# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Text-only prefill / decoder-logit PCC for Dots OCR.

**Reference (PyTorch / HF only)**
``DotsOCRReference`` and the HF text submodel run in **PyTorch** to produce gold ``hf_logits`` and
to read checkpoint weights for ``convert_hf_to_meta``. No separate “TT path” prefill is implemented
in torch.

**TT path (ttnn only for the model forward)**
``Generator.prefill_forward_text(input_ids, rot_mats=None)``: ttnn token ``embd``, device
``HfRotarySetup`` prefill cos/sin, then ``ttnn_prefill_forward`` (decoder + LM head). Logits are
read back to host tensors only for :func:`assert_quality` vs HF.

**Constants:** ``TEXT_PREFILL_PCC_MIN`` — Pearson floor vs HF (**0.98**).
Log lines: ``[dots_ocr text_prefill_pcc] ...``.
"""

import gc

import pytest
import torch
from loguru import logger

from models.demos.dots_ocr.reference.hf_utils import HFLoadSpec
from models.demos.dots_ocr.reference.model import DotsOCRReference
from models.tt_transformers.tt.load_checkpoints import convert_hf_to_meta, convert_hf_to_meta_no_qkv_permute

try:
    import ttnn  # type: ignore

    _HAS_TTNN_RUNTIME = hasattr(ttnn, "open_mesh_device")
except Exception:
    ttnn = None  # type: ignore
    _HAS_TTNN_RUNTIME = False

if not _HAS_TTNN_RUNTIME:
    pytest.skip("TTNN runtime not available (skipping TTNN PCC tests)", allow_module_level=True)

from models.demos.dots_ocr.tt.mesh import close_dots_mesh_device
from models.demos.dots_ocr.tt.model import DotsTransformer
from models.demos.dots_ocr.tt.model_config import DotsModelArgs
from models.tt_dit.utils.check import assert_quality

# PCC floor vs HF.
TEXT_PREFILL_PCC_MIN = 0.98


def _load_dots_reference(model_id: str, *, dtype=torch.bfloat16):
    """
    Load HF reference model via ``load_processor_and_model`` (eager attention; no flash_attn).
    """
    spec = HFLoadSpec(model_id=model_id, dtype=dtype)
    return DotsOCRReference(spec), spec


def _open_mesh_device():
    """Open the default single-device mesh (skips if unavailable)."""
    from models.demos.dots_ocr.tt._ttnn_import import get_ttnn
    from models.demos.dots_ocr.tt.mesh import open_mesh_device as _open

    try:
        ttnn = get_ttnn()
        if ttnn is None:
            raise RuntimeError("ttnn is not available")
        return _open(mesh_shape=ttnn.MeshShape(1, 1))
    except Exception as e:
        pytest.skip(f"Requires TT device runtime (could not open mesh device): {e!r}")
    return None


def test_text_decoder_prefill_pcc(tmp_path):
    """
    ttnn prefill (``prefill_forward_text``, no host rot/embed path) vs HF last **non-pad** token.
    """
    torch.manual_seed(0)

    device = _open_mesh_device()
    try:
        # Use real dots.mocr model by default (HF reference uses eager attention).
        model_id = "rednote-hilab/dots.mocr"
        ref, spec = _load_dots_reference(model_id, dtype=torch.bfloat16)

        # Get test inputs (text-only)
        prompt = "Hello, how are you today?"
        inputs = ref.preprocess_image_and_prompt(None, prompt)  # None image = text-only

        # Prefill kernels require seq_len padded to a multiple of 128, so the minimum usable max_seq_len is 128.
        prefill_max_seq = 128

        # All HF-derived tensors first, then drop the reference model before TTNN + real checkpoint tensors.
        # Peak host RAM otherwise stacks HF weights + large checkpoint state_dict + device buffers → OOM kill.
        with torch.no_grad():

            def _pick_text_submodel(hf_model):
                """
                Return the module that actually implements the text decoder stack.

                Dots OCR is a multimodal wrapper; depending on transformers/remote-code versions,
                the text stack may be exposed as `language_model`, `text_model`, or nested under `model`.
                We pick the first candidate whose state_dict contains decoder layer keys.
                """
                candidates = []
                for name in ("language_model", "text_model", "model"):
                    m = getattr(hf_model, name, None)
                    if m is not None:
                        candidates.append((name, m))
                # Also try nested `model.language_model` if present
                m0 = getattr(hf_model, "model", None)
                if m0 is not None:
                    m1 = getattr(m0, "language_model", None)
                    if m1 is not None:
                        candidates.insert(0, ("model.language_model", m1))

                def looks_like_decoder(sd_keys):
                    """
                    Detect the HF decoder stack.

                    Submodule ``state_dict`` keys are often ``layers.0.*`` (no ``model.`` prefix),
                    while full-module checkpoints use ``model.layers.0.*``. Match both.
                    """
                    has_layer0 = any((".layers.0." in k) or k.startswith("layers.0.") for k in sd_keys)
                    has_attn = any("self_attn" in k or "attention" in k for k in sd_keys)
                    return has_layer0 and has_attn

                for name, m in candidates:
                    try:
                        keys = list(m.state_dict().keys())
                    except Exception:
                        continue
                    if looks_like_decoder(keys):
                        logger.info(f"Prefill test: using text submodel `{name}` for HF logits/weights")
                        return m
                logger.warning("Prefill test: could not find dedicated text submodel; falling back to full model")
                return hf_model

            _text_model = _pick_text_submodel(ref.model)
            # Inner ``Qwen2Model`` / Llama base returns hidden states; causal wrappers expose ``logits``.
            if hasattr(_text_model, "lm_head"):
                text_out = _text_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True,
                )
                hf_logits = text_out.logits
            else:
                base_out = _text_model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    return_dict=True,
                )
                hf_logits = ref.model.lm_head(base_out.last_hidden_state)
            logger.info(f"HF (text submodel) logits shape: {hf_logits.shape} (using eager attention implementation)")

            # Capture the text submodel config/class for later conversion-check.
            torch_text_model_cfg = _text_model.config

            # Capture embedding + lm_head weights explicitly; the text submodule's state_dict may not include lm_head.
            torch_embed_weight = _text_model.get_input_embeddings().weight.detach().cpu()
            _lm_head_mod = getattr(ref.model, "lm_head", None) or getattr(_text_model, "lm_head", None)
            torch_lm_head_weight = (
                _lm_head_mod.weight.detach().cpu()
                if _lm_head_mod is not None and hasattr(_lm_head_mod, "weight")
                else None
            )

            if inputs.attention_mask is not None:
                n_real = int(inputs.attention_mask[0].sum().item())
            else:
                n_real = int(inputs.input_ids.shape[1])
            # int32 [1, S] for ttnn token embd + device RoPE (no host prefill_forward_embeddings).
            input_ids_for_ttnn = inputs.input_ids.to(dtype=torch.int32, device="cpu")
            if input_ids_for_ttnn.dim() == 1:
                input_ids_for_ttnn = input_ids_for_ttnn.unsqueeze(0)

        del ref
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        optimizations = None
        try:
            from models.tt_transformers.tt.model_config import parse_optimizations

            optimizations = parse_optimizations(
                "precision_cfg{activation:bf16,wqkv:bf16,wo:bf16,kv_cache:bf16,ff1_3:bf16,ff2:bf16} "
                "fidelity_cfg{accuracy:hifi4fp32,li_qkv_prefill:hifi4fp32,sdpa_prefill:hifi4fp32,li_o_prefill:hifi4fp32,"
                "li_qkv_decode:hifi4fp32,sdpa_decode:hifi4fp32,li_o_decode:hifi4fp32,li_ff1_3:hifi4fp32,li_ff2:hifi4fp32}"
            )
        except Exception as e:
            logger.warning(f"Prefill test: could not set optimizations ({e!r}); continuing with defaults")

        def _run_once(*, qkv_permute: bool):
            model_args = DotsModelArgs(
                mesh_device=device,
                max_batch_size=1,
                max_seq_len=prefill_max_seq,
                hf_config=torch_text_model_cfg,
                optimizations=optimizations,
            )
            # Align cache tagging with demo loads (QKV layout selection).
            model_args.dots_text_qkv_permute = bool(qkv_permute)
            # For meaningful PCC vs HF, avoid bfloat8 LM head output quantization.
            model_args.lm_head_dtype = ttnn.bfloat16

            # Ensure TT model depth matches HF text model depth.
            # Some remote-code configs don't populate `num_hidden_layers` reliably, so infer from weights too.
            hf_layers_cfg = getattr(torch_text_model_cfg, "num_hidden_layers", None)
            hf_layers_sd = None
            try:
                layer_idxs = []
                for k in _text_model.state_dict().keys():
                    if ".layers." in k:
                        # e.g. "model.layers.0.self_attn.q_proj.weight" or "layers.0...."
                        tail = k.split(".layers.", 1)[1]
                        idx = int(tail.split(".", 1)[0])
                        layer_idxs.append(idx)
                if layer_idxs:
                    hf_layers_sd = max(layer_idxs) + 1
            except Exception:
                hf_layers_sd = None

            hf_layers = int(hf_layers_sd or hf_layers_cfg or model_args.n_layers)
            logger.info(f"Prefill test: HF layers inferred cfg={hf_layers_cfg} sd={hf_layers_sd} -> using {hf_layers}")
            if model_args.n_layers != hf_layers:
                logger.warning(f"Prefill test: forcing n_layers {model_args.n_layers} -> {hf_layers} to match HF")
                model_args.n_layers = hf_layers
                model_args.full_model_n_layers = hf_layers

            # Build the TT state_dict directly from the already-loaded HF text model.
            raw_sd = _text_model.state_dict()
            stripped = {}
            for k, v in raw_sd.items():
                k2 = k[len("model.") :] if k.startswith("model.") else k
                stripped[k2] = v
            # Ensure embeddings + lm_head are present under standard HF key names so the Meta conversion
            # yields `tok_embeddings.weight` and `output.weight`.
            stripped.setdefault("embed_tokens.weight", torch_embed_weight)
            if torch_lm_head_weight is not None:
                stripped.setdefault("lm_head.weight", torch_lm_head_weight)

            # A/B: some stacks want HF-layout Q/K (no permute), others want Meta-layout (permute).
            convert = convert_hf_to_meta if qkv_permute else convert_hf_to_meta_no_qkv_permute
            state_dict = convert(
                stripped,
                model_args.head_dim,
                n_heads=model_args.n_heads,
                n_kv_heads=model_args.n_kv_heads,
            )
            # Belt-and-braces: alias expected TT keys if conversion didn't create them.
            if "tok_embeddings.weight" not in state_dict and "embed_tokens.weight" in state_dict:
                state_dict["tok_embeddings.weight"] = state_dict["embed_tokens.weight"]
            if "output.weight" not in state_dict and "lm_head.weight" in state_dict:
                state_dict["output.weight"] = state_dict["lm_head.weight"]

            # Create TTNN transformer
            weight_cache_path = tmp_path / ("weights_perm" if qkv_permute else "weights_noperm")
            tt_model = DotsTransformer(
                args=model_args,
                dtype=ttnn.bfloat16,
                mesh_device=device,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
            )

            from models.demos.dots_ocr.tt.generator import Generator

            generator = Generator(tt_model, model_args, device)
            prompt_lens = torch.tensor([n_real], dtype=torch.int32)
            tt_logits = generator.prefill_forward_text(input_ids_for_ttnn, rot_mats=None, prompt_lens=prompt_lens)
            pcc_label = "ttnn token embd+device RoPE+decoder"
            last_pos = n_real - 1
            hf_last = hf_logits[:, last_pos : last_pos + 1, :]
            # If vocab differs (should not in real-weight mode), compare overlapping prefix.
            hf_cmp = hf_last.float()
            tt_cmp = tt_logits.float()
            min_vocab = min(hf_cmp.shape[-1], tt_cmp.shape[-1])
            hf_cmp = hf_cmp[..., :min_vocab]
            tt_cmp = tt_cmp[..., :min_vocab]
            logger.info(f"Text-only prefill tensors ready ({pcc_label}, qkv_permute={int(qkv_permute)})")
            return hf_cmp, tt_cmp

        # Run TT quality check: try both QKV conversion variants (same as taking max PCC vs HF).
        try:
            min_pcc = TEXT_PREFILL_PCC_MIN
            last_exc: Exception | None = None
            for qkv_permute in (False, True):
                try:
                    hf_cmp, tt_cmp = _run_once(qkv_permute=qkv_permute)
                    assert_quality(hf_cmp, tt_cmp, pcc=min_pcc)
                    logger.info(f"Text-only prefill assert_quality passed (qkv_permute={int(qkv_permute)})")
                    print(
                        f"[dots_ocr text_prefill_pcc] PASS qkv_permute={int(qkv_permute)} "
                        f"(assert_quality pcc>={min_pcc:.6f})",
                        flush=True,
                    )
                    break
                except Exception as e:
                    last_exc = e
                    logger.warning(f"Prefill quality check failed for qkv_permute={int(qkv_permute)}: {e}")
            else:
                assert last_exc is not None
                raise last_exc
        except Exception as e:
            logger.exception("Prefill/PCC path raised: {}", e)
            raise

    finally:
        if device is not None:
            close_dots_mesh_device(device)

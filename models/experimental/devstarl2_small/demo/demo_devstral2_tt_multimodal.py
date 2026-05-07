# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Tenstorrent demo: Devstral Small 2 (Mistral3) **text** LM on TT, with **autoregressive generation**
using the implemented TT stack (``TtMinistral3Model`` + layers) and Hugging Face ``lm_head`` for logits.

**Prompting** defaults to the **same** chat template as ``reference/inference.py`` (``inference_fixtures``:
system prompt, user fibonacci/tools task, and tool schemas). Use ``--simple-chat`` for a single user turn
from ``--prompt`` only.

**Generation** repeats full TT **prefill** on the growing sequence (no TT decode/KV-cache yet), then
``lm_head`` for the next token—matching ``inference.py`` setup requires **full** ``--text-layers``
(omit the flag) and sampling defaults aligned with ``REFERENCE_GENERATE_KWARGS``.

Usage (from repo root)::

    export HF_MODEL=mistralai/Devstral-Small-2-24B-Instruct-2512
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py --seed 0

    # Bring-up: fewer layers + PCC check
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py \\
        --text-layers 1 --verify --max-new-tokens 20

    # Optional: pure HF ``generate()`` baseline (not TT)
    python models/experimental/devstarl2_small/demo/demo_devstral2_tt_multimodal.py --hf-generate --seed 0

Embeddings are padded so sequence length is **128**-divisible, **KV-shard tile**-aligned, and (when
``L > 1024``) a multiple of **1024** for the attention ``wo`` chunk reshape; hidden states are sliced
back to the real length for ``lm_head`` and PCC.
"""

from __future__ import annotations

import argparse
import os
import types
from pathlib import Path

import torch
from loguru import logger
from transformers import MistralCommonBackend
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.masking_utils import create_causal_mask
from transformers.models.mistral3.modeling_mistral3 import Mistral3Model

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.experimental.devstarl2_small.tt.tt_ministral3_model import TtMinistral3Model
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.model_config import ModelArgs

try:
    from tests.scripts.common import get_updated_device_params
except ImportError:  # minimal fallback if tests package not on PYTHONPATH
    get_updated_device_params = lambda p: p  # type: ignore[assignment]

from models.experimental.devstarl2_small.reference.inference_fixtures import (
    REFERENCE_GENERATE_KWARGS,
    REFERENCE_MESSAGES,
    REFERENCE_TOOLS,
)

_DEFAULT_MODEL_ID = "mistralai/Devstral-Small-2-24B-Instruct-2512"
_DEMO_DIR = Path(__file__).resolve().parent

_ORIGINAL_FP8_DEQUANTIZE_ONE = Fp8Dequantize._dequantize_one


def _dequantize_one_compat(self, quantized: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if scales.ndim == 0:
        fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
        if quantized.dtype == torch.int8 or (fp4_dtype is not None and quantized.dtype == fp4_dtype):
            quantized_fp32 = self._unpack_fp4(quantized)
        else:
            quantized_fp32 = quantized.to(torch.float32)
        out_dtype = scales.dtype if scales.dtype.is_floating_point and scales.element_size() >= 2 else torch.bfloat16
        scale = scales.to(torch.float32)
        return (quantized_fp32 * scale).to(out_dtype)
    return _ORIGINAL_FP8_DEQUANTIZE_ONE(self, quantized, scales)


Fp8Dequantize._dequantize_one = _dequantize_one_compat


def _text_model_root(multimodal_inner: Mistral3Model):
    lm = multimodal_inner.language_model
    return lm.model if hasattr(lm, "model") else lm


def _apply_devstral_hf_trust_patches():
    from models.tt_transformers.tt import model_config as mc

    orig_set = mc.ModelArgs._set_hf_params

    def _set_hf_params_trust(self, checkpoint_dir: str):
        self.trust_remote_code_hf = True
        return orig_set(self, checkpoint_dir)

    mc.ModelArgs._set_hf_params = _set_hf_params_trust  # type: ignore[method-assign]

    def _get_hf_model_cls_devstral_safe(self):
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.modeling_auto import AutoModelForImageTextToText

        if not self.is_multimodal:
            return AutoModelForCausalLM
        if type(self.hf_config) in AutoModelForImageTextToText._model_mapping:
            return AutoModelForImageTextToText
        raise ValueError(
            f"Demo supports multimodal configs in AutoModelForImageTextToText only; got {type(self.hf_config)}"
        )

    mc.ModelArgs.get_hf_model_cls = _get_hf_model_cls_devstral_safe  # type: ignore[method-assign]


def _tt_prefill_target_seqlen(seq_len: int, n_kv_heads: int, mesh_cluster_cols: int) -> int:
    """
    Prefill constraints (see ``models/tt_transformers/tt/attention.py`` ``forward_prefill``):

    - ``seq_len % 128 == 0`` (hard assert before QKV).
    - KV fill path: ``(n_kv // mesh_cols) * L // 64`` must be a multiple of the tile size (32), or
      ``interleaved_to_sharded`` fails.
    - When ``L > 1024``, ``wo`` reuses ``[1, L // 1024, 1024, H]``; **L must be a multiple of 1024** or
      the reshape is invalid and the next ``ttnn.linear`` sees the wrong inner dim (e.g. 5120 vs 4096).
    """
    k = n_kv_heads // mesh_cluster_cols
    assert k > 0
    target = seq_len if seq_len % 128 == 0 else seq_len + (128 - (seq_len % 128)) % 128
    for _ in range(32768):
        kv_ok = (k * target // 64) % 32 == 0
        wo_ok = target <= 1024 or (target % 1024 == 0)
        if kv_ok and wo_ok:
            return target
        target += 128
    raise RuntimeError("Could not find L satisfying TT prefill KV + WO chunking constraints.")


def _pad_language_embeddings_for_tt_prefill(
    merged: torch.Tensor,
    position_ids: torch.LongTensor,
    rotary: torch.nn.Module,
    n_kv_heads: int,
    mesh_cluster_cols: int,
) -> tuple[torch.Tensor, torch.LongTensor, tuple]:
    seq_len = int(merged.shape[1])
    target = _tt_prefill_target_seqlen(seq_len, n_kv_heads, mesh_cluster_cols)
    if target == seq_len:
        pe = rotary(merged, position_ids=position_ids)
        return merged, position_ids, pe
    pad = target - seq_len
    merged_pad = torch.nn.functional.pad(merged, (0, 0, 0, pad), value=0.0)
    extra = torch.arange(seq_len, target, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(0)
    position_ids_pad = torch.cat([position_ids, extra], dim=1)
    pe = rotary(merged_pad, position_ids=position_ids_pad)
    return merged_pad, position_ids_pad, pe


def _open_mesh(mesh_width: int):
    device_params = {
        "trace_region_size": 30000000,
        "num_command_queues": 1,
    }
    mesh_shape = ttnn.MeshShape(1, mesh_width)
    return ttnn.open_mesh_device(mesh_shape=mesh_shape, **get_updated_device_params(device_params))


def _squeeze_tt_hidden_to_bsh(tt_lm_out: ttnn.Tensor, mesh_device, seq_len_keep: int) -> torch.Tensor:
    tt_h = ttnn.to_torch(tt_lm_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
    while tt_h.dim() > 3:
        tt_h = tt_h.squeeze(0)
    if tt_h.dim() == 2:
        tt_h = tt_h.unsqueeze(0)
    return tt_h[:, :seq_len_keep, :]


def _eos_token_ids(config, tokenizer=None) -> set[int]:
    """
    Collect EOS ids for stopping TT/HF loops.

    ``Mistral3ForConditionalGeneration`` often leaves ``config.eos_token_id`` unset while
    ``config.text_config.eos_token_id`` (and the tokenizer) still define EOS (e.g. 2). Missing this
    yields an empty set and generation never stops on end-of-sequence, which causes long repetition.
    """
    ids: set[int] = set()

    def _add(eos):
        if eos is None:
            return
        if isinstance(eos, (list, tuple)):
            ids.update(int(x) for x in eos)
        else:
            ids.add(int(eos))

    _add(getattr(config, "eos_token_id", None))
    tc = getattr(config, "text_config", None)
    if tc is not None:
        _add(getattr(tc, "eos_token_id", None))
    gen = getattr(config, "generation_config", None)
    if gen is not None:
        _add(getattr(gen, "eos_token_id", None))
    if tokenizer is not None:
        _add(getattr(tokenizer, "eos_token_id", None))
    return ids


def _tt_prefill_hidden_states(
    merged_bf16: torch.Tensor,
    rotary: torch.nn.Module,
    mesh_device,
    tt_model: TtMinistral3Model,
    seq_len: int,
    model_args: ModelArgs,
) -> torch.Tensor:
    position_ids_lm = torch.arange(seq_len, dtype=torch.long, device=merged_bf16.device).unsqueeze(0)
    merged_tt, position_ids_tt, position_embeddings_tt = _pad_language_embeddings_for_tt_prefill(
        merged_bf16,
        position_ids_lm,
        rotary,
        int(model_args.n_kv_heads),
        int(model_args.cluster_shape[1]),
    )
    cos, sin = position_embeddings_tt
    cos_tt = ttnn.from_torch(
        cos.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    sin_tt = ttnn.from_torch(
        sin.unsqueeze(0),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    rot_mats = [cos_tt, sin_tt]
    x_tt = ttnn.from_torch(
        merged_tt.unsqueeze(1),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    pos_lm_tt = ttnn.from_torch(
        position_ids_tt.to(torch.int32),
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    tt_lm_out = tt_model.forward_prefill_from_embeddings(x_tt, rot_mats, pos_lm_tt)
    return _squeeze_tt_hidden_to_bsh(tt_lm_out, mesh_device, seq_len)


def main():
    ref_max = int(REFERENCE_GENERATE_KWARGS["max_new_tokens"])
    ref_temp = float(REFERENCE_GENERATE_KWARGS["temperature"])
    ref_do_sample = bool(REFERENCE_GENERATE_KWARGS["do_sample"])

    parser = argparse.ArgumentParser(
        description="Devstral-2 text LM on Tenstorrent: TT transformer + HF lm_head (inference.py-style prompt by default)."
    )
    parser.add_argument("--model-id", default=_DEFAULT_MODEL_ID, help="HF repo id (also set HF_MODEL).")
    parser.add_argument(
        "--prompt",
        default=(
            "Can you implement in Python a method to compute the fibonnaci sequence at the `n`th element "
            "with `n` a parameter passed to the function? Start the sequence from 1."
        ),
        help="Only used with --simple-chat (single user turn).",
    )
    parser.add_argument(
        "--system-prompt",
        default=None,
        help="Optional system message for --simple-chat only.",
    )
    parser.add_argument(
        "--simple-chat",
        action="store_true",
        help="Tokenize a minimal chat from --prompt/--system-prompt only. "
        "Default: same messages+tools as reference/inference.py (inference_fixtures).",
    )
    parser.add_argument(
        "--text-layers",
        type=int,
        default=None,
        help="Decoder layers on TT/HF cache after load (default: all). Required for quality matching full inference.",
    )
    parser.add_argument("--mesh-width", type=int, default=1, help="Device mesh width (1 × N).")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="HF reference forward vs TT PCC on the prompt embeddings.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=ref_max,
        help=f"New tokens after prompt (default {ref_max}, same as reference/inference.py). 0 = skip generation.",
    )
    parser.add_argument(
        "--greedy",
        action="store_true",
        help=f"Greedy argmax (default: sample with temperature {ref_temp}, like reference/inference.py).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=ref_temp,
        help="Sampling temperature when not --greedy.",
    )
    parser.add_argument(
        "--hf-generate",
        action="store_true",
        help="Use Hugging Face model.generate() instead of TT+lm_head (baseline only; not TT).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Torch/CUDA RNG seed before generation (recommended for sampling).",
    )
    args = parser.parse_args()

    if args.hf_generate and args.text_layers is not None:
        logger.warning("--hf-generate uses full HF depth; ignoring --text-layers.")
        args.text_layers = None

    os.environ["HF_MODEL"] = args.model_id
    _apply_devstral_hf_trust_patches()

    mesh_device = _open_mesh(max(1, min(args.mesh_width, ttnn.get_num_devices())))
    try:
        dtype_tt = ttnn.bfloat16

        tokenizer = MistralCommonBackend.from_pretrained(
            args.model_id,
            trust_remote_code=True,
            local_files_only=os.getenv("CI") == "true",
        )
        if args.simple_chat:
            if args.system_prompt:
                messages = [
                    {"role": "system", "content": args.system_prompt},
                    {"role": "user", "content": [{"type": "text", "text": args.prompt}]},
                ]
                tokenized = tokenizer.apply_chat_template(
                    conversation=messages,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                messages = [{"role": "user", "content": [{"type": "text", "text": args.prompt}]}]
                tokenized = tokenizer.apply_chat_template(
                    conversation=messages,
                    return_tensors="pt",
                    return_dict=True,
                )
        else:
            tokenized = tokenizer.apply_chat_template(
                conversation=REFERENCE_MESSAGES,
                tools=REFERENCE_TOOLS,
                return_tensors="pt",
                return_dict=True,
            )

        input_ids = tokenized["input_ids"]
        prompt_len = int(input_ids.shape[1])
        extra_tokens = max(0, args.max_new_tokens)
        # Upper bound padded prefill length: 128-step fixes for KV shard tile height + generation growth.
        max_seq = max(4096, prompt_len + extra_tokens + 2048)

        model_args = ModelArgs(
            mesh_device,
            max_batch_size=1,
            max_seq_len=max_seq,
            dummy_weights=False,
            use_hf_rope=True,
            cache_hf=True,
        )
        model_args.is_distributed_norm = types.MethodType(lambda self, mode: False, model_args)

        logger.info("Loading checkpoint via ModelArgs.load_state_dict() …")
        try:
            meta_state_dict = model_args.load_state_dict()
        except Exception as exc:
            raise RuntimeError(
                f"Checkpoint load failed (memory, hub, FP8, etc.): {exc}\n"
                "Ensure HF access, enough RAM, and compatible transformers."
            ) from exc

        if args.text_layers is not None:
            if args.text_layers < 1 or args.text_layers > model_args.full_model_n_layers:
                raise ValueError(
                    f"--text-layers must be in [1, {model_args.full_model_n_layers}], got {args.text_layers}"
                )
            model_args.n_layers = args.text_layers
            if args.max_new_tokens > 0 and not args.hf_generate:
                logger.warning(
                    "Partial --text-layers: TT generation will not match full-model reference/inference.py quality."
                )

        hf_full = model_args.cached_hf_model
        if hf_full is None:
            raise RuntimeError("Expected cached HF model after load_state_dict with cache_hf=True.")
        hf_inner = hf_full.model
        if not isinstance(hf_inner, Mistral3Model):
            raise TypeError(f"Expected Mistral3Model, got {type(hf_inner)}")

        text_cfg = model_args.hf_config.text_config
        rope_params = getattr(text_cfg, "rope_parameters", None) or {}
        if not isinstance(rope_params, dict):
            rope_params = dict(rope_params)

        tt_model = TtMinistral3Model(
            mesh_device=mesh_device,
            tt_ccl=TT_CCL(mesh_device),
            model_args=model_args,
            meta_state_dict=meta_state_dict,
            weight_cache_path=model_args.weight_cache_path(dtype_tt),
            dtype=dtype_tt,
            transformation_mats={"decode": None, "prefill": None},
            configuration=model_args,
            llama_4_scaling_beta=rope_params.get("llama_4_scaling_beta"),
            original_max_position_embeddings=rope_params.get("original_max_position_embeddings"),
        )

        input_ids = input_ids.to(hf_inner.get_input_embeddings().weight.device)
        merged = hf_inner.get_input_embeddings()(input_ids).to(torch.bfloat16)

        text_root = _text_model_root(hf_inner)
        rotary = text_root.rotary_emb
        rotary.eval()
        seq_len_lm = int(merged.shape[1])
        target_lm = _tt_prefill_target_seqlen(seq_len_lm, int(model_args.n_kv_heads), int(model_args.cluster_shape[1]))
        if target_lm != seq_len_lm:
            logger.info(
                f"Padded language sequence length {seq_len_lm} → {target_lm} for TT prefill "
                "(128-divisible seq + KV prefill shard tile alignment)."
            )
        position_ids_lm = torch.arange(seq_len_lm, dtype=torch.long, device=merged.device).unsqueeze(0)
        position_embeddings_hf = rotary(merged, position_ids=position_ids_lm)

        logger.info(
            f"TT language model prefill ({model_args.n_layers} decoder layers); "
            f"prompt template: {'simple-chat' if args.simple_chat else 'inference_fixtures (inference.py)'}."
        )
        tt_lm_torch = _tt_prefill_hidden_states(merged, rotary, mesh_device, tt_model, seq_len_lm, model_args)

        logger.info(f"TT prefill hidden states shape (batch, seq, dim): {tuple(tt_lm_torch.shape)}")

        if args.verify:
            causal_mask = create_causal_mask(
                config=text_cfg,
                inputs_embeds=merged,
                attention_mask=None,
                past_key_values=None,
                position_ids=position_ids_lm,
            )
            hidden = merged
            for layer in text_root.layers[: model_args.n_layers]:
                hidden = layer(
                    hidden_states=hidden,
                    attention_mask=causal_mask,
                    position_ids=position_ids_lm,
                    past_key_values=None,
                    use_cache=False,
                    position_embeddings=position_embeddings_hf,
                )
            ref_out = text_root.norm(hidden)
            tt_cmp = tt_lm_torch
            if tt_cmp.shape != ref_out.shape:
                tt_cmp = tt_cmp.reshape(ref_out.shape)
            pcc_ok, msg = comp_pcc(ref_out, tt_cmp, 0.90)
            logger.info(comp_allclose(ref_out, tt_cmp))
            logger.info(f"PCC (HF vs TT LM on same token embeddings): {msg}")
            if not pcc_ok:
                logger.warning(f"PCC check did not reach threshold: {msg}")

        if args.max_new_tokens <= 0:
            pass
        elif args.hf_generate:
            gen_device = next(hf_full.parameters()).device
            if args.seed is not None:
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)
            prompt_vec = tokenized["input_ids"][0]
            input_ids_gen = tokenized["input_ids"].to(gen_device)
            logger.info(f"HF baseline generate {REFERENCE_GENERATE_KWARGS} on {gen_device} …")
            with torch.inference_mode():
                out = hf_full.generate(input_ids_gen, **REFERENCE_GENERATE_KWARGS)
            seq = out[0]
            answer_text = tokenizer.decode(seq[len(prompt_vec) :].tolist(), skip_special_tokens=False)
            logger.info(f"HF generate baseline ({seq.numel() - prompt_vec.numel()} new tokens):\n{answer_text}")
        else:
            do_sample = ref_do_sample if not args.greedy else False
            if args.seed is not None:
                torch.manual_seed(args.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(args.seed)

            lm_head = hf_full.lm_head
            wdev, wdtype = lm_head.weight.device, lm_head.weight.dtype
            eos_ids = _eos_token_ids(hf_full.config, tokenizer)
            if not eos_ids:
                logger.warning(
                    "No eos_token_id on config/text_config/tokenizer; TT loop will only stop at --max-new-tokens."
                )
            current_ids = input_ids.clone()
            mode = "greedy" if args.greedy else f"sample (T={args.temperature})"
            logger.info(
                f"TT generation: up to {args.max_new_tokens} new tokens, {mode}; "
                f"HF lm_head on {wdev}; full TT prefill each step."
            )
            for _step in range(args.max_new_tokens):
                merged_cur = hf_inner.get_input_embeddings()(current_ids).to(torch.bfloat16)
                sl = int(merged_cur.shape[1])
                tt_h = _tt_prefill_hidden_states(merged_cur, rotary, mesh_device, tt_model, sl, model_args)
                h_last = tt_h[:, sl - 1 : sl, :].to(device=wdev, dtype=wdtype)
                logits = lm_head(h_last)
                if do_sample:
                    probs = torch.softmax(logits.float().squeeze(1) / max(args.temperature, 1e-6), dim=-1)
                    next_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_id = logits.argmax(dim=-1)
                if next_id.dim() == 1:
                    next_id = next_id.unsqueeze(-1)
                if int(next_id.item()) in eos_ids:
                    break
                current_ids = torch.cat([current_ids, next_id.to(current_ids.device)], dim=1)

            answer_ids = current_ids[0, prompt_len:]
            answer_text = tokenizer.decode(answer_ids.tolist(), skip_special_tokens=False)
            logger.info(f"TT+lm_head generated ({answer_ids.numel()} tokens):\n{answer_text}")
    finally:
        ttnn.close_mesh_device(mesh_device)


if __name__ == "__main__":
    main()

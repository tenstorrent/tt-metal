# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end multimodal PCC test.

Loads vision tower (bfloat8_b, ~50% memory savings) and text model (bfloat16)
into device DRAM together, runs ``encode_image`` → ``prefill_multimodal_full_logits``
on TTNN, and compares the full-sequence logits against an HF Torch reference.

  - Expected PCC: ~0.83-0.84 (vision bf8 quantization is the dominant source of loss)
  - Expected memory: ~99% DRAM utilization (both models fit simultaneously)

Run::

    export MISTRAL4_MM_PCC=1
    export MISTRAL4_MM_IMAGE=Battle.jpg   # optional; local path or URL. Defaults to a sample
                                          # battle-scene URL (like demo); set to "" for a random image.
    export MISTRAL4_WEIGHT_CACHE_DIR=/tmp/mistral4_weights  # optional; cache quantized text weights to skip re-quantization
    export MESH_DEVICE=P150x8
    pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_pcc_unified.py -v -s --timeout=0

Pass/fail:
  The only assertion is the overall flattened-logits PCC ≥ 0.80 (``_PCC_FLOOR``). The
  per-position PCC and greedy-token match are logged as diagnostics only — greedy-token
  agreement is intentionally NOT enforced: bf4/bf8 quantization makes exact argmax brittle
  over a long sequence, while logit correlation is the robust signal.

Layer / long-context sweep:
  Parametrized over (num_text_layers, num_vision_layers, max_text_tokens) with ids like
  L1V1 (light smoke) and L36V24_16384 (full model, 16K text context). max_text_tokens pads
  the prompt with coherent English filler up to that many tokens (image tokens added on top;
  0 = prompt unchanged). Pick one with `-k`, e.g. `-k L36V24_16384`. The HF reference runs in
  chunks (MISTRAL4_MM_PREFILL_CHUNK, default 2048) backed by a DynamicCache and loads the model
  in bf16, keeping attention memory at O(chunk × past) instead of O(seq²).

  16K is the highest context that completes PCC verification. L36V24_16384 / _65536 / _131072 /
  _262144 are marked ``@pytest.mark.slow``; beyond ~16K the chunked HF CPU reference forward
  takes prohibitively long (a 64K run was killed after > 1 h without finishing). Chunking
  already bounds its memory, so the bottleneck is wall-clock time, not RAM — it is the
  verification ceiling, not a device limit.

    pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_pcc_unified.py -v -s --timeout=0 -k L36V24_16384
"""

from __future__ import annotations

import copy
import gc
import os
import psutil

import pytest
import torch
from loguru import logger
from transformers.cache_utils import DynamicCache

import ttnn
from models.common.utility_functions import comp_pcc, run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    HF_MODEL_ID,
    MMP_SPATIAL_MERGE_SIZE,
    VISION_PATCH_SIZE,
    text_decoder_layer_state_dict_prefix,
    vision_layer_state_dict_prefix,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral3_for_conditional_generation_unified import (
    TtMistral3ForConditionalGenerationUnified,
)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered


def _log_memory_usage(label: str = ""):
    """Log current system memory usage."""
    try:
        proc = psutil.Process()
        mem = proc.memory_info()
        percent = proc.memory_percent()
        logger.info(f"[Memory {label}] RSS: {mem.rss / 1e9:.2f}GB ({percent:.1f}%)")
    except Exception as e:
        logger.debug(f"Skipping memory usage logging for '{label}': {e}")


pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral3.modeling_mistral3", reason="Mistral3 required")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")
pytest.importorskip("transformers.models.pixtral.modeling_pixtral", reason="Pixtral required")


# Default sample image (a battle scene), matching demo_multimodal.py. Pass a local path or
# URL via MISTRAL4_MM_IMAGE to override, or MISTRAL4_MM_IMAGE="" to use a random synthetic image.
DEFAULT_IMAGE_URL = (
    "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/" "revision/latest?cb=20220523172438"
)

_IMAGE_PATH = os.environ.get("MISTRAL4_MM_IMAGE", DEFAULT_IMAGE_URL)
_PROMPT = os.environ.get("MISTRAL4_MM_PROMPT", "Describe this image.")
_IMAGE_MAX_SIDE = int(os.environ.get("MISTRAL4_MM_IMAGE_MAX_SIDE", "224"))
_IMG_PATCHES = int(os.environ.get("MISTRAL4_MM_IMG_PATCHES", "10"))
# (num_text_layers, num_vision_layers, max_text_tokens) sweep. max_text_tokens pads the text
# prompt with coherent English filler up to that many tokens (0 = prompt unchanged; image
# tokens are added on top). 16K is the highest context that completes PCC verification; larger
# points are marked @pytest.mark.slow (beyond ~16K the chunked HF CPU reference forward is the
# wall-clock ceiling). Select one with `-k`, e.g. `-k L36V24_16384`.
# Chunk size for the HF reference prefill. Chunking with a DynamicCache keeps attention
# memory at O(chunk × past) instead of O(seq²), so long contexts fit on a CPU host.
_PREFILL_CHUNK = int(os.environ.get("MISTRAL4_MM_PREFILL_CHUNK", "2048"))
# Vision bf8 quantization dominates the accuracy floor.
_PCC_FLOOR = 0.80


def _state_dict_prefixes(n_text: int, n_vision: int) -> tuple:
    p = ["language_model.model.embed_tokens."]
    for i in range(n_text):
        p.append(text_decoder_layer_state_dict_prefix(i))
    p.append("language_model.model.norm.")
    p.append("language_model.lm_head.")
    p.append("vision_tower.patch_conv.")
    p.append("vision_tower.ln_pre.")
    for i in range(n_vision):
        p.append(vision_layer_state_dict_prefix(i))
    p.append("multi_modal_projector.")
    return tuple(p)


def _mesh_params():
    shape = mesh_device_request_param()
    base = {"trace_region_size": 30_000_000, "num_command_queues": 1}
    fabric = ttnn.FabricConfig.DISABLED if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D
    return [pytest.param(shape, {**base, "fabric_config": fabric}, id=f"mesh{shape[0]}x{shape[1]}")]


def _hf_param_to_sd_key(name: str) -> str:
    if name.startswith("model.vision_tower."):
        return name[len("model.") :]
    if name.startswith("model.multi_modal_projector."):
        return name[len("model.") :]
    if name.startswith("model.language_model."):
        return "language_model.model." + name[len("model.language_model.") :]
    if name == "lm_head.weight":
        return "language_model.lm_head.weight"
    return name


def _build_hf_mm_ref(full_config, state_dict: dict, n_text: int, n_vision: int):
    """Build HF reference with activation checkpointing enabled for full-size runs."""
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

    cfg = copy.deepcopy(full_config)
    cfg.text_config.num_hidden_layers = n_text
    cfg.vision_config.num_hidden_layers = n_vision
    for sub in (cfg.text_config, cfg.vision_config):
        for attr in ("attn_implementation", "_attn_implementation"):
            if hasattr(sub, attr):
                setattr(sub, attr, "sdpa")

    if n_text >= 36 and n_vision >= 24:
        logger.info("Enabling activation checkpointing for full-layer model (36+24)…")
        cfg.text_config.gradient_checkpointing = True
        cfg.vision_config.gradient_checkpointing = True

    # Force bf16 meta tensors so set_module_tensor_to_device preserves bf16 dtype.
    # Without this, init_empty_weights creates float32 meta tensors → model loads as fp32
    # (~476 GB for 119B params) instead of bf16 (~238 GB).
    _prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    with init_empty_weights():
        model = Mistral3ForConditionalGeneration(cfg)
    torch.set_default_dtype(_prev_dtype)

    missing = []
    for name, _ in model.named_parameters():
        sd_key = _hf_param_to_sd_key(name)
        if sd_key not in state_dict:
            missing.append(name)
            continue
        v = state_dict[sd_key]
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            scale_inv = state_dict.get(sd_key + "_scale_inv")
            if scale_inv is None:
                scale_inv = state_dict.get(sd_key.replace(".weight", ".weight_scale_inv"))
            v_cast = v.to(torch.float32)
            if scale_inv is not None:
                s = scale_inv.to(torch.float32)
                while s.dim() < v_cast.dim():
                    s = s.unsqueeze(-1)
                v_cast = v_cast * s
            tensor = v_cast.to(torch.bfloat16)
            del v_cast
        else:
            tensor = v.to(torch.bfloat16)
        set_module_tensor_to_device(model, name, "cpu", value=tensor)
        del tensor

    if missing:
        logger.warning(f"HF model missing keys (first 5): {missing[:5]}")
    model = model.eval()
    param_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
    sample_dtype = next(model.parameters()).dtype
    logger.info(f"HF model params: {param_gb:.1f} GB (dtype={sample_dtype}, expected ~238 GB for bf16)")
    return model


_FILLER_SENTENCE = (
    "Context padding for long-window multimodal benchmarking on Tenstorrent hardware. "
    "Please ignore this repeated section when answering; respond only to the final question. "
)


def _build_long_prompt(tokenizer, target_text_tokens: int, question: str) -> str:
    """Coherent natural-language haystack of ~target_text_tokens tokens, then a real question.

    Uses repeated real English (not random token ids) so the MoE router selects the same
    experts in the HF fp32 reference and the TT bf16 path. Random ids make routing scores
    near-tied and collapse PCC, which does not reflect a real accuracy regression.
    """
    per = len(tokenizer(_FILLER_SENTENCE, add_special_tokens=False).input_ids)
    filler = _FILLER_SENTENCE * max(1, target_text_tokens // max(1, per))
    ids = tokenizer(filler, add_special_tokens=False).input_ids
    if len(ids) > target_text_tokens:
        filler = tokenizer.decode(ids[:target_text_tokens], skip_special_tokens=True)
    return f"{filler}\n\nNow answer only this: {question}"


def _chunked_hf_forward(hf_model, pixel_values, input_ids, image_sizes, chunk_size: int) -> torch.Tensor:
    """Run HF forward in chunks using DynamicCache (updated in-place) — O(chunk × past) memory.

    Returns the full float32 logits [seq, vocab] so the per-position + flattened PCC comparison
    keeps working. Chunking only bounds the attention/activation memory (O(chunk × past) instead
    of O(seq²)); the bf16 model load is the other long-context enabler.

    Follows devstral2 logit_pcc_common pattern: DynamicCache created once, mutated in-place,
    no attention_mask, position_ids only for chunks after the first.
    """
    seq_len = input_ids.shape[-1]
    chunk_logits: list[torch.Tensor] = []
    cache = DynamicCache()

    for chunk_start in range(0, seq_len, chunk_size):
        chunk_end = min(chunk_start + chunk_size, seq_len)
        chunk_ids = input_ids[:, chunk_start:chunk_end]

        pv = pixel_values if chunk_start == 0 else None
        iz = image_sizes if chunk_start == 0 else None

        kwargs = dict(pixel_values=pv, input_ids=chunk_ids, image_sizes=iz, past_key_values=cache, use_cache=True)
        if chunk_start > 0:
            kwargs["position_ids"] = torch.arange(chunk_start, chunk_end, dtype=torch.long).unsqueeze(0)

        with torch.inference_mode():
            out = hf_model(**kwargs)

        chunk_logits.append(out.logits[0].float().clone())
        del out
        gc.collect()
        logger.info(f"Chunked prefill: {chunk_end}/{seq_len} tokens")

    return torch.cat(chunk_logits, dim=0)


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_MM_PCC") != "1",
    reason="Set MISTRAL4_MM_PCC=1 to run.",
)
@pytest.mark.timeout(0)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
@pytest.mark.parametrize(
    "num_text_layers, num_vision_layers, max_text_tokens",
    [
        (1, 1, 128),
        (36, 24, 128),
        (36, 24, 4096),
        pytest.param(36, 24, 16384, marks=pytest.mark.slow),
        pytest.param(36, 24, 65536, marks=pytest.mark.slow),
        pytest.param(36, 24, 131072, marks=pytest.mark.slow),
        pytest.param(36, 24, 262144, marks=pytest.mark.slow),
    ],
    ids=[
        "L1V1",
        "L36V24",
        "L36V24_4096",
        "L36V24_16384",
        "L36V24_65536",
        "L36V24_131072",
        "L36V24_262144",
    ],
)
def test_mistral_small_4_multimodal_pcc_unified(
    reset_seeds, mesh_device, num_text_layers, num_vision_layers, max_text_tokens
):
    """
    End-to-end multimodal PCC: vision (bf8) + text (bf16) co-resident on device.

    Expected PCC: ~0.83-0.84 (vision bf8 quantization is the dominant source of loss).
    """
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    image_token_id = int(getattr(cfg, "image_token_index", 10))

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(num_text_layers, num_vision_layers))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    # ── Build inputs via HF chat-template processor ───────────────────
    from PIL import Image
    from transformers import AutoProcessor

    if _IMAGE_PATH and _IMAGE_PATH.startswith(("http://", "https://")):
        import io
        import urllib.request

        logger.info(f"Fetching image from URL: {_IMAGE_PATH}")
        with urllib.request.urlopen(_IMAGE_PATH) as resp:
            img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        if max(img.size) > _IMAGE_MAX_SIDE:
            scale = _IMAGE_MAX_SIDE / max(img.size)
            new_w = max(VISION_PATCH_SIZE, int(round(img.size[0] * scale)))
            new_h = max(VISION_PATCH_SIZE, int(round(img.size[1] * scale)))
            img = img.resize((new_w, new_h))
            logger.info(f"Resized {_IMAGE_PATH!r} → {img.size} (max side ≤ {_IMAGE_MAX_SIDE})")
        else:
            logger.info(f"Loaded {_IMAGE_PATH!r} at {img.size}")
    elif _IMAGE_PATH and os.path.exists(_IMAGE_PATH):
        img = Image.open(_IMAGE_PATH).convert("RGB")
        if max(img.size) > _IMAGE_MAX_SIDE:
            scale = _IMAGE_MAX_SIDE / max(img.size)
            new_w = max(VISION_PATCH_SIZE, int(round(img.size[0] * scale)))
            new_h = max(VISION_PATCH_SIZE, int(round(img.size[1] * scale)))
            img = img.resize((new_w, new_h))
            logger.info(f"Resized {_IMAGE_PATH!r} → {img.size} (max side ≤ {_IMAGE_MAX_SIDE})")
        else:
            logger.info(f"Loaded {_IMAGE_PATH!r} at {img.size}")
    else:
        assert (
            _IMG_PATCHES % MMP_SPATIAL_MERGE_SIZE == 0
        ), f"MISTRAL4_MM_IMG_PATCHES ({_IMG_PATCHES}) must be even for 2x2 merge"
        side = _IMG_PATCHES * VISION_PATCH_SIZE
        gen = torch.Generator().manual_seed(0)
        arr = (torch.rand(side, side, 3, generator=gen) * 255).to(torch.uint8).numpy()
        img = Image.fromarray(arr, mode="RGB")
        logger.info(f"No MISTRAL4_MM_IMAGE set; using deterministic synthetic {side}×{side} RGB image")

    processor = AutoProcessor.from_pretrained(HF_MODEL_ID)

    prompt_text = _PROMPT
    if max_text_tokens:
        from transformers import AutoTokenizer

        tok = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(HF_MODEL_ID)
        prompt_text = _build_long_prompt(tok, max_text_tokens, _PROMPT)
        logger.info(f"Long-context prompt: target {max_text_tokens} text tokens, {len(prompt_text)} chars")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[-1]
    num_image_tokens = int((input_ids[0] == image_token_id).sum().item())
    if "image_sizes" in inputs:
        image_sizes = inputs["image_sizes"]
    else:
        image_sizes = torch.tensor([[pixel_values.shape[-2], pixel_values.shape[-1]]], dtype=torch.long)

    logger.info(
        f"Config: text_layers={num_text_layers}, vision_layers={num_vision_layers}, "
        f"pixel_values {tuple(pixel_values.shape)}, {num_image_tokens} image tokens, "
        f"seq_len={seq_len}, prompt_chars={len(prompt_text)}"
    )

    # ── HF reference forward ─────────────────────────────────────────
    _log_memory_usage("before HF build")
    logger.info("Building HF Mistral3ForConditionalGeneration (CPU, bf16)…")
    hf_model = _build_hf_mm_ref(cfg, state_dict, num_text_layers, num_vision_layers)
    _log_memory_usage("after HF build, before forward")

    # Chunking bounds the HF reference's memory, so genuine OOM is rare. Beyond ~16K the real
    # blocker is wall-clock time — the CPU forward runs for hours, which no `except` can catch
    # (those points are gated @pytest.mark.slow). The handler below only adds guidance for the
    # true-OOM case (smaller chunk / fewer layers / more RAM) before re-raising.
    logger.info(f"Running HF reference forward in chunks of {_PREFILL_CHUNK} tokens…")
    try:
        ref_logits = _chunked_hf_forward(hf_model, pixel_values, input_ids, image_sizes, _PREFILL_CHUNK)
        _log_memory_usage("after HF forward")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                f"OOM during HF forward: {e}\n"
                f"Config: text={num_text_layers} layers, vision={num_vision_layers} layers, chunk={_PREFILL_CHUNK}\n"
                f"Try reducing MISTRAL4_MM_PREFILL_CHUNK (e.g. 512), selecting a smaller layer/context\n"
                f"param (e.g. -k L1V1), or running on a machine with more CPU RAM."
            )
        raise

    del hf_model
    gc.collect()
    _log_memory_usage("after HF cleanup")
    logger.info(f"HF reference logits: {tuple(ref_logits.shape)}")

    # ── TTNN load: vision (bf8) + text (bf16) co-resident on device ───
    _log_memory_usage("before TTNN model load")
    logger.info("Building TtMistral3ForConditionalGenerationUnified (vision bf8 + text bf16)…")
    tt_model = TtMistral3ForConditionalGenerationUnified(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=cfg.text_config,
        image_token_id=image_token_id,
        num_text_layers=num_text_layers,
        num_vision_layers=num_vision_layers,
        max_seq_len=seq_len + 16,
        vision_dtype=ttnn.bfloat8_b,
    )

    logger.info("Calling encode_image() — first call lazy-loads both vision and text on device…")
    img_embeds_host = tt_model.encode_image(pixel_values)
    logger.info(f"Image embeddings: {tuple(img_embeds_host.shape)} bf16 on host")
    _log_memory_usage("after vision + text loaded")

    tt_model.load_text()  # idempotent — text was already loaded by encode_image

    # Position embeddings for prefill.
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    rotary = Mistral4RotaryEmbedding(cfg.text_config).eval().to(torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(dummy, pos_ids)

    logger.info("Running prefill_multimodal_full_logits()…")
    tt_logits = tt_model.prefill_multimodal_full_logits(img_embeds_host, input_ids, (cos, sin))
    tt_logits = tt_logits[0].float()
    _log_memory_usage("after inference")

    assert (
        tt_logits.shape == ref_logits.shape
    ), f"shape mismatch: tt={tuple(tt_logits.shape)} vs ref={tuple(ref_logits.shape)}"

    # ── PCC per position + overall ───────────────────────────────────
    pccs = []
    for i in range(seq_len):
        _, msg = comp_pcc(ref_logits[i], tt_logits[i], _PCC_FLOOR)
        pcc_val = float(msg.split("=")[-1].strip() if "=" in str(msg) else msg)
        pccs.append(pcc_val)
    mean_pcc = sum(pccs) / len(pccs)
    min_pcc = min(pccs)
    logger.info(f"Per-position PCC: mean={mean_pcc:.4f}, min={min_pcc:.4f} (floor {_PCC_FLOOR})")
    logger.info(f"Per-position PCCs: {[f'{p:.3f}' for p in pccs]}")

    # Greedy token agreement at every position.
    ref_tokens = ref_logits.argmax(dim=-1).tolist()
    tt_tokens = tt_logits.argmax(dim=-1).tolist()
    match = sum(r == t for r, t in zip(ref_tokens, tt_tokens))
    logger.info(f"Greedy token match: {match}/{seq_len}")
    logger.info(f"  HF   tokens: {ref_tokens}")
    logger.info(f"  TTNN tokens: {tt_tokens}")

    # Pass/fail gate: flattened-logits PCC ≥ _PCC_FLOOR is the ONLY assertion. The per-position
    # PCC and greedy-token match logged above are diagnostics only — greedy agreement is
    # intentionally not enforced (bf4/bf8 quant makes exact argmax brittle; logit correlation
    # is the robust signal).
    passing, overall_msg = comp_pcc(ref_logits.flatten(), tt_logits.flatten(), _PCC_FLOOR)
    logger.info(f"Overall flattened logits PCC: {overall_msg}")
    assert passing, (
        f"End-to-end multimodal PCC below floor {_PCC_FLOOR}.\n"
        f"mean per-pos PCC={mean_pcc:.4f}, min per-pos PCC={min_pcc:.4f}, "
        f"greedy match={match}/{seq_len}\n{overall_msg}"
    )
    logger.info(f"✅ PASSED — end-to-end multimodal PCC ≥ {_PCC_FLOOR}")

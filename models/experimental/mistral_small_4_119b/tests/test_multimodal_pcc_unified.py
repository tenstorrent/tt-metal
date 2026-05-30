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
    export MISTRAL4_MM_TEXT_LAYERS=36
    export MISTRAL4_MM_VISION_LAYERS=24
    export MISTRAL4_MM_IMAGE=path/to/img.jpg
    export MESH_DEVICE=P150x8
    pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_pcc_unified.py -v -s --timeout=0
"""

from __future__ import annotations

import copy
import gc
import os
import psutil

import pytest
import torch
from loguru import logger

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
    except Exception:
        pass


pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral3.modeling_mistral3", reason="Mistral3 required")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")
pytest.importorskip("transformers.models.pixtral.modeling_pixtral", reason="Pixtral required")


_TEXT_LAYERS = int(os.environ.get("MISTRAL4_MM_TEXT_LAYERS", "2"))
_VISION_LAYERS = int(os.environ.get("MISTRAL4_MM_VISION_LAYERS", "2"))
_IMAGE_PATH = os.environ.get("MISTRAL4_MM_IMAGE", "")
_PROMPT = os.environ.get("MISTRAL4_MM_PROMPT", "Describe this image.")
_IMAGE_MAX_SIDE = int(os.environ.get("MISTRAL4_MM_IMAGE_MAX_SIDE", "224"))
_IMG_PATCHES = int(os.environ.get("MISTRAL4_MM_IMG_PATCHES", "10"))
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
                setattr(sub, attr, "eager")

    if n_text >= 36 and n_vision >= 24:
        logger.info("Enabling activation checkpointing for full-layer model (36+24)…")
        cfg.text_config.gradient_checkpointing = True
        cfg.vision_config.gradient_checkpointing = True

    with init_empty_weights():
        model = Mistral3ForConditionalGeneration(cfg)

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
    return model.eval()


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_MM_PCC") != "1",
    reason="Set MISTRAL4_MM_PCC=1 to run.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_multimodal_pcc_unified(reset_seeds, mesh_device):
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
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_TEXT_LAYERS, _VISION_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    # ── Build inputs via HF chat-template processor ───────────────────
    from PIL import Image
    from transformers import AutoProcessor

    if _IMAGE_PATH and os.path.exists(_IMAGE_PATH):
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
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": _PROMPT},
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
        f"Config: text_layers={_TEXT_LAYERS}, vision_layers={_VISION_LAYERS}, "
        f"pixel_values {tuple(pixel_values.shape)}, {num_image_tokens} image tokens, "
        f"seq_len={seq_len}, prompt={_PROMPT!r}"
    )

    # ── HF reference forward ─────────────────────────────────────────
    _log_memory_usage("before HF build")
    logger.info("Building HF Mistral3ForConditionalGeneration (CPU, bf16)…")
    hf_model = _build_hf_mm_ref(cfg, state_dict, _TEXT_LAYERS, _VISION_LAYERS)
    _log_memory_usage("after HF build, before forward")

    logger.info("Running HF reference forward (activation checkpointing enabled)…")
    try:
        with torch.inference_mode():
            hf_out = hf_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                image_sizes=image_sizes,
            )
            ref_logits = hf_out.logits[0].float()
            ref_logits = ref_logits.detach().clone()
            _log_memory_usage("after HF forward")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            logger.error(
                f"OOM during HF forward: {e}\n"
                f"Config: text={_TEXT_LAYERS} layers, vision={_VISION_LAYERS} layers\n"
                f"Memory optimization: activation checkpointing enabled for full model.\n"
                f"If still OOM, try reducing MISTRAL4_MM_TEXT_LAYERS and/or MISTRAL4_MM_VISION_LAYERS,\n"
                f"or run on a machine with more CPU RAM (>256GB recommended for full model)."
            )
        raise

    del hf_model, hf_out
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
        num_text_layers=_TEXT_LAYERS,
        num_vision_layers=_VISION_LAYERS,
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

    passing, overall_msg = comp_pcc(ref_logits.flatten(), tt_logits.flatten(), _PCC_FLOOR)
    logger.info(f"Overall flattened logits PCC: {overall_msg}")
    assert passing, (
        f"End-to-end multimodal PCC below floor {_PCC_FLOOR}.\n"
        f"mean per-pos PCC={mean_pcc:.4f}, min per-pos PCC={min_pcc:.4f}, "
        f"greedy match={match}/{seq_len}\n{overall_msg}"
    )
    logger.info(f"✅ PASSED — end-to-end multimodal PCC ≥ {_PCC_FLOOR}")

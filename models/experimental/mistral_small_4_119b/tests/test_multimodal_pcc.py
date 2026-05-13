# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end multimodal PCC test — TTNN ``TtMistral3ForConditionalGeneration``
vs HF ``Mistral3ForConditionalGeneration``.

Both models receive the same random ``pixel_values`` and ``input_ids`` (with
the model's ``image_token_id`` placed at the correct number of slots), then
produce ``[1, seq_len, vocab_size]`` logits. We compare per-position and
overall PCC.

PCC notes
---------
The end-to-end stack accumulates bf16 error from (a) the vision tower's 24
attention/MLP blocks, (b) the multi-modal projector, (c) the text model's 36
decoder layers with MoE top-k routing in float vs bf16. Realistic floor is
0.85 overall, with per-position numbers typically ≥0.95 at the last position.
For a 2 + 2 layer reduced run, expect ≥0.95 overall.

Run::

    export MISTRAL4_MM_PCC=1
    export MISTRAL4_MM_TEXT_LAYERS=2     # default 2
    export MISTRAL4_MM_VISION_LAYERS=2   # default 2
    export MISTRAL4_MM_IMG_PATCHES=10    # patches per side; default 10
    export MISTRAL4_MM_TEXT_LEN=8        # flanking text tokens; default 8
    export MESH_DEVICE=T3K
    pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_pcc.py -v -s --timeout=0

For the full configuration, set ``MISTRAL4_MM_TEXT_LAYERS=36`` and
``MISTRAL4_MM_VISION_LAYERS=24`` — the HF reference forward will take a few
minutes on CPU.
"""

from __future__ import annotations

import copy
import gc
import os

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
from models.experimental.mistral_small_4_119b.tt.mistral3_for_conditional_generation import (
    TtMistral3ForConditionalGeneration,
)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral3.modeling_mistral3", reason="Mistral3 required")
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")
pytest.importorskip("transformers.models.pixtral.modeling_pixtral", reason="Pixtral required")


_TEXT_LAYERS = int(os.environ.get("MISTRAL4_MM_TEXT_LAYERS", "2"))
_VISION_LAYERS = int(os.environ.get("MISTRAL4_MM_VISION_LAYERS", "2"))
_IMG_PATCHES = int(os.environ.get("MISTRAL4_MM_IMG_PATCHES", "10"))
_TEXT_LEN = int(os.environ.get("MISTRAL4_MM_TEXT_LEN", "8"))
_PCC_FLOOR = 0.85


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


# ── HF param name → state-dict key mapping ─────────────────────────────────
#
# The on-disk checkpoint was saved when the model was Mistral4ForCausalLM at the
# top level; HF's current Mistral3ForConditionalGeneration uses different prefixes.
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
    """
    Build HF ``Mistral3ForConditionalGeneration`` with truncated layer counts.

    Streams weights via ``accelerate.init_empty_weights`` to avoid a second
    in-memory copy of the (large) checkpoint. FP8 keys in the state dict are
    dequantised on the fly to bfloat16.
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

    cfg = copy.deepcopy(full_config)
    cfg.text_config.num_hidden_layers = n_text
    cfg.vision_config.num_hidden_layers = n_vision
    # Force eager attention so we don't depend on flash-attn at test time.
    for sub in (cfg.text_config, cfg.vision_config):
        for attr in ("attn_implementation", "_attn_implementation"):
            if hasattr(sub, attr):
                setattr(sub, attr, "eager")

    with init_empty_weights():
        model = Mistral3ForConditionalGeneration(cfg)

    missing = []
    for name, _ in model.named_parameters():
        sd_key = _hf_param_to_sd_key(name)
        if sd_key not in state_dict:
            missing.append(name)
            continue
        v = state_dict[sd_key]
        # Dequantise FP8 weights on the fly (text-model experts in the original
        # checkpoint are stored as FP8 with a companion ``_scale_inv`` tensor).
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
def test_mistral_small_4_multimodal_pcc(reset_seeds, mesh_device):
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    image_token_id = int(getattr(cfg, "image_token_index", 10))
    assert _IMG_PATCHES % MMP_SPATIAL_MERGE_SIZE == 0, f"--patches ({_IMG_PATCHES}) must be even for 2x2 merge"

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_TEXT_LAYERS, _VISION_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    # ── Build inputs ─────────────────────────────────────────────────────
    img_size = _IMG_PATCHES * VISION_PATCH_SIZE
    num_image_tokens = (_IMG_PATCHES // MMP_SPATIAL_MERGE_SIZE) ** 2
    text_before = _TEXT_LEN // 2
    text_after = _TEXT_LEN - text_before
    seq_len = text_before + num_image_tokens + text_after

    pixel_values = torch.rand(1, 3, img_size, img_size, dtype=torch.bfloat16) * 2 - 1

    rng = torch.Generator().manual_seed(0)
    txt_before = torch.randint(100, 1000, (text_before,), generator=rng, dtype=torch.long)
    txt_after = torch.randint(100, 1000, (text_after,), generator=rng, dtype=torch.long)
    img_slots = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)
    input_ids = torch.cat([txt_before, img_slots, txt_after]).unsqueeze(0)
    image_sizes = torch.tensor([[img_size, img_size]], dtype=torch.long)

    logger.info(
        f"Config: text_layers={_TEXT_LAYERS}, vision_layers={_VISION_LAYERS}, "
        f"image {img_size}×{img_size}, {num_image_tokens} image tokens, "
        f"seq_len={seq_len}"
    )

    # ── HF reference forward ─────────────────────────────────────────────
    logger.info("Building HF Mistral3ForConditionalGeneration (CPU, bf16)…")
    hf_model = _build_hf_mm_ref(cfg, state_dict, _TEXT_LAYERS, _VISION_LAYERS)
    logger.info("Running HF reference forward…")
    hf_out = hf_model(
        pixel_values=pixel_values,
        input_ids=input_ids,
        image_sizes=image_sizes,
    )
    ref_logits = hf_out.logits[0].float()  # [seq_len, vocab_size]
    del hf_model, hf_out
    gc.collect()
    logger.info(f"HF reference logits: {tuple(ref_logits.shape)}")

    # ── TTNN multimodal pipeline ─────────────────────────────────────────
    logger.info("Building TtMistral3ForConditionalGeneration (orchestrator)…")
    tt_model = TtMistral3ForConditionalGeneration(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=cfg.text_config,
        image_token_id=image_token_id,
        num_text_layers=_TEXT_LAYERS,
        num_vision_layers=_VISION_LAYERS,
        max_seq_len=seq_len + 16,
    )

    logger.info("Phase 1 — encode_image…")
    img_embeds_host = tt_model.encode_image(pixel_values)
    logger.info(f"image embeddings: {tuple(img_embeds_host.shape)} bf16 on host")

    logger.info("Phase 2 — load_text…")
    tt_model.load_text()

    # Position embeddings for prefill.
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    rotary = Mistral4RotaryEmbedding(cfg.text_config).eval().to(torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    cos, sin = rotary(dummy, pos_ids)

    logger.info("Phase 3 — prefill_multimodal_full_logits…")
    tt_logits = tt_model.prefill_multimodal_full_logits(
        img_embeds_host, input_ids, (cos, sin)
    )  # [1, seq_len, vocab_size] bf16 host
    tt_logits = tt_logits[0].float()  # [seq_len, vocab_size]

    assert (
        tt_logits.shape == ref_logits.shape
    ), f"shape mismatch: tt={tuple(tt_logits.shape)} vs ref={tuple(ref_logits.shape)}"

    # ── PCC per position + overall ───────────────────────────────────────
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
        f"Multimodal end-to-end PCC below floor {_PCC_FLOOR}.\n"
        f"mean per-pos PCC={mean_pcc:.4f}, min per-pos PCC={min_pcc:.4f}, "
        f"greedy match={match}/{seq_len}\n{overall_msg}"
    )
    logger.info(f"PASSED — end-to-end multimodal PCC ≥ {_PCC_FLOOR}")

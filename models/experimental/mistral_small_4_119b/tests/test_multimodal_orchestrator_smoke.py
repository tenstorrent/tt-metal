# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
TTNN-only smoke test for the full multimodal pipeline (phase-based):

    Phase 1 (vision)  : image  →  vision tower  →  projector  →  host img_embeds
                                   (vision/projector freed from DRAM)
    Phase 2 (load)    : text model construction on device
    Phase 3 (inference): host img_embeds  +  input_ids  →  prefill  →  one decode

No HF reference, no PCC. Just verifies the orchestrator wiring runs end-to-end
and produces valid token ids of the expected shape.

Run::

    export MISTRAL4_MM_SMOKE=1
    export MISTRAL4_MM_TEXT_LAYERS=2     # text decoder layers; default 2
    export MISTRAL4_MM_VISION_LAYERS=2   # pixtral layers; default 2
    export MISTRAL4_MM_IMG_PATCHES=10    # patches per side BEFORE merge; default 10
    export MISTRAL4_MM_TEXT_LEN=8        # non-image text tokens flanking the image; default 8
    export MESH_DEVICE=T3K
    pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_orchestrator_smoke.py -v -s --timeout=0

For the full configuration (36 text layers + 24 vision layers), set
``MISTRAL4_MM_TEXT_LAYERS=36`` and ``MISTRAL4_MM_VISION_LAYERS=24``.
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.experimental.mistral_small_4_119b.constants import (
    EXPECTED_VOCAB_SIZE,
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
pytest.importorskip("transformers.models.mistral4.modeling_mistral4", reason="Mistral4 required")
pytest.importorskip("transformers.models.pixtral.modeling_pixtral", reason="Pixtral required")
pytest.importorskip("transformers.models.mistral3.modeling_mistral3", reason="Mistral3 required")


_TEXT_LAYERS = int(os.environ.get("MISTRAL4_MM_TEXT_LAYERS", "2"))
_VISION_LAYERS = int(os.environ.get("MISTRAL4_MM_VISION_LAYERS", "2"))
_IMG_PATCHES = int(os.environ.get("MISTRAL4_MM_IMG_PATCHES", "10"))
_TEXT_LEN = int(os.environ.get("MISTRAL4_MM_TEXT_LEN", "8"))


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


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_MM_SMOKE") != "1",
    reason="Set MISTRAL4_MM_SMOKE=1 to run.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_multimodal_smoke(reset_seeds, mesh_device):
    from transformers import AutoConfig
    from transformers.models.mistral4.modeling_mistral4 import Mistral4RotaryEmbedding

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    text_cfg = cfg.text_config
    image_token_id = int(getattr(cfg, "image_token_index", 10))

    assert _IMG_PATCHES % MMP_SPATIAL_MERGE_SIZE == 0, f"--patches ({_IMG_PATCHES}) must be even for 2x2 merge"

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _state_dict_prefixes(_TEXT_LAYERS, _VISION_LAYERS))
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    # ── Build orchestrator (lightweight — defers all on-device construction) ──
    num_image_tokens = (_IMG_PATCHES // MMP_SPATIAL_MERGE_SIZE) ** 2
    text_before = _TEXT_LEN // 2
    text_after = _TEXT_LEN - text_before
    seq_len = text_before + num_image_tokens + text_after

    logger.info(
        f"Config: text_layers={_TEXT_LAYERS}, vision_layers={_VISION_LAYERS}, "
        f"image patches={_IMG_PATCHES}×{_IMG_PATCHES} → {num_image_tokens} image tokens, "
        f"text tokens={_TEXT_LEN}, total seq_len={seq_len}"
    )
    model = TtMistral3ForConditionalGeneration(
        mesh_device=mesh_device,
        state_dict=state_dict,
        text_config=text_cfg,
        image_token_id=image_token_id,
        num_text_layers=_TEXT_LAYERS,
        num_vision_layers=_VISION_LAYERS,
        max_seq_len=seq_len + 16,
    )

    # ── Build inputs ─────────────────────────────────────────────────────
    img_size = _IMG_PATCHES * VISION_PATCH_SIZE
    pixel_values = torch.rand(1, 3, img_size, img_size, dtype=torch.bfloat16) * 2 - 1

    # Random non-image token ids (avoid image_token_id collision) in [100, 1000).
    rng = torch.Generator().manual_seed(0)
    txt_before = torch.randint(100, 1000, (text_before,), generator=rng, dtype=torch.long)
    txt_after = torch.randint(100, 1000, (text_after,), generator=rng, dtype=torch.long)
    img_slots = torch.full((num_image_tokens,), image_token_id, dtype=torch.long)
    input_ids = torch.cat([txt_before, img_slots, txt_after]).unsqueeze(0)  # [1, seq_len]
    assert (input_ids == image_token_id).sum().item() == num_image_tokens

    # Precompute RoPE table for prefill + one decode step.
    rotary = Mistral4RotaryEmbedding(text_cfg).eval().to(torch.bfloat16)
    dummy = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16)
    pos_ids = torch.arange(seq_len + 1, dtype=torch.long).unsqueeze(0)
    cos_full, sin_full = rotary(dummy, pos_ids)

    def slice_rope(start: int, end: int):
        if cos_full.dim() == 3:
            return cos_full[:, start:end, :].contiguous(), sin_full[:, start:end, :].contiguous()
        return cos_full[:, :, start:end, :].contiguous(), sin_full[:, :, start:end, :].contiguous()

    # ── Phase 1: vision (vision tower + projector are loaded here and freed before phase 2) ──
    logger.info("Phase 1 — encode_image (vision tower + projector → host img_embeds)…")
    img_embeds_host = model.encode_image(pixel_values)
    assert img_embeds_host.shape == (
        num_image_tokens,
        4096,
    ), f"image embeddings have wrong shape: {tuple(img_embeds_host.shape)}"
    logger.info(f"image embeddings: {tuple(img_embeds_host.shape)} bf16 on host")

    # ── Phase 2: text model load ─────────────────────────────────────────
    logger.info("Phase 2 — load_text (text model construction on device)…")
    model.load_text()

    # ── Phase 3: prefill + one decode step ───────────────────────────────
    logger.info("Phase 3a — prefill_multimodal…")
    next_id = model.prefill_multimodal(img_embeds_host, input_ids, slice_rope(0, seq_len))
    assert (
        isinstance(next_id, int) and 0 <= next_id < EXPECTED_VOCAB_SIZE
    ), f"prefill returned invalid token id {next_id}"
    logger.info(f"Prefill next-token id: {next_id}")

    logger.info("Phase 3b — one decode step…")
    nxt = torch.tensor([[next_id]], dtype=torch.long)
    decoded = model.decode_next_token(nxt, slice_rope(seq_len, seq_len + 1), seq_len)
    assert (
        isinstance(decoded, int) and 0 <= decoded < EXPECTED_VOCAB_SIZE
    ), f"decode returned invalid token id {decoded}"
    logger.info(f"Decode next-token id: {decoded}")

    logger.info("PASSED — multimodal orchestrator prefill + decode produced valid token ids")

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-modal projector PCC test — TTNN vs HF reference.

Feeds random vision-tower-shaped features through both:
  - HF ``Mistral3MultiModalProjector`` (CPU, bf16)
  - ``TtMistral3MultiModalProjector``  (device, bf16)
and compares per-token outputs.

PCC notes:
  No MoE, no quantization. Tight floor (0.99 overall) is realistic since the
  only divergence sources are bf16 mat-mul accumulation order and ttnn tile
  reductions.

Run::

    export MISTRAL4_MMP_PCC=1
    export MISTRAL4_MMP_IMG_PATCHES=10   # patches per side BEFORE merge (must be even); default 10
    export MESH_DEVICE=P150x8
    pytest models/experimental/mistral_small_4_119b/tests/test_multimodal_projector_pcc.py -v -s --timeout=0
"""

from __future__ import annotations

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
    VISION_HIDDEN_SIZE,
    VISION_PATCH_SIZE,
)
from models.experimental.mistral_small_4_119b.tests.mesh_param import mesh_device_request_param
from models.experimental.mistral_small_4_119b.tt.mistral4_multimodal_projector import (
    TtMistral3MultiModalProjector,
)
from models.tt_transformers.tt.load_checkpoints import load_hf_state_dict_filtered

pytest.importorskip("transformers")
pytest.importorskip("transformers.models.mistral3.modeling_mistral3", reason="Mistral3 required")


_IMG_PATCHES = int(os.environ.get("MISTRAL4_MMP_IMG_PATCHES", "10"))
_PCC_FLOOR = 0.99

_MMP_PREFIXES = ("multi_modal_projector.",)


def _mesh_params():
    shape = mesh_device_request_param()
    base = {"trace_region_size": 30_000_000, "num_command_queues": 1}
    fabric = ttnn.FabricConfig.DISABLED if shape == (1, 1) else ttnn.FabricConfig.FABRIC_1D
    return [pytest.param(shape, {**base, "fabric_config": fabric}, id=f"mesh{shape[0]}x{shape[1]}")]


def _build_hf_projector_ref(full_config, state_dict: dict):
    """
    Build HF ``Mistral3MultiModalProjector`` with weights from our filtered state dict.

    Uses ``accelerate.init_empty_weights`` so the model shell costs ~0 RAM.
    """
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device
    from transformers.models.mistral3.modeling_mistral3 import Mistral3MultiModalProjector

    with init_empty_weights():
        model = Mistral3MultiModalProjector(full_config)

    missing = []
    for name, _ in model.named_parameters():
        key = f"multi_modal_projector.{name}"
        if key not in state_dict:
            missing.append(name)
            continue
        set_module_tensor_to_device(model, name, "cpu", value=state_dict[key].to(torch.bfloat16))

    if missing:
        logger.warning(f"HF projector missing keys: {missing}")
    return model.eval()


@torch.no_grad()
@run_for_wormhole_b0_or_blackhole()
@pytest.mark.skipif(
    os.environ.get("MISTRAL4_MMP_PCC") != "1",
    reason="Set MISTRAL4_MMP_PCC=1 to run.",
)
@pytest.mark.parametrize("mesh_device, device_params", _mesh_params(), indirect=True)
def test_mistral_small_4_mmp_pcc(reset_seeds, mesh_device):
    """Compare TTNN multi-modal projector outputs to HF reference."""
    from transformers import AutoConfig

    try:
        cfg = AutoConfig.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    except Exception as exc:
        pytest.skip(f"Could not load HF config: {exc}")

    assert _IMG_PATCHES % MMP_SPATIAL_MERGE_SIZE == 0, f"--patches ({_IMG_PATCHES}) must be even for 2x2 merge"

    try:
        state_dict = load_hf_state_dict_filtered(HF_MODEL_ID, _MMP_PREFIXES)
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"Checkpoint load failed: {exc}")

    # ── Random vision features (same on both paths) ──────────────────────
    h_patches = w_patches = _IMG_PATCHES
    num_patches = h_patches * w_patches
    image_features = torch.randn(num_patches, VISION_HIDDEN_SIZE, dtype=torch.bfloat16) * 0.1

    # Pixel-size image bounds for HF's patch_merger (it converts (H_px, W_px) → grid).
    image_sizes_px = torch.tensor([[h_patches * VISION_PATCH_SIZE, w_patches * VISION_PATCH_SIZE]], dtype=torch.long)
    logger.info(
        f"Patches {h_patches}×{w_patches} = {num_patches}; "
        f"after 2×2 merge: {(h_patches // 2) * (w_patches // 2)} tokens"
    )

    # ── HF reference ─────────────────────────────────────────────────────
    logger.info("Building HF Mistral3MultiModalProjector reference (CPU, bf16)…")
    hf_proj = _build_hf_projector_ref(cfg, state_dict)
    logger.info("Running HF reference forward…")
    ref_out = hf_proj(image_features, image_sizes_px).float()  # [num_out, 4096]
    del hf_proj
    gc.collect()
    logger.info(f"HF reference output: {tuple(ref_out.shape)}")

    # ── TTNN projector ───────────────────────────────────────────────────
    logger.info("Building TtMistral3MultiModalProjector…")
    tt_proj = TtMistral3MultiModalProjector(mesh_device=mesh_device, state_dict=state_dict)

    # Upload the same vision features as a [1, 1, num_patches, vision_hidden] device tensor.
    feats_tt = ttnn.as_tensor(
        image_features.reshape(1, 1, num_patches, VISION_HIDDEN_SIZE),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    logger.info("Running TTNN projector forward…")
    out_tt = tt_proj.forward(feats_tt, h_patches, w_patches)
    ttnn.deallocate(feats_tt)

    tt_features = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0),
    )[
        0, 0
    ].float()  # [num_out, 4096]
    ttnn.deallocate(out_tt)

    assert (
        tt_features.shape == ref_out.shape
    ), f"shape mismatch: tt={tuple(tt_features.shape)} vs ref={tuple(ref_out.shape)}"

    # ── Per-token + overall PCC ──────────────────────────────────────────
    num_out = ref_out.shape[0]
    pccs = []
    for i in range(num_out):
        _, msg = comp_pcc(ref_out[i], tt_features[i], _PCC_FLOOR)
        pcc_val = float(msg.split("=")[-1].strip() if "=" in str(msg) else msg)
        pccs.append(pcc_val)

    mean_pcc = sum(pccs) / len(pccs)
    min_pcc = min(pccs)
    logger.info(f"Per-token PCC: mean={mean_pcc:.4f}, min={min_pcc:.4f}  (floor {_PCC_FLOOR})")

    passing, overall_msg = comp_pcc(ref_out.flatten(), tt_features.flatten(), _PCC_FLOOR)
    logger.info(f"Overall flattened PCC: {overall_msg}")
    assert passing, (
        f"Projector PCC below floor {_PCC_FLOOR}.\n"
        f"mean per-token PCC={mean_pcc:.4f}, min per-token PCC={min_pcc:.4f}\n"
        f"{overall_msg}"
    )
    logger.info(f"PASSED — TTNN projector outputs within PCC≥{_PCC_FLOOR} of HF reference")

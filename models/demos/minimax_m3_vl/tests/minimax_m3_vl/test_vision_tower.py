# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Full-tower PCC test for M3VLVisionModel vs the HF reference goldens.

Two checks per grid:
  - `forward_tower` (patch_embed -> pre_layrnorm -> 32 blocks) vs `tower_out`.
  - full `forward` (tower + projector) vs `final`.

These accumulate bf16 error across 32 layers, so the threshold is 0.95
(per the plan's full-tower target).

Run:
    MESH_DEVICE=N150 TT_METAL_LOGGER_LEVEL=FATAL \
        pytest models/demos/minimax_m3_vl/tests/minimax_m3_vl/test_vision_tower.py -q
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3_vl.tt.model import M3VLVisionModel


def _to_torch(out_tt, mesh_device, shape):
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    out = ttnn.to_torch(out_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None)
    if is_mesh and out.shape[0] != 1:
        out = out[:1]
    return out.view(*shape)


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("grid_tag", ["224x224", "448x448"])
def test_m3vl_vision_tower(mesh_device, model_args, reference, goldens, grid_tag):
    """Full vision tower (and tower+projector) match the HF goldens at PCC >= 0.95."""
    pcc_threshold = 0.95
    g = goldens(grid_tag)
    pixel_values = g["pixel_values"].float()  # (L, 1176)
    grid = g["image_grid_thw"]
    tower_ref = g["tower_out"].float()  # (L, 1280)
    final_ref = g["final"].float()  # (L//4, 6144)

    model = M3VLVisionModel.from_torch(mesh_device, reference, model_args, with_projector=True, dtype=ttnn.bfloat16)

    # Tower-only.
    tower_tt = _to_torch(model.forward_tower(pixel_values, grid), mesh_device, tower_ref.shape)
    t_pass, t_msg = comp_pcc(tower_ref, tower_tt, pcc_threshold)
    t_rel = (torch.linalg.norm(tower_tt.float() - tower_ref) / torch.linalg.norm(tower_ref)).item()
    logger.info(f"[tower {grid_tag}] {comp_allclose(tower_ref, tower_tt)} {t_msg} relL2={t_rel:.4f}")

    # Full (tower + projector).
    final_tt = _to_torch(model.forward(pixel_values, grid), mesh_device, final_ref.shape)
    f_pass, f_msg = comp_pcc(final_ref, final_tt, pcc_threshold)
    f_rel = (torch.linalg.norm(final_tt.float() - final_ref) / torch.linalg.norm(final_ref)).item()
    logger.info(f"[final {grid_tag}] {comp_allclose(final_ref, final_tt)} {f_msg} relL2={f_rel:.4f}")

    assert t_pass, f"tower PCC below threshold for {grid_tag}: {t_msg}"
    assert f_pass, f"final PCC below threshold for {grid_tag}: {f_msg}"

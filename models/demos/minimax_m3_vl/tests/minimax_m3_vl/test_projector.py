# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLProjector vs the HF reference golden.

The projector consumes the tower output (`tower_out`, (L, 1280)) and
produces the merged vision tokens (`final`, (L//4, 6144)).

Run:
    MESH_DEVICE=N150 TT_METAL_LOGGER_LEVEL=FATAL \
        pytest models/demos/minimax_m3_vl/tests/minimax_m3_vl/test_projector.py -q
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3_vl.tt.projector import M3VLProjector


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("grid_tag", ["224x224", "448x448"])
def test_m3vl_projector(mesh_device, model_args, reference, goldens, grid_tag):
    """projector(tower_out) matches the HF golden `final` at PCC >= 0.99."""
    pcc_threshold = 0.99
    g = goldens(grid_tag)
    x_in = g["tower_out"].float()  # (L, 1280)
    ref_out = g["final"].float()  # (L//4, 6144)

    tt_proj = M3VLProjector.from_torch(
        mesh_device, reference.projector, spatial_merge_size=model_args.spatial_merge_size, dtype=ttnn.bfloat16
    )

    L, D = x_in.shape
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_tt = ttnn.from_torch(
        x_in.to(torch.bfloat16).view(1, 1, L, D).contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    out_tt = ttnn.to_torch(
        tt_proj(x_tt),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_tt.shape[0] != 1:
        out_tt = out_tt[:1]
    Lout, Dout = ref_out.shape
    out_tt = out_tt.view(Lout, Dout)

    passing, pcc_msg = comp_pcc(ref_out, out_tt, pcc_threshold)
    rel_l2 = (torch.linalg.norm(out_tt.float() - ref_out) / torch.linalg.norm(ref_out)).item()
    logger.info(f"[projector {grid_tag} L={L}->{Lout}] {comp_allclose(ref_out, out_tt)} {pcc_msg} relL2={rel_l2:.4f}")
    assert passing, f"projector PCC below threshold for {grid_tag}: {pcc_msg}"

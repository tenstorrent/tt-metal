# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLMLP vs the HF reference (golden activations).

Weights come from the checkpoint shards (via `_m3_loader`); the input/output
activations come from goldens generated in the transformers-5.12 venv
(`tests/gen_goldens.py`, layer-0 `mlp` forward hook).

Run:
    MESH_DEVICE=N150 TT_METAL_LOGGER_LEVEL=FATAL \
        pytest models/demos/minimax_m3_vl/tests/minimax_m3_vl/test_mlp.py -q
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3_vl.tt.mlp import M3VLMLP


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("grid_tag", ["224x224", "448x448"])
def test_m3vl_mlp(mesh_device, model_args, reference, goldens, grid_tag):
    """encoder layer-0 MLP matches the HF golden at PCC >= 0.99."""
    pcc_threshold = 0.99
    g = goldens(grid_tag)
    x_in = g["mlp.in"].float()  # (L, 1280)
    ref_out = g["mlp.out"].float()

    tt_mlp = M3VLMLP.from_torch(mesh_device, reference.layers[0].mlp, dtype=ttnn.bfloat16)

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
        tt_mlp(x_tt),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_tt.shape[0] != 1:
        out_tt = out_tt[:1]
    out_tt = out_tt.view(L, D)

    passing, pcc_msg = comp_pcc(ref_out, out_tt, pcc_threshold)
    rel_l2 = (torch.linalg.norm(out_tt.float() - ref_out) / torch.linalg.norm(ref_out)).item()
    logger.info(f"[mlp {grid_tag} L={L}] {comp_allclose(ref_out, out_tt)} {pcc_msg} relL2={rel_l2:.4f}")
    assert passing, f"mlp PCC below threshold for {grid_tag}: {pcc_msg}"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLPatchEmbed (Conv3d-as-Linear) vs the HF reference.

The M3 image processor pre-flattens patches, so the golden `pixel_values`
is already `(L, 1176)`. M3-VL has no class token and no learned posemb, so
the embeddings output feeds straight into `pre_layrnorm` — i.e. the
patch-embed output equals the golden `pre_layrnorm.in`.

Run:
    MESH_DEVICE=N150 TT_METAL_LOGGER_LEVEL=FATAL \
        pytest models/demos/minimax_m3_vl/tests/minimax_m3_vl/test_patch_embed.py -q
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3_vl.tt.patch_embed import M3VLPatchEmbed


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("grid_tag", ["224x224", "448x448"])
def test_m3vl_patch_embed(mesh_device, model_args, reference, goldens, grid_tag):
    """Conv3d-as-Linear patch projection matches the HF golden at PCC >= 0.9999."""
    pcc_threshold = 0.9999  # pure linear projection — should be near-exact
    g = goldens(grid_tag)
    x_in = g["pixel_values"].float()  # (L, 1176)
    ref_out = g["pre_layrnorm.in"].float()  # (L, 1280) == embeddings output

    tt_pe = M3VLPatchEmbed.from_torch(mesh_device, reference.patch_embed, dtype=ttnn.bfloat16)

    L, Din = x_in.shape
    Dout = ref_out.shape[1]
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_tt = ttnn.from_torch(
        x_in.to(torch.bfloat16).view(1, 1, L, Din).contiguous(),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    out_tt = ttnn.to_torch(
        tt_pe(x_tt),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_tt.shape[0] != 1:
        out_tt = out_tt[:1]
    out_tt = out_tt.view(L, Dout)

    passing, pcc_msg = comp_pcc(ref_out, out_tt, pcc_threshold)
    rel_l2 = (torch.linalg.norm(out_tt.float() - ref_out) / torch.linalg.norm(ref_out)).item()
    logger.info(f"[patch_embed {grid_tag} L={L}] {comp_allclose(ref_out, out_tt)} {pcc_msg} relL2={rel_l2:.4f}")
    assert passing, f"patch_embed PCC below threshold for {grid_tag}: {pcc_msg}"

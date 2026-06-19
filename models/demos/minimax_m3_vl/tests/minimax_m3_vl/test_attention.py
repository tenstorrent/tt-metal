# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
PCC test for M3VLAttention vs the HF reference (golden activations).

The encoder block applies layer_norm1 then self_attn, so the attention
input is the golden `layer_norm1.out`; the expected output is `attn.out`.
cos/sin are computed host-side from `image_grid_thw`.

Run:
    MESH_DEVICE=N150 TT_METAL_LOGGER_LEVEL=FATAL \
        pytest models/demos/minimax_m3_vl/tests/minimax_m3_vl/test_attention.py -q
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.minimax_m3_vl.tt.attention import M3VLAttention
from models.demos.minimax_m3_vl.tt.rope import rope_cos_sin_padded


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize("grid_tag", ["224x224", "448x448"])
def test_m3vl_attention(mesh_device, model_args, reference, goldens, grid_tag):
    """encoder layer-0 self-attention matches the HF golden at PCC >= 0.99."""
    pcc_threshold = 0.99
    g = goldens(grid_tag)
    x_in = g["layer_norm1.out"].float()  # attn input (L, 1280)
    ref_out = g["attn.out"].float()  # (L, 1280)
    grid = g["image_grid_thw"].to(torch.int64)

    tt_attn = M3VLAttention.from_torch(
        mesh_device,
        reference.layers[0].self_attn,
        hidden_size=model_args.hidden_size,
        num_heads=model_args.num_attention_heads,
        head_dim=model_args.head_dim,
        dtype=ttnn.bfloat16,
    )

    cos_pt, sin_pt = rope_cos_sin_padded(
        grid,
        head_dim=model_args.head_dim,
        padded_head_dim=model_args.padded_head_dim,
        theta=model_args.rope_theta,
        spatial_merge_size=model_args.spatial_merge_size,
    )
    cos_tt, sin_tt = tt_attn.stage_cos_sin(cos_pt, sin_pt)

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
        tt_attn(x_tt, cos_tt, sin_tt),
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_tt.shape[0] != 1:
        out_tt = out_tt[:1]
    out_tt = out_tt.view(L, D)

    passing, pcc_msg = comp_pcc(ref_out, out_tt, pcc_threshold)
    rel_l2 = (torch.linalg.norm(out_tt.float() - ref_out) / torch.linalg.norm(ref_out)).item()
    logger.info(f"[attention {grid_tag} L={L}] {comp_allclose(ref_out, out_tt)} {pcc_msg} relL2={rel_l2:.4f}")
    assert passing, f"attention PCC below threshold for {grid_tag}: {pcc_msg}"

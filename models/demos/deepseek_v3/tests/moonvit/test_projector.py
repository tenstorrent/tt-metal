# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
PCC test for MoonViTProjector.

Compares our projector against HF KimiVLMultiModalProjector.forward.
Input is the 3D merged form `(L_new, merge_groups, vision_hidden)` per
image — what `patch_merger` produces (one tensor per image).

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_projector.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.projector import MoonViTProjector


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_shapes",
    [
        [(64, 1152)],  # single image, L_new=64.
        [(96, 1152), (256, 1152)],  # two images packed.
    ],
)
def test_moonvit_projector(mesh_device, model_args, image_shapes):
    """Projector forward matches HF at PCC >= 0.99."""
    pcc_threshold = 0.99
    kh, kw = model_args.merge_kernel_size
    merge_groups = kh * kw  # 4

    # 1. HF reference.
    ref_proj = model_args.reference_projector()
    ref_proj_fp32 = ref_proj.float()
    text_hidden = ref_proj.linear_2.out_features

    # 2. Build list of (L_new_i, merge_groups, vision_hidden) tensors — what patch_merger emits.
    torch.manual_seed(0)
    image_feats_fp32 = [
        torch.randn(l, merge_groups, model_args.hidden_size, dtype=torch.float32) for l, _ in image_shapes
    ]
    image_feats_bf16 = [t.to(torch.bfloat16) for t in image_feats_fp32]
    total_l_new = sum(l for l, _ in image_shapes)

    # HF expects a list, concats internally.
    ref_out = ref_proj_fp32(image_feats_fp32)
    assert ref_out.shape == (total_l_new, text_hidden)

    # 3. Build ttnn projector.
    tt_proj = MoonViTProjector.from_torch(
        mesh_device=mesh_device,
        ref=ref_proj,
        dtype=ttnn.bfloat16,
    )

    # 4. Concat list on host, push as (1, 1, L_total * merge_groups, vision_hidden).
    concat = torch.cat(image_feats_bf16, dim=0)  # (L_total, merge_groups, vision_hidden)
    # Flatten (L_total, merge_groups) -> L_total*merge_groups; keep vision_hidden last.
    x_flat = concat.reshape(total_l_new * merge_groups, model_args.hidden_size)
    x_4d = x_flat.view(1, 1, total_l_new * merge_groups, model_args.hidden_size).contiguous()

    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_tt = ttnn.from_torch(
        x_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    out_tt = tt_proj(x_tt)
    out_pt = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_pt.shape[0] != 1:
        out_pt = out_pt[:1]
    out_pt = out_pt.view(total_l_new, text_hidden)

    passing, pcc_msg = comp_pcc(ref_out, out_pt, pcc_threshold)
    logger.info(f"[shapes={image_shapes} L={total_l_new}] {comp_allclose(ref_out, out_pt)} {pcc_msg}")
    assert passing, f"PCC below threshold for {image_shapes}: {pcc_msg}"

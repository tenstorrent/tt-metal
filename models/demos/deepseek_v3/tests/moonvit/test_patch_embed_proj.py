# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Per-submodule PCC test for MoonVisionPatchProj (the Conv2d-only path).

Image processing pre-cuts the input into shape (L, 3, 14, 14), so the
Conv2d projection is equivalent to a per-patch Linear from 588 to 1152.
We test that equivalence: HF Conv2d(L, 3, 14, 14) vs ttnn Linear(L, 588)
with the reshaped Conv2d weight.

The full patch embed (Conv2d + posemb) is step 5; this test covers just
the projection.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_patch_embed_proj.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.patch_embed import MoonVisionPatchProj


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize(
    "grid_hw",
    [
        (16, 16),  # 256-patch image (e.g., 224x224)
        (32, 24),  # 768-patch image (asymmetric — exercises non-square grids)
    ],
)
def test_moonvit_patch_proj(mesh_device, model_args, grid_hw):
    """Per-patch projection matches HF Conv2d at PCC >= 0.99."""

    pcc_threshold = 0.99
    H, W = grid_hw
    L = H * W

    # 1. HF reference: pull the full patch_embed, take .proj (the Conv2d).
    ref_patch_embed = model_args.reference_patch_embed()
    ref_proj = ref_patch_embed.proj
    ref_proj_fp32 = ref_proj.float()
    assert ref_proj.weight.shape == (model_args.hidden_size, 3, 14, 14)

    # 2. ttnn module built from the same weights.
    tt_proj = MoonVisionPatchProj.from_torch(
        mesh_device=mesh_device,
        ref_proj=ref_proj,
        out_dim=model_args.hidden_size,
        dtype=ttnn.bfloat16,
    )

    # 3. Random per-patch input. Image-processor pre-patches images so the
    # tensor that hits self.proj is already (L, 3, 14, 14).
    torch.manual_seed(0)
    x_patches_fp32 = torch.randn(L, 3, 14, 14, dtype=torch.float32)
    x_patches_bf16 = x_patches_fp32.to(torch.bfloat16)

    # Torch reference: Conv2d on (L, 3, 14, 14) -> (L, 1152, 1, 1) -> (L, 1152).
    ref_out_4d = ref_proj_fp32(x_patches_fp32)
    ref_out = ref_out_4d.view(L, -1)  # (L, 1152)

    # TT-Metal: host-flatten (L, 3, 14, 14) -> (L, 588), then ttnn.linear.
    x_flat_pt = MoonVisionPatchProj.flatten_patches(x_patches_bf16)  # (L, 588)
    # Pad to a 4D layout that ttnn likes for matmul: (1, 1, L, 588).
    x_flat_4d = x_flat_pt.view(1, 1, L, -1).contiguous()

    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_tt = ttnn.from_torch(
        x_flat_4d,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )

    tt_out_tensor = tt_proj(x_tt)
    tt_out = ttnn.to_torch(
        tt_out_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and tt_out.shape[0] != 1:
        tt_out = tt_out[:1]
    tt_out = tt_out.view(L, model_args.hidden_size)

    # 4. PCC gate.
    passing, pcc_msg = comp_pcc(ref_out, tt_out, pcc_threshold)
    logger.info(f"[grid={grid_hw} L={L}] {comp_allclose(ref_out, tt_out)} {pcc_msg}")
    assert passing, f"PCC below threshold for grid={grid_hw}: {pcc_msg}"

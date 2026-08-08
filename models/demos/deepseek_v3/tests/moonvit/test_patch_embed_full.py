# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Per-submodule PCC test for the full MoonVisionPatchEmbed.

Composes the Conv2d-as-Linear patch projection (step 3) with the
host-bicubic 2D learned posemb (step 4). The test feeds the same
random (L, 3, 14, 14) patches and grid_hws into both HF and our
implementation, then PCC-compares the output.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_patch_embed_full.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit.patch_embed import MoonVisionPatchEmbed, MoonVisionPatchProj


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize(
    "grid_hws",
    [
        [[16, 16]],  # 224x224 — standard
        [[32, 24]],  # asymmetric
        [[64, 64]],  # matches base posemb shape (fast path)
        [[16, 16], [32, 24]],  # two images packed into one call
    ],
)
def test_moonvit_patch_embed_full(mesh_device, model_args, grid_hws):
    """Full patch embed (Conv2d + posemb) matches HF at PCC >= 0.99."""

    pcc_threshold = 0.99

    # 1. HF reference.
    ref_patch_embed = model_args.reference_patch_embed()
    ref_fp32 = ref_patch_embed.float()
    D = model_args.hidden_size

    # 2. Our composed ttnn module.
    tt_patch_embed = MoonVisionPatchEmbed.from_torch(
        mesh_device=mesh_device,
        ref=ref_patch_embed,
        dtype=ttnn.bfloat16,
    )

    # 3. Input — same per-patch tensor for both. Total L = sum of H*W.
    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    L = int(grid_tensor.prod(dim=1).sum().item())

    torch.manual_seed(0)
    x_patches_fp32 = torch.randn(L, 3, 14, 14, dtype=torch.float32)
    x_patches_bf16 = x_patches_fp32.to(torch.bfloat16)

    # Torch reference: HF forward returns (L, D).
    ref_out = ref_fp32(x_patches_fp32, grid_tensor)

    # TT-Metal: flatten patches host-side, push to device, run module.
    x_flat_pt = MoonVisionPatchProj.flatten_patches(x_patches_bf16).view(1, 1, L, -1).contiguous()
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    x_tt = ttnn.from_torch(
        x_flat_pt,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device) if is_mesh else None,
    )
    tt_out_tensor = tt_patch_embed(x_tt, grid_tensor)
    tt_out = ttnn.to_torch(
        tt_out_tensor,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and tt_out.shape[0] != 1:
        tt_out = tt_out[:1]
    tt_out = tt_out.view(L, D)

    # 4. PCC gate.
    passing, pcc_msg = comp_pcc(ref_out, tt_out, pcc_threshold)
    logger.info(f"[grid_hws={grid_hws} L={L}] {comp_allclose(ref_out, tt_out)} {pcc_msg}")
    assert passing, f"PCC below threshold for grid_hws={grid_hws}: {pcc_msg}"

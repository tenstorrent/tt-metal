# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
End-to-end PCC test for the full MoonViT vision tower.

Composes patch_embed + 27 encoder blocks + final_layernorm + patch_merger
+ projector. The PCC threshold is loosened to 0.95 to absorb the
compounded BF16 drift through 27 layers.

Run:
    HF_HOME=/localdev/zbaczewski/hf_cache MESH_DEVICE=N150 \\
        pytest models/demos/deepseek_v3/tests/moonvit/test_vision_transformer.py -v
"""
from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_allclose, comp_pcc
from models.demos.deepseek_v3.tt.moonvit._references import _vision_tower
from models.demos.deepseek_v3.tt.moonvit.model import MoonViT


@torch.no_grad()
@pytest.mark.parametrize(
    "mesh_device",
    [{"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(os.environ.get("MESH_DEVICE"), (1, 1))],
    indirect=True,
)
@pytest.mark.parametrize(
    "grid_hws",
    [
        [[16, 16]],  # single 224x224-equivalent image.
    ],
)
def test_moonvit_full_tower(mesh_device, model_args, grid_hws):
    """End-to-end vision tower forward matches HF at PCC >= 0.95."""
    pcc_threshold = 0.95

    grid_tensor = torch.tensor(grid_hws, dtype=torch.long)
    L = int(grid_tensor.prod(dim=1).sum().item())

    # 1. Input — random per-patch tensors.
    torch.manual_seed(0)
    pixel_patches = torch.randn(L, 3, 14, 14, dtype=torch.float32)

    # 2. HF reference end-to-end (vision_tower + projector).
    vt = _vision_tower(model_args)
    # Force sdpa on every layer for correct multi-image masking (doesn't affect single-image).
    for blk in vt.encoder.blocks:
        blk.attn_implementation = "sdpa"
    vt_fp32 = vt.float()
    proj_fp32 = model_args.reference_projector().float()

    ref_merged_list = vt_fp32(pixel_patches, grid_tensor)
    ref_out = proj_fp32(ref_merged_list)
    expected_l_new = sum((h // 2) * (w // 2) for h, w in grid_hws)
    text_hidden = proj_fp32.linear_2.out_features
    assert ref_out.shape == (expected_l_new, text_hidden)

    # 3. Build ttnn vision tower with projector.
    tt_tower = MoonViT.from_torch(
        model_args=model_args,
        mesh_device=mesh_device,
        with_projector=True,
        dtype=ttnn.bfloat16,
    )

    # 4. Run end-to-end.
    out_tt = tt_tower(pixel_patches, grid_tensor)
    is_mesh = type(mesh_device).__name__ == "MeshDevice"
    out_pt = ttnn.to_torch(
        out_tt,
        mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0) if is_mesh else None,
    )
    if is_mesh and out_pt.shape[0] != 1:
        out_pt = out_pt[:1]
    out_pt = out_pt.view(expected_l_new, text_hidden)

    passing, pcc_msg = comp_pcc(ref_out, out_pt, pcc_threshold)
    logger.info(f"[grid_hws={grid_hws} L_new={expected_l_new}] {comp_allclose(ref_out, out_pt)} {pcc_msg}")
    assert passing, f"end-to-end PCC below threshold for grid_hws={grid_hws}: {pcc_msg}"

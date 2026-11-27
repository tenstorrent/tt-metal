# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Device test of M3VLMLP against Hugging Face's MiniMaxM3VLVisionMLP.

It runs layer 0's MLP on its input in the reference tower, with fuse_gelu off, and with M3_VISION_ALL_GRIDS=1 also on.
The weights are seeded random values (tests/conftest.py), so no checkpoint is needed.
"""
from __future__ import annotations

import pytest
import torch

import ttnn
from models.demos.minimax_m3.tt.vision.mlp import M3VLMLP

from ..test_factory import parametrize_mesh_with_fabric
from .vision_helpers import (
    MAX_REL_L2,
    all_grids,
    assert_close,
    grid_name,
    parametrize_vision_grids,
    rows_to_device,
    to_host,
)

# fuse_gelu is off by default (mlp.py), so the CI entry checks only the separate GELU. M3_VISION_ALL_GRIDS=1 adds the
# fused one; otherwise it is left out rather than skipped, as the extra grids are.
FUSE_GELU = [pytest.param(False, id="separate_gelu")]
if all_grids():
    FUSE_GELU.append(pytest.param(True, id="fused_gelu"))


@torch.no_grad()
@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@parametrize_vision_grids()
@pytest.mark.parametrize("fuse_gelu", FUSE_GELU)
def test_vision_mlp_vs_ref(mesh_device, device_params, grid_thw, fuse_gelu, vision_reference, vision_activations):
    """Layer 0 MLP matches the reference at PCC >= 0.9999, within its relL2 bound and with no NaN or Inf, with
    fuse_gelu off, and with M3_VISION_ALL_GRIDS=1 also on."""
    x_in = vision_activations(grid_thw)["layer0.mlp.in"].bfloat16().float()
    mlp = vision_reference().tower.layers[0].mlp
    ref_out = mlp(x_in)

    tt_mlp = M3VLMLP.from_torch(mesh_device, mlp, dtype=ttnn.bfloat16)
    tt_mlp.fuse_gelu = fuse_gelu
    out = to_host(tt_mlp(rows_to_device(x_in, mesh_device)), mesh_device, ref_out.shape)

    name = f"{'mlp_fused_gelu' if fuse_gelu else 'mlp'} {grid_name(grid_thw)} L={x_in.shape[0]}"
    assert_close(name, ref_out, out, 0.9999, MAX_REL_L2["mlp"])

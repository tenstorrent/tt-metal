# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `RMSNorm` against each `nn.RMSNorm` of the reference, at the row count it sees in the model.

The encoder norms have `eps=None`, which torch resolves to `finfo(bf16).eps` for bf16 input, so the
golden runs in bf16 and the TT norm uses `config.norm_eps`.
"""

from __future__ import annotations

import pytest
import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    DEVICE_GRIDS,
    device_config,
    P150_AND_RING,
    REFERENCE,
    REFERENCE_LAYERS,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.rms_norm import RMSNorm

# Per-layer norms, under `vision_tower.encoder.blocks.<layer>`.
ENCODER_NORMS = ["norm0", "norm1"]


def check_rms_norm(mesh_device, sp_axis, torch_norm, eps: float, n: int, rows: int) -> None:
    (dim,) = torch_norm.normalized_shape
    tt = RMSNorm(dim, eps=eps, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config(mesh_device))
    tt.load_torch_state_dict(torch_norm.state_dict())

    x = torch.randn(n, dim, dtype=torch.bfloat16)
    with torch.no_grad():
        golden = torch_norm(x)

    out = tt(sp_shard(x, mesh_device, sp_axis, rows))
    assert_pcc(golden, sp_gather(out, mesh_device, sp_axis, n))


@P150_AND_RING
@REFERENCE_LAYERS
@DEVICE_GRIDS
@pytest.mark.parametrize("name", ENCODER_NORMS)
def test_encoder_rms_norm(mesh_device, sp_axis, num_links, config, reference, grid, layer, name):
    config = device_config(config, sp_axis)
    torch_norm = reference.vision_tower.encoder.blocks[layer].get_submodule(name)
    assert torch_norm.eps is None
    _, h, w = grid[0]
    check_rms_norm(mesh_device, sp_axis, torch_norm, config.norm_eps, h * w, config.max_patches)


@P150_AND_RING
@REFERENCE
@DEVICE_GRIDS
def test_final_rms_norm(mesh_device, sp_axis, num_links, config, reference, grid):
    config = device_config(config, sp_axis)
    torch_norm = reference.vision_tower.encoder.final_layernorm
    assert torch_norm.eps is None
    _, h, w = grid[0]
    check_rms_norm(mesh_device, sp_axis, torch_norm, config.norm_eps, h * w, config.max_patches)


@P150_AND_RING
@REFERENCE
@DEVICE_GRIDS
def test_projector_rms_norm(mesh_device, sp_axis, num_links, config, reference, grid):
    config = device_config(config, sp_axis)
    torch_norm = reference.mm_projector.post_norm
    assert torch_norm.eps == config.projector_norm_eps
    _, h, w = grid[0]
    n, rows = h * w // config.merge_area, config.max_patches // config.merge_area
    check_rms_norm(mesh_device, sp_axis, torch_norm, config.projector_norm_eps, n, rows)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Vision_MLP` against the reference `blocks[layer].mlp`."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    DEVICE_GRIDS,
    device_config,
    NOT_WRITTEN,
    REFERENCE_LAYERS,
    P150_AND_RING,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.vision_mlp import TT_Vision_MLP


@P150_AND_RING
@REFERENCE_LAYERS
@DEVICE_GRIDS
@NOT_WRITTEN
def test_vision_mlp(mesh_device, sp_axis, num_links, config, reference, grid, layer):
    config = device_config(config, sp_axis)
    tt = TT_Vision_MLP(config, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config(mesh_device))
    tt.load_torch_state_dict(reference.vision_tower.encoder.blocks[layer].mlp.state_dict())

    _, h, w = grid[0]
    x = torch.randn(h * w, config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        golden = reference.vision_tower.encoder.blocks[layer].mlp(x)

    out = tt(sp_shard(x, mesh_device, sp_axis, config.max_patches))
    assert_pcc(golden, sp_gather(out, mesh_device, sp_axis, h * w))

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Projector` against the reference `mm_projector`."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    device_config,
    row_grid,
    ROW_GRIDS,
    NOT_WRITTEN,
    REFERENCE,
    P150_AND_RING,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.projector import TT_Projector


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
@NOT_WRITTEN
def test_projector(mesh_device, sp_axis, num_links, config, reference, size):
    config = device_config(config, sp_axis)
    grid = row_grid(size, sp_axis)
    tt = TT_Projector(config, mesh_device=mesh_device, compute_kernel_config=compute_kernel_config(mesh_device))
    tt.load_torch_state_dict(reference.mm_projector.state_dict())

    _, h, w = grid[0]
    n = h * w // config.merge_area
    x = torch.randn(n, config.merge_area, config.hidden_size, dtype=torch.bfloat16)
    with torch.no_grad():
        (golden,) = reference.mm_projector([x])

    rows = config.max_patches // config.merge_area
    out = tt(sp_shard(x.reshape(n, config.merged_dim), mesh_device, sp_axis, rows))
    assert_pcc(golden, sp_gather(out, mesh_device, sp_axis, n))

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Patch_Merger` against the reference `tpool_patch_merger`."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.reference.modeling_kimi_k3 import tpool_patch_merger
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    assert_pcc,
    DEVICE_GRIDS,
    device_config,
    NOT_WRITTEN,
    P150_AND_RING,
    ring_parallel,
    sp_gather,
    sp_shard,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import resolve_parallel
from models.experimental.kimi_k3_moonvit_v2.tt.patch_merger import TT_Patch_Merger


@P150_AND_RING
@DEVICE_GRIDS
@NOT_WRITTEN
def test_patch_merger(mesh_device, sp_axis, num_links, config, grid):
    config = device_config(config, sp_axis)
    sequence_parallel, ccl = ring_parallel(mesh_device, sp_axis, num_links)
    parallel = resolve_parallel(mesh_device, sequence_parallel, ccl, config)
    tt = TT_Patch_Merger(config, mesh_device=mesh_device, parallel=parallel)

    _, h, w = grid[0]
    x = torch.randn(h * w, config.hidden_size, dtype=torch.bfloat16)
    (golden,) = tpool_patch_merger(x, torch.tensor(grid), merge_kernel_size=config.merge_kernel_size)
    golden = golden.reshape(-1, config.merged_dim)

    out = tt(sp_shard(x, mesh_device, sp_axis, config.max_patches), h, w)
    assert_pcc(golden, sp_gather(out, mesh_device, sp_axis, golden.shape[0]))

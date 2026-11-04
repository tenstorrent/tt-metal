# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Patch_Merger` on window-ordered rows against the reference `tpool_patch_merger` on raster
rows."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tt import host_prep
from models.experimental.kimi_k3_moonvit_v2.reference.modeling_kimi_k3 import tpool_patch_merger
from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    random_tensor,
    assert_quality,
    device_config,
    row_grid,
    ROW_GRIDS,
    MeshRows,
    P150_AND_RING,
)
from models.experimental.kimi_k3_moonvit_v2.tt.patch_merger import TT_Patch_Merger


@P150_AND_RING
@ROW_GRIDS
def test_patch_merger(mesh_device, sp_axis, num_links, config, size):
    tt = TT_Patch_Merger(device_config(config, sp_axis))

    grid = row_grid(size)
    _, h, w = grid[0]
    x = random_tensor(h * w, config.hidden_size)
    (golden,) = tpool_patch_merger(x, torch.tensor(grid), merge_kernel_size=config.merge_kernel_size)
    golden = golden.reshape(-1, config.merged_dim)

    # A window is 4 consecutive rows and each chip's share is whole windows, so the merged tokens on
    # the mesh are those of its patch rows.
    rows = MeshRows(h * w, config.max_patches, sp_axis)
    merged = MeshRows(h * w // config.merge_area, config.max_patches // config.merge_area, sp_axis)
    out = tt(rows.shard(host_prep.to_window_order(x, h, w, config.merge_kernel_size), mesh_device))
    # A pure copy: the result must match exactly.
    golden, actual = merged.take(golden), merged.gather(out, mesh_device)
    assert_quality(golden, actual)
    assert torch.equal(golden, actual)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Device: `TT_Projector` against the reference `mm_projector`."""

from __future__ import annotations

import torch

from models.experimental.kimi_k3_moonvit_v2.tests.common import (
    random_tensor,
    assert_quality,
    device_config,
    row_grid,
    ROW_GRIDS,
    NOT_WRITTEN,
    REFERENCE,
    MeshRows,
    P150_AND_RING,
)
from models.experimental.kimi_k3_moonvit_v2.tt.model_config import compute_kernel_config
from models.experimental.kimi_k3_moonvit_v2.tt.projector import TT_Projector


@P150_AND_RING
@REFERENCE
@ROW_GRIDS
@NOT_WRITTEN
def test_projector(mesh_device, sp_axis, num_links, config, reference, size):
    ckc = compute_kernel_config(mesh_device)
    tt = TT_Projector(device_config(config, sp_axis), mesh_device=mesh_device, compute_kernel_config=ckc)
    tt.load_torch_state_dict(reference.mm_projector.state_dict())

    ((_, h, w),) = row_grid(size)
    rows = MeshRows(h * w // config.merge_area, config.max_patches // config.merge_area, sp_axis)
    x = random_tensor(rows.n, config.merge_area, config.hidden_size)
    with torch.no_grad():
        (golden,) = reference.mm_projector([rows.take(x)])

    out = tt(rows.shard(x.reshape(rows.n, config.merged_dim), mesh_device))
    assert_quality(golden, rows.gather(out, mesh_device))

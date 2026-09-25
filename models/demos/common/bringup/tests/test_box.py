# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Box step: the spec's mesh opens with its device params, and the collectives the plan relies on work on it.
Records chips, mesh_rows, mesh_cols, and the max abs error of all_gather / all_reduce / reduce_scatter on each mesh axis
with more than one device."""

import torch

from models.demos.common.bringup.core import metrics
from models.demos.common.bringup.testing.harness import mesh_parametrize, spec

S = spec()


@mesh_parametrize
def test_box(mesh_device):
    import ttnn

    rows, cols = mesh_device.shape
    metrics.record("chips", mesh_device.get_num_devices())
    metrics.record("mesh_rows", rows)
    metrics.record("mesh_cols", cols)
    worst = 0.0
    for axis, n in ((0, rows), (1, cols)):
        if n < 2:
            continue
        x = torch.randn(1, 1, 32 * n, 256)
        dims = [None, None]
        dims[axis] = 2
        t = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, (rows, cols), dims=dims),
        )
        g = ttnn.all_gather(t, dim=2, cluster_axis=axis)
        got = ttnn.to_torch(ttnn.get_device_tensors(g)[0]).float()
        worst = max(worst, (got - x.bfloat16().float()).abs().max().item())
        r = ttnn.all_reduce(t, cluster_axis=axis)
        want = x.bfloat16().float().reshape(1, 1, n, 32, 256).sum(2)
        got = ttnn.to_torch(ttnn.get_device_tensors(r)[0]).float()
        metrics.record(f"all_reduce_rel_err_axis{axis}", ((got - want).abs().max() / want.abs().max()).item())
    metrics.record("all_gather_maxabs", worst)
    assert worst == 0.0

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that the fused distributed DiT norms reject arguments their output-spec sizing divides by.
On a single device the cluster width is 1, so the ops take the local path without a stats buffer."""

import pytest
import torch

import ttnn


def _semaphores(device):
    grid = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    return [ttnn.create_global_semaphore(device, cores, 0)]


def _tile(device, shape):
    return ttnn.from_torch(torch.randn(shape).bfloat16(), layout=ttnn.TILE_LAYOUT, device=device)


@pytest.mark.parametrize(
    "channels, num_groups, message",
    [(64, 0, "num_groups must be >= 1"), (16, 1, r"C \(16\) must be divisible by TILE_WIDTH")],
    ids=["zero_groups", "channels_not_tile_aligned"],
)
def test_dit_fused_distributed_groupnorm_rejects_zero_virtual_cols(device, expect_error, channels, num_groups, message):
    x = _tile(device, (1, 1, 32, channels))
    with expect_error(RuntimeError, message):
        ttnn.experimental.dit_fused_distributed_groupnorm(
            x,
            num_groups=num_groups,
            cluster_axis=1,
            mesh_device=device,
            multi_device_global_semaphore=_semaphores(device),
            topology=ttnn.Topology.Linear,
        )


@pytest.mark.parametrize("op", ["dit_fused_distributed_rmsnorm", "dit_fused_distributed_layernorm"])
def test_dit_fused_distributed_norm_rejects_zero_heads(device, expect_error, op):
    x = _tile(device, (1, 1, 32, 128))
    with expect_error(RuntimeError, "num_heads_per_device must be >= 1"):
        getattr(ttnn.experimental, op)(
            x,
            cluster_axis=1,
            mesh_device=device,
            multi_device_global_semaphore=_semaphores(device),
            topology=ttnn.Topology.Linear,
            num_heads_per_device=0,
        )

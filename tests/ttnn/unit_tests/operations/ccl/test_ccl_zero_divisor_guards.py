# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Checks that experimental CCL ops reject arguments their host-side sizing divides by. On a single
device the cluster width is 1, so the DiT norms take the local path without a stats buffer, and the
argument checks run before any check that needs more than one device."""

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


@pytest.mark.parametrize(
    "num_links, pad2_num_links, message",
    [(0, 1, "np_num_links must be > 0"), (1, 0, "np_pad2_num_links must be > 0")],
    ids=["zero_h_links", "zero_w_links"],
)
def test_neighbor_pad_halo_rejects_zero_links(device, expect_error, num_links, pad2_num_links, message):
    x = ttnn.from_torch(torch.randn(1, 1, 8, 8, 32).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    sems = [_semaphores(device)[0] for _ in range(3)]
    with expect_error(RuntimeError, message):
        ttnn.experimental.neighbor_pad_halo(
            x,
            x,
            np_padding_h=1,
            np_padding_w=1,
            np_cluster_axis=0,
            np_num_links=num_links,
            np_topology=ttnn.Topology.Linear,
            h_neighbor_semaphore=sems[0],
            barrier_semaphore=sems[1],
            w_neighbor_semaphore=sems[2],
            np_pad_dim2=3,
            np_pad2_left=1,
            np_pad2_right=1,
            np_pad2_cluster_axis=1,
            np_pad2_num_links=pad2_num_links,
        )


# With more than one virtual column the factory wraps the gamma/beta and mask tile offsets modulo their tile counts.
@pytest.mark.parametrize(
    "mask_shape, weight_shape, message",
    [
        ((0, 2, 32, 32), None, "input_mask must be non-empty"),
        ((1, 2, 32, 32), (1, 1, 0, 32), "Weight must be non-empty"),
    ],
    ids=["empty_mask", "empty_weight"],
)
def test_dit_fused_distributed_groupnorm_rejects_empty_mask_and_affine(
    device, expect_error, mask_shape, weight_shape, message
):
    x = _tile(device, (1, 1, 32, 64))
    mask = _tile(device, mask_shape)
    weight = None
    if weight_shape is not None:
        weight = ttnn.from_torch(torch.ones(weight_shape).bfloat16(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with expect_error(RuntimeError, message):
        ttnn.experimental.dit_fused_distributed_groupnorm(
            x,
            num_groups=2,
            cluster_axis=1,
            mesh_device=device,
            multi_device_global_semaphore=_semaphores(device),
            topology=ttnn.Topology.Linear,
            input_mask=mask,
            weight=weight,
        )

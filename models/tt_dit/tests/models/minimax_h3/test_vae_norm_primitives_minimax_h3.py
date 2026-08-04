# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Primitives probe for a spatially-distributed per-frame GroupNorm.

A fused distributed GroupNorm device op cannot serve H3, which is why these primitives
exist. It hard-rejects ``N > 1`` (``v1 supports batch N==1 only``) because it folds the
spatial extent as ``physical_volume()/C``, which spans batches -- and H3's norm is
per-frame, so ``N = T`` (17/9/5). It also takes a single ``cluster_axis``, so it cannot
reduce over both mesh axes. It measured 1.6x slower regardless, so it was dropped from this
branch (STATE.md amendment 56).

The replacement computes the statistics itself: per ``(frame, group)`` local sums, an
all-reduce of **just those** (T x 32 scalars, against the fused op's full-activation
gather), then an elementwise normalise. Two passes -- mean, then centred variance -- to
avoid the ``E[x^2] - E[x]^2`` cancellation that is why ``GroupNorm3D`` uses Welford.

That needs four things to hold on device, and each is cheaper to probe here than to
debug inside a thirteen-norm encoder:

1. reduce over the flattened spatial axis of a ``(T, 1, H*W, C)`` tile tensor;
2. channel -> group contraction (and its transpose back) as a matmul with a 0/1 matrix;
3. broadcast of a ``(T, 1, 1, C)`` statistic against ``(T, 1, H*W, C)``;
4. all-reduce of a small statistic tensor along one mesh axis.

Together they are the whole norm, so this file also serves as its numerical gate: the
last test assembles them and compares against ``torch.nn.GroupNorm``.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....parallel.manager import CCLManager
from ....utils.check import assert_quality

NUM_GROUPS = 32
EPS = 1e-6

MESH_4X8_AXIS1 = [
    pytest.param(
        (4, 8),
        1,
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D, "require_exact_physical_num_devices": True},
        id="sp8_axis1",
    )
]

# (C, T, H, W) -- the shallowest and deepest encoder norm sites, i.e. the largest spatial
# extent and the one where H/8 collapses to 2 rows against C=1024.
PROBE_SITES = [(128, 17, 256, 256), (1024, 5, 16, 16)]


def _group_matrix(channels: int, groups: int) -> torch.Tensor:
    """``(C, G)`` 0/1 contraction: column g selects the channels belonging to group g."""
    per_group = channels // groups
    matrix = torch.zeros(channels, groups)
    for group in range(groups):
        matrix[group * per_group : (group + 1) * per_group, group] = 1.0
    return matrix


@pytest.mark.parametrize(
    ("mesh_device", "cluster_axis", "device_params"), MESH_4X8_AXIS1, indirect=["mesh_device", "device_params"]
)
@pytest.mark.parametrize("site", PROBE_SITES, ids=[f"c{c}_t{t}_{h}x{w}" for c, t, h, w in PROBE_SITES])
def test_distributed_frame_group_norm_from_primitives(mesh_device, cluster_axis, site):
    channels, frames, height, width = site
    cluster = tuple(mesh_device.shape)[cluster_axis]
    local_height = height // cluster
    assert height % cluster == 0

    torch.manual_seed(0)
    ref = torch.nn.GroupNorm(num_groups=NUM_GROUPS, num_channels=channels, eps=EPS)
    torch.nn.init.normal_(ref.weight)
    torch.nn.init.normal_(ref.bias)
    ref.eval()

    x_TCHW = torch.randn(frames, channels, height, width)
    with torch.no_grad():
        expected = ref(x_TCHW)

    ccl = CCLManager(mesh_device=mesh_device, topology=ttnn.Topology.Linear)

    # (T, H_local, W, C) sharded on H, flattened to (T, 1, H_local*W, C).
    x_THWC = x_TCHW.permute(0, 2, 3, 1).contiguous()
    x_flat = x_THWC.reshape(frames, 1, height * width, channels)
    x_device = ttnn.from_torch(
        x_flat,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device, dims=(None, 2) if cluster_axis == 1 else (2, None), mesh_shape=tuple(mesh_device.shape)
        ),
    )
    logger.info(f"local shard {tuple(x_device.shape)} (expect HW={local_height * width})")

    group_matrix = _group_matrix(channels, NUM_GROUPS)
    to_groups = ttnn.from_torch(
        group_matrix.reshape(1, 1, channels, NUM_GROUPS),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    from_groups = ttnn.from_torch(
        group_matrix.t().contiguous().reshape(1, 1, NUM_GROUPS, channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    elements_per_group = (channels // NUM_GROUPS) * height * width

    def group_reduce(tensor):
        """(T,1,HW_local,C) -> global per-(frame,group) sum, shaped (T,1,1,G).

        The all-reduce is an ``all_gather`` on the singleton dim 1 followed by a local
        sum: ``tt_dit`` exposes no all-reduce (there is not one call to it anywhere in
        the tree), and at ``T x 32`` scalars the gather is free either way.
        """
        channel_sum = ttnn.sum(tensor, dim=2, keepdim=True)  # (T,1,1,C)
        group_sum = ttnn.matmul(channel_sum, to_groups)  # (T,1,1,G)
        gathered = ccl.all_gather(group_sum, dim=1, mesh_axis=cluster_axis, use_hyperparams=False)
        return ttnn.sum(gathered, dim=1, keepdim=True)  # (T,1,1,G)

    def spread(group_sum):
        return ttnn.matmul(group_sum, from_groups)  # (T,1,1,C)

    mean = spread(ttnn.multiply(group_reduce(x_device), 1.0 / elements_per_group))
    centred = ttnn.subtract(x_device, mean)
    variance = spread(ttnn.multiply(group_reduce(ttnn.multiply(centred, centred)), 1.0 / elements_per_group))
    normed = ttnn.multiply(centred, ttnn.rsqrt(ttnn.add(variance, EPS)))

    weight = ttnn.from_torch(
        ref.weight.detach().reshape(1, 1, 1, channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    bias = ttnn.from_torch(
        ref.bias.detach().reshape(1, 1, 1, channels),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = ttnn.add(ttnn.multiply(normed, weight), bias)

    dims = [None, None]
    dims[cluster_axis] = 2
    dims[1 - cluster_axis] = 0
    actual = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=dims, mesh_shape=tuple(mesh_device.shape))
    ).float()
    if actual.ndim == 5:
        actual = actual[0]
    actual = actual[:frames]
    actual_TCHW = actual.reshape(frames, height, width, channels).permute(0, 3, 1, 2)

    assert actual_TCHW.shape == expected.shape, f"{actual_TCHW.shape} != {expected.shape}"
    assert_quality(expected, actual_TCHW, pcc=0.999)

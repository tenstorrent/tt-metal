# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Device parity for 3D neighborhood attention.

The host executor in ``layers.na3d`` is already checked against upstream's own natten-free
na3d, so this compares device against host: same plan, same masks, so any gap is ttnn
execution (gather ordering, layout round trips, bf16 rounding) rather than window arithmetic.

Shapes are the ones the DiffVAE decoder actually uses — kernels ``(3,7,7)``, ``(3,5,5)`` and
``(11,11,11)`` — plus the boundary cases that a window rule can get wrong: an axis shorter
than the kernel, and a grid large enough to force more than one tile per axis.
"""

import pytest
import torch

import ttnn

from ...layers.na3d import (
    DEFAULT_CHUNK_BUDGET,
    NA3DShard,
    build_device_plan,
    na3d_torch,
    neighborhood_attention_3d,
    plan_na3d,
)
from ...parallel.manager import CCLManager
from ...utils.check import assert_quality
from ...utils.tensor import to_torch as to_torch_replicated


@pytest.mark.parametrize(
    "dims, kernel",
    [
        ((3, 7, 7), (3, 7, 7)),  # stage-1/2 kernel, exact fit
        ((2, 5, 5), (3, 7, 7)),  # every axis shorter than the kernel
        ((6, 9, 9), (3, 7, 7)),  # interior plus both boundary regimes
        ((8, 12, 6), (3, 5, 5)),  # stage-3/4 kernel, non-cubic
        ((12, 16, 16), (11, 11, 11)),  # stage-5 kernel
        # Grids large enough that a group holds several tiles, which is what the chunked path
        # needs: on the shapes above the planner fits each regime in one tile, so chunking has
        # nothing to split. The first is the real stage-1 grid at 1920x1088.
        ((6, 34, 60), (3, 7, 7)),  # 9 groups, up to 4 tiles each
        ((25, 40, 40), (11, 11, 11)),  # 24 groups, up to 12 tiles each
    ],
)
@pytest.mark.parametrize("heads, head_dim", [(4, 64)])
@pytest.mark.parametrize(
    "chunk_budget, chunked",
    [
        (DEFAULT_CHUNK_BUDGET, False),
        # Small enough to force one tile per call on these grids. Chunking is what makes full
        # resolution fit, but it only runs there, so without this the shapes that have ground
        # truth would all take the single-chunk path and never exercise it.
        (1, True),
    ],
    ids=["one_chunk", "per_tile_chunks"],
)
def test_na3d_matches_host(*, device, dims, kernel, heads, head_dim, chunk_budget, chunked):
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, heads, head_dim, dtype=torch.float32) for _ in range(3))

    plan = plan_na3d(dims, kernel)
    expected = na3d_torch(q, k, v, kernel, scale=1.0, plan=plan)
    expected = expected.reshape(1, *dims, heads * head_dim)

    if chunked and max(len(group.query_slices) for group in plan.groups) == 1:
        pytest.skip(f"{dims}/{kernel} fits each regime in one tile; chunking has nothing to split")

    tt_q, tt_k, tt_v = (
        ttnn.from_torch(x, device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for x in (q, k, v)
    )
    device_plan = build_device_plan(plan, mesh_device=device, dtype=ttnn.bfloat16)
    actual = neighborhood_attention_3d(
        tt_q, tt_k, tt_v, kernel_size=kernel, scale=1.0, device_plan=device_plan, chunk_budget=chunk_budget
    )

    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, ttnn.to_torch(actual), pcc=0.999)


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D_RING}],
    indirect=True,
    ids=["ring"],
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True, ids=["4x8"])
@pytest.mark.parametrize(
    "dims, kernel",
    [
        # Every group here needs padding on both axes, which is the point: 9 groups of 1 to 4
        # tiles against a mesh axis of 8, and 810 query rows against a mesh axis of 4. This is
        # the real stage-1 grid at 1920x1088.
        ((6, 34, 60), (3, 7, 7)),
        # Enough tiles that the dominant groups split without padding, so the padded and unpadded
        # regimes are both covered.
        ((25, 40, 40), (11, 11, 11)),
    ],
)
@pytest.mark.parametrize(
    "chunk_budget",
    # Chunking and sharding interact: chunks are rejoined before the mesh gather so that the
    # reassembled order stays independent of how the local tiles were split. Forcing one tile
    # per chunk is what would expose that if it were wrong.
    [DEFAULT_CHUNK_BUDGET, 1],
    ids=["one_chunk", "per_tile_chunks"],
)
def test_na3d_sharded_matches_host(*, mesh_device, dims, kernel, chunk_budget):
    """A mesh-sharded plan against the host executor: same plan, same masks, split 32 ways.

    Sharding partitions the query tiles exactly, so this is not a tolerance question — the
    device is doing the same arithmetic on the same values in a different place, and the only
    gap should be the bf16 rounding the replicated path already has.
    """
    heads, head_dim = 4, 64
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, *dims, heads, head_dim, dtype=torch.float32) for _ in range(3))

    plan = plan_na3d(dims, kernel)
    expected = na3d_torch(q, k, v, kernel, scale=1.0, plan=plan).reshape(1, *dims, heads * head_dim)

    tt_q, tt_k, tt_v = (
        ttnn.from_torch(x, device=mesh_device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT) for x in (q, k, v)
    )
    ccl_manager = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    device_plan = build_device_plan(plan, mesh_device=mesh_device, dtype=ttnn.bfloat16, ccl_manager=ccl_manager)

    shard = device_plan.shard
    assert shard == NA3DShard(tile_axis=1, tile_factor=8, row_axis=0, row_factor=4), f"unexpected split {shard}"

    actual = neighborhood_attention_3d(
        tt_q, tt_k, tt_v, kernel_size=kernel, scale=1.0, device_plan=device_plan, chunk_budget=chunk_budget
    )

    assert tuple(actual.shape) == tuple(expected.shape), f"{tuple(actual.shape)} != {tuple(expected.shape)}"
    assert_quality(expected, to_torch_replicated(actual), pcc=0.999)

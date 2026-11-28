# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""A sub_core_grids that misses a shard core of an L1-sharded eltwise-binary operand must fail loud.

The native L1 path splits per-core work over the operands' shard grid while the kernels go on the worker
grid, so a worker grid that misses a shard core leaves that shard without a kernel: the values there are
never written (silent wrong values) and a grid smaller than the shard count deadlocks the program. A grid
that contains the shard grid, and every interleaved operand, are unaffected. Only grids that overlap or
are disjoint from the shard grid are rejected here; the smaller-than-shard-count hang case is not included
because it hangs on main too.
"""

import pytest
import ttnn
import torch

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

SHAPE = [1, 1, 32, 256]
SHARD_SHAPE = [32, 64]


def sharded_pair(device):
    """One sharded input and its torch twin, plus a matching second operand."""
    torch.manual_seed(0)
    torch_a = torch.rand(SHAPE, dtype=torch.bfloat16)
    torch.manual_seed(1)
    torch_b = torch.rand(SHAPE, dtype=torch.bfloat16)

    shard_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))])
    input_shard_spec = ttnn.ShardSpec(shard_grid, SHARD_SHAPE, ttnn.ShardOrientation.ROW_MAJOR)
    memory_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, input_shard_spec)

    def to_dev(t):
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=memory_config,
        )

    return torch_a, torch_b, to_dev(torch_a), to_dev(torch_b)


def grid(x0, y0, x1, y1):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])


def torch_ref(ttnn_op):
    torch.manual_seed(0)
    torch_a = torch.rand(SHAPE, dtype=torch.bfloat16)
    torch.manual_seed(1)
    torch_b = torch.rand(SHAPE, dtype=torch.bfloat16)
    return torch.add(torch_a, torch_b) if ttnn_op is ttnn.add else torch.mul(torch_a, torch_b)


@pytest.mark.parametrize(
    "sub_core_grids",
    [
        # Overlaps the shard grid (0,0)-(3,0) but drops core (0,0): silent wrong values.
        grid(1, 0, 4, 0),
        # Disjoint from the shard grid: the whole output is wrong.
        grid(1, 1, 4, 4),
    ],
    ids=["overlaps_missing_a_shard_core", "disjoint"],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.multiply], ids=["add", "multiply"])
def test_binary_ng_sub_core_grids_must_contain_shard_cores(ttnn_op, sub_core_grids, device, expect_error):
    torch_a, _, a, b = sharded_pair(device)
    assert a.memory_config().shard_spec is not None, "test setup must produce a legacy 2-D shard spec"

    with expect_error(RuntimeError, "must contain all shard cores"):
        ttnn_op(a, b, sub_core_grids=sub_core_grids)


@pytest.mark.parametrize(
    "sub_core_grids",
    [
        # A superset of the shard grid, the shard grid itself, and no grid at all all keep the result.
        None,
        grid(0, 0, 7, 0),
        grid(0, 0, 3, 0),
        grid(0, 0, 7, 6),
    ],
    ids=["no_grid", "superset", "equal_to_shard_grid", "whole_grid"],
)
@pytest.mark.parametrize("ttnn_op", [ttnn.add, ttnn.multiply], ids=["add", "multiply"])
def test_binary_ng_sub_core_grids_containing_shard_cores_stay_exact(ttnn_op, sub_core_grids, device):
    torch_a, _, a, b = sharded_pair(device)

    # Sibling with no grid: the grid must not move a single value.
    ref = ttnn_op(a, b)

    kwargs = {} if sub_core_grids is None else {"sub_core_grids": sub_core_grids}
    out = ttnn_op(a, b, **kwargs)

    # A binary op defaults its output memory config to the input's sharded one, so only compare it
    # against the sibling rather than asserting a layout the caller never asked for.
    assert out.memory_config() == ref.memory_config()
    assert_equal(ttnn.to_torch(ref), ttnn.to_torch(out))
    assert_with_pcc(torch_ref(ttnn_op), ttnn.to_torch(out))


@pytest.mark.parametrize(
    "sub_core_grids",
    [
        None,
        grid(0, 0, 7, 0),
        grid(0, 0, 3, 0),
    ],
    ids=["no_grid", "superset", "equal_to_shard_grid"],
)
def test_binary_ng_interleaved_output_stays_interleaved(sub_core_grids, device):
    # With an explicit interleaved output the layout really must come back interleaved, and the grid
    # must not move a single value.
    torch.manual_seed(0)
    torch_a = torch.rand(SHAPE, dtype=torch.bfloat16)
    torch.manual_seed(1)
    torch_b = torch.rand(SHAPE, dtype=torch.bfloat16)
    a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    assert not a.is_sharded()

    ref = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    kwargs = {} if sub_core_grids is None else {"sub_core_grids": sub_core_grids}
    out = ttnn.add(a, b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs)

    assert out.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert_equal(ttnn.to_torch(ref), ttnn.to_torch(out))
    assert_with_pcc(torch_a + torch_b, ttnn.to_torch(out))

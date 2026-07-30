# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Coverage for https://github.com/tenstorrent/tt-metal/issues/43050:
# generic reduce (sum/mean/max/min/std/var) already supports L1-sharded input/output, but
# combining sharding with a DRAM buffer is explicitly rejected by
# ttnn/cpp/ttnn/operations/reduction/generic/device/common.cpp::validate_reduce_sharded_buffer_types.
# std/var dispatch through a separate WelfordReduceDeviceOperation (not the ReduceDeviceOperation
# that sum/mean/max/min use), but it calls the exact same validate_reduce_sharded_buffer_types, just
# with op_name="Std/Var reduction" (welford_reduce_device_operation.cpp:36) - same restriction, same
# fix surface.
# These tests pin down both sides of that boundary so a future fix (comprehensive DRAM-sharded
# support) has a failing test to flip green.

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

TEST_PADDING_VALUE = -42

REDUCE_OPS = {
    "sum": (ttnn.sum, lambda t, dim, keepdim: torch.sum(t, dim=dim, keepdim=keepdim)),
    "mean": (ttnn.mean, lambda t, dim, keepdim: torch.mean(t, dim=dim, keepdim=keepdim)),
    "max": (ttnn.max, lambda t, dim, keepdim: torch.max(t, dim=dim, keepdim=keepdim).values),
    "min": (ttnn.min, lambda t, dim, keepdim: torch.min(t, dim=dim, keepdim=keepdim).values),
    # correction defaults to True (Bessel's correction) on both ttnn.std/var and torch.std/var.
    "std": (ttnn.std, lambda t, dim, keepdim: torch.std(t, dim=dim, keepdim=keepdim)),
    "var": (ttnn.var, lambda t, dim, keepdim: torch.var(t, dim=dim, keepdim=keepdim)),
}


_L1_SHARD_CORE_GRIDS = {
    ttnn.ShardStrategy.HEIGHT: ttnn.CoreGrid(x=1, y=4),
    ttnn.ShardStrategy.WIDTH: ttnn.CoreGrid(x=5, y=1),  # 160 / 5 = 32, tile-aligned
    ttnn.ShardStrategy.BLOCK: ttnn.CoreGrid(x=5, y=8),
}


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize(
    "shard_strategy", [ttnn.ShardStrategy.HEIGHT, ttnn.ShardStrategy.WIDTH, ttnn.ShardStrategy.BLOCK]
)
def test_reduce_l1_sharded(device, op_name, shard_strategy):
    """L1-sharded input/output already works for generic reduce; guard against regressions."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    shape = (1, 1024, 160)
    core_grid = _L1_SHARD_CORE_GRIDS[shard_strategy]

    torch_input_tensor = torch.randn(shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor, dim=-1, keepdim=True)

    sharded_config = ttnn.create_sharded_memory_config(
        shape=shape,
        core_grid=core_grid,
        strategy=shard_strategy,
        use_height_and_width_as_shard_shape=False,
    )

    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=sharded_config,
    )

    output_tensor = ttnn_op(input_tensor, dim=-1, keepdim=True, memory_config=sharded_config)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == ttnn.BufferType.L1
    assert output_mem_config.is_sharded()

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


# Per-layout (tensor_shape, shard_shape, shard_grid) that satisfies the *physical* shard-geometry
# constraints each layout imposes on tensor construction (e.g. WIDTH_SHARDED requires shard height ==
# full physical height). Only used to legally materialize a DRAM-sharded tensor for the input-side
# test; the output-side test never materializes an output tensor with this shape, since
# validate_reduce_sharded_buffer_types fires before any shape-specific validation runs.
_DRAM_SHARD_GEOMETRY = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: {
        "tensor_shape": (1, 1, 416, 32),
        "shard_shape": (128, 32),
        "shard_grid": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
    },
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: {
        "tensor_shape": (1, 1, 32, 128),
        "shard_shape": (32, 32),
        "shard_grid": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))}),
    },
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: {
        "tensor_shape": (1, 1, 128, 128),
        "shard_shape": (32, 32),
        "shard_grid": ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
    },
}


def _dram_sharded_memory_config(shard_layout):
    geometry = _DRAM_SHARD_GEOMETRY[shard_layout]
    shard_spec = ttnn.ShardSpec(geometry["shard_grid"], geometry["shard_shape"], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(shard_layout, ttnn.BufferType.DRAM, shard_spec)


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize(
    "shard_layout",
    # BLOCK_SHARDED is excluded here: ttnn.interleaved_to_sharded itself rejects DRAM block sharding
    # ("We don't support DRAM block sharding", interleaved_to_sharded_op.cpp:112) before a tensor can
    # even be materialized, so the reduce op's own input-side check is unreachable via this path.
    [ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.TensorMemoryLayout.WIDTH_SHARDED],
)
def test_reduce_dram_sharded_input_not_supported(device, op_name, shard_layout, expect_error):
    """DRAM-sharded *input* is rejected today, for height/width shard layouts; see issue #43050."""
    ttnn_op, _ = REDUCE_OPS[op_name]
    dram_sharded_config = _dram_sharded_memory_config(shard_layout)

    tensor_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["tensor_shape"]
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dram_sharded_input = ttnn.interleaved_to_sharded(interleaved_input, dram_sharded_config)

    # Output memory config defaults to mirroring the (sharded) input, which would trip the
    # *output*-side check first. Force an interleaved output so this isolates the input-side check.
    with expect_error(RuntimeError, "sharded input memory layout"):
        ttnn_op(dram_sharded_input, dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize("shard_layout", list(_DRAM_SHARD_GEOMETRY.keys()))
def test_reduce_dram_sharded_output_not_supported(device, op_name, shard_layout, expect_error):
    """DRAM-sharded *output* is rejected today, for any shard layout; see issue #43050."""
    ttnn_op, _ = REDUCE_OPS[op_name]
    dram_sharded_config = _dram_sharded_memory_config(shard_layout)

    tensor_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["tensor_shape"]
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    with expect_error(RuntimeError, "sharded output memory layout"):
        ttnn_op(interleaved_input, dim=-1, keepdim=True, memory_config=dram_sharded_config)

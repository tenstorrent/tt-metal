# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Sharded-topology coverage for generic reduce (sum/mean/max/min/std/var) on natural-axis reduces.
# L1 and DRAM shard grids are disjoint coordinate spaces - worker-core (x,y) vs bank ids on row
# y=0 - so buffer type is varied independently on input and output.

import pytest

pytestmark = pytest.mark.use_module_device

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_numeric_metrics

REDUCE_OPS = {
    "sum": (ttnn.sum, lambda t, dim, keepdim: torch.sum(t, dim=dim, keepdim=keepdim)),
    "mean": (ttnn.mean, lambda t, dim, keepdim: torch.mean(t, dim=dim, keepdim=keepdim)),
    # amax/amin accept a tuple dim, needed by the full-HW tests.
    "max": (ttnn.max, lambda t, dim, keepdim: torch.amax(t, dim=dim, keepdim=keepdim)),
    "min": (ttnn.min, lambda t, dim, keepdim: torch.amin(t, dim=dim, keepdim=keepdim)),
    # Bessel's correction defaults to True on both ttnn and torch.
    "std": (ttnn.std, lambda t, dim, keepdim: torch.std(t, dim=dim, keepdim=keepdim)),
    "var": (ttnn.var, lambda t, dim, keepdim: torch.var(t, dim=dim, keepdim=keepdim)),
}


def test_reduce_dram_block_sharded_construction_is_impossible(device, expect_error):
    """DRAM's 1D, row-y=0 bank address space cannot hold BLOCK_SHARDED's 2D shard grid, so no
    positive DRAM BLOCK_SHARDED case exists to test - for reduce or any other op."""
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))})
    shard_spec = ttnn.ShardSpec(shard_grid, (32, 32), ttnn.ShardOrientation.ROW_MAJOR)
    mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch_input_tensor = torch.randn((1, 1, 128, 128), dtype=torch.bfloat16)
    with expect_error(RuntimeError, "DRAM banks are 1D"):
        ttnn.from_torch(
            torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem_config
        )


# Geometries satisfying each layout's shard constraints (e.g. WIDTH_SHARDED needs shard height ==
# full physical height) on a single-row DRAM bank grid. BLOCK_SHARDED is unconstructible on DRAM.
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
}


def _dram_sharded_memory_config(shard_layout, shard_grid=None, shard_shape=None):
    geometry = _DRAM_SHARD_GEOMETRY[shard_layout]
    shard_spec = ttnn.ShardSpec(
        shard_grid or geometry["shard_grid"],
        shard_shape or geometry["shard_shape"],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(shard_layout, ttnn.BufferType.DRAM, shard_spec)


def _dram_sharded_input(device, shard_layout, dtype=ttnn.bfloat16, torch_input_tensor=None):
    tensor_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["tensor_shape"]
    if torch_input_tensor is None:
        torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
    dram_sharded_input = ttnn.interleaved_to_sharded(interleaved_input, _dram_sharded_memory_config(shard_layout))
    return torch_input_tensor, dram_sharded_input


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize("shard_layout", list(_DRAM_SHARD_GEOMETRY.keys()))
@pytest.mark.parametrize("dim", [-1, -2])
def test_reduce_dram_sharded(device, op_name, shard_layout, dim):
    """DRAM-sharded input+output across both reduce-dim program factories: dim=-1 uses the W
    factory, dim=-2 the H factory, whose width-sharded fast path is L1-only. WIDTH_SHARDED + dim=-1
    reduces along the tensor's own sharded dimension, collapsing it to a single shard."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    torch_input_tensor, dram_sharded_input = _dram_sharded_input(device, shard_layout)
    torch_output_tensor = torch_op(torch_input_tensor, dim, True)

    dram_sharded_config = _dram_sharded_memory_config(shard_layout)
    output_tensor = ttnn_op(dram_sharded_input, dim=dim, keepdim=True, memory_config=dram_sharded_config)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == ttnn.BufferType.DRAM
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


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize("shard_layout", list(_DRAM_SHARD_GEOMETRY.keys()))
def test_reduce_dram_sharded_full_hw_reduce(device, op_name, shard_layout):
    """Full-HW reduce from a DRAM-sharded input; the output is interleaved since a collapsed 1x1
    result cannot stay sharded across cores. sum/mean/max/min decompose host-side into a W-reduce
    then an H-reduce, each reading the sharded or intermediate tensor; std/var use one Welford
    call."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    torch_input_tensor, dram_sharded_input = _dram_sharded_input(device, shard_layout)
    torch_output_tensor = torch_op(torch_input_tensor, (-2, -1), True)

    output_tensor = ttnn_op(dram_sharded_input, dim=(-2, -1), keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == ttnn.BufferType.DRAM
    assert not output_mem_config.is_sharded()

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


@pytest.mark.parametrize("op_name", ["sum", "max"])
def test_reduce_dram_sharded_full_bank_width_h_reduce(device, op_name):
    """A WIDTH_SHARDED grid spanning every DRAM bank."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    num_banks = device.dram_grid_size().x
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(num_banks - 1, 0))})
    tensor_shape = (1, 1, 32, 32 * num_banks)
    shard_shape = (32, 32)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor, -2, True)

    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dram_sharded_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
        ttnn.BufferType.DRAM,
        ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR),
    )
    dram_sharded_input = ttnn.interleaved_to_sharded(interleaved_input, dram_sharded_config)

    output_tensor = ttnn_op(dram_sharded_input, dim=-2, keepdim=True, memory_config=dram_sharded_config)
    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


@pytest.mark.parametrize("op_name", ["sum", "max"])
@pytest.mark.parametrize("dram_side", ["input", "output"])
def test_reduce_h_width_sharded_mixed_l1_dram(device, op_name, dram_side):
    """One side DRAM-WIDTH_SHARDED, the other L1-WIDTH_SHARDED, on H-reduce: the use_width_sharding
    gate must key off buffer type per side, not just layout."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    tensor_shape = _DRAM_SHARD_GEOMETRY[ttnn.TensorMemoryLayout.WIDTH_SHARDED]["tensor_shape"]
    shard_shape = _DRAM_SHARD_GEOMETRY[ttnn.TensorMemoryLayout.WIDTH_SHARDED]["shard_shape"]
    shard_grid = _DRAM_SHARD_GEOMETRY[ttnn.TensorMemoryLayout.WIDTH_SHARDED]["shard_grid"]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    dram_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)
    l1_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor, -2, True)

    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_config = dram_config if dram_side == "input" else l1_config
    output_config = l1_config if dram_side == "input" else dram_config
    input_tensor = ttnn.interleaved_to_sharded(interleaved_input, input_config)

    output_tensor = ttnn_op(input_tensor, dim=-2, keepdim=True, memory_config=output_config)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == (ttnn.BufferType.L1 if dram_side == "input" else ttnn.BufferType.DRAM)

    output_tensor = ttnn.to_torch(output_tensor)
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


@pytest.mark.parametrize("op_name", list(REDUCE_OPS.keys()))
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.bfloat8_b])
def test_reduce_dram_sharded_dtypes(device, op_name, dtype):
    """Every supported dtype through the H-reduce generic branch, at one fixed geometry."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]
    shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED

    torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
    tensor_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["tensor_shape"]
    torch_input_tensor = torch.randn(tensor_shape, dtype=torch_dtype)
    torch_output_tensor = torch_op(torch_input_tensor, -2, True)

    _, dram_sharded_input = _dram_sharded_input(
        device, shard_layout, dtype=dtype, torch_input_tensor=torch_input_tensor
    )
    dram_sharded_config = _dram_sharded_memory_config(shard_layout)
    output_tensor = ttnn_op(dram_sharded_input, dim=-2, keepdim=True, memory_config=dram_sharded_config)
    output_tensor = ttnn.to_torch(output_tensor)

    # bfloat8_b's block-float quantization inflates relative error on near-zero sums; absolute
    # error and PCC stay well within tolerance.
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.999,
        rtol=0.1,
        atol=0.2,
        frobenius_threshold=0.02,
    )


def test_reduce_dram_sharded_non_natural_dim(device, expect_error):
    """Non-natural-dim reduce moves the axis into H/W position via transpose or permute, whose
    DRAM-sharded handling is out of scope here. Behavior differs across ops - ttnn.sum succeeds on
    this input while mean/max/min/std/var fail building the intermediate's TensorSpec - so only
    ttnn.mean is pinned."""
    torch.manual_seed(0)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (416, 32), ttnn.ShardOrientation.ROW_MAJOR)  # one shard per (n, c) slice
    dram_sharded_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    torch_input_tensor = torch.randn(1, 4, 416, 32, dtype=torch.bfloat16)
    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    dram_sharded_input = ttnn.interleaved_to_sharded(interleaved_input, dram_sharded_config)

    with expect_error(RuntimeError, "Number of shards along height"):
        ttnn.mean(dram_sharded_input, dim=1, keepdim=True)


def test_reduce_h_width_sharded_l1_and_dram_use_distinct_programs(device):
    """L1- and DRAM-WIDTH_SHARDED H-reduce take different branches, so the buffer type must reach
    the program-cache key as distinct entries."""
    shard_layout = ttnn.TensorMemoryLayout.WIDTH_SHARDED
    tensor_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["tensor_shape"]
    shard_shape = _DRAM_SHARD_GEOMETRY[shard_layout]["shard_shape"]
    shard_grid = _DRAM_SHARD_GEOMETRY[shard_layout]["shard_grid"]
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    l1_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1, shard_spec)
    dram_config = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.DRAM, shard_spec)

    torch_input_tensor = torch.randn(tensor_shape, dtype=torch.bfloat16)
    interleaved_input = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # Ignore whatever earlier tests in this session compiled.
    device.enable_program_cache()
    device.clear_program_cache()
    l1_input = ttnn.interleaved_to_sharded(interleaved_input, l1_config)
    ttnn.sum(l1_input, dim=-2, keepdim=True, memory_config=l1_config)
    entries_after_l1 = device.num_program_cache_entries()

    dram_input = ttnn.interleaved_to_sharded(interleaved_input, dram_config)
    ttnn.sum(dram_input, dim=-2, keepdim=True, memory_config=dram_config)
    entries_after_dram = device.num_program_cache_entries()

    assert entries_after_dram > entries_after_l1, "expected a new program cache entry for the DRAM-sharded case"


def test_reduce_dram_sharded_requires_explicit_output_shard_spec_across_buffer_types(device, expect_error):
    """A spec-less sharded output may only borrow the input's shard grid when both share a buffer
    type; a DRAM input with an L1 spec-less output must be rejected."""
    shard_layout = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
    torch_input_tensor, dram_sharded_input = _dram_sharded_input(device, shard_layout)
    output_config_no_spec = ttnn.MemoryConfig(shard_layout, ttnn.BufferType.L1)

    with expect_error(RuntimeError, "requires an explicit shard_spec"):
        ttnn.sum(dram_sharded_input, dim=-1, keepdim=True, memory_config=output_config_no_spec)


# Sharding C as well as H and W keeps the layout ND_SHARDED: no legacy layout can express a
# split along C, so it cannot normalize to WIDTH/HEIGHT/BLOCK.
_DRAM_ND_TENSOR_SHAPE = (1, 4, 128, 128)
_DRAM_ND_SHARD_SHAPE = [1, 2, 64, 64]


def _dram_nd_sharded_config():
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    return ttnn.MemoryConfig(ttnn.BufferType.DRAM, ttnn.NdShardSpec(ttnn.Shape(_DRAM_ND_SHARD_SHAPE), grid))


@pytest.mark.parametrize("op_name", ["sum", "std"])
@pytest.mark.parametrize("dim", [-1, -2])
def test_reduce_dram_nd_sharded(device, op_name, dim):
    """DRAM ND_SHARDED input and output, covering build_reduce_output_tensor_spec's ND branch."""
    torch.manual_seed(0)
    ttnn_op, torch_op = REDUCE_OPS[op_name]

    nd_config = _dram_nd_sharded_config()
    torch_input_tensor = torch.randn(_DRAM_ND_TENSOR_SHAPE, dtype=torch.bfloat16)
    torch_output_tensor = torch_op(torch_input_tensor, dim, True)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=nd_config
    )
    assert input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.ND_SHARDED

    output_tensor = ttnn_op(input_tensor, dim=dim, keepdim=True, memory_config=nd_config)

    output_mem_config = output_tensor.memory_config()
    assert output_mem_config.buffer_type == ttnn.BufferType.DRAM
    assert output_mem_config.is_sharded()

    assert_numeric_metrics(
        torch_output_tensor,
        ttnn.to_torch(output_tensor),
        pcc_threshold=0.999,
        rtol=0.05,
        atol=0.05,
        frobenius_threshold=0.01,
    )


def test_reduce_dram_nd_sharded_requires_explicit_output_nd_shard_spec(device, expect_error):
    """ND counterpart of the legacy-fallback guard: an L1 ND output with no nd_shard_spec of its
    own cannot borrow a DRAM input's bank grid."""
    torch_input_tensor = torch.randn(_DRAM_ND_TENSOR_SHAPE, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_dram_nd_sharded_config(),
    )
    output_config_no_spec = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.ND_SHARDED, ttnn.BufferType.L1)

    with expect_error(RuntimeError, "requires an explicit nd_shard_spec"):
        ttnn.sum(input_tensor, dim=-2, keepdim=True, memory_config=output_config_no_spec)

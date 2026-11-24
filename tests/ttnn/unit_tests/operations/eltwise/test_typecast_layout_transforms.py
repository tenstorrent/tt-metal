# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.typecast changing dtype, layout and memory config in one call."""

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc

TILE = ttnn.TILE_LAYOUT
RM = ttnn.ROW_MAJOR_LAYOUT

DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8]
BFP_DTYPES = [ttnn.bfloat8_b, ttnn.bfloat4_b]

_TORCH_SOURCE_DTYPE = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.bfloat8_b: torch.float32,
    ttnn.bfloat4_b: torch.float32,
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.uint8: torch.int32,
}


def _values(shape, seed=0):
    # Small non-negative integers are exact in every supported dtype, so all casts must be bit-exact.
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 200, shape, generator=generator, dtype=torch.int32)


def _to_device(values, dtype, layout, device, memory_config):
    return ttnn.from_torch(
        values.to(_TORCH_SOURCE_DTYPE[dtype]),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config,
    )


def _check(output, values, dtype, layout, memory_config=None):
    assert output.dtype == dtype
    assert output.layout == layout
    assert list(output.shape) == list(values.shape)
    if memory_config is not None:
        assert output.memory_config().memory_layout == memory_config.memory_layout
        assert output.memory_config().buffer_type == memory_config.buffer_type
        if memory_config.shard_spec is not None:
            assert output.memory_config().shard_spec == memory_config.shard_spec
    result = ttnn.to_torch(output)
    if dtype in BFP_DTYPES:
        assert_with_pcc(values.to(torch.float32), result.to(torch.float32), 0.999 if dtype == ttnn.bfloat8_b else 0.98)
    else:
        assert torch.equal(result.to(torch.int64), values.to(torch.int64))


def _sharded(shape, strategy, core_grid, buffer_type=ttnn.BufferType.L1):
    memory_config = ttnn.create_sharded_memory_config(
        shape, core_grid=core_grid, strategy=strategy, orientation=ttnn.ShardOrientation.ROW_MAJOR
    )
    if buffer_type == ttnn.BufferType.L1:
        return memory_config
    return ttnn.MemoryConfig(memory_config.memory_layout, buffer_type, memory_config.shard_spec)


SHARDED_SHAPE = (1, 1, 64, 128)
SHARDED_CONFIGS = {
    "height": lambda: _sharded(SHARDED_SHAPE, ttnn.ShardStrategy.HEIGHT, ttnn.CoreGrid(y=1, x=2)),
    "width": lambda: _sharded(SHARDED_SHAPE, ttnn.ShardStrategy.WIDTH, ttnn.CoreGrid(y=1, x=4)),
    "block": lambda: _sharded(SHARDED_SHAPE, ttnn.ShardStrategy.BLOCK, ttnn.CoreGrid(y=2, x=2)),
}
INTERLEAVED_CONFIGS = {"dram": ttnn.DRAM_MEMORY_CONFIG, "l1": ttnn.L1_MEMORY_CONFIG}


def _memory_config(name):
    if name in INTERLEAVED_CONFIGS:
        return INTERLEAVED_CONFIGS[name]
    return SHARDED_CONFIGS[name]()


@pytest.mark.parametrize("shape", [(1, 1, 32, 64), (2, 3, 45, 70)], ids=["aligned", "unaligned"])
@pytest.mark.parametrize("input_layout, output_layout", [(TILE, RM), (RM, TILE)], ids=["tile_to_rm", "rm_to_tile"])
@pytest.mark.parametrize("output_dtype", DTYPES, ids=str)
@pytest.mark.parametrize("input_dtype", DTYPES, ids=str)
def test_typecast_layout_change(device, input_dtype, output_dtype, input_layout, output_layout, shape):
    values = _values(shape)
    input_tensor = _to_device(values, input_dtype, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, output_dtype, output_layout=output_layout)
    _check(output, values, output_dtype, output_layout, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize("shape", [(1, 1, 32, 64), (1, 2, 45, 70)], ids=["aligned", "unaligned"])
@pytest.mark.parametrize("output_dtype", [ttnn.bfloat16, ttnn.float32], ids=str)
@pytest.mark.parametrize("input_dtype", BFP_DTYPES, ids=str)
def test_typecast_bfp_tile_to_rm(device, input_dtype, output_dtype, shape):
    values = _values(shape)
    input_tensor = _to_device(values, input_dtype, TILE, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, output_dtype, output_layout=RM)
    expected = ttnn.to_torch(input_tensor).to(torch.float32)
    assert output.layout == RM and output.dtype == output_dtype
    assert torch.equal(ttnn.to_torch(output).to(torch.float32), expected)


@pytest.mark.parametrize("shape", [(1, 1, 32, 64), (1, 2, 45, 70)], ids=["aligned", "unaligned"])
@pytest.mark.parametrize("output_dtype", BFP_DTYPES, ids=str)
@pytest.mark.parametrize("input_dtype", [ttnn.bfloat16, ttnn.float32], ids=str)
def test_typecast_rm_to_bfp_tile(device, input_dtype, output_dtype, shape):
    values = _values(shape)
    input_tensor = _to_device(values, input_dtype, RM, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, output_dtype, output_layout=TILE)
    _check(output, values, output_dtype, TILE)


@pytest.mark.parametrize("output_memory", ["dram", "l1", "height", "width", "block"])
@pytest.mark.parametrize("input_memory", ["dram", "l1", "height", "width", "block"])
@pytest.mark.parametrize("layout", [TILE, RM], ids=["tile", "rm"])
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.int32), (ttnn.uint16, ttnn.uint16)],
    ids=str,
)
def test_typecast_memory_config_change(device, input_dtype, output_dtype, layout, input_memory, output_memory):
    values = _values(SHARDED_SHAPE)
    input_tensor = _to_device(values, input_dtype, layout, device, _memory_config(input_memory))
    output_memory_config = _memory_config(output_memory)
    output = ttnn.typecast(input_tensor, output_dtype, memory_config=output_memory_config)
    _check(output, values, output_dtype, layout, output_memory_config)


@pytest.mark.parametrize("output_memory", ["dram", "l1", "height", "width", "block"])
@pytest.mark.parametrize("input_memory", ["dram", "height", "block"])
@pytest.mark.parametrize("input_layout, output_layout", [(TILE, RM), (RM, TILE)], ids=["tile_to_rm", "rm_to_tile"])
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.bfloat16), (ttnn.int32, ttnn.uint16), (ttnn.uint8, ttnn.int32)],
    ids=str,
)
def test_typecast_combined_change(
    device, input_dtype, output_dtype, input_layout, output_layout, input_memory, output_memory
):
    values = _values(SHARDED_SHAPE)
    input_tensor = _to_device(values, input_dtype, input_layout, device, _memory_config(input_memory))
    output_memory_config = _memory_config(output_memory)
    output = ttnn.typecast(input_tensor, output_dtype, memory_config=output_memory_config, output_layout=output_layout)
    _check(output, values, output_dtype, output_layout, output_memory_config)


@pytest.mark.parametrize("input_layout, output_layout", [(TILE, RM), (RM, TILE), (TILE, TILE)], ids=str)
def test_typecast_input_dtype_overload_with_layout(device, input_layout, output_layout):
    values = _values((1, 1, 64, 64))
    input_tensor = _to_device(values, ttnn.int32, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, ttnn.int32, ttnn.float32, output_layout=output_layout)
    _check(output, values, ttnn.float32, output_layout)


@pytest.mark.parametrize("input_layout, output_layout", [(TILE, RM), (RM, TILE)], ids=["tile_to_rm", "rm_to_tile"])
@pytest.mark.parametrize("input_dtype, output_dtype", [(ttnn.bfloat16, ttnn.float32), (ttnn.int32, ttnn.bfloat16)])
def test_typecast_layout_change_program_cache(device, input_dtype, output_dtype, input_layout, output_layout):
    for seed, shape in enumerate([(1, 1, 64, 96), (1, 1, 64, 96), (1, 2, 33, 40), (1, 2, 33, 40)]):
        values = _values(shape, seed)
        input_tensor = _to_device(values, input_dtype, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.typecast(input_tensor, output_dtype, output_layout=output_layout)
        _check(output, values, output_dtype, output_layout)


@pytest.mark.parametrize("input_layout, output_layout", [(TILE, RM), (RM, TILE)], ids=["tile_to_rm", "rm_to_tile"])
@pytest.mark.parametrize("input_dtype, output_dtype", [(ttnn.bfloat16, ttnn.float32), (ttnn.float32, ttnn.int32)])
def test_typecast_host_layout_change(input_dtype, output_dtype, input_layout, output_layout):
    values = _values((1, 1, 32, 64))
    input_tensor = ttnn.from_torch(values.to(_TORCH_SOURCE_DTYPE[input_dtype]), dtype=input_dtype, layout=input_layout)
    output = ttnn.typecast(input_tensor, output_dtype, output_layout=output_layout)
    _check(output, values, output_dtype, output_layout)


@pytest.mark.parametrize("output_dtype", BFP_DTYPES, ids=str)
@pytest.mark.parametrize("input_layout", [TILE, RM], ids=["tile", "rm"])
def test_typecast_bfp_row_major_output_rejected(device, expect_error, input_layout, output_dtype):
    input_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.float32, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, "requires TILE output layout"):
        ttnn.typecast(input_tensor, output_dtype, output_layout=RM)


def test_typecast_layout_change_rejects_preallocated_output(device, expect_error):
    input_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.float32, TILE, device, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.int32, RM, device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, "preallocated output"):
        ttnn.typecast(input_tensor, ttnn.int32, output_tensor=output_tensor)


def test_typecast_layout_change_rejects_sub_core_grids(device, expect_error):
    input_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.float32, TILE, device, ttnn.DRAM_MEMORY_CONFIG)
    sub_core_grids = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    with expect_error(RuntimeError, "sub_core_grids"):
        ttnn.typecast(input_tensor, ttnn.int32, sub_core_grids=sub_core_grids, output_layout=RM)

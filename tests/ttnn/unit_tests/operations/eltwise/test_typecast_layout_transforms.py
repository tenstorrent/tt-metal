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
    # Small non-negative integers are exact in every non-block-float dtype, so those casts must be bit-exact.
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
        # Block-float shares one exponent per 16 values; bfloat4_b keeps 3 mantissa bits, hence the lower bar.
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
@pytest.mark.parametrize("input_memory", ["dram", "l1", "height", "width", "block"])
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


# The layout changes only through output_layout; a preallocated output in another layout is rejected.
def test_typecast_preallocated_output_layout_without_output_layout_rejected(device, expect_error):
    input_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.float32, TILE, device, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.int32, RM, device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, "must match the output layout"):
        ttnn.typecast(input_tensor, ttnn.int32, output_tensor=output_tensor)


# "dram" runs in one pass; a width-sharded row-major side goes through separate ops and a copy.
@pytest.mark.parametrize("row_major_memory", ["dram", "width"])
@pytest.mark.parametrize("input_layout, output_layout", [(TILE, RM), (RM, TILE)], ids=["tile_to_rm", "rm_to_tile"])
def test_typecast_layout_change_preallocated_output(device, input_layout, output_layout, row_major_memory):
    values = _values(SHARDED_SHAPE)
    input_memory = _memory_config(row_major_memory) if input_layout == RM else ttnn.DRAM_MEMORY_CONFIG
    output_memory = _memory_config(row_major_memory) if output_layout == RM else ttnn.DRAM_MEMORY_CONFIG
    input_tensor = _to_device(values, ttnn.bfloat16, input_layout, device, input_memory)
    output_tensor = _to_device(
        torch.zeros(SHARDED_SHAPE, dtype=torch.int32), ttnn.float32, output_layout, device, output_memory
    )
    output = ttnn.typecast(input_tensor, ttnn.float32, output_tensor=output_tensor, output_layout=output_layout)
    assert output.buffer_address() == output_tensor.buffer_address()
    _check(output_tensor, values, ttnn.float32, output_layout, output_memory)


@pytest.mark.parametrize("input_layout, output_layout", [(TILE, RM), (RM, TILE)], ids=["tile_to_rm", "rm_to_tile"])
@pytest.mark.parametrize(
    "input_dtype, output_dtype", [(ttnn.bfloat16, ttnn.float32), (ttnn.int32, ttnn.uint16)], ids=str
)
def test_typecast_layout_change_sub_core_grids(device, input_dtype, output_dtype, input_layout, output_layout):
    # Two disjoint ranges, 7 cores.
    sub_core_grids = ttnn.CoreRangeSet(
        {
            ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 2)),
            ttnn.CoreRange(ttnn.CoreCoord(4, 0), ttnn.CoreCoord(4, 2)),
        }
    )
    values = _values((1, 2, 32 * 10 + 5, 64))
    input_tensor = _to_device(values, input_dtype, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, output_dtype, sub_core_grids=sub_core_grids, output_layout=output_layout)
    _check(output, values, output_dtype, output_layout, ttnn.DRAM_MEMORY_CONFIG)


# Width-sharded row-major pages hold part of a row, so this input cannot run in one pass.
def test_typecast_layout_change_sub_core_grids_without_one_pass_rejected(device, expect_error):
    input_tensor = _to_device(_values(SHARDED_SHAPE), ttnn.bfloat16, RM, device, _memory_config("width"))
    sub_core_grids = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 1))})
    with expect_error(RuntimeError, "sub_core_grids only when it runs in one pass"):
        ttnn.typecast(input_tensor, ttnn.float32, sub_core_grids=sub_core_grids, output_layout=TILE)


LAYOUT_DIRECTIONS = [(TILE, RM), (RM, TILE)]
LAYOUT_DIRECTION_IDS = ["tile_to_rm", "rm_to_tile"]


# More blocks than cores (uneven split, several blocks per core) and several blocks per tile row,
# including a last block that ends inside the last tile.
@pytest.mark.parametrize(
    "shape",
    [(1, 1, 32 * 150, 64), (1, 1, 45, 288), (1, 1, 45, 293), (1, 2, 32, 1280)],
    ids=["many_blocks", "3_blocks_per_row", "partial_last_block", "8_tile_blocks"],
)
@pytest.mark.parametrize("input_layout, output_layout", LAYOUT_DIRECTIONS, ids=LAYOUT_DIRECTION_IDS)
@pytest.mark.parametrize(
    "input_dtype, output_dtype",
    [(ttnn.bfloat16, ttnn.float32), (ttnn.uint16, ttnn.int32), (ttnn.int32, ttnn.uint16), (ttnn.float32, ttnn.uint8)],
    ids=str,
)
def test_typecast_layout_change_block_split(device, input_dtype, output_dtype, input_layout, output_layout, shape):
    values = _values(shape)
    input_tensor = _to_device(values, input_dtype, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, output_dtype, output_layout=output_layout)
    _check(output, values, output_dtype, output_layout, ttnn.DRAM_MEMORY_CONFIG)


@pytest.mark.parametrize(
    "shape", [(64,), (1, 64), (1, 1, 1, 64), (2, 1, 1, 3, 70)], ids=["rank1", "rank2", "one_row", "rank5"]
)
@pytest.mark.parametrize("input_layout, output_layout", LAYOUT_DIRECTIONS, ids=LAYOUT_DIRECTION_IDS)
def test_typecast_layout_change_small_shapes(device, input_layout, output_layout, shape):
    values = _values(shape)
    input_tensor = _to_device(values, ttnn.bfloat16, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, ttnn.float32, output_layout=output_layout)
    _check(output, values, ttnn.float32, output_layout, ttnn.DRAM_MEMORY_CONFIG)


def _bfp8_exact_values(shape):
    # Multiples of 4 below 256 are exact in bfloat8_b, so casts to integers must be exact too.
    return _values(shape) // 4 * 4


@pytest.mark.parametrize("shape", [(1, 1, 32, 64), (1, 2, 45, 70)], ids=["aligned", "unaligned"])
@pytest.mark.parametrize("output_dtype", [ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8], ids=str)
def test_typecast_bfp8_tile_to_int_rm(device, output_dtype, shape):
    values = _bfp8_exact_values(shape)
    input_tensor = _to_device(values, ttnn.bfloat8_b, TILE, device, ttnn.DRAM_MEMORY_CONFIG)
    assert torch.equal(ttnn.to_torch(input_tensor).to(torch.int64), values.to(torch.int64))
    output = ttnn.typecast(input_tensor, output_dtype, output_layout=RM)
    _check(output, values, output_dtype, RM)


@pytest.mark.parametrize("shape", [(1, 1, 32, 64), (1, 2, 45, 70)], ids=["aligned", "unaligned"])
@pytest.mark.parametrize("input_dtype", [ttnn.int32, ttnn.uint32, ttnn.uint16, ttnn.uint8], ids=str)
def test_typecast_int_rm_to_bfp8_tile(device, input_dtype, shape):
    values = _bfp8_exact_values(shape)
    input_tensor = _to_device(values, input_dtype, RM, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, ttnn.bfloat8_b, output_layout=TILE)
    assert output.layout == TILE and output.dtype == ttnn.bfloat8_b
    assert torch.equal(ttnn.to_torch(output).to(torch.int64), values.to(torch.int64))


@pytest.mark.parametrize("input_layout, output_layout", LAYOUT_DIRECTIONS, ids=LAYOUT_DIRECTION_IDS)
def test_typecast_layout_change_program_cache_entries(device, input_layout, output_layout):
    shape = (1, 1, 64, 96)
    entries = device.num_program_cache_entries()
    for seed in range(2):
        values = _values(shape, seed)
        input_tensor = _to_device(values, ttnn.bfloat16, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
        output = ttnn.typecast(input_tensor, ttnn.float32, output_layout=output_layout)
        _check(output, values, ttnn.float32, output_layout)
        if seed == 0:
            assert device.num_program_cache_entries() > entries
            entries = device.num_program_cache_entries()
        else:
            assert device.num_program_cache_entries() == entries, "same-shape call must hit the program cache"
    # Same shape and dtypes without a layout change is a different program.
    same_layout = ttnn.typecast(_to_device(_values(shape), ttnn.bfloat16, input_layout, device, None), ttnn.float32)
    assert same_layout.layout == input_layout
    assert device.num_program_cache_entries() > entries


def test_typecast_layout_change_tiny_tile_rejected(device, expect_error):
    input_tensor = ttnn.from_torch(
        _values((1, 1, 32, 64)).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=TILE,
        device=device,
        tile=ttnn.Tile((16, 32)),
    )
    with expect_error(RuntimeError, "only 32x32 tiles"):
        ttnn.typecast(input_tensor, ttnn.float32, output_layout=RM)


def test_typecast_output_layout_and_output_tensor_mismatch_rejected(device, expect_error):
    input_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.float32, TILE, device, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.int32, TILE, device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, "must match the output layout"):
        ttnn.typecast(input_tensor, ttnn.int32, output_tensor=output_tensor, output_layout=RM)


_FULL_RANGE_TORCH_DTYPE = {**_TORCH_SOURCE_DTYPE, ttnn.int8: torch.int8}


def _full_range_values(dtype, shape, seed=0):
    generator = torch.Generator().manual_seed(seed)
    if dtype == ttnn.float32:
        values = torch.randn(shape, generator=generator) * 1000
        values.view(-1)[::7] = 1 + 2**-20  # needs more mantissa than bfloat16 / tf32 hold
        return values
    if dtype == ttnn.bfloat16:
        return (torch.randn(shape, generator=generator) * 100).to(torch.bfloat16)
    low, high = {
        ttnn.int32: (-(2**31), 2**31 - 1),
        ttnn.uint32: (0, 2**31 - 1),
        ttnn.uint16: (0, 2**16),
        ttnn.uint8: (0, 2**8),
        ttnn.int8: (-(2**7), 2**7),
    }[dtype]
    return torch.randint(low, high, shape, generator=generator, dtype=torch.int64).to(_FULL_RANGE_TORCH_DTYPE[dtype])


# Full-range values (large and negative integers, fp32 beyond bfloat16 precision) must come out exactly
# as from the same-layout typecast, so 32-bit Dest and unpack modes are exercised.
@pytest.mark.parametrize("shape", [(1, 1, 64, 128), (1, 2, 45, 70)], ids=["aligned", "unaligned"])
@pytest.mark.parametrize("input_layout, output_layout", LAYOUT_DIRECTIONS, ids=LAYOUT_DIRECTION_IDS)
@pytest.mark.parametrize("output_dtype", DTYPES + [ttnn.int8], ids=str)
@pytest.mark.parametrize("input_dtype", DTYPES + [ttnn.int8], ids=str)
def test_typecast_layout_change_full_range(device, input_dtype, output_dtype, input_layout, output_layout, shape):
    values = _full_range_values(input_dtype, shape)
    input_tensor = ttnn.from_torch(
        values, dtype=input_dtype, layout=input_layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    expected = ttnn.to_torch(ttnn.typecast(input_tensor, output_dtype))
    output = ttnn.typecast(input_tensor, output_dtype, output_layout=output_layout)
    assert output.dtype == output_dtype and output.layout == output_layout
    result = ttnn.to_torch(output)
    if output_dtype in (ttnn.float32, ttnn.bfloat16):
        assert torch.equal(
            result.view(torch.int32 if output_dtype == ttnn.float32 else torch.int16),
            expected.view(torch.int32 if output_dtype == ttnn.float32 else torch.int16),
        )
    else:
        assert torch.equal(result.to(torch.int64), expected.to(torch.int64))


# Same layout, preallocated output in a different memory config.
@pytest.mark.parametrize(
    "layout, input_memory, output_memory", [(TILE, "dram", "block"), (RM, "width", "height"), (RM, "dram", "l1")]
)
def test_typecast_memory_config_change_preallocated_output(device, layout, input_memory, output_memory):
    values = _values(SHARDED_SHAPE)
    output_memory_config = _memory_config(output_memory)
    input_tensor = _to_device(values, ttnn.bfloat16, layout, device, _memory_config(input_memory))
    output_tensor = _to_device(torch.zeros_like(values), ttnn.float32, layout, device, output_memory_config)
    output = ttnn.typecast(input_tensor, ttnn.float32, output_tensor=output_tensor)
    assert output.buffer_address() == output_tensor.buffer_address()
    _check(output_tensor, values, ttnn.float32, layout, output_memory_config)


@pytest.mark.parametrize(
    "input_dtype, output_dtype, shape, input_memory, output_memory",
    [
        (ttnn.bfloat8_b, ttnn.bfloat16, SHARDED_SHAPE, "height", "block"),
        (ttnn.bfloat16, ttnn.bfloat8_b, SHARDED_SHAPE, "width", "dram"),
        (ttnn.bfloat16, ttnn.bfloat8_b, (1, 1, 64, 70), "dram", "l1"),
    ],
    ids=["bfp8_height_to_bf16_block", "bf16_width_to_bfp8_dram", "bf16_dram_to_bfp8_l1_w70"],
)
def test_typecast_block_float_memory_config_change(
    device, input_dtype, output_dtype, shape, input_memory, output_memory
):
    values = _values(shape)
    output_memory_config = _memory_config(output_memory)
    input_tensor = _to_device(values, input_dtype, TILE, device, _memory_config(input_memory))
    output = ttnn.typecast(input_tensor, output_dtype, memory_config=output_memory_config)
    expected = ttnn.to_torch(input_tensor).to(torch.float32)
    assert output.dtype == output_dtype and output.memory_config().memory_layout == output_memory_config.memory_layout
    assert_with_pcc(expected, ttnn.to_torch(output).to(torch.float32), 0.999)


# 8-bit row-major data in L1 can use one-tile-wide blocks (32 and 11 tiles wide have no other block width).
@pytest.mark.parametrize("width", [32, 352])
@pytest.mark.parametrize("input_layout, output_layout", LAYOUT_DIRECTIONS, ids=LAYOUT_DIRECTION_IDS)
def test_typecast_layout_change_uint8_l1_narrow_blocks(device, input_layout, output_layout, width):
    values = _values((1, 1, 64, width))
    input_tensor = _to_device(values, ttnn.uint8, input_layout, device, ttnn.L1_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, ttnn.uint8, memory_config=ttnn.L1_MEMORY_CONFIG, output_layout=output_layout)
    _check(output, values, ttnn.uint8, output_layout, ttnn.L1_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, ttnn.int32, memory_config=ttnn.L1_MEMORY_CONFIG, output_layout=output_layout)
    _check(output, values, ttnn.int32, output_layout, ttnn.L1_MEMORY_CONFIG)


# Columns past the logical width are padding inside a block-float exponent group; a large value in the next
# row must not leak into them and coarsen the small values that share the group.
def test_typecast_rm_to_bfp8_padding_isolated(device):
    values = torch.randint(0, 8, (1, 1, 64, 45), generator=torch.Generator().manual_seed(0), dtype=torch.int32)
    values[..., 0] = 2**14
    input_tensor = _to_device(values, ttnn.bfloat16, RM, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, ttnn.bfloat8_b, output_layout=TILE)
    tiled = ttnn.to_layout(input_tensor, TILE)
    expected = ttnn.to_torch(ttnn.typecast(tiled, ttnn.bfloat8_b))
    assert torch.equal(ttnn.to_torch(output), expected)
    assert torch.equal(ttnn.to_torch(output)[..., 32:].to(torch.int64), values[..., 32:].to(torch.int64))


@pytest.mark.parametrize("input_layout, output_layout", LAYOUT_DIRECTIONS, ids=LAYOUT_DIRECTION_IDS)
def test_typecast_layout_change_empty(device, input_layout, output_layout):
    values = torch.zeros((1, 1, 0, 32), dtype=torch.int32)
    input_tensor = _to_device(values, ttnn.bfloat16, input_layout, device, ttnn.DRAM_MEMORY_CONFIG)
    output = ttnn.typecast(input_tensor, ttnn.float32, output_layout=output_layout)
    assert output.dtype == ttnn.float32 and output.layout == output_layout
    assert list(output.shape) == [1, 1, 0, 32]


def test_typecast_layout_change_preallocated_wrong_shape_rejected(device, expect_error):
    input_tensor = _to_device(_values((1, 1, 32, 64)), ttnn.bfloat16, TILE, device, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = _to_device(_values((1, 1, 32, 32)), ttnn.float32, RM, device, ttnn.DRAM_MEMORY_CONFIG)
    with expect_error(RuntimeError, "Preallocated output tensor must be"):
        ttnn.typecast(input_tensor, ttnn.float32, output_tensor=output_tensor, output_layout=RM)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Data-movement ops that take a dtype argument must convert integer <-> float values, not reinterpret bits.

Every entry point below is compared against torch's .to() for int32/uint32 <-> float32/bfloat16 in both directions.
All values are integral and exactly representable in bfloat16, so the comparison is exact.
"""

import pytest
import torch

import ttnn

SHAPE = [1, 1, 64, 32]
NUM_ELEMENTS = 64 * 32

TORCH_DTYPES = {
    ttnn.int32: torch.int32,
    ttnn.uint32: torch.int32,  # torch has no uint32 for from_torch; non-negative int32 values are used
    ttnn.float32: torch.float32,
    ttnn.bfloat16: torch.bfloat16,
}

INT_DTYPES = [ttnn.int32, ttnn.uint32]
FLOAT_DTYPES = [ttnn.float32, ttnn.bfloat16]

SHARD_MEMCFG = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    ttnn.BufferType.L1,
    ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))}),
        [32, 32],
        ttnn.ShardOrientation.ROW_MAJOR,
    ),
)


def make_values(in_dtype, out_dtype):
    # Integral values 0..255 exactly representable in bf16. Negatives only when both ends are signed.
    base = torch.arange(NUM_ELEMENTS, dtype=torch.int32) % 256
    if ttnn.uint32 not in (in_dtype, out_dtype):
        base = base - 100
    return base.reshape(SHAPE)


def upload(device, values, dtype, layout):
    torch_values = values.to(TORCH_DTYPES[dtype])
    return torch_values, ttnn.from_torch(torch_values, dtype=dtype, layout=layout, device=device)


def check(result, expected_values, out_dtype):
    assert result.dtype == out_dtype
    got = ttnn.to_torch(result).reshape(SHAPE)
    expected = expected_values.to(TORCH_DTYPES[out_dtype]).reshape(SHAPE)
    assert torch.equal(
        got.to(torch.float32), expected.to(torch.float32)
    ), f"{int((got.to(torch.float32) != expected.to(torch.float32)).sum())} of {NUM_ELEMENTS} values differ"


def run_tile_entry(device, entry, in_dtype, out_dtype):
    values = make_values(in_dtype, out_dtype)
    torch_in, tt_in = upload(device, values, in_dtype, ttnn.TILE_LAYOUT)
    if entry == "clone":
        out = ttnn.clone(tt_in, dtype=out_dtype)
    elif entry == "experimental_typecast":
        out = ttnn.experimental.typecast(tt_in, out_dtype)
    elif entry == "to_memory_config_interleaved":
        out = ttnn.to_memory_config(tt_in, ttnn.L1_MEMORY_CONFIG, dtype=out_dtype)
    elif entry == "to_memory_config_sharded":
        out = ttnn.to_memory_config(tt_in, SHARD_MEMCFG, dtype=out_dtype)
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
    elif entry == "interleaved_to_sharded":
        out = ttnn.interleaved_to_sharded(tt_in, SHARD_MEMCFG, output_dtype=out_dtype)
        out = ttnn.sharded_to_interleaved(out, ttnn.DRAM_MEMORY_CONFIG)
    elif entry == "sharded_to_interleaved":
        sharded = ttnn.interleaved_to_sharded(tt_in, SHARD_MEMCFG)
        out = ttnn.sharded_to_interleaved(sharded, ttnn.DRAM_MEMORY_CONFIG, output_dtype=out_dtype)
    else:
        raise AssertionError(entry)
    check(out, torch_in, out_dtype)


def run_row_major_entry(device, entry, in_dtype, out_dtype):
    values = make_values(in_dtype, out_dtype)
    torch_in, tt_in = upload(device, values, in_dtype, ttnn.ROW_MAJOR_LAYOUT)
    if entry == "tilize":
        out = ttnn.tilize(tt_in, dtype=out_dtype)
    elif entry == "tilize_with_val_padding":
        out = ttnn.tilize_with_val_padding(tt_in, SHAPE, 0, dtype=out_dtype)
    elif entry == "to_layout":
        out = ttnn.to_layout(tt_in, ttnn.TILE_LAYOUT, dtype=out_dtype)
    else:
        raise AssertionError(entry)
    assert out.layout == ttnn.TILE_LAYOUT
    check(out, torch_in, out_dtype)


TILE_ENTRIES = [
    "clone",
    "to_memory_config_interleaved",
    "to_memory_config_sharded",
    "interleaved_to_sharded",
    "sharded_to_interleaved",
]
ROW_MAJOR_ENTRIES = ["tilize_with_val_padding"]


@pytest.mark.parametrize("entry", TILE_ENTRIES)
@pytest.mark.parametrize("int_dtype", INT_DTYPES)
@pytest.mark.parametrize("float_dtype", FLOAT_DTYPES)
def test_tile_entry_int_to_float(device, entry, int_dtype, float_dtype):
    run_tile_entry(device, entry, int_dtype, float_dtype)


@pytest.mark.parametrize("entry", TILE_ENTRIES)
@pytest.mark.parametrize("int_dtype", INT_DTYPES)
@pytest.mark.parametrize("float_dtype", FLOAT_DTYPES)
def test_tile_entry_float_to_int(device, entry, int_dtype, float_dtype):
    run_tile_entry(device, entry, float_dtype, int_dtype)


@pytest.mark.parametrize("entry", ROW_MAJOR_ENTRIES)
@pytest.mark.parametrize("int_dtype", INT_DTYPES)
@pytest.mark.parametrize("float_dtype", FLOAT_DTYPES)
def test_row_major_entry_int_to_float(device, entry, int_dtype, float_dtype):
    run_row_major_entry(device, entry, int_dtype, float_dtype)


@pytest.mark.parametrize("entry", ROW_MAJOR_ENTRIES)
@pytest.mark.parametrize("int_dtype", INT_DTYPES)
@pytest.mark.parametrize("float_dtype", FLOAT_DTYPES)
def test_row_major_entry_float_to_int(device, entry, int_dtype, float_dtype):
    run_row_major_entry(device, entry, float_dtype, int_dtype)


@pytest.mark.parametrize("entry", TILE_ENTRIES)
@pytest.mark.parametrize("dtype_pair", [(ttnn.float32, ttnn.bfloat16), (ttnn.bfloat16, ttnn.float32)])
def test_tile_entry_float_to_float_unchanged(device, entry, dtype_pair):
    run_tile_entry(device, entry, *dtype_pair)


@pytest.mark.parametrize("entry", TILE_ENTRIES)
@pytest.mark.parametrize("dtype", [ttnn.int32, ttnn.float32])
def test_tile_entry_same_dtype_unchanged(device, entry, dtype):
    run_tile_entry(device, entry, dtype, dtype)

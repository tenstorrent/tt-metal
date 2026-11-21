# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn

pytestmark = pytest.mark.use_module_device

TILE_HEIGHT, TILE_WIDTH = 32, 64
TILE_SHAPE = (1, 1, TILE_HEIGHT, TILE_WIDTH)
TILE_ELEMS = TILE_HEIGHT * TILE_WIDTH

# Covers 0, the low-byte boundary, a value above the 16-bit sign bit and UINT16_MAX, because the
# defect landed the source bits in the target format's exponent field.
UINT16_VALUES = [0, 1, 2, 7, 15, 100, 255, 256, 511, 1024, 32767, 32768, 40000, 65534, 65535]
UINT8_VALUES = [0, 1, 2, 7, 15, 100, 127, 128, 254, 255]

WIDENING_DTYPES = [ttnn.int32, ttnn.uint32, ttnn.bfloat16]

# Every sharded tensor in this file is TILE layout, so the shard shape must be a multiple of the
# tile: tt_metal/impl/tensor/spec/layout/tensor_layout.cpp:44 rejects anything else with
# "Physical shard shape ... must be tile ... sized!". Two cores across the width give exactly one
# 32x32 tile each for a 32x64 tensor. The row-major inputs below (tilize and friends) are never
# sharded, so they need no shard shape at all.
SHARD_SHAPE = (TILE_HEIGHT, TILE_HEIGHT)
SHARD_GRID = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0))})

ENTRY_POINTS = [
    "copy_into_preallocated",
    "assign_dtype",
    "assign_into_preallocated",
    "interleaved_to_sharded_mem_config",
    "interleaved_to_sharded_grid",
    "sharded_to_interleaved",
    "tilize",
    "tilize_with_val_padding",
    "to_layout",
]


def _tiled(values):
    vals = torch.tensor(values, dtype=torch.int32)
    return vals.repeat(TILE_ELEMS // len(vals) + 1)[:TILE_ELEMS].reshape(TILE_SHAPE)


def _to_device(tiled, dtype, layout, device, memory_config=None):
    return ttnn.from_torch(
        tiled,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG,
    )


def _sharded_memory_config():
    # sharded_to_interleaved rejects a DRAM-sharded input outright
    # (sharded_to_interleaved_device_operation.cpp:29), and L1 is what i2s targets in practice.
    shard_spec = ttnn.ShardSpec(SHARD_GRID, list(SHARD_SHAPE), ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _sharded(tiled, dtype, device):
    # Built by an exact same-dtype interleaved_to_sharded, so only the entry point under test sees
    # a dtype change and a shard spec is present for sharded_to_interleaved's early return.
    interleaved = _to_device(tiled, dtype, ttnn.TILE_LAYOUT, device)
    return ttnn.interleaved_to_sharded(interleaved, _sharded_memory_config(), dtype)


def _preallocated(output_dtype, device):
    return _to_device(torch.zeros(TILE_SHAPE, dtype=torch.int32), output_dtype, ttnn.TILE_LAYOUT, device)


def _run_entry_point(entry, tiled, tiled_input, rm_input, device, output_dtype):
    if entry == "copy_into_preallocated":
        return ttnn.copy(tiled_input, _preallocated(output_dtype, device))
    if entry == "assign_dtype":
        return ttnn.assign(tiled_input, memory_config=ttnn.DRAM_MEMORY_CONFIG, dtype=output_dtype)
    if entry == "assign_into_preallocated":
        return ttnn.assign(tiled_input, _preallocated(output_dtype, device))
    if entry == "interleaved_to_sharded_mem_config":
        return ttnn.interleaved_to_sharded(tiled_input, _sharded_memory_config(), output_dtype=output_dtype)
    if entry == "interleaved_to_sharded_grid":
        # The grid overload always stages through L1, so it needs no separate memory config.
        return ttnn.interleaved_to_sharded(
            tiled_input,
            SHARD_GRID,
            list(SHARD_SHAPE),
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            output_dtype=output_dtype,
        )
    if entry == "sharded_to_interleaved":
        sharded_input = _sharded(tiled, tiled_input.dtype, device)
        return ttnn.sharded_to_interleaved(sharded_input, ttnn.DRAM_MEMORY_CONFIG, output_dtype=output_dtype)
    if entry == "tilize":
        return ttnn.tilize(rm_input, dtype=output_dtype)
    if entry == "tilize_with_val_padding":
        # The padded output shape must carry the input's rank: the op indexes it as [-1]/[-2] and the
        # binding validates that rank against the input, so a rank-2 shape fails for a rank-4 input.
        return ttnn.tilize_with_val_padding(rm_input, list(TILE_SHAPE), 0, dtype=output_dtype)
    if entry == "to_layout":
        return ttnn.to_layout(rm_input, ttnn.TILE_LAYOUT, dtype=output_dtype)
    raise ValueError(f"unknown entry point {entry}")


def _assert_values(output, tiled, output_dtype):
    got = ttnn.to_torch(output)
    if output_dtype == ttnn.bfloat16:
        assert torch.equal(got.to(torch.float32), tiled.to(torch.bfloat16).to(torch.float32))
    else:
        assert torch.equal(
            got.to(torch.int64), tiled.to(torch.int64)
        ), f"got {got.flatten()[:8].tolist()} where {tiled.to(torch.int64).flatten()[:8].tolist()} was expected"


def _inputs(tiled, dtype, device):
    return (
        _to_device(tiled, dtype, ttnn.TILE_LAYOUT, device),
        _to_device(tiled, dtype, ttnn.ROW_MAJOR_LAYOUT, device),
    )


@pytest.mark.parametrize("output_dtype", WIDENING_DTYPES)
@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_uint16_widening_is_by_value(device, entry, output_dtype):
    """A uint16 source converted by a data-movement entry point must zero-extend, not re-read the bits.

    The copy kernel moves whole tiles between differently formatted CBs, so before the fix every one of
    these returned the source bits reinterpreted in the output format (uint16 0 as int32 1024).
    """
    tiled = _tiled(UINT16_VALUES)
    tiled_input, rm_input = _inputs(tiled, ttnn.uint16, device)
    output = _run_entry_point(entry, tiled, tiled_input, rm_input, device, output_dtype)
    _assert_values(output, tiled, output_dtype)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_uint16_same_dtype_stays_exact(device, entry):
    """Control: no dtype change means no conversion, so the values must still be the source bits."""
    tiled = _tiled(UINT16_VALUES)
    tiled_input, rm_input = _inputs(tiled, ttnn.uint16, device)
    output = _run_entry_point(entry, tiled, tiled_input, rm_input, device, ttnn.uint16)
    assert output.dtype == ttnn.uint16
    _assert_values(output, tiled, ttnn.uint16)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_uint8_widening_stays_exact(device, entry):
    """Control: a uint8 source already converts correctly through the copy kernel."""
    if entry == "tilize_with_val_padding":
        # This op has no dtype conversion for a uint8 source to widen into: its validator accepts only
        # bfloat16, float32, int32, uint32, uint16 and fp8_e4m3 inputs.
        pytest.skip("tilize_with_val_padding does not accept a uint8 source")
    tiled = _tiled(UINT8_VALUES)
    tiled_input, rm_input = _inputs(tiled, ttnn.uint8, device)
    output = _run_entry_point(entry, tiled, tiled_input, rm_input, device, ttnn.int32)
    assert output.dtype == ttnn.int32
    _assert_values(output, tiled, ttnn.int32)

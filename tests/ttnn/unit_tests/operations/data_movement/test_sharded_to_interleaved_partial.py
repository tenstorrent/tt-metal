# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input


def _docstring_sharded_input_params(op):
    """One ``pytest.param(memory_layout, buffer_type)`` per shard layout and buffer type on the ``- Input:`` item of
    the op's Memory Support list, which reads like ``HEIGHT_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED in DRAM``.

    Every word before ``in`` other than ``and`` must name a ``ttnn.TensorMemoryLayout``, and the word after it a
    ``ttnn.BufferType``. If the item can't be read, a single case fails with the reason.
    """
    try:
        lines = op.__doc__.split("Memory Support:", 1)[1].splitlines()
        start = next((i for i, line in enumerate(lines) if line.strip().startswith("- Input:")), None)
        assert start is not None, "no '- Input:' Memory Support item"
        indent = len(lines[start]) - len(lines[start].lstrip())
        item = lines[start].strip()[len("- Input:") :]
        for line in lines[start + 1 :]:  # join the item's wrapped lines
            if not line.strip() or line.strip().startswith("- ") or len(line) - len(line.lstrip()) <= indent:
                break
            item += " " + line.strip()
        params = []
        for clause in item.split(";"):
            match = re.fullmatch(r"\s*(.+?)\s+in\s+(\w+)\s*", clause)
            assert match, f"no '<shard layouts> in <buffer type>' in {clause!r}"
            buffer_type = getattr(ttnn.BufferType, match[2], None)
            assert isinstance(buffer_type, ttnn.BufferType), f"not a ttnn.BufferType: {match[2]}"
            names = [name for name in re.findall(r"[^\s,]+", match[1]) if name != "and"]
            assert names, f"no shard layouts in {clause!r}"
            for name in names:
                memory_layout = getattr(ttnn.TensorMemoryLayout, name, None)
                assert isinstance(memory_layout, ttnn.TensorMemoryLayout), f"not a ttnn.TensorMemoryLayout: {name}"
                params.append(pytest.param(memory_layout, buffer_type, id=f"{name}-{match[2]}"))
        return params
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the Input Memory Support item from the {op} docstring: {error}")
        return [pytest.param(reason, reason, id="docstring-unreadable")]


def _docstring_cache_dtype_pairs(op):
    """One ``pytest.param(input_dtype, cache_dtype)`` per ordered pair of the dtypes that the op's ``cache_tensor``
    line lists in "input of <dtypes> can be converted". Each name must be a ``ttnn.DataType``."""
    try:
        line = next((line.strip() for line in op.__doc__.splitlines() if line.strip().startswith("cache_tensor (")), "")
        match = re.search(r"input of (.+?) can be converted", line)
        assert match, f"no 'input of <dtypes> can be converted' on: {line!r}"
        names = re.split(r",\s*|\s+or\s+", match[1])
        unknown = [name for name in names if not isinstance(getattr(ttnn.DataType, name, None), ttnn.DataType)]
        assert not unknown, f"not a ttnn.DataType: {unknown} on: {line!r}"
        assert len(names) > 1, f"fewer than two dtypes on: {line!r}"
        return [
            pytest.param(getattr(ttnn.DataType, a), getattr(ttnn.DataType, b), id=f"{a}-to-{b}")
            for a in names
            for b in names
            if a != b
        ]
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the cache_tensor line from the {op} docstring: {error}")
        return [pytest.param(reason, reason, id="docstring-unreadable")]


def _convertible_input(shape, input_dtype, cache_dtype):
    """Values that both dtypes hold exactly, and no NaN, as the cache_tensor line asks. With BFLOAT8_B, the values are
    +-(1 + k / 64): they share one exponent and fit the 6 fraction bits a BFLOAT8_B value with its block's exponent
    has."""
    if ttnn.bfloat8_b not in (input_dtype, cache_dtype):
        return full_range_input(shape, ttnn.bfloat16, nan=False).float()
    torch.manual_seed(0)
    magnitude = 1 + torch.randint(0, 64, shape) / 64
    return torch.where(torch.randint(0, 2, shape, dtype=torch.bool), -magnitude, magnitude)


# The cache tensor has 2 * 3 * 64 = 384 rows of width 128, written in 3 slices of 128 rows that cross the batch and
# channel boundaries. Shard shape and core grid size (x, y) of each shard layout, for one [1, 1, 128, 128] slice.
_CACHE_SHAPE = (2, 3, 64, 128)
_NUM_SLICES = 3
_SLICE_SHARDS = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: ((32, 128), (2, 2)),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: ((128, 64), (2, 1)),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: ((64, 64), (2, 2)),
}


def _write_every_slice(device, values, dtype, layout, memory_layout, buffer_type, cache_dtype, cache_buffer_type):
    """Shards each slice of ``values`` (``_CACHE_SHAPE``) in ``dtype`` and writes it with
    ttnn.sharded_to_interleaved_partial into a cache tensor of ``cache_dtype`` whose values all differ from the slices'.
    After each write, the slice's rows must hold the slice and every other row must be unchanged. When the dtypes
    differ, zeros are compared by value."""
    width = _CACHE_SHAPE[-1]
    slice_height = values.numel() // width // _NUM_SLICES
    shard_shape, (grid_x, grid_y) = _SLICE_SHARDS[memory_layout]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    slice_memory_config = ttnn.MemoryConfig(
        memory_layout, buffer_type, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )
    slices = [
        ttnn.from_torch(
            slice_values.reshape(1, 1, slice_height, width),
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=slice_memory_config,
        )
        for slice_values in values.reshape(-1, width).split(slice_height)
    ]
    slice_rows = torch.cat([ttnn.to_torch(sharded_slice).reshape(-1, width) for sharded_slice in slices])
    # Every value of the cache differs from the slices', so the test sees each one written.
    cache_memory_config = interleaved_memory_config(cache_buffer_type)
    cache_tensor = ttnn.from_torch(
        torch.where(slice_rows == 1, 2, 1).to(slice_rows.dtype).reshape(_CACHE_SHAPE),
        dtype=cache_dtype,
        layout=layout,
        device=device,
        memory_config=cache_memory_config,
    )
    expected = ttnn.to_torch(cache_tensor).reshape(-1, width).to(torch.float64)

    for slice_index, sharded_slice in enumerate(slices):
        output = ttnn.sharded_to_interleaved_partial(sharded_slice, cache_tensor, _NUM_SLICES, slice_index)

        assert output.buffer_address() == cache_tensor.buffer_address()
        assert output.dtype == cache_dtype
        assert output.layout == layout
        assert output.memory_config() == cache_memory_config
        rows = slice(slice_index * slice_height, (slice_index + 1) * slice_height)
        expected[rows] = slice_rows[rows].to(torch.float64)
        assert_exact(expected.reshape(_CACHE_SHAPE), ttnn.to_torch(cache_tensor), signed_zeros=dtype == cache_dtype)


@pytest.mark.parametrize(
    "cache_buffer_type",
    docstring_interleaved_buffer_type_params(ttnn.sharded_to_interleaved_partial, label="Output"),
)
@pytest.mark.parametrize(
    "memory_layout, buffer_type", _docstring_sharded_input_params(ttnn.sharded_to_interleaved_partial)
)
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.sharded_to_interleaved_partial))
def test_sharded_to_interleaved_partial_docstring_supported_inputs(
    dtype, layout, memory_layout, buffer_type, cache_buffer_type, device
):
    """Writes every slice of a sharded input of each dtype, layout and shard layout the
    ttnn.sharded_to_interleaved_partial docstring lists into a cache tensor in each interleaved buffer type it lists."""
    check_readable(dtype, layout, memory_layout, buffer_type, cache_buffer_type)
    _write_every_slice(
        device,
        full_range_input(_CACHE_SHAPE, dtype),
        dtype,
        layout,
        memory_layout,
        buffer_type,
        dtype,
        cache_buffer_type,
    )


@pytest.mark.parametrize(
    "cache_buffer_type",
    docstring_interleaved_buffer_type_params(ttnn.sharded_to_interleaved_partial, label="Output"),
)
@pytest.mark.parametrize("input_dtype, cache_dtype", _docstring_cache_dtype_pairs(ttnn.sharded_to_interleaved_partial))
def test_sharded_to_interleaved_partial_docstring_cache_dtype(input_dtype, cache_dtype, cache_buffer_type, device):
    """Writes every slice of a HEIGHT_SHARDED TILE input into a cache tensor of every other dtype the
    ttnn.sharded_to_interleaved_partial cache_tensor line names. The input values are exact in both dtypes, so the
    cache tensor holds them."""
    check_readable(input_dtype, cache_dtype, cache_buffer_type)
    _write_every_slice(
        device,
        _convertible_input(_CACHE_SHAPE, input_dtype, cache_dtype),
        input_dtype,
        ttnn.TILE_LAYOUT,
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        cache_dtype,
        cache_buffer_type,
    )


@pytest.mark.parametrize(
    "memory_layout, buffer_type", _docstring_sharded_input_params(ttnn.sharded_to_interleaved_partial)
)
def test_sharded_to_interleaved_partial_docstring_round_trip(memory_layout, buffer_type, device):
    """Copies every slice of a tensor with ttnn.interleaved_to_sharded_partial into each shard layout the
    ttnn.sharded_to_interleaved_partial docstring lists, and writes it back with the same num_slices and slice_index
    into a cache tensor whose values all differ from the input's: the cache tensor ends up equal to the input."""
    check_readable(memory_layout, buffer_type)
    if buffer_type != ttnn.BufferType.L1:
        pytest.skip("ttnn.interleaved_to_sharded_partial copies slices into L1")
    input_tensor = ttnn.from_torch(
        full_range_input(_CACHE_SHAPE, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    expected = ttnn.to_torch(input_tensor)
    cache_tensor = ttnn.from_torch(
        torch.where(expected == 1, 2, 1).to(expected.dtype), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    shard_shape, grid_size = _SLICE_SHARDS[memory_layout]
    for slice_index in range(_NUM_SLICES):
        sharded_slice = ttnn.interleaved_to_sharded_partial(
            input_tensor,
            ttnn.CoreCoord(*grid_size),
            shard_shape,
            _NUM_SLICES,
            slice_index,
            memory_layout,
            ttnn.ShardOrientation.ROW_MAJOR,
        )
        ttnn.sharded_to_interleaved_partial(sharded_slice, cache_tensor, _NUM_SLICES, slice_index)

    assert_exact(expected, ttnn.to_torch(cache_tensor))

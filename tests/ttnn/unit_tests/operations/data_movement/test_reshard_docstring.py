# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    _memory_support_items,
    check_readable,
    docstring_dtype_layout_params,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input

HEIGHT = ttnn.TensorMemoryLayout.HEIGHT_SHARDED
WIDTH = ttnn.TensorMemoryLayout.WIDTH_SHARDED
BLOCK = ttnn.TensorMemoryLayout.BLOCK_SHARDED

# The Memory Support item that limits ROW_MAJOR reshards.
_ROW_MAJOR_ITEM = "ROW_MAJOR: the input and output must have the same memory layout, and both be HEIGHT_SHARDED when either is in DRAM"

# _SHAPE has 2 * 2 * 64 = 256 rows of width 128. Shard shape and core grid size (x, y) of each memory layout, for the
# input and for the output: they differ, so a reshard between two equal placements still moves the data. Every shard
# is tile-aligned, at least 32 elements (16 bytes of UINT8) wide, and DRAM shards use at most 4 banks.
_SHAPE = (2, 2, 64, 128)
_INPUT_SHARDS = {
    HEIGHT: ((64, 128), (4, 1)),
    WIDTH: ((256, 64), (2, 1)),
    BLOCK: ((128, 64), (2, 2)),
}
_OUTPUT_SHARDS = {
    HEIGHT: ((128, 128), (2, 1)),
    WIDTH: ((256, 32), (4, 1)),
    BLOCK: ((64, 32), (4, 4)),
}


def _sharded_placements(doc, label):
    """``(id, (memory_layout, buffer_type))`` per shard layout and buffer type on the ``- <label>:`` item of the
    Memory Support list, which reads like ``HEIGHT_SHARDED and BLOCK_SHARDED in L1; HEIGHT_SHARDED in DRAM``.

    Each clause names at least one shard layout. Every name before ``in`` must be a sharded
    ``ttnn.TensorMemoryLayout`` and the word after it a ``ttnn.BufferType``; anything else raises.
    """
    items = [item for item in _memory_support_items(doc) if item.startswith(f"{label}:")]
    assert len(items) == 1, f"expected one '- {label}:' Memory Support item, got {items}"
    placements = []
    for clause in items[0][len(label) + 1 :].split(";"):
        match = re.fullmatch(r"\s*(.+?)\s+in\s+(\w+)\s*", clause)
        assert match, f"no '<shard layouts> in <buffer type>' in {clause!r}"
        layout_names, buffer_name = match.groups()
        buffer_type = getattr(ttnn.BufferType, buffer_name, None)
        assert isinstance(buffer_type, ttnn.BufferType), f"not a ttnn.BufferType: {buffer_name!r}"
        for name in re.split(r"\s*,\s*|\s+and\s+", layout_names):
            memory_layout = getattr(ttnn.TensorMemoryLayout, name, None)
            assert isinstance(memory_layout, ttnn.TensorMemoryLayout) and name.endswith(
                "_SHARDED"
            ), f"not a sharded ttnn.TensorMemoryLayout: {name!r} in {clause!r}"
            placements.append((f"{name}-{buffer_name}", (memory_layout, buffer_type)))
    return placements


def _unreadable(op, error, arity):
    reason = UnreadableDocstring(f"could not read the Memory Support list from the {op} docstring: {error}")
    return [pytest.param(*([reason] * arity), id="docstring-unreadable")]


def _row_major_pair_documented(input_layout, input_buffer, output_layout, output_buffer):
    """The ROW_MAJOR item: the same memory layout on both sides, HEIGHT_SHARDED when either side is in DRAM."""
    if input_layout != output_layout:
        return False
    return input_layout == HEIGHT or ttnn.BufferType.DRAM not in (input_buffer, output_buffer)


def _docstring_reshard_params(op):
    """One ``pytest.param(dtype, layout, input_placement, output_placement)`` per row of the op's dtype/layout table
    and per pair of a placement on its ``- Input:`` and one on its ``- Output:`` Memory Support item. While the
    docstring has the ROW_MAJOR item, ROW_MAJOR pairs keep the memory layout, and are HEIGHT_SHARDED when either side
    is in DRAM."""
    try:
        inputs = _sharded_placements(op.__doc__, "Input")
        outputs = _sharded_placements(op.__doc__, "Output")
        row_major_item = _ROW_MAJOR_ITEM in _memory_support_items(op.__doc__)
    except Exception as error:
        return _unreadable(op, error, arity=4)
    params = []
    for dtype_layout in docstring_dtype_layout_params(op):
        dtype, layout = dtype_layout.values
        if isinstance(dtype, UnreadableDocstring):
            return [pytest.param(dtype, dtype, dtype, dtype, id=dtype_layout.id)]
        for input_id, (input_layout, input_buffer) in inputs:
            for output_id, (output_layout, output_buffer) in outputs:
                if (
                    row_major_item
                    and layout == ttnn.ROW_MAJOR_LAYOUT
                    and not _row_major_pair_documented(input_layout, input_buffer, output_layout, output_buffer)
                ):
                    continue
                params.append(
                    pytest.param(
                        dtype,
                        layout,
                        (input_layout, input_buffer),
                        (output_layout, output_buffer),
                        id=f"{dtype_layout.id}-{input_id}-to-{output_id}",
                    )
                )
    return params


def _docstring_row_major_height_params(op):
    """One ``pytest.param(dtype, input_buffer_type, output_buffer_type)`` per ROW_MAJOR dtype of the op's table and
    per buffer type that holds HEIGHT_SHARDED on its ``- Input:`` and ``- Output:`` Memory Support items."""
    try:
        inputs = [(i, bt) for i, (ml, bt) in _sharded_placements(op.__doc__, "Input") if ml == HEIGHT]
        outputs = [(i, bt) for i, (ml, bt) in _sharded_placements(op.__doc__, "Output") if ml == HEIGHT]
        assert inputs and outputs, "no HEIGHT_SHARDED input or output"
    except Exception as error:
        return _unreadable(op, error, arity=3)
    params = []
    for dtype_layout in docstring_dtype_layout_params(op):
        dtype, layout = dtype_layout.values
        if isinstance(dtype, UnreadableDocstring):
            return [pytest.param(dtype, dtype, dtype, id=dtype_layout.id)]
        if layout != ttnn.ROW_MAJOR_LAYOUT:
            continue
        for input_id, input_buffer in inputs:
            for output_id, output_buffer in outputs:
                params.append(
                    pytest.param(dtype, input_buffer, output_buffer, id=f"{dtype_layout.id}-{input_id}-to-{output_id}")
                )
    return params


def _sharded_memory_config(memory_layout, buffer_type, shard_shape, grid_size):
    grid_x, grid_y = grid_size
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(memory_layout, buffer_type, shard_spec)


def _differing_tensor(expected, torch_dtype, dtype, layout, memory_config, device):
    """A tensor in ``memory_config`` whose every value differs from ``expected``: 2 where it is 1, else 1."""
    values = torch.where(expected.to(torch.float64) == 1, 2, 1).to(torch_dtype)
    tensor = ttnn.from_torch(values, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    assert not (ttnn.to_torch(tensor).to(torch.float64) == expected.to(torch.float64)).any()
    return tensor


def _run_reshard(input_tensor, torch_dtype, memory_config, device):
    """Reshards into ``memory_config`` as a new tensor and into ``output_tensor``, and compares both exactly."""
    expected = ttnn.to_torch(input_tensor)
    dtype, layout = input_tensor.dtype, input_tensor.layout
    output_tensor = _differing_tensor(expected, torch_dtype, dtype, layout, memory_config, device)
    outputs = [
        ttnn.reshard(input_tensor, memory_config),
        ttnn.reshard(input_tensor, memory_config, output_tensor=output_tensor),
        output_tensor,
    ]
    for output in outputs:
        assert output.dtype == dtype
        assert output.layout == layout
        assert output.memory_config() == memory_config
        assert_exact(expected, ttnn.to_torch(output))


@pytest.mark.parametrize("dtype, layout, input_placement, output_placement", _docstring_reshard_params(ttnn.reshard))
def test_reshard_docstring_supported_inputs(dtype, layout, input_placement, output_placement, device):
    """Reshards a tensor of every dtype and layout the ttnn.reshard docstring lists, from every sharded input
    placement it lists into every sharded output placement, as a new tensor and into ``output_tensor``."""
    check_readable(dtype, layout, input_placement, output_placement)
    values = full_range_input(_SHAPE, dtype)
    input_tensor = ttnn.from_torch(
        values,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=_sharded_memory_config(*input_placement, *_INPUT_SHARDS[input_placement[0]]),
    )
    memory_config = _sharded_memory_config(*output_placement, *_OUTPUT_SHARDS[output_placement[0]])
    _run_reshard(input_tensor, values.dtype, memory_config, device)


@pytest.mark.parametrize(
    "dtype, input_buffer_type, output_buffer_type", _docstring_row_major_height_params(ttnn.reshard)
)
def test_reshard_docstring_row_major_height_sharded_any_width(dtype, input_buffer_type, output_buffer_type, device):
    """HEIGHT_SHARDED ROW_MAJOR reshards take any shard width: the 16-byte rule on the output_memory_config line of
    the ttnn.reshard docstring is for WIDTH_SHARDED and BLOCK_SHARDED. 9 elements are 9, 18 or 36 bytes."""
    check_readable(dtype, input_buffer_type, output_buffer_type)
    shape = (1, 1, 64, 9)
    values = full_range_input(shape, dtype)
    input_tensor = ttnn.from_torch(
        values,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_sharded_memory_config(HEIGHT, input_buffer_type, (32, 9), (2, 1)),
    )
    memory_config = _sharded_memory_config(HEIGHT, output_buffer_type, (16, 9), (4, 1))
    _run_reshard(input_tensor, values.dtype, memory_config, device)

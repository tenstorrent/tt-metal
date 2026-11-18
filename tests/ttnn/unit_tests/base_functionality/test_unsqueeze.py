# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

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


@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [
        ((1, 1, 253), 2, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 253), -2, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 253), 1, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 253), -2, ttnn.TILE_LAYOUT),
        ((1, 253), 1, ttnn.TILE_LAYOUT),
        ((57, 83), 1, ttnn.TILE_LAYOUT),
        ((123, 259), -2, ttnn.TILE_LAYOUT),
        ((57, 83), 1, ttnn.ROW_MAJOR_LAYOUT),
        ((123, 259), -2, ttnn.ROW_MAJOR_LAYOUT),
        ((8732,), 1, ttnn.ROW_MAJOR_LAYOUT),
        ((8732,), -1, ttnn.ROW_MAJOR_LAYOUT),
        ((8732,), 0, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_unsqueeze(device, input_shape, dim, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_unsqueeze_tensor = torch.unsqueeze(torch_input_tensor, dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    ttnn_output = ttnn.unsqueeze(input_tensor, dim)
    torch_output_tensor = ttnn.to_torch(ttnn_output)
    assert torch.allclose(torch_output_tensor, torch_unsqueeze_tensor)


@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [
        ((1, 1, 253), 4, ttnn.ROW_MAJOR_LAYOUT),
        ((1, 1, 253), -5, ttnn.ROW_MAJOR_LAYOUT),
    ],
)
def test_invalid_cases(device, input_shape, dim, layout, expect_error):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    with expect_error(RuntimeError, r"Dimension out of range"):
        ttnn.unsqueeze(input_tensor, dim)


# Caption of the ttnn.unsqueeze docstring table of dtypes and layouts that need the new dim before the last two.
_VIEW_CAPTION = "Also supported when the new dimension comes before the last two dimensions of the output"

# For a rank-3 input: dims that put the new dim before the last two of the output, and dims that make it one of them.
_VIEW_DIMS = (0, 1, -4, -3)
_LAST_TWO_DIMS = (2, 3, -2, -1)


def _docstring_shard_layouts(op):
    """One ``pytest.param(memory_layout)`` per shard layout the docstring's ``- Sharded:`` Memory Support item names
    before " in L1". Any other name there fails the test."""
    try:
        line = next(line.strip() for line in op.__doc__.splitlines() if line.strip().startswith("- Sharded:"))
        names = [name.strip() for name in re.split(r",| and ", line[len("- Sharded:") :].split(" in L1", 1)[0])]
        unknown = [name for name in names if name not in ("HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED")]
        assert not unknown, f"not a shard layout: {unknown} on {line!r}"
        return [pytest.param(getattr(ttnn.TensorMemoryLayout, name), id=name) for name in names]
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the Sharded Memory Support item from the {op} docstring: {error}")
        return [pytest.param(reason, id="docstring-unreadable")]


def _assert_unsqueeze_like_torch(input_tensor, dim):
    """Unsqueezes ``input_tensor`` and checks the output against ``torch.unsqueeze`` and the input's dtype, layout and
    memory configuration."""
    output = ttnn.unsqueeze(input_tensor, dim)

    assert output.dtype == input_tensor.dtype
    assert output.layout == input_tensor.layout
    assert output.memory_config().memory_layout == input_tensor.memory_config().memory_layout
    assert output.memory_config().buffer_type == input_tensor.memory_config().buffer_type
    assert output.memory_config().shard_spec == input_tensor.memory_config().shard_spec
    assert_exact(torch.unsqueeze(ttnn.to_torch(input_tensor), dim), ttnn.to_torch(output))
    return output


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.unsqueeze))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.unsqueeze))
def test_unsqueeze_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.unsqueeze docstring lists, inserting the new dim
    at each position, with positive and negative ``dim``."""
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    x = full_range_input((3, 50, 64), dtype)
    input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    for dim in _VIEW_DIMS:
        output = _assert_unsqueeze_like_torch(input_tensor, dim)
        assert output.buffer_address() == input_tensor.buffer_address()

    # A new last dim, or a new second-to-last dim of a TILE input, moves data. Each block-float value then stays in
    # its block of 16, or gets a block of its own, so block floats keep their values.
    for dim in _LAST_TWO_DIMS:
        _assert_unsqueeze_like_torch(input_tensor, dim)


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.unsqueeze))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.unsqueeze, caption=_VIEW_CAPTION))
def test_unsqueeze_docstring_view_inputs(dtype, layout, buffer_type, device):
    """Runs the dtypes and layouts the ttnn.unsqueeze docstring supports when the new dim comes before the last two
    of the output, on each interleaved buffer type it lists."""
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    x = full_range_input((3, 50, 64), dtype)
    input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    for dim in _VIEW_DIMS:
        output = _assert_unsqueeze_like_torch(input_tensor, dim)
        assert output.buffer_address() == input_tensor.buffer_address()


_SHARDED_INPUTS = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: ((2, 64, 32), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.HEIGHT),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: ((2, 32, 128), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: ((2, 32, 64), ttnn.CoreGrid(y=2, x=2), ttnn.ShardStrategy.BLOCK),
}


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_layout", _docstring_shard_layouts(ttnn.unsqueeze))
def test_unsqueeze_docstring_sharded_inputs(memory_layout, layout, device):
    """An L1 input of each shard layout the ttnn.unsqueeze docstring names gives a view with the input's memory
    configuration when the new dim comes before the last two of the output."""
    check_readable(memory_layout)
    shape, core_grid, strategy = _SHARDED_INPUTS[memory_layout]
    sharded = ttnn.create_sharded_memory_config(shape, core_grid, strategy)
    x = full_range_input(shape, ttnn.bfloat16)
    input_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=sharded)
    for dim in _VIEW_DIMS:
        output = _assert_unsqueeze_like_torch(input_tensor, dim)
        assert output.buffer_address() == input_tensor.buffer_address()

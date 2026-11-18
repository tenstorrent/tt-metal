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
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dim",
    [
        ((), 0),
        ((), -1),
        ((1), 0),
        ((19), 0),
        ((1, 19), 0),
        ((1, 32, 16), 0),
        ((1, 1, 24576), 0),
        ((1, 1, 1, 30), 2),
        ((1, 1, 1, 256), 2),
        ((1, 1, 1, 30), -1),
        ((1, 1, 1, 256), -1),
        ((1, 1, 480, 640), 1),
        ((3, 50, 1, 1, 768), -2),
        ((3, 197, 1, 1, 768), -2),
        ((3, 50, 1, 1, 1024), -2),
        ((3, 197, 1, 1, 1024), -2),
        ((3, 1370, 1, 1, 1280), -2),
    ],
)
def test_squeeze(device, input_shape, dim, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor, dim)
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    tt_squeeze_tensor = ttnn.squeeze(tt_input_tensor, dim)
    ttnn_squeeze_tensor = ttnn.to_torch(tt_squeeze_tensor)
    assert ttnn_squeeze_tensor.shape == torch_squeeze_tensor.shape
    assert torch.allclose(ttnn_squeeze_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape",
    [
        (),
        (1,),
        (1, 2),
        (1, 32, 16),
        (1, 1, 1, 256),
        (1, 1, 480, 640),
        (3, 1, 1, 1, 1280),
    ],
)
def test_squeeze_default(device, input_shape, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor)
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    ttnn_squeeze_tensor = ttnn.squeeze(tt_input_tensor)
    ttnn_squeeze_tensor = ttnn.to_torch(ttnn_squeeze_tensor)
    assert ttnn_squeeze_tensor.shape == torch_squeeze_tensor.shape
    assert torch.allclose(ttnn_squeeze_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dims",
    [
        ((), []),
        ((), [-1]),
        ((1,), [0]),
        ((1, 2), [0]),
        ((1, 1, 1, 256), []),
        ((1, 1, 1, 256), [0, 1]),
        ((1, 1, 1, 256), [-4, -3]),
        ((1, 1, 480, 640), [0, 1]),
        ((3, 1, 1, 1, 1280), [1, 2, 3]),
    ],
)
def test_squeeze_multiple_dims(device, input_shape, dims, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_squeeze_tensor = torch.squeeze(torch_input_tensor, list(dims))
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    tt_squeeze_tensor = ttnn.squeeze(tt_input_tensor, list(dims))
    tt_squeeze_tensor = ttnn.to_torch(tt_squeeze_tensor)
    assert tt_squeeze_tensor.shape == torch_squeeze_tensor.shape
    assert torch.allclose(tt_squeeze_tensor, torch_squeeze_tensor)


@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dims, expected_exception, expected_message",
    [
        ((1, 1, 1, 256), [4], RuntimeError, "Dimension out of range"),  # Out of range positive index
        ((1, 1, 1, 256), [-5], RuntimeError, "Dimension out of range"),  # Out of range negative index
        ((1, 1, 1, 256), [0, 0], RuntimeError, "appears multiple times"),  # Duplicate indices
        (
            (1, 1, 1, 256),
            [0, -4],
            RuntimeError,
            "appears multiple times",
        ),  # Duplicate indices (positive and negative)
    ],
)
def test_squeeze_error_cases(device, input_shape, dims, expected_exception, expected_message, layout, expect_error):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)
    with expect_error(expected_exception, expected_message):
        ttnn.squeeze(input_tensor, dims)


# This test verifies that padded_shape of tensor after squeezing is properly updated
@pytest.mark.parametrize(
    "layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.TILE_LAYOUT),
    ],
)
@pytest.mark.parametrize(
    "input_shape, dims",
    [
        ((1, 1, 1, 256), [0, 1]),
        ((1, 1, 480, 640), [0]),
        ((3, 1, 1, 1, 1280), [1, 2, 3]),
    ],
)
def test_squeeze_padded_shape_integrity(device, input_shape, dims, layout):
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    tt_input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, device=device)

    tt_squeeze_tensor = ttnn.squeeze(tt_input_tensor, dims)
    new_tensor = ttnn.empty(tt_squeeze_tensor.shape, layout=layout, device=device, dtype=ttnn.bfloat16)
    assert tt_squeeze_tensor.padded_shape == new_tensor.padded_shape


_BLOCK_FLOATS = (ttnn.bfloat8_b, ttnn.bfloat4_b)

# Caption of the ttnn.squeeze docstring table of dtypes and layouts that need every removed dim before the last two.
_VIEW_CAPTION = "Also supported when every removed dimension comes before the last two"

# Each form of `dim` for an input of shape (1, 3, 1, 50, 64): the dim of size 3 is kept, negative dims count from the
# last, and none of these removes one of the last two dims.
_VIEW_DIMS = (None, 0, -3, 1, [0, 2], [-5, 1, -3])


def _one_exponent_input(shape, seed=0):
    """Values in +-[1, 2) for a block float. They share one exponent, so values an op moves into another block of the
    tile come back exactly."""
    torch.manual_seed(seed)
    magnitude = 1 + torch.rand(shape)
    return torch.where(torch.rand(shape) < 0.5, -magnitude, magnitude)


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


def _assert_squeeze_like_torch(input_tensor, dim):
    """Squeezes ``input_tensor`` and checks the output against ``torch.squeeze`` and the input's dtype, layout and
    memory configuration."""
    output = ttnn.squeeze(input_tensor, dim)
    expected = ttnn.to_torch(input_tensor)
    expected = torch.squeeze(expected) if dim is None else torch.squeeze(expected, dim)

    assert output.dtype == input_tensor.dtype
    assert output.layout == input_tensor.layout
    assert output.memory_config().memory_layout == input_tensor.memory_config().memory_layout
    assert output.memory_config().buffer_type == input_tensor.memory_config().buffer_type
    assert output.memory_config().shard_spec == input_tensor.memory_config().shard_spec
    assert_exact(expected, ttnn.to_torch(output))
    return output


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.squeeze))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.squeeze))
def test_squeeze_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.squeeze docstring lists, with each form of
    ``dim``, and removes the last dim and the second-to-last dim."""
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    x = full_range_input((1, 3, 1, 50, 64), dtype)
    input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    for dim in _VIEW_DIMS:
        output = _assert_squeeze_like_torch(input_tensor, dim)
        assert output.buffer_address() == input_tensor.buffer_address()

    # Removing the last dim, or the second-to-last dim of a TILE input, moves data. Removing the last dim regroups
    # block-float values, so block floats get values with one exponent.
    for shape, dim in (((2, 50, 1), -1), ((2, 1, 64), -2)):
        x = _one_exponent_input(shape) if dtype in _BLOCK_FLOATS else full_range_input(shape, dtype)
        input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
        _assert_squeeze_like_torch(input_tensor, dim)


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.squeeze))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.squeeze, caption=_VIEW_CAPTION))
def test_squeeze_docstring_view_inputs(dtype, layout, buffer_type, device):
    """Runs the dtypes and layouts the ttnn.squeeze docstring supports when every removed dim comes before the last
    two, on each interleaved buffer type it lists, with each form of ``dim``."""
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    x = full_range_input((1, 3, 1, 50, 64), dtype)
    input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    for dim in _VIEW_DIMS:
        output = _assert_squeeze_like_torch(input_tensor, dim)
        assert output.buffer_address() == input_tensor.buffer_address()


_SHARDED_INPUTS = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: ((1, 2, 1, 64, 32), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.HEIGHT),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: ((1, 2, 1, 32, 128), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: ((1, 2, 1, 32, 64), ttnn.CoreGrid(y=2, x=2), ttnn.ShardStrategy.BLOCK),
}


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_layout", _docstring_shard_layouts(ttnn.squeeze))
def test_squeeze_docstring_sharded_inputs(memory_layout, layout, device):
    """An L1 input of each shard layout the ttnn.squeeze docstring names gives a view with the input's memory
    configuration when every removed dim comes before the last two."""
    check_readable(memory_layout)
    shape, core_grid, strategy = _SHARDED_INPUTS[memory_layout]
    sharded = ttnn.create_sharded_memory_config(shape, core_grid, strategy)
    input_tensor = ttnn.from_torch(
        full_range_input(shape, ttnn.bfloat16), dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=sharded
    )
    for dim in (None, 0, [0, -3]):
        output = _assert_squeeze_like_torch(input_tensor, dim)
        assert output.buffer_address() == input_tensor.buffer_address()

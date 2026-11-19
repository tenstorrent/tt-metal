# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tt_py_test_utils_common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    check_readable,
    docstring_blackhole_dtype_layout_params,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input


# Reshape in Tile layout with shapes that are not divisible by 32
@pytest.mark.parametrize(
    "input_shape, output_shape, layout",
    [
        ((1, 15), (15,), ttnn.ROW_MAJOR_LAYOUT),  # RM_last dimension matches, 1D output
        ((2, 1, 1, 1, 15), (2, 15), ttnn.ROW_MAJOR_LAYOUT),  # RM_last dimension matches
        ((16, 1, 1, 247, 13), (1, 16, 247, 13), ttnn.TILE_LAYOUT),  # last two dimensions match
        (
            (16, 1, 1, 256, 16),
            (8, 16, 32, 16),
            ttnn.TILE_LAYOUT,
        ),  # last dimension match but second last multiple of 32 but does not match
        ((32, 32, 32, 15), (32768, 15), ttnn.TILE_LAYOUT),  # Very large tensor
    ],
)
def test_view(input_shape, output_shape, layout, device):
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_result = torch_input_tensor.reshape(output_shape)

    input_tensor = ttnn.from_torch(torch_input_tensor, layout=layout, dtype=ttnn.bfloat16, device=device)
    ttnn_output = ttnn.view(input_tensor, output_shape)
    assert layout == ttnn_output.layout
    output = ttnn.to_torch(ttnn_output)
    assert_equal(torch_result, output)


@pytest.mark.parametrize(
    "input_shape, output_shape, layout, expected_message",
    [
        (
            (2, 1, 1, 1, 15),
            (1, 30),
            ttnn.ROW_MAJOR_LAYOUT,
            "The last dimension can not change in view",
        ),  # RM last dimension doesn't match
        (
            (16, 1, 256, 1, 16),
            (8, 16, 32, 16),
            ttnn.TILE_LAYOUT,
            "Invalid second last dims",
        ),  # TILE last dimension match but second last does not match, shape mult of 32 only
        (
            (16, 1, 1, 256, 16),
            (8, 16, 32, 1, 16),
            ttnn.TILE_LAYOUT,
            "Invalid second last dims",
        ),  # TILE last dimension match but second last does not match, tensor mult of 32 only
        (
            (256, 1, 1, 16, 16),
            (8, 16, 32, 1, 16),
            ttnn.TILE_LAYOUT,
            "Invalid second last dims",
        ),  # TILE last dimension match but second last does not match, none mult of 32
        (
            (16, 8, 1, 32, 16),
            (8, 16, 31, 16),
            ttnn.TILE_LAYOUT,
            "Invalid arguments to reshape",
        ),  # Volume doesn't match but padded volume does
    ],
)
def test_invalid_cases(input_shape, output_shape, layout, expected_message, device, expect_error):
    # Verifies invalid cases do cause an assertion
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input_tensor, dtype=ttnn.bfloat16, layout=layout, device=device)
    with expect_error(RuntimeError, expected_message):
        ttnn.view(input_tensor, output_shape)


def _assert_is_view(input_tensor, shape):
    """Views ``input_tensor`` as ``shape``. The output must have ``shape``, the input's dtype and layout, the input's
    memory layout, buffer type, shard spec and buffer, and the input's values."""
    expected = ttnn.to_torch(input_tensor).reshape(list(shape))
    output = ttnn.view(input_tensor, shape)
    assert list(output.shape) == list(expected.shape)
    assert output.dtype == input_tensor.dtype
    assert output.layout == input_tensor.layout
    assert output.memory_config().memory_layout == input_tensor.memory_config().memory_layout
    assert output.memory_config().buffer_type == input_tensor.memory_config().buffer_type
    assert output.memory_config().shard_spec == input_tensor.memory_config().shard_spec
    assert output.buffer_address() == input_tensor.buffer_address()
    assert_exact(expected, ttnn.to_torch(output))


# (input shape, new shape, layouts): the last two dimensions stay the same; the second-to-last goes from 64 to 384,
# multiples of 32 in both, with -1 inferred; and, for ROW_MAJOR only, the second-to-last goes from 50 to 300 and the
# rank goes down to 1.
_DOCSTRING_VIEWS = [
    ((2, 3, 50, 70), (6, 1, 50, 70), (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)),
    ((2, 3, 64, 70), (-1, 70), (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT)),
    ((2, 3, 50, 70), (300, 70), (ttnn.ROW_MAJOR_LAYOUT,)),
    ((1, 100), (100,), (ttnn.ROW_MAJOR_LAYOUT,)),
]


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.view))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.view))
def test_view_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.view docstring lists.

    Each view must keep the input's dtype, layout, memory configuration and buffer, and hold the input's values in the
    same row-major order. A new shape without -1 runs both as a tuple and as a ttnn.Shape.
    """
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    for input_shape, shape, layouts in _DOCSTRING_VIEWS:
        if layout not in layouts:
            continue
        input_tensor = ttnn.from_torch(
            full_range_input(input_shape, dtype), dtype=dtype, layout=layout, device=device, memory_config=memory_config
        )
        _assert_is_view(input_tensor, shape)
        if -1 not in shape:
            _assert_is_view(input_tensor, ttnn.Shape(list(shape)))


def _docstring_sharded_params(op):
    """One ``pytest.param(memory_layout, buffer_type)`` per pair on the docstring's ``- Sharded:`` Memory Support item.

    The item must start ``- Sharded: <memory layouts> in <buffer types>``, with the names joined by commas, ``and`` or
    ``or``. A name that is not a sharded ``ttnn.TensorMemoryLayout`` or a ``ttnn.BufferType`` fails the test.
    """
    names = r"\w+(?:(?:, | and | or )\w+)*"
    try:
        ((layout_names, buffer_names),) = re.findall(
            rf"^\s*- Sharded: ({names}) in ({names})(?:[;.]|$)", op.__doc__, re.MULTILINE
        )
        params = []
        for layout_name in re.split(", | and | or ", layout_names):
            memory_layout = getattr(ttnn.TensorMemoryLayout, layout_name, None)
            assert isinstance(memory_layout, ttnn.TensorMemoryLayout) and layout_name.endswith(
                "_SHARDED"
            ), f"not a sharded ttnn.TensorMemoryLayout: {layout_name}"
            for buffer_name in re.split(", | and | or ", buffer_names):
                buffer_type = getattr(ttnn.BufferType, buffer_name, None)
                assert isinstance(buffer_type, ttnn.BufferType), f"not a ttnn.BufferType: {buffer_name}"
                params.append(pytest.param(memory_layout, buffer_type, id=f"{layout_name}-{buffer_name}"))
        return params
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the Sharded Memory Support item from the {op} docstring: {error}")
        return [pytest.param(reason, reason, id="docstring-unreadable")]


# An ND shard shape for each memory layout that gives a (2, 2, 64, 64) tensor the shard spec of _sharded_memory_config.
_ND_SHARD_SHAPES = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: [1, 2, 64, 64],
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: [2, 2, 64, 32],
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: [1, 2, 64, 32],
}


def _sharded_memory_config(memory_layout, buffer_type):
    """Shards a tensor of 256 rows of 64 values over two cores, or over 2x2 cores for BLOCK_SHARDED."""
    if memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        end, shard_shape = (1, 0), (128, 64)
    elif memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        end, shard_shape = (1, 0), (256, 32)
    else:
        end, shard_shape = (1, 1), (128, 32)
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(*end))})
    shard_spec = ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(memory_layout, buffer_type, shard_spec)


@pytest.mark.parametrize("memory_layout, buffer_type", _docstring_sharded_params(ttnn.view))
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT], ids=["TILE", "ROW_MAJOR"])
def test_view_docstring_sharded_inputs(memory_layout, buffer_type, layout, device):
    """Views a sharded input in every sharded placement the ttnn.view docstring lists, to rank 3 and rank 2 shapes.
    The input's memory configuration is made from a ShardSpec, and from an NdShardSpec with the same shards."""
    check_readable(memory_layout, buffer_type)
    memory_config = _sharded_memory_config(memory_layout, buffer_type)
    nd_shard_spec = ttnn.NdShardSpec(ttnn.Shape(_ND_SHARD_SHAPES[memory_layout]), memory_config.shard_spec.grid)
    for input_memory_config in (memory_config, ttnn.MemoryConfig(buffer_type, nd_shard_spec)):
        input_tensor = ttnn.from_torch(
            full_range_input((2, 2, 64, 64), ttnn.bfloat16),
            dtype=ttnn.bfloat16,
            layout=layout,
            device=device,
            memory_config=input_memory_config,
        )
        assert input_tensor.memory_config().memory_layout == memory_layout
        assert input_tensor.memory_config().shard_spec == memory_config.shard_spec
        for shape in [(4, 64, 64), (256, 64)]:
            _assert_is_view(input_tensor, shape)


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.view))
@pytest.mark.parametrize("dtype, layout", docstring_blackhole_dtype_layout_params(ttnn.view))
def test_view_docstring_blackhole_inputs(dtype, layout, buffer_type, device):
    """Views a tensor of each dtype the ttnn.view docstring lists for Blackhole, in every interleaved buffer type. The
    values are compared after a conversion to FLOAT32 on host, which holds every FP8_E4M3 value."""
    check_readable(dtype, layout, buffer_type)
    if not is_blackhole():
        pytest.skip("the docstring lists these dtypes for Blackhole")
    torch.manual_seed(0)
    input_tensor = ttnn.from_torch(
        torch.rand((2, 3, 50, 70)) * 2 - 1,
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=interleaved_memory_config(buffer_type),
    )
    output = ttnn.view(input_tensor, (300, 70))

    assert list(output.shape) == [300, 70]
    assert output.dtype == dtype
    assert output.layout == layout
    assert output.memory_config().memory_layout == input_tensor.memory_config().memory_layout
    assert output.memory_config().buffer_type == input_tensor.memory_config().buffer_type
    assert output.buffer_address() == input_tensor.buffer_address()
    expected = ttnn.to_torch(ttnn.to_dtype(input_tensor.cpu(), ttnn.float32)).reshape(300, 70)
    assert_exact(expected, ttnn.to_torch(ttnn.to_dtype(output.cpu(), ttnn.float32)))

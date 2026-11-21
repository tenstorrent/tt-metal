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

# Every pair of dims of a rank-4 tensor.
_DIM_PAIRS = [pytest.param((a, b), id=f"{a}-{b}") for a, b in [(2, 3), (1, 2), (0, 1), (0, 2), (0, 3), (1, 3)]]

_BLOCK_FLOATS = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def _one_exponent_input(shape):
    """Values in +-[1, 2). They share one exponent, so block-float values that move into another block come back
    exactly."""
    torch.manual_seed(0)
    return (1 + torch.rand(shape)) * (2 * torch.randint(0, 2, shape) - 1)


def _docstring_pad_value_dtypes():
    """Dtypes whose exposed TILE padding the ``pad_value`` line of the ttnn.transpose docstring says it fills."""
    try:
        line = next(line for line in ttnn.transpose.__doc__.splitlines() if line.strip().startswith("pad_value ("))
        match = re.search(r"fills the tile padding that the transpose exposes on an? (.+?) TILE output", line)
        assert match, f"no 'fills the tile padding ... TILE output' phrase on: {line.strip()!r}"
        names = [name.strip() for name in re.split(r",|\bor\b", match.group(1))]
        assert all(re.fullmatch(r"``\w+``", name) for name in names), f"not a list of dtypes: {match.group(1)!r}"
        return frozenset(getattr(ttnn.DataType, name.strip("`")) for name in names)
    except Exception as error:
        return UnreadableDocstring(f"could not read the pad_value dtypes from the ttnn.transpose docstring: {error}")


def _names(text):
    """The names in a list such as ``A, B and C``; each must be non-empty."""
    names = [name.strip() for name in re.split(r",|\band\b", text)]
    assert all(names), f"empty name in {text!r}"
    return names


_SHARD_LAYOUTS = ("HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED")


def _docstring_sharded_params():
    """One ``pytest.param(dtype, layout, memory_layout)`` per dtype and shard layout on the docstring's Sharded item,
    with each layout the dtype/layout table lists for that dtype."""
    try:
        memory_support = ttnn.transpose.__doc__.split("Memory Support:", 1)[1]
        items = [line.strip() for line in memory_support.splitlines() if line.strip().startswith("- Sharded:")]
        assert len(items) == 1, f"expected one '- Sharded:' item, got {items}"
        match = re.fullmatch(r"- Sharded: (.+) input and output in L1, for (.+) tensors", items[0])
        assert match, f"unreadable Sharded item: {items[0]!r}"
        layout_names = _names(match.group(1))
        unknown = [name for name in layout_names if name not in _SHARD_LAYOUTS]
        assert not unknown, f"not a shard layout: {unknown}"
        dtype_names = _names(match.group(2))
        assert all(re.fullmatch(r"``\w+``", name) for name in dtype_names), f"not a list of dtypes: {match.group(2)!r}"
        dtypes = {getattr(ttnn.DataType, name.strip("`")) for name in dtype_names}
        table = docstring_dtype_layout_params(ttnn.transpose)
        params = [
            pytest.param(*row.values, getattr(ttnn.TensorMemoryLayout, name), id=f"{row.id}-{name}")
            for row in table
            if row.values[0] in dtypes
            for name in layout_names
        ]
        listed = {param.values[0] for param in params}
        assert listed == dtypes, f"sharded dtypes missing from the dtype/layout table: {dtypes - listed}"
        return params
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the Sharded item from the ttnn.transpose docstring: {error}")
        return [pytest.param(reason, reason, reason, id="docstring-unreadable")]


@pytest.mark.parametrize("dims", _DIM_PAIRS)
@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.transpose))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.transpose))
def test_transpose_docstring_supported_inputs(dtype, layout, buffer_type, dims, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.transpose docstring lists, swapping each pair of
    dims of a rank-4 tensor.

    Each case swaps the dims with the default ``memory_config``, and again with negative dims in reverse order and a
    given ``memory_config``. When exactly one of the dims is among the last two, it also checks ``pad_value``: the tile
    padding of a TILE output holds it for the dtypes the docstring's ``pad_value`` line names, and a ROW_MAJOR output
    has no padding.
    """
    check_readable(dtype, layout, buffer_type)
    pad_value_dtypes = _docstring_pad_value_dtypes()
    check_readable(pad_value_dtypes)
    memory_config = interleaved_memory_config(buffer_type)
    shape = (2, 3, 32, 64)
    # A swap with the last dim runs a compute kernel that holds BFLOAT16 zeros by value: a BFLOAT16 input leaves out
    # NaN, as the docstring requires, and its zeros are compared by value. Values also move into new blocks, so block
    # floats get values with one exponent.
    last_dim_bfloat16 = 3 in dims and dtype == ttnn.bfloat16
    if 3 in dims and dtype in _BLOCK_FLOATS:
        x = _one_exponent_input(shape)
    else:
        x = full_range_input(shape, dtype, nan=not last_dim_bfloat16)
    input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    expected = ttnn.to_torch(input_tensor).transpose(*dims)

    output = ttnn.transpose(input_tensor, *dims)
    assert output.dtype == dtype
    assert output.layout == layout
    assert output.memory_config() == memory_config
    assert_exact(expected, ttnn.to_torch(output), signed_zeros=not last_dim_bfloat16)

    # A given memory_config sets the output's: here the other interleaved buffer type.
    other_memory_config = interleaved_memory_config(
        ttnn.BufferType.L1 if buffer_type == ttnn.BufferType.DRAM else ttnn.BufferType.DRAM
    )
    output = ttnn.transpose(input_tensor, dims[1] - 4, dims[0] - 4, memory_config=other_memory_config)
    assert output.memory_config() == other_memory_config
    assert_exact(expected, ttnn.to_torch(output), signed_zeros=not last_dim_bfloat16)

    if (dims[0] >= 2) != (dims[1] >= 2):
        # The swapped dim of size 2 or 3 lands in the last two dims, so a TILE output has tile padding to fill.
        output = ttnn.transpose(input_tensor, *dims, memory_config=memory_config, pad_value=7.0)
        assert_exact(expected, ttnn.to_torch(output), signed_zeros=not last_dim_bfloat16)
        if layout == ttnn.ROW_MAJOR_LAYOUT:
            assert list(output.padded_shape) == list(output.shape)
        elif dtype in pad_value_dtypes:
            height, width = list(output.shape)[-2:]
            padded = ttnn.from_device(output).to_torch_with_padded_shape()
            assert torch.all(padded[..., height:, :] == 7)
            assert torch.all(padded[..., :, width:] == 7)


# Input shape per pair of dims. The flattened height and the width of the input and of the output are multiples of 64,
# so each splits into tile-aligned shards on the grids below.
_SHARDED_SHAPES = {
    (2, 3): (1, 2, 64, 128),
    (1, 2): (1, 32, 64, 64),
    (0, 1): (2, 4, 32, 64),
    (0, 2): (32, 1, 64, 64),
    (0, 3): (64, 1, 32, 64),
    (1, 3): (1, 64, 32, 64),
}

_SHARD_GRIDS = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: (ttnn.CoreGrid(y=1, x=2), ttnn.ShardStrategy.HEIGHT),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: (ttnn.CoreGrid(y=1, x=2), ttnn.ShardStrategy.WIDTH),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: (ttnn.CoreGrid(y=2, x=2), ttnn.ShardStrategy.BLOCK),
}


@pytest.mark.parametrize("dims", _DIM_PAIRS)
@pytest.mark.parametrize("sharded", ["input", "output"])
@pytest.mark.parametrize("dtype, layout, memory_layout", _docstring_sharded_params())
def test_transpose_docstring_sharded_inputs(dtype, layout, memory_layout, sharded, dims, device):
    """Runs each dtype and shard layout the Sharded item of the ttnn.transpose docstring lists, swapping each pair of
    dims of a rank-4 tensor: an L1 sharded input with the default ``memory_config`` gives an L1 output, and an
    interleaved input with a sharded ``memory_config`` gives an output with that memory configuration."""
    check_readable(dtype, layout, memory_layout)
    shape = _SHARDED_SHAPES[dims]
    output_shape = list(shape)
    output_shape[dims[0]], output_shape[dims[1]] = shape[dims[1]], shape[dims[0]]
    core_grid, strategy = _SHARD_GRIDS[memory_layout]
    # A swap with the last dim holds BFLOAT16 zeros by value and needs input values that are not NaN.
    last_dim_bfloat16 = 3 in dims and dtype == ttnn.bfloat16
    x = full_range_input(shape, dtype, nan=not last_dim_bfloat16)

    if sharded == "input":
        input_config = ttnn.create_sharded_memory_config(shape, core_grid, strategy, ttnn.ShardOrientation.ROW_MAJOR)
        memory_config = None
    else:
        input_config = ttnn.DRAM_MEMORY_CONFIG
        memory_config = ttnn.create_sharded_memory_config(
            output_shape, core_grid, strategy, ttnn.ShardOrientation.ROW_MAJOR
        )
    input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=input_config)
    output = ttnn.transpose(input_tensor, *dims, memory_config=memory_config)

    if sharded == "input":
        assert input_tensor.memory_config().memory_layout == memory_layout
        assert output.memory_config().buffer_type == ttnn.BufferType.L1
    else:
        assert output.memory_config() == memory_config
    assert output.dtype == dtype
    assert output.layout == layout
    expected = ttnn.to_torch(input_tensor).transpose(*dims)
    assert_exact(expected, ttnn.to_torch(output), signed_zeros=not last_dim_bfloat16)

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


def _assert_unsqueeze_to_4D_view(input_tensor, read=ttnn.to_torch):
    """Checks that ttnn.unsqueeze_to_4D gives a view of ``input_tensor`` with its dtype, layout and memory
    configuration, holding its values, read before the call, with 1s added in front of its shape up to rank 4.
    ``read`` brings a tensor's values to host."""
    expected = read(input_tensor)
    expected = expected.reshape((1,) * (4 - expected.dim()) + tuple(expected.shape))
    output = ttnn.unsqueeze_to_4D(input_tensor)

    assert output.dtype == input_tensor.dtype
    assert output.layout == input_tensor.layout
    assert output.memory_config().memory_layout == input_tensor.memory_config().memory_layout
    assert output.memory_config().buffer_type == input_tensor.memory_config().buffer_type
    assert output.memory_config().shard_spec == input_tensor.memory_config().shard_spec
    assert output.buffer_address() == input_tensor.buffer_address()
    assert_exact(expected, read(output))


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.unsqueeze_to_4D))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.unsqueeze_to_4D))
def test_unsqueeze_to_4D_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.unsqueeze_to_4D docstring lists, with inputs of
    rank 0 to 4."""
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    # The rank-0 input holds the first value of a full-range input: +inf for floats, the smallest value for integers.
    inputs = [full_range_input((64,), dtype)[:1].reshape(())]
    inputs += [full_range_input(shape, dtype) for shape in ((64,), (50, 64), (3, 50, 64), (2, 3, 50, 64))]
    for x in inputs:
        input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
        _assert_unsqueeze_to_4D_view(input_tensor)


_SHARDED_INPUTS = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: ((2, 64, 32), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.HEIGHT),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: ((2, 32, 128), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: ((2, 32, 64), ttnn.CoreGrid(y=2, x=2), ttnn.ShardStrategy.BLOCK),
}


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
@pytest.mark.parametrize("memory_layout", _docstring_shard_layouts(ttnn.unsqueeze_to_4D))
def test_unsqueeze_to_4D_docstring_sharded_inputs(memory_layout, layout, device):
    """An L1 input of each shard layout the ttnn.unsqueeze_to_4D docstring names gives a view with the input's memory
    configuration."""
    check_readable(memory_layout)
    shape, core_grid, strategy = _SHARDED_INPUTS[memory_layout]
    sharded = ttnn.create_sharded_memory_config(shape, core_grid, strategy)
    x = full_range_input(shape, ttnn.bfloat16)
    input_tensor = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=sharded)
    _assert_unsqueeze_to_4D_view(input_tensor)


def test_unsqueeze_to_4D_rank_above_4(device, expect_error):
    x = ttnn.from_torch(torch.zeros((1, 1, 1, 32, 32), dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with expect_error(RuntimeError, "Tensor rank is greater than 4"):
        ttnn.unsqueeze_to_4D(x)

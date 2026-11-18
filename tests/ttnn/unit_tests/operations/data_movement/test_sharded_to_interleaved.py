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


def _docstring_output_dtype_pairs(op):
    """One ``pytest.param(input_dtype, output_dtype)`` per ordered pair of the dtypes that the op's ``output_dtype``
    line lists in "input of <dtypes> can be converted". Each name must be a ``ttnn.DataType``."""
    try:
        line = next((line.strip() for line in op.__doc__.splitlines() if line.strip().startswith("output_dtype (")), "")
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
        reason = UnreadableDocstring(f"could not read the output_dtype line from the {op} docstring: {error}")
        return [pytest.param(reason, reason, id="docstring-unreadable")]


def _convertible_input(shape, input_dtype, output_dtype):
    """Values that both dtypes hold exactly, and no NaN, as the output_dtype line asks. With BFLOAT8_B, the values are
    +-(1 + k / 64): they share one exponent and fit the 6 fraction bits a BFLOAT8_B value with its block's exponent
    has."""
    if ttnn.bfloat8_b not in (input_dtype, output_dtype):
        return full_range_input(shape, ttnn.bfloat16, nan=False).float()
    torch.manual_seed(0)
    magnitude = 1 + torch.randint(0, 64, shape) / 64
    return torch.where(torch.randint(0, 2, shape, dtype=torch.bool), -magnitude, magnitude)


# _S2I_SHAPE has 2 * 2 * 64 = 256 rows of width 128. Shard shape and core grid size (x, y) of each shard layout.
_S2I_SHAPE = (2, 2, 64, 128)
_S2I_SHARDS = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: ((64, 128), (4, 1)),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: ((256, 64), (2, 1)),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: ((128, 64), (2, 2)),
}


def _s2i_sharded_memory_config(memory_layout, buffer_type):
    shard_shape, (grid_x, grid_y) = _S2I_SHARDS[memory_layout]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        memory_layout, buffer_type, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )


@pytest.mark.parametrize(
    "output_buffer_type", docstring_interleaved_buffer_type_params(ttnn.sharded_to_interleaved, label="Output")
)
@pytest.mark.parametrize("memory_layout, buffer_type", _docstring_sharded_input_params(ttnn.sharded_to_interleaved))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.sharded_to_interleaved))
def test_sharded_to_interleaved_docstring_supported_inputs(
    dtype, layout, memory_layout, buffer_type, output_buffer_type, device
):
    """Copies a sharded input of every dtype, layout and shard layout the ttnn.sharded_to_interleaved docstring lists
    into every interleaved output it lists, and into the default DRAM output."""
    check_readable(dtype, layout, memory_layout, buffer_type, output_buffer_type)
    input_tensor = ttnn.from_torch(
        full_range_input(_S2I_SHAPE, dtype),
        dtype=dtype,
        layout=layout,
        device=device,
        memory_config=_s2i_sharded_memory_config(memory_layout, buffer_type),
    )
    memory_config = interleaved_memory_config(output_buffer_type)
    outputs = [ttnn.sharded_to_interleaved(input_tensor, memory_config)]
    if output_buffer_type == ttnn.BufferType.DRAM:
        outputs.append(ttnn.sharded_to_interleaved(input_tensor))

    expected = ttnn.to_torch(input_tensor)
    for output in outputs:
        assert output.dtype == dtype
        assert output.layout == layout
        assert output.memory_config() == memory_config
        assert_exact(expected, ttnn.to_torch(output))


@pytest.mark.parametrize(
    "output_buffer_type", docstring_interleaved_buffer_type_params(ttnn.sharded_to_interleaved, label="Output")
)
@pytest.mark.parametrize("input_dtype, output_dtype", _docstring_output_dtype_pairs(ttnn.sharded_to_interleaved))
def test_sharded_to_interleaved_docstring_output_dtype(input_dtype, output_dtype, output_buffer_type, device):
    """Converts a HEIGHT_SHARDED TILE input between every two dtypes the ttnn.sharded_to_interleaved output_dtype line
    names, into every interleaved output the docstring lists. The input values are exact in both dtypes, so the output
    keeps them. Zeros are compared by value."""
    check_readable(input_dtype, output_dtype, output_buffer_type)
    input_tensor = ttnn.from_torch(
        _convertible_input(_S2I_SHAPE, input_dtype, output_dtype),
        dtype=input_dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_s2i_sharded_memory_config(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1),
    )
    memory_config = interleaved_memory_config(output_buffer_type)
    output = ttnn.sharded_to_interleaved(input_tensor, memory_config, output_dtype)

    assert output.dtype == output_dtype
    assert output.layout == ttnn.TILE_LAYOUT
    assert output.memory_config() == memory_config
    assert_exact(ttnn.to_torch(input_tensor), ttnn.to_torch(output), signed_zeros=False)

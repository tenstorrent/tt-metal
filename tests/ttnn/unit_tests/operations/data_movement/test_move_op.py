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

# Neither of the last two dims is a multiple of 32, so TILE inputs carry tile padding.
INTERLEAVED_SHAPE = (2, 3, 50, 72)
# 128 x 128 when flattened to 2D.
SHARDED_SHAPE = (1, 2, 64, 128)
_SHARDED_INPUTS = {
    "HEIGHT_SHARDED": (ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.HEIGHT),  # 32 x 128 shards
    "WIDTH_SHARDED": (ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),  # 128 x 32 shards
    "BLOCK_SHARDED": (ttnn.CoreGrid(y=2, x=2), ttnn.ShardStrategy.BLOCK),  # 64 x 64 shards
}


def _to_device(values, dtype, layout, memory_config, device):
    """``values`` on device in ``memory_config``. A FP8_E4M3 tensor reaches a sharded memory configuration through an
    interleaved L1 tensor, since ``ttnn.from_torch`` doesn't shard that dtype."""
    if dtype == ttnn.fp8_e4m3 and memory_config.is_sharded():
        interleaved = ttnn.from_torch(
            values, dtype=dtype, layout=layout, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        tensor = ttnn.to_memory_config(interleaved, memory_config)
        ttnn.deallocate(interleaved)
        return tensor
    return ttnn.from_torch(values, dtype=dtype, layout=layout, device=device, memory_config=memory_config)


def _move_and_check(x, dtype, layout, memory_config, free, move_memory_config, device, read=ttnn.to_torch):
    """Creates ``x`` on device in ``memory_config``, moves it with ``move_memory_config`` and checks the output, and
    that the move deallocated the input.

    ``free`` is a tensor allocated just before the input and deallocated just before the move, so it leaves memory free
    next to the input: ``None`` leaves none, ``"part"`` (one 16-byte row) lets the output overlap the input, and
    ``"all"`` (a tensor like the input, holding other values) gives the output memory of its own.
    """
    dummy = None
    if free == "part":
        dummy = ttnn.from_torch(
            torch.ones((1, 8), dtype=torch.bfloat16),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, memory_config.buffer_type),
        )
    elif free == "all":
        other = torch.where(x.to(torch.float64) == 1, 2, 1).to(x.dtype)
        dummy = _to_device(other, dtype, layout, memory_config, device)
    input_tensor = _to_device(x, dtype, layout, memory_config, device)
    if dummy is not None:
        ttnn.deallocate(dummy)
    input_memory_config = input_tensor.memory_config()
    expected = read(input_tensor)

    output = ttnn.move(input_tensor, memory_config=move_memory_config)

    assert not input_tensor.is_allocated()
    assert output.dtype == dtype
    assert output.layout == layout
    if move_memory_config is None or move_memory_config.is_sharded():
        assert output.memory_config() == input_memory_config
    else:
        assert output.memory_config() == move_memory_config
    assert_exact(expected, read(output))


@pytest.mark.parametrize("output_buffer_type", docstring_interleaved_buffer_type_params(ttnn.move))
@pytest.mark.parametrize("input_buffer_type", docstring_interleaved_buffer_type_params(ttnn.move))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.move))
def test_move_docstring_supported_inputs(dtype, layout, input_buffer_type, output_buffer_type, device):
    """Moves an input of every dtype, layout and interleaved buffer type the ttnn.move docstring lists into every
    interleaved buffer type it lists, and with the default memory_config, with no memory, a 16-byte row, or an input's
    worth of memory freed next to the input."""
    check_readable(dtype, layout, input_buffer_type, output_buffer_type)
    x = full_range_input(INTERLEAVED_SHAPE, dtype)
    for free in (None, "part", "all"):
        for move_memory_config in (interleaved_memory_config(output_buffer_type), None):
            _move_and_check(
                x, dtype, layout, interleaved_memory_config(input_buffer_type), free, move_memory_config, device
            )


def _docstring_shard_layouts():
    """One ``pytest.param(name)`` per shard layout on the Sharded Memory Support item of the ttnn.move docstring."""
    try:
        line = next(line.strip() for line in ttnn.move.__doc__.splitlines() if line.strip().startswith("- Sharded:"))
        match = re.fullmatch(r"- Sharded: (.+?) in L1, with the output in the input's memory configuration", line)
        assert match, f"unexpected Sharded item: {line!r}"
        names = re.split(r", | and ", match.group(1))
        unknown = [name for name in names if name not in _SHARDED_INPUTS]
        assert not unknown, f"not a shard layout: {unknown} on the Sharded item: {line!r}"
        return [pytest.param(name, id=name) for name in names]
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the Sharded item from the ttnn.move docstring: {error}")
        return [pytest.param(reason, id="docstring-unreadable")]


def _sharded_memory_config(shard_layout):
    core_grid, strategy = _SHARDED_INPUTS[shard_layout]
    return ttnn.create_sharded_memory_config(list(SHARDED_SHAPE), core_grid, strategy, ttnn.ShardOrientation.ROW_MAJOR)


@pytest.mark.parametrize("shard_layout", _docstring_shard_layouts())
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.move))
def test_move_docstring_sharded_inputs(dtype, layout, shard_layout, device):
    """Moves an L1 input of every dtype and layout the ttnn.move docstring lists, in each shard layout its Sharded item
    names, with the input's memory_config and the default one, with no memory, a 16-byte row, or an input's worth of
    memory freed next to the input."""
    check_readable(dtype, layout, shard_layout)
    sharded = _sharded_memory_config(shard_layout)
    x = full_range_input(SHARDED_SHAPE, dtype)
    for free in (None, "part", "all"):
        for move_memory_config in (sharded, None):
            _move_and_check(x, dtype, layout, sharded, free, move_memory_config, device)


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.move))
def test_move_fp8_e4m3_row_major(buffer_type, device):
    """On Blackhole, moves a FP8_E4M3 ROW_MAJOR input, as the ttnn.move docstring allows, from each interleaved buffer
    type the docstring lists into the other, and an L1 HEIGHT_SHARDED one in its own memory configuration."""
    check_readable(buffer_type)
    if not ttnn.device.is_blackhole(device):
        pytest.skip("FP8_E4M3 tensors need Blackhole")

    def read(tensor):
        return ttnn.to_torch(ttnn.to_dtype(tensor.cpu(), ttnn.float32))

    torch.manual_seed(0)
    x = torch.randn(SHARDED_SHAPE, dtype=torch.float32)
    other_buffer_type = ttnn.BufferType.L1 if buffer_type == ttnn.BufferType.DRAM else ttnn.BufferType.DRAM
    for free in (None, "part", "all"):
        for move_memory_config in (interleaved_memory_config(other_buffer_type), None):
            _move_and_check(
                x,
                ttnn.fp8_e4m3,
                ttnn.ROW_MAJOR_LAYOUT,
                interleaved_memory_config(buffer_type),
                free,
                move_memory_config,
                device,
                read,
            )
    if buffer_type == ttnn.BufferType.L1:
        sharded = _sharded_memory_config("HEIGHT_SHARDED")
        for free in (None, "part", "all"):
            _move_and_check(x, ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT, sharded, free, sharded, device, read)

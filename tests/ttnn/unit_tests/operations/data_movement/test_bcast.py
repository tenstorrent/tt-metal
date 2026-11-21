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
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact

_TORCH_OPS = {
    ttnn.BcastOpMath.ADD: torch.add,
    ttnn.BcastOpMath.SUB: torch.sub,
    ttnn.BcastOpMath.MUL: torch.mul,
}


def _docstring_enum_params(arg, enum_type):
    """One ``pytest.param`` per ``ttnn.<enum>.<value>`` that the ttnn.bcast docstring names on the line of ``arg``."""
    enum_name = enum_type.__name__
    try:
        line = next(line for line in ttnn.bcast.__doc__.splitlines() if line.strip().startswith(f"{arg} ("))
        names = re.findall(rf"ttnn\.{enum_name}\.(\w+)", line)
        assert names, f"no ttnn.{enum_name} value on: {line.strip()!r}"
        return [pytest.param(getattr(enum_type, name), id=name) for name in dict.fromkeys(names)]
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the {arg} values from the ttnn.bcast docstring: {error}")
        return [pytest.param(reason, id="docstring-unreadable")]


def _docstring_sharded_params():
    """One ``pytest.param(dim, memory_layout)`` per shard layout that a Sharded item of the ttnn.bcast docstring names
    for its dim."""
    try:
        items = re.findall(r"- Sharded, with ``ttnn\.BcastOpDim\.(\w+)``:(.*)", ttnn.bcast.__doc__)
        params = [
            pytest.param(getattr(ttnn.BcastOpDim, dim), getattr(ttnn.TensorMemoryLayout, layout), id=f"{dim}-{layout}")
            for dim, item in items
            for layout in dict.fromkeys(re.findall(r"\b((?:HEIGHT|WIDTH|BLOCK)_SHARDED)\b", item))
        ]
        assert params, "no Sharded item names a shard layout"
        return params
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the Sharded items from the ttnn.bcast docstring: {error}")
        return [pytest.param(reason, reason, id="docstring-unreadable")]


def _docstring_tile_dtype_layout_params():
    """The TILE combinations of the docstring's dtype/layout table; the Sharded items are for a TILE input_a."""
    return [param for param in docstring_dtype_layout_params(ttnn.bcast) if param.values[1] != ttnn.ROW_MAJOR_LAYOUT]


def _exact_operands(shape, seed):
    """Multiples of 1/8 in [-11/8, 11/8]. A sum or difference of two of them is a multiple of 1/8 below 3, and a
    product a multiple of 1/64 below 2, so every result is a BFLOAT16 value, and a BFLOAT8_B value within a block of
    such results. The device result then does not depend on how the hardware rounds, and compares exactly."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-11, 12, shape, generator=generator).to(torch.float32) / 8


def _input_b_last_dims(dim, height, width):
    """The last two dims of input_b that the docstring gives for ``dim``."""
    return {ttnn.BcastOpDim.H: (1, width), ttnn.BcastOpDim.W: (height, 1), ttnn.BcastOpDim.HW: (1, 1)}[dim]


def _expected(math_op, input_a, input_b):
    return _TORCH_OPS[math_op](ttnn.to_torch(input_a).float(), ttnn.to_torch(input_b).float())


# Neither of the last two dims is a multiple of 32.
_SHAPE_A = (2, 3, 40, 70)


@pytest.mark.parametrize("dim", _docstring_enum_params("dim", ttnn.BcastOpDim))
@pytest.mark.parametrize("math_op", _docstring_enum_params("math_op", ttnn.BcastOpMath))
@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.bcast))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.bcast))
def test_bcast_docstring_supported_inputs(dtype, layout, buffer_type, math_op, dim, device):
    """Runs every dtype, layout, interleaved buffer type, math_op and dim the ttnn.bcast docstring lists.

    input_b has its dims before the last two all 1, then equal to input_a's, and the second call also passes
    ``memory_config``. A preallocated ``output_tensor`` then receives the output. The inputs make every result exact
    (see _exact_operands), so the output is compared exactly with torch. The docstring makes no claim on the sign of
    a zero result, so zeros are compared by value.
    """
    check_readable(dtype, layout, buffer_type, math_op, dim)
    memory_config = interleaved_memory_config(buffer_type)
    *batch, height, width = _SHAPE_A
    input_a = ttnn.from_torch(
        _exact_operands(_SHAPE_A, seed=0), dtype=dtype, layout=layout, device=device, memory_config=memory_config
    )
    for b_batch, output_memory_config in (((1, 1), None), (tuple(batch), memory_config)):
        input_b = ttnn.from_torch(
            _exact_operands(b_batch + _input_b_last_dims(dim, height, width), seed=1),
            dtype=dtype,
            layout=layout,
            device=device,
            memory_config=memory_config,
        )
        output = ttnn.bcast(input_a, input_b, math_op, dim, memory_config=output_memory_config)

        assert output.dtype == dtype
        assert output.layout == ttnn.TILE_LAYOUT
        assert output.memory_config() == memory_config
        expected = _expected(math_op, input_a, input_b)
        assert_exact(expected, ttnn.to_torch(output), signed_zeros=False)

    preallocated = ttnn.from_torch(
        torch.zeros(_SHAPE_A), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config
    )
    output = ttnn.bcast(input_a, input_b, math_op, dim, output_tensor=preallocated)
    assert output.buffer_address() == preallocated.buffer_address()
    assert_exact(expected, ttnn.to_torch(preallocated), signed_zeros=False)


# The shard grids start at core (0, 0); the WIDTH_SHARDED one is one row of cores. Each value: the shape of input_a,
# the last core of its grid, and its shard shape.
_SHARDED_INPUTS = {
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: ((1, 2, 32, 128), (3, 0), (64, 32)),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: ((1, 2, 32, 128), (1, 1), (32, 64)),
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: ((1, 2, 64, 64), (3, 0), (32, 64)),
}


@pytest.mark.parametrize("math_op", _docstring_enum_params("math_op", ttnn.BcastOpMath))
@pytest.mark.parametrize("dim, memory_layout", _docstring_sharded_params())
@pytest.mark.parametrize("dtype, layout", _docstring_tile_dtype_layout_params())
def test_bcast_docstring_sharded_inputs(dtype, layout, dim, memory_layout, math_op, device):
    """Runs each shard layout that a Sharded item of the ttnn.bcast docstring names, with its dim, every TILE dtype
    and every math_op the docstring lists. input_b is interleaved with its dims before the last two all 1, and the
    output keeps the sharding of input_a, with ``memory_config`` left out and set to that of input_a."""
    check_readable(dtype, layout, dim, memory_layout, math_op)
    shape, (end_x, end_y), shard_shape = _SHARDED_INPUTS[memory_layout]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})
    sharded = ttnn.MemoryConfig(
        memory_layout, ttnn.BufferType.L1, ttnn.ShardSpec(grid, list(shard_shape), ttnn.ShardOrientation.ROW_MAJOR)
    )
    input_a = ttnn.from_torch(
        _exact_operands(shape, seed=0), dtype=dtype, layout=layout, device=device, memory_config=sharded
    )
    input_b = ttnn.from_torch(
        _exact_operands((1, 1) + _input_b_last_dims(dim, *shape[-2:]), seed=1),
        dtype=dtype,
        layout=layout,
        device=device,
    )
    expected = _expected(math_op, input_a, input_b)
    for memory_config in (None, input_a.memory_config()):
        output = ttnn.bcast(input_a, input_b, math_op, dim, memory_config=memory_config)

        assert output.dtype == dtype
        assert output.memory_config() == input_a.memory_config()
        assert_exact(expected, ttnn.to_torch(output), signed_zeros=False)

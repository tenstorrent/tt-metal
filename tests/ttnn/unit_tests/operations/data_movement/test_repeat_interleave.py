# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import re

import pytest

import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_equal
from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    assert_exact,
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
    torch_input,
)


@pytest.mark.parametrize("repeats", [1, 2, 3, 58])
@pytest.mark.parametrize("dim", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.uint16])
def test_repeat_interleave(device, repeats, dim, dtype):
    if dtype == ttnn.uint16:
        torch_dtype = torch.int16
        torch_input_tensor = torch.randint(0, 100, (1, 1, 32, 32), dtype=torch_dtype)
    else:
        torch_dtype = torch.bfloat16
        torch_input_tensor = torch.rand(1, 1, 32, 32, dtype=torch_dtype)

    torch_result = torch.repeat_interleave(torch_input_tensor, repeats, dim=dim)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=dim)
    output = ttnn.to_torch(output)
    assert_equal(torch_result, output)


# --- Codegen-path coverage ---
#
# ttnn.repeat_interleave routes gate-supported cases to codegen and the rest to native, and offers
# no way to ask for one: the verification-only entries below live in the private module for that
# reason (see repeat_interleave_force.hpp). These pin the codegen path so the suite exercises it
# regardless of the gate's verdict.
#
# The codegen path supports only a subset of cases (see repeat_interleave_codegen_supported.cpp):
# interleaved input and output, rank 2-4, repeats > 1, bfloat16/float32/int32, and a repeated dim
# outside the pages the layout subdivides -- TILE defers the last two (sub-tile) dims, ROW_MAJOR
# defers the last (within-stick) dim. Every case below is hand-picked to satisfy that gate, so the
# forced entry resolves instead of raising. repeat_interleave copies values verbatim, so a lossless
# round-trip means assert_equal holds.
_force_native = ttnn._ttnn.operations.data_movement.repeat_interleave_force_native
_force_codegen = ttnn._ttnn.operations.data_movement.repeat_interleave_force_codegen

_CODEGEN_CASES = [
    # (shape, repeats, dim, layout)
    ((2, 3, 32, 64), 2, 1, ttnn.TILE_LAYOUT),
    ((2, 3, 32, 64), 3, 0, ttnn.TILE_LAYOUT),
    ((2, 3, 32, 64), 2, -3, ttnn.TILE_LAYOUT),
    ((2, 3, 32, 64), 2, 1, ttnn.ROW_MAJOR_LAYOUT),
    ((2, 3, 32, 64), 3, 0, ttnn.ROW_MAJOR_LAYOUT),
    ((4, 6, 8), 2, 1, ttnn.ROW_MAJOR_LAYOUT),
]
_CODEGEN_CASE_IDS = [
    "[2, 3, 32, 64]|dim=1&repeats=2|tile",
    "[2, 3, 32, 64]|dim=0&repeats=3|tile",
    "[2, 3, 32, 64]|dim=-3&repeats=2|tile",
    "[2, 3, 32, 64]|dim=1&repeats=2|row_major",
    "[2, 3, 32, 64]|dim=0&repeats=3|row_major",
    "[4, 6, 8]|dim=1&repeats=2|row_major",
]
_CODEGEN_DTYPES = [ttnn.bfloat16, ttnn.float32, ttnn.int32]
_CODEGEN_DTYPE_IDS = ["bfloat16", "float32", "int32"]


def _codegen_input(shape, dtype):
    if dtype == ttnn.int32:
        return torch.randint(0, 100, shape, dtype=torch.int32)
    return torch.rand(shape, dtype=torch.bfloat16)


@pytest.mark.parametrize("dtype", _CODEGEN_DTYPES, ids=_CODEGEN_DTYPE_IDS)
@pytest.mark.parametrize("shape,repeats,dim,layout", _CODEGEN_CASES, ids=_CODEGEN_CASE_IDS)
def test_repeat_interleave_codegen(device, shape, repeats, dim, layout, dtype):
    """Bit-exactness against the implementation codegen replaces, on the same input."""
    input_tensor = ttnn.from_torch(_codegen_input(shape, dtype), layout=layout, dtype=dtype, device=device)
    golden = ttnn.to_torch(_force_native(input_tensor, repeats, dim))
    assert_equal(golden, ttnn.to_torch(_force_codegen(input_tensor, repeats, dim)))


@pytest.mark.parametrize("shape,repeats,dim,layout", _CODEGEN_CASES, ids=_CODEGEN_CASE_IDS)
def test_pc_repeat_interleave_codegen(device, shape, repeats, dim, layout):
    """A second dispatch of the same spec must reuse the cached program with the new buffers.

    The descriptor factory hands raw Buffer*s to emplace_runtime_args, so a cache hit relies on the
    framework re-resolving those bindings; a stale binding would read or write the first
    invocation's allocation. Cache identity does not vary with dtype, so one is enough here.
    """
    dtype = ttnn.bfloat16
    first = ttnn.from_torch(_codegen_input(shape, dtype), layout=layout, dtype=dtype, device=device)
    first_golden = ttnn.to_torch(_force_native(first, repeats, dim))
    assert_equal(first_golden, ttnn.to_torch(_force_codegen(first, repeats, dim)))
    entries_after_miss = device.num_program_cache_entries()

    # A distinct allocation with the same spec: same program hash, different Buffer*.
    second = ttnn.from_torch(_codegen_input(shape, dtype), layout=layout, dtype=dtype, device=device)
    second_golden = ttnn.to_torch(_force_native(second, repeats, dim))
    assert_equal(second_golden, ttnn.to_torch(_force_codegen(second, repeats, dim)))
    msg = "second codegen dispatch missed the program cache"
    assert device.num_program_cache_entries() == entries_after_miss, msg


@pytest.mark.skip(reason="ttnn.repeat_interleave only supports `repeats` as int")
def test_repeat_interleave_with_repeat_tensor(device):
    torch_input_tensor = torch.rand(1, 2, 32, 32, dtype=torch.bfloat16)
    torch_repeats = torch.tensor([1, 2])
    torch_result = torch.repeat_interleave(torch_input_tensor, torch_repeats, dim=1)
    input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
    repeats = ttnn.from_torch(torch_repeats)
    output = ttnn.repeat_interleave(input_tensor, repeats, dim=1)
    output = ttnn.to_torch(output)

    assert_equal(torch_result, output)


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.repeat_interleave))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.repeat_interleave))
def test_repeat_interleave_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.repeat_interleave docstring lists, repeating
    an outer dim and the last dim."""
    memory_config = interleaved_memory_config(buffer_type)
    input_tensor = ttnn.from_torch(
        torch_input((2, 3, 32, 64), dtype), dtype=dtype, layout=layout, device=device, memory_config=memory_config
    )
    expected = ttnn.to_torch(input_tensor)
    for dim in (1, -1):
        output = ttnn.repeat_interleave(input_tensor, 2, dim=dim, memory_config=memory_config)

        assert output.dtype == dtype
        assert output.layout == layout
        assert output.memory_config().buffer_type == buffer_type
        assert_exact(torch.repeat_interleave(expected, 2, dim=dim), ttnn.to_torch(output))


def _docstring_kept_shard_layouts():
    """Shard layouts the ttnn.repeat_interleave docstring says keep their sharding when an outer dim is repeated."""
    try:
        line = next(line for line in ttnn.repeat_interleave.__doc__.splitlines() if "keeps the same sharding" in line)
        names = re.findall(r"\b((?:HEIGHT|WIDTH|BLOCK)_SHARDED)\b", line.split("keeps the same sharding")[0])
        assert names, f"no shard layouts named on: {line.strip()!r}"
        return [pytest.param(getattr(ttnn.TensorMemoryLayout, name), id=name) for name in names]
    except Exception as error:
        return [
            pytest.param(UnreadableDocstring(f"could not read the sharded line: {error}"), id="docstring-unreadable")
        ]


_SHARDED_INPUTS = {
    ttnn.TensorMemoryLayout.HEIGHT_SHARDED: ((1, 32, 8, 16), ttnn.CoreGrid(y=1, x=8), ttnn.ShardStrategy.HEIGHT),
    ttnn.TensorMemoryLayout.WIDTH_SHARDED: ((1, 2, 8, 128), ttnn.CoreGrid(y=1, x=4), ttnn.ShardStrategy.WIDTH),
    ttnn.TensorMemoryLayout.BLOCK_SHARDED: ((2, 32, 8, 32), ttnn.CoreGrid(y=2, x=4), ttnn.ShardStrategy.BLOCK),
}


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("dim", [0, 1, 2, -1])
@pytest.mark.parametrize("memory_layout", _docstring_kept_shard_layouts())
def test_repeat_interleave_docstring_sharded_inputs(memory_layout, dim, device):
    """A ROW_MAJOR rank-4 L1 input of each shard layout the docstring names keeps that sharding when a dim other than
    the last is repeated, and a last-dim repeat gives an L1 output."""
    check_readable(memory_layout)
    shape, core_grid, strategy = _SHARDED_INPUTS[memory_layout]
    x = torch.rand(shape, dtype=torch.bfloat16)
    sharded = ttnn.create_sharded_memory_config(list(shape), core_grid, strategy, ttnn.ShardOrientation.ROW_MAJOR)
    input_tensor = ttnn.to_memory_config(ttnn.from_torch(x, layout=ttnn.ROW_MAJOR_LAYOUT, device=device), sharded)
    output = ttnn.repeat_interleave(input_tensor, 2, dim=dim)

    assert output.memory_config().buffer_type == ttnn.BufferType.L1
    if dim != -1:
        assert output.memory_config().memory_layout == memory_layout
    assert_exact(torch.repeat_interleave(x, 2, dim=dim), ttnn.to_torch(output))

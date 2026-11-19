# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import re

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, assert_equal
from tests.ttnn.unit_tests.operations.data_movement.docstring_support import (
    UnreadableDocstring,
    check_readable,
    docstring_dtype_layout_params,
    docstring_interleaved_buffer_type_params,
    interleaved_memory_config,
)
from tests.ttnn.unit_tests.operations.data_movement.data_movement_test_utils import assert_exact, full_range_input


@pytest.mark.parametrize(
    "input_tensor, shifts, dim, layout, dtype, memory_config",
    [
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 1, 0, ttnn.ROW_MAJOR_LAYOUT, torch.float32, ttnn.DRAM_MEMORY_CONFIG),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], -1, 0, ttnn.TILE_LAYOUT, torch.float32, ttnn.L1_MEMORY_CONFIG),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 0, 0, ttnn.ROW_MAJOR_LAYOUT, torch.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 4, 0, ttnn.TILE_LAYOUT, torch.float32, ttnn.L1_MEMORY_CONFIG),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 5, 0, ttnn.ROW_MAJOR_LAYOUT, torch.int32, ttnn.DRAM_MEMORY_CONFIG),
        ([[1, 1], [1, 1], [1, 1], [1, 1]], 1, 1, ttnn.TILE_LAYOUT, torch.bfloat16, ttnn.L1_MEMORY_CONFIG),
        (
            [
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
                [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29], [30, 31, 32, 33, 34], [35, 36, 37, 38, 39]],
                [[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]],
                [[60, 61, 62, 63, 64], [65, 66, 67, 68, 69], [70, 71, 72, 73, 74], [75, 76, 77, 78, 79]],
                [[80, 81, 82, 83, 84], [85, 86, 87, 88, 89], [90, 91, 92, 93, 94], [95, 96, 97, 98, 99]],
                [
                    [100, 101, 102, 103, 104],
                    [105, 106, 107, 108, 109],
                    [110, 111, 112, 113, 114],
                    [115, 116, 117, 118, 119],
                ],
            ],
            [-2, -1],
            [0, 1],
            ttnn.TILE_LAYOUT,
            torch.float32,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]],
            [1, 0],
            [0, 1],
            ttnn.ROW_MAJOR_LAYOUT,
            torch.bfloat16,
            ttnn.L1_MEMORY_CONFIG,
        ),
        (
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
            [1, 0, -2],
            [0, 1, 2],
            ttnn.TILE_LAYOUT,
            torch.float32,
            ttnn.DRAM_MEMORY_CONFIG,
        ),
        (
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]],
            [0, -1, -2],
            [0, 1, 2],
            ttnn.ROW_MAJOR_LAYOUT,
            torch.bfloat16,
            ttnn.L1_MEMORY_CONFIG,
        ),
    ],
)
def test_roll(device, input_tensor, shifts, dim, layout, dtype, memory_config):
    tensor = torch.tensor(input_tensor, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(tensor)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config)

    pytorch_out = torch.roll(tensor, shifts, dim)
    ttnn_out = ttnn.roll(ttnn_tensor, shifts, dim)
    ttnn_result_torch = ttnn.to_torch(ttnn_out)

    assert_equal(pytorch_out, ttnn_result_torch)


@pytest.mark.parametrize(
    "input_tensor, shifts",
    [
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 2),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 3),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], -2),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 4),
        ([[1, 2], [3, 4], [5, 6], [7, 8]], 5),
        ([[1, 1], [1, 1], [1, 1], [1, 1]], 2),
        (
            [
                [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
                [[20, 21, 22, 23, 24], [25, 26, 27, 28, 29], [30, 31, 32, 33, 34], [35, 36, 37, 38, 39]],
                [[40, 41, 42, 43, 44], [45, 46, 47, 48, 49], [50, 51, 52, 53, 54], [55, 56, 57, 58, 59]],
                [[60, 61, 62, 63, 64], [65, 66, 67, 68, 69], [70, 71, 72, 73, 74], [75, 76, 77, 78, 79]],
                [[80, 81, 82, 83, 84], [85, 86, 87, 88, 89], [90, 91, 92, 93, 94], [95, 96, 97, 98, 99]],
                [
                    [100, 101, 102, 103, 104],
                    [105, 106, 107, 108, 109],
                    [110, 111, 112, 113, 114],
                    [115, 116, 117, 118, 119],
                ],
            ],
            -2,
        ),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], 4),
        ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]], 2),
        ([[[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]], [[13, 14], [15, 16]]]], -3),
    ],
)
def test_roll_without_dim(device, input_tensor, shifts):
    tensor = torch.tensor(input_tensor, dtype=torch.float32)
    ttnn_tensor = ttnn.from_torch(tensor)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, ttnn.ROW_MAJOR_LAYOUT)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pytorch_out = torch.roll(tensor, shifts)
    ttnn_out = ttnn.roll(ttnn_tensor, shifts)
    ttnn_result_torch = ttnn.to_torch(ttnn_out)
    assert_equal(pytorch_out, ttnn_result_torch)


@pytest.mark.parametrize(
    "input_tensor, shifts, dim, layout, dtype",
    [
        (
            [
                [
                    [
                        [1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9],
                        [10, 11, 12],
                        [13, 14, 15],
                        [16, 17, 18],
                        [19, 20, 21],
                        [22, 23, 24],
                    ],
                    [
                        [25, 26, 27],
                        [28, 29, 30],
                        [31, 32, 33],
                        [34, 35, 36],
                        [37, 38, 39],
                        [40, 41, 42],
                        [43, 44, 45],
                        [46, 47, 48],
                    ],
                    [
                        [49, 50, 51],
                        [52, 53, 54],
                        [55, 56, 57],
                        [58, 59, 60],
                        [61, 62, 63],
                        [64, 65, 66],
                        [67, 68, 69],
                        [70, 71, 72],
                    ],
                    [
                        [73, 74, 75],
                        [76, 77, 78],
                        [79, 80, 81],
                        [82, 83, 84],
                        [85, 86, 87],
                        [88, 89, 90],
                        [91, 92, 93],
                        [94, 95, 96],
                    ],
                ]
            ],
            1,
            1,
            ttnn.TILE_LAYOUT,
            torch.bfloat16,
        ),
        (
            [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]],
            [1, 0],
            [2, 3],
            ttnn.TILE_LAYOUT,
            torch.float32,
        ),
    ],
)
def test_roll_tile_padding(device, input_tensor, shifts, dim, layout, dtype):
    tensor = torch.tensor(input_tensor, dtype=dtype)
    ttnn_tensor = ttnn.from_torch(tensor)
    ttnn_tensor = ttnn.to_layout(ttnn_tensor, layout)
    ttnn_tensor = ttnn.to_device(ttnn_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    pytorch_out = torch.roll(tensor, shifts, dim)
    ttnn_out = ttnn.roll(ttnn_tensor, shifts, dim)
    ttnn_result_torch = ttnn.to_torch(ttnn_out)

    assert_equal(pytorch_out, ttnn_result_torch)


@pytest.mark.parametrize(
    "shape, shifts, dims",
    [
        ((4, 4, 32, 32), [1, 0], [2, 3]),
        ((2, 2, 64, 64), [2, 1], [2, 3]),
        ((1, 1, 32, 32), [1, -1], [2, 3]),
    ],
)
def test_roll_with_program_cache(device, shape, shifts, dims):
    dtype = torch.float32
    torch_input_tensor = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=dtype).reshape(shape)
    torch_output_tensor = torch.roll(torch_input_tensor, shifts=shifts, dims=dims)

    ttnn_input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)

    for i in range(2):
        ttnn_output_tensor = ttnn.roll(ttnn_input_tensor, shifts, dims)
        ttnn_output_torch = ttnn.to_torch(ttnn_output_tensor)

        assert_with_pcc(torch_output_tensor, ttnn_output_torch)
        assert torch.allclose(torch_output_tensor, ttnn_output_torch)


@pytest.mark.parametrize(
    "first_shape, second_shape, shifts, dims",
    [
        ((1, 1, 32, 32), (1, 1, 128, 128), [1, 1], [2, 3]),
        ((2, 2, 32, 32), (2, 2, 96, 64), [2, -1], [2, 3]),
    ],
)
def test_roll_cache_hit_larger_shape(device, first_shape, second_shape, shifts, dims):
    # shift/dim/dtype/layout/memory_config match, so both shapes share one program-cache entry by design:
    # the hit path re-applies the new shape's transfer plan. Guards that reuse against a stale plan.
    for shape in (first_shape, second_shape):
        torch_input = torch.arange(1, 1 + torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
        expected = torch.roll(torch_input, shifts=shifts, dims=dims)
        ttnn_input = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
        actual = ttnn.to_torch(ttnn.roll(ttnn_input, shifts, dims))
        assert_with_pcc(expected, actual)


def _empty_rm(device):
    return ttnn.from_torch(
        torch.randn(0, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )


# Rolling an empty tensor is a no-op; it used to take `shift % 0`.
@pytest.mark.parametrize("dims", [(0,), None], ids=["dim0", "flattened"])
def test_roll_empty_tensor(device, dims):
    x = _empty_rm(device)
    out = ttnn.roll(x, 1, dims[0]) if dims else ttnn.roll(x, 1)
    assert tuple(out.shape) == (0, 32)


def test_roll_empty_tensor_honors_memory_config(device):
    out = ttnn.roll(_empty_rm(device), 1, 0, memory_config=ttnn.L1_MEMORY_CONFIG)
    assert tuple(out.shape) == (0, 32)
    assert out.memory_config().buffer_type == ttnn.BufferType.L1


_BLOCK_FLOATS = (ttnn.bfloat8_b, ttnn.bfloat4_b)


def _one_exponent_input(shape, seed=0):
    """Values in +-[1, 2) for a block float. They share one exponent, so values a roll moves into another block of the
    tile come back exactly."""
    torch.manual_seed(seed)
    magnitude = 1 + torch.rand(shape)
    return torch.where(torch.rand(shape) < 0.5, -magnitude, magnitude)


def _listed(value):
    return [value] if isinstance(value, int) else list(value)


def _rolls_last_two_dims(rank, dims):
    """True when the roll has no dim, or includes one of the last two dims."""
    return dims is None or any(dim % rank >= rank - 2 for dim in _listed(dims))


def _bfloat8_b_roll(shape, shifts, dims):
    """The rolls the docstring lists for BFLOAT8_B: a dim is given, and each rolled dim among the last two has a size and
    a shift that are multiples of 32."""
    rank = len(shape)
    return dims is not None and all(
        shape[dim] % 32 == 0 and shift % 32 == 0
        for shift, dim in zip(_listed(shifts), _listed(dims))
        if dim % rank >= rank - 2
    )


def _torch_roll(tensor, shifts, dims):
    # Integers are compared by value, so they widen to int64, which torch.roll takes for every integer dtype.
    tensor = tensor if tensor.is_floating_point() else tensor.to(torch.int64)
    return torch.roll(tensor, shifts) if dims is None else torch.roll(tensor, shifts, dims)


# (input shape, shifts, dims): int and list forms, negative shifts and dims, a shift larger than its dim, tile-aligned and
# unaligned rolls of the last two dims, a shape with tile padding, and no dim (the flattened tensor is rolled).
_ROLLS = [
    ((2, 3, 64, 96), 1, 1),
    ((2, 3, 64, 96), [-1, 2], [0, -3]),
    ((2, 3, 64, 96), [32, -32], [-1, 2]),
    ((2, 3, 64, 96), [-5, 7, 100], [-1, -2, 1]),
    ((2, 3, 30, 50), 1, 0),
    ((2, 3, 30, 50), [7, -3], [-1, -2]),
    ((2, 3, 30, 50), 3, None),
]


@pytest.mark.parametrize("buffer_type", docstring_interleaved_buffer_type_params(ttnn.roll))
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.roll))
def test_roll_docstring_supported_inputs(dtype, layout, buffer_type, device):
    """Runs every dtype, layout and interleaved buffer type the ttnn.roll docstring lists, through each call form: an
    int shift and dim, lists of shifts and dims, and no dim."""
    check_readable(dtype, layout, buffer_type)
    memory_config = interleaved_memory_config(buffer_type)
    for shape, shifts, dims in _ROLLS:
        if dtype == ttnn.bfloat8_b and not _bfloat8_b_roll(shape, shifts, dims):
            continue
        if dtype == ttnn.uint8 and dims is None:  # the docstring asks for `dim` with UINT8
            continue
        # With no dim, or with one of the last two dims rolled, a BFLOAT16 input leaves out NaN, as the docstring
        # requires, and its zeros are compared by value. Block-float values can then move into other blocks, so they get
        # one exponent.
        last_two = _rolls_last_two_dims(len(shape), dims)
        bfloat16_last_two = dtype == ttnn.bfloat16 and last_two
        if dtype in _BLOCK_FLOATS and last_two:
            x = _one_exponent_input(shape)
        else:
            x = full_range_input(shape, dtype, nan=not bfloat16_last_two)
        input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
        output = ttnn.roll(input_tensor, shifts) if dims is None else ttnn.roll(input_tensor, shifts, dims)

        assert output.dtype == dtype
        assert output.layout == layout
        assert output.memory_config() == input_tensor.memory_config()
        expected = _torch_roll(ttnn.to_torch(input_tensor), shifts, dims)
        assert_exact(expected, ttnn.to_torch(output), signed_zeros=not bfloat16_last_two)

    # A given memory_config sets the output's memory configuration.
    other_config = interleaved_memory_config(
        ttnn.BufferType.L1 if buffer_type == ttnn.BufferType.DRAM else ttnn.BufferType.DRAM
    )
    x = full_range_input((2, 3, 30, 50), dtype)
    input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=memory_config)
    output = ttnn.roll(input_tensor, 1, 0, memory_config=other_config)
    assert output.memory_config().buffer_type == other_config.buffer_type
    assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert_exact(_torch_roll(ttnn.to_torch(input_tensor), 1, 0), ttnn.to_torch(output))


_SHARDINGS = ("HEIGHT_SHARDED", "WIDTH_SHARDED", "BLOCK_SHARDED")


def _docstring_sharded_params():
    """One ``pytest.param(memory_layout, buffer_type)`` per sharding the Sharded item of the ttnn.roll docstring names in
    its first sentence, written as ``<shardings> in <buffer type>`` clauses separated by ``;``."""
    try:
        items = [line.strip() for line in ttnn.roll.__doc__.splitlines() if line.strip().startswith("- Sharded:")]
        assert len(items) == 1, f"expected one '- Sharded:' item, got {items}"
        placement = re.split(r"\.(?:\s|$)", items[0][len("- Sharded:") :], maxsplit=1)[0]
        params = []
        for clause in placement.split(";"):
            match = re.fullmatch(r"\s*(\S.*?) in (\w+)\s*", clause)
            assert match, f"not '<shardings> in <buffer type>': {clause!r}"
            buffer_name = match.group(2)
            buffer_type = getattr(ttnn.BufferType, buffer_name, None)
            assert isinstance(buffer_type, ttnn.BufferType), f"not a ttnn.BufferType: {buffer_name!r} in {clause!r}"
            for name in re.split(r",\s*|\s+and\s+", match.group(1)):
                assert name in _SHARDINGS, f"not a sharding: {name!r} in {clause!r}"
                memory_layout = getattr(ttnn.TensorMemoryLayout, name)
                params.append(pytest.param(memory_layout, buffer_type, id=f"{name}-{buffer_name}"))
        return params
    except Exception as error:
        reason = UnreadableDocstring(f"could not read the Sharded item from the ttnn.roll docstring: {error}")
        return [pytest.param(reason, reason, id="docstring-unreadable")]


_SHARDED_SHAPE = (2, 2, 64, 64)
# Last core of the shard grid, which starts at core (0, 0), and the shard shape, per sharding of _SHARDED_SHAPE.
_SHARD_SPECS = {
    (ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1): ((3, 0), (64, 64)),
    (ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1): ((1, 0), (256, 32)),
    (ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1): ((1, 3), (64, 32)),
}
_SHARDED_ROLLS = [(1, 0), (-7, -1), ([32, 1], [-1, 0]), ([-1, 5], [1, -2]), (32, -2)]


def _within_shard_height_rule(shape, dims, shard_height):
    """The docstring's rule: for each rolled dim other than the last, the shard height is at most the product of the
    sizes of the dims from that one through the second-to-last."""
    rank = len(shape)
    return all(
        shard_height <= math.prod(shape[dim % rank : rank - 1]) for dim in _listed(dims) if dim % rank != rank - 1
    )


@pytest.mark.parametrize("memory_layout, buffer_type", _docstring_sharded_params())
@pytest.mark.parametrize("dtype, layout", docstring_dtype_layout_params(ttnn.roll))
def test_roll_docstring_sharded_inputs(dtype, layout, memory_layout, buffer_type, device):
    """An input of each dtype, layout and sharding the ttnn.roll docstring lists keeps its memory configuration, for
    each roll that meets the docstring's shard height rule and, for BFLOAT8_B, its BFLOAT8_B rule."""
    check_readable(dtype, layout, memory_layout, buffer_type)
    if (memory_layout, buffer_type) not in _SHARD_SPECS:
        pytest.fail(f"the docstring names {memory_layout} in {buffer_type}, which this test has no shard spec for")
    (end_x, end_y), shard_shape = _SHARD_SPECS[(memory_layout, buffer_type)]
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(end_x, end_y))})
    sharded = ttnn.MemoryConfig(
        memory_layout, buffer_type, ttnn.ShardSpec(grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    )
    rolls = [(s, d) for s, d in _SHARDED_ROLLS if _within_shard_height_rule(_SHARDED_SHAPE, d, shard_shape[0])]
    for shifts, dims in rolls:
        if dtype == ttnn.bfloat8_b and not _bfloat8_b_roll(_SHARDED_SHAPE, shifts, dims):
            continue
        # As in test_roll_docstring_supported_inputs: once one of the last two dims rolls, a BFLOAT16 input leaves out
        # NaN and its zeros are compared by value, and block floats get values with one exponent.
        last_two = _rolls_last_two_dims(len(_SHARDED_SHAPE), dims)
        bfloat16_last_two = dtype == ttnn.bfloat16 and last_two
        if dtype in _BLOCK_FLOATS and last_two:
            x = _one_exponent_input(_SHARDED_SHAPE)
        else:
            x = full_range_input(_SHARDED_SHAPE, dtype, nan=not bfloat16_last_two)
        input_tensor = ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, memory_config=sharded)
        expected = _torch_roll(ttnn.to_torch(input_tensor), shifts, dims)

        output = ttnn.roll(input_tensor, shifts, dims)

        case = f"shifts {shifts}, dims {dims}"
        assert output.dtype == dtype, case
        assert output.layout == layout, case
        assert output.memory_config() == input_tensor.memory_config(), case
        assert_exact(expected, ttnn.to_torch(output), signed_zeros=not bfloat16_last_two)

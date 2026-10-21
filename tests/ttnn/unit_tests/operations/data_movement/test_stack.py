# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc

RM = ttnn.ROW_MAJOR_LAYOUT
TILE = ttnn.TILE_LAYOUT
MEMORY_CONFIGS = [
    pytest.param(ttnn.L1_MEMORY_CONFIG, id="l1"),
    pytest.param(ttnn.DRAM_MEMORY_CONFIG, id="dram"),
]

# A new trailing axis on interleaved row-major (rank >= 1) or tile (rank >= 2) inputs goes through
# concat at 0 plus one permute; everything else goes through unsqueeze-then-concat. Routing is not
# observable from the values, so the ids only record which route each case covers; the program-cache
# test below is what asserts the route.
STACK_CASES = [
    # unsqueeze-then-concat: the new axis is not trailing
    pytest.param([(1, 1, 253)] * 3, 2, RM, id="rm-rank3-dim2-unsqueeze"),
    pytest.param([(1, 253)] * 3, 1, RM, id="rm-rank2-dim1-unsqueeze"),
    pytest.param([(57, 83)] * 3, 0, TILE, id="tile-rank2-dim0-unsqueeze"),
    pytest.param([(8732,)] * 3, 0, RM, id="rm-rank1-dim0-unsqueeze"),
    pytest.param([(1, 1, 253)] * 3, -3, RM, id="rm-rank3-dim-3-unsqueeze"),
    pytest.param([(2, 32, 64)] * 2, -2, RM, id="rm-rank3-dim-2-unsqueeze"),
    pytest.param([(120, 1, 253, 3)] * 4, 3, RM, id="rm-rank4-dim3-unsqueeze"),
    pytest.param([(24, 50, 83)] * 6, 2, TILE, id="tile-rank3-dim2-unsqueeze"),
    # unsqueeze-then-concat: rank-1 tile inputs stay there even on a trailing axis
    pytest.param([(1444,)] * 8, -1, TILE, id="tile-rank1-trailing-unsqueeze"),
    # leading-axis route, row-major
    pytest.param([(13600,)] * 4, 1, RM, id="rm-rank1-trailing-n4"),
    pytest.param([(1444,)] * 8, -1, RM, id="rm-rank1-trailing-n8"),
    pytest.param([(400,)] * 12, -1, RM, id="rm-rank1-trailing-n12"),
    pytest.param([(253,)] * 3, 1, RM, id="rm-rank1-trailing-odd-stick"),
    pytest.param([(1, 253)] * 3, 2, RM, id="rm-rank2-trailing-odd-stick"),
    pytest.param([(8732, 2)] * 2, 2, RM, id="rm-rank2-trailing"),
    pytest.param([(1, 23, 40, 64)] * 2, 4, RM, id="rm-rank4-trailing"),
    pytest.param([(1, 2, 3, 32, 64)] * 2, 5, RM, id="rm-rank5-trailing"),
    pytest.param([(5, 7)], -1, RM, id="rm-rank2-trailing-single-input"),
    # leading-axis route, tile (rank >= 2)
    pytest.param([(123, 259)] * 3, -1, TILE, id="tile-rank2-trailing-n3"),
    pytest.param([(57, 83)] * 7, -1, TILE, id="tile-rank2-trailing-n7"),
    pytest.param([(64, 32)] * 12, -1, TILE, id="tile-rank2-trailing-n12"),
    pytest.param([(8732, 1)] * 4, 2, TILE, id="tile-rank2-trailing-width1"),
    pytest.param([(8732, 2)] * 2, 2, TILE, id="tile-rank2-trailing-width2"),
    pytest.param([(1, 23, 40, 64)] * 2, 4, TILE, id="tile-rank4-trailing"),
    pytest.param([(1, 5, 16, 16)] * 2, -1, TILE, id="tile-rank4-trailing-small"),
    pytest.param([(1, 2, 3, 32, 64)] * 2, 5, TILE, id="tile-rank5-trailing"),
    pytest.param([(5, 7)], -1, TILE, id="tile-rank2-trailing-single-input"),
]


def _expected_padded_shape(shape, layout):
    if layout != TILE:
        return list(shape)
    padded = list(shape)
    for axis in range(max(len(padded) - 2, 0), len(padded)):
        padded[axis] = -(-padded[axis] // 32) * 32
    return padded


def _check_result(torch_result, ttnn_result, layout):
    ttnn_result_torch = ttnn.to_torch(ttnn_result)
    assert torch_result.shape == ttnn_result_torch.shape
    assert_equal(torch_result, ttnn_result_torch)
    # Both routes end in concat's default placement, DRAM, whatever the inputs' placement.
    assert ttnn_result.memory_config().buffer_type == ttnn.BufferType.DRAM
    assert list(ttnn_result.padded_shape) == _expected_padded_shape(torch_result.shape, layout)


@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
@pytest.mark.parametrize("input_shapes, dim, layout", STACK_CASES)
def test_stack(device, input_shapes, dim, layout, memory_config):
    torch_tensors = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]

    ttnn_tensors = [ttnn.from_torch(tensor) for tensor in torch_tensors]
    ttnn_tensors = [ttnn.to_device(tensor, device, memory_config=memory_config) for tensor in ttnn_tensors]
    ttnn_tensors = [ttnn.to_layout(tensor, layout) for tensor in ttnn_tensors]
    torch_result = torch.stack(torch_tensors, dim=dim)
    ttnn_result = ttnn.stack(ttnn_tensors, dim=dim)

    _check_result(torch_result, ttnn_result, layout)


def _make_torch(shape, torch_dtype):
    if torch_dtype.is_floating_point:
        return torch.rand(shape, dtype=torch_dtype)
    return torch.randint(0, 1000, shape, dtype=torch_dtype)


@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
@pytest.mark.parametrize("layout", [pytest.param(RM, id="rm"), pytest.param(TILE, id="tile")])
@pytest.mark.parametrize(
    "input_shapes, dim, torch_dtype, ttnn_dtype",
    [
        # Trailing axis, 4-byte and float dtypes: concat's stick size and permute's element-size
        # compile-time arg and 32-bit dest path all change with the dtype.
        pytest.param([(1444,)] * 8, -1, torch.int32, ttnn.int32, id="rank1-int32"),
        pytest.param([(8732, 1)] * 4, 2, torch.int32, ttnn.int32, id="rank2-int32"),
        pytest.param([(8732, 2)] * 2, 2, torch.int32, ttnn.uint32, id="rank2-uint32"),
        pytest.param([(1, 23, 40, 64)] * 2, 4, torch.float32, ttnn.float32, id="rank4-float32"),
    ],
)
def test_stack_trailing_axis_dtypes(device, input_shapes, dim, torch_dtype, ttnn_dtype, layout, memory_config):
    torch_tensors = [_make_torch(shape, torch_dtype) for shape in input_shapes]

    ttnn_tensors = [
        ttnn.from_torch(tensor, dtype=ttnn_dtype, layout=layout, device=device, memory_config=memory_config)
        for tensor in torch_tensors
    ]
    torch_result = torch.stack(torch_tensors, dim=dim)
    ttnn_result = ttnn.stack(ttnn_tensors, dim=dim)

    _check_result(torch_result, ttnn_result, layout)


def test_stack_negative_non_trailing_dim(device):
    torch_tensors = [_make_torch((13600,), torch.int32) for _ in range(3)]
    ttnn_tensors = [
        ttnn.from_torch(tensor, dtype=ttnn.int32, layout=RM, device=device, memory_config=ttnn.L1_MEMORY_CONFIG)
        for tensor in torch_tensors
    ]
    torch_result = torch.stack(torch_tensors, dim=-2)
    ttnn_result = ttnn.stack(ttnn_tensors, dim=-2)

    _check_result(torch_result, ttnn_result, RM)


@pytest.mark.parametrize(
    "input_shapes, dim",
    [
        pytest.param([(8732, 2)] * 2, 2, id="rank2-trailing"),
        pytest.param([(1, 23, 40, 64)] * 2, 4, id="rank4-trailing"),
    ],
)
def test_stack_block_float_tile(device, input_shapes, dim):
    # Block floats stay on unsqueeze-then-concat: the rank-5 permute of the leading-axis route has
    # no bfloat8_b typecast. Both routes round-trip through bfloat16, hence PCC rather than equality.
    torch_tensors = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    ttnn_tensors = [
        ttnn.from_torch(tensor, dtype=ttnn.bfloat8_b, layout=TILE, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for tensor in torch_tensors
    ]
    torch_result = torch.stack(torch_tensors, dim=dim)
    ttnn_result = ttnn.stack(ttnn_tensors, dim=dim)

    ttnn_result_torch = ttnn.to_torch(ttnn_result)
    assert torch_result.shape == ttnn_result_torch.shape
    assert_with_pcc(torch_result, ttnn_result_torch, 0.999)


def test_stack_trailing_axis_takes_leading_axis_route(device):
    # Correctness cannot tell the routes apart, so count the programs a fresh trailing-axis stack
    # compiles: the leading-axis route is one concat and one permute; unsqueeze-then-concat compiles
    # the per-input reshape, the concat and its transpose fallback as well.
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    try:

        def stack_fresh(width):
            torch_tensors = [torch.rand((width,), dtype=torch.bfloat16) for _ in range(4)]
            ttnn_tensors = [
                ttnn.from_torch(tensor, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                for tensor in torch_tensors
            ]
            ttnn_result = ttnn.stack(ttnn_tensors, dim=1)
            assert_equal(torch.stack(torch_tensors, dim=1), ttnn.to_torch(ttnn_result))

        stack_fresh(1444)
        entries_after_prime = device.num_program_cache_entries()
        stack_fresh(1444)
        assert device.num_program_cache_entries() == entries_after_prime, "same shapes must hit the cache"
        stack_fresh(1500)
        assert (
            device.num_program_cache_entries() == entries_after_prime + 2
        ), "a new trailing-axis shape must compile exactly one concat and one permute"
    finally:
        device.disable_and_clear_program_cache()
        device.enable_program_cache()


@pytest.mark.parametrize("dim", [2, -3], ids=["dim=rank+1", "dim=-(rank+2)"])
def test_stack_dim_out_of_range(device, expect_error, dim):
    ttnn_tensors = [ttnn.from_torch(torch.rand((8,), dtype=torch.bfloat16), layout=RM, device=device) for _ in range(2)]
    with expect_error(RuntimeError, "out of range"):
        ttnn.stack(ttnn_tensors, dim=dim)


@pytest.mark.parametrize("memory_config", MEMORY_CONFIGS)
def test_stack_trailing_axis_uint16_row_major(device, memory_config):
    # Row-major only: a rank-1 tile input keeps the unsqueeze-then-concat path, whose device reshape
    # rejects uint16.
    torch_tensors = [_make_torch((1444,), torch.uint16) for _ in range(8)]
    ttnn_tensors = [
        ttnn.from_torch(tensor, dtype=ttnn.uint16, layout=RM, device=device, memory_config=memory_config)
        for tensor in torch_tensors
    ]
    torch_result = torch.stack(torch_tensors, dim=-1)
    ttnn_result = ttnn.stack(ttnn_tensors, dim=-1)

    _check_result(torch_result, ttnn_result, RM)

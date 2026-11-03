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

# A new trailing axis on same-layout, interleaved, non-block-float row-major (rank >= 1) or tile
# (rank >= 2) inputs goes through concat at 0 plus one permute; everything else goes through
# unsqueeze-then-concat, marked by the "-unsqueeze" id suffix. Routing is not observable from the
# values, so the ids only record which route the gate in stack.cpp gives each case; the device-op and
# program-cache tests below assert the route for one case per layout.
STACK_CASES = [
    # unsqueeze-then-concat: the new axis is not trailing
    pytest.param([(1, 1, 253)] * 3, 2, RM, id="rm-rank3-dim2-unsqueeze"),
    pytest.param([(1, 253)] * 3, 1, RM, id="rm-rank2-dim1-unsqueeze"),
    pytest.param([(57, 83)] * 3, 0, TILE, id="tile-rank2-dim0-unsqueeze"),
    pytest.param([(8732,)] * 3, 0, RM, id="rm-rank1-dim0-unsqueeze"),
    pytest.param([(1, 1, 253)] * 3, -3, RM, id="rm-rank3-dimneg3-unsqueeze"),
    pytest.param([(2, 32, 64)] * 2, -2, RM, id="rm-rank3-dimneg2-unsqueeze"),
    pytest.param([(120, 1, 253, 3)] * 4, 3, RM, id="rm-rank4-dim3-unsqueeze"),
    pytest.param([(24, 50, 83)] * 6, 2, TILE, id="tile-rank3-dim2-unsqueeze"),
    # unsqueeze-then-concat: rank-1 tile inputs stay there even on a trailing axis
    pytest.param([(1444,)] * 8, -1, TILE, id="tile-rank1-trailing-unsqueeze"),
    # unsqueeze-then-concat: a single tile input is one tiled reshape, cheaper than the permute
    pytest.param([(5, 7)], -1, TILE, id="tile-rank2-trailing-single-input-unsqueeze"),
    # leading-axis route, row-major
    pytest.param([(13600,)] * 4, 1, RM, id="rm-rank1-trailing-n4"),
    pytest.param([(1444,)] * 8, -1, RM, id="rm-rank1-trailing-n8"),
    pytest.param([(400,)] * 12, -1, RM, id="rm-rank1-trailing-n12"),
    pytest.param([(253,)] * 3, 1, RM, id="rm-rank1-trailing-odd-stick"),
    pytest.param([(1, 253)] * 3, 2, RM, id="rm-rank2-trailing-odd-stick"),
    pytest.param([(8732, 2)] * 2, 2, RM, id="rm-rank2-trailing"),
    # rank 3 is the one input rank whose stacked tensor is exactly 4-D, so permute takes it unpadded
    pytest.param([(2, 37, 64)] * 3, 3, RM, id="rm-rank3-trailing"),
    pytest.param([(1, 23, 40, 64)] * 2, 4, RM, id="rm-rank4-trailing"),
    pytest.param([(1, 2, 3, 32, 64)] * 2, 5, RM, id="rm-rank5-trailing"),
    # single input: concat short-circuits to a copy, so only the permute runs
    pytest.param([(5, 7)], -1, RM, id="rm-rank2-trailing-single-input"),
    # leading-axis route, tile (rank >= 2)
    pytest.param([(123, 259)] * 3, -1, TILE, id="tile-rank2-trailing-n3"),
    pytest.param([(57, 83)] * 7, -1, TILE, id="tile-rank2-trailing-n7"),
    pytest.param([(64, 32)] * 12, -1, TILE, id="tile-rank2-trailing-n12"),
    pytest.param([(8732, 1)] * 4, 2, TILE, id="tile-rank2-trailing-width1"),
    pytest.param([(8732, 2)] * 2, 2, TILE, id="tile-rank2-trailing-width2"),
    pytest.param([(2, 37, 64)] * 3, 3, TILE, id="tile-rank3-trailing"),
    pytest.param([(1, 23, 40, 64)] * 2, 4, TILE, id="tile-rank4-trailing"),
    pytest.param([(1, 5, 16, 16)] * 2, -1, TILE, id="tile-rank4-trailing-small"),
    pytest.param([(1, 2, 3, 32, 64)] * 2, 5, TILE, id="tile-rank5-trailing"),
]


def _expected_padded_shape(shape, layout):
    # Only stack outputs come through here, and every input has rank >= 1, so the shape has the two
    # tile-padded trailing dims.
    assert len(shape) >= 2
    if layout != TILE:
        return list(shape)
    padded = list(shape)
    for axis in (-2, -1):
        padded[axis] = -(-padded[axis] // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
    return padded


def _check_result(torch_result, ttnn_result, layout):
    ttnn_result_torch = ttnn.to_torch(ttnn_result)
    assert torch_result.shape == ttnn_result_torch.shape
    assert_equal(torch_result, ttnn_result_torch)
    # Both routes end in concat's default placement, DRAM, whatever the inputs' placement (the
    # permute on the leading-axis route keeps its input's placement).
    assert (
        ttnn_result.memory_config().buffer_type == ttnn.BufferType.DRAM
    ), "the stack docstring promises DRAM output; update stack_nanobind.cpp if concat's default changes"
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
        # compile-time arg and 32-bit dest path all change with the dtype. The rank-1 case under the
        # tile layout stays on unsqueeze-then-concat and covers that path's dtype handling instead.
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
    # Block floats stay on unsqueeze-then-concat at every rank: for rank >= 4 inputs the leading-axis
    # route's rank-5 permute would skip permute's bfloat8_b typecast, so the rank-4 case is the
    # regression case and the rank-2 case is coverage. PCC rather than equality because the golden
    # is torch.stack of the unquantized bfloat16 inputs.
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


@pytest.mark.parametrize(
    "input_shapes, dim, layout, expected_ops",
    [
        pytest.param([(1500,)] * 4, 1, RM, ["Concat", "Permute"], id="rm-rank1"),
        pytest.param([(8732, 1)] * 4, 2, TILE, ["Concat", "Permute"], id="tile-rank2"),
        pytest.param([(8732, 1)], 2, TILE, ["Reshape"], id="tile-single-input"),
    ],
)
def test_stack_trailing_axis_device_ops(device, input_shapes, dim, layout, expected_ops):
    # Values cannot tell the routes apart, so record the device operations: the leading-axis route
    # is one concat and one permute, while unsqueeze-then-concat starts with a reshape per input and,
    # for row-major, wraps the concat in transposes. A single tile input stays on the reshape.
    # Expected ops are name prefixes because an op can run as more than one device operation
    # (reshape picks ReshapeView or ReshapeCodegen by shape and arch); the route is what is asserted.
    torch_tensors = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    ttnn_tensors = [
        ttnn.from_torch(tensor, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        for tensor in torch_tensors
    ]
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        ttnn_result = ttnn.stack(ttnn_tensors, dim=dim)
    finally:
        captured = ttnn.graph.end_graph_capture()
    device_ops = [name for name in ttnn.graph.extract_calltrace(captured) if name.endswith("DeviceOperation")]
    assert len(device_ops) == len(expected_ops), device_ops
    assert all(op.startswith(prefix) for op, prefix in zip(device_ops, expected_ops)), device_ops
    _check_result(torch.stack(torch_tensors, dim=dim), ttnn_result, layout)


@pytest.mark.parametrize(
    "input_shape, dim, layout",
    [
        pytest.param((13600, 1), 2, RM, id="rm-last-dim-1"),
        pytest.param((1024, 1, 1), 3, TILE, id="tile-last-two-dims-1"),
    ],
)
def test_stack_single_input_view_runs_no_device_ops(device, input_shape, dim, layout):
    # One input whose trailing unsqueeze is a view: the stack is metadata plus concat's single-input
    # copy, a no-op for a DRAM input. The leading-axis route would spend a permute on it.
    torch_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    ttnn_tensor = ttnn.from_torch(torch_tensor, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        ttnn_result = ttnn.stack([ttnn_tensor], dim=dim)
    finally:
        captured = ttnn.graph.end_graph_capture()
    device_ops = [name for name in ttnn.graph.extract_calltrace(captured) if name.endswith("DeviceOperation")]
    assert device_ops == [], device_ops
    _check_result(torch.stack([torch_tensor], dim=dim), ttnn_result, layout)


def test_stack_trailing_axis_compiles_two_programs(device):
    # The device caches one program per distinct op and tensor spec; from_torch, to_torch and the
    # unsqueeze-at-0 view add none, so a fresh row-major shape must add exactly concat and permute and
    # a repeated shape must add nothing.
    device.disable_and_clear_program_cache()
    device.enable_program_cache()
    counter = device.cache_entries_counter

    def run_stack(width):
        torch_tensors = [torch.rand((width,), dtype=torch.bfloat16) for _ in range(4)]
        ttnn_tensors = [
            ttnn.from_torch(tensor, layout=RM, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            for tensor in torch_tensors
        ]
        counter.reset()
        with counter.measure():
            ttnn_result = ttnn.stack(ttnn_tensors, dim=1)
        assert_equal(torch.stack(torch_tensors, dim=1), ttnn.to_torch(ttnn_result))
        return counter.total

    assert run_stack(1444) == 2, "a fresh trailing-axis shape must compile exactly one concat and one permute"
    assert run_stack(1444) == 0, "same shapes must hit the cache"
    assert run_stack(1500) == 2, "a new trailing-axis shape must compile exactly one concat and one permute"


@pytest.mark.parametrize("dim", [2, -3], ids=["dim=rank+1", "dim=-(rank+2)"])
def test_stack_dim_out_of_range(device, expect_error, dim):
    ttnn_tensors = [ttnn.from_torch(torch.rand((8,), dtype=torch.bfloat16), layout=RM, device=device) for _ in range(2)]
    with expect_error(RuntimeError, "Stack dim"):
        ttnn.stack(ttnn_tensors, dim=dim)


@pytest.mark.parametrize("dim", [1, 0], ids=["trailing", "leading"])
def test_stack_zero_volume_rejected(device, expect_error, dim):
    # Both routes would otherwise reach concat's division by the padded last dim, a SIGFPE.
    ttnn_tensors = [ttnn.from_torch(torch.rand((0,), dtype=torch.bfloat16), layout=RM, device=device) for _ in range(4)]
    with expect_error(RuntimeError, "zero-volume"):
        ttnn.stack(ttnn_tensors, dim=dim)


def test_stack_trailing_axis_rejects_mixed_inputs_in_concat(device, expect_error):
    # The gate reads dtype and volume from the first input only; concat rejects a later input that
    # differs (shape in the composite, dtype in the device op's validate) before the leading-axis
    # route runs anything on device.
    bf16 = ttnn.from_torch(torch.rand((1444,), dtype=torch.bfloat16), layout=RM, device=device)
    f32 = ttnn.from_torch(torch.rand((1444,), dtype=torch.float32), layout=RM, device=device)
    with expect_error(RuntimeError, "same dtypes"):
        ttnn.stack([bf16, f32], dim=1)
    longer = ttnn.from_torch(torch.rand((1500,), dtype=torch.bfloat16), layout=RM, device=device)
    with expect_error(RuntimeError, "All dimensions must be the same size"):
        ttnn.stack([bf16, longer], dim=1)


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

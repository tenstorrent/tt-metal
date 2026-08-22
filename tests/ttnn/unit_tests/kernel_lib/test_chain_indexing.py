# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Index and broadcast axes for eltwise helpers.

    InputTileMapping   per-iter tile index
    Scalar        0
    Block         ht*Wt + wt
    Row           wt          (one tile per column)
    Col           ht          (one tile per row)
    TileAddressing    base + index

Golden changes if the wrong tile is read: a Row<->Col swap or dropped TileAddressing base fails PCC.
The broadcast case separately validates which row, column, or scalar is replicated within a tile.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

ADDRESSING_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/tile_addressing.cpp"
INDEX_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/index_2d.cpp"
STRIDED_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/strided_tile_range.cpp"
BCAST_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/eltwise/chain/axes/bcast_binary_add.cpp"
MODE = {"row": 2, "col": 1}


# =============================================================================
# TileAddressing — Block walker reading tiles [base, base+n). output[i] == input[base+i].
# =============================================================================
@pytest.mark.parametrize("base", [0, 3])
def test_tile_addressing_base(device, base):
    n = 4
    dt = ttnn.bfloat16
    total = base + n  # input holds base+n tiles; chain reads the last n.
    in_shape = [1, 1, 32, 32 * total]
    out_shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(in_shape, dt, device, seed=401)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    # Bulk reader stages all `total` tiles upfront -> size cb_in for total, cb_out for n.
    cbs = [lib.cb_descriptor(0, dt, total, core_grid), lib.cb_descriptor(16, dt, n, core_grid)]
    reader = lib.build_reader_kernel([tt_in], total, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel_rt(ADDRESSING_KERNEL, [n], [base], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)

    in_f = torch_in.to(torch.float32)
    golden = in_f[:, :, :, 32 * base : 32 * (base + n)]
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.debug(f"TileAddressing base={base} -> output[i]==input[base+i] | {msg}")
    assert pcc_ok, msg


# =============================================================================
# Row / Col inter-tile index — 2D grid add where B's tile is selected by its index mode.
# =============================================================================
def _run_index_2d(device, axis, Ht=2, Wt=4):
    dt = ttnn.bfloat16
    core_grid = lib.single_core_grid()
    a_shape = [1, 1, 32 * Ht, 32 * Wt]
    # Row: B is one tile-row (Wt tiles); Col: B is one tile-column (Ht tiles).
    b_shape = [1, 1, 32, 32 * Wt] if axis == "row" else [1, 1, 32 * Ht, 32]
    b_count = Wt if axis == "row" else Ht

    torch_a, tt_a = lib.make_input(a_shape, dt, device, seed=411)
    torch_b, tt_b = lib.make_input(b_shape, dt, device, seed=412)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(a_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [
        lib.cb_descriptor(0, dt, Ht * Wt, core_grid),
        lib.cb_descriptor(1, dt, b_count, core_grid),
        lib.cb_descriptor(16, dt, Ht * Wt, core_grid),
    ]
    reader = lib.build_reader_asym_kernel([tt_a, tt_b], [Ht * Wt, b_count], core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, Ht * Wt, core_grid)
    compute = lib.build_compute_kernel(INDEX_KERNEL, [Ht, Wt, MODE[axis]], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)

    a_f = torch_a.to(torch.float32)
    b_f = torch_b.to(torch.float32)
    if axis == "row":  # B (one tile-row) repeated down all Ht tile-rows
        golden = a_f + b_f.repeat(1, 1, Ht, 1)
    else:  # B (one tile-col) repeated across all Wt tile-cols
        golden = a_f + b_f.repeat(1, 1, 1, Wt)
    return golden, ttnn.to_torch(output).to(torch.float32)


def test_index_2d_axes_are_correct_and_distinct(device):
    """Run each axis once, check both goldens, then prove the outputs discriminate an axis swap."""
    results = {axis: _run_index_2d(device, axis) for axis in ("row", "col")}
    for axis, (golden, out) in results.items():
        pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
        logger.debug(f"index axis={axis} | {msg}")
        assert pcc_ok, msg

    col_golden, _ = results["col"]
    _, row_out = results["row"]
    ok_cross, msg = comp_pcc(col_golden, row_out, 0.99)
    logger.debug(f"index cross-check: ROW-out vs COL-golden (expect low) | {msg}")
    assert not ok_cross, "ROW-index output matched COL golden — a Row<->Col index swap would slip through."


def test_strided_tile_range(device):
    Ht, Wt = 2, 2
    input_stride, output_stride = 4, 5
    input_base, output_base = 1, 2
    dt = ttnn.bfloat16
    core_grid = lib.single_core_grid()

    in_shape = [1, 1, 32 * Ht, 32 * input_stride]
    out_shape = [1, 1, 32 * Ht, 32 * output_stride]
    torch_in, tt_in = lib.make_input(in_shape, dt, device, seed=421)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(out_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )

    cbs = [
        lib.cb_descriptor(0, dt, Ht * input_stride, core_grid),
        lib.cb_descriptor(16, dt, Ht * output_stride, core_grid),
    ]
    reader = lib.build_reader_kernel([tt_in], Ht * input_stride, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, Ht * output_stride, core_grid)
    compute = lib.build_compute_kernel(
        STRIDED_KERNEL, [Ht, Wt, input_stride, output_stride, input_base, output_base], core_grid
    )

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)
    out = ttnn.to_torch(output).to(torch.float32)

    golden_region = torch_in.to(torch.float32)[:, :, :, 32 * input_base : 32 * (input_base + Wt)]
    output_region = out[:, :, :, 32 * output_base : 32 * (output_base + Wt)]
    pcc_ok, msg = comp_pcc(golden_region, output_region, lib.pcc_threshold([dt]))
    logger.debug(f"strided tile range | {msg}")
    assert pcc_ok, msg


BCAST_DIM = {"row": 2, "col": 1, "scalar": 3}


def _broadcast_operand(torch_b, axis):
    if axis == "row":
        return torch_b[:, :, 0:1, :].expand_as(torch_b)
    if axis == "col":
        return torch_b[:, :, :, 0:1].expand_as(torch_b)
    return torch_b[:, :, 0:1, 0:1].expand_as(torch_b)


def _run_bcast_add(device, axis):
    shape = [1, 1, 32, 32]
    dt = ttnn.bfloat16
    core_grid = lib.single_core_grid()
    torch_a, tt_a = lib.make_input(shape, dt, device, seed=201)
    torch_b, tt_b = lib.make_input(shape, dt, device, seed=202)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [
        lib.cb_descriptor(0, dt, 2, core_grid),
        lib.cb_descriptor(1, dt, 2, core_grid),
        lib.cb_descriptor(16, dt, 2, core_grid),
    ]
    program = ttnn.ProgramDescriptor(
        kernels=[
            lib.build_reader_kernel([tt_a, tt_b], 1, core_grid),
            lib.build_writer_1out_kernel(tt_out, 1, core_grid),
            lib.build_compute_kernel(BCAST_KERNEL, [1, BCAST_DIM[axis]], core_grid),
        ],
        semaphores=[],
        cbs=cbs,
    )
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)
    golden = torch_a.to(torch.float32) + _broadcast_operand(torch_b, axis).to(torch.float32)
    return golden, ttnn.to_torch(output).to(torch.float32)


def test_bcast_axes_are_correct_and_distinct(device):
    """Validate all intra-tile broadcast axes and ensure row/column are distinguishable."""
    results = {axis: _run_bcast_add(device, axis) for axis in ("row", "col", "scalar")}
    for axis, (golden, out) in results.items():
        pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
        logger.debug(f"bcast add axis={axis} | {msg}")
        assert pcc_ok, msg

    col_golden, _ = results["col"]
    _, row_out = results["row"]
    ok_cross, msg = comp_pcc(col_golden, row_out, 0.99)
    logger.debug(f"cross-check: ROW-out vs COL-golden pcc (expect low) | {msg}")
    assert not ok_cross, "ROW output matched COL golden; an axis swap would slip through."

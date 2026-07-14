# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Operand-kind & broadcast tests for eltwise_chain.

Broadcast picks which part of the B tile is replicated before A (op) B. A swapped axis silently
corrupts (no crash/hang), so the defense is a RANDOM B: a row B[0][c], a column B[r][0] and a scalar
B[0][0] are all different, so a ROW<->COL swap fails PCC instead of matching by accident.
Each case builds A + bcast(B) on one 32x32 tile (caller owns init_bcast; chain owns per-element work).
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/bcast_binary_add.cpp"

# BroadcastDim compile-time arg (ckernel values) -> the torch reduction that produces the
# broadcast operand from a full tile.
DIM = {"row": 2, "col": 1, "scalar": 3}


def _bcast_b(torch_b, axis):
    """Replicate B the way the hardware bcast does, so the golden matches."""
    if axis == "row":  # replicate row 0 down all rows
        return torch_b[:, :, 0:1, :].expand_as(torch_b)
    if axis == "col":  # replicate col 0 across all cols
        return torch_b[:, :, :, 0:1].expand_as(torch_b)
    return torch_b[:, :, 0:1, 0:1].expand_as(torch_b)  # scalar: element [0][0]


def _run_bcast_add(device, axis):
    """Returns (golden, output) for A + bcast(B) on a single tile."""
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
    reader = lib.build_reader_kernel([tt_a, tt_b], 1, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, 1, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [1, DIM[axis]], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_out], program)

    golden = torch_a.to(torch.float32) + _bcast_b(torch_b, axis).to(torch.float32)
    return golden, ttnn.to_torch(output).to(torch.float32)


# =============================================================================
# Each broadcast axis produces its own golden.
# =============================================================================
@pytest.mark.parametrize("axis", ["row", "col", "scalar"])
def test_bcast_add_axis(device, axis):
    golden, out = _run_bcast_add(device, axis)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
    logger.info(f"bcast add axis={axis} | {msg}")
    assert pcc_ok, msg


# =============================================================================
# Negative cross-check — a ROW result must NOT match the COL golden (and vice versa).
# This proves the test discriminates the axis rather than passing vacuously: if the helper
# ever swapped ROW<->COL, test_bcast_add_axis would catch it (these asserts confirm the
# goldens are genuinely different on the random input we feed).
# =============================================================================
def test_bcast_axis_goldens_are_distinct(device):
    row_golden, row_out = _run_bcast_add(device, "row")
    col_golden, _ = _run_bcast_add(device, "col")

    # The actual ROW output matches the ROW golden ...
    ok_match, _ = comp_pcc(row_golden, row_out, lib.pcc_threshold([ttnn.bfloat16]))
    assert ok_match, "ROW output should match ROW golden"

    # ... but NOT the COL golden — confirming a swap would be caught.
    ok_cross, msg = comp_pcc(col_golden, row_out, 0.99)
    logger.info(f"cross-check: ROW-out vs COL-golden pcc (expect low) | {msg}")
    assert not ok_cross, (
        "ROW output unexpectedly matched the COL golden — the test cannot tell the axes apart; "
        "a ROW<->COL swap would slip through."
    )

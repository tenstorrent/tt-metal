# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
G3 (indexing axis) — inter-tile index selection for eltwise_chain.

Coverage spec: ttnn/cpp/ttnn/kernel_lib/docs/eltwise_helper_test_coverage.html (group G3, the
OperandKind *index* contract + TileOffset). Distinct from test_chain_operand_kind.py, which tests
the BroadcastDim *intra-tile* replication. Here the variable is WHICH TILE an operand reads:

    OperandKind   per-iter tile index   (eltwise_chain.inl:1855)
    Scalar        0
    Block         ht*Wt + wt
    Row           wt          (one tile per column)
    Col           ht          (one tile per row)
    TileOffset    base + index

Each test makes the tile index the only thing that can be wrong, with a golden that changes if the
wrong tile is read. A Row<->Col index swap, or a dropped TileOffset base, fails PCC.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

OFFSET_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/tile_offset.cpp"
INDEX_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/index_2d.cpp"
MODE = {"row": 2, "col": 1}


# =============================================================================
# TileOffset — Block walker reading tiles [base, base+n). output[i] == input[base+i].
# =============================================================================
@pytest.mark.parametrize("base", [0, 2, 3])
def test_tile_offset_base(device, base):
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
    compute = lib.build_compute_kernel_rt(OFFSET_KERNEL, [n], [base], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_in, tt_out], program)

    in_f = torch_in.to(torch.float32)
    golden = in_f[:, :, :, 32 * base : 32 * (base + n)]
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"TileOffset base={base} -> output[i]==input[base+i] | {msg}")
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


@pytest.mark.parametrize("axis", ["row", "col"])
def test_index_2d_axis(device, axis):
    golden, out = _run_index_2d(device, axis)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([ttnn.bfloat16]))
    logger.info(f"G3 index axis={axis} | {msg}")
    assert pcc_ok, msg


def test_index_2d_axes_are_distinct(device):
    """Discrimination check: a ROW-index result must NOT match the COL golden — proves the test
    actually distinguishes which tile is read, so a Row<->Col index swap would be caught."""
    row_golden, row_out = _run_index_2d(device, "row")
    col_golden, _ = _run_index_2d(device, "col")
    ok_match, _ = comp_pcc(row_golden, row_out, lib.pcc_threshold([ttnn.bfloat16]))
    assert ok_match, "ROW-index output should match ROW golden"
    ok_cross, msg = comp_pcc(col_golden, row_out, 0.99)
    logger.info(f"G3 index cross-check: ROW-out vs COL-golden (expect low) | {msg}")
    assert not ok_cross, "ROW-index output matched COL golden — a Row<->Col index swap would slip through."

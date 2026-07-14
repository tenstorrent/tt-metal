# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Coverage for chain element types that the other suites don't exercise.

  - DestReuseBinary: feeds the DEST result back as an FPU operand (DEST->srcA/srcB) instead of a
    second CB read. out = (A + B) * C.

Fill / Rand (no-CB-input elements with special init) and ternary/quaternary SFPU are tracked as
follow-up.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

DEST_REUSE = "ttnn/cpp/ttnn/kernel_lib/tests/axes/dest_reuse.cpp"
DEST_REUSE_PARAM = "ttnn/cpp/ttnn/kernel_lib/tests/axes/dest_reuse_param.cpp"
TERNARY_WHERE = "ttnn/cpp/ttnn/kernel_lib/tests/axes/ternary_where.cpp"

# reuse selector -> name; op selector -> (name, torch fn applied as `lhs op rhs`)
_REUSE = {0: "DEST_TO_SRCA", 1: "DEST_TO_SRCB"}
_OP = {0: ("add", lambda x, y: x + y), 1: ("sub", lambda x, y: x - y), 2: ("mul", lambda x, y: x * y)}


def test_dest_reuse_binary(device):
    """DestReuseBinary: out = (A + B) * C, with (A+B) threaded through DEST into the multiply."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    ta, tt_a = lib.make_input(shape, dt, device, seed=1101)
    tb, tt_b = lib.make_input(shape, dt, device, seed=1102)
    tc, tt_c = lib.make_input(shape, dt, device, seed=1103)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(i, dt, 2, core_grid) for i in (0, 1, 2)] + [lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_a, tt_b, tt_c], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(DEST_REUSE, [n], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_c, tt_out], program)

    golden = ((ta + tb) * tc).to(torch.float32)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"DestReuseBinary (A+B)*C | {msg}")
    assert pcc_ok, msg


@pytest.mark.parametrize("reuse", [0, 1], ids=["SRCA", "SRCB"])
@pytest.mark.parametrize("op", [0, 1, 2], ids=["add", "sub", "mul"])
def test_dest_reuse_matrix(device, reuse, op):
    """DestReuseBinary reuse-direction x op. Stage1 D0=A+B; stage2 routes DEST per ReuseType:
    SRCA -> (A+B) op C, SRCB -> C op (A+B). The Sub rows differ between directions, proving DEST
    is routed to the correct unpack lane."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    ta, tt_a = lib.make_input(shape, dt, device, seed=1301)
    tb, tt_b = lib.make_input(shape, dt, device, seed=1302)
    tc, tt_c = lib.make_input(shape, dt, device, seed=1303)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(i, dt, 2, core_grid) for i in (0, 1, 2)] + [lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_a, tt_b, tt_c], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(DEST_REUSE_PARAM, [n, reuse, op], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_c, tt_out], program)

    dest = (ta + tb).to(torch.float32)
    c = tc.to(torch.float32)
    _, fn = _OP[op]
    golden = fn(dest, c) if reuse == 0 else fn(c, dest)  # SRCA: DEST op C ; SRCB: C op DEST
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"DestReuse {_REUSE[reuse]} {_OP[op][0]} | {msg}")
    assert pcc_ok, f"{_REUSE[reuse]} {_OP[op][0]}: {msg}"


def test_ternary_where(device):
    """Where ternary: out = where(cond != 0, a, b). cond is a 0/1 mask so both branches are taken."""
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    # cond as a 0/1 mask so where exercises both the 'a' and 'b' branches.
    torch.manual_seed(1201)
    cond_f = (torch.randn(shape) > 0).to(torch.float32)
    tt_cond = ttnn.from_torch(
        cond_f.to(torch.bfloat16),
        dtype=dt,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    ta, tt_a = lib.make_input(shape, dt, device, seed=1202)
    tb, tt_b = lib.make_input(shape, dt, device, seed=1203)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(i, dt, 2, core_grid) for i in (0, 1, 2)] + [lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_cond, tt_a, tt_b], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(TERNARY_WHERE, [n], core_grid)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_cond, tt_a, tt_b, tt_out], program)

    golden = torch.where(cond_f != 0, ta.to(torch.float32), tb.to(torch.float32))
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"Where ternary | {msg}")
    assert pcc_ok, msg

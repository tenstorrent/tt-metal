# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
DEST multi-slot independence for eltwise_chain.

dest_multislot.cpp keeps two independent results live in DEST (D0=A+B, D1=C+E) inside one
tile_regs window, then combines them (D0=D0+D1) and packs. out = A+B+C+E. If the second FPU add
stomped D0, the result would collapse to (A+B) or (C+E) — so the golden directly verifies slot
independence. Runs under fp32_dest_acc {False, True} (the halved-DEST-capacity addressing path).
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/dest_multislot.cpp"


@pytest.mark.parametrize("fp32_dest_acc_en", [False, True])
def test_dest_multislot_independence(device, fp32_dest_acc_en):
    n = 4
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    ta, tt_a = lib.make_input(shape, dt, device, seed=801)
    tb, tt_b = lib.make_input(shape, dt, device, seed=802)
    tc, tt_c = lib.make_input(shape, dt, device, seed=803)
    te, tt_e = lib.make_input(shape, dt, device, seed=804)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(i, dt, 2, core_grid) for i in (0, 1, 2, 3)] + [lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_a, tt_b, tt_c, tt_e], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n], core_grid, fp32_dest_acc_en=fp32_dest_acc_en)

    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    output = ttnn.generic_op([tt_a, tt_b, tt_c, tt_e, tt_out], program)

    golden = (ta + tb + tc + te).to(torch.float32)
    out = ttnn.to_torch(output).to(torch.float32)
    pcc_ok, msg = comp_pcc(golden, out, lib.pcc_threshold([dt]))
    logger.info(f"DEST multi-slot fp32_acc={fp32_dest_acc_en} | {msg}")
    assert pcc_ok, f"D0/D1 not independent (fp32_acc={fp32_dest_acc_en}): {msg}"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
ComputeMode::Skip performance-analysis knob for eltwise_chain.

Skip makes the chain emit ONLY the CB lifecycle (input wait/pop, output reserve/push) plus the
tile_regs dst-sync window, and skip ALL init + reconfig + compute (unpack, exp_tile, pack_tile).
The CB counts are byte-for-byte identical to Run, so the reader/writer dataflow kernels are
unmodified and still handshake — the kernel must NOT hang.

This test runs a streaming exp(x) chain compiled with ComputeMode::Skip and asserts two things:
  1. It runs to completion (reaching the assert at all proves the CB handshake stayed intact — a
     broken handshake would hang and run_safe_pytest.sh would report it).
  2. Output does NOT match exp(x) (low PCC) — proving init + compute were actually elided.

The Run-mode counterpart (same chain producing exp(x) with PCC ~1.0) is covered by
test_chain_setup_owner.py and test_chain_blocking.py.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/skip_compute_exp.cpp"


@pytest.mark.parametrize("n", [8, 32])
def test_skip_compute_skips_compute(device, n):
    """ComputeMode::Skip: chain runs (no hang) but output is garbage (compute skipped)."""
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    core_grid = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=777)
    tt_out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG)
    cbs = [lib.cb_descriptor(0, dt, 2, core_grid), lib.cb_descriptor(16, dt, 2, core_grid)]
    reader = lib.build_reader_kernel([tt_in], n, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, n, core_grid)
    compute = lib.build_compute_kernel(KERNEL, [n], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    # Reaching the line after generic_op proves the CB handshake did not deadlock.
    output = ttnn.generic_op([tt_in, tt_out], program)
    golden = torch.exp(torch_in.to(torch.float32))
    out = ttnn.to_torch(output).to(torch.float32)

    ok_exp, msg = comp_pcc(golden, out, 0.99)
    logger.info(f"ComputeMode::Skip n={n} | ran without hang | PCC(out, exp(x))={msg}")
    assert not ok_exp, f"Skip output matched exp(x) (PCC {msg}) — compute was NOT skipped; the knob is a no-op."

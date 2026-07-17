# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Skip-compute (CKL_ELTWISE_CHAIN_SKIP_COMPUTE) performance-analysis knob for eltwise_chain.

Skip makes the chain emit ONLY the CB lifecycle (input wait/pop, output reserve/push) plus the
tile_regs dst-sync window, and skip ALL init + reconfig + compute (unpack, exp_tile, pack_tile).
The CB counts are byte-for-byte identical to a normal run, so the reader/writer dataflow kernels are
unmodified and still handshake — the kernel must NOT hang.

This test runs a streaming exp(x) chain compiled with the skip-compute macro set to 1 and asserts
two things:
  1. It runs to completion (reaching the assert at all proves the CB handshake stayed intact — a
     broken handshake would hang and run_safe_pytest.sh would report it).
  2. Output does NOT match exp(x) (low PCC) — proving init + compute were actually elided.

The normal-run counterpart (same chain producing exp(x) with PCC ~1.0) is covered by
test_chain_setup_owner.py and test_chain_blocking.py.
"""

import torch
import pytest
import ttnn
from loguru import logger
from tests.ttnn.utils_for_testing import comp_pcc
import tests.ttnn.unit_tests.kernel_lib.chain_test_lib as lib

KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/axes/skip_compute_exp.cpp"
DEST_ACCUM_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/dest_accumulation/skip_compute_dest_accum.cpp"
L1_ACCUM_KERNEL = "ttnn/cpp/ttnn/kernel_lib/tests/l1_accumulation/skip_compute_l1_accum.cpp"


@pytest.mark.parametrize("n", [8, 32])
def test_skip_compute_skips_compute(device, n):
    """Skip-compute (ordinary walk): chain runs (no hang) but output is garbage (compute skipped)."""
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
    logger.info(f"skip-compute n={n} | ran without hang | PCC(out, exp(x))={msg}")
    assert not ok_exp, f"Skip output matched exp(x) (PCC {msg}) — compute was NOT skipped; the knob is a no-op."


@pytest.mark.parametrize("block_size", [1, 8])
@pytest.mark.parametrize("caller_managed", [False, True])
def test_skip_compute_skips_dest_accumulation(device, block_size, caller_managed):
    """Skip-compute (DEST-accumulation walk): the sticky-D0 reduction runs (no hang) but the packed
    output is garbage — proving the knob covers the DEST-accumulation loop too, not just the ordinary
    walk. Golden is the real (local + remote) row reduction; skip must NOT reproduce it."""
    n = 8
    num_outputs = 3
    total_input_tiles = n * num_outputs
    dt = ttnn.bfloat16
    input_shape = [1, 1, 32, 32 * total_input_tiles]
    output_shape = [1, 1, 32, 32 * num_outputs]
    core_grid = lib.single_core_grid()

    # Seeds unique to this test (NOT the functional DEST test's 1701/1702): with the compute
    # skipped, the writer publishes whatever uninitialized/stale L1 the pack never overwrote, so
    # reusing another test's seeds could alias a DRAM address still holding that test's real
    # reduction and mask the skip. Unique seeds keep the golden uncorrelated with any stale tile.
    torch_local, tt_local = lib.make_input(input_shape, dt, device, seed=90011)
    torch_remote, tt_remote = lib.make_input(input_shape, dt, device, seed=90022)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(output_shape), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    cbs = [
        lib.cb_descriptor(0, dt, total_input_tiles, core_grid),
        lib.cb_descriptor(1, dt, total_input_tiles, core_grid),
        lib.cb_descriptor(16, dt, num_outputs, core_grid),
    ]
    reader = lib.build_reader_kernel([tt_local, tt_remote], total_input_tiles, core_grid)
    writer = lib.build_writer_1out_kernel(tt_out, num_outputs, core_grid)
    compute = lib.build_compute_kernel(DEST_ACCUM_KERNEL, [n, block_size, int(caller_managed), num_outputs], core_grid)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    # Reaching the line after generic_op proves the CB handshake did not deadlock.
    out = ttnn.to_torch(ttnn.generic_op([tt_local, tt_remote, tt_out], program)).to(torch.float32)

    local_tiles = torch.stack(torch_local.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    remote_tiles = torch.stack(torch_remote.to(torch.float32).split(32, dim=-1)).reshape(num_outputs, n, 1, 1, 32, 32)
    reduced = (local_tiles + remote_tiles).sum(dim=1)
    golden = torch.cat([reduced[i] for i in range(num_outputs)], dim=-1)

    ok_reduce, msg = comp_pcc(golden, out, 0.99)
    logger.info(
        f"skip-compute DEST-accum block={block_size} caller_managed={caller_managed} | "
        f"ran without hang | PCC(out, reduction)={msg}"
    )
    assert not ok_reduce, f"Skip output matched the reduction (PCC {msg}) — DEST-accum compute was NOT skipped."


@pytest.mark.parametrize("caller_managed", [False, True])
def test_skip_compute_skips_l1_accumulation(device, caller_managed):
    """Skip-compute (L1-accumulation walk): the seed-first accumulate + copy runs (no hang) but the
    output is garbage — proving the knob covers the L1-accumulation loop too. Golden is the real
    n-tile running sum; skip must NOT reproduce it. Private seed (see the note above)."""
    n = 8
    dt = ttnn.bfloat16
    shape = [1, 1, 32, 32 * n]
    cg = lib.single_core_grid()

    torch_in, tt_in = lib.make_input(shape, dt, device, seed=90033, scale=0.125, bias=0.0)
    tt_out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, 32, 32]), dt, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
    )
    cbs = [lib.cb_descriptor(0, dt, 2, cg), lib.cb_descriptor(15, dt, 1, cg), lib.cb_descriptor(16, dt, 2, cg)]
    reader = lib.build_reader_kernel([tt_in], n, cg)
    writer = lib.build_writer_1out_kernel(tt_out, 1, cg)
    compute = lib.build_compute_kernel(L1_ACCUM_KERNEL, [n, int(caller_managed)], cg)
    program = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)

    # Reaching the line after generic_op proves the CB handshake did not deadlock.
    out = ttnn.to_torch(ttnn.generic_op([tt_in, tt_out], program)).to(torch.float32)
    golden = torch.stack(torch_in.to(torch.float32).split(32, dim=-1)).sum(dim=0)  # n-tile running sum

    ok_sum, msg = comp_pcc(golden, out, 0.99)
    logger.info(f"skip-compute L1-accum caller_managed={caller_managed} | ran without hang | PCC(out, sum)={msg}")
    assert not ok_sum, f"Skip output matched the accumulation (PCC {msg}) — L1-accum compute was NOT skipped."

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Per-transaction-ID waits must not return before the transaction they wait for has finished.

The per-trid NIU counters are incremented when NOC_CMD_CTRL is written. A counter read issued right
after that store can be emitted first, see 0, and let the wait return before the transaction is even
counted. The waits now wait on noc_cmd_buf_ready() (a read of NOC_CMD_CTRL) before polling the counter.
"""

import os

import pytest
import torch
import ttnn

KERNEL = "tests/ttnn/unit_tests/operations/debug/kernels/noc_trid_waits/trid_waits.cpp"
MODES = {"read_barrier": 0, "write_barrier": 1, "write_flushed": 2}
IMPLS = {"api": 0, "unordered_poll": 1}
SCRATCH_WORDS = 32 * 1024 // 4 + 64

pytestmark = pytest.mark.skipif(
    os.getenv("TT_METAL_SIMULATOR") is not None, reason="NoC counter ordering is a hardware property"
)


def run_trid_waits(device, mode, impl, iters=20000, nbytes=4096, uncounted_issue=False):
    """Returns (iterations, early returns) for one wait under test."""
    local, remote = ttnn.CoreCoord(0, 0), ttnn.CoreCoord(1, 0)
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(local, remote)])
    rows = SCRATCH_WORDS // 4
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [rows, 4], ttnn.ShardOrientation.ROW_MAJOR),
    )
    # Every shard holds its own word index, so the remote buffer has a known pattern.
    pattern = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4).repeat(2, 1)
    scratch = ttnn.from_torch(
        pattern, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )
    dummy = ttnn.from_torch(torch.zeros(32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    remote_virt = device.worker_core_from_logical_core(remote)

    runtime_args = ttnn.RuntimeArgs()
    runtime_args[local.x][local.y] = [scratch.buffer_address(), remote_virt.x, remote_virt.y]
    kernel = ttnn.KernelDescriptor(
        kernel_source=KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=ttnn.CoreRangeSet([ttnn.CoreRange(local, local)]),
        compile_time_args=[iters, MODES[mode], IMPLS[impl], nbytes, int(uncounted_issue)],
        runtime_args=runtime_args,
        config=ttnn.DataMovementConfigDescriptor(processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_1),
    )
    output = ttnn.generic_op([dummy, scratch], ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[]))
    words = (ttnn.to_torch(output).to(torch.int64) & 0xFFFFFFFF).reshape(-1)
    done, early = words[32 * 1024 // 4].item(), words[32 * 1024 // 4 + 1].item()
    assert done == iters
    return iters, early


@pytest.mark.parametrize(
    "mode, uncounted_issue",
    [
        ("read_barrier", False),
        ("write_barrier", False),
        ("write_barrier", True),
        ("write_flushed", False),
        ("write_flushed", True),
    ],
)
def test_noc_trid_waits_do_not_return_early(device, mode, uncounted_issue):
    iters, early = run_trid_waits(device, mode, "api", uncounted_issue=uncounted_issue)
    assert early == 0, f"{mode}: returned before the transaction finished in {early} of {iters} iterations"


@pytest.mark.parametrize(
    "mode, uncounted_issue",
    [
        ("read_barrier", False),
        # The write issue wrapper updates the software counters after writing NOC_CMD_CTRL, which in
        # practice gives the store time to drain; issuing without counter updates (as manually tracked
        # trid writes do) puts the counter poll right behind the store, like the read path.
        ("write_barrier", True),
        ("write_flushed", True),
    ],
)
def test_unordered_trid_wait_returns_early(device, mode, uncounted_issue):
    # The waits as they were before this fix: poll the per-trid counter straight after issuing. The
    # counter read overtakes the NOC_CMD_CTRL store and the wait returns before the transaction is
    # counted. This also proves the detector works for test_noc_trid_waits_do_not_return_early.
    iters, early = run_trid_waits(device, mode, "unordered_poll", uncounted_issue=uncounted_issue)
    assert early > 0, f"{mode}: the unordered counter poll never returned early in {iters} iterations"

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stress the linked-multicast rule: while a linked multicast chain is open, the initiating core's NoC must carry
only multicasts (the next transaction from the command buffer must be a multicast on the same VC, the core's other
command buffers stay idle). One sender core runs MIMO_LMC_ITERS chains of 4 linked 8 KB multicasts (the last
unlinked) into a 4 x 4 rectangle on NOC0 and breaks the rule per mode:

  legal         nothing else (control)
  write_same    a unicast write inside every open chain (the multicast's own command buffer)
  atomic_same   a unicast semaphore increment inside every open chain (the atomic command buffer)
  read_same     a unicast read inside every open chain (the read command buffer)
  write_other   legal chains on BRISC while NCRISC hammers unicast writes on the same NoC
  read_other    legal chains on BRISC while NCRISC hammers unicast reads on the same NoC

A hang shows up as run_safe_pytest's dispatch timeout; otherwise every receiver's buffer and the unicast target are
checked. Run one mode per invocation (a hang resets the device): -k <mode>.
"""

import os

import pytest
import torch
from loguru import logger

import ttnn

KDIR = "models/demos/mimo_v2_d_p/tests/perf/kernels/noc_probe"
MODES = ["legal", "write_same", "atomic_same", "read_same", "write_other", "read_other"]
ITERS = int(os.environ.get("MIMO_LMC_ITERS", "20000"))
CHAIN, BYTES = 4, 8192
ROWS = BYTES // 64  # bf16 rows of 32 per BYTES


def _l1(device, core, rows, host=None):
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)]) if isinstance(core, ttnn.CoreCoord) else core
    n = crs.num_cores()
    t = host if host is not None else torch.zeros(n * rows, 32)
    return ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(crs, (rows, 32), ttnn.ShardOrientation.ROW_MAJOR),
        ),
    )


@pytest.mark.timeout(600)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
@pytest.mark.parametrize("mode", MODES)
def test_mcast_linked_rule(device, mode):
    sender, target = ttnn.CoreCoord(1, 1), ttnn.CoreCoord(9, 1)
    lo, hi = ttnn.CoreCoord(2, 2), ttnn.CoreCoord(5, 5)
    rect = ttnn.CoreRangeSet([ttnn.CoreRange(lo, hi)])
    phys = lambda c: device.worker_core_from_logical_core(c)
    pk = lambda c: (phys(c).x << 16) | phys(c).y
    torch.manual_seed(0)
    pattern = torch.randn(ROWS, 32)
    src = _l1(device, sender, ROWS, pattern)
    land = _l1(device, sender, ROWS)
    dst = _l1(device, rect, CHAIN * ROWS)
    uni = _l1(device, target, ROWS)
    same = {"legal": 0, "write_same": 1, "atomic_same": 2, "read_same": 3}.get(mode, 0)
    rt = ttnn.RuntimeArgs()
    rt[sender.x][sender.y] = [
        src.buffer_address(),
        dst.buffer_address(),
        pk(lo),
        pk(hi),
        rect.num_cores(),
        pk(target),
        uni.buffer_address(),
        land.buffer_address(),
    ]
    one = ttnn.CoreRangeSet([ttnn.CoreRange(sender, sender)])
    dm = lambda proc: ttnn.DataMovementConfigDescriptor(processor=proc, noc=ttnn.NOC.NOC_0)
    FP = ttnn.KernelDescriptor.SourceType.FILE_PATH
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=f"{KDIR}/mcast_linked_sender.cpp",
            source_type=FP,
            core_ranges=one,
            compile_time_args=[same, ITERS, CHAIN, BYTES],
            runtime_args=rt,
            config=dm(ttnn.DataMovementProcessor.RISCV_0),
        )
    ]
    if mode.endswith("_other"):  # the sender core's other RISC on the same NoC
        hr = ttnn.RuntimeArgs()
        hr[sender.x][sender.y] = [
            (src if mode == "write_other" else land).buffer_address(),
            pk(target),
            uni.buffer_address(),
        ]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=f"{KDIR}/unicast_hammer.cpp",
                source_type=FP,
                core_ranges=one,
                compile_time_args=[1 if mode == "write_other" else 2, 4 * ITERS, BYTES],
                runtime_args=hr,
                config=dm(ttnn.DataMovementProcessor.RISCV_1),
            )
        )
    program = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
    for launch in range(int(os.environ.get("MIMO_LMC_LAUNCHES", "3"))):
        ttnn.generic_op([src, land, dst, uni], program)
        ttnn.synchronize_device(device)
        logger.info(f"{mode}: launch {launch} done ({ITERS} chains of {CHAIN} x {BYTES} B)")
    got = ttnn.to_torch(dst).float().reshape(rect.num_cores(), CHAIN, ROWS, 32)
    want = pattern.to(torch.bfloat16).float()
    bad = [(c, k) for c in range(rect.num_cores()) for k in range(CHAIN) if not torch.equal(got[c, k], want)]
    assert not bad, f"{mode}: multicast data wrong at (receiver, piece) {bad[:8]} of {len(bad)}"
    if mode in ("write_same", "write_other"):
        assert torch.equal(ttnn.to_torch(uni).float(), want), f"{mode}: unicast target data wrong"
    if mode in ("read_same", "read_other"):
        assert torch.equal(ttnn.to_torch(land).float(), torch.zeros(ROWS, 32)), f"{mode}: read landing wrong"
    logger.info(f"{mode}: no hang, all {rect.num_cores()} receivers x {CHAIN} pieces correct")

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Multicast cost on Blackhole: S senders (logical columns 0..6) each multicast N x B bytes into a destination
rectangle of the right block (logical x 8..11 or one column); reports us and GB/s delivered per destination."""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import aiclk_mhz, crs, kpath, read_stamps, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, rect_noc, sharded_l1


def run_mcast(device, senders, dests, N, B, flag, barrier):
    st = stamps_tensor(device, senders)
    buf = sharded_l1(device, senders + dests, [16 * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)  # 32 KB a core
    r = rect_noc(device, dests)
    rt = ttnn.RuntimeArgs()
    for i, (x, y) in enumerate(senders):
        rt[x][y] = [(i * B) % (32768 - B + 1) // 16 * 16]
    k = ttnn.KernelDescriptor(
        kernel_source=kpath("mcast_bench.cpp"),
        core_ranges=crs(senders),
        compile_time_args=[N, B] + r + [len(dests), buf.buffer_address(), buf.buffer_address(), buf.buffer_address()]
        + [int(flag), int(barrier), st.buffer_address()],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
        ),
    )
    prog = ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[])
    ttnn.generic_op([buf, st], prog)
    ttnn.synchronize_device(device)
    s = read_stamps(st, len(senders))
    t0 = int(s[:, 0].min())
    t1 = int(((s[:, 1] - t0) & 0xFFFFFFFF).max())
    clk = aiclk_mhz(0)
    us = t1 / clk
    gbs = len(senders) * N * B / (us * 1e3)
    return us, gbs, t1 / (len(senders) * N)


@pytest.mark.parametrize("ns", [1, 10, 80])
@pytest.mark.parametrize("B", [4096, 40960 // 2])
@pytest.mark.parametrize("dst", ["rect40", "col10"])
@pytest.mark.parametrize("flag,barrier", [(False, False), (False, True), (True, True)])
def test_mcast(device, ns, B, dst, flag, barrier):
    left = [(x, y) for x in range(7) for y in range(10)] + [(7, y) for y in range(10)]
    senders = left[:ns]
    dests = [(x, y) for x in range(8, 12) for y in range(10)] if dst == "rect40" else [(8, y) for y in range(10)]
    N = max(4, (2 * 1024 * 1024) // (B * ns))
    us, gbs, cyc = run_mcast(device, senders, dests, N, B, flag, barrier)
    logger.info(
        f"MCAST senders {ns} dst {dst} B {B} flag {int(flag)} barrier {int(barrier)}: {us:.1f} us, "
        f"{gbs:.1f} GB/s into every destination, {cyc:.0f} cyc per multicast (all senders)"
    )

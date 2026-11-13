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


def run_mcast(device, senders, dests, N, B, flag, barrier, nocs=None, loopback=False):
    """nocs: per sender 0 / 1 (NOC1 senders run on NCRISC; their rectangle is passed end-first)."""
    nocs = nocs or [0] * len(senders)
    allc = list(dict.fromkeys(senders + dests))
    st = stamps_tensor(device, allc)
    buf = sharded_l1(device, allc, [64 * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)  # 128 KB a core
    r = rect_noc(device, dests)
    kernels = []
    for nn in (0, 1):
        ss = [c for c, n in zip(senders, nocs) if n == nn]
        if not ss:
            continue
        rt = ttnn.RuntimeArgs()
        for i, (x, y) in enumerate(ss):
            rt[x][y] = [0]
        rr = r if nn == 0 else [r[2], r[3], r[0], r[1]]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("mcast_bench.cpp"),
                core_ranges=crs(ss),
                compile_time_args=[N, B] + rr + [len(dests), buf.buffer_address(), buf.buffer_address(), buf.buffer_address()]
                + [int(flag), int(barrier), st.buffer_address()],
                defines=[("LOOPBACK", "1")] if loopback else [],
                runtime_args=rt,
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0 if nn == 0 else ttnn.DataMovementProcessor.RISCV_1,
                    noc=ttnn.NOC.RISCV_0_default if nn == 0 else ttnn.NOC.RISCV_1_default,
                ),
            )
        )
    prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])
    ttnn.generic_op([buf, st], prog)
    ttnn.synchronize_device(device)
    s = read_stamps(st, len(allc))
    idx = [allc.index(c) for c in senders]
    s = s[idx]
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


@pytest.mark.parametrize("B", [8192, 16384, 32768, 65536])
@pytest.mark.parametrize("mode", ["noc0", "noc1", "noc0+noc1", "10x_noc0", "5+5"])
def test_mcast_column(device, B, mode):
    """Senders inside the destination column (loopback), as an in-cluster all-gather of x would."""
    col = [(8, y) for y in range(10)]
    if mode == "noc0":
        senders, nocs = [(8, 0)], [0]
    elif mode == "noc1":
        senders, nocs = [(8, 9)], [1]
    elif mode == "noc0+noc1":
        senders, nocs = [(8, 0), (8, 9)], [0, 1]
    elif mode == "10x_noc0":
        senders, nocs = col, [0] * 10
    else:
        senders, nocs = col, [0] * 5 + [1] * 5
    N = max(4, (4 * 1024 * 1024) // (B * len(senders)))
    us, gbs, cyc = run_mcast(device, senders, col, N, B, False, False, nocs, loopback=True)
    logger.info(f"MCASTCOL {mode} B {B}: {us:.1f} us, {gbs:.1f} GB/s into every core of the column")

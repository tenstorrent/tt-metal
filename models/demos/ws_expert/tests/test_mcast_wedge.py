# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Single-chip, data-movement-only repro of the lane design's NOC0 group multicast wedge: a lane group's x multicast
(2 x 10 cores, logical columns 1 and 2; the multicaster's BRISC at column 1, row MW_SRC_ROW, NOC0) with switchable NOC0
background traffic (kernels/mw_*.cpp). Launches of MW_ITER multicasts repeat for MW_SECS seconds.

Knobs: MW_BG, a comma list of background kinds: gather / own (DRAM reads into every other group core), h (2 KB writes
and a semaphore increment from every other group core into the last one), stream (13.8 KB writes from the two cores
just west of the group, rows 4 and 5, to logical column 11 of the same rows, through the group). MW_SELF: the
multicaster's own DRAM reads (1 gather, 2 own, 3 both). MW_PB payload bytes (default 20 bfp8 tiles), MW_NX ring slots.
"""

import os
import time

import torch
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import crs, kpath, read_stamps, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, rect_noc, sharded_l1

PAGE = 16384
NPAGES = 1024
BG_BITS = {"gather": 1, "own": 2, "h": 4, "stream": 8}


def l1buf(device, cores, nbytes):
    """Same-address L1 bytes on every core, with 64 B of slack to align the start (a64)."""
    words = (nbytes + 64 + 63) // 64 * 16
    return sharded_l1(device, cores, [1, words], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, torch.int32)


def a64(t):
    return (t.buffer_address() + 63) // 64 * 64


def test_mcast_wedge(device):
    secs = float(os.environ.get("MW_SECS", "180"))
    ITER = int(os.environ.get("MW_ITER", "4000"))
    PB = int(os.environ.get("MW_PB", str(20 * 1088)))
    NX = int(os.environ.get("MW_NX", "9"))
    row = int(os.environ.get("MW_SRC_ROW", "0"))
    bg = [k for k in os.environ.get("MW_BG", "").split(",") if k]
    assert all(k in BG_BITS for k in bg), bg
    self_rd = int(os.environ.get("MW_SELF", "0"))
    mcast = os.environ.get("MW_MC", "1") != "0"  # 0: a control without the multicast (MW_NO_MCAST)
    bg_noc1 = os.environ.get("MW_BG_NOC1", "0") != "0"  # the group cores' background on NCRISC / NOC1, receivers on BRISC
    gq = int(os.environ.get("MW_GQ", "32"))  # gather reads in flight
    mc_bar = os.environ.get("MW_MC_BARRIER", "0") != "0"  # write barrier after each multicast
    rect_mode = sum(BG_BITS[k] for k in bg if k != "stream")
    stream = "stream" in bg

    rect = [(1, y) for y in range(10)] + [(2, y) for y in range(10)]
    GC = len(rect)
    mc = (1, row)
    others = [c for c in rect if c != mc]
    hdst = rect[-1]
    muxes = [(0, 4), (0, 5)] if stream else []
    sinks = [(11, 4), (11, 5)] if stream else []
    allc = rect + muxes + sinks
    r0 = rect_noc(device, rect)
    assert noc(device, *rect[0]) == (r0[0], r0[1]), "the rectangle's NOC0 start corner is logical (1, 0)"

    ring = l1buf(device, allc, NX * PB)
    flags = l1buf(device, allc, NX * 16)
    prog = l1buf(device, allc, GC * 16)
    src = l1buf(device, allc, max(PB, 13824))
    scr = l1buf(device, allc, 16)
    stop = l1buf(device, allc, 16)
    land = l1buf(device, allc, 32 * 1280 + 4 * 13824)
    hland = l1buf(device, allc, 4 * 2048)
    hsem = l1buf(device, allc, 16)
    bland = l1buf(device, allc, 2 * 13824)
    st = stamps_tensor(device, allc)
    dram = ttnn.from_torch(
        torch.zeros([NPAGES, PAGE // 4], dtype=torch.int32),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    io = [ring, flags, prog, src, scr, stop, land, hland, hsem, bland, st, dram]
    dacc = list(ttnn.TensorAccessorArgs(dram).get_compile_time_args())
    dm = lambda risc: ttnn.DataMovementConfigDescriptor(
        processor=ttnn.DataMovementProcessor.RISCV_0 if risc == 0 else ttnn.DataMovementProcessor.RISCV_1,
        noc=ttnn.NOC.RISCV_0_default if risc == 0 else ttnn.NOC.RISCV_1_default,
    )
    bgc = (others if rect_mode else []) + muxes
    mx, my = noc(device, *mc)
    hx, hy = noc(device, *hdst)

    def program(launch):
        base, stopv = launch * ITER, launch + 1
        kernels = []
        rt = ttnn.RuntimeArgs()
        rt[mc[0]][mc[1]] = [base, stopv, dram.buffer_address(), len(bgc)] + [v for c in bgc for v in noc(device, *c)]
        kernels.append(
            ttnn.KernelDescriptor(
                kernel_source=kpath("mw_mcast.cpp"),
                core_ranges=crs([mc]),
                compile_time_args=[ITER, PB, NX] + r0 + [GC]
                + [a64(ring), a64(flags), a64(prog), a64(src)]
                + [a64(scr), st.buffer_address(), a64(stop), self_rd, a64(land), NPAGES]
                + dacc,
                runtime_args=rt,
                defines=([] if mcast else [("MW_NO_MCAST", "1")]) + ([("MW_MC_BARRIER", "1")] if mc_bar else []),
                config=dm(0),
            )
        )
        recv_groups = [([mc], 1), (others, 0)] if bg_noc1 and rect_mode else [(rect, 1)]
        for cores, risc in recv_groups if mcast else []:
            rrt = ttnn.RuntimeArgs()
            for x, y in cores:
                rrt[x][y] = [base, rect.index((x, y)), mx, my]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("mw_recv.cpp"),
                    core_ranges=crs(cores),
                    compile_time_args=[ITER, NX, a64(flags), a64(prog), st.buffer_address()],
                    runtime_args=rrt,
                    config=dm(risc),
                )
            )
        for cores, mode, risc in ((others, rect_mode, 1 if bg_noc1 else 0), (muxes, 8, 0)):
            if not cores or not mode:
                continue
            brt = ttnn.RuntimeArgs()
            for k, (x, y) in enumerate(cores):
                bx, by = noc(device, *sinks[k % len(sinks)]) if mode & 8 else (0, 0)
                brt[x][y] = [stopv, dram.buffer_address(), hx, hy, bx, by, 101 * k + 7]
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=kpath("mw_bg.cpp"),
                    core_ranges=crs(cores),
                    compile_time_args=[mode, a64(stop), st.buffer_address(), a64(land), a64(hland)]
                    + [a64(hsem), a64(bland), a64(src), NPAGES]
                    + dacc,
                    runtime_args=brt,
                    defines=[("MW_GQ", str(gq))],
                    config=dm(risc),
                )
            )
        return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=[])

    logger.info(
        f"MW knobs: ITER {ITER}, PB {PB}, NX {NX}, multicaster {mc}, background {bg or 'none'} on "
        f"{len(bgc)} cores (NOC{int(bg_noc1)}, {gq} gather reads in flight), self reads {self_rd}, multicast {int(mcast)}"
        f"{' with a write barrier' if mc_bar else ''}, {secs:.0f} s"
    )
    t0, n = time.time(), 0
    while time.time() - t0 < secs:
        ttnn.generic_op(io, program(n))
        ttnn.synchronize_device(device)
        n += 1
        if n % 500 == 0:
            s = read_stamps(st, len(allc))
            bgn = [int(s[allc.index(c), 8]) for c in bgc]
            logger.info(
                f"MW launch {n}, {time.time() - t0:.0f} s: {n * ITER} multicasts, multicaster at "
                f"{int(s[allc.index(mc), 0])}, background loops min {min(bgn) if bgn else 0} max "
                f"{max(bgn) if bgn else 0} (last launch)"
            )
    dt = time.time() - t0
    logger.info(
        f"MW SUMMARY: {n} launches, {n * ITER} multicasts in {dt:.0f} s without a hang "
        f"({n * ITER * PB / dt / 1e9:.1f} GB/s multicast payload), background {bg or 'none'}, self reads {self_rd}, multicast {int(mcast)}"
    )

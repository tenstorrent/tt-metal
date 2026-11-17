# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Unicast cost on Blackhole along a group column (logical column 1, rows 0..9), for the lane chains' roofline:
one hop (flushed after each block, or streamed), every hop of the column at once, and a 10-core chain (per-hop latency
and throughput with a dedicated RISC a hop). NOC0 = BRISC, down the column; NOC1 = NCRISC, up the column."""

import statistics

import pytest
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, crs, kpath, read_stamps, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, sharded_l1

COL = 1
ROWS = list(range(10))


def run_uc(device, pairs, N, B, S, mode, nocid, head=None, tail=None):
    """pairs: [(src logical, dst logical or None)]; the chain (mode 2) gives head / tail. Returns stamps by core."""
    cores = list(dict.fromkeys([p[0] for p in pairs] + [p[1] for p in pairs if p[1]]))
    st = stamps_tensor(device, cores)
    nt = -(-(S * B) // 2048)
    ring = sharded_l1(device, cores, [nt * 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT)
    fl = sharded_l1(device, cores, [S + 4, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    cr = sharded_l1(device, cores, [4, 4], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    rt = ttnn.RuntimeArgs()
    srcs = {p[0]: p[1] for p in pairs}
    hxy = noc(device, *head) if head else (0, 0)
    for c in cores:
        d = srcs.get(c)
        dx, dy = noc(device, *d) if d else (0, 0xFFFF)
        rt[c[0]][c[1]] = [dx, dy, int(c == head), hxy[0], hxy[1]]
    k = ttnn.KernelDescriptor(
        kernel_source=kpath("uc_bench.cpp"),
        core_ranges=crs(cores),
        compile_time_args=[
            N,
            B,
            S,
            ring.buffer_address(),
            fl.buffer_address(),
            cr.buffer_address(),
            st.buffer_address(),
        ]
        + [mode],
        runtime_args=rt,
        config=ttnn.DataMovementConfigDescriptor(
            processor=ttnn.DataMovementProcessor.RISCV_0 if nocid == 0 else ttnn.DataMovementProcessor.RISCV_1,
            noc=ttnn.NOC.RISCV_0_default if nocid == 0 else ttnn.NOC.RISCV_1_default,
        ),
    )
    ttnn.generic_op([ring, fl, cr, st], ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[]))
    ttnn.synchronize_device(device)
    s = read_stamps(st, len(cores))
    return {c: s[i] for i, c in enumerate(cores)}


def span(s, c):
    return int((s[c][1] - s[c][0]) & 0xFFFFFFFF)


def hop_pairs(nocid):
    # NOC0 down the column (row y -> y + 1), NOC1 up (row y -> y - 1): one NoC hop in each NoC's own direction
    return [((COL, y), (COL, y + 1)) for y in ROWS[:-1]] if nocid == 0 else [((COL, y), (COL, y - 1)) for y in ROWS[1:]]


@pytest.mark.parametrize("nocid", [0, 1])
@pytest.mark.parametrize("B", [4352, 10880, 21760])
def test_uc_hop(device, nocid, B):
    """One hop, one pair; then all 9 hops of the column at once (each link one stream, each core sending and
    receiving). Bytes a cycle a stream, flushed after each block vs streamed."""
    clk, N, S = aiclk_mhz(0), 400, 8
    one = [hop_pairs(nocid)[4]]
    res = []
    for name, pairs in (("1 pair", one), ("9 pairs", hop_pairs(nocid))):
        for mode, mname in ((0, "flush"), (1, "stream")):
            s = run_uc(device, pairs, N, B, S, mode, nocid)
            cyc = max(span(s, p[0]) for p in pairs)
            res.append(f"{name} {mname}: {N * B / cyc:.1f} B/cyc a stream ({N * B / cyc * clk / 1e3:.1f} GB/s)")
    logger.info(f"UC hop NOC{nocid} B {B}: " + "; ".join(res))


@pytest.mark.parametrize("nocid", [0, 1])
@pytest.mark.parametrize("B,S", [(21760, 4), (21760, 8), (10880, 8), (4352, 16)])
def test_uc_chain(device, nocid, B, S):
    """10-core chain down (NOC0) / up (NOC1) the column, a dedicated RISC a hop: head send -> tail arrival latency
    (block 0 of a quiet chain, and the median over the first 64 under load) and throughput."""
    clk, N = aiclk_mhz(0), 400
    rows = ROWS if nocid == 0 else ROWS[::-1]
    chain = [(COL, y) for y in rows]
    pairs = [(chain[i], chain[i + 1]) for i in range(9)] + [(chain[9], None)]
    s = run_uc(device, pairs, N, B, S, 2, nocid, head=chain[0], tail=chain[9])
    h, t = s[chain[0]], s[chain[9]]
    lat = [int((t[64 + i] - h[128 + i]) & 0xFFFFFFFF) for i in range(64)]
    cyc = span(s, chain[0])
    logger.info(
        f"UC chain NOC{nocid} B {B} S {S}: quiet 9-hop latency {lat[0]} cyc ({lat[0] / 9:.0f} a hop), loaded median "
        f"{statistics.median(lat):.0f}; throughput {N * B / cyc:.1f} B/cyc ({N * B / cyc * clk / 1e3:.1f} GB/s), "
        f"{cyc / N:.0f} cyc a block"
    )


@pytest.mark.parametrize("nocid", [0, 1])
def test_uc_niu_state(device, nocid):
    """Are the NIU request FIFOs on (NIU_CFG_0 bit 16)? If so, a NOC_CMD_CTRL write while the FIFO is full is dropped
    (only CMD_BUF_OVFL tells), and tt-metal's noc_cmd_buf_ready (polls NOC_CMD_CTRL) does not check CMD_BUF_AVAIL."""
    pairs = hop_pairs(nocid)
    s = run_uc(device, pairs, 400, 4352, 8, 1, nocid)
    for c in (pairs[0][0], pairs[4][0]):
        cfg, avail, ovfl = (int(s[c][200 + i]) for i in range(3))
        logger.info(
            f"UC NIU NOC{nocid} core {c}: NIU_CFG_0 0x{cfg:08x} (request FIFOs {'ON' if cfg >> 16 & 1 else 'off'}), "
            f"CMD_BUF_AVAIL 0x{avail:08x}, CMD_BUF_OVFL 0x{ovfl:08x}"
        )

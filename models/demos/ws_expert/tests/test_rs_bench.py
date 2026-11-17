# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Column reduce-scatter for a K-split gate/up (kernels/rs_bench.cpp): 10 cores of a group column, each owning one
(gate, up) pair; per row every core sends each other core its pair's partial tiles. Capacity (rows as fast as
possible) and, at the compute's row rate, the lag from production to a complete slot at the owner."""

import statistics

import pytest
from loguru import logger

import ttnn
from models.demos.ws_expert.bench_util import aiclk_mhz, crs, kpath, read_stamps, stamps_tensor
from models.demos.ws_expert.ws_expert import noc, sharded_l1

COL, NC, R = 1, 10, 400


def run_rs(device, mode, tb, D, T):
    cores = [(COL, y) for y in range(NC)]
    st = stamps_tensor(device, cores)
    U, RM = ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT
    land = sharded_l1(device, cores, [D * NC * 2 * tb // 16, 4], U, RM)
    flags = sharded_l1(device, cores, [D * NC * 2 + 2, 4], U, RM)
    cred = sharded_l1(device, cores, [NC, 4], U, RM)
    cons = sharded_l1(device, cores, [1, 4], U, RM)
    xy = [noc(device, *c) for c in cores]
    rt = ttnn.RuntimeArgs()
    for i, c in enumerate(cores):
        rt[c[0]][c[1]] = [i] + [p[0] for p in xy] + [p[1] for p in xy]
    ct = [
        R,
        D,
        tb,
        mode,
        land.buffer_address(),
        flags.buffer_address(),
        cred.buffer_address(),
        st.buffer_address(),
        T,
        cons.buffer_address(),
        NC,
    ]
    ks = [
        ttnn.KernelDescriptor(
            kernel_source=kpath("rs_bench.cpp"),
            core_ranges=crs(cores),
            compile_time_args=ct,
            runtime_args=rt,
            config=ttnn.DataMovementConfigDescriptor(processor=proc, noc=n),
        )
        for proc, n in (
            (ttnn.DataMovementProcessor.RISCV_0, ttnn.NOC.RISCV_0_default),
            (ttnn.DataMovementProcessor.RISCV_1, ttnn.NOC.RISCV_1_default),
        )
    ]
    ttnn.generic_op([land, flags, cred, cons, st], ttnn.ProgramDescriptor(kernels=ks, semaphores=[], cbs=[]))
    ttnn.synchronize_device(device)
    s = read_stamps(st, NC)
    span = max(int((max(s[i][1], s[i][3]) - min(s[i][0], s[i][2])) & 0xFFFFFFFF) for i in range(NC))
    lags = [int((s[i][64 + q] - s[i][2] - q * T) & 0xFFFFFFFF) for i in range(NC) for q in range(D, R)]
    return span / R, statistics.median(lags), max(lags)


@pytest.mark.parametrize("D", [2, 4])
@pytest.mark.parametrize("tb", [2048, 1088], ids=["bf16", "bfp8"])
@pytest.mark.parametrize("mode", [0, 1], ids=["dirsplit", "gusplit"])
def test_rs_bench(device, mode, tb, D):
    cap, _, _ = run_rs(device, mode, tb, D, 0)
    T = 6950  # the 10 x 2 row on one core (row bench, fold + SwiGLU)
    per, lag_med, lag_max = run_rs(device, mode, tb, D, T)
    out_b = (NC - 1) * 2 * tb
    logger.info(
        f"RS {['dir split', 'gate/up split'][mode]} {'bf16' if tb == 2048 else 'bfp8'} D {D}: capacity {cap:.0f} cyc "
        f"a row ({out_b / cap:.1f} B/cyc out a core); at T {T}: {per:.0f} cyc a row, lag median {lag_med:.0f} max "
        f"{lag_max:.0f}; aiclk {aiclk_mhz(0)}"
    )

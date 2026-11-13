# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Single-core matmul microbench: resident in0 / in1 in L1, stock matmul_block, fp32 DEST, LoFi.

cycles per tile product = (PACK end - UNPACK start) / (passes * tile products); LoFi peak = 16.
"""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import (
    TILE_BYTES,
    aiclk_mhz,
    cb,
    crs,
    kpath,
    pcc,
    quantize,
    read_stamps,
    stamps_tensor,
    to_dev,
)

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b, "bfp4": ttnn.bfloat4_b, "fp32": ttnn.float32}


def run_bench(device, rt, ct, kt, nbr, nbc, act, x_dt="bf16", w_dt="bfp4", out_dt="bf16", full_sync=False, target=2e6):
    core = (0, 0)
    grid = crs([core])
    ops_per_pass = nbr * nbc * rt * ct * kt
    passes = max(2, int(target / (ops_per_pass * 16)))
    torch.manual_seed(0)
    x = torch.randn(nbr * rt * 32, kt * 32)
    w = torch.randn(kt * 32, nbc * ct * 32) / (kt * 32) ** 0.5
    xd, wd, od = DT[x_dt], DT[w_dt], DT[out_dt]
    x_t = to_dev(x, xd, device)
    w_t = to_dev(w, wd, device)
    out_cols = nbc * ct // (2 if act else 1)
    o_t = to_dev(torch.zeros(nbr * rt * 32, out_cols * 32), od, device)
    st = stamps_tensor(device, [core])
    n0, n1 = nbr * rt * kt, kt * nbc * ct
    out_blk = rt * ct // (2 if act else 1)
    l1_need = n0 * TILE_BYTES[xd] + n1 * TILE_BYTES[wd] + 4 * out_blk * TILE_BYTES[od]
    assert l1_need < 1300 * 1024, l1_need

    rd_rt = ttnn.RuntimeArgs()
    rd_rt[0][0] = [x_t.buffer_address(), w_t.buffer_address(), n0, n1]
    total_out = passes * nbr * nbc * out_blk
    nout = nbr * nbc * out_blk
    wr_rt = ttnn.RuntimeArgs()
    wr_rt[0][0] = [o_t.buffer_address(), total_out, nout, out_blk]
    cp_rt = ttnn.RuntimeArgs()
    cp_rt[0][0] = [st.buffer_address()]
    rd = ttnn.KernelDescriptor(
        kernel_source=kpath("bench_reader.cpp"),
        core_ranges=grid,
        compile_time_args=list(ttnn.TensorAccessorArgs(x_t).get_compile_time_args())
        + list(ttnn.TensorAccessorArgs(w_t).get_compile_time_args()),
        runtime_args=rd_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    wr = ttnn.KernelDescriptor(
        kernel_source=kpath("bench_writer.cpp"),
        core_ranges=grid,
        compile_time_args=list(ttnn.TensorAccessorArgs(o_t).get_compile_time_args()),
        runtime_args=wr_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    cp = ttnn.KernelDescriptor(
        kernel_source=kpath("bench_mm_compute.cpp"),
        core_ranges=grid,
        compile_time_args=[rt, ct, kt, nbr, nbc, passes, int(act)],
        named_compile_time_args=[("expert_activation", 0)],
        defines=[
            ("EXPERT_ACTIVATION_BETA", "1.0f"),
            ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
            ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
            ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
        ],
        runtime_args=cp_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True, dst_full_sync_en=full_sync
        ),
    )
    prog = ttnn.ProgramDescriptor(
        kernels=[rd, wr, cp],
        semaphores=[],
        cbs=[cb(0, xd, n0, grid), cb(1, wd, n1, grid), cb(16, od, 4 * out_blk, grid)],
    )
    ttnn.generic_op([x_t, w_t, st, o_t], prog)
    ttnn.synchronize_device(device)
    s = read_stamps(st, 1)[0]
    cyc = int((s[5] - s[0]) & 0xFFFFFFFF)
    cpt = cyc / (passes * ops_per_pass)
    # correctness of the last pass
    xq, wq = quantize(x, xd), quantize(w, wd)
    ref = xq @ wq
    if act:
        g = ref.reshape(ref.shape[0], -1, 2, 32)[:, :, 0, :].reshape(ref.shape[0], -1)
        u = ref.reshape(ref.shape[0], -1, 2, 32)[:, :, 1, :].reshape(ref.shape[0], -1)
        ref = torch.nn.functional.silu(g) * u
    # out tiles land in block order: unscramble into [rows, cols]
    o = ttnn.to_torch(o_t).float()
    ot = o.reshape(-1, 32, o.shape[1] // 32, 32).permute(0, 2, 1, 3).reshape(-1, 32, 32)  # tile row-major
    full = torch.zeros_like(ref)
    oc = nbc * ct // (2 if act else 1)
    i = 0
    for br in range(nbr):
        for bc in range(nbc):
            for t in range(out_blk):
                if act:
                    r, c = t // (ct // 2), t % (ct // 2)
                    gc = bc * ct // 2 + c
                else:
                    r, c = t // ct, t % ct
                    gc = bc * ct + c
                gr = br * rt + r
                full[gr * 32 : gr * 32 + 32, gc * 32 : gc * 32 + 32] = ot[i]
                i += 1
    p = pcc(full, ref)
    return dict(cpt=cpt, util=16 / cpt, passes=passes, pcc=p, cyc=cyc, aiclk=aiclk_mhz(0))


CASES = [
    # (rt, ct, kt, nbr, nbc, act) gate/up-like (K 120) and down-like (K 20)
    (1, 4, 120, 2, 2, True),
    (2, 2, 120, 2, 2, True),
    (1, 2, 120, 2, 4, True),
    (4, 2, 120, 1, 2, True),
    (1, 4, 120, 2, 2, False),
    (1, 4, 20, 4, 6, False),
    (2, 2, 20, 4, 6, False),
    (1, 3, 20, 4, 8, False),
    (1, 2, 20, 4, 8, False),
    (4, 1, 20, 2, 12, False),
    (1, 1, 20, 4, 12, False),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}x{c[1]}k{c[2]}{'act' if c[5] else ''}")
@pytest.mark.parametrize("x_dt", ["bf16", "bfp8"])
def test_mm_bench(device, case, x_dt):
    rt, ct, kt, nbr, nbc, act = case
    r = run_bench(device, rt, ct, kt, nbr, nbc, act, x_dt=x_dt)
    logger.info(
        f"MMBENCH {rt}x{ct} K{kt} blocks {nbr}x{nbc} act {int(act)} x {x_dt}: {r['cpt']:.2f} cyc/tile "
        f"({100 * r['util']:.1f}% LoFi) pcc {r['pcc']:.5f} passes {r['passes']} aiclk {r['aiclk']}"
    )
    assert r["pcc"] > 0.99


@pytest.mark.parametrize(
    "case",
    [(2, 4, 120, 2, 1, True), (1, 8, 120, 2, 1, True), (2, 4, 20, 4, 3, False), (1, 8, 20, 4, 3, False)],
    ids=lambda c: f"{c[0]}x{c[1]}k{c[2]}",
)
def test_mm_bench_fullsync(device, case):
    rt, ct, kt, nbr, nbc, act = case
    r = run_bench(device, rt, ct, kt, nbr, nbc, act, full_sync=True)
    logger.info(
        f"MMBENCH FULLSYNC {rt}x{ct} K{kt} blocks {nbr}x{nbc} act {int(act)}: {r['cpt']:.2f} cyc/tile "
        f"({100 * r['util']:.1f}% LoFi) pcc {r['pcc']:.5f}"
    )
    assert r["pcc"] > 0.99

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""SFPU SwiGLU cost on the PACK thread (fp32 DEST): cycles per (gate, up) tile pair and error vs torch."""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import aiclk_mhz, cb, crs, kpath, read_stamps, stamps_tensor, to_dev

NAMES = {
    0: "stock silu pass + multiply pass",
    1: "fused, fp32 exp + 2 Newton",
    2: "fused, exp_21f + 1 Newton (current)",
    3: "fused, exp_21f + raw approx recip",
    4: "load g, u, g*u, store (floor)",
    5: "fused, exp_21f only (no recip)",
    6: "custom exp(-|g|) + normalized 1 Newton + sign fix",
    7: "custom, no Newton",
    8: "exp_21f, constants in Prgm regs, clamp 254, plain Newton",
    9: "LUT sigmoid (3 linear pieces)",
}


def run_glu(device, v, n, unroll=1):
    grid = crs([(0, 0)])
    torch.manual_seed(0)
    g = (torch.rand(32, 32) * 20 - 10)
    u = torch.randn(32, 32)
    xt = to_dev(torch.cat([g, u], 0), ttnn.float32, device)
    ot = to_dev(torch.zeros(32, 32), ttnn.float32, device)
    st = stamps_tensor(device, [(0, 0)])
    rd, wr, cp = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd[0][0] = [xt.buffer_address(), xt.buffer_address(), 2, 0]
    wr[0][0] = [ot.buffer_address(), 1, 1, 1]
    cp[0][0] = [st.buffer_address()]
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    utd[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=True)
    cc.unpack_to_dest_mode = utd
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=kpath("bench_reader.cpp"),
                core_ranges=grid,
                compile_time_args=list(ttnn.TensorAccessorArgs(xt).get_compile_time_args()) * 2,
                runtime_args=rd,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("bench_writer.cpp"),
                core_ranges=grid,
                compile_time_args=list(ttnn.TensorAccessorArgs(ot).get_compile_time_args()),
                runtime_args=wr,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("glu_bench.cpp"),
                core_ranges=grid,
                compile_time_args=[n],
                named_compile_time_args=[("expert_activation", 0)],
                defines=[
                    ("GLU_V", str(v)),
                    ("GLU_UNROLL", str(unroll)),
                    ("EXPERT_ACTIVATION_BETA", "1.0f"),
                    ("EXPERT_ACTIVATION_LINEAR_BETA", "1.0f"),
                    ("EXPERT_ACTIVATION_LIMIT", "7.0f"),
                    ("EXPERT_ACTIVATION_ALPHA", "1.702f"),
                ],
                runtime_args=cp,
                config=cc,
            ),
        ],
        semaphores=[],
        cbs=[cb(0, ttnn.float32, 2, grid), cb(1, ttnn.float32, 2, grid), cb(16, ttnn.float32, 2, grid)],
    )
    ttnn.generic_op([xt, ot, st], prog)
    s = read_stamps(st, 1)[0]
    cyc = int((s[1] - s[0]) & 0xFFFFFFFF)
    out = ttnn.to_torch(ot).float()
    ref = torch.nn.functional.silu(g) * u
    err = (out - ref).abs() / (ref.abs() + 1e-3)
    return cyc, float(err.max()), float(((out - ref).norm() / ref.norm()))


@pytest.mark.parametrize("unroll", [1])
@pytest.mark.parametrize("v", [2, 8, 9])
def test_glu_bench(device, v, unroll):
    if v == 0 and unroll == 2:
        pytest.skip("stock path has its own unroll")
    c1, emax, erel = run_glu(device, v, 1, unroll)
    c64, _, _ = run_glu(device, v, 64, unroll)
    per = (c64 - c1) / 63
    logger.info(
        f"GLU v{v} unroll {unroll} [{NAMES[v]}]: {per:.0f} cyc per tile pair ({per / 32:.1f} a row of 32), "
        f"x2 pairs a row of h = {2 * per:.0f}; one-shot {c1}; err max rel {emax:.2e} norm {erel:.2e}"
    )

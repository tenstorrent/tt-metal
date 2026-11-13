# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""What the Blackhole SFPU's approx_exp / approx_recip return (fp32 DEST, one tile)."""

import pytest
import torch
import ttnn
from loguru import logger

from models.demos.ws_expert.bench_util import cb, crs, kpath, to_dev


def run_probe(device, op, x):
    grid = crs([(0, 0)])
    xt = to_dev(x, ttnn.float32, device)
    ot = to_dev(torch.zeros(32, 32), ttnn.float32, device)
    rd_rt, wr_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rd_rt[0][0] = [xt.buffer_address(), xt.buffer_address(), 1, 0]
    wr_rt[0][0] = [ot.buffer_address(), 1, 1, 1]
    utd = ttnn._ttnn.program_descriptor.VectorUnpackToDestMode([ttnn.UnpackToDestMode.Default] * 64)
    utd[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cc = ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    cc.unpack_to_dest_mode = utd
    prog = ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=kpath("bench_reader.cpp"),
                core_ranges=grid,
                compile_time_args=list(ttnn.TensorAccessorArgs(xt).get_compile_time_args()) * 2,
                runtime_args=rd_rt,
                config=ttnn.ReaderConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("bench_writer.cpp"),
                core_ranges=grid,
                compile_time_args=list(ttnn.TensorAccessorArgs(ot).get_compile_time_args()),
                runtime_args=wr_rt,
                config=ttnn.WriterConfigDescriptor(),
            ),
            ttnn.KernelDescriptor(
                kernel_source=kpath("sfpu_probe.cpp"),
                core_ranges=grid,
                defines=[("PROBE_OP", op)],
                config=cc,
            ),
        ],
        semaphores=[],
        cbs=[cb(0, ttnn.float32, 2, grid), cb(1, ttnn.float32, 2, grid), cb(16, ttnn.float32, 2, grid)],
    )
    ttnn.generic_op([xt, ot], prog)
    return ttnn.to_torch(ot).float()


@pytest.mark.parametrize(
    "op", ["sfpi::approx_exp(x)", "sfpi::approx_exp(-sfpi::abs(x))", "sfpi::approx_recip(x)", "-sfpi::abs(x)"]
)
def test_probe(device, op):
    x = torch.linspace(-12, 12, 1024).reshape(32, 32)
    y = run_probe(device, op, x)
    xs = [-12.0, -6.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 6.0, 12.0]
    idx = [int(torch.argmin((x.flatten() - v).abs())) for v in xs]
    pairs = [(round(float(x.flatten()[i]), 3), float(y.flatten()[i])) for i in idx]
    logger.info(f"PROBE {op}: {pairs}")

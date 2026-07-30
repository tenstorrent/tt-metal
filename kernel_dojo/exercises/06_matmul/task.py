# SPDX-License-Identifier: Apache-2.0
"""Exercise 06 — single-core tiled matmul."""

import torch
import ttnn

from dojo import harness
from dojo.exercise import Case, Exercise, Workload

CB_A = 0
CB_B = 1
CB_OUT = 16


class Matmul(Exercise):
    title = "Matmul: the FPU's real job"
    blurb = "K-accumulation in DST, SrcOrder::Reverse, math fidelity."
    kernels = ("reader.cpp", "compute.cpp")

    # A K-deep bfloat16 reduction accumulates rounding; PCC is the meaningful
    # check here, and the elementwise tolerance has to scale with K.
    min_pcc = 0.999
    atol = 0.3
    rtol = 0.1

    def cases(self):
        return [
            Case("1x1x1 tiles", {"Mt": 1, "Kt": 1, "Nt": 1}),
            Case("2x2x2 tiles", {"Mt": 2, "Kt": 2, "Nt": 2}),
            Case("4x8x4 tiles", {"Mt": 4, "Kt": 8, "Nt": 4}),
            Case("8x8x8 tiles", {"Mt": 8, "Kt": 8, "Nt": 8}, perf=True),
        ]

    def make_inputs(self, case):
        Mt, Kt, Nt = case["Mt"], case["Kt"], case["Nt"]
        # Scale down so a K-deep sum stays in a range bfloat16 represents well.
        scale = (Kt * 32) ** -0.5
        a = (torch.randn(1, 1, Mt * 32, Kt * 32) * scale).to(torch.bfloat16)
        b = (torch.randn(1, 1, Kt * 32, Nt * 32) * scale).to(torch.bfloat16)
        return [a, b]

    def golden(self, case, inputs):
        a, b = inputs
        return (a.to(torch.float32) @ b.to(torch.float32)).to(torch.bfloat16)

    def output_shape(self, case, inputs):
        return (1, 1, case["Mt"] * 32, case["Nt"] * 32)

    def program(self, case, tensors, ctx):
        a, b, out = tensors
        Mt, Kt, Nt = case["Mt"], case["Kt"], case["Nt"]
        cores = harness.single_core()

        cbs = [
            harness.cb(CB_A, cores, n_pages=2),
            harness.cb(CB_B, cores, n_pages=2),
            harness.cb(CB_OUT, cores, n_pages=2),
        ]

        reader_rt = harness.RtArgs()
        reader_rt.set((0, 0), [a.buffer_address(), b.buffer_address(), Mt, Kt, Nt])
        writer_rt = harness.RtArgs()
        writer_rt.set((0, 0), [out.buffer_address(), Mt * Nt])
        compute_rt = harness.RtArgs()
        compute_rt.set((0, 0), [Mt, Kt, Nt])

        kernels = [
            harness.reader_kernel(
                "reader.cpp",
                cores,
                ct_args=[CB_A, CB_B, *harness.accessor_args(a), *harness.accessor_args(b)],
                rt_args=reader_rt,
            ),
            harness.writer_kernel(
                "writer.cpp",
                cores,
                ct_args=[CB_OUT, *harness.accessor_args(out)],
                rt_args=writer_rt,
            ),
            harness.compute_kernel(
                "compute.cpp",
                cores,
                ct_args=[CB_A, CB_B, CB_OUT],
                rt_args=compute_rt,
                math_fidelity=ttnn.MathFidelity.HiFi4,
            ),
        ]
        return harness.program(kernels, cbs)

    def workload(self, case):
        Mt, Kt, Nt = case["Mt"], case["Kt"], case["Nt"]
        tile_bytes = harness.tile_size(ttnn.bfloat16)
        # This kernel re-reads B for every row of A and A for every column of
        # B, so the traffic is Mt*Nt*Kt tiles per input, not Mt*Kt + Kt*Nt.
        tiles_read = 2 * Mt * Nt * Kt
        return Workload(
            bytes_moved=(tiles_read + Mt * Nt) * tile_bytes,
            # 2 FLOPs (multiply + add) per element of the K reduction.
            flops=2 * (Mt * 32) * (Nt * 32) * (Kt * 32),
        )


EXERCISE = Matmul

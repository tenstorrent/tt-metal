# SPDX-License-Identifier: Apache-2.0
"""Exercise 07 — row-parallel matmul with A reuse, plus the tuning knobs.

The perf cases form two sweeps: core count at fixed fidelity, and fidelity at
fixed core count. Each is a controlled experiment with one variable.
"""

import torch
import ttnn

from dojo import harness
from dojo.exercise import Case, Exercise, Workload

CB_A = 0
CB_B = 1
CB_OUT = 16

#: Shape for the sweeps. Mt=64 so the 64-core case splits exactly one row per
#: core, keeping the scaling curve free of quantisation artefacts.
SWEEP = {"Mt": 64, "Kt": 8, "Nt": 32}

FIDELITY = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


class MatmulTuned(Exercise):
    title = "Matmul at scale: reuse, parallelism, fidelity"
    blurb = "Operand reuse in L1, row-parallel cores, and the math fidelity knob."
    kernels = ("reader.cpp", "compute.cpp")

    min_pcc = 0.999
    atol = 0.3
    rtol = 0.1

    def cases(self):
        correctness = [
            Case("2x2x2, 1 core", {"Mt": 2, "Kt": 2, "Nt": 2, "cores": 1}),
            Case("4x4x4, 4 cores", {"Mt": 4, "Kt": 4, "Nt": 4, "cores": 4}),
            # 8 rows over 3 cores: deliberately uneven.
            Case("8x8x8, 3 cores", {"Mt": 8, "Kt": 8, "Nt": 8, "cores": 3}),
            # More cores than rows: the extra cores must be dropped.
            Case("4x8x8, 16 cores", {"Mt": 4, "Kt": 8, "Nt": 8, "cores": 16}),
            Case("64x8x32, 64 cores", {**SWEEP, "cores": 64}),
        ]
        core_sweep = [
            Case(f"{c} cores, HiFi4", {**SWEEP, "cores": c}, perf=True)
            for c in (1, 8, 32, 64)
        ]
        # The fidelity comparison is run at both ends of the scaling curve on
        # purpose. At 1 core the kernel is compute-bound and fidelity is the
        # binding constraint; at 64 cores DRAM is saturated and it makes no
        # difference at all. The contrast is the lesson.
        # LoFi truncates the input mantissa, so it is held to a looser
        # accuracy bar — that loss is the point, not a kernel bug.
        lofi = {"fidelity": "LoFi", "min_pcc": 0.98, "atol": 1.0}
        fidelity_sweep = [
            Case("1 core, LoFi", {**SWEEP, "cores": 1, **lofi}, perf=True),
            Case("1 core, HiFi2", {**SWEEP, "cores": 1, "fidelity": "HiFi2"}, perf=True),
            Case("64 cores, LoFi", {**SWEEP, "cores": 64, **lofi}, perf=True),
            Case("64 cores, HiFi2", {**SWEEP, "cores": 64, "fidelity": "HiFi2"}, perf=True),
        ]
        return correctness + core_sweep + fidelity_sweep

    def make_inputs(self, case):
        Mt, Kt, Nt = case["Mt"], case["Kt"], case["Nt"]
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

        # Parallelism is over rows of C, so it is capped by Mt.
        grid = harness.first_n_cores(ctx.device, min(case["cores"], Mt))
        cores = harness.cores_used(grid, Mt)
        work = harness.split_tiles(grid, Mt)  # here "tiles" are output rows

        # A's row must fit entirely in cb_a; 2*Kt lets the reader prefetch the
        # next row while compute is still on the current one.
        cbs = [
            harness.cb(CB_A, cores, n_pages=2 * Kt),
            harness.cb(CB_B, cores, n_pages=2 * Kt),
            harness.cb(CB_OUT, cores, n_pages=2),
        ]

        reader_rt = harness.RtArgs()
        writer_rt = harness.RtArgs()
        compute_rt = harness.RtArgs()
        for w in work:
            reader_rt.set(
                w.core,
                [a.buffer_address(), b.buffer_address(), Kt, Nt, w.start_tile, w.n_tiles],
            )
            writer_rt.set(w.core, [out.buffer_address(), w.start_tile * Nt, w.n_tiles * Nt])
            compute_rt.set(w.core, [Kt, Nt, w.n_tiles])

        fidelity = FIDELITY[case.get("fidelity", "HiFi4")]

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
                math_fidelity=fidelity,
            ),
        ]

        case.params["cores_used"] = len(work)
        return harness.program(kernels, cbs)

    def workload(self, case):
        Mt, Kt, Nt = case["Mt"], case["Kt"], case["Nt"]
        tile_bytes = harness.tile_size(ttnn.bfloat16)
        # With A-row reuse: A is read once per row, B once per output tile.
        # Compare against lesson 06's 2*Mt*Nt*Kt.
        tiles_read = Mt * Kt + Mt * Nt * Kt
        return Workload(
            bytes_moved=(tiles_read + Mt * Nt) * tile_bytes,
            flops=2 * (Mt * 32) * (Nt * 32) * (Kt * 32),
        )


EXERCISE = MatmulTuned

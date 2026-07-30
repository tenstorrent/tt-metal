# SPDX-License-Identifier: Apache-2.0
"""Exercise 08 — 2-D output blocking, and crossing into compute-bound.

The sweep holds the core count fixed and varies Mb, so the only thing changing
is arithmetic intensity. Mb=1 reproduces lesson 07's kernel.
"""

import torch
import ttnn

from dojo import harness
from dojo.exercise import Case, Exercise, Workload

CB_A = 0
CB_B = 1
CB_OUT = 16

#: Mt=128 divides evenly by every Mb in the sweep, at 16 cores.
SWEEP = {"Mt": 128, "Kt": 8, "Nt": 32}
SWEEP_CORES = 16

FIDELITY = {
    "LoFi": ttnn.MathFidelity.LoFi,
    "HiFi2": ttnn.MathFidelity.HiFi2,
    "HiFi4": ttnn.MathFidelity.HiFi4,
}


class OutputBlocking(Exercise):
    title = "Output blocking: finding the real bottleneck"
    blurb = "Reuse B across Mb rows, leave the DRAM ceiling, then find what's next."
    kernels = ("reader.cpp", "compute.cpp")

    min_pcc = 0.999
    atol = 0.3
    rtol = 0.1

    def cases(self):
        correctness = [
            Case("Mb=1, 4x4x4", {"Mt": 4, "Kt": 4, "Nt": 4, "Mb": 1, "cores": 2}),
            Case("Mb=2, 8x4x4", {"Mt": 8, "Kt": 4, "Nt": 4, "Mb": 2, "cores": 2}),
            Case("Mb=4, 16x8x8", {"Mt": 16, "Kt": 8, "Nt": 8, "Mb": 4, "cores": 4}),
            Case("Mb=8, 32x8x8", {"Mt": 32, "Kt": 8, "Nt": 8, "Mb": 8, "cores": 4}),
            Case("Mb=4, 128x8x32", {**SWEEP, "Mb": 4, "cores": 16}),
        ]
        # One variable: arithmetic intensity.
        block_sweep = [
            Case(f"Mb={mb}", {**SWEEP, "Mb": mb, "cores": SWEEP_CORES}, perf=True)
            for mb in (1, 2, 4, 8)
        ]
        # Having left the DRAM ceiling at Mb=8, the natural guess is that the
        # FPU is now the limit. These cases test that guess — and refute it:
        # 4x fewer math passes buys ~2%, because the cost is per-matmul_tiles
        # issue and unpack, not the mantissa passes. See the README.
        fidelity_sweep = [
            Case(
                "Mb=8, LoFi",
                {**SWEEP, "Mb": 8, "cores": SWEEP_CORES, "fidelity": "LoFi",
                 "min_pcc": 0.98, "atol": 1.0},
                perf=True,
            ),
            Case(
                "Mb=8, HiFi2",
                {**SWEEP, "Mb": 8, "cores": SWEEP_CORES, "fidelity": "HiFi2"},
                perf=True,
            ),
        ]
        return correctness + block_sweep + fidelity_sweep

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
        Mt, Kt, Nt, Mb = case["Mt"], case["Kt"], case["Nt"], case["Mb"]

        assert Mt % Mb == 0, f"Mt={Mt} must be a multiple of Mb={Mb}"
        n_blocks = Mt // Mb

        # Parallelism is now over row-blocks, so Mb trades parallelism for
        # arithmetic intensity: doubling Mb halves the number of work items.
        grid = harness.first_n_cores(ctx.device, min(case["cores"], n_blocks))
        cores = harness.cores_used(grid, n_blocks)
        work = harness.split_tiles(grid, n_blocks)  # "tiles" are row-blocks here

        cbs = [
            # A's whole Mb x Kt sub-block must be resident at once.
            harness.cb(CB_A, cores, n_pages=2 * Mb * Kt),
            harness.cb(CB_B, cores, n_pages=2 * Kt),
            harness.cb(CB_OUT, cores, n_pages=2 * Mb),
        ]

        reader_rt = harness.RtArgs()
        writer_rt = harness.RtArgs()
        compute_rt = harness.RtArgs()
        for w in work:
            reader_rt.set(
                w.core,
                [a.buffer_address(), b.buffer_address(), Kt, Nt, w.start_tile, w.n_tiles],
            )
            writer_rt.set(w.core, [out.buffer_address(), Nt, w.start_tile, w.n_tiles])
            compute_rt.set(w.core, [Kt, Nt, w.n_tiles])

        fidelity = FIDELITY[case.get("fidelity", "HiFi4")]

        kernels = [
            harness.reader_kernel(
                "reader.cpp",
                cores,
                ct_args=[CB_A, CB_B, Mb, *harness.accessor_args(a), *harness.accessor_args(b)],
                rt_args=reader_rt,
            ),
            harness.writer_kernel(
                "writer.cpp",
                cores,
                ct_args=[CB_OUT, Mb, *harness.accessor_args(out)],
                rt_args=writer_rt,
            ),
            harness.compute_kernel(
                "compute.cpp",
                cores,
                ct_args=[CB_A, CB_B, CB_OUT, Mb],
                rt_args=compute_rt,
                math_fidelity=fidelity,
            ),
        ]

        case.params["cores_used"] = len(work)
        return harness.program(kernels, cbs)

    def workload(self, case):
        Mt, Kt, Nt, Mb = case["Mt"], case["Kt"], case["Nt"], case["Mb"]
        tile_bytes = harness.tile_size(ttnn.bfloat16)
        # A once per row-block pass; B once per (row-block, column) pair.
        tiles_read = Mt * Kt + (Mt // Mb) * Nt * Kt
        return Workload(
            bytes_moved=(tiles_read + Mt * Nt) * tile_bytes,
            flops=2 * (Mt * 32) * (Nt * 32) * (Kt * 32),
        )


EXERCISE = OutputBlocking

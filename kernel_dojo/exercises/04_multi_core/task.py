# SPDX-License-Identifier: Apache-2.0
"""Exercise 04 — the same add, spread across the core grid.

The perf cases hold the problem size fixed and sweep the core count, so
`dojo bench 04` prints a scaling curve rather than a single number.
"""

import torch
import ttnn

from dojo import harness
from dojo.exercise import Case, Exercise, Workload

CB_A = 0
CB_B = 1
CB_OUT = 16

#: Problem size for the scaling sweep. Big enough that per-dispatch overhead
#: does not dominate at 64 cores.
SWEEP_TILES = 2048


class MultiCore(Exercise):
    title = "Multi-core: making it 64x wider"
    blurb = "Work splitting, per-core runtime args, and where DRAM bandwidth caps you."
    kernels = ("reader.cpp", "compute.cpp", "writer.cpp")

    min_pcc = 0.9999
    atol = 1e-2
    rtol = 1e-2

    def cases(self):
        correctness = [
            # Fewer tiles than cores: exercises the "drop idle cores" path.
            Case("7 tiles / 8 cores", {"n_tiles": 7, "cores": 8}),
            Case("64 tiles / 8 cores", {"n_tiles": 64, "cores": 8}),
            # Deliberately not divisible: 1000 = 15*64 + 40.
            Case("1000 tiles / 64 cores", {"n_tiles": 1000, "cores": 64}),
            Case("2048 tiles / 64 cores", {"n_tiles": 2048, "cores": 64}),
        ]
        sweep = [
            Case(f"{c} core{'s' if c > 1 else ''}", {"n_tiles": SWEEP_TILES, "cores": c}, perf=True)
            for c in (1, 2, 8, 32, 64)
        ]
        return correctness + sweep

    def make_inputs(self, case):
        n = case["n_tiles"]
        return [
            torch.randn(1, 1, 32, 32 * n).to(torch.bfloat16),
            torch.randn(1, 1, 32, 32 * n).to(torch.bfloat16),
        ]

    def golden(self, case, inputs):
        a, b = inputs
        return (a.to(torch.float32) + b.to(torch.float32)).to(torch.bfloat16)

    def program(self, case, tensors, ctx):
        a, b, out = tensors
        n_tiles = case["n_tiles"]

        grid = harness.first_n_cores(ctx.device, case["cores"])
        # Cores that would get zero tiles must not run the kernel at all: they
        # would execute with unset runtime args.
        cores = harness.cores_used(grid, n_tiles)
        work = harness.split_tiles(grid, n_tiles)

        cbs = [
            harness.cb(CB_A, cores, n_pages=2),
            harness.cb(CB_B, cores, n_pages=2),
            harness.cb(CB_OUT, cores, n_pages=2),
        ]

        reader_rt = harness.RtArgs()
        writer_rt = harness.RtArgs()
        compute_rt = harness.RtArgs()
        for w in work:
            reader_rt.set(w.core, [a.buffer_address(), b.buffer_address(), w.n_tiles, w.start_tile])
            writer_rt.set(w.core, [out.buffer_address(), w.n_tiles, w.start_tile])
            compute_rt.set(w.core, [w.n_tiles])

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
            ),
        ]

        # Surfaced in the bench output so the scaling table is self-explanatory.
        case.params["cores_used"] = len(work)
        return harness.program(kernels, cbs)

    def workload(self, case):
        tile_bytes = harness.tile_size(ttnn.bfloat16)
        return Workload(bytes_moved=3 * case["n_tiles"] * tile_bytes)


EXERCISE = MultiCore

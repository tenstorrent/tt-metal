# SPDX-License-Identifier: Apache-2.0
"""Exercise 05 — blocking and double buffering.

The perf cases hold the core count fixed and sweep the block size, isolating
per-core pipeline efficiency from the parallelism of lesson 04.
"""

import torch
import ttnn

from dojo import harness
from dojo.exercise import Case, Exercise, Workload

CB_A = 0
CB_B = 1
CB_OUT = 16

#: Held fixed for the sweep. Low enough that DRAM bandwidth is not the limit,
#: so the pipeline improvements are visible rather than masked.
SWEEP_CORES = 8
SWEEP_TILES = 2048


class Pipelining(Exercise):
    title = "Pipelining: making one core fast"
    blurb = "Batched NoC reads, blocked DST usage, CB depth. The perf lesson."
    kernels = ("reader.cpp", "compute.cpp")

    min_pcc = 0.9999
    atol = 1e-2
    rtol = 1e-2

    def cases(self):
        correctness = [
            Case("block 1, 8 cores", {"n_tiles": 64, "cores": 8, "block": 1}),
            Case("block 2, 8 cores", {"n_tiles": 128, "cores": 8, "block": 2}),
            Case("block 4, 4 cores", {"n_tiles": 256, "cores": 4, "block": 4}),
            Case("block 8, 8 cores", {"n_tiles": 1024, "cores": 8, "block": 8}),
            Case("block 8, 64 cores", {"n_tiles": 2048, "cores": 64, "block": 4}),
        ]
        sweep = [
            Case(
                f"block {b}",
                {"n_tiles": SWEEP_TILES, "cores": SWEEP_CORES, "block": b},
                perf=True,
            )
            for b in (1, 2, 4, 8)
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
        block = case["block"]

        grid = harness.first_n_cores(ctx.device, case["cores"])
        cores = harness.cores_used(grid, n_tiles)
        work = harness.split_tiles(grid, n_tiles)

        # The kernels have no remainder path, so every core's slice must be a
        # whole number of blocks. The case list is chosen to guarantee it;
        # assert rather than silently produce a wrong answer.
        for w in work:
            assert w.n_tiles % block == 0, (
                f"core {w.core} got {w.n_tiles} tiles, not a multiple of block={block}"
            )

        # Double buffering: one block being filled while another is consumed.
        depth = 2 * block
        cbs = [
            harness.cb(CB_A, cores, n_pages=depth),
            harness.cb(CB_B, cores, n_pages=depth),
            harness.cb(CB_OUT, cores, n_pages=depth),
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
                ct_args=[
                    CB_A,
                    CB_B,
                    block,
                    *harness.accessor_args(a),
                    *harness.accessor_args(b),
                ],
                rt_args=reader_rt,
            ),
            harness.writer_kernel(
                "writer.cpp",
                cores,
                ct_args=[CB_OUT, block, *harness.accessor_args(out)],
                rt_args=writer_rt,
            ),
            harness.compute_kernel(
                "compute.cpp",
                cores,
                ct_args=[CB_A, CB_B, CB_OUT, block],
                rt_args=compute_rt,
            ),
        ]

        case.params["cores_used"] = len(work)
        return harness.program(kernels, cbs)

    def workload(self, case):
        tile_bytes = harness.tile_size(ttnn.bfloat16)
        return Workload(bytes_moved=3 * case["n_tiles"] * tile_bytes)


EXERCISE = Pipelining

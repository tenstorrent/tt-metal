# SPDX-License-Identifier: Apache-2.0
"""Exercise 01 — tile copy.

Host side: one core, one circular buffer, a reader and a writer. The learner
supplies both kernels.
"""

import torch
import ttnn

from dojo import harness
from dojo.exercise import Case, Exercise, Workload

CB_IN = 0


class TileCopy(Exercise):
    title = "Tile copy: DRAM → L1 → DRAM"
    blurb = "Data movement only. Circular buffers, NoC reads/writes, barriers."
    kernels = ("reader.cpp", "writer.cpp")

    # A copy is bit-exact; there is no arithmetic to lose precision to.
    min_pcc = 1.0
    atol = 0.0
    rtol = 0.0

    def cases(self):
        return [
            Case("1 tile", {"n_tiles": 1}),
            Case("4 tiles", {"n_tiles": 4}),
            Case("64 tiles", {"n_tiles": 64}),
            Case("256 tiles", {"n_tiles": 256}, perf=True),
        ]

    def make_inputs(self, case):
        n = case["n_tiles"]
        return [torch.randn(1, 1, 32, 32 * n).to(torch.bfloat16)]

    def golden(self, case, inputs):
        return inputs[0]

    def program(self, case, tensors, ctx):
        src, dst = tensors
        n_tiles = case["n_tiles"]
        cores = harness.single_core()

        # Depth 2 so the reader can fill one page while the writer drains the
        # other. Lesson 05 makes this knob the whole point.
        cbs = [harness.cb(CB_IN, cores, n_pages=2)]

        reader_rt = harness.RtArgs()
        reader_rt.set((0, 0), [src.buffer_address(), n_tiles])
        writer_rt = harness.RtArgs()
        writer_rt.set((0, 0), [dst.buffer_address(), n_tiles])

        kernels = [
            harness.reader_kernel(
                "reader.cpp",
                cores,
                ct_args=[CB_IN, *harness.accessor_args(src)],
                rt_args=reader_rt,
            ),
            harness.writer_kernel(
                "writer.cpp",
                cores,
                ct_args=[CB_IN, *harness.accessor_args(dst)],
                rt_args=writer_rt,
            ),
        ]
        return harness.program(kernels, cbs)

    def workload(self, case):
        # Every tile crosses the NoC twice: DRAM→L1 and L1→DRAM.
        tile_bytes = harness.tile_size(ttnn.bfloat16)
        return Workload(bytes_moved=2 * case["n_tiles"] * tile_bytes)


EXERCISE = TileCopy

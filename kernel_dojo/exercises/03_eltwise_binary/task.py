# SPDX-License-Identifier: Apache-2.0
"""Exercise 03 — element-wise binary add on the FPU."""

import torch
import ttnn

from dojo import harness
from dojo.exercise import Case, Exercise, Workload

CB_A = 0
CB_B = 1
CB_OUT = 16


class EltwiseBinary(Exercise):
    title = "Element-wise binary: two inputs on the FPU"
    blurb = "add_tiles, two circular buffers, overlapping NoC reads."
    kernels = ("reader.cpp", "compute.cpp")

    # A single bfloat16 add rounds once; the FPU accumulates in higher
    # precision, so this is tight but not exact.
    min_pcc = 0.9999
    atol = 1e-2
    rtol = 1e-2

    def cases(self):
        return [
            Case("1 tile", {"n_tiles": 1}),
            Case("8 tiles", {"n_tiles": 8}),
            Case("64 tiles", {"n_tiles": 64}),
            Case("256 tiles", {"n_tiles": 256}, perf=True),
        ]

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
        cores = harness.single_core()

        cbs = [
            harness.cb(CB_A, cores, n_pages=2),
            harness.cb(CB_B, cores, n_pages=2),
            harness.cb(CB_OUT, cores, n_pages=2),
        ]

        reader_rt = harness.RtArgs()
        reader_rt.set((0, 0), [a.buffer_address(), b.buffer_address(), n_tiles])
        writer_rt = harness.RtArgs()
        writer_rt.set((0, 0), [out.buffer_address(), n_tiles])
        compute_rt = harness.RtArgs()
        compute_rt.set((0, 0), [n_tiles])

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
        return harness.program(kernels, cbs)

    def workload(self, case):
        tile_bytes = harness.tile_size(ttnn.bfloat16)
        # Two tiles in, one out.
        return Workload(bytes_moved=3 * case["n_tiles"] * tile_bytes)


EXERCISE = EltwiseBinary

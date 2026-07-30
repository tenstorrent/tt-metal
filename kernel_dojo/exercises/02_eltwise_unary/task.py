# SPDX-License-Identifier: Apache-2.0
"""Exercise 02 — element-wise unary (exp) on the SFPU."""

import torch
import ttnn

from dojo import harness
from dojo.exercise import Case, Exercise, Workload

CB_IN = 0
CB_OUT = 16


class EltwiseUnary(Exercise):
    title = "Element-wise unary: your first compute kernel"
    blurb = "SFPU, DST registers, the unpack/math/pack pipeline."
    kernels = ("compute.cpp",)

    # exp() through the SFPU on bfloat16 is approximate.
    min_pcc = 0.999
    atol = 2e-2
    rtol = 2e-2

    def cases(self):
        return [
            Case("1 tile", {"n_tiles": 1}),
            Case("8 tiles", {"n_tiles": 8}),
            Case("64 tiles", {"n_tiles": 64}),
            Case("256 tiles", {"n_tiles": 256}, perf=True),
        ]

    def make_inputs(self, case):
        n = case["n_tiles"]
        # Keep the range modest: exp() of a large bfloat16 overflows to inf and
        # would make the comparison meaningless rather than informative.
        return [(torch.rand(1, 1, 32, 32 * n) * 4.0 - 2.0).to(torch.bfloat16)]

    def golden(self, case, inputs):
        return torch.exp(inputs[0].to(torch.float32)).to(torch.bfloat16)

    def program(self, case, tensors, ctx):
        src, dst = tensors
        n_tiles = case["n_tiles"]
        cores = harness.single_core()

        cbs = [
            harness.cb(CB_IN, cores, n_pages=2),
            harness.cb(CB_OUT, cores, n_pages=2),
        ]

        reader_rt = harness.RtArgs()
        reader_rt.set((0, 0), [src.buffer_address(), n_tiles])
        writer_rt = harness.RtArgs()
        writer_rt.set((0, 0), [dst.buffer_address(), n_tiles])
        compute_rt = harness.RtArgs()
        compute_rt.set((0, 0), [n_tiles])

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
                ct_args=[CB_OUT, *harness.accessor_args(dst)],
                rt_args=writer_rt,
            ),
            harness.compute_kernel(
                "compute.cpp",
                cores,
                ct_args=[CB_IN, CB_OUT],
                rt_args=compute_rt,
            ),
        ]
        return harness.program(kernels, cbs)

    def workload(self, case):
        tile_bytes = harness.tile_size(ttnn.bfloat16)
        return Workload(bytes_moved=2 * case["n_tiles"] * tile_bytes)


EXERCISE = EltwiseUnary

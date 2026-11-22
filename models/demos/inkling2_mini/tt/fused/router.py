# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused decode router: x -> (top-k weights, top-k ids) in one program, one token.

Replaces in tt/moe.py: reshape, the K-split router GEMM (ROUTER_SPLIT batched matmuls), ttnn.sum over the
splits (FastReduceNC) and moe_grouped_topk (sigmoid, + bias, top-k, gather of the unbiased scores,
normalize + eps, * route scale). All values are bit-equal to that chain:
  - chunks x N-tiles cores (36) each compute one fp32 output tile of the K-split GEMM with matmul_tiles
    accumulating in DEST (as ttnn's bmm) and send its row 0 to the collector's partial pages;
  - the collector sums the partials in chunk order on the SFPU (as FastReduceNC's accurate fp32 path) into one
    dense page (expert e at element e), runs moe_grouped_topk's own sigmoid + bias block on it, a scalar top-k
    on the writer that reproduces the op's stable rank-tag order (17-bit mantissa keys, equal keys by lower
    index), and the op's normalize / scale blocks with its CB ids, formats and compute config.
"""

import struct

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F


def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


class FusedRouter:
    def __init__(self, device, router_w, bias, k, route_scale, epsilon, first_row=3):
        """router_w: [1, chunks, K/chunks, E] fp32 TILE device tensor (tt/moe.py's K-split weight); bias: dense
        torch [E] (the centred correction bias). Cores: chunks * E/32, row-major from first_row."""
        _, self.chunks, kc, e = router_w.shape
        assert router_w.dtype == ttnn.float32 and kc % 32 == 0 and e % 32 == 0 and e <= F.PAGE
        self.device = device
        self.w = router_w
        self.kt, self.nt = kc // 32, e // 32
        self.bias = F.dense_tensor(device, bias.float().reshape(-1))
        self.k = k
        self.packed = dict(packed_eps=_f32_bits(epsilon), packed_scale=_f32_bits(route_scale))
        self.cores, self.grid = F.core_list(device, self.chunks * self.nt, first_row)
        c = device.worker_core_from_logical_core(self.cores[0])
        self.collector_noc = (c.x, c.y)
        # as the op's factory: fp32 inputs and biased keys to DEST; the partials too (FastReduceNC's SFPU path)
        self.compute_cfg = F.compute_config([0, 1, 5, 9])
        fp, g = ttnn.float32, self.grid
        self.cbs = [
            F.cb(0, fp, 1, cores=g), F.cb(1, fp, 1, cores=g), F.cb(2, ttnn.bfloat16, 2, cores=g),
            F.cb(3, ttnn.uint16, 2, 2048, cores=g), F.cb(4, fp, 1, cores=g), F.cb(5, fp, 1, cores=g),
            F.cb(6, ttnn.bfloat16, self.kt, cores=g), F.cb(7, fp, self.kt, cores=g), F.cb(8, fp, 1, cores=g),
            F.cb(9, fp, self.chunks, cores=g), F.cb(15, fp, 2, cores=g), F.cb(17, fp, 1, cores=g),
            F.cb(18, fp, 1, cores=g), F.cb(19, fp, 1, cores=g), F.cb(20, fp, 2, cores=g), F.cb(21, fp, 2, cores=g),
            F.cb(22, fp, 2, cores=g),
        ]  # fmt: skip

    def __call__(self, x):
        """x [1,1,1,K] bf16 TILE (post-norm) -> (weights [1,1,1,k] bf16 TILE L1, ids [1,1,1,k] uint16 TILE L1)."""
        assert x.dtype == ttnn.bfloat16 and x.layout == ttnn.TILE_LAYOUT, (x.dtype, x.layout)
        assert tuple(x.shape) == (1, 1, 1, 32 * self.kt * self.chunks), x.shape  # one token
        shape = ttnn.Shape([1, 1, 1, self.k])
        l1 = ttnn.L1_MEMORY_CONFIG
        w = ttnn.allocate_tensor_on_device(shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, self.device, l1)
        ids = ttnn.allocate_tensor_on_device(shape, ttnn.uint16, ttnn.TILE_LAYOUT, self.device, l1)
        acc = lambda t: list(ttnn.TensorAccessorArgs(t).get_compile_time_args())  # noqa: E731
        named = dict(kt=self.kt, nt=self.nt, chunks=self.chunks, sem=0, width_tiles=self.nt, k=self.k, **self.packed)
        r_args, w_args, c_args = {}, {}, {}
        for i, c in enumerate(self.cores):
            chunk, n, col = i // self.nt, i % self.nt, int(i == 0)
            r_args[c] = [x.buffer_address(), self.w.buffer_address(), chunk, n, col, self.bias.buffer_address()]
            w_args[c] = [w.buffer_address(), ids.buffer_address(), chunk, n, col, *self.collector_noc]
            c_args[c] = [col]

        def kern(name, config, rt, cta=()):
            return ttnn.KernelDescriptor(
                kernel_source=f"{F.KDIR}/{name}",
                core_ranges=self.grid,
                compile_time_args=list(cta),
                named_compile_time_args=list(named.items()),
                runtime_args=[(core, args) for core, args in rt.items()],
                config=config,
            )

        kernels = [
            kern("router_reader.cpp", ttnn.ReaderConfigDescriptor(), r_args, acc(x) + acc(self.w)),
            kern("router_writer.cpp", ttnn.WriterConfigDescriptor(), w_args, acc(w) + acc(ids)),
            kern("router_compute.cpp", self.compute_cfg, c_args),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=0, core_ranges=self.grid, initial_value=0)]
        ttnn.generic_op(
            [x, self.w, self.bias, w, ids], ttnn.ProgramDescriptor(kernels=kernels, cbs=self.cbs, semaphores=sems)
        )
        return w, ids

# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill MoE routing on the device: top-k expert ids [1, 1, T, 8] uint16 TILE -> the unified expert op's
routing buffers (region offsets, counts, row index) and the combine's back index, in two programs (per-core
histograms, then offsets + scatter). Same buffer rows as the host routing (a stable sort of the (token, k)
pairs by expert), so a chunk needs no host round trip and traces as one segment."""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

MAX_CORES = 64


class PfRoute:
    def __init__(self, device):
        self.device = device
        self._hist = {}

    def _cores(self, rt):
        nc = min(MAX_CORES, rt)
        per = -(-rt // nc)
        nc = -(-rt // per)
        return nc, per

    def hist_buffer(self, t, ne):
        """[1, n_cores * ne] uint32 scratch of a t-row chunk (allocate with the chunk's other buffers)."""
        nc, _ = self._cores(t // 32)
        return ttnn.allocate_tensor_on_device(ttnn.Shape([1, nc * ne]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT,
                                              self.device, ttnn.DRAM_MEMORY_CONFIG)  # fmt: skip

    def __call__(self, indices, nreal, hist, off, cnt, xri, back, spare, k):
        """indices [1,1,T,k] uint16 TILE; nreal [1,16] uint32 (word 0 = real rows n); hist from hist_buffer;
        off, cnt [1, ne] uint32; xri [1, rows] uint32; back [1, k T] uint32 (all ROW_MAJOR DRAM)."""
        t, ne = indices.shape[-2], off.shape[-1]
        assert indices.dtype == ttnn.uint16 and indices.layout == ttnn.TILE_LAYOUT and t % 32 == 0, indices
        assert indices.padded_shape[-1] == 32 and k <= 16, indices.padded_shape
        rt = t // 32
        nc, per = self._cores(rt)
        assert hist.shape[-1] == nc * ne, (hist.shape, nc, ne)
        cores, crs = F.core_list(self.device, nc)
        idx_dram = int(indices.memory_config().buffer_type == ttnn.BufferType.DRAM)
        rng = [(i * per, min(rt, (i + 1) * per)) for i in range(nc)]
        named = dict(ne=ne, k=k, idx_dram=idx_dram, t=t, nc=nc)  # t, nc: one cached program per chunk shape
        a_args = {c: [indices.buffer_address(), nreal.buffer_address(), hist.buffer_address(), r0, r1, i]
                  for i, (c, (r0, r1)) in enumerate(zip(cores, rng))}  # fmt: skip
        cfg = ttnn.ReaderConfigDescriptor()
        u32 = ttnn.uint32
        cbs = [F.cb(0, u32, 1, per * 1024 + 64, crs), F.cb(1, u32, 1, ne * 4 + 64, crs), F.cb(2, u32, 1, 128, crs)]
        F.run([indices, nreal, hist], [F.kernel("pf_route_count.cpp", cfg, named, a_args, crs)], cbs)
        ntok = per * 32
        named = dict(named, spare=spare)
        b_args = {c: [indices.buffer_address(), nreal.buffer_address(), hist.buffer_address(), off.buffer_address(),
                      cnt.buffer_address(), xri.buffer_address(), back.buffer_address(), r0, r1, i, nc]
                  for i, (c, (r0, r1)) in enumerate(zip(cores, rng))}  # fmt: skip
        cbs = [
            F.cb(0, u32, 1, per * 1024 + 64, crs),
            F.cb(1, u32, 1, nc * ne * 4 + 64, crs),
            F.cb(2, u32, 1, 128, crs),
            F.cb(3, u32, 1, k * ntok * 4 + 64, crs),
            F.cb(4, u32, 1, (2 * k * ntok + 16 * ne) * 4 + 64, crs),  # xs + xri staging (64-byte slots)
            F.cb(5, u32, 1, 4 * ne * 4 + 64, crs),
        ]
        F.run([indices, nreal, hist, off, cnt, xri, back],
              [F.kernel("pf_route_scatter.cpp", cfg, named, b_args, crs)], cbs)  # fmt: skip
        return back

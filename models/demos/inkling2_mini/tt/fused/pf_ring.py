# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Prefill ring write of a long chunk in one program (data movement only).

The op chain per ring: slices of the chunk's bf16 k / v, a concat, multiplies by the slot masks, an add, a
typecast to the ring dtype and fill_cache: 9 programs. The masks select whole rows (slot q takes the last real
row j < n with j % 1024 == q), and the typecast to bfp8 works per 16-wide face row, so the result is rows of the
chunk's k / v already cast to the ring dtype (the SDPA's k_c / v_c). The kernel moves those rows.

Call: PfRing(device)([(src, ring), ...], nreal) with src [1, H, T, 128] and ring [1, H, 1024, 128], both bfp8
TILE DRAM, T > 1024; nreal: uint32 [1, 16] ROW_MAJOR whose first value is n (1024 < n <= T).
"""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

WINDOW = 1024
NB = 8  # tiles per batch


class PfRing:
    def __init__(self, device):
        self.device = device

    @staticmethod
    def ok(src, ring):
        dram = lambda x: x.memory_config().buffer_type == ttnn.BufferType.DRAM and not x.is_sharded()  # noqa: E731
        return (
            all(x.dtype == ttnn.bfloat8_b and x.layout == ttnn.TILE_LAYOUT and dram(x) for x in (src, ring))
            and src.shape[2] > WINDOW
            and src.shape[2] % 32 == 0
            and tuple(ring.shape) == (1, src.shape[1], WINDOW, src.shape[3])
            and src.shape[3] == 128
        )

    def __call__(self, pairs, nreal):
        t, h = pairs[0][0].shape[2], pairs[0][0].shape[1]
        for src, ring in pairs:
            assert self.ok(src, ring) and src.shape[2] == t and src.shape[1] == h, (src, ring)
        assert nreal.dtype == ttnn.uint32 and nreal.layout == ttnn.ROW_MAJOR_LAYOUT and nreal.shape[-1] == 16, nreal
        tiles = h * 128  # ring tiles: (h * 32 + R) * 4 + c
        grid = self.device.compute_with_storage_grid_size()
        n_cores = min(grid.x * grid.y, len(pairs) * tiles // NB)
        cores, crs = F.core_list(self.device, n_cores)
        total = len(pairs) * tiles
        rt = {}
        for i, core in enumerate(cores):  # core i: tiles [g0, g1) of the pairs laid end to end
            g0, g1 = total * i // n_cores, total * (i + 1) // n_cores
            runs = []
            for j, (src, ring) in enumerate(pairs):
                a, b = max(g0, j * tiles) - j * tiles, min(g1, (j + 1) * tiles) - j * tiles
                if a < b:
                    runs += [src.buffer_address(), ring.buffer_address(), a, b]
            rt[core] = [nreal.buffer_address(), len(runs) // 4] + runs
        named = F.fixed_rt(dict(t32=t // 32, nb=NB), rt)
        k = F.kernel("pf_ring.cpp", ttnn.ReaderConfigDescriptor(), named, rt, crs)
        b8 = ttnn.bfloat8_b
        cbs = [F.cb(0, b8, NB, 1088, cores=crs), F.cb(1, b8, NB, 1088, cores=crs),
               F.cb(2, ttnn.bfloat16, 1, 64, cores=crs)]  # fmt: skip
        io = [nreal] + [s for s, _ in pairs] + [r for _, r in pairs]
        F.run(io, [k], cbs)

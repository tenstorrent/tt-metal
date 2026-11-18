# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""B49 (HY4_B49_IHC=fused, default OFF): iHC of a PREFILL chunk (x4 [1,4,n,D] fp32 TILE, n = 32 * TR) as 5 JIT programs instead of ~40 stock ops per site.

pre(x4)   -> (h bf16 [1,1,n,D] = rms_norm(sum_s pre_s x_s) * w, st)    K1 stats partials (token tile-row r x 6 column chunks = 96 cores: M = sum x @ fn, Q = sum x^2), K2 combine + mixes
                                                                      (pre/post, 8 replicated-column tiles per row = ``st``), K3 collapse + partial sums of squares of h, K4 norm
expand(x4, y, st) -> x4 + post * y (fp32), K5.
Same math as Hy4IHCGen.pre_post / collapse + rms_exact / expand (fp32 SFPU for the stream arithmetic, HiFi4 matmul for the mixes)."""
import os
import struct

import torch

import ttnn
from models.demos.blackhole.hy4_preview.tt.hy4_common import upload

KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ihc_pf_kernels")
f32, bf16 = ttnn.float32, ttnn.bfloat16
CT = 32  # column tiles per core
_CACHE = {}


def enabled():
    return os.environ.get("HY4_B49_IHC", "0") == "fused"


def _bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _acc(t):
    return list(ttnn.TensorAccessorArgs(t).get_compile_time_args())


def _cb(core_set, idx, pages, page, dt):
    return ttnn.CBDescriptor(
        total_size=pages * page,
        core_ranges=core_set,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dt, page_size=page)],
    )


def _kernel(name, core_set, ct, rt, cfg, common):
    return ttnn.KernelDescriptor(
        kernel_source=f"{KDIR}/{name}",
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_set,
        compile_time_args=ct,
        runtime_args=rt,
        common_runtime_args=common,
        config=cfg,
    )


def _cores(grid, n):
    gx = grid.x
    full, rem = divmod(n, gx)
    rs = []
    if full:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, full - 1)))
    if rem:
        rs.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(rs), [(i % gx, i // gx) for i in range(n)]


def _rt(coords, fn):
    rt = ttnn.RuntimeArgs()
    for i, (cx, cy) in enumerate(coords):
        rt[cx][cy] = fn(i)
    return rt


def _unpack_modes(*idx):
    m = [ttnn.UnpackToDestMode.Default] * 64
    for i in idx:
        m[i] = ttnn.UnpackToDestMode.UnpackToDestFp32
    return m


class Hy4IHCPF:
    def __init__(self, md, w_t, base8, scale2, norm_w, hc_eps, norm_eps, magnitude=2.0, dim=6144, n=512):
        """w_t: the hc_fn tensor of Hy4IHCGen ([1,4,D,8] fp32 TILE: page (s, c) = s * D/32 + c, shared, no copy); norm_w: [D] (bf16-exact values: kept as a bf16 tile tensor, rows replicated)."""
        assert n % 32 == 0 and dim % (32 * CT) == 0
        self.md, self.dim, self.n = md, dim, n
        self.TR, self.NTD = n // 32, dim // 32
        self.NCH = self.NTD // CT
        self.NC = self.TR * self.NCH
        self.hc_eps, self.norm_eps = hc_eps, norm_eps
        grid = md.compute_with_storage_grid_size()
        assert grid.x * grid.y >= self.NC, "not enough worker cores"
        self.grid = grid
        self.w_t = w_t
        c = torch.zeros(5, 32, 32)
        c[0] = 1.0
        sc = torch.cat([scale2[0].float().reshape(1).repeat(4), scale2[1].float().reshape(1).repeat(4)])
        c[1, :, :8] = sc
        c[2, :, :8] = base8.float()
        c[3, :, :4] = 1.0
        c[3, :, 4:8] = magnitude
        c[4, :, :4] = hc_eps
        self.cst = upload(md, c.reshape(1, 1, 5 * 32, 32), f32)
        self.wn = upload(md, norm_w.float().reshape(1, 1, 1, dim).repeat(1, 1, 32, 1), bf16)
        key = (id(md), n, dim)
        sh = _CACHE.get(key)
        if sh is None:
            mk = lambda shape, dt: ttnn.allocate_tensor_on_device(
                ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
            )
            sh = _CACHE[key] = dict(
                part=mk([1, 1, self.NC * 2 * 32, 32], f32),
                accp=mk([1, 1, self.NC * 32, 32], f32),
                hbuf=mk([1, 1, n, dim], f32),
            )
        self.sh = sh

    # ------------------------------------------------------------------ programs
    def _run(self, tag, kernels, cbs, tensors, key):
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        prog.custom_program_hash = (tag << 40) | (hash(key) & ((1 << 40) - 1))
        ttnn.generic_op(tensors, prog)

    def pre(self, x4):
        md, TR, NTD, NCH, NC = self.md, self.TR, self.NTD, self.NCH, self.NC
        n, D = self.n, self.dim
        assert tuple(x4.shape) == (1, 4, n, D)
        part, accp, hbuf = self.sh["part"], self.sh["accp"], self.sh["hbuf"]
        mk = lambda shape, dt: ttnn.allocate_tensor_on_device(
            ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        st = mk([1, 1, TR * 32, 32], f32)
        hout = mk([1, 1, n, D], bf16)
        cs_all, co_all = _cores(self.grid, NC)
        cs_r, co_r = _cores(self.grid, TR)
        # ---- K1 stats
        B = 4
        cb = lambda *a: _cb(cs_all, *a)
        k1 = [
            _kernel(
                "k1_reader.cpp",
                cs_all,
                [0, 1, TR, NTD, CT, NCH, B] + _acc(x4) + _acc(self.w_t),
                _rt(co_all, lambda i: [i]),
                ttnn.ReaderConfigDescriptor(),
                [x4.buffer_address(), self.w_t.buffer_address()],
            ),
            _kernel(
                "w_generic.cpp",
                cs_all,
                [2, 4096, 2, 1, 0, 2, 0xFFFF, 0, 0] + _acc(part) + _acc(part),
                _rt(co_all, lambda i: [i * 2, 0]),
                ttnn.WriterConfigDescriptor(),
                [part.buffer_address(), part.buffer_address()],
            ),
            _kernel(
                "k1_compute.cpp",
                cs_all,
                [0, 1, 2, 4 * CT, B],
                ttnn.RuntimeArgs(),
                ttnn.ComputeConfigDescriptor(
                    math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
                ),
                [],
            ),
        ]
        self._run(
            0x4B49,
            k1,
            [cb(0, 16, 4096, f32), cb(1, 16, 4096, f32), cb(2, 2, 4096, f32)],
            [x4, self.w_t, part],
            ("k1", TR, NTD),
        )
        # ---- K2 combine
        cb = lambda *a: _cb(cs_r, *a)
        eps = _bits(self.norm_eps)
        cfg2 = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        cfg2.unpack_to_dest_mode = _unpack_modes(1, 5)
        k2 = [
            _kernel(
                "k2_reader.cpp",
                cs_r,
                [0, 1, NCH, 5] + _acc(self.cst) + _acc(part),
                _rt(co_r, lambda i: [i]),
                ttnn.ReaderConfigDescriptor(),
                [self.cst.buffer_address(), part.buffer_address()],
            ),
            _kernel(
                "w_generic.cpp",
                cs_r,
                [4, 4096, 1, 1, 0, 1, 0xFFFF, 0, 0] + _acc(st) + _acc(st),
                _rt(co_r, lambda i: [i, 0]),
                ttnn.WriterConfigDescriptor(),
                [st.buffer_address(), st.buffer_address()],
            ),
            _kernel(
                "k2_compute.cpp", cs_r, [0, 1, 2, 3, 4, 5, NCH, _bits(1.0 / (4 * D)), eps], ttnn.RuntimeArgs(), cfg2, []
            ),
        ]
        self._run(
            0x4B4A,
            k2,
            [
                cb(0, 1, 4096, f32),
                cb(1, 2 * NCH, 4096, f32),
                cb(2, 1, 4096, f32),
                cb(3, 1, 4096, f32),
                cb(4, 1, 4096, f32),
                cb(5, 4, 4096, f32),
            ],
            [self.cst, part, st],
            ("k2", TR, NCH, D, eps),
        )
        # ---- K3 collapse
        cb = lambda *a: _cb(cs_all, *a)
        cfg3 = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        cfg3.unpack_to_dest_mode = _unpack_modes(0, 1, 3)
        k3 = [
            _kernel(
                "k3_reader.cpp",
                cs_all,
                [0, 1, TR, NTD, CT, NCH, 4] + _acc(x4) + _acc(st),
                _rt(co_all, lambda i: [i]),
                ttnn.ReaderConfigDescriptor(),
                [x4.buffer_address(), st.buffer_address()],
            ),
            _kernel(
                "w_generic.cpp",
                cs_all,
                [2, 4096, CT, 1, 0, 4, 3, 4096, 1] + _acc(hbuf) + _acc(accp),
                _rt(co_all, lambda i: [(i // NCH) * NTD + (i % NCH) * CT, i]),
                ttnn.WriterConfigDescriptor(),
                [hbuf.buffer_address(), accp.buffer_address()],
            ),
            _kernel("k3_compute.cpp", cs_all, [0, 1, 2, 3, CT], ttnn.RuntimeArgs(), cfg3, []),
        ]
        self._run(
            0x4B4B,
            k3,
            [cb(0, 16, 4096, f32), cb(1, 4, 4096, f32), cb(2, 8, 4096, f32), cb(3, 1, 4096, f32), cb(4, 1, 4096, f32)],
            [x4, st, hbuf, accp],
            ("k3", TR, NTD),
        )
        # ---- K4 norm
        cfg4 = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        cfg4.unpack_to_dest_mode = _unpack_modes(1, 2, 5)
        k4 = [
            _kernel(
                "k4_reader.cpp",
                cs_all,
                [0, 1, 2, 3, NTD, CT, NCH] + _acc(self.cst) + _acc(accp) + _acc(hbuf) + _acc(self.wn),
                _rt(co_all, lambda i: [i]),
                ttnn.ReaderConfigDescriptor(),
                [self.cst.buffer_address(), accp.buffer_address(), hbuf.buffer_address(), self.wn.buffer_address()],
            ),
            _kernel(
                "w_generic.cpp",
                cs_all,
                [6, 2048, CT, 1, 0, 4, 0xFFFF, 0, 0] + _acc(hout) + _acc(hout),
                _rt(co_all, lambda i: [(i // NCH) * NTD + (i % NCH) * CT, 0]),
                ttnn.WriterConfigDescriptor(),
                [hout.buffer_address(), hout.buffer_address()],
            ),
            _kernel(
                "k4_compute.cpp",
                cs_all,
                [0, 1, 2, 3, 4, 5, 6, CT, NCH, _bits(1.0 / D), eps],
                ttnn.RuntimeArgs(),
                cfg4,
                [],
            ),
        ]
        self._run(
            0x4B4C,
            k4,
            [
                cb(0, 1, 4096, f32),
                cb(1, NCH, 4096, f32),
                cb(2, 4, 4096, f32),
                cb(3, 4, 2048, bf16),
                cb(4, 1, 4096, f32),
                cb(5, 1, 4096, f32),
                cb(6, 4, 2048, bf16),
            ],
            [self.cst, accp, hbuf, self.wn, hout],
            ("k4", TR, NTD, NCH, D, eps),
        )
        return hout, st

    def expand(self, x4, y, st):
        """x4 [1,4,n,D] fp32, y [1,1,n,D] bf16 TILE, st (from pre) -> x4 + post * y (fp32 [1,4,n,D]). Does not free its inputs."""
        md, TR, NTD, NCH, NC, n, D = self.md, self.TR, self.NTD, self.NCH, self.NC, self.n, self.dim
        assert y.dtype == bf16
        out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 4, n, D]), f32, ttnn.TILE_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG
        )
        cs_all, co_all = _cores(self.grid, NC)
        cb = lambda *a: _cb(cs_all, *a)
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        cfg.unpack_to_dest_mode = _unpack_modes(0, 2)
        k5 = [
            _kernel(
                "k5_reader.cpp",
                cs_all,
                [0, 1, 2, TR, NTD, CT, NCH, 4] + _acc(x4) + _acc(y) + _acc(st),
                _rt(co_all, lambda i: [i]),
                ttnn.ReaderConfigDescriptor(),
                [x4.buffer_address(), y.buffer_address(), st.buffer_address()],
            ),
            _kernel(
                "w_generic.cpp",
                cs_all,
                [3, 4096, CT, 4, TR * NTD, 2, 0xFFFF, 0, 0] + _acc(out) + _acc(out),
                _rt(co_all, lambda i: [(i // NCH) * NTD + (i % NCH) * CT, 0]),
                ttnn.WriterConfigDescriptor(),
                [out.buffer_address(), out.buffer_address()],
            ),
            _kernel("k5_compute.cpp", cs_all, [0, 1, 2, 3, CT], ttnn.RuntimeArgs(), cfg, []),
        ]
        self._run(
            0x4B4D,
            k5,
            [
                cb(0, 16, 4096, f32),
                cb(1, 4, 2048, bf16),
                cb(2, 4, 4096, f32),
                cb(3, 16, 4096, f32),
                cb(4, 1, 4096, f32),
            ],
            [x4, y, st, out],
            ("k5", TR, NTD),
        )
        return out

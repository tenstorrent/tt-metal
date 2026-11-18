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
CT = int(
    os.environ.get("HY4_B49_CT", "64")
)  # column tiles per core (cores = n/32 * D/32 / CT; host launch cost ~0.45 ms at 96 cores x 32 chips)
MERGE = (
    os.environ.get("HY4_B49_MERGE", "1") == "1"
)  # PRE as ONE program (one core per token tile-row, no cross-core exchange): 1 launch instead of 4
CTE = int(
    os.environ.get("HY4_B49_CTE", "192" if MERGE else "0")
)  # column tiles per core of the expand program (0 = CT)
_CACHE = {}
HOST_T = {}  # name -> [calls, seconds] host time of pre / expand / each program launch (B49 profiling)


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
        self.CTE = CTE or CT
        self.NCHE = self.NTD // self.CTE
        self.NCE = self.TR * self.NCHE
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
        sh = None
        if (
            not MERGE
        ):  # the 4-program variant needs shared scratch (partials, h row); the merged program keeps everything in L1
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
    def _launch(self, tag, key, build, tensors, commons):
        """Builds the program descriptor of this call (``build()`` -> (kernels, cbs); ``commons[i]`` = buffer addresses of kernels[i]) and launches it; the program hash is a constant of (tag, key).
        (Caching the descriptor and re-assigning the common runtime args does NOT work: ProgramDescriptor copies its kernels.)
        """
        import time

        t0 = time.perf_counter()
        kernels, cbs = build()
        for k, c in zip(kernels, commons):
            k.common_runtime_args = c
        prog = ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
        prog.custom_program_hash = (tag << 40) | (hash(key) & ((1 << 40) - 1))
        t1 = time.perf_counter()
        ttnn.generic_op(tensors, prog)
        t2 = time.perf_counter()
        e = HOST_T.setdefault(hex(tag), [0, 0.0, 0.0])
        e[0] += 1
        e[1] += t2 - t0
        e[2] += t2 - t1  # generic_op call only

    def _alloc(self, shape, dt):
        return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dt, ttnn.TILE_LAYOUT, self.md, ttnn.DRAM_MEMORY_CONFIG)

    def _cfg(self, *unpack_dest_cbs):
        cfg = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        if unpack_dest_cbs:
            cfg.unpack_to_dest_mode = _unpack_modes(*unpack_dest_cbs)
        return cfg

    def pre(self, x4):
        if MERGE:
            return self.pre_merged(x4)
        return self.pre4(x4)

    def pre_merged(self, x4):
        TR, NTD, D, n = self.TR, self.NTD, self.dim, self.n
        assert tuple(x4.shape) == (1, 4, n, D)
        st = self._alloc([1, 1, TR * 32, 32], f32)  # kept for the expand: the pp tiles are rebuilt there from this
        hout = self._alloc([1, 1, n, D], bf16)
        eps = _bits(self.norm_eps)
        A = lambda *ts: [t.buffer_address() for t in ts]

        def build():
            cs, co = _cores(self.grid, TR)
            cfg = self._cfg(3, 4, 7, 8, 9, 10, 12)
            return (
                [
                    _kernel(
                        "p_reader.cpp",
                        cs,
                        [0, 1, 2, 3, 6, 7, 13, 8, TR, NTD]
                        + _acc(x4)
                        + _acc(self.w_t)
                        + _acc(self.cst)
                        + _acc(self.wn)
                        + _acc(st),
                        _rt(co, lambda i: [i]),
                        ttnn.ReaderConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "w_generic.cpp",
                        cs,
                        [14, 2048, NTD, 1, 0, 4, 0xFFFF, 0, 0] + _acc(hout) + _acc(hout),
                        _rt(co, lambda i: [i * NTD, 0]),
                        ttnn.WriterConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "p_compute.cpp",
                        cs,
                        [
                            0,
                            1,
                            2,
                            3,
                            4,
                            5,
                            6,
                            7,
                            8,
                            9,
                            10,
                            11,
                            12,
                            13,
                            14,
                            NTD,
                            _bits(1.0 / (4 * D)),
                            eps,
                            _bits(1.0 / D),
                        ],
                        ttnn.RuntimeArgs(),
                        cfg,
                        [],
                    ),
                ],
                [
                    _cb(cs, 0, 16, 4096, f32),
                    _cb(cs, 1, 16, 4096, f32),
                    _cb(cs, 2, 1, 4096, f32),
                    _cb(cs, 3, 4, 4096, f32),
                    _cb(cs, 4, 1, 4096, f32),
                    _cb(cs, 5, 1, 4096, f32),
                    _cb(cs, 6, 1, 4096, f32),
                    _cb(cs, 7, 4, 4096, f32),
                    _cb(cs, 8, 16, 4096, f32),
                    _cb(cs, 9, 1, 4096, f32),
                    _cb(cs, 10, 1, 4096, f32),
                    _cb(cs, 11, 1, 4096, f32),
                    _cb(cs, 12, 1, 4096, f32),
                    _cb(cs, 13, 4, 2048, bf16),
                    _cb(cs, 14, 4, 2048, bf16),
                ],
            )

        self._launch(
            0x4B4E,
            ("pm", TR, NTD, eps),
            build,
            [x4, self.w_t, self.cst, self.wn, st, hout],
            [A(x4, self.w_t, self.cst, self.wn, st), A(hout, hout), []],
        )
        return hout, st

    def pre4(self, x4):
        md, TR, NTD, NCH, NC = self.md, self.TR, self.NTD, self.NCH, self.NC
        n, D = self.n, self.dim
        assert tuple(x4.shape) == (1, 4, n, D)
        part, accp, hbuf = self.sh["part"], self.sh["accp"], self.sh["hbuf"]
        st = self._alloc([1, 1, TR * 32, 32], f32)
        hout = self._alloc([1, 1, n, D], bf16)
        A = lambda *ts: [t.buffer_address() for t in ts]
        eps = _bits(self.norm_eps)

        def build_k1():
            cs, co = _cores(self.grid, NC)
            return (
                [
                    _kernel(
                        "k1_reader.cpp",
                        cs,
                        [0, 1, TR, NTD, CT, NCH, 4] + _acc(x4) + _acc(self.w_t),
                        _rt(co, lambda i: [i]),
                        ttnn.ReaderConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "w_generic.cpp",
                        cs,
                        [2, 4096, 2, 1, 0, 2, 0xFFFF, 0, 0] + _acc(part) + _acc(part),
                        _rt(co, lambda i: [i * 2, 0]),
                        ttnn.WriterConfigDescriptor(),
                        [],
                    ),
                    _kernel("k1_compute.cpp", cs, [0, 1, 2, 4 * CT, 4], ttnn.RuntimeArgs(), self._cfg(), []),
                ],
                [_cb(cs, 0, 16, 4096, f32), _cb(cs, 1, 16, 4096, f32), _cb(cs, 2, 2, 4096, f32)],
            )

        self._launch(0x4B49, ("k1", TR, NTD, CT), build_k1, [x4, self.w_t, part], [A(x4, self.w_t), A(part, part), []])

        def build_k2():
            cs, co = _cores(self.grid, TR)
            return (
                [
                    _kernel(
                        "k2_reader.cpp",
                        cs,
                        [0, 1, NCH, 5] + _acc(self.cst) + _acc(part),
                        _rt(co, lambda i: [i]),
                        ttnn.ReaderConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "w_generic.cpp",
                        cs,
                        [4, 4096, 1, 1, 0, 1, 0xFFFF, 0, 0] + _acc(st) + _acc(st),
                        _rt(co, lambda i: [i, 0]),
                        ttnn.WriterConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "k2_compute.cpp",
                        cs,
                        [0, 1, 2, 3, 4, 5, NCH, _bits(1.0 / (4 * D)), eps],
                        ttnn.RuntimeArgs(),
                        self._cfg(1, 5),
                        [],
                    ),
                ],
                [
                    _cb(cs, 0, 1, 4096, f32),
                    _cb(cs, 1, 2 * NCH, 4096, f32),
                    _cb(cs, 2, 1, 4096, f32),
                    _cb(cs, 3, 1, 4096, f32),
                    _cb(cs, 4, 1, 4096, f32),
                    _cb(cs, 5, 4, 4096, f32),
                ],
            )

        self._launch(
            0x4B4A, ("k2", TR, NCH, D, eps, CT), build_k2, [self.cst, part, st], [A(self.cst, part), A(st, st), []]
        )

        def build_k3():
            cs, co = _cores(self.grid, NC)
            return (
                [
                    _kernel(
                        "k3_reader.cpp",
                        cs,
                        [0, 1, TR, NTD, CT, NCH, 4] + _acc(x4) + _acc(st),
                        _rt(co, lambda i: [i]),
                        ttnn.ReaderConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "w_generic.cpp",
                        cs,
                        [2, 4096, CT, 1, 0, 4, 3, 4096, 1] + _acc(hbuf) + _acc(accp),
                        _rt(co, lambda i: [(i // NCH) * NTD + (i % NCH) * CT, i]),
                        ttnn.WriterConfigDescriptor(),
                        [],
                    ),
                    _kernel("k3_compute.cpp", cs, [0, 1, 2, 3, CT], ttnn.RuntimeArgs(), self._cfg(0, 1, 3), []),
                ],
                [
                    _cb(cs, 0, 16, 4096, f32),
                    _cb(cs, 1, 4, 4096, f32),
                    _cb(cs, 2, 8, 4096, f32),
                    _cb(cs, 3, 1, 4096, f32),
                    _cb(cs, 4, 1, 4096, f32),
                ],
            )

        self._launch(0x4B4B, ("k3", TR, NTD, CT), build_k3, [x4, st, hbuf, accp], [A(x4, st), A(hbuf, accp), []])

        def build_k4():
            cs, co = _cores(self.grid, NC)
            return (
                [
                    _kernel(
                        "k4_reader.cpp",
                        cs,
                        [0, 1, 2, 3, NTD, CT, NCH] + _acc(self.cst) + _acc(accp) + _acc(hbuf) + _acc(self.wn),
                        _rt(co, lambda i: [i]),
                        ttnn.ReaderConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "w_generic.cpp",
                        cs,
                        [6, 2048, CT, 1, 0, 4, 0xFFFF, 0, 0] + _acc(hout) + _acc(hout),
                        _rt(co, lambda i: [(i // NCH) * NTD + (i % NCH) * CT, 0]),
                        ttnn.WriterConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "k4_compute.cpp",
                        cs,
                        [0, 1, 2, 3, 4, 5, 6, CT, NCH, _bits(1.0 / D), eps],
                        ttnn.RuntimeArgs(),
                        self._cfg(1, 2, 5),
                        [],
                    ),
                ],
                [
                    _cb(cs, 0, 1, 4096, f32),
                    _cb(cs, 1, NCH, 4096, f32),
                    _cb(cs, 2, 4, 4096, f32),
                    _cb(cs, 3, 4, 2048, bf16),
                    _cb(cs, 4, 1, 4096, f32),
                    _cb(cs, 5, 1, 4096, f32),
                    _cb(cs, 6, 4, 2048, bf16),
                ],
            )

        self._launch(
            0x4B4C,
            ("k4", TR, NTD, NCH, D, eps, CT),
            build_k4,
            [self.cst, accp, hbuf, self.wn, hout],
            [A(self.cst, accp, hbuf, self.wn), A(hout, hout), []],
        )
        return hout, st

    def expand(self, x4, y, st):
        """x4 [1,4,n,D] fp32, y [1,1,n,D] bf16 TILE, st (from pre) -> x4 + post * y (fp32 [1,4,n,D]). Does not free its inputs."""
        TR, NTD, NCH, NC, n, D, CTx = self.TR, self.NTD, self.NCHE, self.NCE, self.n, self.dim, self.CTE
        assert y.dtype == bf16
        out = self._alloc([1, 4, n, D], f32)

        def build_k5():
            cs, co = _cores(self.grid, NC)
            return (
                [
                    _kernel(
                        "k5_reader.cpp",
                        cs,
                        [0, 1, 2, TR, NTD, CTx, NCH, 4] + _acc(x4) + _acc(y) + _acc(st),
                        _rt(co, lambda i: [i]),
                        ttnn.ReaderConfigDescriptor(),
                        [],
                    ),
                    _kernel(
                        "w_generic.cpp",
                        cs,
                        [3, 4096, CTx, 4, TR * NTD, 2, 0xFFFF, 0, 0] + _acc(out) + _acc(out),
                        _rt(co, lambda i: [(i // NCH) * NTD + (i % NCH) * CTx, 0]),
                        ttnn.WriterConfigDescriptor(),
                        [],
                    ),
                    _kernel("k5_compute.cpp", cs, [0, 1, 2, 3, CTx], ttnn.RuntimeArgs(), self._cfg(0, 2), []),
                ],
                [
                    _cb(cs, 0, 16, 4096, f32),
                    _cb(cs, 1, 4, 2048, bf16),
                    _cb(cs, 2, 4, 4096, f32),
                    _cb(cs, 3, 16, 4096, f32),
                    _cb(cs, 4, 1, 4096, f32),
                ],
            )

        A = lambda *ts: [t.buffer_address() for t in ts]
        self._launch(0x4B4D, ("k5", TR, NTD, CTx), build_k5, [x4, y, st, out], [A(x4, y, st), A(out, out), []])
        return out

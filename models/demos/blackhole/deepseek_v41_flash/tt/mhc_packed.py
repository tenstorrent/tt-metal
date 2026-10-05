# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""mHC for PREFILL on the PACKED stream layout x [1,1,N,4C] fp32 (token rows, stream i at columns [i*C, (i+1)*C)) instead of the
decode layout [T,1,4,C] whose tiles hold 4 rows of 32 (8x padding). Works on N = 32*R tokens at once (no 32-token loop).

  mixes:    pk_proj generic_op (K split over S cores, all token tile rows) -> partials [R,S,32,32]; sum over S; RMS scale; the
            deepseek_prefill ``mhc_split_sinkhorn`` op (multi-core over token tiles) -> pre [N,4], post [N,4], comb [N,16]
  coefs:    pre/post/comb -> 'column-0 tiles' (tile q holds coefficient q of every token in column 0) with one selection matmul each
  collapse: pk_comb generic_op, h_bf16 = sum_i diag(pre_i) @ x_i  (row scaling = fp32 tile matmul accumulated in DST), then ttnn.rms_norm
  expand:   pk_comb generic_op, new_j = sum_i diag(comb_ij) @ x_i + diag(post_j) @ (y [+ y2])
"""
import os

import torch

import ttnn
from models.demos.blackhole.deepseek_v41_flash.tt.mhc_collapse import _acc, _cb, _hash, _kernel

f32, bf16 = ttnn.float32, ttnn.bfloat16
TILE = 4096


def _flag(name, default="0"):
    return os.environ.get(name, default) == "1"


def _cores(mesh, n):
    g = mesh.compute_with_storage_grid_size()
    cs = [(cx, cy) for cy in range(g.y) for cx in range(g.x)]
    return cs[:n]


def _divisors_le(n, m):
    return [d for d in range(1, min(n, m) + 1) if n % d == 0]


def _core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(cx, cy), ttnn.CoreCoord(cx, cy)) for cx, cy in cores])


def pk_proj(x, wt, ncol, S=None):
    """x [1,1,N,4C] fp32 TILE; wt [4C,32]-tile fn^T (fp32, K rows) -> partials [R,S,32,32] fp32 (R = N/32): tile (r,k) = chunk-k projection of the
    32 tokens of row r (columns 0..mix_hc-1) and their sum of squares (column ncol)."""
    N, W = int(x.shape[-2]), int(x.shape[-1])
    assert N % 32 == 0 and W % 128 == 0
    R, XT = N // 32, W // 32
    mesh = x.device()
    S = S or int(os.environ.get("DSV41_PKPROJ_S", "64"))
    while XT % S:
        S -= 1
    TPC = XT // S
    part = ttnn.allocate_tensor_on_device(
        ttnn.Shape([R, S, 32, 32]), f32, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    cores = _cores(mesh, S)
    assert len(cores) == S, (len(cores), S)
    core_set = _core_set(cores)
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for k, (cx, cy) in enumerate(cores):
        rt[cx][cy] = [k]
        crt[cx][cy] = []
    X, Wc, ONES, SQ, P = range(5)
    cbs = [
        _cb(core_set, X, 2 * TPC, TILE, f32),
        _cb(core_set, Wc, TPC, TILE, f32),
        _cb(core_set, ONES, 1, TILE, f32),
        _cb(core_set, SQ, TPC, TILE, f32),
        _cb(core_set, P, 2, TILE, f32),
    ]
    reader = _kernel(
        "pk_proj_reader.cpp",
        core_set,
        [X, Wc, ONES, R, XT, TPC, ncol] + _acc(x) + _acc(wt),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[x.buffer_address(), wt.buffer_address()],
    )
    writer = _kernel(
        "pk_proj_writer.cpp",
        core_set,
        [P, R, S] + _acc(part),
        rt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[part.buffer_address()],
    )
    compute = _kernel(
        "pk_proj_compute.cpp",
        core_set,
        [X, Wc, ONES, SQ, P, TPC, R],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(0x6B1, R, XT, S, ncol, tuple(_acc(x)), tuple(_acc(wt)), tuple(_acc(part)))
    ttnn.generic_op([x, wt, part], prog)
    return part


def pk_comb(x, cta, ctb, y=None, y2=None, nout=4, out_dtype=f32, max_cores=None):
    """Packed row-scaling combine, see pk_comb_*.cpp.  x [1,1,N,4C] fp32; cta [N,32*NCA] (coefficient tiles; NCA = 16 comb (expand) or 4 pre (collapse)),
    ctb [N,32*4] post tiles (expand) or None; y / y2 [1,1,N,C] fp32.  nout=4 -> [1,1,N,4C] out_dtype; nout=1 -> [1,1,N,C].
    """
    N, W = int(x.shape[-2]), int(x.shape[-1])
    CT = W // 128
    R = N // 32
    NCA = int(cta.shape[-1]) // 32
    NCB = 0 if ctb is None else int(ctb.shape[-1]) // 32
    mesh = x.device()
    g = mesh.compute_with_storage_grid_size()
    max_cores = max_cores or int(g.x) * int(g.y)
    per_row = max(1, max_cores // R)
    cs_opts = [d for d in _divisors_le(CT, CT) if CT // d <= per_row]  # d = column tiles per core
    cs = min(cs_opts)
    nrc = CT // cs  # cores per row
    BLK = max(d for d in _divisors_le(cs, 4))
    cores = _cores(mesh, R * nrc)
    assert len(cores) == R * nrc
    core_set = _core_set(cores)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, N, CT * 32 * nout]), out_dtype, ttnn.TILE_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )
    otile = TILE if out_dtype == f32 else 2048
    rt, crt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for c, (cx, cy) in enumerate(cores):
        r, p = divmod(c, nrc)
        rt[cx][cy] = [r, p * cs, cs]
        crt[cx][cy] = [cs]
    CB_I, CB_CT, CB_X, CB_Y, CB_Y2, CB_D, CB_O = range(7)
    NC = NCA + NCB
    cbs = [
        _cb(core_set, CB_I, 1, TILE, f32),
        _cb(core_set, CB_CT, NC, TILE, f32),
        _cb(core_set, CB_X, 8 * BLK, TILE, f32),
        _cb(core_set, CB_Y, 2 * BLK, TILE, f32),
        _cb(core_set, CB_Y2, 2 * BLK, TILE, f32),
        _cb(core_set, CB_D, NC, TILE, f32),
        _cb(core_set, CB_O, 2 * max(nout, 1), otile, out_dtype),
    ]
    yt = y if y is not None else x
    y2t = y2 if y2 is not None else x
    btt = ctb if ctb is not None else cta
    reader = _kernel(
        "pk_comb_reader.cpp",
        core_set,
        [CB_I, CB_CT, CB_X, CB_Y, CB_Y2, NCA, NCB, CT, BLK, int(y is not None), int(y2 is not None)]
        + _acc(x)
        + _acc(cta)
        + _acc(btt)
        + _acc(yt)
        + _acc(y2t),
        rt,
        None,
        ttnn.ReaderConfigDescriptor(),
        common=[
            x.buffer_address(),
            cta.buffer_address(),
            btt.buffer_address(),
            yt.buffer_address(),
            y2t.buffer_address(),
        ],
    )
    writer = _kernel(
        "pk_comb_writer.cpp",
        core_set,
        [CB_O, CT, nout, otile, BLK] + _acc(out),
        rt,
        None,
        ttnn.WriterConfigDescriptor(),
        common=[out.buffer_address()],
    )
    compute = _kernel(
        "pk_comb_compute.cpp",
        core_set,
        [CB_I, CB_CT, CB_X, CB_Y, CB_Y2, CB_D, CB_O, NC, NCA, BLK, nout, int(y is not None), int(y2 is not None)],
        crt,
        None,
        ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True),
    )
    prog = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)
    prog.custom_program_hash = _hash(
        0x6B2,
        R,
        CT,
        NCA,
        NCB,
        nout,
        str(out_dtype),
        cs,
        BLK,
        y is not None,
        y2 is not None,
        tuple(_acc(x)),
        tuple(_acc(cta)),
        tuple(_acc(btt)),
        tuple(_acc(yt)),
        tuple(_acc(y2t)),
        tuple(_acc(out)),
    )
    ttnn.generic_op([x, cta, btt, yt, y2t, out], prog)
    return out


class PackedMHC:
    """Packed-layout mHC pieces of one sub-block, built on a DSV41MHC (shares its fp32 constants)."""

    def __init__(self, mhc):
        self.m = mhc
        w = mhc._w
        self.w = w
        dev = mhc.device
        self.dev = dev
        self.n, self.C = mhc.n, mhc.dim
        mix_hc = mhc.mix_col
        self.ncol = mix_hc  # sum of squares in column mix_hc of the partial tiles
        up = lambda t: ttnn.from_torch(
            t.contiguous(),
            device=dev,
            dtype=f32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(dev),
        )
        # fn^T [4C, 32] (columns >= mix_hc zero)
        ft = torch.zeros(self.n * self.C, 32)
        ft[:, :mix_hc] = mhc._fn.t().float()
        self.wt = up(ft.reshape(1, 1, self.n * self.C, 32))
        sel = lambda k: torch.zeros(k, 32 * k)
        e4, e16 = sel(4), sel(16)
        for q in range(4):
            e4[q, q * 32] = 1.0
        for q in range(16):
            e16[q, q * 32] = 1.0
        self.e4, self.e16 = up(e4.reshape(1, 1, 4, 128)), up(e16.reshape(1, 1, 16, 512))
        sel_ss = torch.zeros(32, 32)
        sel_ss[mix_hc, :] = 1.0
        self.sel_ss = up(sel_ss.reshape(1, 1, 32, 32))
        self.ckc = w.ckc
        self.norm_ckc = ttnn.init_device_compute_kernel_config(
            dev.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._nw = {}

    def mixes(self, x):
        """x [1,1,N,4C] fp32 -> (pre [1,1,N,4], post [1,1,N,4], comb [1,1,N,16]) fp32."""
        N = int(x.shape[-2])
        part = pk_proj(x, self.wt, self.ncol)
        R = N // 32
        mu = ttnn.sum(part, dim=1, keepdim=False)  # [R,32,32]
        mu = ttnn.reshape(mu, [1, 1, N, 32])
        ss = ttnn.matmul(mu, self.sel_ss, compute_kernel_config=self.ckc)  # every column = sum of squares
        rs = ttnn.rsqrt(ttnn.add(ttnn.multiply(ss, 1.0 / (self.n * self.C)), self.w.norm_eps))
        mixes = ttnn.multiply(mu, rs)
        ttnn.deallocate(ss), ttnn.deallocate(rs), ttnn.deallocate(mu), ttnn.deallocate(part)
        pre, post, comb = ttnn.experimental.deepseek_prefill.mhc_split_sinkhorn(
            mixes, self.w.consts, self.n, self.w.iters, self.w.eps
        )
        ttnn.deallocate(mixes)
        r4 = lambda t: ttnn.reshape(t, [1, 1, N, t.shape[-1]])
        return r4(pre), r4(post), r4(comb)

    def coefs_pre(self, pre):
        return ttnn.matmul(pre, self.e4, compute_kernel_config=self.ckc)  # [1,1,N,128]

    def coefs_expand(self, post, comb):
        return (
            ttnn.matmul(comb, self.e16, compute_kernel_config=self.ckc),  # [1,1,N,512]
            ttnn.matmul(post, self.e4, compute_kernel_config=self.ckc),  # [1,1,N,128]
        )

    def collapse(self, x, pre):
        """-> h bf16 [1,1,N,C] = bf16(sum_i pre_i x_i) (no norm)."""
        ct = self.coefs_pre(pre)
        h = pk_comb(x, ct, None, nout=1, out_dtype=bf16)
        ttnn.deallocate(ct)
        return h

    def collapse_norm(self, x, pre, w, eps):
        """bf16 [1,1,N,C] = rms_norm(bf16(collapse), weight=w) (w: [1,1,1,C] fp32 tile row)."""
        h = self.collapse(x, pre)
        o = ttnn.rms_norm(h, epsilon=eps, weight=w, compute_kernel_config=self.norm_ckc)
        ttnn.deallocate(h)
        return o

    def expand(self, y, x, post, comb, y2=None):
        """new_j = post_j * (y [+ y2]) + sum_i comb[i,j] x_i -> [1,1,N,4C] fp32; y / y2 [1,1,N,C] fp32."""
        cc, cp = self.coefs_expand(post, comb)
        o = pk_comb(x, cc, cp, y=y, y2=y2, nout=4, out_dtype=f32)
        ttnn.deallocate(cc), ttnn.deallocate(cp)
        return o

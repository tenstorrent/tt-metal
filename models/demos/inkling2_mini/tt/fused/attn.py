# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused Inkling-2-Mini attention glue (ttnn.generic_op, JIT kernels, many cores).

`AttnPrep`: one program from the fused q/k/v projection (the conv window [1, 1, 32, N] bf16) to the sdpa inputs,
replacing ~45 ttnn ops per layer application:
- head role, 24 cores: per kv head the k / v short conv (+ dynamic taps on global layers), k L2 clip and v RMS
  norm, rows written straight into the ring caches (and the global far cache), bf16 or bfp8, in place of
  paged_update_cache; per GQA group the q pre-scale + L2 clip (+ tau) -> q [1, 1, 24, 128].
- bias role, 8 cores x 4 slot tiles: rel clip (+ tau) and the relative bias rel @ proj[:, dist(slot)] + near mask
  -> mask [1, 1, 24, 1024] bf16; the first core also builds the sink tile [24, 32].
`AttnOut`: attention output * sigmoid(gate) -> row 0 of the o-projection's L1 input shards (8 cores).
"""

import struct

import torch

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F

HEAD_DIM = 128
N_HEADS = 24
N_KV = 8
GROUPS = 3
D_REL = 16
WINDOW = 1024
TAPS = 4
TABLE_ROWS = WINDOW + 34  # wrap rows so a 1088-byte aligned read never runs past the end
NJ = 4  # bias slot tiles per core
NEG = -1.0e9  # masked logit (attention.py)


def f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _cb(index, dtype, n, cores, page=None):
    page = page or (4096 if dtype == ttnn.float32 else 2048)
    fmt = ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page)
    return ttnn.CBDescriptor(total_size=n * page, core_ranges=cores, format_descriptors=[fmt])


def _stage(index, cores, size=2048 + 128):
    return _cb(index, ttnn.bfloat16, 1, cores, page=size)


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _kernel(name, cores, named, rt, config):
    return ttnn.KernelDescriptor(
        kernel_source=f"{F.KDIR}/{name}",
        core_ranges=_crs(cores),
        named_compile_time_args=list(named.items()),
        runtime_args=[(c, [int(v) for v in a]) for c, a in zip(cores, rt)],
        config=config,
    )


_SHARED = {}


def _shared_l1(device, name, c, mem):
    """Zeroed [1, 1, 32, c] bf16 L1 input shared by every layer: each layer fills it and its matmul reads it right
    away, so one buffer per shape is enough (24 per-layer copies would take ~1 MB of L1 per core)."""
    key = (id(device), name, c)  # the entry keeps `device` alive, so its id is not reused
    if key not in _SHARED or not _SHARED[key][1].is_allocated():  # (a closed and reopened device frees it)
        t = ttnn.from_torch(
            torch.zeros(1, 1, 32, c), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )
        _SHARED[key] = (device, t)
    return _SHARED[key][1]


def _compute_config(fp32_cbs):
    return F.compute_config(fp32_cbs)


class AttnPrep:
    """Per attention layer. q_scale [24, 128], rel_scale [24, 16], rel_proj [16, 1024] (torch, fp32): the module's
    parameters; taps[kv] [4, 1024]: short-conv taps per lag (residual and conv scale folded in); q_inv / k_inv:
    1 / clip threshold^2. dyn: global layers (dynamic taps scaled by conv_scale, tau on q and rel)."""

    def __init__(self, device, q_scale, rel_scale, rel_proj, taps, q_inv, k_inv, rel_inv, eps, kv0, dyn, conv_scale):
        self.device = device
        self.dyn = dyn
        self.kv0 = kv0
        self.q_inv, self.k_inv, self.rel_inv, self.eps, self.cs = q_inv, k_inv, rel_inv, eps, conv_scale
        put = lambda t, dt=ttnn.float32, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(  # noqa: E731
            t, dtype=dt, layout=lay, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        self.q_taps = F.dense_tensor(device, q_scale.reshape(N_HEADS, HEAD_DIM))
        self.kv_taps = {
            kv: F.dense_tensor(device, taps[kv].view(TAPS, N_KV, HEAD_DIM).transpose(0, 1).reshape(-1, HEAD_DIM))
            for kv in ("k", "v")
        }
        rs = torch.zeros(32, 32)
        rs[:N_HEADS, :D_REL] = rel_scale.reshape(N_HEADS, D_REL)
        self.rel_scale = put(rs.view(1, 1, 32, 32))
        g = torch.zeros(32, 32)
        g[:D_REL, :D_REL] = 1.0
        self.g = put(g.view(1, 1, 32, 32), ttnn.bfloat16)
        u = torch.arange(TABLE_ROWS)
        table = rel_proj.reshape(D_REL, WINDOW).T[(-u) % WINDOW]  # row u = profile at distance (-u) mod 1024
        self.table = put(table.reshape(1, 1, 1, -1), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT)
        m = torch.zeros(1, 1, 32, 32)
        m[..., 0, :] = 1.0
        m[..., 16, :] = 1.0
        self.scaler = put(m)

        zeros = lambda shape, mem=ttnn.DRAM_MEMORY_CONFIG: ttnn.from_torch(  # noqa: E731
            torch.zeros(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=mem
        )
        self.q = zeros([1, 1, N_HEADS, HEAD_DIM])
        self.mask = zeros([1, 1, N_HEADS, WINDOW])
        self.sink = zeros([N_HEADS, 32])
        self.sink_blk = None
        if dyn:
            self.sink_blk = ttnn.from_torch(
                torch.zeros(1, N_KV, GROUPS, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
            )
        mass = torch.zeros(1, HEAD_DIM)
        mass[0, 0] = 1.0  # far-cache mass chunk [1, 0 x 15] (+ a zero chunk), packed like the v row
        self.mass = F.dense_tensor(device, mass)

        cores, _ = F.core_list(device, 3 * N_KV + N_KV, first_row=1)
        self.q_cores, self.k_cores = cores[:N_KV], cores[N_KV : 2 * N_KV]
        self.v_cores, self.b_cores = cores[2 * N_KV : 3 * N_KV], cores[3 * N_KV :]
        self.cache = {}

    def _head_group(self, cores, proj, kind, mode_rows, tau, state, positions):
        """kind: q / k / v. Returns (kernels, cbs) for one group of 8 cores."""
        q = kind == "q"
        nitems, nlag = (GROUPS, 1) if q else (1, TAPS)
        dyn = int(self.dyn and not q)
        has_tau = int(self.dyn and q)
        mass = int(self.dyn and kind == "v")
        r_rt, w_rt = [], []
        for g, _ in enumerate(cores):
            r = [proj.buffer_address(), (self.q_taps if q else self.kv_taps[kind]).buffer_address()]
            r += [self.scaler.buffer_address(), tau.buffer_address() if has_tau else 0, self.mass.buffer_address()]
            if q:
                w = []
                for i in range(GROUPS):
                    h = g * GROUPS + i
                    r += [h * HEAD_DIM // 32, mode_rows["q"], h]
                    w += [self.q.buffer_address(), 0, 0, 0, 0, h]
            else:
                c0 = self.kv0 + (0 if kind == "k" else N_KV * HEAD_DIM) + g * HEAD_DIM
                r += [c0 // 32, mode_rows[kind], g * TAPS]
                if dyn:
                    r += [self.kv0 + 2 * N_KV * HEAD_DIM + (0 if kind == "k" else N_KV * TAPS) + g * TAPS]
                ring = state.k_ring if kind == "k" else state.v_ring
                w = [g, ring.buffer_address(), WINDOW // 32, HEAD_DIM // 32, 0, positions.ring.buffer_address()]
                if self.dyn:
                    full = state.kv_full
                    seq = full.padded_shape[-2] // 32
                    w += [full.buffer_address(), seq, full.padded_shape[-1] // 32, 5 if kind == "k" else 0]
                    w += [positions.pos.buffer_address()]
            r_rt.append(r)
            w_rt.append(w)
        named = dict(nitems=nitems, nlag=nlag, dyn=dyn, has_tau=has_tau, mass=mass)
        c_named = dict(
            named,
            norm=int(kind == "v"),
            inv=f32_bits(self.q_inv if q else (1.0 / HEAD_DIM if kind == "v" else self.k_inv)),
            eps=f32_bits(self.eps),
            cs=f32_bits(self.cs),
        )
        crs = _crs(cores)
        n = nitems * nlag
        if q:
            writer = _kernel(
                "attn_head_writer.cpp", cores, dict(nitems=nitems, ndst=1), w_rt, ttnn.WriterConfigDescriptor()
            )
            out_dtype = ttnn.bfloat16
        else:
            out_dtype = state.k_ring.dtype
            assert out_dtype in (ttnn.bfloat16, ttnn.bfloat8_b), out_dtype
            assert not self.dyn or state.kv_full.dtype == out_dtype
            named_w = dict(ndst=1 + int(self.dyn), bfp8=int(out_dtype == ttnn.bfloat8_b), mass=mass)
            writer = _kernel("attn_kv_writer.cpp", cores, named_w, w_rt, ttnn.WriterConfigDescriptor())
        out_page = 1088 if out_dtype == ttnn.bfloat8_b else 2048
        kernels = [
            _kernel("attn_head_reader.cpp", cores, named, r_rt, ttnn.ReaderConfigDescriptor()),
            writer,
            _kernel("attn_head_compute.cpp", cores, c_named, [[]] * len(cores), _compute_config([1, 2, 4, 5])),
        ]
        cbs = [
            _cb(0, ttnn.bfloat16, n, crs),
            _cb(1, ttnn.float32, n, crs),
            _cb(2, ttnn.float32, TAPS, crs),
            _cb(3, ttnn.float32, 1, crs),
            _cb(4, ttnn.float32, 1, crs),
            _cb(5, ttnn.float32, 1, crs),
            _cb(16, ttnn.float32, 1, crs),
            _cb(17, ttnn.float32, 1, crs),
            _cb(18, ttnn.float32, 1, crs),
            _cb(19, out_dtype, 1, crs, page=out_page),
            _cb(20, out_dtype, 1, crs, page=out_page),
            _stage(24, crs),
            _stage(25, crs, 256),
        ]
        return kernels, cbs

    def _bias_group(self, proj, positions, tau):
        cores = self.b_cores
        has_tau = int(self.dyn)
        named = dict(nj=NJ, has_tau=has_tau, rel_tile=6144 // 32)
        r_rt, w_rt = [], []
        for i, _ in enumerate(cores):
            r_rt.append(
                [
                    proj.buffer_address(),
                    self.rel_scale.buffer_address(),
                    self.g.buffer_address(),
                    tau.buffer_address() if has_tau else 0,
                    positions.near_mask.buffer_address(),
                    positions.ring.buffer_address(),
                    self.table.buffer_address(),
                    i * NJ,
                ]
            )
            blk = self.sink_blk.buffer_address() if self.dyn else 0
            w_rt.append([self.mask.buffer_address(), i * NJ, proj.buffer_address(), self.sink.buffer_address(), blk])
        crs = _crs(cores)
        sink_core, rest = cores[:1], cores[1:]
        w_named = dict(nj=NJ, sink_tile=6528 // 32, sink_blk=int(self.dyn), neg=f32_bits(NEG))
        kernels = [
            _kernel("attn_bias_reader.cpp", cores, named, r_rt, ttnn.ReaderConfigDescriptor()),
            _kernel(
                "attn_bias_writer.cpp", sink_core, dict(w_named, has_sink=1), w_rt[:1], ttnn.WriterConfigDescriptor()
            ),
            _kernel("attn_bias_writer.cpp", rest, dict(w_named, has_sink=0), w_rt[1:], ttnn.WriterConfigDescriptor()),
            _kernel(
                "attn_bias_compute.cpp",
                cores,
                dict(nj=NJ, has_tau=has_tau, inv=f32_bits(self.rel_inv)),
                [[]] * len(cores),
                _compute_config([1, 3, 16]),
            ),
        ]
        cbs = [
            _cb(0, ttnn.bfloat16, 1, crs),
            _cb(1, ttnn.float32, 1, crs),
            _cb(2, ttnn.bfloat16, 1, crs),
            _cb(3, ttnn.float32, 1, crs),
            _cb(4, ttnn.bfloat16, NJ, crs),
            _cb(5, ttnn.float32, NJ, crs),
            _cb(6, ttnn.bfloat16, 1, crs),
            _cb(7, ttnn.float32, 1, crs, page=24 * 128),
            _cb(16, ttnn.float32, 1, crs),
            _cb(17, ttnn.float32, 1, crs),
            _cb(19, ttnn.bfloat16, 1, crs),
            _cb(20, ttnn.bfloat16, 1, crs),
            _cb(21, ttnn.bfloat16, 1, crs),
            _stage(24, crs),
            _stage(25, crs, 512),
        ]
        return kernels, cbs

    def __call__(self, proj, positions, mode_rows, state):
        """proj: [1, 1, 32, N] bf16 TILE DRAM (conv window projection). Writes this step's k / v rows into the
        state's caches (ring slot, and the far cache at pos on global layers). Returns (q, mask, sink): sink is the
        sdpa sink tile [24, 32] bf16 (local) or the near-softmax sink block [1, 8, 3, 32] fp32 (global)."""
        F.check_row_input(proj)
        assert proj.dtype == ttnn.bfloat16
        tau = positions.tau
        kernels, cbs = [], []
        for cores, kind in ((self.q_cores, "q"), (self.k_cores, "k"), (self.v_cores, "v")):
            k, c = self._head_group(cores, proj, kind, mode_rows, tau, state, positions)
            kernels += k
            cbs += c
        k, c = self._bias_group(proj, positions, tau)
        kernels += k
        cbs += c
        io = [proj, self.q_taps, self.kv_taps["k"], self.kv_taps["v"], self.scaler, tau, self.rel_scale, self.g]
        io += [positions.near_mask, positions.ring, positions.pos, self.table, self.mass, self.sink, self.mask]
        io += [state.k_ring, state.v_ring] + ([state.kv_full, self.sink_blk] if self.dyn else [])
        io += [self.q]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return self.q, self.mask, self.sink_blk if self.dyn else self.sink


class AttnOut:
    """gated = bf16(attn * sigmoid(gate)) into row 0 of a persistent o-projection input [1, 1, 32, 3072] (L1, width
    sharded over 8 cores x 3 heads = one kv group each, as `act_mem`). Rows 1..31 stay zero. Global layers pass
    the far and near partials and the merge runs here too."""

    GATE_COL = 3072

    def __init__(self, device, act_mem):
        self.device = device
        self.act = _shared_l1(device, "o_in", N_HEADS * HEAD_DIM, act_mem)
        self.cores = [ttnn.CoreCoord(i, 0) for i in range(N_KV)]  # act_mem grid: CoreGrid(x=8, y=1)

    def __call__(self, attn, proj, merge=None):
        """attn: [1, 1, 24, 128] bf16 TILE DRAM, or with merge = (o_near [1, 8, 3, 128] fp32, r [1, 8, 3, 1] fp32,
        far_valid [1, 1, 1, 1] fp32) the far partial [1, 1, 24, 160] bf16. proj as AttnPrep. Returns self.act."""
        F.check_row_input(attn)
        width = HEAD_DIM if merge is None else HEAD_DIM + 32
        assert attn.dtype == ttnn.bfloat16 and tuple(attn.padded_shape)[-2:] == (32, width), attn.padded_shape
        crs = _crs(self.cores)
        nh = N_HEADS // len(self.cores)
        rt = [[attn.buffer_address(), proj.buffer_address(), i * nh] for i in range(len(self.cores))]
        io = [attn, proj]
        if merge is not None:
            for t in merge:
                F.check_row_input(t)
                assert t.dtype == ttnn.float32, t.dtype
            o_near, r, valid = merge
            rt = [
                a + [o_near.buffer_address(), r.buffer_address(), valid.buffer_address(), i] for i, a in enumerate(rt)
            ]
            io += list(merge)
        flag = int(merge is not None)
        kernels = [
            _kernel(
                "attn_out_reader.cpp",
                self.cores,
                dict(nh=nh, gate_tile=self.GATE_COL // 32, merge=flag),
                rt,
                ttnn.ReaderConfigDescriptor(),
            ),
            _kernel(
                "attn_out_writer.cpp",
                self.cores,
                dict(nh=nh),
                [[self.act.buffer_address()]] * len(self.cores),
                ttnn.WriterConfigDescriptor(),
            ),
            _kernel(
                "attn_out_compute.cpp",
                self.cores,
                dict(merge=flag),
                [[]] * len(self.cores),
                F.compute_config([2, 3, 4], full_sync=True),
            ),
        ]
        cbs = [_cb(0, ttnn.bfloat16, 1, crs), _cb(1, ttnn.bfloat16, 1, crs), _cb(16, ttnn.bfloat16, 1, crs)]
        cbs += [_cb(i, ttnn.float32, 1, crs) for i in (2, 3, 4)] if merge is not None else []
        cbs.append(_stage(24, crs))
        ttnn.generic_op(io + [self.act], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return self.act


class AttnWindow:
    """Conv window update straight into the projection input: one data-movement program on the 8 cores of the
    projection's input shards (`act_mem`, width sharded, 9 tiles each). Writes the shifted window back to the
    state (x_hist) and into a persistent L1 input; replaces the shift matmul, 1-3 row adds and the copy to L1."""

    def __init__(self, device, act_mem, c=2304):
        self.device = device
        self.act = _shared_l1(device, "qkv_in", c, act_mem)
        self.cores = [ttnn.CoreCoord(i, 0) for i in range(N_KV)]  # act_mem grid: CoreGrid(x=8, y=1)
        self.ntiles = c // 32 // len(self.cores)

    def __call__(self, hist, xs):
        """hist: window state [1, 1, 32, C] bf16 TILE DRAM; xs: [x] (single) or [xq, xk, xv] rows [1, 1, 1, C] bf16.
        Returns the projection input (self.act)."""
        for t in [hist, *xs]:
            F.check_row_input(t)
            assert t.dtype == ttnn.bfloat16, t.dtype
        xq, xk, xv = xs if len(xs) == 3 else xs * 3
        n = self.ntiles
        rt = [[hist.buffer_address(), self.act.buffer_address(), i * n] for i in range(len(self.cores))]
        rt = [r + [xq.buffer_address(), xk.buffer_address(), xv.buffer_address()] for r in rt]
        crs = _crs(self.cores)
        named = dict(ntiles=n, ngroups=1 if len(xs) == 1 else 2)
        kernels = [_kernel("attn_window.cpp", self.cores, named, rt, ttnn.ReaderConfigDescriptor())]
        cbs = [_cb(0, ttnn.bfloat16, n, crs), _cb(1, ttnn.bfloat16, n, crs), _stage(24, crs, n * 384 + 128)]
        ttnn.generic_op([hist, *xs, self.act], ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=[]))
        return self.act

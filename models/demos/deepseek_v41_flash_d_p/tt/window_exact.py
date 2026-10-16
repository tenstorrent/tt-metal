# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Exact window attention on the FPU (settings WINDOW_SDPA = "exact"; DS41-19).

The reference attends in fp32: s = q . k (q fp32, k on the fp8 grid), p = exp(s - max), o = (p @ v) / (sum p +
exp(sink - max)), and an fp8 quantizer follows (after the inverse RoPE), so every rounding the device adds becomes
whole e4m3 steps (DS41-1). sparse_sdpa rounds q, the scores, the probabilities and the output to bf16, and the FPU
rounds every in-tile partial dot to ~TF32 (DS41-16): rel 0.021 vs the CPU step, 0.016 on the block out (DS41-18).

Here both products are exact, with the ExactMatmul idea (tt/exact_mm.py) moved to the weight side:
    pieces    an fp32 operand x = x1 + x2 + x3, each a bf16 (x1 = bf16(x), x2 = bf16(x - x1), ...; every
              subtraction exact on the SFPU); ``pieces`` = 3 holds every fp32 bit, 2 holds 16 bits. The pieces sit
              side by side along K and the other operand is repeated along K, so one matmul sums them.
    lanes     the other operand (keys / values: e4m3 x 2^k, bf16-exact) is masked to the K rows k % 32 == u, one
              copy per lane u stacked along N: every in-tile dot holds one product, which the FPU keeps exactly
              (bf16 x e4m3: 12 bits, probed exact), and DEST adds the K tiles in fp32. The 32 lane results are
              added on the SFPU (fp32). 32x the matmul work.
    blocks    queries in blocks of B = 32 positions; block b's keys are the 160 table rows of positions
              [p0 - 128, p0 + 32): every query's 128-key window, plus 32 masked by a load-time band bias (0 / -1e30;
              per start class 0, 32, 64, 96, >= 128 also the positions < 0).
    softmax   fp32 SFPU: max, exp(s - max) (probed rel 7e-8), the sink term exp(sink - max) with the fp32 sink.
              The row sum comes out of the PV matmul (a ones block next to V, exact like the numerator).

Layout: the chip's heads (TP split, as wq_b's columns) and its own rows of the chunk (SP split): no head <-> seq
reshard. q [1, H/TP, S/SP, 512] fp32 -> blocks [1, nb, H/TP * 32, 512] (nb = S/SP / 32), the keys of all S/32
blocks built from the replicated table and split over SP with mesh_partition. Everything that depends only on the
chunk, mesh and pieces is built at load; nothing on the host per chunk.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, SP_AXIS

LANES = 32
BLOCK = 32  # queries per block
NEG = -1e30  # masked score (exp -> 0, probed)
BYTES_CAP = 256 << 20  # per chip: each transient of a lane group (repeated x, masked operand, fp32 output)


def _free(*ts):
    for t in ts:
        ttnn.deallocate(t)


def _upload(mesh, t, dtype, dims=None):
    return ttnn.from_torch(
        t.contiguous(),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=DRAM,
        mesh_mapper=(
            ttnn.ReplicateTensorToMesh(mesh)
            if dims is None
            else ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims)
        ),
    )


def _pieces(x: ttnn.Tensor, n: int) -> ttnn.Tensor:
    """fp32 x -> [x1 | x2 | ... | xn] bf16 along the last dim, x = sum x_i up to the last piece's rounding."""
    out, r = [], x
    for i in range(n):
        p = ttnn.typecast(r, ttnn.bfloat16, memory_config=DRAM)
        out.append(p)
        if i + 1 < n:
            pf = ttnn.typecast(p, ttnn.float32, memory_config=DRAM)
            nr = ttnn.subtract(r, pf, dtype=ttnn.float32, memory_config=DRAM)  # exact (SFPU)
            _free(pf)
            if r is not x:
                _free(r)
            r = nr
    if r is not x:
        _free(r)
    if n == 1:
        return out[0]
    res = ttnn.concat(out, dim=-1, memory_config=DRAM)
    _free(*out)
    return res


def _lane_sum(o: ttnn.Tensor, g: int, n: int, acc):
    """o [.., G * n] (lane results side by side) -> acc + sum over the lanes, fp32 SFPU adds. G == 1 with no acc
    returns o itself (a full-range slice aliases o's buffer: the caller frees o only when it is not the result)."""
    if g == 1:
        if acc is None:
            return o
        res = ttnn.add(o, acc, dtype=ttnn.float32, memory_config=DRAM)
        _free(acc)
        return res
    shape = list(o.shape)
    parts = []
    for u in range(g):
        lo, hi = [0] * len(shape), list(shape)
        lo[-1], hi[-1] = u * n, (u + 1) * n
        parts.append(ttnn.slice(o, lo, hi, memory_config=DRAM))
    if acc is not None:
        parts.append(acc)
    while len(parts) > 1:
        nxt = []
        for i in range(0, len(parts) - 1, 2):
            nxt.append(ttnn.add(parts[i], parts[i + 1], dtype=ttnn.float32, memory_config=DRAM))
            _free(parts[i], parts[i + 1])
        if len(parts) % 2:
            nxt.append(parts[-1])
        parts = nxt
    return parts[0]


def _split(nb: int, m: int, n: int, k: int) -> tuple[int, int]:
    """(L, G): lanes per matmul stacked along K (L, summed in DEST: x repeated L times) and along N (G, summed on the
    SFPU after it), powers of two with L G <= 32 and every transient under BYTES_CAP: x [nb, m, L k] bf16, the
    masked operand [nb, G n, L k] bf16, the output [nb, m, G n] fp32. K first: it saves the most traffic."""
    lk = 1
    while lk < LANES and nb * m * 2 * lk * k * 2 <= BYTES_CAP:
        lk *= 2
    g = 1
    while lk * g < LANES and nb * 2 * g * n * lk * k * 2 <= BYTES_CAP and nb * m * 2 * g * n * 4 <= BYTES_CAP:
        g *= 2
    return lk, g


class ExactWindow:
    """Load-time tables of the exact window attention for one chunk size, mesh and heads per chip (shared by every
    window layer). ``__call__(qs, table, start, sink_col)``: qs [1, H/TP, S/SP, 512] fp32 (roped, scaled), table
    [1, 1, 128 + S, 512] bf16 ROW_MAJOR (carry | chunk, replicated), sink_col [1, H/TP, 32, 1] fp32 -> o [1, H/TP,
    S/SP, 512] fp32 (softmax-weighted values, the reference's attend). Does not consume its inputs."""

    def __init__(self, mesh, chunk: int, heads_per_chip: int, width: int, window: int, pieces: int, compute_config):
        self.ckc = compute_config
        self.mesh, self.chunk, self.hc, self.d, self.win, self.n = mesh, chunk, heads_per_chip, width, window, pieces
        self.sp = mesh.shape[SP_AXIS]
        self.rows = chunk // self.sp  # query rows per chip
        assert self.rows % BLOCK == 0 and window % BLOCK == 0 and width % LANES == 0, (self.rows, window, width)
        self.nb = self.rows // BLOCK
        self.nk = window + BLOCK  # keys per block
        self.m = self.hc * BLOCK  # matmul rows per block
        self.vw = width + LANES  # values + the ones block
        # lane masks over K (the last dim of the [N, K] operand, used with transpose_b): [1, G, 1, L K] per matmul,
        # block l of row g keeping lane u = u0 + g L + l, so one broadcast multiply of the operand repeated L times
        # along K [nb, 1, N, L K] gives the G x L masked copies
        kq, kp = width * pieces, self.nk * pieces
        self.split_s = _split(self.nb, self.m, self.nk, kq)
        self.split_p = _split(self.nb, self.m, self.vw, kp)
        self.q_masks = self._masks(kq, *self.split_s)
        self.p_masks = self._masks(kp, *self.split_p)
        self.ones = _upload(mesh, torch.ones(self.nb, 1, LANES, self.nk), ttnn.bfloat16)  # V^T's ones block
        # band bias per start class: global block gb (chip row r holds blocks [r nb, (r + 1) nb)), query i, key j
        # (position start + gb B - window + j): valid iff i < j <= i + window and the position is >= 0
        i = torch.arange(BLOCK).view(1, BLOCK, 1)
        j = torch.arange(self.nk).view(1, 1, self.nk)
        gb = torch.arange(chunk // BLOCK).view(-1, 1, 1)
        band = (j > i) & (j <= i + window)
        self.bias = {}
        for cls in range(0, window + 1, BLOCK):
            ok = band & (cls + gb * BLOCK - window + j >= 0)
            b = torch.where(ok, 0.0, NEG).unsqueeze(1)  # [S/B, 1, B, nk]
            self.bias[cls] = _upload(mesh, b, ttnn.float32, (0, None))

    def _masks(self, k: int, lk: int, group: int) -> list:
        lane = torch.arange(k) % LANES
        out = []
        for u0 in range(0, LANES, lk * group):
            rows = [torch.cat([(lane == u0 + g * lk + l).float() for l in range(lk)]) for g in range(group)]
            out.append(_upload(self.mesh, torch.stack(rows).view(1, group, 1, lk * k), ttnn.bfloat16))
        return out

    def _keys(self, table: ttnn.Tensor) -> ttnn.Tensor:
        """table [1, 1, win + S, 512] bf16 RM -> this chip's key blocks [1, nb, win + B, 512] bf16 TILE."""
        t = ttnn.to_layout(table, ttnn.TILE_LAYOUT, memory_config=DRAM)
        nt = t.shape[2] // BLOCK
        t4 = ttnn.reshape(t, [1, nt, BLOCK, self.d])
        nbg = self.chunk // BLOCK
        parts = [
            ttnn.slice(t4, [0, u, 0, 0], [1, u + nbg, BLOCK, self.d], memory_config=DRAM)
            for u in range(self.nk // BLOCK)
        ]
        kall = ttnn.concat(parts, dim=2, memory_config=DRAM)  # [1, S/B, nk, 512]
        _free(*parts)
        _free(t)  # t4 is a view of t (never freed on its own)
        if self.sp == 1:
            return kall
        kb = ttnn.mesh_partition(kall, dim=1, cluster_axis=SP_AXIS, memory_config=DRAM)
        _free(kall)
        return kb

    def _lane_matmul(self, x, wt, masks, split):
        """x [1, nb, M, K] bf16 @ wt [nb, 1, N, K]^T summed over the 32 lanes (wt masked to the K columns k % 32 ==
        u) -> [1, nb, M, N] fp32. Per matmul L x G lanes: x and wt repeated L times along K (DEST adds those lanes,
        fp32 across tiles), G masked copies stacked along N (added on the SFPU after)."""
        lk, group = split
        nb, _, n, k = wt.shape
        xl = ttnn.concat([x] * lk, dim=-1, memory_config=DRAM) if lk > 1 else x
        wl = ttnn.concat([wt] * lk, dim=-1, memory_config=DRAM) if lk > 1 else wt
        acc = None
        for mask in masks:
            ws = ttnn.multiply(wl, mask, dtype=ttnn.bfloat16, memory_config=DRAM)  # [nb, G, N, L K], exact (x 0 / 1)
            w2 = ttnn.reshape(ws, [1, nb, group * n, lk * k])
            o = ttnn.matmul(
                xl, w2, transpose_b=True, dtype=ttnn.float32, compute_kernel_config=self.ckc, memory_config=DRAM
            )
            _free(ws)  # (w2 is a view of ws)
            acc = _lane_sum(o, group, n, acc)
            if acc is not o:
                _free(o)
        if lk > 1:
            _free(xl, wl)
        return acc

    def __call__(self, qs, table, start: int, sink_col) -> ttnn.Tensor:
        hc, nb, B, d, n = self.hc, self.nb, BLOCK, self.d, self.n
        kb = self._keys(table)
        # q blocks: [1, Hc, R, d] -> [Hc, nb, B, d] -> [nb, Hc, B, d] -> [1, nb, Hc B, d] -> pieces
        q4 = ttnn.permute(ttnn.reshape(qs, [hc, nb, B, d]), (1, 0, 2, 3), memory_config=DRAM)
        qb = ttnn.reshape(q4, [1, nb, self.m, d])
        xq = _pieces(qb, n)
        _free(q4)  # qb is a view of q4
        kb4 = ttnn.reshape(kb, [nb, 1, self.nk, d])
        k3 = ttnn.concat([kb4] * n, dim=-1, memory_config=DRAM) if n > 1 else kb4  # [nb, 1, nk, n d]
        s = self._lane_matmul(xq, k3, self.q_masks, self.split_s)  # [1, nb, m, nk]
        _free(xq)
        if n > 1:
            _free(k3)
        # softmax with the sink, fp32 SFPU
        s4 = ttnn.reshape(s, [nb, hc, B, self.nk])
        sb = ttnn.add(s4, self.bias[min(start, self.win)], dtype=ttnn.float32, memory_config=DRAM)
        _free(s4)  # (a view of s)
        mx = ttnn.max(sb, dim=-1, keepdim=True, memory_config=DRAM)  # [nb, Hc, B, 1]
        z = ttnn.subtract(sb, mx, dtype=ttnn.float32, memory_config=DRAM)
        _free(sb)
        e = ttnn.exp(z, memory_config=DRAM)
        _free(z)
        zs = ttnn.subtract(sink_col, mx, dtype=ttnn.float32, memory_config=DRAM)
        _free(mx)
        es = ttnn.exp(zs, memory_config=DRAM)
        _free(zs)
        xp = ttnn.reshape(_pieces(e, n), [1, nb, self.m, self.nk * n])
        _free(e)
        # V^T and its ones block [nb, 1, d + 32, nk], repeated along K (the pieces' key dim)
        vt = ttnn.transpose(kb4, -2, -1, memory_config=DRAM)  # exact (bf16)
        _free(kb)  # (kb4 is a view of kb)
        v1 = ttnn.concat([vt, self.ones], dim=2, memory_config=DRAM)
        _free(vt)
        v3 = ttnn.concat([v1] * n, dim=-1, memory_config=DRAM) if n > 1 else v1
        if n > 1:
            _free(v1)
        pv = self._lane_matmul(xp, v3, self.p_masks, self.split_p)  # [1, nb, m, d + 32]
        _free(xp, v3)
        pv4 = ttnn.reshape(pv, [nb, hc, B, self.vw])
        num = ttnn.slice(pv4, [0, 0, 0, 0], [nb, hc, B, d], memory_config=DRAM)
        den32 = ttnn.slice(pv4, [0, 0, 0, d], [nb, hc, B, self.vw], memory_config=DRAM)  # every column = sum p
        _free(pv4)
        den = ttnn.max(den32, dim=-1, keepdim=True, memory_config=DRAM)
        _free(den32)
        dt = ttnn.add(den, es, dtype=ttnn.float32, memory_config=DRAM)
        _free(den, es)
        o = ttnn.divide(num, dt, dtype=ttnn.float32, memory_config=DRAM)
        _free(num, dt)
        oh = ttnn.permute(o, (1, 0, 2, 3), memory_config=DRAM)  # [Hc, nb, B, d]
        _free(o)
        return ttnn.reshape(oh, [1, hc, self.rows, d])

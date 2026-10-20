# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 RoPE on the device, exact in fp32 (reference/ds41_ref.py:rope; inference apply_rotary_emb).

The reference rotates the last R = 64 dims of a vector as 32 adjacent pairs (complex numbers, the interleaved / Meta
order) by the fp32 angle pos * freq[k] (``rope_table``). Here the pairs arrive DE-interleaved: the producing matmul's
weight rows are permuted at load (``deinterleave_perm``) so the tail is [x_even (32) | x_odd (32)], two tile-aligned
column blocks. Then, all on the SFPU in fp32 (fp32 x fp32 eltwise never passes the FPU, K-30):

    re = xe c - xo s       ro = xe s + xo c       (the reference's complex product, term for term)
    tail [re | ro] -> ROW_MAJOR [R, 2, 32] -> transpose -> [R, 32, 2] -> [R, 64]: back to the interleaved order
                                                  (data movement only; bit-exact on device, probe in BREADCRUMBS)

ATTN_GLUE=on (P.15): the interleave / de-interleave is a ``PairPerm`` 0/1 matmul on the bf16 tail instead (after
the cast that follows every site; exact at HiFi4), and the rope's last add / subtract write bf16 directly (the same
RNE as ttnn.typecast): ``rope_deinterleaved_bf16`` (kv, compressor), ``rope_heads(..., dtype=bf16)`` (q, indexer q,
the attention output).

cos / sin: ``RopeTable`` holds the reference's own fp32 table (torch.polar of the fp32 angles) for positions
[0, max_seq), [1, 1, max_seq, 32] fp32 TILE, replicated, built once at load. Per chunk it is sliced on the device at
[start, start + S) and split over the SP axis with ``ttnn.mesh_partition``, so mesh row r gets the positions of its
natural block [start + r S/SP, start + (r + 1) S/SP) (the row-split layout of tt/layout.py). No host work per chunk.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, SP_AXIS, TP_AXIS
from .settings import settings


def glue() -> bool:
    """ATTN_GLUE=on (P.15): the attention / RoPE glue folded into ops (``PairPerm``, nlp_create_q_heads_split,
    the bf16 output path; tt/settings.py)."""
    return settings.get("ATTN_GLUE") == "on"


def indexer_glue() -> bool:
    """INDEXER_GLUE=on (P.16): the indexer / shared-source glue folded into ops or computed once per chunk (the
    rope rows memo here, tt/sparse_attention.ChunkMemo, the score op's block max / bias; tt/settings.py)."""
    return settings.get("INDEXER_GLUE") == "on"


class RowsMemo:
    """INDEXER_GLUE=on (P.16): device tensors that depend only on (a load-time table, a chunk start), computed once
    and handed to every caller of the chunk (the rope rows of all 8 layers' kv / attention / indexer steps). Holds
    the entries of the latest start only (another start frees them: the values are constants of the table, so a
    re-run or another slot recomputes them). Callers release what they got through ``free`` (never
    ``ttnn.deallocate``): memoized tensors stay."""

    def __init__(self, on: bool):
        self.on = on
        self._d: dict = {}

    def get(self, key, make):
        if not self.on:
            return make()
        hit = self._d.get(key)
        if hit is not None and all(t.is_allocated() for t in hit):
            return hit
        for k in [k for k in self._d if k[0] != key[0]]:  # another start: the old rows go
            for t in self._d.pop(k):
                if t.is_allocated():
                    ttnn.deallocate(t)
        hit = self._d[key] = make()
        return hit

    def mine(self, t: ttnn.Tensor) -> bool:
        return any(t is u for v in self._d.values() for u in v)

    def free(self, *ts) -> None:
        for t in ts:
            if not self.mine(t):
                ttnn.deallocate(t)


class PairPerm:
    """ATTN_GLUE=on (P.15): the rope tail's pair (de-)interleave as a 0/1 [2 half, 2 half] matmul on a bf16 tensor
    (one nonzero product per output: exact at HiFi3 / HiFi4 for any bf16 operand; HiFi2 / LoFi drop in0's last
    mantissa bit, probe ds41_attn_glue/probe_003), replacing the ROW_MAJOR reshape to 8-byte pages of
    ``interleave_tail`` / ``deinterleave_tail``. ``to_inter``: [even | odd] -> interleaved (e_0, o_0, e_1, ...);
    ``to_de``: the inverse. Load-time [1, 1, 2 half, 2 half] bf16 TILE, replicated."""

    def __init__(self, mesh, half: int):
        w = 2 * half
        src = torch.arange(w).view(2, half).t().reshape(-1)  # interleaved column i reads [even | odd] column src[i]
        p_inter = torch.zeros(w, w)
        p_inter[src, torch.arange(w)] = 1.0  # out[:, i] = in[:, src[i]]
        self.half = half

        def dev(t):
            return ttnn.from_torch(
                t.reshape(1, 1, w, w).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.to_inter, self.to_de = dev(p_inter), dev(p_inter.t())
        # HiFi4 + fp32 DEST: the only exact setting for a bf16 data move through the FPU (not a precision choice)
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def __call__(self, t: ttnn.Tensor, p: ttnn.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
        """t [..., R, 2 half] bf16 TILE -> its columns permuted by ``p`` (``to_inter`` / ``to_de``), in ``dtype``
        (bf16 or fp32: exact either way). Does not consume ``t``."""
        assert t.dtype == ttnn.bfloat16 and t.shape[-1] == 2 * self.half, (t.dtype, t.shape)
        return ttnn.matmul(t, p, dtype=dtype, compute_kernel_config=self.ckc, memory_config=DRAM)


def deinterleave_perm(width: int, rope_dim: int) -> torch.Tensor:
    """Column order [0, width - R) + the tail's even then odd columns (index into the reference's interleaved order)."""
    head = torch.arange(width - rope_dim)
    tail = width - rope_dim + torch.arange(rope_dim).view(-1, 2).t().reshape(-1)
    return torch.cat([head, tail])


class RopeTable:
    """cos / sin of one rope type for positions [0, max_seq): [1, 1, max_seq, R/2] fp32 TILE, replicated."""

    def __init__(self, mesh, freqs: torch.Tensor, max_seq: int):
        from models.demos.deepseek_v41_flash_d_p.reference.ds41_ref import rope_table

        max_seq = -(-max_seq // 32) * 32
        rot = rope_table(freqs.float(), torch.arange(max_seq))  # complex64 [max_seq, R/2], the reference's
        self.mesh, self.max_seq, self.half = mesh, max_seq, freqs.numel()
        assert self.half % 32 == 0, f"rope half width {self.half} is not a tile multiple"

        def dev(t):
            return ttnn.from_torch(
                t.float().reshape(1, 1, max_seq, self.half).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        self.cos, self.sin = dev(rot.real), dev(rot.imag)
        self.pairs = PairPerm(mesh, self.half) if glue() else None  # ATTN_GLUE=on (P.15)
        self.memo = RowsMemo(indexer_glue())  # INDEXER_GLUE=on (P.16): the rows once per (start, length)

    def free(self, *ts) -> None:
        """Release rows from ``rows``: freed unless memoized (INDEXER_GLUE=on)."""
        self.memo.free(*ts)

    def rows(self, start: int, length: int, tp_split: bool = False) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """cos, sin of the chip row's natural block of [start, start + length): [1, 1, length / SP, R/2] fp32.
        ``tp_split``: this chip's Sq = length / (SP TP) rows of that block (mesh column c: rows [c Sq, (c+1) Sq);
        the indexer's queries). Release them with ``free``."""
        if tp_split:
            return self.memo.get((start, length, True), lambda: self._tp_split(start, length))
        return self.memo.get((start, length, False), lambda: self._rows(start, length))

    def _tp_split(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        rows = self.rows(start, length)
        tp = self.mesh.shape[TP_AXIS]
        out = tuple(
            (
                ttnn.clone(t, memory_config=DRAM)
                if tp == 1
                else ttnn.mesh_partition(t, dim=2, cluster_axis=TP_AXIS, memory_config=DRAM)
            )
            for t in rows
        )
        self.free(*rows)
        return out

    def _rows(self, start: int, length: int) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        sp = self.mesh.shape[SP_AXIS]
        assert start % 32 == 0 and length % (32 * sp) == 0, (start, length, sp)
        assert start + length <= self.max_seq, f"positions up to {start + length} > the table's {self.max_seq}"
        out = []
        for t in (self.cos, self.sin):
            sl = ttnn.slice(t, [0, 0, start, 0], [1, 1, start + length, self.half], memory_config=DRAM)
            if sp > 1:
                part = ttnn.mesh_partition(sl, dim=2, cluster_axis=SP_AXIS, memory_config=DRAM)
                ttnn.deallocate(sl)
                sl = part
            out.append(sl)
        return out[0], out[1]


def rope_deinterleaved(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor) -> ttnn.Tensor:
    """x [1, 1, R, W] fp32 TILE whose last 2 * half columns are [even | odd] (``deinterleave_perm``) -> the rotated
    vector in the reference's interleaved order, [1, 1, R, W] fp32 TILE. Does not consume its inputs."""
    rows, width, half = x.shape[-2], x.shape[-1], cos.shape[-1]
    r0 = width - 2 * half
    xe = ttnn.slice(x, [0, 0, 0, r0], [1, 1, rows, r0 + half], memory_config=DRAM)
    xo = ttnn.slice(x, [0, 0, 0, r0 + half], [1, 1, rows, width], memory_config=DRAM)

    def mul(a, b):
        return ttnn.multiply(a, b, dtype=ttnn.float32, memory_config=DRAM)

    a, b = mul(xe, cos), mul(xo, sin)
    re = ttnn.subtract(a, b, dtype=ttnn.float32, memory_config=DRAM)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
    a, b = mul(xe, sin), mul(xo, cos)
    ro = ttnn.add(a, b, dtype=ttnn.float32, memory_config=DRAM)
    for t in (a, b, xe, xo):
        ttnn.deallocate(t)
    tail = ttnn.concat([re, ro], dim=-1, memory_config=DRAM)
    ttnn.deallocate(re)
    ttnn.deallocate(ro)
    # [re | ro] -> interleaved (re_k, ro_k): a transpose of [2, half] per row, row-major (data movement only)
    rm = ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
    ttnn.deallocate(tail)
    t = ttnn.transpose(ttnn.reshape(rm, [1, rows, 2, half]), -2, -1, memory_config=DRAM)
    ttnn.deallocate(rm)
    inter = ttnn.to_layout(ttnn.reshape(t, [1, 1, rows, 2 * half]), ttnn.TILE_LAYOUT, memory_config=DRAM)
    ttnn.deallocate(t)
    if r0 == 0:
        return inter
    head = ttnn.slice(x, [0, 0, 0, 0], [1, 1, rows, r0], memory_config=DRAM)
    out = ttnn.concat([head, inter], dim=-1, memory_config=DRAM)
    ttnn.deallocate(head)
    ttnn.deallocate(inter)
    return out


def rope_deinterleaved_bf16(x: ttnn.Tensor, cos: ttnn.Tensor, sin: ttnn.Tensor, pairs: PairPerm) -> ttnn.Tensor:
    """ATTN_GLUE=on, NUMERICS=native (P.15): ``rope_deinterleaved`` then a cast to bf16, bit for bit, with the
    re-interleave as a ``PairPerm`` matmul on the bf16 tail (the rope's add / subtract write bf16: the same RNE as
    ttnn.typecast, probe ds41_attn_glue/probe_001). x [1, 1, R, W] fp32 TILE -> [1, 1, R, W] bf16 TILE, interleaved
    tail. Does not consume its inputs."""
    rows, width, half = x.shape[-2], x.shape[-1], cos.shape[-1]
    r0 = width - 2 * half
    tail = ttnn.slice(x, [0, 0, 0, r0], [1, 1, rows, width], memory_config=DRAM)
    rot = rope_heads(tail, cos, sin, dtype=ttnn.bfloat16)  # [re | ro] bf16
    ttnn.deallocate(tail)
    inter = pairs(rot, pairs.to_inter)
    ttnn.deallocate(rot)
    if r0 == 0:
        return inter
    head = ttnn.slice(x, [0, 0, 0, 0], [1, 1, rows, r0], memory_config=DRAM)
    hb = ttnn.typecast(head, ttnn.bfloat16, memory_config=DRAM)
    ttnn.deallocate(head)
    out = ttnn.concat([hb, inter], dim=-1, memory_config=DRAM)
    ttnn.deallocate(hb)
    ttnn.deallocate(inter)
    return out


def interleave_tail(t: ttnn.Tensor, half: int) -> ttnn.Tensor:
    """[B, H, R, 2 * half] TILE whose columns are [even | odd] -> the interleaved order (e_0, o_0, e_1, ...): a
    ROW_MAJOR transpose of [2, half] per row (data movement only, exact). Does not consume ``t``."""
    b, h, rows = t.shape[0], t.shape[1], t.shape[2]
    rm = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
    tr = ttnn.transpose(ttnn.reshape(rm, [b * h, rows, 2, half]), -2, -1, memory_config=DRAM)
    if rm is not t:
        ttnn.deallocate(rm)
    out = ttnn.to_layout(ttnn.reshape(tr, [b, h, rows, 2 * half]), ttnn.TILE_LAYOUT, memory_config=DRAM)
    ttnn.deallocate(tr)
    return out


def deinterleave_tail(t: ttnn.Tensor, half: int) -> ttnn.Tensor:
    """[B, H, R, W] TILE whose last 2 * half columns are interleaved (e_0, o_0, e_1, ...) -> the same with the tail
    [even | odd] (``rope_heads``' input order): a ROW_MAJOR transpose of [half, 2] per row (data movement only,
    exact). Does not consume ``t``."""
    b, h, rows, width = t.shape
    r0 = width - 2 * half
    tail = ttnn.slice(t, [0, 0, 0, r0], [b, h, rows, width], memory_config=DRAM)
    rm = ttnn.to_layout(tail, ttnn.ROW_MAJOR_LAYOUT, memory_config=DRAM)
    ttnn.deallocate(tail)
    tr = ttnn.transpose(ttnn.reshape(rm, [b * h, rows, half, 2]), -2, -1, memory_config=DRAM)
    ttnn.deallocate(rm)
    de = ttnn.to_layout(ttnn.reshape(tr, [b, h, rows, 2 * half]), ttnn.TILE_LAYOUT, memory_config=DRAM)
    ttnn.deallocate(tr)
    if r0 == 0:
        return de
    head = ttnn.slice(t, [0, 0, 0, 0], [b, h, rows, r0], memory_config=DRAM)
    out = ttnn.concat([head, de], dim=-1, memory_config=DRAM)
    ttnn.deallocate(head)
    ttnn.deallocate(de)
    return out


def rope_heads(
    x: ttnn.Tensor,
    cos: ttnn.Tensor,
    sin: ttnn.Tensor,
    inverse: bool = False,
    interleave: bool = False,
    dtype=ttnn.float32,
) -> ttnn.Tensor:
    """RoPE of the last 2 * half columns of x [B, H, R, W] fp32 TILE (every head), held DE-interleaved ([even | odd],
    ``deinterleave_perm``), by cos / sin [1, 1, R, half] fp32 (broadcast over the heads). The reference's complex
    product term for term on the SFPU (fp32): forward re = e c - o s, ro = e s + o c; ``inverse`` (rope(...,
    inverse=True), the conjugate) re = e c + o s, ro = o c - e s. ``interleave``: return the tail in the reference's
    interleaved order (else [re | ro]). ``dtype``: what the final add / subtract write (bf16 = the fp32 result cast
    to bf16, bit for bit; only with ``interleave`` False and no head columns, W == 2 half). Returns [B, H, R, W]
    TILE; does not consume its inputs."""
    b, h, rows, width = x.shape
    half = cos.shape[-1]
    r0 = width - 2 * half
    assert dtype == ttnn.float32 or (r0 == 0 and not interleave), (dtype, r0, interleave)
    xe = ttnn.slice(x, [0, 0, 0, r0], [b, h, rows, r0 + half], memory_config=DRAM)
    xo = ttnn.slice(x, [0, 0, 0, r0 + half], [b, h, rows, width], memory_config=DRAM)

    def mul(a, c):
        return ttnn.multiply(a, c, dtype=ttnn.float32, memory_config=DRAM)

    ec, os_, es, oc = mul(xe, cos), mul(xo, sin), mul(xe, sin), mul(xo, cos)
    if inverse:
        re = ttnn.add(ec, os_, dtype=dtype, memory_config=DRAM)
        ro = ttnn.subtract(oc, es, dtype=dtype, memory_config=DRAM)
    else:
        re = ttnn.subtract(ec, os_, dtype=dtype, memory_config=DRAM)
        ro = ttnn.add(es, oc, dtype=dtype, memory_config=DRAM)
    for t in (ec, os_, es, oc, xe, xo):
        ttnn.deallocate(t)
    tail = ttnn.concat([re, ro], dim=-1, memory_config=DRAM)
    ttnn.deallocate(re)
    ttnn.deallocate(ro)
    if interleave:
        inter = interleave_tail(tail, half)
        ttnn.deallocate(tail)
        tail = inter
    if r0 == 0:
        return tail
    head = ttnn.slice(x, [0, 0, 0, 0], [b, h, rows, r0], memory_config=DRAM)
    out = ttnn.concat([head, tail], dim=-1, memory_config=DRAM)
    ttnn.deallocate(head)
    ttnn.deallocate(tail)
    return out

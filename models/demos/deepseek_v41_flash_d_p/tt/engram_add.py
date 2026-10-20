# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 engram residual add (reference/ds41_ref.py:_engram_add, inference engram.py) on the SP x TP mesh,
all fp32:

    key_n = kv[:, n H : (n+1) H] (n = 0..3), value = kv[:, 4H : 5H], w_n = q_weight[n] * k_weight[n]
    dot_n = <h_n, w_n key_n> rsqrt(mean(h_n^2) + eps) rsqrt(mean(key_n^2) + eps) / sqrt(H)          eps = 1e-20
    gate_n = sigmoid(copysign(sqrt(max(|dot_n|, 1e-6)), dot_n))
    out_n = h_n + gate_n value                         the pending pre (DS41-3) passes through unchanged

Per chip (r, c): the streams h [1, 1, S/SP, 4 x H/TP] fp32 (tt/layout.py, stream-major), engram_kv
[1, 1, S/SP, 5 x H/TP] fp32 (tt/engram_kv.py: chip column c's hidden slice of the 4 keys and the value, so no reshard)
and the pending pre [1, 1, S/SP, 4] (row-split, replicated over TP). Composition:

    ttnn.multiply (fp32, SFPU)    h_n * w_n * key_n, h_n^2, key_n^2 over the chip's H/TP columns (w_n: the chip's
                                  slice of q_weight * k_weight, fp32, built at load; bf16 x bf16 is fp32-exact)
    ttnn.sum(dim=-1)              the 12 partial row sums; on fp32 with fp32_dest_acc_en this is the accurate SFPU
                                  reduce (generic_reductions_nanobind.hpp: fast_and_approximate_mode=False), not the
                                  FPU's TF32 read (K-30)
    tt/common.py:tp_sum           [S/SP, 12] all_gather over axis 1 + SFPU ttnn.add in mesh-column order (exact fp32,
                                  K-30; bit-identical on every chip of the row)
    rsqrt / multiply / abs / maximum / sqrt / where(ltz) / sigmoid    the gate [S/SP, 4] (fp32 SFPU)
    ttnn.addcmul                  h_n + value * gate_n per stream (fp32), then concat -> the new streams

Output: (streams [1, 1, S/SP, 4 x H/TP] fp32, the pending pre as given). No host work in __call__; the only collective
is the [S/SP, 12] fp32 exchange (plan.md swa_engram).

MHC_FUSED=on (P.14): the 12 partial sums come from ONE ttnn.bringup.mhc_pre_xing_pack engram pass over h and the keys
(exact fp32 SFPU; the weight row stored on 32 rows at load) as a tile-aligned [S/SP, 32] row, exchanged exactly
(tp_sum exact=True), and the add is ONE ttnn.bringup.mhc_post(value, h, post=gate, comb=I) (X'_n = gate_n value + h_n:
the identity comb's zero terms add exact zeros, so each output is round(h_n + round(gate_n value)), the addcmul's
arithmetic), replacing the 4 h slices, 4 gate slices, 4 addcmul and the concat; the identity [S/SP, 16] is built at
load (``rows``) or once per row count.
"""

from __future__ import annotations

import torch

import ttnn

from .common import DRAM, TP_AXIS, compute_config, tp_sum
from .layout import HC, row_split_to_device, streams_cols_to_chip_major
from .settings import settings

GATE_FLOOR = 1e-6  # reference: dot.abs().clamp_min(1e-6)


class TtEngramAdd:
    """engram_add of one layer. Call with (streams [1, 1, R, 4 x H/TP] fp32, engram_kv [1, 1, R, 5 x H/TP] fp32,
    pre [1, 1, R, 4] or None); returns (new streams [1, 1, R, 4 x H/TP] fp32, pre). Consumes no input."""

    def __init__(
        self, mesh, q_weight: torch.Tensor, k_weight: torch.Tensor, hidden: int, eps: float, rows: int | None = None
    ):
        assert q_weight.shape == (HC, hidden) and k_weight.shape == (HC, hidden), (q_weight.shape, k_weight.shape)
        tp = mesh.shape[TP_AXIS]
        assert hidden % (32 * tp) == 0, f"hidden {hidden} does not split into tiles over TP {tp}"
        self.mesh, self.hidden, self.w = mesh, hidden, hidden // tp
        self.eps = float(eps)
        self.inv_h = 1.0 / float(hidden)
        self.inv_sqrt_h = float(hidden) ** -0.5
        self.ckc = compute_config(mesh)
        wt = (q_weight.float() * k_weight.float()).reshape(1, HC * hidden)  # the reference's fp32 product
        wt = streams_cols_to_chip_major(wt, hidden, tp)  # chip column c's [4 x H/TP] block (stream-major)
        dims = (None, 3) if TP_AXIS == 1 else (3, None)
        self.weight = ttnn.from_torch(
            wt.reshape(1, 1, 1, HC * hidden).contiguous(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=mesh,
            memory_config=DRAM,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
        )
        self.fused = settings.get("MHC_FUSED") == "on"
        self.weight32 = None
        if self.fused:  # the engram mode of mhc_pre_xing_pack reads the weight row on all 32 rows of a tile
            self.weight32 = ttnn.from_torch(
                wt.reshape(1, 1, 1, HC * hidden).expand(1, 1, 32, HC * hidden).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=DRAM,
                mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=tuple(mesh.shape), dims=dims),
            )
        self._eye = {}  # rows per chip -> identity comb [1, 1, rows, 16] fp32 (row-split, TP-replicated)
        if self.fused and rows:
            self.eye(rows)

    def eye(self, rows: int) -> ttnn.Tensor:
        """The identity comb for mhc_post, [1, 1, rows, n n] fp32 per chip; built once per row count (load time
        when the model passes ``rows``)."""
        t = self._eye.get(rows)
        if t is None:
            sp = self.mesh.shape[0]
            e = torch.eye(HC, dtype=torch.float32).reshape(1, HC * HC).expand(rows * sp, HC * HC)
            t = self._eye[rows] = row_split_to_device(self.mesh, e.contiguous())
        return t

    def _rowsum(self, t: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.sum(t, dim=-1, keepdim=True, compute_kernel_config=self.ckc, memory_config=DRAM)

    def _partials_fused(self, h: ttnn.Tensor, kv: ttnn.Tensor) -> ttnn.Tensor:
        """MHC_FUSED=on: the 12 exact partial sums [dot 0..3 | sum h^2 | sum k^2] of this chip's columns in ONE
        ttnn.bringup.mhc_pre_xing_pack engram pass over h and the keys -> [1, 1, R, 32], summed exactly over TP."""
        part = ttnn.bringup.mhc_pre_xing_pack(None, h, n=HC, engram_keys=kv, engram_weight=self.weight32)
        full = tp_sum(self.mesh, part, exact=True)
        ttnn.deallocate(part)
        return full

    def gates(self, h: ttnn.Tensor, kv: ttnn.Tensor) -> ttnn.Tensor:
        """[1, 1, R, 4] fp32 gate per stream, replicated over TP."""
        rows, w = h.shape[-2], self.w
        if self.fused:
            return self._gate_math(self._partials_fused(h, kv), rows)
        hw = ttnn.multiply(h, self.weight, dtype=ttnn.float32, memory_config=DRAM)  # h * (q_w k_w), all 4 streams
        cols = []  # [dot_0..3 | ss_h 0..3 | ss_k 0..3] partials, each [R, 1]
        sq_h, sq_k = [], []
        for n in range(HC):
            hn = ttnn.slice(h, [0, 0, 0, n * w], [1, 1, rows, (n + 1) * w], memory_config=DRAM)
            kn = ttnn.slice(kv, [0, 0, 0, n * w], [1, 1, rows, (n + 1) * w], memory_config=DRAM)
            hwn = ttnn.slice(hw, [0, 0, 0, n * w], [1, 1, rows, (n + 1) * w], memory_config=DRAM)
            p = ttnn.multiply(hwn, kn, dtype=ttnn.float32, memory_config=DRAM)
            ttnn.deallocate(hwn)
            cols.append(self._rowsum(p))
            ttnn.deallocate(p)
            p = ttnn.multiply(hn, hn, dtype=ttnn.float32, memory_config=DRAM)
            sq_h.append(self._rowsum(p))
            ttnn.deallocate(p)
            p = ttnn.multiply(kn, kn, dtype=ttnn.float32, memory_config=DRAM)
            sq_k.append(self._rowsum(p))
            ttnn.deallocate(p)
            ttnn.deallocate(hn)
            ttnn.deallocate(kn)
        ttnn.deallocate(hw)
        parts = cols + sq_h + sq_k
        part = ttnn.concat(parts, dim=-1, memory_config=DRAM)  # [1, 1, R, 12]
        for t in parts:
            ttnn.deallocate(t)
        full = tp_sum(self.mesh, part)
        ttnn.deallocate(part)
        return self._gate_math(full, rows)

    def _gate_math(self, full: ttnn.Tensor, rows: int) -> ttnn.Tensor:
        """The gates from the exchanged partials [1, 1, R, >= 12] (consumed)."""
        dot = ttnn.slice(full, [0, 0, 0, 0], [1, 1, rows, HC], memory_config=DRAM)
        ssh = ttnn.slice(full, [0, 0, 0, HC], [1, 1, rows, 2 * HC], memory_config=DRAM)
        ssk = ttnn.slice(full, [0, 0, 0, 2 * HC], [1, 1, rows, 3 * HC], memory_config=DRAM)
        ttnn.deallocate(full)
        rs = []
        for ss in (ssh, ssk):
            ms = ttnn.multiply(ss, self.inv_h, dtype=ttnn.float32, memory_config=DRAM)
            ttnn.deallocate(ss)
            t = ttnn.add(ms, self.eps, dtype=ttnn.float32, memory_config=DRAM)
            ttnn.deallocate(ms)
            rs.append(ttnn.rsqrt(t, memory_config=DRAM))
            ttnn.deallocate(t)
        r2 = ttnn.multiply(rs[0], rs[1], dtype=ttnn.float32, memory_config=DRAM)  # rsqrt(h) * rsqrt(key)
        for t in rs:
            ttnn.deallocate(t)
        d0 = ttnn.multiply(dot, r2, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(dot)
        ttnn.deallocate(r2)
        d = ttnn.multiply(d0, self.inv_sqrt_h, dtype=ttnn.float32, memory_config=DRAM)
        ttnn.deallocate(d0)
        a = ttnn.abs(d, memory_config=DRAM)
        am = ttnn.maximum(a, GATE_FLOOR, memory_config=DRAM)
        ttnn.deallocate(a)
        s = ttnn.sqrt(am, memory_config=DRAM)
        ttnn.deallocate(am)
        neg = ttnn.neg(s, memory_config=DRAM)
        isneg = ttnn.ltz(d, memory_config=DRAM)
        ttnn.deallocate(d)
        sg = ttnn.where(isneg, neg, s, memory_config=DRAM)  # copysign(s, d)
        for t in (neg, isneg, s):
            ttnn.deallocate(t)
        gate = ttnn.sigmoid(sg, memory_config=DRAM)
        ttnn.deallocate(sg)
        return gate

    def __call__(self, h: ttnn.Tensor, kv: ttnn.Tensor, pre: ttnn.Tensor | None = None):
        out, pre, _ = self.add(h, kv, pre)
        return out, pre

    def add(self, h: ttnn.Tensor, kv: ttnn.Tensor, pre: ttnn.Tensor | None = None, hc=None):
        """(new streams, pre, row): with MHC_FUSED=on and ``hc`` (this layer's attn_hc, tt/mhc.py:TtHcMixes) the
        mhc_post pass also emits that step's partial row of the new streams (stats_weight), else row is None."""
        assert h.dtype == ttnn.float32, h.dtype
        rows, w = h.shape[-2], self.w
        assert h.shape[-1] == HC * w and kv.shape[-1] == (HC + 1) * w and kv.shape[-2] == rows, (h.shape, kv.shape)
        kv32 = kv if kv.dtype == ttnn.float32 else ttnn.typecast(kv, ttnn.float32, memory_config=DRAM)
        gate = self.gates(h, kv32)
        value = ttnn.slice(kv32, [0, 0, 0, HC * w], [1, 1, rows, (HC + 1) * w], memory_config=DRAM)
        if kv32 is not kv:
            ttnn.deallocate(kv32)
        if self.fused:
            row = None
            if hc is not None:  # X'_n = gate_n value + h_n, and attn_hc's partial row of X'
                out, row = ttnn.bringup.mhc_post(
                    value, h, gate, self.eye(rows), stats_weight=hc.fn_t, compute_kernel_config=hc.post_cfg
                )
            else:
                out = ttnn.bringup.mhc_post(value, h, gate, self.eye(rows))  # X'_n = gate_n value + h_n
            ttnn.deallocate(value)
            ttnn.deallocate(gate)
            return out, pre, row
        outs = []
        for n in range(HC):
            hn = ttnn.slice(h, [0, 0, 0, n * w], [1, 1, rows, (n + 1) * w], memory_config=DRAM)
            g = ttnn.slice(gate, [0, 0, 0, n], [1, 1, rows, n + 1], memory_config=DRAM)
            outs.append(ttnn.addcmul(hn, value, g, memory_config=DRAM))
            ttnn.deallocate(hn)
            ttnn.deallocate(g)
        ttnn.deallocate(value)
        ttnn.deallocate(gate)
        out = ttnn.concat(outs, dim=-1, memory_config=DRAM)
        for t in outs:
            ttnn.deallocate(t)
        return out, pre, None


def build_engram_add(mesh, ck, cfg, layer: int, rows: int | None = None) -> TtEngramAdd:
    """Layer ``layer``'s engram residual add (q_weight * k_weight, fp32, built at load; ``rows`` per chip: the
    MHC_FUSED identity comb at load too)."""
    assert cfg.is_engram(layer), f"layer {layer} has no engram"
    qw = ck.weight(f"layers.{layer}.engram.q_weight", torch.float32)
    kw = ck.weight(f"layers.{layer}.engram.k_weight", torch.float32)
    return TtEngramAdd(mesh, qw, kw, cfg.dim, cfg.eps, rows=rows)

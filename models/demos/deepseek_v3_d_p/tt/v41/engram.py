# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash Engram for the prefill (before layers 1 and 14), ``model.py`` ``Engram.forward``:

    kv          = wkv(rows)                     # rows: the 24 table rows x 256 of each position's n-grams, [S, 6144]
    key, value  = kv[:, :4 * D] (one key per hc copy), kv[:, 4 * D:]
    dot_c       = sum_d(h_c * q_w_c * k_w_c * key_c) * rstd(h_c) * rstd(key_c) / sqrt(D)     per hc copy c
    gate_c      = sigmoid(sign(dot_c) * sqrt(max(|dot_c|, 1e-6)))
    h_c        += gate_c * value

The hash and the table gather are host work (the tables are ~98 GB per layer): tt-blaze's ``EngramHost`` (the
checkpoint's own ``engram.py`` hashing) returns ``rows`` per position; they are uploaded per chunk, SP-sharded on S and
replicated over TP. The device does ``wkv`` and the gate. ``wkv``'s output columns are laid out per chip as
``[key_0 | key_1 | key_2 | key_3 | value]`` of that chip's D slice, so every key / value slice lines up with the
TP-sharded streams; the three per-(token, copy) sums (``h * w * key``, ``h^2``, ``key^2``) are TP partials, packed into
one fp32 [S_l, 32] row and finished by the one TP all-reduce (``tt/v4/hyper_connection.py``'s pattern).
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.deepseek_v3_d_p.tt.v4.hyper_connection import _TtHyperBase

HC = 4
ROW = 32
DOT0, HSQ0, KSQ0 = 0, 4, 8  # columns of the partial-sum row


class V41PrefillEngram(_TtHyperBase):
    def __init__(self, device, cfg, layer: int, w: dict, *, sp_axis: int = 0, tp_axis: int = 1):
        D = int(cfg.dim)
        super().__init__(device, hidden=D, rms_eps=cfg.norm_eps, hc_eps=cfg.hc_eps, sp_axis=sp_axis, tp_axis=tp_axis)
        self.layer, self.D, self.eps = int(layer), D, float(cfg.norm_eps)
        tp = self.tp_factor
        D_l = D // tp
        self.D_l = D_l
        wkv = w["engram.wkv.weight"].float()  # [5 * D, K] (fp8 dequantised by V41Checkpoint)
        K = int(wkv.shape[1])
        assert wkv.shape[0] == (HC + 1) * D, wkv.shape
        # [K, 5, tp, D_l] -> [K, tp, 5, D_l]: chip t's columns are (key_0..key_3, value) of its D slice
        wt = wkv.t().reshape(K, HC + 1, tp, D_l).permute(0, 2, 1, 3).reshape(1, 1, K, tp * (HC + 1) * D_l)
        self.wkv = ttnn.from_torch(
            wt.contiguous(),
            device=device,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mesh_mapper(tp_dim=3),
        )
        qk = (w["engram.q_weight"].float() * w["engram.k_weight"].float()).reshape(HC, 1, 1, D)  # only the product
        self.qk = [self._const(qk[c].reshape(1, 1, 1, D), tp_dim=3) for c in range(HC)]
        # one-hot column selectors [D_l, 32] (the partial sums land in their own columns of one row)
        self.E = {}
        for c in range(3 * HC):
            e = torch.zeros(1, 1, D_l, ROW)
            e[..., c] = 1.0
            self.E[c] = self._const(e)
        # row -> per-copy columns 0..3: dot (identity on 0..3), mean h^2, mean key^2
        sel = lambda src, scale: torch.zeros(ROW, ROW).index_put_(  # noqa: E731
            (torch.arange(src, src + HC), torch.arange(HC)), torch.full((HC,), float(scale))
        )
        self.M_DOT = self._const(sel(DOT0, 1.0).view(1, 1, ROW, ROW))
        self.M_HSQ = self._const(sel(HSQ0, 1.0 / D).view(1, 1, ROW, ROW))
        self.M_KSQ = self._const(sel(KSQ0, 1.0 / D).view(1, 1, ROW, ROW))

    def upload_rows(self, rows: torch.Tensor, seq_pad: int):
        """[S, 6144] host rows -> [1, 1, S_pad / sp, 6144] bf16 (SP-sharded on S, replicated over TP); pad rows are zero."""
        S, K = rows.shape
        t = torch.zeros(1, 1, seq_pad, K, dtype=torch.bfloat16)
        t[0, 0, :S] = rows.to(torch.bfloat16)
        return ttnn.from_torch(
            t,
            device=self.device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._mesh_mapper(sp_dim=2),
        )

    def forward(self, streams: list, rows_dev) -> list:
        assert len(streams) == HC
        D_l = self.D_l
        S_l = int(streams[0].shape[2])
        kv = ttnn.linear(rows_dev, self.wkv, memory_config=self.memory_config)  # [1, 1, S_l, 5 * D_l]
        kvf = ttnn.typecast(kv, ttnn.float32)
        ttnn.deallocate(kv)
        key = [ttnn.slice(kvf, [0, 0, 0, c * D_l], [1, 1, S_l, (c + 1) * D_l]) for c in range(HC)]
        value = ttnn.slice(kvf, [0, 0, 0, HC * D_l], [1, 1, S_l, (HC + 1) * D_l])
        xf = [ttnn.typecast(x, ttnn.float32) for x in streams]
        part = None
        for c in range(HC):
            terms = (
                (ttnn.multiply(ttnn.multiply(xf[c], self.qk[c]), key[c]), DOT0 + c),
                (ttnn.multiply(xf[c], xf[c]), HSQ0 + c),
                (ttnn.multiply(key[c], key[c]), KSQ0 + c),
            )
            for t, col in terms:
                p = self._mm(t, self.E[col])
                ttnn.deallocate(t)
                part = p if part is None else ttnn.add(part, p)
        row = self._tp_all_reduce(part)  # [1, 1, S_l, 32] fp32 full sums
        dot = self._mm(row, self.M_DOT)
        rstd = ttnn.multiply(
            ttnn.rsqrt(ttnn.add(self._mm(row, self.M_HSQ), self.eps)),
            ttnn.rsqrt(ttnn.add(self._mm(row, self.M_KSQ), self.eps)),
        )
        dot = ttnn.multiply(ttnn.multiply(dot, rstd), self.D**-0.5)
        signed = ttnn.multiply(ttnn.sign(dot), ttnn.sqrt(ttnn.clamp(ttnn.abs(dot), min=1e-6)))
        gate = ttnn.sigmoid(signed)
        out = []
        for c in range(HC):
            o = ttnn.add(xf[c], ttnn.multiply(value, self._col(gate, c)))
            out.append(ttnn.typecast(o, streams[c].dtype))
            ttnn.deallocate(o)
        for t in xf + key + [value, kvf]:
            ttnn.deallocate(t)
        return out

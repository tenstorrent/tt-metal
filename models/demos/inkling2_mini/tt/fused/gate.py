# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused message gate (1 - tanh(F z)) * h + tanh(M z) * m, z = down(cat(rms h, rms m)): 4 device programs
instead of ~17 (E2 residual.MessageGate) or ~19 (model._QKVGate):
    prep     (generic op, one core per vector): rms of each h / m row, bf16, written straight into the split-K
             layout of the down_proj input (no concat / reshape / permute / typecast)
    matmul + sum  (E2's split-K down_proj, unchanged)
    combine  (generic op, one core per output column tile, no cross-core sync): bf16 z [* block mask], both
             rank-64 up projections, tanh, and the fp32 mix, rows 0..R-1 written to a zeroed output.

FusedMessageGate: drop-in for residual.MessageGate (rows = vectors, each through the same gate).
FusedQKVGate: drop-in for model._QKVGate (row i of h / m through gate q, k, v; block-masked like _QKVGate).
"""

import torch

import ttnn
from models.demos.inkling2_mini.tt import residual as R
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.attnres import f32_bits

RANK = 64


def _t(w):
    return ttnn.to_torch(w) if isinstance(w, ttnn.Tensor) else w


class _FusedGate:
    def __init__(self, device, downs, forgets, gains, rows, eps=1e-6, combine_cores=None):
        """downs: [2C, 64] per gate; forgets / gains: [64, C] per gate; rows: number of gated vectors R.
        With several gates, row i goes through gate i (block mask)."""
        self.device = device
        self.eps = eps
        self.rows = rows
        self.c = forgets[0].shape[-1]
        self.ntc = self.c // 32
        self.ckc = R.compute_config(device)
        g = len(downs)
        self.kz = g * RANK // 32
        self.down = R.split_k_weight(torch.cat(downs, -1), device)  # [1, SPLIT, 2C/SPLIT, 64 g] bf16
        self.f = R.to_device(torch.cat(forgets).float(), device)  # [1, 1, 64 g, C]
        self.g = R.to_device(torch.cat(gains).float(), device)
        self.mask = None
        if g > 1:
            m = torch.zeros(rows, g * RANK)
            for i in range(rows):
                m[i, i * RANK : (i + 1) * RANK] = 1
            self.mask = R.to_device(m, device)
        ks = 2 * self.c // R.SPLIT
        assert ks % 32 == 0 and self.ntc % 1 == 0
        self.xs = ttnn.from_torch(
            torch.zeros(1, R.SPLIT, rows, ks), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )
        m = torch.zeros(1, 1, 32, 32)
        m[..., 0, :] = 1.0
        m[..., 16, :] = 1.0
        self.scaler = ttnn.from_torch(m, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        self.out = F.OutCache(device)
        n = combine_cores or self.ntc
        assert self.ntc % n == 0
        self.nt = self.ntc // n
        self.prep_cores = F.core_list(device, 2 * rows)
        self.comb_cores = F.core_list(device, n)

    def _prep(self, h, m):
        T = F.pages(self.c)
        cores, crs = self.prep_cores
        rd, wr = {}, {}
        for v, core in enumerate(cores):
            src, r = (h, v) if v < self.rows else (m, v - self.rows)
            rd[core] = [
                src.buffer_address(),
                F.tile_bytes(src.dtype),
                F.ES[src.dtype],
                0,
                0,
                self.scaler.buffer_address(),
                r,
                0,
            ]
            wr[core] = [self.xs.buffer_address(), r, 0 if v < self.rows else self.ntc]
        named = dict(ntc=self.ntc, pages=T, nq=0, k=0, has_weight=0)
        assert h.dtype == m.dtype
        kernels = [
            F.kernel("attnres_reader.cpp", ttnn.ReaderConfigDescriptor(), named, rd, crs),
            F.kernel("gate_prep_writer.cpp", ttnn.WriterConfigDescriptor(), named, wr, crs),
            F.kernel(
                "attnres_compute.cpp",
                F.compute_config([], full_sync=True),
                dict(pages=T, nq=0, k=0, has_weight=0, inv_c=f32_bits(1.0 / self.c), eps=f32_bits(self.eps)),
                None,
                crs,
            ),
        ]
        f32 = ttnn.float32
        cbs = [
            F.cb(0, h.dtype, T, cores=crs),
            F.cb(1, f32, 1, cores=crs),
            F.cb(2, f32, 1, cores=crs),
            F.cb(3, f32, 1, cores=crs),
            F.cb(16, ttnn.bfloat16, T, cores=crs),
            F.cb(17, f32, 1, cores=crs),
            F.stage_cb(24, self.ntc, crs),
            F.cb(25, f32, 1, cores=crs),
            F.cb(26, f32, 1, cores=crs),
            F.cb(27, f32, 1, cores=crs),
            F.cb(28, f32, 1, cores=crs),
        ]
        F.run([h, m, self.scaler, self.xs], kernels, cbs)

    def _combine(self, z, h, m):
        cores, crs = self.comb_cores
        out = self.out.get("out", self.rows, self.c, ttnn.float32)
        has_mask = int(self.mask is not None)
        rd, wr = {}, {}
        for i, core in enumerate(cores):
            col0 = i * self.nt
            rd[core] = [
                z.buffer_address(),
                self.mask.buffer_address() if has_mask else 0,
                self.f.buffer_address(),
                self.g.buffer_address(),
                h.buffer_address(),
                m.buffer_address(),
                col0,
            ]
            wr[core] = [out.buffer_address(), col0]
        named = dict(
            kz=self.kz, nt=self.nt, nct=self.ntc, has_mask=has_mask, h_tile=F.tile_bytes(h.dtype), rows=self.rows
        )
        fp32_cbs = [0, 1] + ([4, 5] if h.dtype == ttnn.float32 else [])
        kernels = [
            F.kernel("gate_combine_reader.cpp", ttnn.ReaderConfigDescriptor(), named, rd, crs),
            F.kernel("gate_combine_writer.cpp", ttnn.WriterConfigDescriptor(), named, wr, crs),
            F.kernel("gate_combine_compute.cpp", F.compute_config(fp32_cbs), named, None, crs),
        ]
        f32 = ttnn.float32
        cbs = [
            F.cb(0, f32, self.kz, cores=crs),
            F.cb(1, f32, self.kz, cores=crs),
            F.cb(2, f32, self.kz * self.nt, cores=crs),
            F.cb(3, f32, self.kz * self.nt, cores=crs),
            F.cb(4, h.dtype, self.nt, cores=crs),
            F.cb(5, h.dtype, self.nt, cores=crs),
            F.cb(16, ttnn.bfloat16, self.kz, cores=crs),
            F.cb(17, f32, self.nt, cores=crs),
        ]
        io = [z, h, m, self.f, self.g] + ([self.mask] if has_mask else []) + [out]
        return F.run(io, kernels, cbs)

    def __call__(self, h, m):
        """h, m: [1,1,R,C] row-form DRAM tensors of one dtype (fp32 or bf16). Returns fp32 [1,1,R,C]."""
        F.check_row_input(h)
        F.check_row_input(m)
        if h.dtype != m.dtype:
            h, m = R._f32(h), R._f32(m)
        assert h.shape[-2] == self.rows and m.shape[-2] == self.rows, (h.shape, m.shape, self.rows)
        self._prep(h, m)
        z = ttnn.sum(
            ttnn.matmul(self.xs, self.down, dtype=ttnn.float32, compute_kernel_config=self.ckc), dim=1, keepdim=True
        )
        return self._combine(z, h, m)


class FusedMessageGate(_FusedGate):
    """Drop-in for residual.MessageGate(rows, weights, which). rows: [1,1,r,C] count (default 1, the model's mlp
    gate); the T5 test gates 4 rows at once."""

    def __init__(self, rows, weights, which, eps=1e-6, num_rows=1):
        p = f"msg_gate.{which}."
        c = _t(weights[p + "forget"]).shape[-1]
        down = _t(weights[p + "down"]).reshape(2 * c, RANK)
        forget = _t(weights[p + "forget"]).reshape(RANK, c)
        gain = _t(weights[p + "gain"]).reshape(RANK, c)
        super().__init__(rows.device, [down], [forget], [gain], num_rows, eps)


class FusedQKVGate(_FusedGate):
    """Drop-in for model._QKVGate(rows, w): rows 0/1/2 of h, m through gates q, k, v."""

    def __init__(self, rows, w, eps=1e-6):
        g3 = ("q", "k", "v")
        c = _t(w["msg_gate.q.forget"]).shape[-1]
        downs = [_t(w[f"msg_gate.{g}.down"]).reshape(2 * c, RANK) for g in g3]
        forgets = [_t(w[f"msg_gate.{g}.forget"]).reshape(RANK, c) for g in g3]
        gains = [_t(w[f"msg_gate.{g}.gain"]).reshape(RANK, c) for g in g3]
        super().__init__(rows.device, downs, forgets, gains, 3, eps)

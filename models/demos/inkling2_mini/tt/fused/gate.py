# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""Fused message gate (1 - tanh(F z)) * h + tanh(M z) * m, z = down(cat(rms h, rms m)): 4 device programs
instead of ~17 (unfused message gate) or ~19 (three unfused gates in one call):
    prep     (generic op, one core per vector): rms of each h / m row, bf16, written straight into the split-K
             layout of the down_proj input (no concat / reshape / permute / typecast)
    matmul + sum  (residual.split_k_matmul, unchanged)
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


Z_COPIES = 8  # compact z pages (one per DRAM bank): the combine cores spread their z reads over them


class _FusedGate:
    def __init__(self, device, downs, forgets, gains, rows, eps=1e-6, combine_cores=None):
        """downs: [2C, 64] per gate; forgets / gains: [64, C] per gate; rows: number of gated vectors R.
        With several gates, row i goes through gate i (block mask)."""
        self.device = device
        self.eps = eps
        self.rows = rows
        self.c = forgets[0].shape[-1]
        self.ntc = self.c // 32
        g = len(downs)
        self.kz = g * RANK // 32
        self.down = R.split_k_weight(torch.cat(downs, -1), device)  # [1, SPLIT, 2C/SPLIT, 64 g] bf16
        # forget / gain: bf16 when every value is bf16 (checkpoint dtype): same operand values for the FPU
        # (it reads fp32 as tf32, which holds any bf16 exactly), half the combine's DRAM bytes
        f, g_ = torch.cat(forgets).float(), torch.cat(gains).float()
        exact = all(torch.equal(t, t.to(torch.bfloat16).float()) for t in (f, g_))
        self.fg_dtype = ttnn.bfloat16 if exact else ttnn.float32
        self.f = R.to_device(f, device, self.fg_dtype)  # [1, 1, 64 g, C]
        self.g = R.to_device(g_, device, self.fg_dtype)
        self.block = g > 1  # row i goes through gate i only (z tiles 2i, 2i + 1; applied by the combine reader)
        assert not self.block or (g == rows and RANK == 64)
        ks = 2 * self.c // R.SPLIT
        assert ks % 32 == 0 and self.ntc % 1 == 0
        self.kt = ks // 32  # K tiles per split
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
        assert 2 * rows <= R.SPLIT
        self.z_parts = 2  # K parts per split: 18 cores of 8 K tiles (fewer matmul tiles per core)
        self.z_cores = F.core_list(device, R.SPLIT * self.z_parts)
        # packed partial rows on the root: sender s row r -> packed row (s - 1) * rows + r; S_b[r, j] selects them
        n_rows = (R.SPLIT * self.z_parts - 1) * rows
        self.z_nb = -(-n_rows // 32)
        sel = torch.zeros(32, 32 * self.z_nb)
        for g in range(n_rows):
            sel[g % rows, g] = 1.0
        self.sel = ttnn.from_torch(sel[None, None], dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)

    def _prep(self, h, m):
        """The gate prep part of _prep_z (bf16 rms rows of h and m, one prep core each): per-core args and CBs."""
        T = F.pages(self.c)
        cores, crs = self.prep_cores[0], self.z_cores[1]
        rd, wr = {}, {}
        # one CB format for all prep cores: read dense twins only when h and m then share a dtype
        use_twin = F.input_dtype(h) == F.input_dtype(m)
        in_dtype = F.input_dtype(h) if use_twin else h.dtype
        for v, core in enumerate(cores):
            src, r = (h, v) if v < self.rows else (m, v - self.rows)
            if use_twin:
                args = F.input_args(src, r)
            else:
                args = [src.buffer_address(), F.tile_bytes(src.dtype), F.ES[src.dtype], r, 0]
            rd[core] = [*args, 0, 0, self.scaler.buffer_address(), 0]
            wr[core] = [0, 0, 0]
        named = dict(ntc=self.ntc, pages=T, nq=0, k=0, has_weight=0, split_out=0)
        assert h.dtype == m.dtype
        f32 = ttnn.float32
        cbs = [
            F.cb(0, in_dtype, T, cores=crs),
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
        tws = [t[0] for t in (F.twin(h), F.twin(m)) if t is not None]
        io = [h, m, *tws, self.scaler]
        cnamed = dict(pages=T, nq=0, k=0, has_weight=0, inv_c=f32_bits(1.0 / self.c), eps=f32_bits(self.eps))
        return dict(rd=[rd[c] for c in cores], wr=[wr[c] for c in cores], named=named, cnamed=cnamed, cbs=cbs, io=io)

    def _prep_z(self, h, m):
        """Gate prep + z = x @ down in one program: one core per K part; the first 2 R cores also run the prep
        (bf16 rms rows, kept in their L1) and signal every core, which then reads its x tiles from those rows;
        partial z rows are summed on core 0.
        Returns z in compact form: Z_COPIES pages, each rows 0..R-1 of the kz z tiles (128 B per tile row)."""
        prep = self._prep(h, m)
        cores, crs = self.z_cores
        n_prep = len(prep["rd"])  # the prep cores are the first split cores
        assert self.kz * self.rows * 128 <= 4096
        z = self.out.get("zc", 32, 32 * Z_COPIES, ttnn.float32)  # Z_COPIES pages of compact z rows
        virt = [self.device.worker_core_from_logical_core(c) for c in cores]
        xy = [v for c in virt for v in (c.x, c.y)]
        kt = self.kt // self.z_parts
        rd, cp, wr = {}, {}, {}
        for s, core in enumerate(cores):
            is_prep = int(s < n_prep)
            prd, pwr = (prep["rd"][s], prep["wr"][s]) if is_prep else ([0] * 9, [0] * 3)
            x = self.sel.buffer_address()  # reader arg A + 0: the root reads the selection tiles
            first = s * kt  # K tile of the whole x: split s // parts, its part s % parts
            rd[core] = [*prd, x, self.down.buffer_address(), first, first * self.kz, is_prep, *xy]
            cp[core] = [s, is_prep]
            wr[core] = [*pwr, z.buffer_address(), s, is_prep, *xy, *(prd[:5] if is_prep else [0] * 5)]
        named = dict(
            prep["named"],
            kt=kt,
            kz=self.kz,
            n_split=len(cores),
            n_prep=n_prep,
            sem_x=0,
            sem_z=1,
            rows=self.rows,
            nb=self.z_nb,
            z_copies=Z_COPIES,
        )
        cnamed = dict(prep["cnamed"], kt=kt, kz=self.kz, n_split=len(cores), nb=self.z_nb)
        kernels = [
            F.kernel("gatez_reader.cpp", ttnn.ReaderConfigDescriptor(), named, rd, crs),
            F.kernel("gatez_writer.cpp", ttnn.WriterConfigDescriptor(), named, wr, crs),
            F.kernel("gatez_compute.cpp", F.compute_config([]), cnamed, cp, crs),
        ]
        f32 = ttnn.float32
        cbs = [c for c in prep["cbs"]] + [
            F.cb(5, ttnn.bfloat16, kt, cores=crs),
            F.cb(6, ttnn.bfloat16, kt * self.kz, cores=crs),
            F.cb(7, f32, self.z_nb * self.kz, cores=crs),
            F.cb(8, f32, self.z_nb, cores=crs),
            F.cb(9, f32, 1, cores=crs),
            F.cb(18, f32, self.kz, cores=crs),
            F.cb(29, ttnn.bfloat16, 1, 32, cores=crs),
            F.cb(31, f32, 1, cores=crs),  # root: compact z rows  # prep writer -> reader: second half of the row landed
            F.stage_cb(30, self.ntc, crs),
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in (0, 1)]
        io = prep["io"] + [self.down, self.sel, z]
        return ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))

    def _combine(self, z, h, m):
        cores, crs = self.comb_cores
        out, out_dense = self.out.get_twin("out", self.rows, self.c, ttnn.float32)
        block = int(self.block)

        def src(t):  # [addr, dense, nvec]: fp32 dense twins are read row by row
            tw = F.twin(t)
            if tw is not None and t.dtype == ttnn.float32:
                return tw[0].buffer_address(), 1, tw[1]
            return t.buffer_address(), 0, 0

        (h_a, h_d, h_n), (m_a, m_d, m_n) = src(h), src(m)
        rd, wr = {}, {}
        for i, core in enumerate(cores):
            col0 = i * self.nt
            rd[core] = [z.buffer_address(), 0, self.f.buffer_address(), self.g.buffer_address(), h_a, m_a, col0]
            rd[core] += [h_d, m_d, h_n, m_n, i % Z_COPIES]
            wr[core] = [out.buffer_address(), col0, out_dense.buffer_address(), self.g.buffer_address()]
            wr[core] += [self.f.buffer_address()]
        named = dict(
            kz=self.kz,
            nt=self.nt,
            nct=self.ntc,
            has_mask=0,
            # F rows the writer reads (with G): F + G are DRAM-bound, split them evenly over both RISCs
            f_w=self.kz // 2 if self.kz > 2 else 0,
            block=block,
            h_tile=F.tile_bytes(h.dtype),
            fg_page=F.tile_bytes(self.fg_dtype),
            rows=self.rows,
            pages=F.pages(self.c),
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
            F.cb(2, self.fg_dtype, self.kz * self.nt, cores=crs),
            F.cb(3, self.fg_dtype, self.kz * self.nt, cores=crs),
            F.cb(4, h.dtype, self.nt, cores=crs),
            F.cb(5, h.dtype, self.nt, cores=crs),
            F.cb(6, ttnn.bfloat16, 1, 32, cores=crs),  # writer -> reader: its F rows landed
            F.cb(16, ttnn.bfloat16, self.kz, cores=crs),
            F.cb(17, f32, self.nt, cores=crs),
        ]
        tws = [t[0] for t in (F.twin(h), F.twin(m)) if t is not None]
        io = [z, h, m, *tws, self.f, self.g, out_dense, out]
        F.run(io, kernels, cbs)
        return out  # the cached object itself: its dense twin is keyed by identity

    def __call__(self, h, m):
        """h, m: [1,1,R,C] row-form DRAM tensors of one dtype (fp32 or bf16). Returns fp32 [1,1,R,C]."""
        F.check_row_input(h)
        F.check_row_input(m)
        if h.dtype != m.dtype:
            h, m = R._f32(h), R._f32(m)
        assert h.shape[-2] == self.rows and m.shape[-2] == self.rows, (h.shape, m.shape, self.rows)
        return self._combine(self._prep_z(h, m), h, m)


class FusedMessageGate(_FusedGate):
    """Drop-in for residual.MessageGate(rows, weights, which). rows: [1,1,r,C] count (default 1, the model's mlp
    gate); the tests gate 4 rows at once."""

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

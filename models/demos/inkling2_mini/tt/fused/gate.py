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
    def __init__(self, device, downs, forgets, gains, rows, eps=1e-6, combine_cores=None, fused=True, group=None):
        """downs: [2C, 64] per gate; forgets / gains: [64, C] per gate; rows: number of gated vectors R.
        With several gates, row i goes through gate i (block mask). fused: z and combine in one program (else two
        programs, z through DRAM); same outputs."""
        self.device = device
        self.eps = eps
        self.rows = rows
        group = group or rows  # rows per token (the partial-sum layout of a 1-token gate of `group` rows)
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
        # row i goes through gate i % g only (z tiles 2 (i % g), +1; applied by the combine reader); R verify rows of
        # the q/k/v gates: rows 3r + j, gate j
        self.block = g > 1
        assert not self.block or (rows % g == 0 and RANK == 64)
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
        assert 2 * rows <= R.SPLIT or fused  # the fused gate preps on cores of their own
        self.z_parts = 2  # K parts per split: 18 cores of 8 K tiles (fewer matmul tiles per core)
        n_z = R.SPLIT * self.z_parts
        # z and combine in ONE program (fused): combine on rectangles below row 0, z on the other cores (row 0 first).
        # The z root multicasts z into each rectangle; combine reads F / G from the start. A multicast rectangle must
        # stay inside one run of logical columns with consecutive virtual x (Blackhole has non-Tensix columns
        # inside the grid).
        g = device.compute_with_storage_grid_size()
        self.rects = _rects(device, n) if fused else None
        self.fused = self.rects is not None and g.x * g.y >= n + n_z + 2 * rows
        if self.fused:
            comb = [
                ttnn.CoreCoord(x, y) for a, b in self.rects for y in range(a.y, b.y + 1) for x in range(a.x, b.x + 1)
            ]
            inside = {(c.x, c.y) for c in comb}
            rest = [ttnn.CoreCoord(x, 0) for x in range(g.x)] + [
                ttnn.CoreCoord(x, y) for y in range(1, g.y) for x in range(g.x) if (x, y) not in inside
            ]
            z = rest[: n_z + 2 * rows]  # K parts, then prep-only cores (their weight reads start at once)
            ranges = [ttnn.CoreRange(a, b) for a, b in self.rects]
            self.comb_cores = comb, ttnn.CoreRangeSet(ranges)
            self.z_cores = z, ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in z])
            self.all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in z] + ranges)
        else:
            self.comb_cores = F.core_list(device, n)
            self.z_cores = F.core_list(device, n_z)
        # packed partial rows on the root, per token of `group` rows (rows = tokens x group): row j of sender s ->
        # row (s - 1) * group + j of the token's block of nbt tiles, as the 1-token gate packs them, so each token's
        # sums round as there (S_b[r, g] = 1 where packed row g came from row r)
        n_send = R.SPLIT * self.z_parts - 1
        assert rows % group == 0
        self.group = group
        nbt = -(-n_send * group // 32)
        self.z_nb = rows // group * nbt
        sel = torch.zeros(32, 32 * self.z_nb)
        for r in range(rows):
            for s_ in range(n_send):
                sel[r, (r // group) * nbt * 32 + s_ * group + r % group] = 1.0
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
        named = dict(ntc=self.ntc, pages=T, nq=0, k=0, has_weight=0, has_norm=0, split_out=0)
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
        cnamed = dict(
            pages=T, nq=0, k=0, has_weight=0, has_norm=0, inv_c=f32_bits(1.0 / self.c), eps=f32_bits(self.eps)
        )
        return dict(rd=[rd[c] for c in cores], wr=[wr[c] for c in cores], named=named, cnamed=cnamed, cbs=cbs, io=io)

    def _prep_z(self, h, m, parts=False):
        """Gate prep + z = x @ down in one program: one core per K part; the first 2 R cores also run the prep
        (bf16 rms rows, kept in their L1) and signal every core, which then reads its x tiles from those rows;
        partial z rows are summed on core 0.
        Returns z in compact form: Z_COPIES pages, each rows 0..R-1 of the kz z tiles (128 B per tile row)."""
        prep = self._prep(h, m)
        cores, crs = self.z_cores
        n_prep = len(prep["rd"])
        n_split = R.SPLIT * self.z_parts
        # fused: prep-only cores after the K parts (role 2); else the first n_prep K-part cores also prep (role 1)
        mm, pc = cores[:n_split], (cores[n_split:] if self.fused else cores[:n_prep])
        zb = self.kz * self.rows * 128  # compact z bytes (one DRAM page per copy unless fused)
        assert (zb <= 4096 or self.fused) and len(pc) == n_prep
        z = self.out.get("zc", 32, 32 * Z_COPIES, ttnn.float32)  # Z_COPIES pages of compact z rows
        vxy = lambda cs: [
            v for c in cs for v in (lambda w: (w.x, w.y))(self.device.worker_core_from_logical_core(c))
        ]  # noqa: E731
        mm_xy, pc_xy = vxy(mm), vxy(pc)
        kt = self.kt // self.z_parts
        mc = []
        if self.fused:
            mc = [len(self.rects)]
            for a, b in self.rects:
                a0, a1 = (self.device.worker_core_from_logical_core(c) for c in (a, b))
                mc += [a0.x, a0.y, a1.x, a1.y, (b.x - a.x + 1) * (b.y - a.y + 1)]
        rd, cp, wr = {}, {}, {}
        roles = [(c, s, int(not self.fused and s < n_prep), s if not self.fused else None) for s, c in enumerate(mm)]
        if self.fused:
            roles += [(c, n_split + v, 2, v) for v, c in enumerate(pc)]
        for core, s, role, v in roles:
            prd, pwr = (prep["rd"][v], prep["wr"][v]) if role else ([0] * 9, [0] * 3)
            x = self.sel.buffer_address()  # reader arg A + 0: the root reads the selection tiles
            first = s * kt if role != 2 else 0  # K tile of the whole x: split s // parts, its part s % parts
            rd[core] = [*prd, x, self.down.buffer_address(), first, first * self.kz, role, *pc_xy, *mc]
            cp[core] = [s, role]
            wr[core] = [*pwr, z.buffer_address(), s, role, *mm_xy, *(prd[:5] if role else [0] * 5), *mc]
        named = dict(
            prep["named"],
            kt=kt,
            kz=self.kz,
            n_split=n_split,
            n_prep=n_prep,
            sem_x=0,
            sem_z=1,
            rows=self.rows,
            nb=self.z_nb,
            group=self.group,
            z_copies=Z_COPIES,
            fused=int(self.fused),
            sem_zl=2,
            sem_go_r=3,
            sem_go_w=4,
        )
        cnamed = dict(prep["cnamed"], kt=kt, kz=self.kz, n_split=n_split, nb=self.z_nb)
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
            # root: compact z rows (fused: also where they land on the combine cores; one address everywhere)
            F.cb(31, f32, -(-zb // 4096), cores=self.all_cores if self.fused else crs),
            F.stage_cb(30, self.ntc, crs),  # prep writer -> reader: second half of the row landed
        ]
        sems = [ttnn.SemaphoreDescriptor(id=i, core_ranges=crs, initial_value=0) for i in (0, 1)]
        io = prep["io"] + [self.down, self.sel, z]
        if parts:
            return kernels, cbs, sems, io
        return ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=kernels, cbs=cbs, semaphores=sems))

    def _combine(self, z, h, m, parts=False):
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
            fused=int(self.fused),
            sem_zl=2,
            sem_go_r=3,
            sem_go_w=4,
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
        if parts:
            return kernels, cbs, io, out
        F.run(io, kernels, cbs)
        return out  # the cached object itself: its dense twin is keyed by identity

    def _fused_call(self, h, m):
        """z and combine as one program on disjoint cores (z multicast core to core, no DRAM round trip)."""
        zk, zcb, zsem, zio = self._prep_z(h, m, parts=True)
        z = zio[-1]  # the DRAM z buffer (unused here, kept for the arg layout)
        ck, ccb, cio, out = self._combine(z, h, m, parts=True)
        sems = zsem + [ttnn.SemaphoreDescriptor(id=i, core_ranges=self.all_cores, initial_value=0) for i in (2, 3, 4)]
        io = zio + [t for t in cio if all(t is not u for u in zio)]
        io = [t for t in io if t is not out] + [out]
        ttnn.generic_op(io, ttnn.ProgramDescriptor(kernels=zk + ck, cbs=zcb + ccb, semaphores=sems))
        return out

    def __call__(self, h, m):
        """h, m: [1,1,R,C] row-form DRAM tensors of one dtype (fp32 or bf16). Returns fp32 [1,1,R,C]."""
        F.check_row_input(h)
        F.check_row_input(m)
        if h.dtype != m.dtype:
            h, m = R._f32(h), R._f32(m)
        assert h.shape[-2] == self.rows and m.shape[-2] == self.rows, (h.shape, m.shape, self.rows)
        if self.fused:
            return self._fused_call(h, m)
        return self._combine(self._prep_z(h, m), h, m)


def _rects(device, n):
    """<= 2 rectangles of logical cores (rows 1..) holding exactly n cores, each inside a run of logical columns with
    consecutive virtual x; None if there is no such split."""
    g = device.compute_with_storage_grid_size()
    vx = [device.worker_core_from_logical_core(ttnn.CoreCoord(x, 1)).x for x in range(g.x)]
    runs, a = [], 0
    for x in range(1, g.x + 1):
        if x == g.x or vx[x] != vx[x - 1] + 1:
            runs.append((a, x - 1))
            a = x
    rows = g.y - 1
    for ra in runs:
        wa = ra[1] - ra[0] + 1
        for ha in range(rows, 0, -1):
            rem = n - wa * ha
            if rem == 0:
                return [(ttnn.CoreCoord(ra[0], 1), ttnn.CoreCoord(ra[1], ha))]
            for rb in runs:
                wb = rb[1] - rb[0] + 1
                if rb != ra and rem > 0 and rem % wb == 0 and rem // wb <= rows:
                    return [
                        (ttnn.CoreCoord(ra[0], 1), ttnn.CoreCoord(ra[1], ha)),
                        (ttnn.CoreCoord(rb[0], 1), ttnn.CoreCoord(rb[1], rem // wb)),
                    ]
    return None


class FusedMessageGate(_FusedGate):
    """Drop-in for residual.MessageGate(rows, weights, which). rows: [1,1,r,C] count (default 1, the model's mlp
    gate); the tests gate 4 rows at once."""

    def __init__(self, rows, weights, which, eps=1e-6, num_rows=1):
        p = f"msg_gate.{which}."
        c = _t(weights[p + "forget"]).shape[-1]
        down = _t(weights[p + "down"]).reshape(2 * c, RANK)
        forget = _t(weights[p + "forget"]).reshape(RANK, c)
        gain = _t(weights[p + "gain"]).reshape(RANK, c)
        # num_rows rows = num_rows verify rows of the model's 1-row gate (each sums as the 1-row gate)
        super().__init__(rows.device, [down], [forget], [gain], num_rows, eps, group=1)


class FusedQKVGate(_FusedGate):
    """Drop-in for model._QKVGate(rows, w): rows 0/1/2 of h, m through gates q, k, v."""

    def __init__(self, rows, w, eps=1e-6, num_tokens=1):
        """num_tokens = R: R verify rows, rows 3r + j of h / m through gate j."""
        g3 = ("q", "k", "v")
        c = _t(w["msg_gate.q.forget"]).shape[-1]
        downs = [_t(w[f"msg_gate.{g}.down"]).reshape(2 * c, RANK) for g in g3]
        forgets = [_t(w[f"msg_gate.{g}.forget"]).reshape(RANK, c) for g in g3]
        gains = [_t(w[f"msg_gate.{g}.gain"]).reshape(RANK, c) for g in g3]
        super().__init__(rows.device, downs, forgets, gains, 3 * num_tokens, eps, group=3)

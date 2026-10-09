# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Temperature / top-k / top-p sampling INSIDE the decode trace (DeepSeek-V4.1-Flash, vocab sharded over the 8 mesh columns).

The host supplies, per user and per step, four numbers in one small tensor ``params`` [T, 4] fp32 (row-sharded like the tokens):
``invT`` (1 / temperature), ``top_k`` (0 = off), ``top_p`` (1 = off), ``u`` (uniform draw in [0, 1), from the request's own generator: seeds stay on the host)
and a greedy flag. The device returns the token id per user: [T, 1] uint32, identical on every device. No candidate read, no host draw, no full-row fallback.

Algorithm (over the FULL vocabulary row, no top-k truncation of the candidates):
  s = (logits - max) * invT <= 0,  e = exp(s)                 (per column, 16160 entries; Z = cross-column sum of e)
  top-k   : tau_k = largest tau with  count(s >= tau) >= k      (k-ary threshold search, ``levels`` x ``J`` thresholds, one cross-column sum per level)
  top-p   : tau_p = largest tau >= tau_k with mass(s >= tau) >= top_p * mass(s >= tau_k)
  keep    : s >= tau_p;  draw = inverse CDF of e * keep in VOCABULARY order at u * total (per-column cumsum, 8 column totals gathered)
Because the nucleus is a threshold set, no sort is needed; the draw in vocabulary order has the same distribution as the draw in sorted order. The only approximation is
the resolution of the threshold search: tokens whose scaled logit lies within (range / J**levels) of the exact boundary may be kept or dropped wrongly. The top-k count search resolves one level more than the
top-p mass search (``levels_k``): on a flat row the k-th and (k+1)-th largest scaled logits are closer than range / J**levels, which kept k + 1 tokens in ~20% of the cases (tests/test_device_sampler*.py).

Rows with 0 < top_k <= ``kcand`` (32) take a second, much cheaper path (``forward_topk``, ~0.7 ms at 8 users per mesh row, ~1.1 ms at 32, against ~3.8 / ~9.5 ms): the kept set lives in the ``kcand`` largest logits of each
mesh column, so the top-k set, the nucleus and the draw are computed exactly over those 8 * kcand candidates (no search, no resolution band). ``set_params`` selects it for a step when every sampled row qualifies
(``DSV41DeviceSampler.fast``); a step with a row without top_k (or top_k > kcand) replays the full-vocabulary trace. ``SamplerTraces`` holds the two traces.

Execution: the sampler is its own trace (``DSV41Decoder.forward_sampler`` / ``SpecDecoder.forward_sampler``), replayed after the step trace only when a row of the step is sampled: the step trace itself
stays greedy (``sample_global``), so greedy serving does not pay the ~3.8 ms (8 users per mesh row) - 9.5 ms (32 rows) of the sampler.
"""

import torch

import ttnn

VOCAB = 129280


class DSV41DeviceSampler:
    def __init__(self, mesh_device, mesh_config, ccl, users_per_row, J=32, levels=4, levels_k=None, kcand=32):
        self.md, self.mc, self.ccl = mesh_device, mesh_config, ccl
        rows, cols = tuple(mesh_device.shape)
        self.rows, self.cols, self.T, self.J, self.levels = rows, cols, users_per_row, J, levels
        self.levels_k = (
            levels if levels_k is None else levels_k
        )  # the top-k count search resolves ties of a flat row: one more (cheap, count only) level than the top-p mass search
        self.shard = VOCAB // cols
        rep = ttnn.ReplicateTensorToMesh(mesh_device)
        mk = lambda t, mapper=rep: ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        # fraction grid j / J, j = 0..J-1 : [1,1,1,J] -> broadcast against [T,1] columns as [1,T?]; kept as [1,1,J] rows below
        self.frac = mk((torch.arange(J, dtype=torch.float32) / J).reshape(1, J, 1, 1))
        self.col_ids = mk(torch.arange(cols, dtype=torch.float32).reshape(1, 1, 1, cols))
        # this device's own column index [1,1,1,1] (sharded over the mesh columns)
        own = torch.arange(cols, dtype=torch.float32).reshape(1, cols, 1, 1)
        self.own_col = mk(own, ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 1), mesh_shape=(rows, cols)))
        # lower-triangular [cols, cols]: inclusive prefix over the column totals
        self.tri = mk(torch.triu(torch.ones(cols, cols)).reshape(1, 1, cols, cols))  # [i, j] = 1 for i <= j
        # the vocabulary slice as [PR, 32] tiles (16160 = 505 * 32, padded to 512 rows with -inf logits): no 32-row padding of the T users
        self.PR = 512
        self.u32 = mk(torch.triu(torch.ones(32, 32)).reshape(1, 1, 32, 32))  # inclusive prefix inside a row of 32
        self.ustrict = mk(
            torch.triu(torch.ones(self.PR, self.PR), diagonal=1).reshape(1, 1, self.PR, self.PR)
        )  # exclusive prefix over rows
        self.ckc = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.kcand = (
            kcand  # candidates per mesh column of the top-k path (``forward_topk``): rows with 0 < top_k <= kcand
        )
        self.fast = False  # set by ``set_params``: every sampled row has 0 < top_k <= kcand -> ``forward_topk`` is exact and cheaper
        self.params = None
        self.stop = None  # stage name: forward returns right after it (cost bisection)
        self.prof = None  # list of (stage, seconds) when profiling eagerly

    def _mark(self, name):
        if self.prof is not None:
            import time

            ttnn.synchronize_device(self.md)
            now = time.perf_counter()
            self.prof.append((name, now - self._t))
            self._t = now

    # host side -------------------------------------------------------------------------------------------------------------------------
    def alloc_params(self):
        """Persistent device params [1,1,T,8] per device (invT, top_k, top_p, u, greedy, pad...), row sharded: written by ``set_params`` before a replay."""
        rows, cols = self.rows, self.cols
        self._pmap = ttnn.ShardTensor2dMesh(self.md, dims=(2, None), mesh_shape=(rows, cols))
        host = torch.zeros(1, 1, rows * self.T, 32)
        host[..., 0] = 1.0
        host[..., 2] = 1.0
        host[..., 4] = 1.0
        self.params = ttnn.from_torch(
            host,
            device=self.md,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self._pmap,
        )
        return self.params

    def set_params(self, invT, top_k, top_p, u, greedy):
        """Each argument a [B] tensor / list (B = rows * T, mesh-row order). top_k <= 0 / top_p >= 1: off. greedy: 1.0 = argmax row."""
        B = self.rows * self.T
        host = torch.zeros(1, 1, B, 32)
        host[0, 0, :, 0] = torch.as_tensor(invT, dtype=torch.float32)
        host[0, 0, :, 1] = torch.as_tensor(top_k, dtype=torch.float32).clamp(min=0)
        host[0, 0, :, 2] = torch.as_tensor(top_p, dtype=torch.float32)
        host[0, 0, :, 3] = torch.as_tensor(u, dtype=torch.float32)
        host[0, 0, :, 4] = torch.as_tensor(greedy, dtype=torch.float32)
        kk = torch.as_tensor(top_k, dtype=torch.float32)
        gr = torch.as_tensor(greedy, dtype=torch.float32) > 0
        self.fast = bool(((kk > 0) & (kk <= self.kcand) | gr).all())
        h = ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, mesh_mapper=self._pmap)
        ttnn.copy_host_to_device_tensor(h, self.params)

    # device side (inside the trace) -----------------------------------------------------------------------------------------------------
    def _allsum(self, x):
        """sum over the 8 mesh columns of x [1,1,T,n] -> [1,1,T,n] (all-gather + local sum)."""
        n = x.shape[-1]
        g = self.mc.allgather(x, self.ccl, axis=1, dim=3)  # [1,1,T,cols*n]
        g = ttnn.reshape(g, [1, self.T, self.cols, n])
        return ttnn.reshape(ttnn.sum(g, dim=2, keepdim=False), [1, 1, self.T, n])

    def _v4(self, t):  # [1,1,T,1] -> [T,1,1,1]
        return ttnn.reshape(t, [self.T, 1, 1, 1])

    def _search(self, s4, e4, lo, hi, target, use_mass, levels=None):
        """Largest threshold tau in [lo, hi) (grid refined ``levels`` times) with F(tau) >= target, F = count or mass of {s >= tau} over the whole vocabulary. lo / hi / target [1,1,T,1]."""
        T, J = self.T, self.J
        for _ in range(levels or self.levels):
            width = ttnn.subtract(hi, lo)  # [1,1,T,1]
            taus = ttnn.add(ttnn.multiply(self._v4(width), self.frac), self._v4(lo))  # [T,J,1,1]
            m = ttnn.ge(s4, taus)  # [T,J,PR,32] 1.0 / 0.0
            if use_mass:
                m = ttnn.multiply(m, e4)
            part = ttnn.reshape(ttnn.sum(m, dim=[2, 3], keepdim=True), [1, 1, T, J])
            tot = self._allsum(part)  # [1,1,T,J]
            ok = ttnn.ge(tot, target)  # target [1,1,T,1] broadcasts over J
            jstar = ttnn.maximum(
                ttnn.subtract(ttnn.sum(ok, dim=-1, keepdim=True), 1.0), 0.0
            )  # monotone: satisfied thresholds - 1 (F(lo) >= target)
            step = ttnn.multiply(width, 1.0 / J)
            lo = ttnn.add(lo, ttnn.multiply(step, jstar))
            hi = ttnn.add(lo, step)
        return lo

    def _mass_ge(self, s4, e4, tau):
        part = ttnn.reshape(
            ttnn.sum(ttnn.multiply(ttnn.ge(s4, self._v4(tau)), e4), dim=[2, 3], keepdim=True), [1, 1, self.T, 1]
        )
        return self._allsum(part)

    def run(self, logits, params, fast):
        """The sampler body of a variant: ``fast`` = ``forward_topk`` (rows with 0 < top_k <= kcand), else ``forward`` (full vocabulary)."""
        return self.forward_topk(logits, params) if fast else self.forward(logits, params)

    def forward_topk(self, logits, params):
        """Rows with 0 < top_k <= ``kcand`` (``self.fast``): top-k / top-p / draw over the CANDIDATES alone (the ``kcand`` largest logits of every mesh column, 8 * kcand per user) instead of the full vocabulary.
        The top-k set is exact (ties of the k-th value are kept), the nucleus is the exact one over it and the draw the exact vocabulary-order inverse CDF: no threshold search, no resolution band.
        Per column: top-``kcand`` of the logits rounded to bf16 (a fast op; its order is the order of the fp32 logits except among logits that round to the same bf16 value) with the rounding remainder
        gathered back, so that the candidate values are fp32 to 2**-17 relative. All-gather of (values, ids) over the mesh columns, then pairwise [Kt, Kt] comparisons on the candidates
        (rank / mass above a candidate for the nucleus, mass before it in vocabulary order for the draw).
        """
        T, cols, shard, Kc = self.T, self.cols, self.shard, self.kcand
        Kt = cols * Kc
        col = lambda i: ttnn.reshape(ttnn.slice(params, [0, 0, 0, i], [1, 1, T, i + 1]), [T, 1, 1, 1])
        invT, kk, pp, u, greedy = col(0), col(1), col(2), col(3), col(4)
        xb = ttnn.typecast(logits, ttnn.bfloat16)
        hi, li = ttnn.topk(xb, Kc, dim=-1)  # [1,1,T,Kc]
        lo = ttnn.typecast(ttnn.subtract(logits, ttnn.typecast(xb, ttnn.float32)), ttnn.bfloat16)
        lo_g = ttnn.gather(lo, -1, index=ttnn.typecast(li, ttnn.uint32))
        vals = ttnn.add(ttnn.typecast(hi, ttnn.float32), ttnn.typecast(lo_g, ttnn.float32))
        ids = ttnn.add(
            ttnn.typecast(li, ttnn.float32), ttnn.multiply(self.own_col, float(shard))
        )  # global vocabulary ids
        g = self.mc.allgather(ttnn.concat([vals, ids], dim=-1), self.ccl, axis=1, dim=3)  # [1,1,T,cols*2Kc]
        g = ttnn.reshape(ttnn.to_layout(g, ttnn.ROW_MAJOR_LAYOUT), [T, cols, 2, Kc])
        v_row = ttnn.to_layout(
            ttnn.reshape(ttnn.slice(g, [0, 0, 0, 0], [T, cols, 1, Kc]), [T, 1, 1, Kt]), ttnn.TILE_LAYOUT
        )
        i_row = ttnn.to_layout(
            ttnn.reshape(ttnn.slice(g, [0, 0, 1, 0], [T, cols, 2, Kc]), [T, 1, 1, Kt]), ttnn.TILE_LAYOUT
        )
        v_col, i_col = ttnn.transpose(v_row, 2, 3), ttnn.transpose(i_row, 2, 3)  # [T,1,Kt,1]
        M = ttnn.max(v_row, dim=-1, keepdim=True)  # [T,1,1,1]
        e_row = ttnn.exp(ttnn.multiply(ttnn.subtract(v_row, M), invT))
        e_col = ttnn.transpose(e_row, 2, 3)
        # exact top-k set and nucleus over the candidates: G[j, i] = v_i > v_j
        G = ttnn.gt(v_row, v_col)  # [T,1,Kt,Kt]
        rank = ttnn.sum(G, dim=-1, keepdim=True)  # [T,1,Kt,1] number of larger candidates
        above = ttnn.sum(ttnn.multiply(G, e_row), dim=-1, keepdim=True)  # mass of the larger candidates
        in_k = ttnn.lt(rank, kk)
        mass_k = ttnn.sum(ttnn.multiply(e_col, in_k), dim=2, keepdim=True)  # [T,1,1,1]
        no_p = ttnn.ge(pp, 1.0)
        keep = ttnn.multiply(in_k, ttnn.maximum(no_p, ttnn.lt(above, ttnn.multiply(pp, mass_k))))
        w_col = ttnn.multiply(e_col, keep)
        w_row = ttnn.transpose(w_col, 2, 3)
        # draw: inverse CDF in vocabulary order (mass before candidate j = sum of the kept candidates with a smaller id)
        before = ttnn.sum(ttnn.multiply(ttnn.lt(i_row, i_col), w_row), dim=-1, keepdim=True)  # [T,1,Kt,1]
        tgt = ttnn.multiply(u, ttnn.sum(w_col, dim=2, keepdim=True))
        cond = ttnn.multiply(ttnn.le(before, tgt), ttnn.gt(w_col, 0.0))
        tok_s = ttnn.subtract(ttnn.max(ttnn.multiply(cond, ttnn.add(i_col, 1.0)), dim=2, keepdim=True), 1.0)
        # greedy: the first (lowest id) maximum
        big = float(1 << 17)
        tok_g = ttnn.subtract(
            big, ttnn.max(ttnn.multiply(ttnn.ge(v_col, M), ttnn.subtract(big, i_col)), dim=2, keepdim=True)
        )
        tok = ttnn.add(ttnn.multiply(greedy, tok_g), ttnn.multiply(ttnn.subtract(1.0, greedy), tok_s))
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [T, 1])

    def forward(self, logits, params):
        """logits [1,1,T,shard] fp32 (this column's vocab slice) -> token ids [T,1] uint32 (identical on all devices)."""
        T, cols, shard, PR = self.T, self.cols, self.shard, self.PR
        col = lambda i: ttnn.slice(params, [0, 0, 0, i], [1, 1, T, i + 1])
        invT, kk, pp, u, greedy = col(0), col(1), col(2), col(3), col(4)
        if self.prof is not None:
            import time

            ttnn.synchronize_device(self.md)
            self._t = time.perf_counter()
        # global max / min of the row and the greedy token (first max wins, like torch)
        mx = ttnn.max(logits, dim=-1, keepdim=True)
        mn = ttnn.min(logits, dim=-1, keepdim=True)
        idx = ttnn.argmax(ttnn.to_layout(logits, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True)
        idxf = ttnn.typecast(ttnn.to_layout(idx, ttnn.TILE_LAYOUT), ttnn.float32)
        small = self.mc.allgather(ttnn.concat([mx, idxf, mn], dim=-1), self.ccl, axis=1, dim=3)  # [1,1,T,cols*3]
        small = ttnn.reshape(small, [1, T, cols, 3])
        mx_all = ttnn.reshape(ttnn.slice(small, [0, 0, 0, 0], [1, T, cols, 1]), [1, 1, T, cols])
        ix_all = ttnn.reshape(ttnn.slice(small, [0, 0, 0, 1], [1, T, cols, 2]), [1, 1, T, cols])
        mn_all = ttnn.reshape(ttnn.slice(small, [0, 0, 0, 2], [1, T, cols, 3]), [1, 1, T, cols])
        M = ttnn.max(mx_all, dim=-1, keepdim=True)
        mn_g = ttnn.min(mn_all, dim=-1, keepdim=True)
        best = ttnn.typecast(
            ttnn.to_layout(
                ttnn.argmax(ttnn.to_layout(mx_all, ttnn.ROW_MAJOR_LAYOUT), dim=-1, keepdim=True), ttnn.TILE_LAYOUT
            ),
            ttnn.float32,
        )
        tok_greedy = ttnn.add(
            ttnn.multiply(best, float(shard)),
            ttnn.sum(ttnn.multiply(ttnn.eq(best, self.col_ids), ix_all), dim=-1, keepdim=True),
        )
        self._mark("greedy_part")
        if self.stop == "greedy_part":
            return tok_greedy
        # the slice as [T,1,PR,32] tiles; padding entries have logit -1e30 (never kept, zero weight)
        pad = ttnn.pad(logits, [(0, 0), (0, 0), (0, 0), (0, PR * 32 - shard)], value=-1e30)
        x4 = ttnn.to_layout(ttnn.reshape(ttnn.to_layout(pad, ttnn.ROW_MAJOR_LAYOUT), [T, 1, PR, 32]), ttnn.TILE_LAYOUT)
        s4 = ttnn.multiply(ttnn.subtract(x4, self._v4(M)), self._v4(invT))  # <= 0
        e4 = ttnn.exp(s4)
        self._mark("scale_exp")
        if self.stop == "scale_exp":
            return e4
        lo0 = ttnn.multiply(ttnn.subtract(mn_g, M), invT)  # <= s of every token (scaled minimum)
        hi0 = ttnn.add(ttnn.multiply(lo0, 0.0), 1e-3)  # a hair above the max so that the grid covers s = 0
        off_k = ttnn.eq(kk, 0.0)  # top-k off -> k = vocabulary
        k_eff = ttnn.add(ttnn.multiply(ttnn.subtract(1.0, off_k), kk), ttnn.multiply(off_k, float(VOCAB)))
        tau_k = self._search(s4, e4, lo0, hi0, k_eff, use_mass=False, levels=self.levels_k)
        self._mark("search_k")
        if self.stop == "search_k":
            return tau_k
        mass_k = self._mass_ge(s4, e4, tau_k)
        tau_p = self._search(s4, e4, tau_k, hi0, ttnn.multiply(pp, mass_k), use_mass=True)
        no_p = ttnn.ge(
            pp, 1.0
        )  # top_p off: keep the whole top-k set (mass below fp32 resolution of the sum would be cut otherwise)
        tau_p = ttnn.add(ttnn.multiply(no_p, tau_k), ttnn.multiply(ttnn.subtract(1.0, no_p), tau_p))
        self._mark("search_p")
        if self.stop == "search_p":
            return tau_p
        w = ttnn.multiply(e4, ttnn.ge(s4, self._v4(tau_p)))
        # inclusive prefix over the slice in vocabulary order: within rows of 32 (matmul with a triangular matrix), then the exclusive prefix over the row totals
        P = ttnn.matmul(w, self.u32, compute_kernel_config=self.ckc)  # [T,1,PR,32]
        R = ttnn.slice(P, [0, 0, 0, 31], [T, 1, PR, 32])  # [T,1,PR,1] row totals
        offs = ttnn.transpose(
            ttnn.matmul(ttnn.transpose(R, 2, 3), self.ustrict, compute_kernel_config=self.ckc), 2, 3
        )  # [T,1,PR,1]
        cs = ttnn.add(P, offs)
        self._mark("prefix")
        if self.stop == "prefix":
            return cs
        tot_c = ttnn.reshape(ttnn.slice(cs, [0, 0, PR - 1, 31], [T, 1, PR, 32]), [1, 1, T, 1])
        tot_all = self.mc.allgather(tot_c, self.ccl, axis=1, dim=3)  # [1,1,T,cols]
        total = ttnn.sum(tot_all, dim=-1, keepdim=True)
        incl = ttnn.matmul(
            tot_all, self.tri, compute_kernel_config=self.ckc
        )  # inclusive prefix over columns [1,1,T,cols]
        tgt = ttnn.multiply(u, total)
        colsel = ttnn.minimum(
            ttnn.sum(ttnn.le(incl, tgt), dim=-1, keepdim=True), float(cols - 1)
        )  # selected column [1,1,T,1]
        excl_sel = ttnn.sum(
            ttnn.multiply(ttnn.eq(self.col_ids, colsel), ttnn.subtract(incl, tot_all)), dim=-1, keepdim=True
        )
        local_t = ttnn.subtract(tgt, excl_sel)  # same on every device
        cnt = ttnn.reshape(
            ttnn.sum(ttnn.le(cs, self._v4(local_t)), dim=[2, 3], keepdim=True), [1, 1, T, 1]
        )  # first index with cs > local_t
        cnt = ttnn.minimum(cnt, float(shard - 1))
        mine = ttnn.add(ttnn.multiply(self.own_col, float(shard)), cnt)  # this device's candidate token
        cand = self.mc.allgather(mine, self.ccl, axis=1, dim=3)  # [1,1,T,cols]
        tok_s = ttnn.sum(ttnn.multiply(ttnn.eq(self.col_ids, colsel), cand), dim=-1, keepdim=True)
        tok = ttnn.add(ttnn.multiply(greedy, tok_greedy), ttnn.multiply(ttnn.subtract(1.0, greedy), tok_s))
        self._mark("draw_rest")
        return ttnn.reshape(ttnn.to_layout(ttnn.typecast(tok, ttnn.uint32), ttnn.ROW_MAJOR_LAYOUT), [T, 1])


class SamplerTraces:
    """The two traces of one sampler body (full-vocabulary and top-k variant, see ``DSV41DeviceSampler.forward_topk``): ``replay`` runs the one the last ``set_params`` selected
    (``smp.fast``: every sampled row has 0 < top_k <= kcand)."""

    def __init__(self, smp):
        self.smp, self.tids = smp, {}

    def capture(self, body):
        """``body(fast)`` runs the sampler variant and whatever follows it (eager ops; captured into one trace per variant)."""
        md = self.smp.md
        for fast in (False, True):
            self.tids[fast] = ttnn.begin_trace_capture(md, cq_id=0)
            body(fast)
            ttnn.end_trace_capture(md, self.tids[fast], cq_id=0)

    def replay(self):
        ttnn.execute_trace(self.smp.md, self.tids[self.smp.fast], cq_id=0, blocking=False)

    def release(self):
        for tid in self.tids.values():
            ttnn.release_trace(self.smp.md, tid)
        self.tids = {}


def sampling_rows(temperature, top_k, top_p, gens, B):
    """Host side of one step: per-row (invT, top_k, top_p, u, greedy) [B] for ``DSV41DeviceSampler.set_params``. ``temperature`` / ``top_k`` / ``top_p``: a scalar or a list (padded with the
    last value / greedy); ``gens``: a list of per-row ``torch.Generator`` (a seeded request keeps its own stream) or one generator shared by all rows. Greedy rows: temperature 0 or top_k 1.
    """

    def vec(x, default):
        x = list(x) if isinstance(x, (list, tuple)) else [x]
        x = [default if v is None else v for v in x]
        return (x + [x[-1] if x else default] * B)[:B]

    t = torch.tensor(vec(temperature, 0.0), dtype=torch.float32)
    k = torch.tensor(vec(top_k, 0), dtype=torch.float32)
    p = torch.tensor(vec(top_p, 1.0), dtype=torch.float32)
    greedy = ((t <= 0) | (k == 1)).float()
    invT = torch.where(t > 0, 1.0 / t.clamp(min=1e-6), torch.ones(B))
    if isinstance(gens, (list, tuple)):
        # one batched draw per distinct generator (a seeded request keeps its own stream, the unseeded rows share one): no per-row python loop of generator calls
        u = torch.empty(B)
        groups = {}
        for i in range(B):
            g = gens[i % len(gens)]
            groups.setdefault(id(g), (g, []))[1].append(i)
        for g, idx in groups.values():
            u[torch.tensor(idx)] = torch.rand(len(idx), generator=g)
    else:
        u = torch.rand(B, generator=gens)
    return invT, k, p, u.clamp(max=1 - 1e-7), greedy


# ---- CPU model of the device algorithm (tests/test_device_sampler_cpu.py; the device test compares the traced sampler with the exact reference) -------------------------------------------
def emulate_keep_weights(logits, temperature, top_k=0, top_p=1.0, J=16, levels=3, levels_k=None):
    """Weights e * keep [V] of one row, computed exactly like ``DSV41DeviceSampler.forward`` (fp32, k-ary threshold searches over the full vocabulary)."""
    l = logits.float()
    s = (l - l.max()) * (1.0 / float(temperature))
    e = torch.exp(s)

    def search(lo, hi, target, use_mass, levels=levels):
        for _ in range(levels):
            width = hi - lo
            taus = lo + width * (torch.arange(J, dtype=torch.float32) / J)
            m = (s[None, :] >= taus[:, None]).float()
            f = (m * e[None, :]).sum(1) if use_mass else m.sum(1)
            jstar = max(int((f >= target).sum()) - 1, 0)
            step = width / J
            lo = lo + step * jstar
            hi = lo + step
        return lo

    lo0, hi0 = s.min(), torch.tensor(1e-3)
    k_eff = float(top_k) if top_k and top_k > 0 else float(l.numel())
    tau_k = search(lo0, hi0, k_eff, False, levels if levels_k is None else levels_k)
    mass_k = (e * (s >= tau_k)).sum()
    tau_p = tau_k if float(top_p) >= 1.0 else search(tau_k, hi0, float(top_p) * mass_k, True)
    return e * (s >= tau_p)


def emulate_draw(w, u):
    """Vocabulary-order inverse CDF of the weights ``w`` [V] at ``u`` (scalar or tensor of uniforms): the first index whose cumulative weight exceeds u * total."""
    cs = w.double().cumsum(0)
    idx = torch.searchsorted(cs, torch.as_tensor(u, dtype=torch.float64) * cs[-1], right=True)
    return idx.clamp(max=w.numel() - 1)


def emulate_topk_weights(logits, temperature, top_k, top_p=1.0, kcand=32, cols=8):
    """Weights e * keep [V] of one row, computed like ``DSV41DeviceSampler.forward_topk`` (candidates = the kcand largest logits of each of ``cols`` vocabulary slices; pairwise rank / mass-above tests)."""
    l = logits.float()
    V = l.numel()
    shard = V // cols
    ids = torch.cat([torch.topk(l[c * shard : (c + 1) * shard], kcand).indices + c * shard for c in range(cols)])
    v = l[ids]
    e = torch.exp((v - v.max()) * (1.0 / float(temperature)))
    G = (v[None, :] > v[:, None]).float()  # G[j, i] = v_i > v_j
    rank = G.sum(1)
    above = (G * e[None, :]).sum(1)
    in_k = rank < top_k
    mass_k = (e * in_k).sum()
    keep = in_k & ((above < top_p * mass_k) | (top_p >= 1.0))
    out = torch.zeros(V)
    out[ids] = e * keep
    return out

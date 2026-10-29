# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Repetition / presence / frequency penalties inside the fused MTP spec step (phase 3 part 2, design PEN-1;
work/research_1026/80 section 4 and 84).

State per device, per ENGINE row r = u * T + j (u = verify row, j = column, T = K + 1; the T rows of one user hold the same state):
  counts fp32 [1, 1, 32, V/TP]    how often each token of the vocabulary shard is in the request's OUTPUT (committed tokens only)
  mask   bf16 [1, 1, 32, V/TP]    1.0 where the token is in the request's PROMPT (only the repetition penalty reads it)
  rep, 1/rep, freq, pres fp32 [1, 1, 32, 1]   the per-row parameters of the step (neutral rows: 1, 1, 0, 0)
The state is indexed by engine row, not by runner slot: under the host slot map (QWEN36_SPEC_SLOTMAP=1) a request keeps its engine
row when the runner remaps slots, and a row bucket b < B only reads the rows 0 .. b*T-1, so neither event moves any state.

Two traces use it:
  apply_body   (the "pen trace", per verify bucket, replayed between the main trace and the topk trace only when a live row has a
               non-neutral penalty): adds the draft history of column j (drafts 1..j of the row's verify tokens, read from the
               verify token buffer) to the counts, applies repetition, frequency and presence in fp32 in the reference order and
               writes the bf16 result IN PLACE into the bucket's shard logits. Neutral rows are unchanged bit for bit
               (x * 1.0 - 0 * c - 0 * (c > 0), no rounding because x is already a bf16 value).
  update_body  (the "pen update trace", one trace, replayed after the sampled accept trace): counts[u] += one-hot of the committed
               ids of user u (result_out row 0; -1 past the count adds nothing). A user whose overflow word (result_out row 2) is
               not 0 is skipped: its device result is meaningless and the host redo walk commits other tokens (the engine adds those
               with add_committed_host).
Exactness (accepted by the user on 2026-10-28): the fp32 chain rounds to bf16 once, so a penalized token can differ from the host
reference by about half a bf16 ulp (measured max 0.567 ulp from fp32 cancellation, work/research_1026/81); tokens with no penalty
are bit-identical.

The bodies take ``ops`` (default: the ttnn module) so a host test can run the same op sequence on a torch emulation.
"""

from dataclasses import dataclass, field

import torch

REP_NEUTRAL, FREQ_NEUTRAL, PRES_NEUTRAL = 1.0, 0.0, 0.0
PEN_ROWS = 32  # one tile row of engine rows (B * T <= 32)


# --------------------------------------------------------------------------- #
# Host helpers and references (pure torch)
# --------------------------------------------------------------------------- #
def penalties_neutral(rep, freq, pres):
    """[B] bool: True where a row's penalties change nothing (repetition 1, frequency 0, presence 0, exactly)."""
    rep, freq, pres = (torch.as_tensor(x) for x in (rep, freq, pres))
    return (rep == REP_NEUTRAL) & (freq == FREQ_NEUTRAL) & (pres == PRES_NEUTRAL)


def counts_from_ids(ids, vocab):
    """fp32 [V]: how many times each token id is in ``ids``; ids outside [0, V) (the plugin's padding) are ignored."""
    t = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
    t = t[(t >= 0) & (t < vocab)]
    return torch.bincount(t, minlength=vocab).to(torch.float32)


def mask_from_ids(ids, vocab):
    """fp32 [V] 0/1: 1.0 where the token id is in ``ids`` (ids outside [0, V) ignored)."""
    return (counts_from_ids(ids, vocab) > 0).to(torch.float32)


def column_history_counts(out_counts, drafts, num_valid, T, *, device_rule=False):
    """fp32 [T, V]: the output counts column j of a verify row scores under. Plugin rule (``device_rule=False``): committed output +
    drafts[:min(j, num_valid)]. Device rule: drafts[:j] (the device does not know num_valid; a column past num_valid is never
    examined, so the two rules agree on every column the walk reads)."""
    V = out_counts.numel()
    out = out_counts.reshape(1, V).repeat(T, 1).clone()
    dr = [int(d) for d in torch.as_tensor(drafts).reshape(-1).tolist()]
    for j in range(T):
        n = j if device_rule else min(j, int(num_valid))
        for d in dr[:n]:
            if 0 <= d < V:
                out[j, d] += 1.0
    return out


def apply_penalties_exact(x, counts, prompt_mask, rep, freq, pres):
    """The plugin / vLLM penalty arithmetic in fp32, no rounding: x [R, V], counts [R, V] fp32 (history incl. drafts), prompt_mask
    [R, V] (any dtype, > 0 = in the prompt), rep / freq / pres [R]. Order: repetition (x > 0 ? x / rep : x * rep on tokens seen in
    the prompt or the history), then frequency (x - freq * count), then presence (x - pres * (count > 0)). Returns fp32 [R, V].
    """
    x = torch.as_tensor(x).to(torch.float32)
    c = torch.as_tensor(counts).to(torch.float32)
    rep, freq, pres = (torch.as_tensor(v).to(torch.float32).reshape(-1, 1) for v in (rep, freq, pres))
    seen = (c > 0) | (torch.as_tensor(prompt_mask) > 0)
    scaled = torch.where(x > 0, x / rep, x * rep)
    x = torch.where(seen, scaled, x)
    x = x - freq * c
    return x - pres * (c > 0).to(torch.float32)


def apply_penalties_device_model(x, counts, prompt_mask, rep, freq, pres):
    """What the pen trace computes, on the host: the same op order in fp32 with the repetition division as a multiplication by 1 / rep
    (rep and 1 / rep are separate fp32 inputs), rounded once to bf16. x [R, V] (bf16 values), returns bf16 [R, V]. The audit and the
    exactness tests model the device with this; the plugin reference (apply_penalties_exact) differs from it by the rounding.
    """
    xf = torch.as_tensor(x).to(torch.float32)
    c = torch.as_tensor(counts).to(torch.float32)
    rep32 = torch.as_tensor(rep).to(torch.float32).reshape(-1, 1)
    inv = (1.0 / rep32.double()).to(torch.float32)  # the engine stages 1 / rep computed in float64, rounded to fp32
    freq, pres = (torch.as_tensor(v).to(torch.float32).reshape(-1, 1) for v in (freq, pres))
    cpos = (c > 0).to(torch.float32)
    seen = (cpos + torch.as_tensor(prompt_mask).to(torch.float32)) > 0
    scale = torch.where(xf > 0, inv.expand_as(xf), rep32.expand_as(xf))
    scale = torch.where(seen, scale, torch.ones_like(scale))
    out = xf * scale
    out = out - c * freq
    out = out - cpos * pres
    return out.to(torch.bfloat16)


def update_reference(counts, committed, overflow):
    """The update trace on the host: counts [32, V] fp32 (engine rows), committed [B, T] (-1 past the count), overflow [B]. Returns
    the new counts (a copy): every row of user u gains one for each committed id of u unless overflow[u] != 0."""
    counts = counts.clone()
    B, T = committed.shape
    for u in range(B):
        if int(overflow[u]) != 0:
            continue
        for j in range(T):
            t = int(committed[u, j])
            if 0 <= t < counts.shape[1]:
                counts[u * T : (u + 1) * T, t] += 1.0
    return counts


@dataclass
class PenaltyStep:
    """The penalties of one step in ENGINE rows (see MTPSpecEngine._engine_sampling): rep / freq / pres fp32 [B], ``live`` [B] bool (a
    non-neutral row), ``rebuild`` {engine row: (prompt ids int64 [P], output ids int64 [N])} (state to write before the step).
    """

    rep: torch.Tensor
    freq: torch.Tensor
    pres: torch.Tensor
    live: torch.Tensor
    rebuild: dict = field(default_factory=dict)

    @property
    def any_live(self):
        return bool(self.live.any())


def parse_penalties(pen, B, to_rows, *, active=None):
    """The plugin's ``spec_sampling["penalties"]`` (runner slots) as a PenaltyStep over engine rows. ``pen``: {"repetition", "frequency",
    "presence": fp32 [B], optional "rebuild": {runner slot: (prompt ids, output ids)}}. ``to_rows(*xs)`` permutes slot-ordered tensors
    to engine rows; the rebuild keys are mapped with the same permutation (slot s -> the engine row that holds it)."""
    rep, freq, pres = to_rows(
        *(torch.as_tensor(pen[k]).to(torch.float32).reshape(B) for k in ("repetition", "frequency", "presence"))
    )
    rep, freq, pres = rep.clone(), freq.clone(), pres.clone()
    live = ~penalties_neutral(rep, freq, pres)
    if active is not None:  # a padding row owns no request: its parameters are ignored
        act = torch.as_tensor(active).bool().reshape(B)
        rep[~act], freq[~act], pres[~act] = REP_NEUTRAL, FREQ_NEUTRAL, PRES_NEUTRAL
        live = live & act
    if bool((rep <= 0).any()) or not bool(torch.isfinite(torch.stack([rep, freq, pres])).all()):
        raise ValueError(
            f"penalties must be finite with repetition > 0: repetition {rep.tolist()} frequency {freq.tolist()} presence {pres.tolist()}"
        )
    rebuild = {}
    marker = torch.arange(B, dtype=torch.int64)
    (slot_of_row,) = to_rows(marker)  # engine row u holds runner slot slot_of_row[u]
    slot_to_row = {int(s): u for u, s in enumerate(slot_of_row.tolist())}
    for slot, (prompt, out) in (pen.get("rebuild") or {}).items():
        slot = int(slot)
        if slot not in slot_to_row:
            raise ValueError(f"penalties rebuild for slot {slot} outside range({B})")
        rebuild[slot_to_row[slot]] = (
            torch.as_tensor(prompt, dtype=torch.int64).reshape(-1),
            torch.as_tensor(out, dtype=torch.int64).reshape(-1),
        )
    return PenaltyStep(rep, freq, pres, live, rebuild)


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
class PenaltyDevice:
    """The device state and bodies of PEN-1 (module docstring). ``allocate`` creates every persistent buffer (before any trace capture);
    ``apply_body`` / ``update_body`` are trace capturable (no host round trip, no data-dependent shape); ``stage_params``,
    ``rebuild`` and ``add_committed_host`` are the eager host-driven writes; ``read_rows`` the exact readback (fallback walks, tests).
    """

    @classmethod
    def allocate(cls, mesh, B, K, per_shard, vocab, tok_in, *, ops=None):
        """``tok_in``: the model's verify token buffer (uint32 ROW_MAJOR [1, B*T]) the pen trace reads the drafts from."""
        if ops is None:
            import ttnn as ops
        T, R = K + 1, B * (K + 1)
        if R > PEN_ROWS:
            raise ValueError(f"penalties need B * T <= {PEN_ROWS} engine rows, got B={B} T={T}")
        if vocab % per_shard:
            raise ValueError(f"vocab {vocab} is not a multiple of the shard width {per_shard}")
        if vocab >= (1 << 24):
            raise ValueError(f"token ids must stay exact in fp32: vocab {vocab} >= 2**24")
        self = cls()
        self.ops, self.mesh, self.B, self.K, self.T, self.R = ops, mesh, B, K, T, R
        self.per_shard, self.vocab, self.tok_in = per_shard, vocab, tok_in
        self.TP = vocab // per_shard
        self.result_in = None
        f32, bf16, TL = ops.float32, ops.bfloat16, ops.TILE_LAYOUT
        rep = ops.ReplicateTensorToMesh(mesh)
        shard = ops.ShardTensorToMesh(mesh, dim=3)

        def dev(t, dt, mapper=rep):
            return ops.from_torch(
                t, dtype=dt, layout=TL, device=mesh, mesh_mapper=mapper, memory_config=ops.DRAM_MEMORY_CONFIG
            )

        col = lambda v: v.reshape(1, 1, PEN_ROWS, 1).to(torch.float32)  # noqa: E731
        self._dev = dev
        self.iota = dev(torch.arange(vocab, dtype=torch.float32).reshape(1, 1, 1, vocab), f32, shard)
        self.ones_t = dev(torch.ones(1, T), f32)
        jr = torch.arange(PEN_ROWS) % T
        self.m_keep = [dev(col((jr >= i).to(torch.float32)), f32) for i in range(1, K + 1)]  # 1 where column j >= i
        self.m_neg = [dev(col((jr >= i).to(torch.float32) - 1.0), f32) for i in range(1, K + 1)]  # 0 or -1
        self.counts = dev(torch.zeros(1, 1, PEN_ROWS, vocab), f32, shard)
        self.mask = dev(torch.zeros(1, 1, PEN_ROWS, vocab), bf16, shard)
        self.rep, self.inv, self.freq, self.pres = (
            dev(col(torch.full((PEN_ROWS,), v)), f32) for v in (1.0, 1.0, 0.0, 0.0)
        )
        self.stage_counts = dev(torch.zeros(1, 1, PEN_ROWS, vocab), f32, shard)
        self.stage_mask = dev(torch.zeros(1, 1, PEN_ROWS, vocab), bf16, shard)
        self.sel_f, self.keep_f = dev(col(torch.zeros(PEN_ROWS)), f32), dev(col(torch.ones(PEN_ROWS)), f32)
        self.sel_b, self.keep_b = dev(col(torch.zeros(PEN_ROWS)), bf16), dev(col(torch.ones(PEN_ROWS)), bf16)
        self.add_ids = [dev(col(torch.full((PEN_ROWS,), -1.0)), f32) for _ in range(T)]
        self._par_host = None
        self.stats = {"param_stages": 0, "rebuilds": 0, "rebuild_rows": 0, "repairs": 0}
        return self

    def tensors(self):
        """Every persistent device tensor (shutdown frees them)."""
        out = [
            self.iota,
            self.ones_t,
            *self.m_keep,
            *self.m_neg,
            self.counts,
            self.mask,
            self.rep,
            self.inv,
            self.freq,
            self.pres,
        ]
        return out + [
            self.stage_counts,
            self.stage_mask,
            self.sel_f,
            self.keep_f,
            self.sel_b,
            self.keep_b,
            *self.add_ids,
        ]

    def bind_result(self, result_in):
        """The sampled accept's result buffer (int32 ROW_MAJOR [3, W]) the update trace reads."""
        self.result_in = result_in

    # ---- bodies --------------------------------------------------------------
    def _draft_ids(self, tokf, R, i):
        """fp32 [1,1,32,1]: for engine row r = u*T + j the id of draft i of user u (verify token column i) where j >= i, else -1."""
        ops, T = self.ops, self.T
        nu = R // T
        col = ops.slice(tokf, [0, i], [nu, i + 1])  # [nu, 1]
        idr = ops.reshape(ops.multiply(col, self.ones_t), (R, 1))  # [R, 1]: the user's draft on each of its rows
        if R < PEN_ROWS:
            idr = ops.pad(idr, [(0, PEN_ROWS - R), (0, 0)], value=-1.0)
        idr = ops.reshape(idr, (1, 1, PEN_ROWS, 1))
        return ops.add(ops.multiply(idr, self.m_keep[i - 1]), self.m_neg[i - 1])

    def apply_body(self, x):
        """The pen trace body over the bucket's shard logits ``x`` (bf16 TILE [1,1,R,V/TP], R = b*T rows, written in place)."""
        ops, T, K = self.ops, self.T, self.K
        TL, f32, bf16 = ops.TILE_LAYOUT, ops.float32, ops.bfloat16
        fl = lambda t: ops.typecast(t, f32)  # noqa: E731
        R = int(x.shape[2])
        counts = self.counts
        if K > 0:  # the draft history of column j: one-hot of drafts 1..j of the row's verify tokens
            nu = R // T
            tok = self.tok_in if int(self.tok_in.shape[-1]) == R else ops.slice(self.tok_in, [0, 0], [1, R])
            tokf = fl(ops.to_layout(ops.reshape(tok, (nu, T)), TL))  # [nu, T]
            for i in range(1, K + 1):
                ids = self._draft_ids(tokf, R, i)
                oh = ops.eq(self.iota, ids)  # [1,1,32,V] one-hot, none on a -1 id
                nxt = ops.add(counts, oh)
                ops.deallocate(oh)
                ops.deallocate(ids)
                if counts is not self.counts:
                    ops.deallocate(counts)
                counts = nxt
        xp = ops.pad(x, [(0, 0), (0, 0), (0, PEN_ROWS - R), (0, 0)], value=0.0) if R < PEN_ROWS else x
        xf = fl(xp)
        cpos = ops.gt(counts, 0.0)
        mpos = ops.typecast(self.mask, f32)
        anyp = ops.add(cpos, mpos)
        ops.deallocate(mpos)
        anyb = ops.gt(anyp, 0.0)
        ops.deallocate(anyp)
        gtz = ops.gt(xf, 0.0)
        scale = ops.where(gtz, self.inv, self.rep)
        ops.deallocate(gtz)
        scale2 = ops.where(anyb, scale, 1.0)
        ops.deallocate(scale)
        ops.deallocate(anyb)
        x2 = ops.multiply(xf, scale2)
        ops.deallocate(xf)
        ops.deallocate(scale2)
        fc = ops.multiply(counts, self.freq)
        x3 = ops.subtract(x2, fc)
        ops.deallocate(x2)
        ops.deallocate(fc)
        pc = ops.multiply(cpos, self.pres)
        ops.deallocate(cpos)
        x4 = ops.subtract(x3, pc)
        ops.deallocate(x3)
        ops.deallocate(pc)
        out = ops.typecast(x4, bf16)
        ops.deallocate(x4)
        if R < PEN_ROWS:
            out = ops.slice(out, [0, 0, 0, 0], [1, 1, R, int(x.shape[3])])
        ops.copy(out, x)
        ops.deallocate(out)
        if counts is not self.counts:
            ops.deallocate(counts)

    def _add_ids_body(self, ids_by_column):
        """counts += one-hot of each [1,1,32,1] id tensor in ``ids_by_column`` (in place; -1 adds nothing)."""
        ops = self.ops
        for ids in ids_by_column:
            oh = ops.eq(self.iota, ids)
            ops.add_(self.counts, oh)
            ops.deallocate(oh)

    def update_body(self):
        """The update trace body: counts[u rows] += one-hot of the committed ids of user u in ``result_in`` (row 0, user-major
        r = u*T + j, -1 past the count), skipping a user whose overflow word (row 2) is not 0."""
        ops, B, T, R = self.ops, self.B, self.T, self.R
        assert self.result_in is not None, "bind_result first"
        TL, f32 = ops.TILE_LAYOUT, ops.float32
        fl = lambda t: ops.typecast(t, f32)  # noqa: E731
        comm = fl(ops.to_layout(ops.reshape(ops.slice(self.result_in, [0, 0], [1, R]), (B, T)), TL))  # [B, T]
        rest = ops.transpose(fl(ops.to_layout(self.result_in, TL)), 0, 1)  # [W, 3]
        ovf = ops.slice(rest, [0, 2], [B, 3])  # [B, 1]
        ok = fl(ops.le(ovf, 0.0))  # 1.0 for a user whose device result is valid
        okm1 = ops.subtract(ok, 1.0)
        ids = []
        for j in range(T):
            cj = ops.slice(comm, [0, j], [B, j + 1])  # [B, 1]
            cjm = ops.add(ops.multiply(cj, ok), okm1)  # committed id, or -1 for an overflow user / past the count
            idr = ops.reshape(ops.multiply(cjm, self.ones_t), (R, 1))
            if R < PEN_ROWS:
                idr = ops.pad(idr, [(0, PEN_ROWS - R), (0, 0)], value=-1.0)
            ids.append(ops.reshape(idr, (1, 1, PEN_ROWS, 1)))
        self._add_ids_body(ids)
        for t in ids:
            ops.deallocate(t)

    # ---- eager host-driven writes ------------------------------------------------
    def _host_copy(self, t, dev_t, dtype, mapper=None):
        ops = self.ops
        mapper = mapper if mapper is not None else ops.ReplicateTensorToMesh(self.mesh)
        h = ops.from_torch(t, dtype=dtype, layout=ops.TILE_LAYOUT, device=None, mesh_mapper=mapper)
        ops.copy_host_to_device_tensor(h, dev_t)

    def stage_params(self, step):
        """Write the step's rep / 1/rep / freq / pres (per engine row, each user's value on its T rows) into the persistent parameter
        tensors. Skipped when the same values are already there."""
        T, B = self.T, self.B
        rows = torch.arange(PEN_ROWS) // T
        live = torch.arange(PEN_ROWS) < self.R

        def per_row(v, neutral):
            full = torch.full((PEN_ROWS,), neutral, dtype=torch.float64)
            full[live] = torch.as_tensor(v).to(torch.float64).reshape(B)[rows[live]]
            return full

        rep, freq, pres = per_row(step.rep, 1.0), per_row(step.freq, 0.0), per_row(step.pres, 0.0)
        vals = torch.stack([rep, 1.0 / rep, freq, pres]).to(torch.float32)  # 1 / rep from float64
        if self._par_host is not None and torch.equal(self._par_host, vals):
            return False
        for v, dev_t in zip(vals, (self.rep, self.inv, self.freq, self.pres)):
            self._host_copy(v.reshape(1, 1, PEN_ROWS, 1), dev_t, self.ops.float32)
        self._par_host = vals.clone()
        self.stats["param_stages"] += 1
        return True

    def rebuild(self, entries):
        """Write the state of the engine rows in ``entries`` {row u: (prompt ids, output ids)} from host token lists and leave every
        other row untouched: the dense host tensors go to the staging buffers, then counts = counts * keep + stage * sel on the
        device (queued behind every step already enqueued). Cost: two dense uploads of [32, V] (host build ~ tens of ms).
        """
        if not entries:
            return
        ops, V, T = self.ops, self.vocab, self.T
        cnt = torch.zeros(PEN_ROWS, V, dtype=torch.float32)
        msk = torch.zeros(PEN_ROWS, V, dtype=torch.float32)
        sel = torch.zeros(PEN_ROWS, dtype=torch.float32)
        for u, (prompt, out) in sorted(entries.items()):
            if not 0 <= int(u) < self.B:
                raise ValueError(f"penalty rebuild for engine row {u} outside range({self.B})")
            lo, hi = u * T, (u + 1) * T
            cnt[lo:hi] = counts_from_ids(out, V)
            msk[lo:hi] = mask_from_ids(prompt, V)
            sel[lo:hi] = 1.0
        shard = ops.ShardTensorToMesh(self.mesh, dim=3)
        self._host_copy(cnt.reshape(1, 1, PEN_ROWS, V), self.stage_counts, ops.float32, shard)
        self._host_copy(msk.reshape(1, 1, PEN_ROWS, V), self.stage_mask, ops.bfloat16, shard)
        col = lambda v: v.reshape(1, 1, PEN_ROWS, 1)  # noqa: E731
        self._host_copy(col(sel), self.sel_f, ops.float32)
        self._host_copy(col(1.0 - sel), self.keep_f, ops.float32)
        self._host_copy(col(sel), self.sel_b, ops.bfloat16)
        self._host_copy(col(1.0 - sel), self.keep_b, ops.bfloat16)
        self._merge()
        self.stats["rebuilds"] += 1
        self.stats["rebuild_rows"] += len(entries)

    def _merge(self):
        """counts = counts * keep + stage * sel, mask likewise (eager, exact: 0/1 weights of finite values)."""
        ops = self.ops
        for state, stage, sel, keep in (
            (self.counts, self.stage_counts, self.sel_f, self.keep_f),
            (self.mask, self.stage_mask, self.sel_b, self.keep_b),
        ):
            a, b = ops.multiply(state, keep), ops.multiply(stage, sel)
            new = ops.add(a, b)
            ops.copy(new, state)
            for t in (a, b, new):
                ops.deallocate(t)

    def add_committed_host(self, committed, rows):
        """Eager: counts[u] += one-hot of the committed ids of each user u in ``rows`` (``committed`` [B, T], -1 past the count). The
        engine calls it for a step the host walked (fallback, overflow redo): the update trace did not run for those users.
        """
        rows = sorted(int(u) for u in rows)
        if not rows:
            return
        T = self.T
        comm = torch.as_tensor(committed, dtype=torch.int64).reshape(self.B, T)
        for j in range(T):
            ids = torch.full((PEN_ROWS,), -1.0, dtype=torch.float32)
            for u in rows:
                ids[u * T : (u + 1) * T] = float(comm[u, j])
            self._host_copy(ids.reshape(1, 1, PEN_ROWS, 1), self.add_ids[j], self.ops.float32)
        self._add_ids_body(self.add_ids)
        self.stats["repairs"] += len(rows)

    def read_rows(self, rows):
        """(counts fp32 [n, V], prompt mask bool [n, V]) of the verify rows ``rows`` (the first of each user's T rows), after a synchronize."""
        ops = self.ops
        ops.synchronize_device(self.mesh)
        comp = ops.ConcatMeshToTensor(self.mesh, dim=3)
        c = ops.to_torch(self.counts, mesh_composer=comp).reshape(PEN_ROWS, self.vocab)
        m = ops.to_torch(self.mask, mesh_composer=comp).reshape(PEN_ROWS, self.vocab)
        idx = [int(u) * self.T for u in rows]
        return c[idx].to(torch.float32), m[idx].to(torch.float32) > 0

    def warm(self, x_by_rows):
        """Compile every program before capture forbids cache misses: the apply body over each stand-in shard-logits tensor in
        ``x_by_rows`` (one per row bucket and the B-row shape), the update body, the merge and the host-id add. The state is left zero
        and neutral (stand-in logits are written in place and discarded)."""
        for x in x_by_rows:
            self.apply_body(x)
        self.update_body()
        self._merge()
        self._add_ids_body(self.add_ids)
        self.ops.synchronize_device(self.mesh)
        self.reset_state()

    def reset_state(self):
        """Zero the state and neutral parameters in place (after warm-up)."""
        ops = self.ops
        z = lambda shape: torch.zeros(*shape)  # noqa: E731
        shard = ops.ShardTensorToMesh(self.mesh, dim=3)
        self._host_copy(z((1, 1, PEN_ROWS, self.vocab)), self.counts, ops.float32, shard)
        self._host_copy(z((1, 1, PEN_ROWS, self.vocab)), self.mask, ops.bfloat16, shard)
        self._host_copy(z((1, 1, PEN_ROWS, self.vocab)), self.stage_counts, ops.float32, shard)
        self._host_copy(z((1, 1, PEN_ROWS, self.vocab)), self.stage_mask, ops.bfloat16, shard)
        for t, v in ((self.rep, 1.0), (self.inv, 1.0), (self.freq, 0.0), (self.pres, 0.0)):
            self._host_copy(torch.full((1, 1, PEN_ROWS, 1), v), t, ops.float32)
        for t in self.add_ids:
            self._host_copy(torch.full((1, 1, PEN_ROWS, 1), -1.0), t, ops.float32)
        self._par_host = None

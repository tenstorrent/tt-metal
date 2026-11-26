# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Presence, frequency and repetition penalties on the MTP verify (vllm-tt-plugin ``supports_spec_device_penalties``).

On a device-sampled step the plugin sends a verify whose rows may carry penalties, with each row's prompt and committed
output (``prompt_tokens``, ``output_tokens``). Column t of row u is scored after the row's history: the committed
output plus the row's first t drafts. The penalties change the logits of column t as vLLM's ``apply_penalties`` does,
in float32 and in its order:

  1. repetition: a token in the prompt or the history is multiplied by 1 / repetition (logit > 0) or by repetition;
  2. frequency: minus frequency x (the token's count in the history);
  3. presence: minus presence where the count is above 0.

The device holds per verify row r = u * T + j (T = K + 1, at most 32 rows) the counts of the committed output of the
row's request (``counts``, float32, the same on the T rows of one user) and its prompt mask (``prompt``, bfloat16
0/1), each device its vocabulary shard. ``VerifyPenalties.apply_body`` adds the drafts of column j from the verify step
buffer, applies the penalties in float32 to the verify's vocabulary shard of the logits, then:

  * a penalty row the step does not draw (greedy, or top-k 1) takes the exact float32 argmax of its penalized logits
    (the fused per-shard argmax), which replaces its lanes of the verify ids, so the id walk certifies it as it
    certifies a greedy row;
  * the shard is written back in bfloat16, so the sampler trace draws a sampled penalty row from penalized logits.

A row with neutral penalties (repetition 1, frequency 0, presence 0) is unchanged bit for bit (x * 1 - 0 - 0).

Who keeps the counts current:

  * a host-staged verify: the engine compares each penalty row with its own record (``PenaltyRow``: the prompt and
    output the device row holds) and adds the committed tokens the record lacks (``add``), or rewrites the row from
    the plugin's lists (``rebuild``) for a new request, a moved row or a long gap;
  * a chained verify (spec_chain): the plugin does not know the predecessor's commit yet, so the commit trace
    (``commit_body``) adds it on the device from the predecessor's ids (draws for a drawn user) and accepted count (the
    drafter pack's anchor row); the engine adds the same tokens to its record once it reads the predecessor.

Draws of a sampled penalty row come from the bfloat16 penalized logits: a penalized logit can differ from the float32
value vLLM samples from by half a bfloat16 ulp, as the unpenalized logits are bfloat16 already. A greedy penalty row
takes the float32 argmax, which equals the plugin's host path exactly when the device float32 arithmetic rounds as
torch does.

The device class takes an op set ``ops`` (default ttnn) and an argmax, so a host test runs the same op sequence in
torch.
"""

from dataclasses import dataclass, field

import torch

from models.demos.blackhole.qwen36.tt.spec_sample import _row_value
from models.demos.blackhole.qwen36.tt.spec_step import VERIFY_TOKENS, draft_anchor

REPETITION_NEUTRAL, FREQUENCY_NEUTRAL, PRESENCE_NEUTRAL = 1.0, 0.0, 0.0
# A host-staged verify adds at most this many committed tokens per row with one-hot adds; a longer gap rewrites the row.
MAX_ADDED_TOKENS = 16


# --------------------------------------------------------------------------------------------------------------- host
@dataclass
class StepPenalties:
    """The penalties of one verify of B users: per user ``repetition``, ``frequency``, ``presence`` (float32 [B]) and
    ``live`` [B] bool (an active user whose penalties are not neutral)."""

    repetition: torch.Tensor
    frequency: torch.Tensor
    presence: torch.Tensor
    live: torch.Tensor

    @property
    def users(self):
        return [u for u, x in enumerate(self.live.tolist()) if x]


def step_penalties(sampling, active):
    """The ``StepPenalties`` of a verify whose ``sampling`` (the plugin's per-row ``sampling_params``, or None) covers
    the rows ``active`` [B] bool, or None when no active row has a penalty. Raises ValueError on a repetition penalty
    at or below 0 or a value that is not finite."""
    if sampling is None:
        return None
    active = torch.as_tensor(active, dtype=torch.bool).reshape(-1)
    B = int(active.numel())
    names = (
        ("repetition_penalty", REPETITION_NEUTRAL),
        ("frequency_penalty", FREQUENCY_NEUTRAL),
        ("presence_penalty", PRESENCE_NEUTRAL),
    )
    values = []
    for name, neutral in names:
        per = [float(_row_value(getattr(sampling, name, None), u, neutral)) if active[u] else neutral for u in range(B)]
        values.append(torch.tensor(per, dtype=torch.float32))
    rep, freq, pres = values
    if not bool(torch.isfinite(torch.stack(values)).all()) or bool((rep <= 0).any()):
        raise ValueError(
            f"penalties must be finite with repetition above 0: repetition {rep.tolist()}, frequency {freq.tolist()}, "
            f"presence {pres.tolist()}"
        )
    live = active & ((rep != REPETITION_NEUTRAL) | (freq != FREQUENCY_NEUTRAL) | (pres != PRESENCE_NEUTRAL))
    if not bool(live.any()):
        return None
    return StepPenalties(rep, freq, pres, live)


def token_row(tokens, row):
    """Row ``row`` of a plugin token tensor (prompt or output, -1 padded) as int64 ids, padding removed."""
    t = torch.as_tensor(tokens)[int(row)].reshape(-1).to(torch.int64)
    return t[t >= 0]


def counts_of(ids, vocab):
    """float32 [vocab]: how often each token id is in ``ids`` (ids outside [0, vocab) ignored)."""
    t = torch.as_tensor(ids, dtype=torch.int64).reshape(-1)
    t = t[(t >= 0) & (t < vocab)]
    return torch.bincount(t, minlength=vocab).to(torch.float32)


def reference_penalties(logits, prompt, history, repetition, frequency, presence):
    """vLLM's apply_penalties on one row: ``logits`` [V] (any float dtype, computed in float32), ``prompt`` and
    ``history`` the prompt ids and the output ids before the scored token. Returns float32 [V]."""
    x = torch.as_tensor(logits).to(torch.float32).reshape(1, -1).clone()
    V = x.shape[1]
    counts = counts_of(history, V).reshape(1, V)
    seen = (counts > 0) | (counts_of(prompt, V).reshape(1, V) > 0)
    rep = torch.tensor([float(repetition)], dtype=torch.float32)
    penalties = torch.where(seen, rep.reshape(1, 1).expand(1, V), torch.ones(1, V))
    x *= torch.where(x > 0, 1.0 / penalties, penalties)
    x -= torch.tensor([float(frequency)], dtype=torch.float32).reshape(1, 1) * counts
    x -= torch.tensor([float(presence)], dtype=torch.float32).reshape(1, 1) * (counts > 0).to(torch.float32)
    return x.reshape(V)


@dataclass
class PenaltyRow:
    """What the device's counts row of one request hold: its ``prompt`` and committed ``output`` ids (int64), the device
    row ``row`` they are in (the state slot at the last write; a row move leaves them where they were), and
    ``pending``: the verifies (PendingVerify, row) whose commit a chained verify added on the device and the record
    does not have yet."""

    prompt: torch.Tensor
    output: torch.Tensor
    row: int
    pending: list = field(default_factory=list)

    def resolve(self, T):
        """Append each pending verify's commit for its row: its ids up to the drafter pack's anchor count. Reads the
        verify if it is not read yet."""
        for step, r in self.pending:
            ids = step.wait()
            m = int(step.pack[draft_anchor(step.K), r]) - r * T
            self.output = torch.cat([self.output, ids[r, : m + 1].to(torch.int64)])
        self.pending = []

    def matches(self, prompt, output, row):
        """Whether the device row ``row`` holds exactly ``prompt`` and ``output`` (pending commits resolved)."""
        return (
            self.row == row
            and not self.pending
            and torch.equal(self.prompt, prompt)
            and torch.equal(self.output, torch.as_tensor(output, dtype=torch.int64))
        )


def reconcile(records, users, prompts, outputs):
    """Bring the records of the users ``users`` to ``prompts[u]`` and ``outputs[u]`` (int64 ids): returns (rebuild
    {u: (prompt, output)}, add {u: ids}) and replaces the records. A user whose record holds its prompt on its own row
    and a prefix of its output at most MAX_ADDED_TOKENS short gets the missing tokens added; any other is rebuilt."""
    rebuild, add = {}, {}
    for u in users:
        prompt, output = prompts[u], outputs[u]
        rec = records[u]
        n = 0 if rec is None else int(rec.output.numel())
        if (
            rec is not None
            and not rec.pending
            and rec.row == u
            and torch.equal(rec.prompt, prompt)
            and n <= int(output.numel()) <= n + MAX_ADDED_TOKENS
            and torch.equal(output[:n], rec.output)
        ):
            if int(output.numel()) > n:
                add[u] = output[n:]
        else:
            rebuild[u] = (prompt, output)
        records[u] = PenaltyRow(prompt.clone(), output.clone(), u)
    return rebuild, add


# ------------------------------------------------------------------------------------------------------------- device
class VerifyPenalties:
    """The device penalty state of a verify of B users x T columns (R = B * T <= 32 rows) over a vocabulary of
    ``vocab`` ids in ``shards`` shards (each device holds [1, 1, R, vocab / shards] of the counts and the prompt mask,
    as the vocab-sharded LM head lays out the logits; 1 when every device holds the whole vocabulary).

    Inputs it reads, bound by the owner: ``step`` (the verify step buffer, spec_step), ``ids_out`` (the verify ids,
    [1, 1, 1, 32] uint32 lanes), ``draws`` (the verify sampler's draws, [1, 1, 1, R] uint32) and, for the commit body,
    ``pack`` (the drafter pack). ``argmax`` maps a float32 vocabulary shard [1, 1, r, vocab / shards] to its global
    argmax ids in the ids lanes' layout (the fused per-shard argmax).

    Persistent tensors (allocated here, before any capture): the state ``counts`` and ``prompt``, the per-row
    parameters of the step (``repetition``, ``inverse`` = 1 / repetition, ``frequency``, ``presence``, ``track`` = 1 on
    the rows whose counts a chained verify keeps), the per-lane ``greedy`` mask (lanes whose id the penalized argmax
    replaces), the per-user ``drawn`` mask of the predecessor (commit body), the staging buffers of ``rebuild`` and
    ``add``, and constants. ``apply_body`` and ``commit_body`` are trace bodies, which the owner compiles by running them
    once before any capture; ``rebuild``, ``add`` and ``stage`` are eager host-driven writes, compiled by
    ``warm_eager``."""

    def __init__(self, mesh, B, T, vocab, shards=1, argmax=None, ops=None):
        if ops is None:
            import ttnn as ops
        R = int(B) * int(T)
        if R > 32:
            raise ValueError(f"{B} users x {T} columns exceed the 32 verify rows")
        if int(vocab) % int(shards):
            raise ValueError(f"a vocabulary of {vocab} does not split into {shards} shards")
        if int(vocab) >= 1 << 24:
            raise ValueError(f"token ids must stay exact in float32: vocabulary {vocab} >= 2**24")
        self.ops, self.mesh, self.argmax = ops, mesh, argmax
        self.B, self.T, self.K, self.R = int(B), int(T), int(T) - 1, R
        self.vocab, self.shards = int(vocab), int(shards)
        self.width = self.vocab // self.shards
        self.step = self.ids_out = self.draws = self.pack = None
        f32, bf16 = ops.float32, ops.bfloat16
        col = lambda v: torch.as_tensor(v, dtype=torch.float32).reshape(1, 1, R, 1)  # noqa: E731
        j = torch.arange(R) % T
        self.iota = self._sharded(torch.arange(self.vocab, dtype=torch.float32).reshape(1, 1, 1, -1), f32)
        self.counts = self._sharded(torch.zeros(1, 1, R, self.vocab), f32)
        self.prompt = self._sharded(torch.zeros(1, 1, R, self.vocab), bf16)
        self.stage_counts = self._sharded(torch.zeros(1, 1, R, self.vocab), f32)
        self.stage_prompt = self._sharded(torch.zeros(1, 1, R, self.vocab), bf16)
        self.repetition, self.inverse, self.frequency, self.presence, self.track = (
            self._replicated(col(torch.full((R,), v)), f32) for v in (1.0, 1.0, 0.0, 0.0, 0.0)
        )
        self.sel_f, self.keep_f = self._replicated(col(torch.zeros(R)), f32), self._replicated(col(torch.ones(R)), f32)
        self.sel_b, self.keep_b = self._replicated(col(torch.zeros(R)), bf16), self._replicated(
            col(torch.ones(R)), bf16
        )
        self.add_ids = self._replicated(col(torch.full((R,), -1.0)), f32)
        self.greedy = self._replicated(torch.zeros(1, 1, 1, 32), f32)
        self.drawn = self._replicated(torch.zeros(self.B, 1), f32)
        self.ones_t = self._replicated(torch.ones(1, T), f32)
        self.iota_t = self._replicated(torch.arange(T, dtype=torch.float32).reshape(1, T), f32)
        self.user_rows = self._replicated((torch.arange(self.B) * T).to(torch.float32).reshape(self.B, 1), f32)
        # keep[i]: 1 on the rows of column j >= i, which score draft i
        self.keep = [None] + [self._replicated(col((j >= i).to(torch.float32)), f32) for i in range(1, T)]
        self._staged = {}  # name -> host tensor its device buffer holds
        self.buffers = [
            self.iota, self.counts, self.prompt, self.stage_counts, self.stage_prompt, self.repetition, self.inverse,
            self.frequency, self.presence, self.track, self.sel_f, self.keep_f, self.sel_b, self.keep_b, self.add_ids,
            self.greedy, self.drawn, self.ones_t, self.iota_t, self.user_rows, *self.keep[1:],
        ]  # fmt: skip

    def _mapper(self, sharded):
        ops = self.ops
        if sharded and self.shards > 1:
            return ops.ShardTensorToMesh(self.mesh, dim=3)
        return ops.ReplicateTensorToMesh(self.mesh)

    def _host(self, t, dtype, sharded=False):
        ops = self.ops
        return ops.from_torch(t, dtype=dtype, layout=ops.TILE_LAYOUT, mesh_mapper=self._mapper(sharded))

    def _device(self, t, dtype, sharded):
        ops = self.ops
        return ops.to_device(self._host(t, dtype, sharded), self.mesh, memory_config=ops.DRAM_MEMORY_CONFIG)

    def _sharded(self, t, dtype):
        return self._device(t, dtype, True)

    def _replicated(self, t, dtype):
        return self._device(t, dtype, False)

    def _copy(self, t, dst, dtype, sharded=False):
        self.ops.copy_host_to_device_tensor(self._host(t, dtype, sharded), dst)

    def bind(self, step=None, ids_out=None, draws=None, pack=None):
        """Point the bodies at the buffers they read and write (a stand-in until the real one exists)."""
        for name, t in (("step", step), ("ids_out", ids_out), ("draws", draws), ("pack", pack)):
            if t is not None:
                setattr(self, name, t)

    # ------------------------------------------------------------------------------------------------------ bodies
    def _float(self, t, owned):
        """A uint32 ROW_MAJOR tensor as float32 TILE (exact below 2**24)."""
        ops = self.ops
        tile = ops.to_layout(t, ops.TILE_LAYOUT)
        out = ops.typecast(tile, ops.float32)
        owned += [tile, out]
        return out

    def _rows(self, col, owned):
        """[B, 1] per user -> [1, 1, R, 1] per verify row (the user's value on each of its T rows)."""
        ops = self.ops
        wide = ops.multiply(col, self.ones_t)
        rows = ops.reshape(wide, (1, 1, self.R, 1))
        owned += [wide, rows]
        return rows

    def _one_hot_add(self, counts, ids, owned):
        """counts + one-hot of ``ids`` [1, 1, R, 1] (-1 adds nothing) as a new tensor."""
        ops = self.ops
        hot = ops.eq(self.iota, ids, dtype=ops.float32)
        out = ops.add(counts, hot)
        ops.deallocate(hot)
        owned.append(out)
        return out

    def _free(self, owned, keep=()):
        ops, done = self.ops, []
        for t in owned:
            if t is None or any(t is k for k in keep) or any(t is d for d in done):
                continue
            if any(_aliases(t, k) for k in (*keep, self.step, self.ids_out, self.draws, self.pack) if k is not None):
                continue
            ops.deallocate(t)
            done.append(t)

    def apply_body(self, x):
        """Trace body over the verify's vocabulary shard ``x`` (bfloat16 TILE [1, 1, r, vocab / shards], r = b * T <=
        R: a verify row bucket's first rows), written in place, and the verify ids' lanes of the greedy penalty rows
        (module doc)."""
        ops, B, T, R = self.ops, self.B, self.T, self.R
        r = int(x.shape[2])
        owned = []
        tok = ops.slice(self.step, (VERIFY_TOKENS, 0), (VERIFY_TOKENS + 1, R))
        owned.append(tok)
        shaped = ops.reshape(tok, (B, T))
        owned.append(shaped)
        tokf = self._float(shaped, owned)
        counts = self.counts
        for i in range(1, T):
            draft = ops.slice(tokf, (0, i), (B, i + 1))
            owned.append(draft)
            on_rows = self._rows(draft, owned)
            # -1 on the rows of the columns before i: (id + 1) * keep - 1
            plus = ops.add(on_rows, 1.0)
            kept = ops.multiply(plus, self.keep[i])
            ids = ops.subtract(kept, 1.0)
            owned += [plus, kept, ids]
            counts = self._one_hot_add(counts, ids, owned)
        prompt, params = self.prompt, (self.repetition, self.inverse, self.frequency, self.presence)
        if r < R:
            cut = lambda t: ops.slice(t, (0, 0, 0, 0), (1, 1, r, int(t.shape[3])))  # noqa: E731
            counts, prompt = cut(counts), cut(prompt)
            params = tuple(cut(p) for p in params)
            owned += [counts, prompt, *params]
        rep, inv, freq, pres = params
        xf = ops.typecast(x, ops.float32)
        present = ops.gt(counts, 0.0)
        in_prompt = ops.typecast(prompt, ops.float32)
        either = ops.add(present, in_prompt)
        seen = ops.gt(either, 0.0)
        positive = ops.gt(xf, 0.0)
        scale = ops.where(positive, inv, rep)
        scale_seen = ops.where(seen, scale, 1.0)
        scaled = ops.multiply(xf, scale_seen)
        freq_term = ops.multiply(counts, freq)
        after_freq = ops.subtract(scaled, freq_term)
        pres_term = ops.multiply(present, pres)
        penalized = ops.subtract(after_freq, pres_term)
        owned += [xf, present, in_prompt, either, seen, positive, scale, scale_seen, scaled, freq_term, after_freq]
        owned += [pres_term, penalized]
        ids = self.argmax(penalized)
        owned.append(ids)
        self._merge_ids(ids, owned)
        out = ops.typecast(penalized, ops.bfloat16)
        owned.append(out)
        ops.copy(out, x)
        self._free(owned, keep=(x, self.counts, self.prompt, *self._params()))

    def _params(self):
        return (self.repetition, self.inverse, self.frequency, self.presence)

    def _merge_ids(self, ids, owned):
        """ids_out lanes <- ``ids`` lanes where ``greedy`` is 1 (float32, exact below 2**24)."""
        ops = self.ops
        old = self._float(self.ids_out, owned)
        new = self._float(ids, owned)
        diff = ops.subtract(new, old)
        picked = ops.multiply(diff, self.greedy)
        merged = ops.add(old, picked)
        words = ops.typecast(merged, ops.uint32)
        rm = ops.to_layout(words, ops.ROW_MAJOR_LAYOUT)
        owned += [diff, picked, merged, words, rm]
        ops.copy(rm, self.ids_out)

    def commit_body(self):
        """Trace body of a chained verify, enqueued before its verify trace: on every ``track`` row, counts += one-hot
        of the predecessor's commit of the row's user u, ids[u, 0 .. m] with m = anchor[u] - u * T from the drafter
        pack and ids the verify ids, or the draws where ``drawn`` is 1."""
        ops, B, T, R = self.ops, self.B, self.T, self.R
        owned = []
        anchor = ops.slice(self.pack, (draft_anchor(self.K), 0), (draft_anchor(self.K) + 1, B))
        owned.append(anchor)
        a = self._float(anchor, owned)
        a_col = ops.transpose(a, 0, 1)
        m = ops.subtract(a_col, self.user_rows)
        count = ops.add(m, 1.0)
        owned += [a_col, m, count]

        def lanes(t):
            s = ops.slice(t, (0, 0, 0, 0), (1, 1, 1, R))
            shaped = ops.reshape(s, (B, T))
            owned.extend([s, shaped])
            return self._float(shaped, owned)

        ids, draws = lanes(self.ids_out), lanes(self.draws)
        diff = ops.subtract(draws, ids)
        picked = ops.multiply(diff, self.drawn)
        merged = ops.add(ids, picked)
        within = ops.lt(self.iota_t, count, dtype=ops.float32)  # [B, T]: column < count
        plus = ops.add(merged, 1.0)
        kept = ops.multiply(plus, within)
        committed = ops.subtract(kept, 1.0)  # the id, or -1 past the count
        owned += [diff, picked, merged, within, plus, kept, committed]
        for i in range(T):
            column = ops.slice(committed, (0, i), (B, i + 1))
            owned.append(column)
            on_rows = self._rows(column, owned)
            plus_r = ops.add(on_rows, 1.0)
            tracked = ops.multiply(plus_r, self.track)
            ids_r = ops.subtract(tracked, 1.0)
            hot = ops.eq(self.iota, ids_r, dtype=ops.float32)
            ops.add_(self.counts, hot)
            owned += [plus_r, tracked, ids_r, hot]
        self._free(owned)

    # ------------------------------------------------------------------------------------------------ eager writes
    def stage(self, penalties, greedy_lanes, track=None, drawn=None):
        """Host -> device copies of a step's parameters (each user's on its T rows; neutral elsewhere), its ``greedy``
        lanes (verify rows whose ids the penalized argmax replaces), and for a chained verify the ``track`` users and
        the predecessor's ``drawn`` users. A buffer that already holds the values is not copied again."""
        ops, B, T, R = self.ops, self.B, self.T, self.R
        user = torch.arange(R) // T

        def per_row(v):
            return torch.as_tensor(v, dtype=torch.float32).reshape(B)[user].reshape(1, 1, R, 1)

        rep = penalties.repetition.to(torch.float32)
        lanes = torch.zeros(1, 1, 1, 32)
        lanes[0, 0, 0, : len(greedy_lanes)] = torch.as_tensor(greedy_lanes, dtype=torch.float32)
        tracked = torch.zeros(B) if track is None else torch.as_tensor(track, dtype=torch.float32)
        drawn = torch.zeros(B) if drawn is None else torch.as_tensor(drawn, dtype=torch.float32)
        values = {
            "repetition": per_row(rep),
            "inverse": per_row(1.0 / rep),  # vLLM's own float32 division: 1.0 / penalties
            "frequency": per_row(penalties.frequency),
            "presence": per_row(penalties.presence),
            "track": per_row(tracked),
            "greedy": lanes,
            "drawn": drawn.reshape(B, 1),
        }
        for name, host in values.items():
            held = self._staged.get(name)
            if held is not None and torch.equal(held, host):
                continue
            self._copy(host, getattr(self, name), ops.float32)
            self._staged[name] = host

    def rebuild(self, rows):
        """Write the state of users ``rows`` {u: (prompt ids, output ids)} from the host and leave the other rows:
        dense staging buffers, then state = state * keep + staged * sel on the device (eager, exact: 0/1 weights)."""
        if not rows:
            return
        ops, T, R, V = self.ops, self.T, self.R, self.vocab
        counts = torch.zeros(R, V, dtype=torch.float32)
        mask = torch.zeros(R, V, dtype=torch.float32)
        sel = torch.zeros(R, dtype=torch.float32)
        for u, (prompt, output) in sorted(rows.items()):
            if not 0 <= int(u) < self.B:
                raise ValueError(f"penalty state for user {u} outside range({self.B})")
            lo, hi = int(u) * T, (int(u) + 1) * T
            counts[lo:hi] = counts_of(output, V)
            mask[lo:hi] = (counts_of(prompt, V) > 0).to(torch.float32)
            sel[lo:hi] = 1.0
        self._copy(counts.reshape(1, 1, R, V), self.stage_counts, ops.float32, sharded=True)
        self._copy(mask.reshape(1, 1, R, V), self.stage_prompt, ops.bfloat16, sharded=True)
        col = sel.reshape(1, 1, R, 1)
        self._copy(col, self.sel_f, ops.float32)
        self._copy(1.0 - col, self.keep_f, ops.float32)
        self._copy(col, self.sel_b, ops.bfloat16)
        self._copy(1.0 - col, self.keep_b, ops.bfloat16)
        self._merge()

    def _merge(self):
        ops = self.ops
        for state, staged, sel, keep in (
            (self.counts, self.stage_counts, self.sel_f, self.keep_f),
            (self.prompt, self.stage_prompt, self.sel_b, self.keep_b),
        ):
            kept, taken = ops.multiply(state, keep), ops.multiply(staged, sel)
            merged = ops.add(kept, taken)
            ops.copy(merged, state)
            for t in (kept, taken, merged):
                ops.deallocate(t)

    def add(self, tokens):
        """counts of user u += one-hot of each id of ``tokens[u]`` (eager; one pass per token position)."""
        if not tokens:
            return
        ops, T, R = self.ops, self.T, self.R
        n = max(int(torch.as_tensor(t).numel()) for t in tokens.values())
        for k in range(n):
            ids = torch.full((R,), -1.0)
            for u, t in tokens.items():
                t = torch.as_tensor(t).reshape(-1)
                if k < t.numel():
                    ids[int(u) * T : (int(u) + 1) * T] = float(t[k])
            self._copy(ids.reshape(1, 1, R, 1), self.add_ids, ops.float32)
            self._add_pass()

    def _add_pass(self):
        ops = self.ops
        hot = ops.eq(self.iota, self.add_ids, dtype=ops.float32)
        ops.add_(self.counts, hot)
        ops.deallocate(hot)

    def warm_eager(self):
        """Compile the eager programs (the merge of ``rebuild`` and the pass of ``add``) before a capture; the owner
        compiles the bodies by running them. Leaves the state zero and the parameters neutral."""
        self._merge()
        self._add_pass()
        self.reset()

    def reset(self):
        """Zero the state and make every parameter neutral (host copies)."""
        ops, R, V = self.ops, self.R, self.vocab
        for t, dtype in ((self.counts, ops.float32), (self.prompt, ops.bfloat16)):
            self._copy(torch.zeros(1, 1, R, V), t, dtype, sharded=True)
        for t, v in ((self.repetition, 1.0), (self.inverse, 1.0), (self.frequency, 0.0), (self.presence, 0.0)):
            self._copy(torch.full((1, 1, R, 1), v), t, ops.float32)
        self._copy(torch.zeros(1, 1, R, 1), self.track, ops.float32)
        self._copy(torch.zeros(1, 1, 1, 32), self.greedy, ops.float32)
        self._staged = {}

    def read(self, users):
        """(counts float32 [n, vocab], prompt mask bool [n, vocab]) of ``users`` (their first row), after a wait."""
        ops = self.ops
        ops.synchronize_device(self.mesh)
        if self.shards > 1:
            compose = ops.ConcatMeshToTensor(self.mesh, dim=3)
            c, p = (ops.to_torch(t, mesh_composer=compose) for t in (self.counts, self.prompt))
        else:
            c, p = (ops.to_torch(ops.get_device_tensors(t)[0]) for t in (self.counts, self.prompt))
        idx = [int(u) * self.T for u in users]
        c, p = c.reshape(-1, self.vocab)[: self.R], p.reshape(-1, self.vocab)[: self.R]
        return c[idx].to(torch.float32), p[idx].to(torch.float32) > 0

    def release(self):
        for t in self.buffers:
            self.ops.deallocate(t)
        self.buffers = []


def _aliases(a, b):
    """Whether two device handles name one buffer; a handle without an address (a host stand-in) aliases only
    itself."""
    if a is b:
        return True
    try:
        return a.buffer_address() == b.buffer_address()
    except Exception:
        return False

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""DFlash2 speculative decode as a long-lived MULTI-SLOT serving session (vLLM block-output contract).

The demo decoder (DFlash2Decoder.generate) owns a closed batch of B prompts end to end. A server has
S = max_batch_size physical SLOTS that requests join and leave at arbitrary steps, and cannot
re-capture anything per request. This decoder keeps every trace and buffer for the server's lifetime
and expresses each request as a slot of the batched substrate:

  alloc()     ONCE: tap bufs (B*T rows), the drafter's per-slot RING context KV store + one draft
              LAYOUT per bucket.
  warm()      ONCE, eager, before any capture: the GDN spec cfg of every bucket, the verify cfg of
              every bucket (its eager verify pass), then the warm SWEEP: every join seed (bucket, row,
              slot), every bucket-switch realization (row, mi) in both directions, the draft/extend of
              every layout -- so nothing spec-related compiles once a trace is parked.
  capture()   ONCE: the verify trace of every bucket (dummy per-row tables), then the first traced
              steps capture each layout's draft/extend traces. Nothing spec-related compiles or
              allocates after this (warm_sweep() re-run + program-cache count = the caller's guard).
  ingest_prompt(phys, taps, T, chunk_start)   per request: its prompt taps -> slot phys's drafter ring.
  begin(phys, first, T, page_table_row)       per request: seat slot phys on a ROW of the current
              bucket and seed that row's GDN spec state from the slot's prefilled decode row
              (seed_spec_row); every other row HOLDS (HOLD ring index, window copied through,
              position -1 rows: no KV write), so its durable spec state is bit-identical afterwards.
              QWEN36_DFLASH_FOLD_SEED=1: no replay in begin(); the slot is FRESH with pending = first,
              p = T-1 and its first step() performs the seed row-0 computation inside a real
              speculative iteration (see _FOLD_SEED).
  plan(live_after)  per engine step, BEFORE any begin/set_table/step of that step: the bucket
              hysteresis; may switch buckets (set_bucket) and returns the bucket id in force.
  step()      ONE draft (all rows of the current layout) + verify replay (live rows speculate, every
              other row holds) + greedy accept + extend; returns {slot: committed ids} for the live
              slots (a FRESH slot's list omits com[0] == first, which the caller already holds).
  end(phys)   slot phys leaves (its row keeps holding; in a compacting bucket the row is freed).
  release()   shutdown.

BUCKETS. A bucket is a verify geometry (B_cfg users x T_cfg = K_cfg+1 rows) with its own verify trace,
GDN spec cfg (conv-window pair + ctrl page) and drafter layout, all sharing ONE fp32 GDN ring, one
32-row set of tap buffers and one drafter KV store. The bucket with B_cfg == S is the IDENTITY bucket
(row == slot, never compacted); a bucket with B_cfg < S seats its live slots on rows (stable lowest-free
assignment) and HOLDS its empty rows permanently. Host state (active / pending / p / mi / ctx_len ...)
is per PHYSICAL slot; ``row_tables[phys]`` is the only page-table truth and each bucket's [B_cfg, nb]
verify tables are rebuilt from it through the row map. A single-bucket construction (the batch4 /
single-user profiles, tests/test_dflash2_serving.py) is exactly today's decoder: one bucket 'default',
row == slot, no switch code path reachable.

BUCKET SWITCH (set_bucket, the spec's WHOLE-BATCH reseed; see bucket_switch_protocol in the design):
with L = the active slots (|L| <= B_tgt) and their (row_cur, mi): per GDN layer ONE switch_spec_cfg
moves ring block (mi*B_cur + row_cur)*Nv -> the target's slot-0 block row_tgt*Nv and the conv window
rows E_prev_cur[row_cur, mi:mi+K] -> E_prev_tgt[row_tgt, 0:K] (an exact byte move: slices, concats,
copies, no arithmetic); empty target rows get their own slot-0 block (identity) and a zero window row.
Then mi <- 0 for every moved slot, the target's tables are rebuilt from row_tables and restaged, the
target layout's rows are restaged (set_rows), and cur <- tgt. NOTHING else moves: pending / p / ctx_len
/ fresh / carry are untouched, the attention KV is per request (the next verify rewrites p+1..p+T
before reading it), the drafter store is per slot. No token is emitted, lost or duplicated by a switch:
tokens enter the caller's carry only from step(). The old bucket's ring blocks / windows / tables are
simply stale (re-seeded whole at the next switch into it).

HYSTERESIS (BucketPlanner; env-tunable, frozen after set_bucket is measured): UP (to a bucket that
seats every live slot) is forced as soon as |live_after| exceeds the current capacity; DOWN happens
after QWEN36_DFLASH_BUCKET_DOWN_STEPS consecutive plan() calls that fit a smaller bucket AND at least
QWEN36_DFLASH_BUCKET_COOLDOWN steps since the last switch; the cooldown doubles (cap 64) when an
up-switch follows a down-switch within 2x cooldown and halves back after 64 quiet steps. Never twice in
one decode step (a forced up-switch is the only exception, for correctness), never mid-step.

Deferred commit: like the demo, a slot's accepted-prefix index mi is folded into the NEXT replay's
selectors; a held row's replay reads (and rewrites, unchanged) exactly that row.
Page tables: a bucket's per-row tables are restaged (host->device copy) whenever a seated slot's vLLM
row changes; idle / empty rows point at block 0 (vLLM's never-allocated null block) so their held rows
write nowhere that matters.
"""

import inspect
import os
import time
from dataclasses import dataclass, field

import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tt.dflash2_decode import (
    _DEBUG,
    _DRAFT_TRACED,
    DFlash2Decoder,
    _dbg,
    default_draft_len,
    get_drafter,
)
from models.tt_transformers.tt.common import get_block_size

# QWEN36_DFLASH_FOLD_SEED=1: fold the seed replay into the slot's first speculative step. begin() then
# only seeds the GDN spec state (ring slot-0 block + window row of the slot's ROW) and marks the slot FRESH
# with pending = first, p = T-1, ctx_len = T. Its first step() drafts from ``first`` at position T over
# context 0..T-1 (exactly the prompt taps the drafter holds), verifies [first, drafts] at positions T..
# with mi_prev = 0 and, for row 0, performs the seed replay's row-0 computation to the bit: the recurrence,
# the conv and the attention are causal per row, so row 0 reads only the seeded ring block / window row /
# prompt KV and the other rows and slots cannot reach it. That step emits com[1:] (com[0] == first was
# emitted by the caller at prefill) while mi / next pending / the extend (row 0's tap -> ring slot T) use
# the full com, so every later invariant (ctx_len == p+1, draft at p+1, verify at p+1, ring block mi) is
# unchanged. Saves one verify replay + extend per join. Default OFF until validated on device
# (tests/test_dflash2_serving.py with the knob set).
_FOLD_SEED = os.environ.get("QWEN36_DFLASH_FOLD_SEED", "0") == "1"

# The bucket / cfg / layout id a SINGLE-bucket construction uses: the GDN and model registries' default
# cfg, so their back-compat accessors (verify_win_cur(), _vfy_trace_id, _verify_win_buf) name this bucket.
DEFAULT_BUCKET = "default"


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else int(default)


def parse_buckets(spec, default=None):
    """'8x4,4x8' -> ((8, 4), (4, 8)) as (B_cfg, T_cfg) pairs; an empty / None spec returns ``default``."""
    if not spec or not str(spec).strip():
        return default
    out = []
    for item in str(spec).split(","):
        item = item.strip().lower()
        if not item:
            continue
        b, t = item.split("x")
        out.append((int(b), int(t)))
    assert out, f"no buckets in {spec!r}"
    return tuple(out)


def bucket_id(B, T):
    return f"{int(B)}x{int(T)}"


# ---------------------------------------------------------------------------------------------- host-only policy
class BucketPlanner:
    """The bucket hysteresis, host-only (no device state): which bucket ``n`` live slots should run in.

    ``capacities``: bucket id -> B_cfg (how many slots it seats). UP to the smallest bucket that seats n
    is FORCED whenever n exceeds the current capacity. DOWN to the smallest bucket that seats n needs
    ``down_steps`` consecutive decide() calls that fit it AND >= ``cooldown`` steps since the last switch;
    the cooldown doubles (up to ``cooldown_cap``) when an up-switch follows a down-switch within 2x the
    cooldown, and halves back towards its base after ``decay_steps`` quiet steps. ``step`` is the
    caller's step clock (the decoder's total_steps).
    """

    def __init__(self, capacities, start, down_steps=None, cooldown=None, cooldown_cap=64, decay_steps=64):
        assert start in capacities
        self.caps = dict(capacities)
        self.order = sorted(self.caps, key=lambda k: (self.caps[k], k))  # ascending capacity
        self.cur = start
        self.down_steps = _env_int("QWEN36_DFLASH_BUCKET_DOWN_STEPS", 8) if down_steps is None else int(down_steps)
        self.cooldown0 = _env_int("QWEN36_DFLASH_BUCKET_COOLDOWN", 16) if cooldown is None else int(cooldown)
        self.cooldown = self.cooldown0
        self.cooldown_cap = max(int(cooldown_cap), self.cooldown0)
        self.decay_steps = int(decay_steps)
        self.last_switch_step = None
        self.last_switch_dir = None
        self.last_decay_step = None
        self.consec_small = 0
        self.n_up = 0
        self.n_down = 0

    @property
    def max_capacity(self):
        return max(self.caps.values())

    def required(self, n):
        """The smallest bucket that seats n slots."""
        for k in self.order:
            if self.caps[k] >= n:
                return k
        raise AssertionError(f"{n} live slots exceed every bucket ({self.caps})")

    def decide(self, n, step):
        """(target id, 'up' | 'down') if a switch is due for n live slots at this step, else (None, None).
        Pure: call note_switch() once the switch has actually happened."""
        n = int(n)
        assert 0 <= n <= self.max_capacity, f"{n} live slots exceed the largest bucket ({self.max_capacity})"
        cur_cap = self.caps[self.cur]
        if n > cur_cap:
            return self.required(n), "up"  # forced: the current bucket cannot seat them
        cand = self.required(n)
        if self.caps[cand] >= cur_cap:
            self.consec_small = 0
            return None, None
        self.consec_small += 1
        # Back-off decay: halve the cooldown towards its base after a quiet stretch.
        since = [s for s in (self.last_switch_step, self.last_decay_step) if s is not None]
        if self.cooldown > self.cooldown0 and (not since or step - max(since) >= self.decay_steps):
            self.cooldown = max(self.cooldown0, self.cooldown // 2)
            self.last_decay_step = step
        quiet = float("inf") if self.last_switch_step is None else step - self.last_switch_step
        if self.consec_small >= self.down_steps and quiet >= self.cooldown:
            return cand, "down"
        return None, None

    def note_switch(self, target, direction, step):
        """Record a switch that happened (planned or forced by a caller) at ``step``."""
        assert target in self.caps and direction in ("up", "down")
        if (
            direction == "up"
            and self.last_switch_dir == "down"
            and self.last_switch_step is not None
            and step - self.last_switch_step <= 2 * self.cooldown
        ):
            self.cooldown = min(self.cooldown_cap, self.cooldown * 2)  # churn: back off
        self.last_switch_step = step
        self.last_switch_dir = direction
        self.consec_small = 0
        self.cur = target
        if direction == "up":
            self.n_up += 1
        else:
            self.n_down += 1

    def stats(self):
        return {
            "cur": self.cur,
            "cooldown": self.cooldown,
            "cooldown_base": self.cooldown0,
            "down_steps": self.down_steps,
            "consec_small": self.consec_small,
            "last_switch_step": self.last_switch_step,
            "n_up": self.n_up,
            "n_down": self.n_down,
        }


@dataclass
class Bucket:
    """One verify geometry: B users x T rows, its row -> slot map and its [B, nb] verify tables."""

    id: str
    B: int
    T: int
    K: int
    identity: bool  # B == S: row == slot, never compacted
    rows: list  # row -> phys | None
    tables: torch.Tensor  # int32 [B, nb_v], row r = row_tables[rows[r]] (zeros for an empty row)
    layout: object = None  # the drafter's DraftLayout (alloc)
    stats: dict = field(
        default_factory=lambda: {"steps": 0, "committed": 0, "accepted": 0, "drafted": 0, "switches_in": 0}
    )

    def row_of(self, phys):
        for r, p in enumerate(self.rows):
            if p == phys:
                return r
        return None

    def free_rows(self):
        return [r for r, p in enumerate(self.rows) if p is None]


def assign_rows(bucket, live, n_slots):
    """The target row map for a switch into ``bucket`` with the live slots ``live``: the identity bucket
    seats every slot on its own row; a compacting bucket keeps the row a live slot already had in it
    (stable) and gives the others the lowest free rows. Pure (no device)."""
    live = sorted(set(int(p) for p in live))
    assert len(live) <= bucket.B, f"{len(live)} live slots do not fit bucket {bucket.id} (B={bucket.B})"
    if bucket.identity:
        return list(range(n_slots))
    rows = [None] * bucket.B
    for phys in live:
        r = bucket.row_of(phys)
        if r is not None and rows[r] is None:
            rows[r] = phys
    for phys in live:
        if phys not in rows:
            rows[rows.index(None)] = phys
    return rows


def build_moves(cur, tgt_rows, mi, active):
    """The whole-batch move list of a switch out of ``cur``: {row_tgt: (row_cur, mi) for a live slot | None}
    with one entry per target row (empty rows get None = identity block + zero window). Pure."""
    moves = {}
    for r, phys in enumerate(tgt_rows):
        if phys is None or not active[phys]:
            moves[r] = None
            continue
        rc = cur.row_of(phys)
        assert rc is not None, f"live slot {phys} has no row in bucket {cur.id}"
        moves[r] = (rc, int(mi[phys]))
    return moves


def switch_realizations(src, dst):
    """Every (row_src, mi) a switch src -> dst can read, as a list of move dicts, plus the all-empty move
    (the warm sweep executes each so every slice offset is compiled before any capture)."""
    out = []
    for mi in range(src.T):
        for c0 in range(0, src.B, dst.B):
            chunk = list(range(src.B))[c0 : c0 + dst.B]
            moves = {r: None for r in range(dst.B)}
            for rd, rs in enumerate(chunk):
                moves[rd] = (rs, mi)
            out.append(moves)
    out.append({r: None for r in range(dst.B)})
    return out


# ---------------------------------------------------------------------------------------------- the decoder
class DFlash2DualBucketDecoder(DFlash2Decoder):
    """DFlash2Decoder whose captures and buffers live for the server's lifetime, with S slots seated on
    the rows of one of several verify BUCKETS (see the module doc). ``buckets``: (B_cfg, T_cfg) pairs;
    the largest B_cfg must equal max_batch_size and all must share B*T (the tap-buf row count)."""

    _multi_bucket = True

    def __init__(
        self,
        model,
        num_blocks,
        buckets=((8, 4), (4, 8)),
        weights_dir=None,
        stop_tokens=None,
        ring_tokens=None,
    ):
        S = int(model.args.max_batch_size)
        nb_v = int(num_blocks)  # width of vLLM's per-request page-table rows (the verify trace's table width)
        bl = [(int(B), int(T)) for B, T in buckets]
        assert bl and len(set(bl)) == len(bl), f"buckets {bl} must be distinct (B, T) pairs"
        for B, T in bl:
            assert 1 <= B <= S, f"bucket {B}x{T}: B must be in [1, {S}] (max_batch_size)"
            assert T >= 2, f"bucket {B}x{T}: T = K+1 must be >= 2"
            assert B * T <= ttnn.TILE_SIZE, f"bucket {B}x{T}: {B * T} verify rows exceed one {ttnn.TILE_SIZE}-row tile"
        assert max(B for B, _ in bl) == S, f"the largest bucket must seat every slot (max_batch_size={S}): {bl}"
        assert len({B * T for B, T in bl}) == 1, f"buckets must share B*T (one set of tap bufs): {bl}"
        single = len(bl) == 1
        K_max = max(T for _, T in bl) - 1
        # The base class asserts S*(K+1) <= 32 and sizes its histograms by K: construct it at the IDENTITY
        # bucket's K (which satisfies the assert by construction) and widen to K_max below.
        K_base = next(T for B, T in bl if B == S) - 1
        # Dummy per-slot tables for the capture: width nb_v, and DISJOINT block sets across slots (the
        # grouped verify KV write asserts that at capture). Slot u names blocks [u*nbu, (u+1)*nbu).
        pt = self._disjoint_tables(S, nb_v)
        super().__init__(model, pt, draft_len=K_base, stop_tokens=stop_tokens, weights_dir=weights_dir)
        assert self.tp, "the serving decoder needs the TP drafter (QWEN36_DFLASH_TP=1)"
        if K_max != K_base:
            self.K = K_max
            self.accept_hist = [0] * (K_max + 1)
            self.depth_hits = [0] * K_max
            # The drafter's DEFAULT block follows K_max (layouts carry their own; this keeps drafter.K == self.K).
            self.drafter = get_drafter(model, self._weights_dir, block=K_max + 1)
        assert self.drafter.K == self.K and K_max <= self.drafter.K_max
        self.n_slots = S
        self.nb_v = nb_v
        self.buckets = {}
        for B, T in sorted(bl):
            bid = DEFAULT_BUCKET if single else bucket_id(B, T)
            self.buckets[bid] = Bucket(
                id=bid,
                B=B,
                T=T,
                K=T - 1,
                identity=(B == S),
                rows=list(range(S)) if B == S else [None] * B,
                tables=torch.zeros(B, nb_v, dtype=torch.int32),
            )
        self.cur = next(b for b in self.buckets.values() if b.identity)
        self.row_tables = torch.zeros(S, nb_v, dtype=torch.int32)  # per-SLOT page-table truth (zeros = idle)
        self._planner = BucketPlanner({b.id: b.B for b in self.buckets.values()}, start=self.cur.id)
        self._ring_req = ring_tokens
        self.ring = None
        self.block_size = None
        self._alloc_done = False
        self._warm_done = False
        self._captured = False
        # Per-slot session state.
        self.active = [False] * S
        self.joined = [False] * S  # ever seeded: its hold inputs are real
        self.fresh = [False] * S  # folded seed: the slot's next step consumes ``first`` (emits com[1:])
        self.p = [0] * S
        self.pending = [0] * S
        self.mi = [0] * S  # accepted-prefix index the slot's NEXT replay commits (0 right after a switch)
        self.iters = [0] * S
        self.accepted = [0] * S
        self.drafted = [0] * S
        self.committed = [0] * S  # tokens step() returned for the slot (excludes the caller-held ``first``)
        self.hist = [[0] * (self.K + 1) for _ in range(S)]
        self._t_begin = [0.0] * S
        self.total_steps = 0
        # Ordering / switch bookkeeping.
        self._planned = False
        self._switched_since_step = False
        self.switch_count = 0
        self.switch_ms = []  # per-switch wall time (host issue time unless QWEN36_SPEC_TIMING / DEBUG fences)
        self._last_switch_step = None

    # ------------------------------------------------------------------ small helpers
    @staticmethod
    def _disjoint_tables(B, nb_v):
        """[B, nb_v] int32 dummy tables with DISJOINT block sets per row (row r names [r*nbu, (r+1)*nbu))."""
        nbu = max(1, nb_v // B)
        return torch.stack(
            [torch.tensor([r * nbu + (i % nbu) for i in range(nb_v)], dtype=torch.int32) for r in range(B)]
        )

    def _dummy_tables(self, b):
        return self._disjoint_tables(b.B, self.nb_v)

    @property
    def K_max(self):
        return self.K

    @property
    def cur_id(self):
        return self.cur.id

    @property
    def bucket_ids(self):
        return list(self.buckets)

    @property
    def tables(self):
        """The CURRENT bucket's live per-row verify tables (single bucket: row == slot, as before)."""
        return self.cur.tables

    @property
    def multi(self):
        return len(self.buckets) > 1

    def row_of(self, phys, bucket_id=None):
        """The row slot ``phys`` occupies in the current (or the given) bucket, or None."""
        b = self.cur if bucket_id is None else self.buckets[bucket_id]
        return b.row_of(int(phys))

    def row_map(self, bucket_id=None):
        b = self.cur if bucket_id is None else self.buckets[bucket_id]
        return list(b.rows)

    def _refresh_tables(self, b):
        if self._captured:
            self.model.refresh_verify_page_tables(b.tables, cfg_id=b.id)

    def _rebuild_tables(self, b):
        for r, phys in enumerate(b.rows):
            b.tables[r] = self.row_tables[phys] if phys is not None else 0

    # ------------------------------------------------------------------ one-time: buffers + programs
    def alloc(self):
        """Allocate every buffer the session traces will bake (before any capture, once)."""
        model = self.model
        assert not self._alloc_done
        S = self.n_slots
        T_max = self.K + 1
        # Identity (weights_dir, tp) only: the buckets' blocks are LAYOUT properties of one drafter.
        self.drafter = get_drafter(model, self._weights_dir)
        assert self.drafter.is_tp and self.K <= self.drafter.K_max, f"K={self.K} > drafter K_max={self.drafter.K_max}"
        self.block_size = bs = get_block_size(model._paged_kv_caches)
        widest = max(self.drafter.windows) if all(self.drafter.windows) else 0
        assert widest, "the serving ring needs an all-windowed drafter (DFlash2); DFlash v1 has a full-attention layer"
        ring = (
            int(self._ring_req) if self._ring_req else -(-(widest + 2 * bs) // bs) * bs
        )  # window + >= 1 block of slack
        assert ring % bs == 0 and ring >= widest + T_max
        self.ring = ring
        nblk = ring // bs
        model._dflash_tap_layers = tuple(self.drafter.taps)
        rows = next(iter(self.buckets.values())).B * next(iter(self.buckets.values())).T  # equal for every bucket
        if model._dflash_tap_bufs is not None and (
            len(model._dflash_tap_bufs) != len(model._dflash_tap_layers) or model._dflash_tap_bufs[0].shape[-2] != rows
        ):
            model.free_dflash_tap_bufs()
        if model._dflash_tap_bufs is None:
            model._dflash_tap_bufs = model.alloc_dflash_tap_bufs(rows)
        ring_pt = torch.stack([torch.arange(u * nblk, (u + 1) * nblk, dtype=torch.int32) for u in range(S)])
        self.drafter.alloc_store(ring_pt, bs, ring_tokens=ring, tap_bufs=model._dflash_tap_bufs)
        for b in self.buckets.values():
            b.layout = self.drafter.add_layout(b.id, b.B, b.T, model._dflash_tap_bufs)
            self.drafter.set_rows(b.layout, b.rows)
        self.ctx_len = [0] * S
        self._alloc_done = True
        logger.info(
            f"[dflash2-serve] allocated: slots={S} buckets={[f'{b.id}(B={b.B},T={b.T})' for b in self.buckets.values()]} "
            f"K_max={self.K} verify table width {self.nb_v}, drafter ring {ring} positions ({nblk} blocks x {bs}) per slot"
        )

    def _prepare_verify(self, b, warm_position):
        """Lane A's prepare_verify_trace for bucket b: buffers + the EAGER verify warm pass (no capture)."""
        model = self.model
        model._dflash_tap = True
        try:
            model.prepare_verify_trace(
                self._dummy_tables(b),
                b.T,
                warm_positions=[int(warm_position)] * b.B,
                decode_cfg=True,
                cfg_id=b.id,
            )
        finally:
            model._dflash_tap = False

    def _capture_verify(self, b, warm_position):
        """Lane A's capture_verify_trace for bucket b. The frozen form takes the cfg id only (prepare ran in
        warm()); the design's longer form also takes the tables / positions -- pass them when accepted."""
        model = self.model
        fn = model.capture_verify_trace
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            params = {}
        model._dflash_tap = True
        try:
            if "page_tables" in params:
                fn(
                    self._dummy_tables(b),
                    b.T,
                    warm_positions=[int(warm_position)] * b.B,
                    decode_cfg=True,
                    cfg_id=b.id,
                )
            else:
                fn(cfg_id=b.id)
        finally:
            model._dflash_tap = False

    def warm(self, warm_position=64):
        """Compile every program the post-capture path needs, EAGERLY and before any capture: every bucket's
        GDN spec cfg and verify cfg (the eager verify pass writes throwaway KV at ``warm_position`` into the
        dummy tables), then the warm sweep (every join seed, every switch realization, every layout's draft /
        extend)."""
        assert self._alloc_done and not self._warm_done and not self._captured
        for b in self.buckets.values():
            for dn in self._gdn:
                dn.prepare_spec_cfg(b.id, b.B, b.T)
        for b in self.buckets.values():
            self._prepare_verify(b, warm_position)
        ttnn.synchronize_device(self.mesh)
        self.warm_sweep()
        self._warm_done = True
        _dbg("warm done")

    def warm_sweep(self):
        """EXECUTE every seed (bucket, row, slot) and every bucket-switch realization (row, mi) in both
        directions, then an eager draft + extend in every layout. Phase 1 compiles them; the caller re-runs
        it after every capture and fails startup if the program cache grew (a spec program compiled with a
        trace parked). Scribbles ring slot 0 / the windows of every bucket: no slot may be ACTIVE."""
        assert self._alloc_done
        assert not any(self.active), "warm_sweep() re-seeds every bucket's slot-0 state: end() every active slot first"
        gdn = self._gdn
        t0 = time.perf_counter()
        n_seed = n_sw = 0
        for b in self.buckets.values():
            pairs = (
                [(p, p) for p in range(self.n_slots)]
                if b.identity
                else [(r, p) for r in range(b.B) for p in range(self.n_slots)]
            )
            for row, phys in pairs:
                for dn in gdn:
                    dn.seed_spec_row(b.id, row, phys)
                n_seed += 1
        for src in self.buckets.values():
            for dst in self.buckets.values():
                if src is dst:
                    continue
                for moves in switch_realizations(src, dst):
                    for dn in gdn:
                        dn.switch_spec_cfg(src.id, dst.id, moves)
                    n_sw += 1
        ttnn.synchronize_device(self.mesh)
        for b in self.buckets.values():
            anchors, Cs = self._draft_inputs(b)  # finite rows; never clobbers an ingested slot's prompt context
            self.drafter.draft(anchors, Cs, layout_id=b.id, traced=False)
            self.drafter.extend_context([0] * b.B, [0] * b.B, layout_id=b.id, traced=False)
        ttnn.synchronize_device(self.mesh)
        logger.info(
            f"[dflash2-serve] warm sweep: {n_seed} seeds, {n_sw} switch realizations, "
            f"{len(self.buckets)} layout draft/extend in {time.perf_counter() - t0:.1f}s"
        )

    def capture(self, warm_position=64):
        """Capture every bucket's verify trace against its dummy per-row tables (the drafter's own traces are
        captured by the first traced step in each layout). ``warm_position``: where the capture pass writes
        when lane A's capture takes positions (else the prepare-time position of warm() applies)."""
        assert self._alloc_done and self._warm_done and not self._captured
        model = self.model
        for b in self.buckets.values():
            self._capture_verify(b, warm_position)
        model._vfy_owner = self
        self._vfy_captured = True
        ttnn.synchronize_device(self.mesh)
        self._captured = True
        # From here on an IDLE / EMPTY row points at block 0 (vLLM's never-allocated null block): its held
        # rows keep replaying, and their attention KV writes must land nowhere a request owns. The dummy
        # disjoint tables were only for the capture-time grouped-write check.
        for b in self.buckets.values():
            b.tables[:] = 0
            model.refresh_verify_page_tables(b.tables, cfg_id=b.id)
        self.row_tables[:] = 0
        logger.info(
            "[dflash2-serve] verify traces captured: "
            + ", ".join(f"{b.id} (B={b.B} x T={b.T})" for b in self.buckets.values())
            + f"; current bucket {self.cur.id}"
        )

    # ------------------------------------------------------------------ per-request context
    def ingest_prompt(self, phys, taps, T, chunk_start=0):
        """Prompt taps of slot phys (5 fractured [1,1,S,dim/tp] device tensors from an eager prefill chunk) ->
        the drafter's ring for slot phys at positions chunk_start..; ``T`` = positions covered after this
        chunk. Frees the taps."""
        assert self._alloc_done
        _dbg(f"ingest slot={phys} T={T} rows={taps[0].shape[-2]} start={chunk_start}")
        self.drafter.fill_context(taps, int(chunk_start), phys=int(phys), valid_len=int(T) - int(chunk_start))
        for t in taps:
            ttnn.deallocate(t)
        self.ctx_len[int(phys)] = int(T)

    def fit_row(self, row):
        """vLLM's page-table row (torch [nb] or [1, nb]) -> [nb_v] int32 (zero-padded / trimmed)."""
        row = torch.as_tensor(row).reshape(-1).to(torch.int32)
        n = row.shape[0]
        if n < self.nb_v:
            row = torch.cat([row, torch.zeros(self.nb_v - n, dtype=torch.int32)])
        elif n > self.nb_v:
            row = row[: self.nb_v]
        return row.contiguous()

    def set_table(self, phys, row):
        """Point slot phys's verify row (if it has one in the current bucket) at this request's blocks
        (host->device restage, no capture). Always records the row as the slot's page-table truth."""
        phys = int(phys)
        fitted = self.fit_row(row)
        if torch.equal(self.row_tables[phys], fitted):
            return
        self.row_tables[phys] = fitted
        r = self.cur.row_of(phys)
        if r is not None:
            self.cur.tables[r] = fitted
            self._refresh_tables(self.cur)

    # ------------------------------------------------------------------ rows
    def _seat(self, b, phys):
        """The row slot phys runs on in bucket b, seating it on the lowest free row of a compacting bucket
        (and restaging that layout's rows) if it has none."""
        r = b.row_of(phys)
        if r is not None:
            return r
        assert not b.identity
        free = b.free_rows()
        assert free, (
            f"bucket {b.id} (B={b.B}) has no free row for slot {phys}: plan(live_after) must run before "
            f"begin() so the bucket in force seats every joining slot"
        )
        r = free[0]
        b.rows[r] = phys
        b.tables[r] = self.row_tables[phys]
        if self._alloc_done:
            self.drafter.set_rows(b.layout, b.rows)
        return r

    def _unseat(self, b, phys):
        """Free slot phys's row in a compacting bucket (the row is held from now on; its tables are zero)."""
        if b.identity:
            return False
        r = b.row_of(phys)
        if r is None:
            return False
        b.rows[r] = None
        b.tables[r] = 0
        if b is self.cur and self._alloc_done:
            self.drafter.set_rows(b.layout, b.rows)
        return True

    def _hold_inputs(self, b):
        """Replay inputs for a step in which every row of bucket b HOLDS (tokens and positions irrelevant:
        verify_traced stages -1 positions and the HOLD ring index for held rows; mi is passed for the
        stepping ones)."""
        mi_prev = [self.mi[phys] if phys is not None else 0 for phys in b.rows]
        return [[0] * b.T for _ in range(b.B)], [0] * b.B, mi_prev

    def _draft_inputs(self, b):
        """(anchors, Cs) per user-row of bucket b's layout, FINITE for every row: a live or active-held slot
        drafts at its real next position (anchor = pending, C = p+1: the block's K/V land at p+1..p+K+1 in
        its ring, the very rows its next real draft rewrites before reading them); the row of an inactive
        slot drafts the dummy at C = max(1, ctx_len) -- past its prompt frontier, so an ingested-not-begun
        slot's context is never clobbered (its first real draft at C = T rewrites those rows first); an
        empty row (no slot) drafts the C=1 dummy into the scratch block."""
        anchors = [0] * b.B
        Cs = [1] * b.B
        for r, phys in enumerate(b.rows):
            if phys is None:
                continue
            if self.active[phys]:
                anchors[r] = self.pending[phys]
                Cs[r] = self.p[phys] + 1
            else:
                Cs[r] = max(1, self.ctx_len[phys])
        return anchors, Cs

    # ------------------------------------------------------------------ join
    def begin(self, phys, first, T, page_table_row):
        """Start slot phys's session after ingest_prompt(phys, ...): seat it on a row of the current bucket
        and seed that row from the slot's prefilled decode state."""
        phys = int(phys)
        assert self._captured, "begin() needs the captured session (warm-up must have run)"
        assert self.ctx_len[phys] == int(
            T
        ), f"slot {phys}: ingest_prompt({T}) must precede begin (ctx_len={self.ctx_len[phys]})"
        assert not self.active[phys], f"slot {phys} is still active; end() it first"
        if self.multi and not self._planned and _DEBUG:
            logger.warning(f"[dflash2-serve] begin(slot {phys}) without plan() this step (bucket {self.cur.id})")
        b = self.cur
        self._t_begin[phys] = time.perf_counter()
        row = self._seat(b, phys)
        self.set_table(phys, page_table_row)
        # Slot phys's durable GDN row (written by its prefill) -> the cfg's ring slot-0 block row*Nv + window
        # row ``row``. Every other row's ring blocks / window rows are untouched.
        for dn in self._gdn:
            dn.seed_spec_row(b.id, row, phys)
        if _FOLD_SEED:
            # No replay: the slot's first step() consumes ``first`` at position T as row 0 of a real
            # iteration (see _FOLD_SEED). p = T-1 keeps ctx_len == p+1 (the drafter holds 0..T-1).
            self.pending[phys] = int(first)
            self.p[phys] = int(T) - 1
            self.fresh[phys] = True
            _dbg(f"seed slot={phys} row={row}/{b.id} T={T} folded into its first step (pending {self.pending[phys]})")
        else:
            tokens, positions, mi_prev = self._hold_inputs(b)
            tokens[row] = [int(first)] * b.T  # row 0 = the seed; rows 1..K are junk the next verify overwrites
            positions[row] = int(T)
            mi_prev[row] = 0
            hold = [r for r in range(b.B) if r != row]
            _dbg(f"seed slot={phys} row={row}/{b.id} T={T} hold={hold}")
            ids, _feed, _ = self.model.verify_traced(
                tokens, positions, mi_prev, read_logits=False, hold=hold, cfg_id=b.id
            )
            self.pending[phys] = int(ids[row * b.T])
            # The verify trace copied its taps: this row's row 0 (position T) -> the slot's drafter ring slot T.
            self.drafter.extend_context(
                [int(T) if r == row else 0 for r in range(b.B)],
                [1 if r == row else 0 for r in range(b.B)],
                layout_id=b.id,
                traced=_DRAFT_TRACED,
            )
            self.ctx_len[phys] = int(T) + 1
            self.p[phys] = int(T)
            self.fresh[phys] = False
            _dbg(f"seed slot={phys} -> pending {self.pending[phys]}")
        self.mi[phys] = 0
        self.active[phys] = self.joined[phys] = True
        self.iters[phys] = self.accepted[phys] = self.drafted[phys] = self.committed[phys] = 0
        self.hist[phys] = [0] * (self.K + 1)

    # ------------------------------------------------------------------ bucket policy
    def plan(self, live_after):
        """Run the bucket hysteresis for the coming step: ``live_after`` = the slots that will be live after
        this step's joins (active slots + prefilled slots about to begin). May switch buckets. Call BEFORE
        any begin() / set_table() / step() of the step. Returns the bucket id in force."""
        live = set(int(x) for x in live_after) | {u for u in range(self.n_slots) if self.active[u]}
        assert all(
            0 <= u < self.n_slots for u in live
        ), f"live_after {sorted(live)} names a slot outside 0..{self.n_slots - 1}"
        n = len(live)
        assert n <= self.n_slots
        if self.multi:
            tgt, direction = self._planner.decide(n, self.total_steps)
            if tgt is not None and tgt != self.cur.id:
                if direction == "down" and self._switched_since_step:
                    _dbg(f"plan: down-switch to {tgt} deferred (already switched this step)")
                else:
                    self.set_bucket(tgt)
        self._planned = True
        return self.cur.id

    def moves_for(self, target_id):
        """(target row map, move dict) a switch to ``target_id`` would use right now. Pure (no device)."""
        tgt = self.buckets[target_id]
        live = [u for u in range(self.n_slots) if self.active[u]]
        rows = assign_rows(tgt, live, self.n_slots)
        return rows, build_moves(self.cur, rows, self.mi, self.active)

    def set_bucket(self, target_id):
        """Switch the session to bucket ``target_id`` (the whole-batch reseed; see the module doc). Requires
        the captured session and every active slot to fit the target. Between steps only."""
        tgt = self.buckets[target_id]
        cur = self.cur
        if tgt is cur:
            return
        assert self._captured, "set_bucket() needs the captured session"
        live = [u for u in range(self.n_slots) if self.active[u]]
        assert len(live) <= tgt.B, f"{len(live)} active slots do not fit bucket {tgt.id} (B={tgt.B})"
        direction = "up" if tgt.B > cur.B else "down"
        t0 = time.perf_counter()
        new_rows, moves = self.moves_for(tgt.id)
        _dbg(f"set_bucket {cur.id} -> {tgt.id}: live={live} rows={new_rows} moves={moves}")
        # STEP 1: the exact state move, per GDN layer (ring blocks + conv window rows; nothing else).
        for dn in self._gdn:
            dn.switch_spec_cfg(cur.id, tgt.id, moves)
        # STEP 2-5: host commit (all-or-nothing after the device ops were issued).
        for u in live:
            self.mi[u] = 0  # every moved slot resumes from the target's slot-0 block / window rows 0..K-1
        tgt.rows = new_rows
        self._rebuild_tables(tgt)
        self.model.refresh_verify_page_tables(tgt.tables, cfg_id=tgt.id)
        self.drafter.set_rows(tgt.layout, tgt.rows)
        self.cur = tgt
        tgt.stats["switches_in"] += 1
        self.switch_count += 1
        self._switched_since_step = True
        self._last_switch_step = self.total_steps
        self._planner.note_switch(tgt.id, direction, self.total_steps)
        if self._timing or _DEBUG:
            ttnn.synchronize_device(self.mesh)  # measure the device time, not the issue time
        dt = (time.perf_counter() - t0) * 1e3
        self.switch_ms.append(dt)
        logger.info(
            f"[dflash2-serve] bucket switch {cur.id} -> {tgt.id} ({direction}) at step {self.total_steps}: "
            f"{len(live)} live slot(s) moved, {dt:.1f} ms{'' if (self._timing or _DEBUG) else ' (issue time)'}; "
            f"planner {self._planner.stats()}"
        )

    # ------------------------------------------------------------------ the loop
    def step(self, only=None):
        """One speculative iteration over the live slots (or over ``only`` those live slots: the others
        HOLD -- replay their last inputs, advance nothing -- which bounds how far a fast slot can run
        ahead of what its caller has consumed and, with it, its KV write reach). Returns
        {slot: committed ids} for the slots that stepped."""
        assert self._captured
        b = self.cur
        B, T, K = b.B, b.T, b.K
        only = None if only is None else set(int(u) for u in only)
        live = [u for u in range(self.n_slots) if self.active[u] and (only is None or u in only)]
        if not live:
            return {}
        for u in live:
            assert self.ctx_len[u] >= self.p[u] + 1, f"slot {u}: context {self.ctx_len[u]} < p+1 {self.p[u] + 1}"
            assert b.row_of(u) is not None, f"active slot {u} has no row in bucket {b.id}"
        # The draft is one R-row forward over every row of the current layout (finite inputs per row: see
        # _draft_inputs). A live slot drafts for real; an active-held slot at its real next position too.
        anchors, Cs = self._draft_inputs(b)
        if not self._armed and _DRAFT_TRACED:
            self.drafter.arm_traces()
            self._armed = True
        drafts = self.drafter.draft(anchors, Cs, layout_id=b.id, traced=_DRAFT_TRACED)
        tokens, positions, mi_prev = self._hold_inputs(b)
        row_of = {}
        for u in live:
            r = b.row_of(u)
            row_of[u] = r
            tokens[r] = [self.pending[u]] + [int(d) for d in drafts[r]]
            positions[r] = self.p[u] + 1
            mi_prev[r] = self.mi[u]
        hold = [r for r in range(B) if r not in row_of.values()]  # every non-stepping row (empty rows included)
        ids, _feed, _ = self.model.verify_traced(tokens, positions, mi_prev, read_logits=False, hold=hold, cfg_id=b.id)
        committed = {}
        slot0 = [0] * B
        nrows = [0] * B
        next_pending = {}
        for u in live:
            r = row_of[u]
            row_ids = ids[r * T : (r + 1) * T]
            m = self._accept_greedy(drafts[r], row_ids)  # updates the shared histogram fields
            com = [self.pending[u]] + [int(d) for d in drafts[r][:m]]
            stop_i = next((i for i, t in enumerate(com) if t in self.stop_tokens), None)
            if stop_i is not None:
                com = com[: stop_i + 1]
            mi_u = len(com) - 1
            # A FRESH slot's com[0] is its seed token ``first`` (folded seed): the caller emitted it at
            # prefill, so it is not returned again; mi / next pending / the extend still use the full com.
            committed[u] = com[1:] if self.fresh[u] else com
            self.fresh[u] = False
            self.committed[u] += len(committed[u])
            next_pending[u] = int(row_ids[mi_u])
            self.mi[u] = mi_u
            slot0[r], nrows[r] = self.p[u] + 1, len(com)
            self.iters[u] += 1
            self.accepted[u] += m
            self.drafted[u] += K
            self.hist[u][m] += 1
            b.stats["accepted"] += m
            b.stats["drafted"] += K
            b.stats["committed"] += len(com)
        self.drafter.extend_context(slot0, nrows, layout_id=b.id, traced=_DRAFT_TRACED)
        for u in live:
            r = row_of[u]
            self.ctx_len[u] = slot0[r] + nrows[r]
            self.p[u] += nrows[r]
            self.pending[u] = next_pending[u]
        self.total_steps += 1
        b.stats["steps"] += 1
        self._planned = False
        self._switched_since_step = False
        return committed

    def end(self, phys):
        """Slot phys's request is done: its row keeps holding (identity bucket) or is freed (compacting
        bucket); its blocks go back to vLLM now, so no held row may touch them any more."""
        phys = int(phys)
        if not self.active[phys]:
            return
        self.active[phys] = False
        zeros = torch.zeros(self.nb_v, dtype=torch.int32)
        changed = not torch.equal(self.row_tables[phys], zeros)
        self.row_tables[phys] = 0
        r = self.cur.row_of(phys)
        if r is not None:
            changed = changed or bool(self.cur.tables[r].any())
            self.cur.tables[r] = 0
        for b in self.buckets.values():
            self._unseat(b, phys)
        if changed:
            self._refresh_tables(self.cur)
        if self.iters[phys]:
            dt = time.perf_counter() - self._t_begin[phys]
            n_tok = self.committed[phys]  # == iters + accepted, minus the folded seed's ``first`` if any
            logger.info(
                f"[dflash2-serve] slot {phys} session: {self.iters[phys]} iters, {n_tok} committed tokens, "
                f"accept {self.accepted[phys] / self.iters[phys]:.2f}/{self.K} -> {n_tok / self.iters[phys]:.2f} tok/iter, "
                f"{n_tok / max(dt, 1e-6):.1f} tok/s in-model (incl. seed); histogram {self.hist[phys]}"
                + (f"; bucket {self.cur.id}" if self.multi else "")
            )

    # ------------------------------------------------------------------ statistics
    def reset_bucket_policy(self):
        """Clean hysteresis: a fresh planner (env-derived down_steps / cooldown, starting in the CURRENT bucket) and
        zeroed switch counters / per-bucket stats. The warm-up's dummy sessions and the post-capture guard drive real
        switches (and a test's direct set_bucket() churn does too), which back the planner's cooldown off; a server
        calls this once before its first request so no synthetic churn lingers for its life. Host-only."""
        self._planner = BucketPlanner({b.id: b.B for b in self.buckets.values()}, start=self.cur.id)
        self.switch_count = 0
        self.switch_ms = []
        self._last_switch_step = None
        for b in self.buckets.values():
            for k in b.stats:
                b.stats[k] = 0

    def bucket_stats(self):
        """Per-bucket counters + switch statistics (for logs and tests)."""
        out = {b.id: dict(b.stats) for b in self.buckets.values()}
        out["_switches"] = {
            "count": self.switch_count,
            "ms": list(self.switch_ms),
            "mean_ms": (sum(self.switch_ms) / len(self.switch_ms)) if self.switch_ms else None,
            "planner": self._planner.stats(),
            "cur": self.cur.id,
        }
        return out

    def log_bucket_stats(self):
        st = self.bucket_stats()
        for b in self.buckets.values():
            s = st[b.id]
            acc = f"{s['accepted'] / s['drafted']:.3f}" if s["drafted"] else "n/a"
            per = f"{s['committed'] / s['steps']:.2f}" if s["steps"] else "n/a"
            logger.info(
                f"[dflash2-serve] bucket {b.id} (B={b.B}, K={b.K}): {s['steps']} steps, {s['committed']} committed, "
                f"accept {acc}, {per} tok/step, switched into {s['switches_in']}x"
            )
        sw = st["_switches"]
        if sw["count"]:
            logger.info(
                f"[dflash2-serve] {sw['count']} bucket switch(es), mean {sw['mean_ms']:.1f} ms "
                f"(min {min(sw['ms']):.1f}, max {max(sw['ms']):.1f}); planner {sw['planner']}"
            )

    # ------------------------------------------------------------------ test hooks (device reads)
    def _spec_cfg_window(self, dn, cfg_id):
        """The cfg's CURRENT conv window E_prev: the fused path's pair half at its parity, else the composite buffer."""
        cfg = dn.spec_cfg(cfg_id)
        return cfg.win_pair[cfg.parity] if getattr(cfg, "win_pair", None) is not None else cfg.win_buf

    def snapshot_row(self, bucket_id, row, gdn_layers=None, mi=0, kv_blocks=None):
        """Bytes a bucket ROW's spec state consists of, as host tensors: per GDN layer the ring block
        (mi*B + row)*Nv .. +Nv (fp32) and the current conv window rows E_prev[row, mi:mi+Kc] (bf16), plus
        the first full-attention layer's K blocks named in ``kv_blocks`` (if given). ``gdn_layers``: indices
        into self._gdn (default: all)."""
        b = self.buckets[bucket_id]
        assert 0 <= row < b.B and 0 <= mi < b.T
        layers = range(len(self._gdn)) if gdn_layers is None else list(gdn_layers)
        rings, wins = [], []
        for li in layers:
            dn = self._gdn[li]
            Nv, Kc = dn.Nv, dn.K
            blk = (mi * b.B + row) * Nv
            rings.append(ttnn.to_torch(ttnn.get_device_tensors(dn._spec_ring)[0])[blk : blk + Nv].clone())
            win = self._spec_cfg_window(dn, b.id)
            wins.append(ttnn.to_torch(ttnn.get_device_tensors(win)[0])[row, mi : mi + Kc].clone())
        out = {"bucket": b.id, "row": row, "mi": mi, "ring": rings, "win": wins}
        if kv_blocks is not None:
            att0 = next(layer.attention for layer in self.model.layers if layer.is_full_attention)
            blocks = [int(x) for x in kv_blocks]
            out["kv"] = ttnn.to_torch(ttnn.get_device_tensors(att0.paged_k)[0])[blocks].clone()
        return out

    def snapshot(self, phys, gdn_layers=None, with_kv=True):
        """snapshot_row for slot phys's row in the CURRENT bucket at its current mi (an active slot), with
        its attention K blocks (the non-zero entries of row_tables[phys]) when ``with_kv``."""
        phys = int(phys)
        r = self.cur.row_of(phys)
        assert r is not None, f"slot {phys} has no row in bucket {self.cur.id}"
        blocks = None
        if with_kv:
            blocks = [int(x) for x in self.row_tables[phys].tolist() if int(x) != 0]
        out = self.snapshot_row(self.cur.id, r, gdn_layers=gdn_layers, mi=self.mi[phys], kv_blocks=blocks)
        out["phys"] = phys
        return out

    # ------------------------------------------------------------------ shutdown
    def release(self):
        model = self.model
        for u in range(self.n_slots):
            self.active[u] = False
        try:
            ttnn.synchronize_device(self.mesh)
        except Exception:
            pass
        if self.multi or self.switch_count:
            try:
                self.log_bucket_stats()
            except Exception as e:
                logger.warning(f"[dflash2-serve] bucket stats failed: {e!r}")
        try:
            self.drafter.free()
        except Exception as e:
            logger.warning(f"[dflash2-serve] drafter free failed: {e!r}")
        try:
            model.release_verify_trace()  # every cfg
        except Exception as e:
            logger.warning(f"[dflash2-serve] verify trace release failed: {e!r}")
        self._vfy_captured = False
        self._captured = False
        model._free_dflash_eager_taps()
        model.free_dflash_tap_bufs()
        model._dflash_tap = False
        self._alloc_done = False
        self._warm_done = False


class DFlash2ServingDecoder(DFlash2DualBucketDecoder):
    """Today's single-bucket serving decoder: S slots x one verify geometry (K = draft_len, default the
    checkpoint's / QWEN36_DFLASH_BLOCK's), row == slot. Exactly DFlash2DualBucketDecoder with
    buckets=((max_batch_size, K+1),) and bucket id 'default'."""

    def __init__(self, model, num_blocks, draft_len=None, weights_dir=None, stop_tokens=None, ring_tokens=None):
        K = default_draft_len(weights_dir) if draft_len is None else int(draft_len)
        B = int(model.args.max_batch_size)
        super().__init__(
            model,
            num_blocks,
            buckets=((B, K + 1),),
            weights_dir=weights_dir,
            stop_tokens=stop_tokens,
            ring_tokens=ring_tokens,
        )


def dummy_prompt(S, seed=1234):
    """S plain text token ids for warmup prefills (no multimodal placeholder ids: they are > 200000)."""
    g = torch.Generator().manual_seed(seed + S)
    return torch.randint(100, 20000, (1, S), generator=g, dtype=torch.int32)


def serve_block_size():
    """Tokens one vLLM decode step commits in spec mode (QWEN36_DFLASH_SERVE_BLOCK; 1 = spec off)."""
    return max(1, int(os.environ.get("QWEN36_DFLASH_SERVE_BLOCK", "32")))

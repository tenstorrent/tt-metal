# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only (no device) tests for the dual-bucket serving plan (profiles/dual_bucket_spec.json).

  1. The bucket HYSTERESIS (dflash2_serving.BucketPlanner, device-free): up is forced the moment the live set
     exceeds the current bucket; down needs DOWN_STEPS consecutive plan() calls that fit the smaller bucket AND
     the cooldown since the last switch; the cooldown doubles (cap 64) when an up-switch follows a down-switch
     within 2x cooldown and halves back after a quiet spell; never two switches per call; adversarial 4<->5 churn
     is bounded.
  2. 4x8 ROW ASSIGNMENT (assign_rows): stable while live, lowest-free at a join / switch-down, freed at end, a full
     bucket refuses; the whole-batch MOVE LIST (build_moves) in both directions, including the shared-ring aliasing
     case the spec's concat-before-write protocol exists for; the warm sweep's switch realizations cover every
     (row_src, mi) in both directions.
  3. QWEN36_DFLASH_BUCKETS parsing + the per-bucket startup checks of the vLLM class (qwen36_vllm_dflash.parse_buckets
     / check_buckets), and agreement with the decoder's own parser.
  4. decode_forward ORDERING with a recording fake decoder: dec.plan(live_after) runs BEFORE any begin() /
     set_table() / step() of a decode step, with live_after = the active sessions + the prefilled slots that begin
     in that step (composed through the plugin's slot_remap); rows without a session are excluded.
  5. TOKEN ACCOUNTING across bucket switches: a fake decoder that models the served decoder's per-slot host state
     (pending / p / ctx_len / mi, folded seed) commits deterministic per-request trajectories and switches buckets
     in plan() (forced up, coin-flip down); every request's emitted ids must be its trajectory, in order, nothing
     lost or duplicated, while requests join and leave -- and a switch must touch nothing but mi.
  6. The gdn_spec_step ring contract at BOTH geometries (4,8) and (8,4) with Nv=12: every (row, head) stays in its
     own lane inside the ONE shared [32*Nv] ring, the switch's source-block formula matches spec_state_blk_idx, and
     the per-cfg ctrl pages differ in size.

The vLLM class imports vllm at module level; on a box without the serving venv the three vllm names it touches (a
protocol base class, three processor classes, a decorator registry) are shimmed in sys.modules -- nothing from
vllm is exercised here. ttnn must import (no device is opened).

Run: pytest models/demos/blackhole/qwen36/tests/test_dual_bucket_planner.py -q
"""

import os
import random
import sys
import types
from types import SimpleNamespace

import pytest
import torch

# The vLLM class reads these at import (module constants): the ragged contract is what the multi-bucket profile runs.
os.environ.setdefault("QWEN36_DRAFTER", "dflash2")
os.environ["QWEN36_DFLASH_RAGGED"] = "1"
os.environ.setdefault("QWEN36_DFLASH_SERVE_BLOCK", "32")

from models.demos.blackhole.qwen36.tests.test_spec_batch_helpers import _mi_cases  # noqa: E402
from models.demos.blackhole.qwen36.tt import dflash2_serving as serving  # noqa: E402
from models.demos.blackhole.qwen36.tt.dflash2_serving import (  # noqa: E402
    DEFAULT_BUCKET,
    Bucket,
    BucketPlanner,
    assign_rows,
    bucket_id,
    build_moves,
    switch_realizations,
)
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_page, spec_ctrl_words, spec_state_blk_idx  # noqa: E402

S = 8  # physical slots (max_num_seqs of the batch8-dflash2 profile)
CAPS = {"8x4": 8, "4x8": 4}
NV = 12  # linear_num_value_heads 48 / TP 4


# ------------------------------------------------------------------------------------------------ vllm shim
def _shim_vllm():
    """sys.modules stand-ins for the vllm names qwen36_vllm(.py|_dflash.py) import at module level."""
    if "vllm" in sys.modules:
        return

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    class _Registry:
        def register_processor(self, *a, **k):
            return lambda cls: cls

    mod("vllm")
    mod("vllm.model_executor")
    mod("vllm.model_executor.models")
    mod("vllm.model_executor.models.interfaces", SupportsMultiModal=type("SupportsMultiModal", (), {}))
    mod(
        "vllm.model_executor.models.qwen3_5",
        Qwen3_5ProcessingInfo=type("Qwen3_5ProcessingInfo", (), {}),
        Qwen3VLDummyInputsBuilder=type("Qwen3VLDummyInputsBuilder", (), {}),
        Qwen3VLMultiModalProcessor=type("Qwen3VLMultiModalProcessor", (), {}),
    )
    mod("vllm.multimodal", MULTIMODAL_REGISTRY=_Registry())


def _import_vllm_dflash():
    try:
        import vllm  # noqa: F401
    except ImportError:
        _shim_vllm()
    from models.demos.blackhole.qwen36.tt import qwen36_vllm_dflash as m

    # The module reads QWEN36_DFLASH_RAGGED (and the other spec knobs) at import time. If an earlier test in the
    # same session imported it before this file set RAGGED=1, its _RAGGED is stale -- reload so the multi-bucket
    # ragged contract this file drives is in force regardless of collection order.
    if os.environ.get("QWEN36_DFLASH_RAGGED") == "1" and not getattr(m, "_RAGGED", False):
        import importlib

        m = importlib.reload(m)
    return m


try:
    vd = _import_vllm_dflash()
    _VD_ERR = None
except Exception as e:  # pragma: no cover - environment without the model stack
    vd = None
    _VD_ERR = repr(e)

requires_vd = pytest.mark.skipif(vd is None, reason=f"qwen36_vllm_dflash not importable host-side: {_VD_ERR}")


# ------------------------------------------------------------------------------------------------ 1. hysteresis
def _drive(planner, ns, start_step=0):
    """decide()+note_switch() over a live-count trace; returns [(step, from, to, direction)] of the switches."""
    switches = []
    for k, n in enumerate(ns):
        step = start_step + k
        src = planner.cur
        tgt, direction = planner.decide(n, step)
        assert (tgt is None) == (direction is None)
        if tgt is not None:
            planner.note_switch(tgt, direction, step)
            switches.append((step, src, tgt, direction))
    return switches


def test_planner_forces_up_the_moment_the_bucket_is_too_small(expect_error):
    p = BucketPlanner(CAPS, start="4x8", down_steps=8, cooldown=16)
    assert p.decide(4, 0) == (None, None), "4 live fit 4x8: nothing to do"
    assert p.decide(5, 1) == ("8x4", "up"), "a 5th live slot cannot be seated in 4x8: up is FORCED"
    # ...regardless of cooldown / consecutive-call bookkeeping (a fresh planner has neither).
    p.note_switch("8x4", "up", 1)
    assert p.cur == "8x4" and p.n_up == 1
    with expect_error(AssertionError, "exceed the largest bucket"):
        p.decide(9, 2)  # more than the largest bucket seats: a planner-level impossibility


def test_planner_down_needs_consecutive_fits_and_the_cooldown():
    p = BucketPlanner(CAPS, start="8x4", down_steps=8, cooldown=16)
    # 7 fitting calls, then a 5th user: the run resets.
    assert _drive(p, [2] * 7 + [5]) == []
    assert p.consec_small == 0
    # 8 consecutive fitting calls -> down (never switched before: no cooldown applies).
    sw = _drive(p, [1] * 8, start_step=8)
    assert sw == [(15, "8x4", "4x8", "down")]
    # Forced up right away; the up-after-down within 2x cooldown doubles the cooldown (16 -> 32).
    sw = _drive(p, [5], start_step=16)
    assert sw == [(16, "4x8", "8x4", "up")]
    assert p.cooldown == 32
    # Down again only once 32 steps have passed since that switch (not after 8 fitting calls).
    sw = _drive(p, [1] * 40, start_step=17)
    assert len(sw) == 1 and sw[0][3] == "down"
    assert sw[0][0] - 16 == 32, f"down-switch after {sw[0][0] - 16} steps, cooldown was 32"


def test_planner_never_switches_twice_per_call_and_lands_in_the_required_bucket():
    rng = random.Random(7)
    p = BucketPlanner(CAPS, start="8x4", down_steps=3, cooldown=4)
    for step in range(2000):
        n = rng.choice([0, 1, 2, 3, 4, 4, 5, 6, 7, 8])
        tgt, direction = p.decide(n, step)
        if tgt is not None:
            p.note_switch(tgt, direction, step)
        # invariants after every call
        assert CAPS[p.cur] >= n, f"step {step}: {n} live in {p.cur} (seats {CAPS[p.cur]})"
        assert direction != "up" or tgt == p.required(n)
        assert direction != "down" or CAPS[tgt] < CAPS[p.cur] or tgt == p.cur
    assert p.n_up > 10 and p.n_down > 10


def test_planner_backoff_bounds_adversarial_churn_then_decays():
    p = BucketPlanner(CAPS, start="8x4", down_steps=8, cooldown=16, cooldown_cap=64, decay_steps=64)
    # Adversarial 4 <-> 5: the moment we sit in 4x8 a 5th user arrives; once in 8x4 it leaves again.
    switches = []
    for step in range(600):
        n = 5 if p.cur == "4x8" else 4
        tgt, direction = p.decide(n, step)
        if tgt is not None:
            p.note_switch(tgt, direction, step)
            switches.append((step, direction))
    assert p.cooldown == 64, f"the cooldown should have backed off to the cap, is {p.cooldown}"
    # Without the backoff a cycle costs 1 + max(8, 16) steps -> ~70 switches in 600; with it the cycle grows to 65.
    assert len(switches) < 30, f"{len(switches)} switches in 600 steps of churn"
    downs = [s for s, d in switches if d == "down"]
    gaps = [b - a for a, b in zip(downs, downs[1:])]
    assert gaps and gaps[-1] >= 64 and gaps == sorted(gaps), f"down-switch gaps must grow with the backoff: {gaps}"
    # Decay: sitting in 8x4 with 4 live, the cooldown halves after decay_steps quiet steps and the down goes through.
    last = switches[-1][0]
    assert p.cur == "8x4"
    sw = _drive(p, [4] * 70, start_step=last + 1)
    assert len(sw) == 1 and sw[0][3] == "down" and sw[0][0] - last == 64
    assert p.cooldown == 32
    # An up-switch long after the down (> 2x cooldown) does not back off again.
    sw = _drive(p, [5], start_step=sw[0][0] + 200)
    assert len(sw) == 1 and sw[0][3] == "up" and p.cooldown == 32
    up_step = sw[0][0]
    # NOTE (BucketPlanner semantics): the decay is evaluated only while a smaller bucket fits, and the down-switch
    # fires as soon as quiet >= cooldown -- so with cooldown (32) < decay_steps (64) the down pre-empts the decay and
    # the cooldown stays at 32; it only halves from the cap (64 == decay_steps), as above. Pinned here so a change
    # of that policy is a deliberate one.
    sw = _drive(p, [3] * 70, start_step=up_step + 1)
    assert len(sw) == 1 and sw[0][3] == "down" and sw[0][0] - up_step == 32
    assert p.cooldown == 32


def test_planner_env_defaults_and_stats():
    p = BucketPlanner(CAPS, start="8x4")
    assert p.down_steps == int(os.environ.get("QWEN36_DFLASH_BUCKET_DOWN_STEPS", 8))
    assert p.cooldown == int(os.environ.get("QWEN36_DFLASH_BUCKET_COOLDOWN", 16))
    st = p.stats()
    assert st["cur"] == "8x4" and st["n_up"] == 0 and st["n_down"] == 0


# ------------------------------------------------------------------------------------------------ 2. rows / moves
def _bucket(B, T, rows=None):
    return Bucket(
        id=bucket_id(B, T),
        B=B,
        T=T,
        K=T - 1,
        identity=(B == S),
        rows=list(range(S)) if B == S else (list(rows) if rows is not None else [None] * B),
        tables=torch.zeros(B, 4, dtype=torch.int32),
    )


def test_assign_rows_is_stable_lowest_free_and_refuses_a_full_bucket(expect_error):
    big, small = _bucket(8, 4), _bucket(4, 8)
    assert assign_rows(big, {5, 2, 7}, S) == list(range(S)), "the identity bucket seats every slot on its own row"
    rows = assign_rows(small, {5, 2}, S)
    assert rows == [2, 5, None, None], "a first join takes the lowest free rows (in slot order)"
    small.rows = rows
    rows = assign_rows(small, {5, 2, 7}, S)
    assert rows == [2, 5, 7, None], "seated slots keep their rows (stable); the newcomer gets the lowest free"
    small.rows = rows
    rows = assign_rows(small, {5, 7}, S)
    assert rows == [None, 5, 7, None], "a slot that left frees its row; the others do not move"
    small.rows = rows
    rows = assign_rows(small, {5, 7, 1}, S)
    assert rows == [1, 5, 7, None], "the freed row is the lowest free for the next join"
    with expect_error(AssertionError, "do not fit"):
        assign_rows(small, {0, 1, 2, 3, 4}, S)


def test_build_moves_both_directions_cover_live_slots_once_and_expose_the_ring_alias(expect_error):
    active = [False] * S
    for u in (0, 5, 2):
        active[u] = True
    mi = {0: 3, 5: 1, 2: 0}
    # UP: cur 4x8 rows [0, 5, 2, None] -> tgt 8x4 (identity rows).
    cur = _bucket(4, 8, rows=[0, 5, 2, None])
    tgt_rows = assign_rows(_bucket(8, 4), [0, 5, 2], S)
    moves = build_moves(cur, tgt_rows, mi, active)
    assert moves == {0: (0, 3), 1: None, 2: (2, 0), 3: None, 4: None, 5: (1, 1), 6: None, 7: None}
    assert sorted(r for r, m in moves.items() if m is not None) == [0, 2, 5], "one target row per live slot"
    # The shared-ring alias the protocol's concat-before-write exists for: a live source block (mi*B_cur + row_cur)
    # coincides with a target slot-0 block (row_tgt) that the single slice_write rewrites.
    src_blocks = {m[1] * cur.B + m[0] for m in moves.values() if m is not None}
    tgt_blocks = set(range(len(tgt_rows)))
    assert src_blocks & tgt_blocks, f"expected an aliased block between sources {src_blocks} and targets {tgt_blocks}"
    assert (
        1 * 4 + 1 in src_blocks and 5 in tgt_blocks
    ), "the spec's example: cur 4x8 row 1 at mi 1 = block 5 = 8x4 row 5"
    # DOWN: cur 8x4 (identity) -> tgt 4x8 with the previous 4x8 seating remembered (stable rows).
    cur = _bucket(8, 4)
    small = _bucket(4, 8, rows=[0, 5, 2, None])
    mi = {0: 2, 5: 3, 2: 0}
    tgt_rows = assign_rows(small, [0, 5, 2], S)
    assert tgt_rows == [0, 5, 2, None]
    moves = build_moves(cur, tgt_rows, mi, active)
    assert moves == {0: (0, 2), 1: (5, 3), 2: (2, 0), 3: None}
    # An ended slot still seated in the remembered rows is NOT moved (build_moves checks active).
    active[5] = False
    moves = build_moves(cur, tgt_rows, mi, active)
    assert moves[1] is None
    with expect_error(AssertionError, "has no row"):
        build_moves(_bucket(4, 8, rows=[None] * 4), [0, None, None, None], {0: 0}, [True] + [False] * 7)


@pytest.mark.parametrize("src,dst", [((4, 8), (8, 4)), ((8, 4), (4, 8))])
def test_switch_realizations_cover_every_row_and_mi(src, dst):
    sb, db = _bucket(*src), _bucket(*dst)
    reals = switch_realizations(sb, db)
    seen = set()
    for moves in reals:
        assert sorted(moves) == list(range(db.B)), "one entry per target row"
        for m in moves.values():
            if m is not None:
                rs, mi = m
                assert 0 <= rs < sb.B and 0 <= mi < sb.T
                seen.add((rs, mi))
    assert seen == {(r, mi) for r in range(sb.B) for mi in range(sb.T)}, "every (row_src, mi) slice offset warmed"
    assert any(all(m is None for m in moves.values()) for moves in reals), "the all-empty move is warmed too"


def test_bucket_ids_and_default():
    assert bucket_id(8, 4) == "8x4" and bucket_id(4, 8) == "4x8"
    assert DEFAULT_BUCKET == "default"
    assert serving.parse_buckets("8x4,4x8") == ((8, 4), (4, 8))
    assert serving.parse_buckets("", default=((4, 8),)) == ((4, 8),)


# ------------------------------------------------------------------------------------------------ 3. env parsing / checks
@requires_vd
def test_parse_buckets_and_agreement_with_the_decoder_parser(expect_error):
    assert vd.parse_buckets("8x4,4x8") == ((8, 4), (4, 8))
    assert vd.parse_buckets(" 8X4 ; 4x8 ,") == ((8, 4), (4, 8))
    assert vd.parse_buckets("") == () and vd.parse_buckets(None) == ()
    assert vd.parse_buckets("8x4,4x8") == serving.parse_buckets("8x4,4x8")
    for bad in ("8", "8x", "x4", "8x4x2", "8-4", "ax4"):
        with expect_error(RuntimeError, "BxT"):
            vd.parse_buckets(bad)


@requires_vd
def test_check_buckets_accepts_the_profile_and_the_single_bucket_defaults():
    assert vd.check_buckets(((8, 4), (4, 8)), 8, 7, True) == ((8, 4), (4, 8))
    assert vd.check_buckets(((4, 8),), 4, 7, False) == ((4, 8),)  # batch4-dflash2: one bucket, ragged optional
    assert vd.check_buckets(((1, 8),), 1, 7, False) == ((1, 8),)  # single-user-dflash2
    assert vd.check_buckets(((8, 4),), 8, 3, True) == ((8, 4),)  # the committed K=3 batch8 profile
    assert vd.bucket_id((8, 4)) == "8x4"


@requires_vd
@pytest.mark.parametrize(
    "buckets,max_num_seqs,k_default,ragged,match",
    [
        ((), 8, 7, True, "no verify bucket"),
        (((8, 5),), 8, 7, True, "exceed one 32-row"),
        (((8, 4), (4, 8)), 4, 7, True, "more users than"),
        (((8, 4), (4, 8)), 8, 3, True, "drafts at most K=3"),
        (((8, 4), (4, 8)), 8, 7, False, "RAGGED"),
        (((8, 4), (8, 2)), 8, 7, True, "distinct B"),
        (((8, 4), (4, 4)), 8, 7, True, "share B x T"),
        (((4, 8), (2, 16)), 8, 15, True, "largest bucket seats 4"),
        (((8, 1),), 8, 7, True, "T >= 2"),
        (((1, 17),), 1, 16, True, "lookahead bound"),
    ],
)
def test_check_buckets_rejects(buckets, max_num_seqs, k_default, ragged, match, expect_error):
    with expect_error(RuntimeError, match):
        vd.check_buckets(buckets, max_num_seqs, k_default, ragged)


# ------------------------------------------------------------------------------------------------ 4./5. fake decoders
class _RecordingDec:
    """A decoder double that records the ORDER of plan / begin / set_table / step / end calls and commits one
    fresh id per stepping slot (the plan-placement contract, nothing else)."""

    def __init__(self, S, cur_id="8x4"):
        self.S = S
        self.active = [False] * S
        self.ctx_len = [0] * S
        self.calls = []
        self.cur_id = cur_id
        self.K_max = 7
        self._n = [0] * S

    def plan(self, live_after):
        self.calls.append(("plan", frozenset(live_after)))
        return self.cur_id

    def begin(self, u, first, T, row):
        self.calls.append(("begin", u))
        assert not self.active[u]
        self.active[u] = True

    def set_table(self, u, row):
        self.calls.append(("set_table", u))

    def step(self, only=None):
        live = [u for u in range(self.S) if self.active[u] and (only is None or u in only)]
        self.calls.append(("step", tuple(live)))
        out = {}
        for u in live:
            self._n[u] += 1
            out[u] = [u * 1000 + self._n[u]]
        return out

    def end(self, u):
        self.calls.append(("end", u))
        self.active[u] = False


def _vllm_obj(dec, S, eos=151645, vocab=248320):
    """A Qwen36DFlashForCausalLM without its __init__ (no model, no device): the fields decode_forward reads."""
    cls = vd.Qwen36DFlashForCausalLM
    obj = cls.__new__(cls)
    obj._spec, obj._spec_pre, obj._in_warmup = dec, None, False
    obj.data_parallel = 1  # Generator.__del__ reads it (no __init__ ran)
    obj.model = [SimpleNamespace(vocab_size=vocab)]
    obj._eos, obj._eos_fill = {eos}, eos
    obj._B = S
    obj._phys = list(range(S))
    obj._pending = [None] * S
    obj._carry = [[] for _ in range(S)]
    obj._stopped = [False] * S
    obj._prev_tail = [None] * S
    obj._anchor_warned = obj._oov_warned = obj._nosession_warned = False
    obj._buckets, obj._multi_bucket, obj._last_bucket = ((S, 4), (S // 2, 8)), True, None
    return obj


@requires_vd
def test_decode_forward_plans_before_any_begin_set_table_or_step():
    assert vd._W > 1 and vd._RAGGED, "this test drives the ragged multi-bucket profile's contract"
    dec = _RecordingDec(S)
    obj = _vllm_obj(dec, S)
    nb = 16
    dec.active[0] = dec.active[1] = True  # live sessions
    obj._pending[2] = (130, torch.arange(nb, dtype=torch.int32))  # prefilled this engine step
    obj._pending[5] = (147, torch.arange(nb, dtype=torch.int32))
    tokens = torch.zeros(S, 1, dtype=torch.int32)
    start_pos = torch.tensor([200, 210, 130, -1, -1, 147, 300, -1])  # rows 3/4/7 padding; row 6 live, no session
    page_table = torch.arange(S * nb, dtype=torch.int32).reshape(S, nb)
    out = obj.decode_forward(tokens=tokens, start_pos=start_pos, page_table=page_table)
    kinds = [c[0] for c in dec.calls]
    assert kinds[0] == "plan" and kinds.count("plan") == 1, kinds
    assert dec.calls[0][1] == frozenset({0, 1, 2, 5}), "live_after = active sessions + slots that begin this step"
    dev_calls = [i for i, k in enumerate(kinds) if k in ("begin", "set_table", "step")]
    assert dev_calls and min(dev_calls) > 0, "plan() precedes every begin/set_table/step"
    assert [c[1] for c in dec.calls if c[0] == "begin"] == [2, 5]
    assert [c[1] for c in dec.calls if c[0] == "set_table"] == [0, 1]
    steps = [i for i, k in enumerate(kinds) if k == "step"]
    assert steps and steps[0] > max(i for i, k in enumerate(kinds) if k in ("begin", "set_table"))
    assert dec.calls[steps[0]][1] == (0, 1, 2, 5), "one ragged iteration over every live row"
    # The ragged block: one real id per live row, -1 padding; the session-less live row ends with EOS.
    assert out.shape == (S, vd._W)
    for i in (0, 1, 2, 5):
        assert int(out[i, 0]) >= 0 and bool((out[i, 1:] == vd._PAD).all())
    assert int(out[6, 0]) == obj._eos_fill and bool((out[6, 1:] == vd._PAD).all())
    for i in (3, 4, 7):
        assert bool((out[i] == vd._PAD).all())
    assert obj._pending[2] is None and obj._pending[5] is None
    assert obj._last_bucket == "8x4"


@requires_vd
def test_decode_forward_live_after_is_composed_through_slot_remap(expect_error):
    S4 = 4
    dec = _RecordingDec(S4, cur_id="4x8")
    obj = _vllm_obj(dec, S4)
    dec.active[3] = True
    obj._pending[1] = (100, torch.zeros(8, dtype=torch.int32))
    remap = [3, 1, 0, 2]  # row 0 now reads the state that was at slot 3; row 1 slot 1
    obj.decode_forward(
        tokens=torch.zeros(S4, 1, dtype=torch.int32),
        start_pos=torch.tensor([500, 100, -1, -1]),
        page_table=torch.zeros(S4, 8, dtype=torch.int32),
        slot_remap=torch.tensor(remap),
    )
    assert obj._phys == remap
    assert dec.calls[0] == ("plan", frozenset({3, 1})), "live_after names PHYSICAL slots after the remap"
    assert [c[1] for c in dec.calls if c[0] == "begin"] == [1]
    with expect_error(RuntimeError, "not a permutation"):
        obj.decode_forward(
            tokens=torch.zeros(S4, 1, dtype=torch.int32),
            start_pos=torch.tensor([-1] * S4),
            page_table=None,
            slot_remap=torch.tensor([0, 0, 1, 2]),
        )


@requires_vd
def test_decode_forward_without_a_planner_is_todays_path():
    """A decoder without plan() (no bucket policy) is driven exactly as before: begins / steps, no plan call."""

    class _NoPlan(_RecordingDec):
        plan = None

    dec = _NoPlan(S)
    obj = _vllm_obj(dec, S)
    dec.active[0] = True
    obj.decode_forward(
        tokens=torch.zeros(S, 1, dtype=torch.int32),
        start_pos=torch.tensor([10] + [-1] * (S - 1)),
        page_table=torch.zeros(S, 4, dtype=torch.int32),
    )
    assert [c[0] for c in dec.calls] == ["set_table", "step"]
    assert obj._last_bucket is None


class _SimDec(_RecordingDec):
    """Models the served decoder's per-slot HOST state (pending / p / ctx_len / mi, folded seed) with a deterministic
    per-request trajectory; step() commits like the real one (random acceptance in the current bucket's K; a FRESH
    slot omits com[0] == first); plan() switches buckets the way the served planner can in the extreme (forced up
    above the small bucket, a coin-flip down otherwise). A switch touches nothing but mi (the spec's STEP 2)."""

    def __init__(self, S, small_B, rng):
        super().__init__(S, cur_id=f"{S}x4")
        self.big_id, self.small_id = f"{S}x4", f"{small_B}x8"
        self.Ks = {self.big_id: 3, self.small_id: 7}
        self.small_B = small_B
        self.rng = rng
        self.p = [0] * S
        self.pending = [0] * S
        self.mi = [0] * S
        self.fresh = [False] * S
        self.gen = [0] * S  # request generation per slot (a re-used slot gets a new trajectory)
        self.idx = [0] * S  # trajectory index of ``pending``
        self.switches = []

    @staticmethod
    def traj(u, gen, i):
        return 100 + (u * 7919 + gen * 104729 + i * 31) % 200000

    def begin(self, u, first, T, row):
        super().begin(u, first, T, row)
        self.gen[u] += 1
        assert first == self.traj(u, self.gen[u], 0), "the anchor the runner passes is the prefill's first token"
        self.pending[u], self.idx[u] = first, 0
        self.p[u], self.ctx_len[u] = T - 1, T
        self.mi[u], self.fresh[u] = 0, True

    def plan(self, live_after):
        self.calls.append(("plan", frozenset(live_after)))
        n = len(set(live_after) | {u for u in range(self.S) if self.active[u]})
        if n > self.small_B and self.cur_id != self.big_id:
            self._switch(self.big_id)
        elif n <= self.small_B and self.cur_id != self.small_id and self.rng.random() < 0.3:
            self._switch(self.small_id)
        return self.cur_id

    def _switch(self, tgt):
        before = (list(self.pending), list(self.p), list(self.ctx_len), list(self.fresh), list(self.idx))
        self.calls.append(("switch", self.cur_id, tgt))
        self.switches.append((self.cur_id, tgt))
        self.cur_id = tgt
        for u in range(self.S):
            if self.active[u]:
                self.mi[u] = 0
        assert before == (list(self.pending), list(self.p), list(self.ctx_len), list(self.fresh), list(self.idx))

    def step(self, only=None):
        live = [u for u in range(self.S) if self.active[u] and (only is None or u in only)]
        self.calls.append(("step", tuple(live)))
        K = self.Ks[self.cur_id]
        out = {}
        for u in live:
            assert self.ctx_len[u] >= self.p[u] + 1
            m = self.rng.randint(0, K)
            i0 = self.idx[u]
            com = [self.traj(u, self.gen[u], i) for i in range(i0, i0 + 1 + m)]
            assert com[0] == self.pending[u]
            out[u] = com[1:] if self.fresh[u] else com
            self.fresh[u] = False
            self.mi[u] = len(com) - 1
            self.idx[u] = i0 + len(com)
            self.pending[u] = self.traj(u, self.gen[u], self.idx[u])
            self.ctx_len[u] = self.p[u] + 1 + len(com)
            self.p[u] += len(com)
        return out


@requires_vd
def test_token_accounting_is_exact_across_bucket_switches_joins_and_leaves():
    rng = random.Random(2026)
    dec = _SimDec(S, small_B=4, rng=rng)
    obj = _vllm_obj(dec, S)
    nb = 8
    emitted = {}  # (phys, gen) -> ids the runner received, in order (first from "prefill")
    ended = set()
    for engine_step in range(400):
        # Arrivals: a free slot is "prefilled" (the vLLM prefill_forward bookkeeping) with some probability.
        for phys in range(S):
            if not dec.active[phys] and obj._pending[phys] is None and rng.random() < 0.2:
                gen = dec.gen[phys] + 1
                T = 100 + 17 * phys
                obj._pending[phys] = (T, torch.zeros(nb, dtype=torch.int32))
                obj._carry[phys], obj._stopped[phys], obj._prev_tail[phys] = [], False, None
                emitted[(phys, gen)] = [_SimDec.traj(phys, gen, 0)]
        rows = [phys for phys in range(S) if dec.active[phys] or obj._pending[phys] is not None]
        if not rows:
            continue
        tokens = torch.zeros(S, 1, dtype=torch.int32)
        start_pos = torch.full((S,), -1, dtype=torch.int32)
        for phys in rows:
            if obj._pending[phys] is not None:
                tokens[phys, 0] = _SimDec.traj(phys, dec.gen[phys] + 1, 0)
                start_pos[phys] = obj._pending[phys][0]
            else:
                tokens[phys, 0] = obj._prev_tail[phys]
                start_pos[phys] = dec.p[phys] + 1
        n_calls = len(dec.calls)
        out = obj.decode_forward(tokens=tokens, start_pos=start_pos, page_table=torch.zeros(S, nb, dtype=torch.int32))
        assert dec.calls[n_calls][0] == "plan", "every engine step starts with plan()"
        for phys in rows:
            ids = [int(t) for t in out[phys].tolist() if int(t) >= 0]
            assert ids, f"a live row must emit >= 1 id per ragged step (slot {phys})"
            emitted[(phys, dec.gen[phys])].extend(ids)
            assert obj._carry[phys] == [], "a ragged step emits its whole carry (<= K+1 < W ids)"
        # Departures: a request leaves after enough tokens, or at random (vLLM's release_request).
        for phys in list(rows):
            key = (phys, dec.gen[phys])
            if dec.active[phys] and (len(emitted[key]) >= 40 or rng.random() < 0.05):
                obj.release_request(phys)
                ended.add(key)
                assert not dec.active[phys] and obj._pending[phys] is None
    # Every request's ids are its own trajectory, in order, complete and unique.
    assert emitted, "no request ran"
    for (phys, gen), ids in emitted.items():
        want = [_SimDec.traj(phys, gen, i) for i in range(len(ids))]
        assert ids == want, f"slot {phys} gen {gen}: emitted {ids[:8]}... != trajectory {want[:8]}..."
        assert len(set(ids)) == len(ids), f"slot {phys} gen {gen}: duplicated ids"
    ups = [s for s in dec.switches if s[1] == dec.big_id]
    downs = [s for s in dec.switches if s[1] == dec.small_id]
    assert ups and downs, f"the simulation must switch in both directions (up {len(ups)}, down {len(downs)})"
    assert len(ended) >= 10, f"only {len(ended)} requests completed"


# ------------------------------------------------------------------------------------------------ 6. ring contract
@pytest.mark.parametrize("B,T", [(4, 8), (8, 4)])
def test_ring_contract_holds_for_both_bucket_geometries(B, T):
    """Both buckets address the ONE [32*Nv] ring: every (row, head) lane stays in its own lane and inside the ring,
    the switch's source block formula (mi*B + row)*Nv is spec_state_blk_idx's, HOLD rows carry the sentinel."""
    bh = B * NV
    assert B * T * NV == 32 * NV, "the two geometries share one 32-row ring"
    for mi in _mi_cases(B, T):
        idx = spec_state_blk_idx(mi, B, NV)
        for h in range(bh):
            assert int(idx[h]) % bh == h, f"idx[{h}]={int(idx[h])} leaves lane {h} (BH={bh})"
            assert 0 <= int(idx[h]) < 32 * NV, f"idx[{h}]={int(idx[h])} outside the shared ring"
        for row in range(B):
            assert int(idx[row * NV]) == (mi[row] * B + row) * NV, "switch_spec_cfg's source block formula"
        assert len(set(idx.tolist())) == bh
    held = spec_state_blk_idx([0] * B, B, NV, hold={1})
    assert bool((held[NV : 2 * NV] == -1).all()) and int(held[0]) == 0


def test_ctrl_pages_are_per_cfg():
    assert spec_ctrl_words(4, NV) == 64 and spec_ctrl_words(8, NV) == 112  # 1 + B + B*Nv rounded up to 16 words
    for B, T in ((4, 8), (8, 4)):
        page = spec_ctrl_page([1] * B, B, NV, parity=1, hold={0})
        assert page.shape == (1, spec_ctrl_words(B, NV))
        assert int(page[0, 0]) == 1, "word 0 = window parity"
        assert int(page[0, 1]) == 0 and int(page[0, 2]) == 1, "mi words: 0 for a held row, mi otherwise"
        blocks = page[0, 1 + B : 1 + B + B * NV]
        assert bool((blocks[:NV] == -1).all()), "HOLD sentinel for every head of the held row"
        assert int(blocks[NV]) == (1 * B + 1) * NV


# ------------------------------------------------------------------------------------------------ 7. warm-up sequencing
class _FakeDev:
    def __init__(self):
        self.n_programs = 1000
        self.compile_on_sweep = False  # simulate a program variant missing from phase 1

    def num_program_cache_entries(self):
        return self.n_programs


class _CaptureDec(_RecordingDec):
    """Fake DFlash2DualBucketDecoder for the phase-2 warm-up: a bucket registry, capture(), warm_sweep() (refuses
    live slots, may 'compile'), set_bucket() (live must fit) and a plan() that only forces up (the real
    hysteresis is BucketPlanner, tested above). step() records the bucket it ran in."""

    def __init__(self, S, buckets, dev):
        big = max(B for B, _ in buckets)
        super().__init__(S)
        single = len(buckets) == 1
        self.buckets = {("default" if single else f"{B}x{T}"): SimpleNamespace(B=B, T=T) for B, T in buckets}
        self.cur_id = next(k for k, b in self.buckets.items() if b.B == big)
        self.K = self.K_max = max(T for _, T in buckets) - 1
        self.dev = dev

    def capture(self, warm_position=64):
        self.calls.append(("capture", warm_position))

    def warm_sweep(self):
        assert not any(self.active), "warm_sweep() with a live slot"
        self.calls.append(("warm_sweep",))
        if self.dev.compile_on_sweep:
            self.dev.n_programs += 1

    def plan(self, live_after):
        n = len(set(live_after) | {u for u in range(self.S) if self.active[u]})
        self.calls.append(("plan", frozenset(live_after)))
        if n > self.buckets[self.cur_id].B:
            self.set_bucket(min((k for k, b in self.buckets.items() if b.B >= n), key=lambda k: self.buckets[k].B))
        return self.cur_id

    def set_bucket(self, bid):
        assert bid in self.buckets
        assert sum(self.active) <= self.buckets[bid].B, "live slots do not fit the target bucket"
        self.calls.append(("set_bucket", self.cur_id, bid))
        self.cur_id = bid

    def step(self, only=None):
        out = super().step(only)
        self.calls[-1] = self.calls[-1] + (self.cur_id,)
        return out


def _capture_harness(monkeypatch, buckets, used_mib=300, total_mib=1024):
    dev = _FakeDev()
    dec = _CaptureDec(S, buckets, dev)
    obj = _vllm_obj(dec, S)
    obj._spec, obj._spec_pre = None, dec
    obj._buckets, obj._multi_bucket = buckets, len(buckets) > 1
    obj._warm_rows = [torch.zeros(4, dtype=torch.int32) for _ in range(S)]
    model = SimpleNamespace(
        mesh_device=dev, _PREFILL_MASK_BUCKETS=[128, 256, 512], _trace_region_bytes=lambda d: used_mib << 20
    )
    obj.model = [model]
    obj._spec_prefill = lambda model, dec, phys, prompt, T, row: torch.zeros(1, 8)
    monkeypatch.setattr(vd.ttnn, "synchronize_device", lambda d: None)
    monkeypatch.setattr(
        vd.ttnn, "get_memory_view", lambda d, bt: SimpleNamespace(total_bytes_per_bank=total_mib << 20, num_banks=1)
    )
    return obj, dec, dev


def _plans_precede_begins(calls):
    """Every begin() is preceded by a plan() in its engine step: the most recent plan/begin/step-kind call before each
    begin is a plan or another begin of the same step (a set_bucket issued INSIDE plan() -- the forced up-switch --
    is not a boundary)."""
    kinds = [c[0] for c in calls]
    for i, k in enumerate(kinds):
        if k == "begin":
            prev = [x for x in kinds[:i] if x in ("plan", "begin", "step")]
            assert prev and prev[-1] in ("plan", "begin"), f"begin at {i} without a preceding plan(): {kinds[: i + 1]}"


@requires_vd
def test_spec_capture_sequences_both_buckets_and_guards():
    """Phase 2 on the dual profile: capture -> every slot joins in 8x4 (plan before each begin) -> 4 steps -> 4 slots
    leave, explicit switch to 4x8 -> 4 steps -> the 4 re-join with ONE plan() that forces 8x4 before their begins ->
    a step -> all end; then the guard: warm_sweep with no live slot, one traced step in 8x4 and 4x8, ending in 4x8."""
    obj, dec, dev = _capture_harness(pytest.MonkeyPatch(), ((8, 4), (4, 8)))
    obj._spec_capture()
    assert obj._spec is dec and obj._spec_pre is None
    c = dec.calls
    assert c[0] == ("capture", 129), "capture at S0 + 1 (smallest mask bucket 128)"
    _plans_precede_begins(c)
    joins = [x for x in c if x[0] in ("plan", "begin")][:16]
    assert [x[0] for x in joins] == ["plan", "begin"] * 8
    assert [x[1] for x in joins if x[0] == "begin"] == list(range(8))
    steps = [x for x in c if x[0] == "step"]
    assert [s[2] for s in steps[:4]] == ["8x4"] * 4, "4 traced steps in the identity bucket first"
    sw = [x for x in c if x[0] == "set_bucket"]
    assert sw[0] == ("set_bucket", "8x4", "4x8"), "explicit down-switch once the surplus slots left"
    i_sw = c.index(sw[0])
    assert [x[1] for x in c[:i_sw] if x[0] == "end"] == [4, 5, 6, 7]
    assert [s[2] for s in steps[4:8]] == ["4x8"] * 4
    assert sw[1] == ("set_bucket", "4x8", "8x4"), "plan() with 8 live forces the up-switch before the re-joins begin"
    i_up = c.index(sw[1])
    assert c[i_up - 1] == ("plan", frozenset(range(8)))
    rejoin = [x[1] for x in c[i_up:] if x[0] == "begin"][:4]
    assert rejoin == [4, 5, 6, 7]
    assert steps[8][2] == "8x4" and steps[8][1] == tuple(range(8))
    # the guard: warm sweep with nobody live, then one traced step per bucket (8x4 then 4x8), slot 0 ended
    i_sweep = c.index(("warm_sweep",))
    assert [x[1] for x in c[i_up:i_sweep] if x[0] == "end"] == list(range(8)), "every dummy slot ends before the sweep"
    tail = c[i_sweep:]
    assert [x[0] for x in tail] == ["warm_sweep", "plan", "begin", "step", "set_bucket", "step", "end"], tail
    assert tail[3][2] == "8x4" and tail[4] == ("set_bucket", "8x4", "4x8") and tail[5][2] == "4x8"
    assert dec.cur_id == "4x8", "the server starts serving in the small bucket"
    assert not any(dec.active)


@requires_vd
def test_spec_capture_guard_fails_startup_on_a_post_capture_compile_and_a_full_trace_region(expect_error):
    obj, dec, dev = _capture_harness(pytest.MonkeyPatch(), ((8, 4), (4, 8)))
    dev.compile_on_sweep = True
    with expect_error(RuntimeError, "compiled AFTER the spec traces"):
        obj._spec_capture()
    obj, dec, dev = _capture_harness(pytest.MonkeyPatch(), ((8, 4), (4, 8)), used_mib=900, total_mib=1024)
    with expect_error(RuntimeError, "TRACE region"):
        obj._spec_capture()


@requires_vd
def test_spec_capture_single_bucket_is_todays_dummy_session_plus_a_warning_only_guard():
    obj, dec, dev = _capture_harness(pytest.MonkeyPatch(), ((8, 4),))
    dev.compile_on_sweep = True  # single-bucket profiles: the guard warns, never fails startup
    obj._spec_capture()
    c = dec.calls
    assert c[0] == ("capture", 129)
    assert [x[0] for x in c[1:17]] == ["plan", "begin"] * 8
    assert [x[2] for x in c if x[0] == "step"][:4] == ["default"] * 4
    assert not any(x[0] == "set_bucket" for x in c), "no switch path on a single bucket"
    tail = c[c.index(("warm_sweep",)) :]
    assert [x[0] for x in tail] == ["warm_sweep", "plan", "begin", "step", "end"], tail
    assert obj._spec is dec

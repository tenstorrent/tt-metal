# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the engine side of the spec-step row buckets (QWEN36_SPEC_BUCKETS, work/notes/77 tasks T13-T15): the bucket of a step,
its kwargs / drafter trace id / handle fields / counters, the lowest-free-row placement in prefill, and the ``bucket_change`` chain
break (sync chain and deferred serving). Fake device (the rigs of test_spec_async_engine_host.py), no device.

Only B-row traces exist today, so a bucket b < B without an entry in ``engine._dr_traces`` runs at B and counts a fallback; the tests
register fake drafter trace ids for the buckets they want to use.

Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_bucket_engine_host.py
"""

import random
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_async_engine_host import _AsyncRig, _Pipe
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine_host import _Flow
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import DRAFT
from models.demos.blackhole.qwen36.tests.test_spec_slotmap_host import _engine as _slot_engine
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine


@pytest.fixture(autouse=True)
def _no_hold(monkeypatch):
    """The tests below the hold section run the immediate down-shift (QWEN36_SPEC_BUCKET_HOLD=0 = the behavior before the hold)."""
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")


DR = {1: 31, 2: 32, 4: 34}  # fake per-bucket drafter trace ids (B = 8 would use DRAFT)


def _rig(monkeypatch, B, buckets=None, traces=(), chain=True):
    """An _AsyncRig whose model records the ``bucket`` kwarg of every verify_stage_and_enqueue (and drops it: the real model does not
    take it yet). ``buckets`` None leaves the engine as without the flag."""
    r = _AsyncRig(monkeypatch, B, chain=chain)
    if buckets is not None:
        r.e._buckets = tuple(buckets)
    r.e._dr_traces = {b: DR[b] for b in traces}
    r.bk = []
    orig = r.m.verify_stage_and_enqueue

    def wrap(*a, **k):
        r.bk.append(k.pop("bucket", "absent"))
        return orig(*a, **k)

    r.m.verify_stage_and_enqueue = wrap
    return r


def _bare(B=8, buckets=(1, 2, 4, 8), traces=(1, 2, 4)):
    e = MTPSpecEngine.__new__(MTPSpecEngine)
    e.B, e.K, e.T, e._buckets, e._dr_traces = B, 3, 4, tuple(buckets), {b: DR[b] for b in traces}
    return e


def _sync_run(monkeypatch, B, nv, accepts, finish_after, **rig_kw):
    """Sync serving loop (the _Flow of the chain tests): per step the engine's commit, its propose, the next block; ``finish_after``
    maps step -> rows released after that step. Returns (rig, [(ids, counts)], [drafter trace ids of each step])."""
    r = _rig(monkeypatch, B, **rig_kw)
    r.e._active = {u for u in range(B) if nv[u] >= 0}
    f = _Flow(r, B, list(nv))
    out, drs = [], []
    for i, acc in enumerate(accepts):
        f.step(acc)
        out.append((f.res.accepted_token_ids.clone(), f.res.accepted_counts.clone()))
        drs.append([x for x in r.names("trace") if x == DRAFT or x in DR.values()])
        f.propose()
        f.next_block(finish=tuple(finish_after.get(i, ())))
    return r, out, drs


def _pipe_run(monkeypatch, B, nv, accepts, finish, **rig_kw):
    """Deferred serving, two steps in flight (as _run_pipe of the async tests); ``finish`` maps resolved step j -> rows released
    after it (they leave from step j + 2). Returns (rig, handles, [(ids, counts)])."""
    r = _rig(monkeypatch, B, **rig_kw)
    r.queue = [("acc", a) for a in accepts]
    p = _Pipe(r, B, nv)
    hs, out, pend = [], [], []
    for _ in range(len(accepts)):
        h = p.enq(pend[-1] if pend else None)
        hs.append(h)
        pend.append(h)
        if len(pend) == 2:
            j = len(out)
            ids, counts, _, _ = p.resolve(pend.pop(0), release=tuple(finish.get(j, ())))
            out.append((ids, counts))
    while pend:
        j = len(out)
        ids, counts, _, _ = p.resolve(pend.pop(0), release=tuple(finish.get(j, ())))
        out.append((ids, counts))
    return r, hs, out


def _same_streams(a, b, rows=None):
    assert len(a) == len(b)
    for i, ((ia, ca), (ib, cb)) in enumerate(zip(a, b)):
        for u in range(ia.shape[0]) if rows is None else rows(i):
            assert int(ca[u]) == int(cb[u]), f"step {i} row {u} counts"
            assert torch.equal(ia[u, : int(ca[u])], ib[u, : int(cb[u])]), f"step {i} row {u} tokens"


# --------------------------------------------------------------------------- #
# bucket selection
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "rows, want",
    [([0], 1), ([0, 1], 2), ([1], 2), ([2], 4), ([0, 3], 4), ([4], 8), ([0, 7], 8), ([], 1), ([2, 0, 1], 4)],
)
def test_bucket_of_is_the_smallest_bucket_above_the_highest_live_row(rows, want):
    e = _bare()
    assert e._bucket_of(rows) == want
    mask = torch.zeros(8, dtype=torch.bool)
    mask[rows] = True
    assert e._bucket_of(e._active_rows(mask)) == want


def test_bucket_of_falls_back_to_B_for_a_bucket_without_a_trace():
    e = _bare(traces=(2,))  # only bucket 2 has a drafter trace
    assert [e._bucket_of(r) for r in ([0], [0, 1], [2], [0, 3], [5])] == [8, 2, 8, 8, 8]
    e._bucket_step([0])
    assert e._vb == 8 and e._vb_fell
    e._bucket_step([1])
    assert e._vb == 2 and not e._vb_fell


def test_flag_off_or_only_B_means_bucket_B_and_no_hooks():
    for buckets in (None, (8,)):
        e = _bare()
        e._buckets = buckets
        assert not e._bucketed() and e._bucket_of([0]) == 8
        e._bucket_step([0])
        kw = {"read_logits": False}
        assert e._bucket_kw(kw) == {"read_logits": False} and e._vb is None
        e._dr_trace = DRAFT
        assert e._dr_trace_for(None) == DRAFT and e._dr_trace_for(8) == DRAFT
        assert e._bucket_note_text() == ""
        assert e.__dict__.get("bucket_steps") is None and e.bucket_fallbacks == 0 and e.row_swaps == 0


def test_dr_trace_for_picks_the_bucket_trace_and_B_uses_the_default():
    e = _bare(traces=(1, 2))
    e._dr_trace = DRAFT
    assert [e._dr_trace_for(b) for b in (None, 1, 2, 4, 8)] == [DRAFT, DR[1], DR[2], DRAFT, DRAFT]


def test_engine_init_reads_the_flag(monkeypatch):
    monkeypatch.setattr(ttnn, "init_device_compute_kernel_config", lambda *a, **k: None)

    def make_model(max_batch_size=8):
        return types.SimpleNamespace(
            mtp=object(),
            num_devices=2,
            args=types.SimpleNamespace(vocab_size=8, dim=16, max_batch_size=max_batch_size),
            mesh_device=types.SimpleNamespace(arch=lambda: None),
            gdn_fused_decode=True,
            layers=[],
        )

    model = make_model()
    monkeypatch.delenv("QWEN36_SPEC_BUCKETS", raising=False)
    e = MTPSpecEngine(model, 8, 3)
    assert e._buckets == (1, 2, 4, 8) and e._bucketed()  # unset = the default = on
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "8")
    e = MTPSpecEngine(model, 8, 3)
    assert e._buckets == (8,) and not e._bucketed() and e._dr_traces == {}
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    e = MTPSpecEngine(model, 8, 3)
    assert e._buckets == (1, 2, 4, 8) and e._bucketed()
    # an engine narrower than 8 keeps only the buckets <= B (plus B): B=1 is not bucketed, B=4 keeps 1,2,4
    e = MTPSpecEngine(make_model(1), 1, 3)
    assert e._buckets == (1,) and not e._bucketed()
    e = MTPSpecEngine(make_model(4), 4, 3)
    assert e._buckets == (1, 2, 4) and e._bucketed()


# --------------------------------------------------------------------------- #
# the step: kwargs, drafter trace, counters, flag "8" = HEAD
# --------------------------------------------------------------------------- #
ACC = [[1, 1, 0, 0], [2, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0], [0, 2, 0, 0]]


def test_default_flag_runs_exactly_the_head_calls_and_counters(monkeypatch):
    """Flag unset, "8" and a bucket list with no traces all run the same device calls; the first two add no kwarg and no counter."""
    nv = (3, 2, 1, 0)
    runs = {}
    for name, kw in (("none", {}), ("only8", dict(buckets=(4,))), ("no_traces", dict(buckets=(1, 2, 4)))):
        monkeypatch.delenv("QWEN36_SPEC_BUCKETS", raising=False)
        r = _rig(monkeypatch, 4, **kw)
        r.e._active = set(range(4))
        f = _Flow(r, 4, list(nv))
        logs = []
        for acc in ACC[:3]:
            f.step(acc)
            logs.append(r.log[:])
            f.propose()
            f.next_block()
        runs[name] = (r, logs)
    r0, l0 = runs["none"]
    for name in ("only8", "no_traces"):
        r1, l1 = runs[name]
        assert [[x for x in lg if x[0] == "trace"] for lg in l1] == [
            [x for x in lg if x[0] == "trace"] for lg in l0
        ], name
        assert l1 == l0, name
        assert r1.e.chain_breaks == r0.e.chain_breaks and r1.e.chain_steps == r0.e.chain_steps
    assert r0.bk == ["absent"] * 3 and r0.e.__dict__.get("bucket_steps") is None
    assert r0.e.row_swaps == 0 and r0.e.bucket_fallbacks == 0
    # bucket list (1, 2, 4) on B = 4: four rows run bucket 4 = B, no fallback
    r2 = runs["no_traces"][0]
    assert r2.bk == [4] * 3 and r2.e.bucket_steps == {4: 3} and r2.e.bucket_fallbacks == 0


def test_bucket_reaches_the_verify_and_the_drafter_trace_and_is_counted(monkeypatch):
    # rows 0, 1 live: bucket 2 has a trace; row 2 later joins: bucket 4 = B
    r = _rig(monkeypatch, 4, buckets=(1, 2, 4), traces=(1, 2))
    r.e._active = {0, 1}
    f = _Flow(r, 4, [3, 2, -1, -1])
    f.step([1, 1, 0, 0])
    assert r.bk == [2]
    assert DR[2] in r.names("trace") and DRAFT not in r.names("trace")
    assert r.e._vb == 2 and r.e.bucket_steps == {2: 1} and r.e.bucket_fallbacks == 0
    # one row only: bucket 1
    r1 = _rig(monkeypatch, 4, buckets=(1, 2, 4), traces=(1, 2))
    r1.e._active = {0}
    f1 = _Flow(r1, 4, [3, -1, -1, -1])
    f1.step([1, 0, 0, 0])
    assert r1.bk == [1] and DR[1] in r1.names("trace")
    # rows 0..2: bucket 4 = B runs the default drafter trace
    r4 = _rig(monkeypatch, 4, buckets=(1, 2, 4), traces=(1, 2))
    r4.e._active = {0, 1, 2}
    f4 = _Flow(r4, 4, [3, 2, 1, -1])
    f4.step([1, 1, 1, 0])
    assert r4.bk == [4] and DRAFT in r4.names("trace")
    assert r4.e._dr_trace_for(r4.e._vb) == DRAFT


def test_missing_bucket_trace_runs_at_B_and_counts_a_fallback(monkeypatch):
    r = _rig(monkeypatch, 4, buckets=(1, 2, 4), traces=(2,))
    r.e._active = {0}
    f = _Flow(r, 4, [3, -1, -1, -1])
    f.step([1, 0, 0, 0])  # bucket 1 has no trace
    assert r.bk == [4] and DRAFT in r.names("trace") and not any(t in DR.values() for t in r.names("trace"))
    assert r.e.bucket_fallbacks == 1 and r.e.bucket_steps == {4: 1}


def test_stats_line_lists_steps_per_bucket_and_swaps(monkeypatch):
    e = _bare()
    assert (
        e._bucket_note_text()
        == "; buckets 1:0 2:0 4:0 8:0, fallbacks 0, row swaps 0, down-shifts hold 0 free 0, free switches 0"
    )
    e.bucket_steps, e.bucket_fallbacks, e.row_swaps = {1: 3, 8: 2}, 1, 4
    e.bucket_downs_hold, e.bucket_downs_free, e.bucket_free_switch = 5, 6, 7
    assert e._bucket_note_text() == (
        "; buckets 1:3 2:0 4:0 8:2, fallbacks 1, row swaps 4, down-shifts hold 5 free 6, free switches 7"
    )


# --------------------------------------------------------------------------- #
# row placement (prefill)
# --------------------------------------------------------------------------- #
def _placing_engine(B=8, buckets=(1, 2, 4, 8), slotmap=True):
    e = _slot_engine(B, slotmap=slotmap)
    e._buckets, e._dr_traces = tuple(buckets), {}
    e.landed = []

    def prefill_row(row, *a, **k):
        e._active.add(row)
        e.landed.append(row)
        return None

    e._prefill_row = prefill_row
    return e


def _admit(e, slot):
    e.prefill(slot, [1, 2, 3], None)
    return e._phys[slot]


def test_new_requests_land_on_the_lowest_free_engine_row():
    e = _placing_engine()
    rows = [_admit(e, s) for s in (5, 3, 6, 0)]  # the plugin picks any free runner slot
    assert rows == [0, 1, 2, 3] and e.row_swaps == 4  # every admit swapped: slot 0 had moved to row 5 at the first one
    assert sorted(e._phys) == list(range(8)) and e._active == {0, 1, 2, 3}


def test_admit_4_free_1_admit_1_reuses_the_hole():
    e = _placing_engine()
    slots = [5, 3, 6, 7]
    for s in slots:
        _admit(e, s)
    assert e._active == {0, 1, 2, 3}
    e.release(3)  # the request on slot 3 (engine row 1) leaves
    assert e._active == {0, 2, 3}
    assert _admit(e, 1) == 1  # lowest free engine row
    assert e._active == {0, 1, 2, 3} and sorted(e._phys) == list(range(8))


def test_occupied_rows_never_move_and_a_reprefill_does_nothing():
    e = _placing_engine()
    rng = random.Random(3)
    live = {}  # runner slot -> engine row
    for _ in range(200):
        if live and (len(live) == 8 or rng.random() < 0.45):
            s = rng.choice(sorted(live))
            e.release(s)
            del live[s]
        elif live and rng.random() < 0.15:
            s = rng.choice(sorted(live))
            before, swaps = list(e._phys), e.row_swaps
            assert _admit(e, s) == live[s]  # re-prefill of a live request: no move
            assert e._phys == before and e.row_swaps == swaps
        else:
            free_slots = [s for s in range(8) if s not in live]
            s = rng.choice(free_slots)
            free_rows = sorted(set(range(8)) - e._active)
            assert _admit(e, s) == free_rows[0]
            live[s] = e._phys[s]
        assert sorted(e._phys) == list(range(8))
        assert {s: e._phys[s] for s in live} == live and set(live.values()) == e._active


def test_placement_off_without_the_flag_or_the_host_slot_map():
    for kw in (dict(buckets=(8,)), dict(slotmap=False)):
        e = _placing_engine(**kw)
        assert [_admit(e, s) for s in (5, 3)] == [5, 3] and e.row_swaps == 0 and e._phys == list(range(8))
    e = _placing_engine()
    e._buckets = None
    assert _admit(e, 6) == 6 and e.row_swaps == 0


def _handle(rows, B=8, finished=False, prev=None):
    act = torch.zeros(B, dtype=torch.bool)
    act[list(rows)] = True
    return types.SimpleNamespace(active=act, finished=finished, prev=prev, device=True, fin=None, chainable=True)


def test_no_swap_while_a_deferred_step_in_flight_describes_either_row():
    # rows 0, 1 live; the plugin picks slot 6 (row 6); the lowest free row is 2
    for pending_rows, swaps in (({2}, 0), ({6}, 0), ({2, 6}, 0), ({3}, 1), ({0, 1}, 1)):
        e = _placing_engine()
        _admit(e, 0), _admit(e, 1)
        e._def_last = _handle(pending_rows)
        assert _admit(e, 6) == (6 if swaps == 0 else 2), pending_rows
        assert e.row_swaps == swaps and sorted(e._phys) == list(range(8))
        assert e._def_last.chainable is False  # prefill's existing rule still runs
    # an older handle in the prev chain counts too; a finished one does not
    e = _placing_engine()
    _admit(e, 0), _admit(e, 1)
    e._def_last = _handle({0}, prev=_handle({2}))
    assert _admit(e, 6) == 6 and e.row_swaps == 0
    e = _placing_engine()
    _admit(e, 0), _admit(e, 1)
    e._def_last = _handle({2, 6}, finished=True)
    assert _admit(e, 6) == 2 and e.row_swaps == 1


@pytest.mark.parametrize("seed", range(4))
def test_request_to_engine_row_identity_holds_with_slot_remaps(seed):
    B = 8
    e = _placing_engine(B)
    rng = random.Random(seed)
    rows = [None] * B  # runner row i -> request id (the plugin's view)
    home, nxt = {}, 0  # request -> engine row
    for _ in range(300):
        op = rng.random()
        live_i = [i for i in range(B) if rows[i] is not None]
        if op < 0.25 and live_i:  # a request ends
            i = rng.choice(live_i)
            e.release(i)
            rows[i] = None
        elif op < 0.65 and len(live_i) < B:  # a request is admitted into a free runner slot
            i = rng.choice([i for i in range(B) if rows[i] is None])
            free_rows = sorted(set(range(B)) - e._active)
            row = _admit(e, i)
            assert row == free_rows[0]
            rows[i], home[nxt], nxt = nxt, row, nxt + 1
        else:  # the plugin condenses / shuffles runner rows
            remap = list(range(B))
            rng.shuffle(remap)
            e.remap_slots(remap)
            rows = [rows[j] for j in remap]
        assert sorted(e._phys) == list(range(B))
        for i in range(B):
            if rows[i] is not None:
                assert e._phys[i] == home[rows[i]], (i, rows[i])
        assert e._active == {home[q] for q in rows if q is not None}


# --------------------------------------------------------------------------- #
# the bucket_change chain break
# --------------------------------------------------------------------------- #
def test_chain_reason_bucket_change_row_set_and_event_order():
    e = _bare(B=4, buckets=(1, 2, 4), traces=(1, 2))
    prev = dict(taken=True, active=torch.tensor([True, True, True, True]), bucket=4)
    act = torch.tensor([True, True, False, False])
    args = (None, None, None)
    e._bucket_step(e._active_rows(act))  # the step's bucket (2), as verify / the deferred step decide it
    # a later reason masks the bucket change (the full bucket_change case runs in the sync / deferred tests below)
    prev["kind"], e._ch_trace_g = "g", None
    assert e._chain_reason(prev, None, None, None, None, act, None, args) == "not_captured"
    assert e._chain_reason(prev, None, None, None, None, act, None, args, skip_bucket=True) == "not_captured"
    # an event reason comes first; an up-shift through a joining row is a row_set
    assert e._chain_reason(prev, None, "park", None, None, act, None, args) == "park"
    up = dict(taken=True, active=torch.tensor([True, True, False, False]), bucket=2)
    assert e._chain_reason(up, None, None, None, None, torch.tensor([True, True, True, False]), None, args) == "row_set"
    e._bucket_step([0, 1])
    # flag off: never bucket_change (the next check, not_captured, answers instead)
    e._buckets = (4,)
    e._ch_trace_g = e._ch_trace_s = None
    assert e._chain_reason(prev, None, None, None, None, act, None, args) == "not_captured"
    # a bucket without a trace runs at B: no change
    e._buckets, e._dr_traces = (1, 2, 4), {}
    assert e._chain_reason(prev, None, None, None, None, act, None, args) == "not_captured"


def test_sync_chain_down_shift_breaks_once_and_chains_again(monkeypatch):
    nv, acc = (3, 2, 1, 0), [[1, 1, 0, 0]] * 6
    fin = {0: (3,), 2: (2,)}  # rows {0..3} -> {0,1,2} (still bucket 4) -> {0,1} (bucket 2)
    r, out, drs = _sync_run(monkeypatch, 4, nv, acc, fin, buckets=(1, 2, 4), traces=(1, 2))
    assert r.bk == [4, 4, 4, 2, 2, 2]
    assert r.e.chain_breaks == {"first": 1, "bucket_change": 1}, r.e.chain_breaks
    assert r.e.chain_steps == 4  # steps 1, 2 and 4, 5 chained; step 3 (the shift) ran host-staged
    assert r.e.bucket_steps == {4: 3, 2: 3}
    assert [d == [DRAFT] for d in drs] == [True] * 3 + [False] * 3 and drs[3] == [DR[2]]
    # the same script with the flag off: same streams, no bucket_change
    r0, out0, _ = _sync_run(monkeypatch, 4, nv, acc, fin)
    assert r0.e.chain_breaks == {"first": 1} and r0.e.chain_steps == 5
    _same_streams(out, out0)


def test_sync_chain_without_bucket_traces_never_breaks_on_buckets(monkeypatch):
    nv, acc = (3, 2, 1, 0), [[1, 1, 0, 0]] * 5
    r, _, _ = _sync_run(monkeypatch, 4, nv, acc, {0: (3,), 2: (2,)}, buckets=(1, 2, 4), traces=())
    assert r.e.chain_breaks == {"first": 1} and r.bk == [4] * 5 and r.e.bucket_fallbacks == 2


def test_deferred_down_shift_materializes_with_reason_bucket_change(monkeypatch):
    nv, acc = (3, 2, 1, 0), [[1, 1, 0, 0]] * 10
    fin = {0: (3,), 2: (2,)}  # row 3 leaves from step 2 (bucket 4 -> 4), row 2 from step 4 (bucket 4 -> 2)
    r, hs, out = _pipe_run(monkeypatch, 4, nv, acc, fin, buckets=(1, 2, 4), traces=(1, 2))
    assert r.e.chain_breaks == {"bucket_change": 1}, r.e.chain_breaks
    assert r.e.__dict__.get("def_breaks") == {"bucket_change": 1}
    assert [h.bucket for h in hs] == [4, 4, 4, 4, 2, 2, 2, 2, 2, 2]
    assert r.bk == [h.bucket for h in hs] and r.e.bucket_steps == {4: 4, 2: 6}
    assert len(r.syncs()) == 1  # the one synchronize is the materialize of the break
    r0, hs0, out0 = _pipe_run(monkeypatch, 4, nv, acc, fin)
    assert r0.e.__dict__.get("def_breaks", {}) == {} and all(h.bucket is None for h in hs0) and set(r0.bk) == {"absent"}
    _same_streams(out, out0)


def test_deferred_up_shift_by_a_join_reports_host_row_not_bucket_change(monkeypatch):
    r = _rig(monkeypatch, 4, buckets=(1, 2, 4), traces=(1, 2))
    r.queue = [("acc", [1, 1, 0, 0])] * 4
    r.e._active = {0, 1, 2}
    p = _Pipe(r, 4, (3, 2, -1, -1))
    h0 = p.enq()
    assert h0.bucket == 2
    h1 = p.enq(h0, host={2: (7, 60)})  # a new request on row 2: host tokens, no chain for that row
    assert h1.bucket == 4
    assert r.e.__dict__.get("def_breaks") == {"host_row": 1}, r.e.def_breaks


# --------------------------------------------------------------------------- #
# QWEN36_SPEC_BUCKET_HOLD: the down-shift delay
# --------------------------------------------------------------------------- #
def _hold_ref(buckets, hold, seq):
    """The switch rule as a plain reference: per step the live rows' wanted bucket in, (bucket used, cause) out."""
    cb, streak, smax, out = None, 0, 0, []
    for w in seq:
        cause = None
        if cb is None or w >= cb:
            cb, streak, smax = w, 0, 0
        else:
            streak, smax = streak + 1, max(smax, w)
            if streak >= hold:
                cb, streak, smax, cause = smax, 0, 0, "hold"
        out.append((cb, cause))
    return out


def test_hold_flag_default_and_parse(monkeypatch):
    from models.demos.blackhole.qwen36.tt.tp_common import spec_bucket_hold

    monkeypatch.delenv("QWEN36_SPEC_BUCKET_HOLD", raising=False)
    assert spec_bucket_hold() == 16
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")
    assert spec_bucket_hold() == 0
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "5")
    assert spec_bucket_hold() == 5


def test_hold_zero_is_the_bucket_of_the_live_rows_every_step(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")
    rng = random.Random(1)
    e = _bare(traces=(1, 2, 4))
    for _ in range(200):
        rows = sorted(rng.sample(range(8), rng.randint(1, 8)))
        e._bucket_step(rows)
        assert e._vb == e._bucket_of(rows)
        assert e._bk_cand is None


def test_hold_16_oscillating_1_and_5_rows_never_down_shifts_until_16_steps_fit(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "16")
    e = _bare()
    e._bucket_step([0, 1, 2, 3, 4])
    assert e._vb == 8
    for _ in range(40):  # 1 row (wants 1), then 5 rows (wants 8): the streak never reaches 16
        e._bucket_step([0])
        assert e._vb == 8
        e._bucket_step([0, 1, 2, 3, 4])
        assert e._vb == 8
    assert e.bucket_downs_hold == 0 and e.bucket_downs_free == 0
    for i in range(15):
        e._bucket_step([0])
        assert e._vb == 8, i
    e._bucket_step([0])  # the 16th consecutive step that fits a smaller bucket
    assert e._vb == 1 and e.bucket_downs_hold == 1


def test_hold_switches_down_to_the_largest_wanted_of_the_streak(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "4")
    e = _bare()
    e._bucket_step([0, 1, 2, 3, 4, 5])
    used = []
    for rows in ([0], [0, 1], [0], [0, 2]):  # wanted 1, 2, 1, 4: the new bucket fits the 4
        e._bucket_step(rows)
        used.append(e._vb)
    assert used == [8, 8, 8, 4] and e.bucket_downs_hold == 1
    e._bucket_step([0, 1, 2, 3, 4])  # up-shift: at once, resets the streak
    assert e._vb == 8 and e._bk_streak == 0 and e._bk_smax == 0


@pytest.mark.parametrize("hold", [0, 1, 3, 16])
def test_hold_random_live_sets_follow_the_rule_and_always_cover_the_live_rows(monkeypatch, hold):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", str(hold))
    rng = random.Random(7 + hold)
    e = _bare()
    seq, used = [], []
    for _ in range(400):
        rows = sorted(rng.sample(range(8), rng.choice((1, 1, 2, 3, 5, 8))))
        e._bucket_step(rows)
        seq.append(e._bucket_of(rows))
        used.append(e._vb)
        assert e._vb in (1, 2, 4, 8) and e._vb >= max(rows) + 1, (rows, e._vb)
        assert e._vb >= e._bucket_of(rows)
    ref = _hold_ref((1, 2, 4, 8), hold, seq)
    assert used == [b for b, _ in ref]
    assert e.bucket_downs_hold == sum(c == "hold" for _, c in ref)


def test_free_down_shift_at_another_break_costs_no_bucket_change(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "16")
    e = _bare(B=4, buckets=(1, 2, 4), traces=(1, 2))
    e._ch_g, e._chain_why, e._chain_ev = object(), None, "park"
    e._chain_in = dict(taken=True, active=torch.tensor([True, True, True, False]), bucket=4)
    e.bucket_steps = {4: 5}
    e._bucket_step([0, 1, 2])
    e._bucket_step([0, 1])  # wants 2, held at 4: pending
    assert e._vb == 4 and e._bk_cand == 2
    kw = {"bucket": 4}
    e.bucket_steps[4] += 1
    act = torch.tensor([True, True, False, False])
    assert e._chain_plan(None, None, act, None, (None, None, None), kw) is None
    assert e.chain_breaks == {"park": 1}  # no bucket_change on top
    assert e._vb == 2 and kw["bucket"] == 2 and e._bk_cb == 2 and e._bk_cand is None
    assert e.bucket_downs_free == 1 and e.bucket_downs_hold == 0 and e.bucket_free_switch == 1
    assert e.bucket_steps == {4: 5, 2: 1}
    # the same switch when the hold runs out on a step that has a later reason: counted under that reason only
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")
    e2 = _bare(B=4, buckets=(1, 2, 4), traces=(1, 2))
    e2._ch_g, e2._chain_why, e2._chain_ev = object(), None, None
    e2._ch_trace_g = None
    e2._chain_in = dict(taken=True, active=torch.tensor([True, True, True, False]), bucket=4, kind="g")
    e2._bucket_step([0, 1, 2])
    e2._bucket_step([0, 1])
    assert e2._vb == 2 and e2.bucket_downs_hold == 1
    assert e2._chain_plan(None, None, act, None, (None, None, None)) is None
    assert e2.chain_breaks == {"not_captured": 1} and e2.bucket_free_switch == 1


def test_sync_hold_keeps_the_chain_through_the_would_be_down_shift(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "16")
    nv, acc = (3, 2, 1, 0), [[1, 1, 0, 0]] * 6
    r, out, _ = _sync_run(monkeypatch, 4, nv, acc, {0: (3,), 2: (2,)}, buckets=(1, 2, 4), traces=(1, 2))
    assert r.bk == [4] * 6 and r.e.chain_breaks == {"first": 1} and r.e.chain_steps == 5
    assert r.e.bucket_steps == {4: 6} and r.e.bucket_downs_hold == 0
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")
    r0, out0, _ = _sync_run(monkeypatch, 4, nv, acc, {0: (3,), 2: (2,)}, buckets=(1, 2, 4), traces=(1, 2))
    _same_streams(out, out0)


def test_sync_hold_runs_out_then_one_bucket_change_break(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "2")
    nv, acc = (3, 2, 1, 0), [[1, 1, 0, 0]] * 6
    r, _, _ = _sync_run(monkeypatch, 4, nv, acc, {0: (3,), 2: (2,)}, buckets=(1, 2, 4), traces=(1, 2))
    assert r.bk == [4, 4, 4, 4, 2, 2]  # step 3 wants 2 (streak 1), step 4 reaches the hold
    assert r.e.chain_breaks == {"first": 1, "bucket_change": 1} and r.e.bucket_downs_hold == 1


def test_deferred_hold_matches_the_sync_rule(monkeypatch):
    nv, acc = (3, 2, 1, 0), [[1, 1, 0, 0]] * 10
    fin = {0: (3,), 2: (2,)}
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "16")
    r, hs, out = _pipe_run(monkeypatch, 4, nv, acc, fin, buckets=(1, 2, 4), traces=(1, 2))
    assert [h.bucket for h in hs] == [4] * 10 and r.bk == [4] * 10
    assert r.e.chain_breaks == {} and r.e.__dict__.get("def_breaks", {}) == {} and len(r.syncs()) == 0
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "2")  # wanted 2 from step 4: held at step 4, switched at step 5
    r2, hs2, out2 = _pipe_run(monkeypatch, 4, nv, acc, fin, buckets=(1, 2, 4), traces=(1, 2))
    assert [h.bucket for h in hs2] == [4] * 5 + [2] * 5
    assert r2.e.__dict__.get("def_breaks") == {"bucket_change": 1} and r2.e.bucket_downs_hold == 1
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")
    r0, hs0, out0 = _pipe_run(monkeypatch, 4, nv, acc, fin, buckets=(1, 2, 4), traces=(1, 2))
    _same_streams(out, out0)
    _same_streams(out2, out0)


def test_deferred_free_down_shift_at_a_park_costs_no_bucket_change(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "16")
    r = _rig(monkeypatch, 4, buckets=(1, 2, 4), traces=(1, 2))
    r.queue = [("acc", [1, 1, 0, 0])] * 3
    r.e._active = {0, 1, 2}
    p = _Pipe(r, 4, (3, 2, 1, -1))
    h0 = p.enq()
    assert h0.bucket == 4
    h1 = p.enq(h0, absent=(2,))  # row 2 left out of the step (a park): rows 0, 1 want bucket 2, free now
    assert r.e.__dict__.get("def_breaks") == {"park": 1}, r.e.__dict__.get("def_breaks")
    assert h1.bucket == 2 and r.e.bucket_downs_free == 1 and r.e.bucket_free_switch == 1
    assert r.e.bucket_steps == {4: 1, 2: 1} and r.e._bk_cb == 2


# --------------------------------------------------------------------------- #
# T16: spec_memory_costs and the row buckets

_TC = dict(
    hidden_size=512,
    vocab_size=1024,
    num_attention_heads=8,
    num_key_value_heads=2,
    head_dim=64,
    linear_num_key_heads=4,
    linear_num_value_heads=8,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    num_hidden_layers=4,
)


def test_spec_memory_costs_row_buckets(monkeypatch):
    from models.demos.blackhole.qwen36.tt.spec_engine import spec_memory_costs

    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "8")
    base = spec_memory_costs(_TC, 2, 8, 3, nb=16)
    assert spec_memory_costs(_TC, 2, 8, 3, nb=16, buckets=(8,)) == base  # "8" is off
    monkeypatch.delenv("QWEN36_SPEC_BUCKETS", raising=False)
    assert spec_memory_costs(_TC, 2, 8, 3, nb=16) == spec_memory_costs(
        _TC, 2, 8, 3, nb=16, buckets=(1, 2, 4, 8)
    )  # unset = on
    tot = {}
    for bks in ((8,), (4, 8), (2, 4, 8), (1, 2, 4, 8)):
        c = spec_memory_costs(_TC, 2, 8, 3, nb=16, buckets=bks)
        assert c["per_seq"] == base["per_seq"] and c["per_token"] == base["per_token"]
        tot[bks] = c["fixed"]["total"]
    keys = list(tot)
    assert all(tot[a] < tot[b] for a, b in zip(keys, keys[1:])), tot  # monotone: every extra bucket costs bytes
    # the env flag is the default
    monkeypatch.setenv("QWEN36_SPEC_BUCKETS", "1,2,4,8")
    assert spec_memory_costs(_TC, 2, 8, 3, nb=16) == spec_memory_costs(_TC, 2, 8, 3, nb=16, buckets=(1, 2, 4, 8))

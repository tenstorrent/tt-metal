# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for the tail planner in the spec engine: QWEN36_PREFILL_INTRACE_SNAP (one segment per spec
fast-prefill call, GDN prefix-cache snapshots taken inside the replays) with the fake model of
test_spec_fastprefill_host.py and a real GdnPrefixStateCache (copies faked):

* flags off = today's segmented calls and saves; INTRACE_SNAP on = one segment, the same save positions / keys /
  low-priority and end marks in the same order, in-replay points saved through the snapshot copy function;
* the fallbacks to the legacy split (eager tail, too many points in one replay, trash-block collision);
* GdnPrefixStateCache.save with a snapshot copy function; Qwen36Model._snap_stage's change-only staging.
The pure planner is tested in test_prefill_plan_host.py.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_prefill_tail_planner_host.py -q
"""
import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import (
    _engine,
    _fake_model,
    _row,
    fake_ttnn,
)
from models.demos.blackhole.qwen36.tests.test_spec_prefix_fast_host import _cache
from models.demos.blackhole.qwen36.tt import tp_common
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.prefix_cache import SpecPoints
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine, _SnapRun

__all__ = ["fake_ttnn"]  # fixture re-export

@pytest.fixture(autouse=True)
def _flags(monkeypatch):
    """Every flag of this track off and re-read (they are read once)."""
    for name in ("QWEN36_PREFILL_INTRACE_SNAP", "QWEN36_PREFILL_MASKED_BIG", "QWEN36_PREFILL_COST_TABLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(tp_common, "_PREFILL_INTRACE_SNAP", None)
    monkeypatch.setattr(tp_common, "_PREFILL_MASKED_BIG", None)


def test_flags_default_off_and_read_once(monkeypatch):
    assert tp_common.prefill_intrace_snap_enabled() is False
    assert tp_common.prefill_masked_big_enabled() is False
    monkeypatch.setenv("QWEN36_PREFILL_INTRACE_SNAP", "1")
    monkeypatch.setenv("QWEN36_PREFILL_MASKED_BIG", "1")
    assert tp_common.prefill_intrace_snap_enabled() is False  # read once
    monkeypatch.setattr(tp_common, "_PREFILL_INTRACE_SNAP", None)
    monkeypatch.setattr(tp_common, "_PREFILL_MASKED_BIG", None)
    assert tp_common.prefill_intrace_snap_enabled() is True and tp_common.prefill_masked_big_enabled() is True


def _prompt(L, seed=0):
    return [(seed * 7919 + 3 * i) % 997 + 1 for i in range(L)]


def _points(pts, low=(), end=()):
    p = SpecPoints(sorted(pts))
    p.low, p.end = frozenset(low), frozenset(end)
    return p


def _snap_model(calls, cache, trace_lengths, big=8192, snap_on=True):
    """The fake model of test_spec_fastprefill_host with a big chunk trace, the snapshot hooks of _replay_chunks
    (inside / after per replayed chunk) and the in-replay snapshot API of Qwen36Model."""
    m = _fake_model(calls, trace_lengths=trace_lengths, bucket_for=lambda L: Qwen36Model._mask_bucket_for(L))
    m._prefix_cache = cache
    m.big_chunk_rows = lambda: big
    if snap_on:
        m.gdn_snap_active = lambda: True
    m.snap_rows_ok = lambda rows: snap_on
    m.gdn_snap_copy_fn = lambda k: ("snapcopy", k)
    m.prefill_trash_block = 0

    def _chunks(*a, **k):
        num_full, chunk = a[3], a[4]
        start, snap = k.get("start", 0), k.get("snap")
        calls.append(("chunks", num_full, a[5], start, k.get("reset_gdn", True), k.get("run_tail"), snap is not None))
        rows = num_full * chunk
        bg = big if (k.get("allow_big") and big) else 0
        n_big = rows // bg if bg else 0
        layout = [(start + i * bg, bg) for i in range(n_big)]
        layout += [(start + n_big * bg + j * chunk, chunk) for j in range((rows - n_big * bg) // chunk)]
        for cs, r in layout:
            inside = snap.inside(cs, r) if snap is not None else []
            calls.append(("replay_chunk", cs, r, tuple(p - cs for p in inside)))
            if snap is not None:
                snap.after(cs, r, inside)
        return "REPL"

    def _replay(tokens, pt, L, strict=True, chunk_start=0, carry_gdn=False, spec_inputs=None, snap_rows=(), bucket=None):
        calls.append(("replay", L, chunk_start, carry_gdn, tuple(snap_rows), bucket))
        tr = m._short_trace_for(L)
        return tr

    m._prefill_traced_chunked_tp = _chunks
    m._replay_short_trace = _replay
    return m


class _RecCache:
    """Wraps a GdnPrefixStateCache: records every save (pos, low, protect, copy source)."""

    def __init__(self, inner):
        self.inner, self.saves = inner, []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def save(self, tokens, pos, low_prio=False, protect=False, copy_fn=None):
        self.saves.append((int(pos), bool(low_prio), bool(protect), copy_fn))
        return self.inner.save(tokens, pos, low_prio=low_prio, protect=protect)


def _run_fast(Tp, points, trace_lengths, snap_on, s0=0, final_row=True, save_end=False, tokens=None):
    tokens = tokens or _prompt(Tp)
    calls = []
    cache = _RecCache(_cache(num_slots=16))
    model = _snap_model(calls, cache, trace_lengths, snap_on=snap_on)
    e = _engine(8, model=model)
    prompt = torch.tensor([tokens], dtype=torch.int32)
    captured = [False]
    src = (types.SimpleNamespace(restore=lambda slot: calls.append(("restore", slot))), 0) if s0 > 0 else None
    plan = (s0, src, points)
    out = MTPSpecEngine._prefill_fast(
        e,
        3,
        tokens,
        prompt,
        _row(),
        Tp,
        lambda *a: None,
        captured,
        plan=plan,
        mtp_prompt=[tokens],
        final_row=final_row,
        save_end=save_end,
    )
    return out, calls, cache.saves, e


def test_flag_off_keeps_the_legacy_split(fake_ttnn):
    """gdn_snap_active absent (flag off): the end point splits the final chunk, as today."""
    Tp = 32768
    pts = _points([24576, 32640], low=[24576], end=[32640])
    _, calls, saves, _ = _run_fast(Tp, pts, trace_lengths=(1920, 128), snap_on=False, s0=16384)
    kinds = [c for c in calls if c[0] in ("chunks", "replay")]
    # segments [16384, 24576) (1 big), [24576, 32640) (3 x 2048 + tail 1920), [32640, 32768) (tail 128)
    assert [c[0] for c in kinds] == ["chunks", "chunks", "replay", "replay"]
    assert [(p, low, prot, src) for p, low, prot, src in saves] == [(24576, True, False, None), (32640, False, True, None)]
    assert all(not c[-1] for c in calls if c[0] == "chunks")  # no snap kwarg reaches the model


def test_intrace_one_segment_final_8k_chunk(fake_ttnn):
    """The final 8K chunk [24576, 32768) of a 32K prompt: ONE big replay snapshotting 32640 inside it."""
    Tp = 32768
    pts = _points([24576, 32640], low=[24576], end=[32640])
    _, calls, saves, e = _run_fast(Tp, pts, trace_lengths=(1920, 128), snap_on=True, s0=16384)
    reps = [c for c in calls if c[0] in ("replay_chunk", "replay")]
    assert reps == [("replay_chunk", 16384, 8192, ()), ("replay_chunk", 24576, 8192, (32640 - 24576,))]
    # same positions, marks and order as the legacy saves; 24576 from the bound state, 32640 from snapshot 0
    assert [(p, low, prot) for p, low, prot, _ in saves] == [(24576, True, False), (32640, False, True)]
    assert saves[0][3] is None and saves[1][3] == ("snapcopy", 0)
    assert e.intrace_prefills == 1 and e.intrace_snapshots == 1


def test_intrace_resumed_join_one_short_replay(fake_ttnn):
    """+916 tokens after an 8.4K prefix: cap 8320 restored, learn point 8448 and end point 9088 inside ONE short
    replay [8320, 9236) instead of three (8320-8448, 8448-9088, 9088-9236)."""
    Tp = 9236
    pts = _points([8448, 9088], end=[9088])
    _, calls, saves, _ = _run_fast(Tp, pts, trace_lengths=(916, 128, 640, 148), snap_on=True, s0=8320)
    reps = [c for c in calls if c[0] == "replay"]
    assert reps == [("replay", 916, 8320, True, (128, 768), None)]
    assert [(p, prot, src) for p, _, prot, src in saves] == [(8448, False, ("snapcopy", 0)), (9088, True, ("snapcopy", 1))]


def test_intrace_falls_back_when_a_point_sits_in_an_eager_tail(fake_ttnn):
    """No short trace for the tail that would hold the end point: the legacy split serves the call."""
    Tp = 16384 + 3000
    pts = _points([16384 + 2944], end=[16384 + 2944])
    _, calls, saves, _ = _run_fast(Tp, pts, trace_lengths=(2944 - 2048, 56), snap_on=True, s0=16384)
    # legacy: [16384, 19328) = 1 x 2048 + tail 896, then [19328, 19384) tail 56
    assert [c[0] for c in calls if c[0] in ("chunks", "replay")] == ["chunks", "replay", "replay"]
    assert [s[3] for s in saves] == [None]


def test_intrace_falls_back_with_more_points_than_slots(fake_ttnn):
    Tp = 2048 + 1024
    pts = _points([2048 + 128, 2048 + 256, 2048 + 384, 2048 + 512, 2048 + 640])
    _, calls, saves, _ = _run_fast(Tp, pts, trace_lengths=tuple(range(1, 2048)), snap_on=True, s0=2048)
    assert sum(1 for c in calls if c[0] == "replay") == 6  # six legacy segments
    assert all(s[3] is None for s in saves)


def test_intrace_falls_back_on_trash_block_collision(fake_ttnn):
    """The request owns the trash block (a harness table): the short traces would decline, so the legacy split."""
    Tp = 9236
    pts = _points([8448, 9088], end=[9088])
    tokens = _prompt(Tp)
    calls = []
    cache = _RecCache(_cache(num_slots=16))
    model = _snap_model(calls, cache, trace_lengths=(916, 128, 640, 148), snap_on=True)
    model.prefill_trash_block = 7  # block 7 is one of the request's blocks in _row()
    e = _engine(8, model=model)
    run, choice = MTPSpecEngine._snap_run_plan(e, torch.tensor([tokens]), _row(), 8320, Tp, pts)
    assert run is None and choice is None


def test_intrace_keeps_the_split_when_it_is_cheaper(fake_ttnn):
    """A resumed join [8960, 10016) with its end point at 9856: one segment would round 1056 rows up to the 2048 bucket,
    the split (896 -> 1024, 160 -> 256) costs less, so the legacy split runs."""
    Tp = 10016
    pts = _points([9856], end=[9856])
    _, calls, saves, e = _run_fast(Tp, pts, trace_lengths=(1056, 896, 160), snap_on=True, s0=8960)
    assert [c[1] for c in calls if c[0] == "replay"] == [896, 160]
    assert [s[3] for s in saves] == [None] and e.intrace_declined == 1


def test_intrace_min_rows_keeps_the_split_for_small_bodies(fake_ttnn):
    """QWEN36_PREFILL_INTRACE_SNAP_MIN_ROWS above the 1024 bucket: the join's points cannot sit inside its short replay."""
    Tp = 9236
    pts = _points([8448, 9088], end=[9088])
    tokens = _prompt(Tp)
    calls = []
    cache = _RecCache(_cache(num_slots=16))
    model = _snap_model(calls, cache, trace_lengths=(916, 128, 640, 148), snap_on=True)
    model.snap_rows_ok = lambda rows: rows >= 2048
    e = _engine(8, model=model)
    assert MTPSpecEngine._snap_run_plan(e, torch.tensor([tokens]), _row(), 8320, Tp, pts) == (None, None)


def test_intrace_non_final_chunk_saves_end(fake_ttnn):
    """A non-final chunk with Option B save_end: the end save after the last replay, from the bound state."""
    Tp = 16384 + 8192
    pts = _points([16384 + 8192 - 128])  # a forced point inside the second chunk's replay
    _, calls, saves, _ = _run_fast(Tp, pts, trace_lengths=(), snap_on=True, s0=16384, final_row=False, save_end=True)
    assert [c for c in calls if c[0] == "replay_chunk"] == [("replay_chunk", 16384, 8192, (8064,))]
    assert [(p, src) for p, _, _, src in saves] == [(24448, ("snapcopy", 0)), (24576, None)]


def test_snaprun_orders_inside_then_end():
    log = []
    cache = types.SimpleNamespace(save=lambda t, p, low_prio, protect, **k: log.append((p, low_prio, protect, k)))
    model = types.SimpleNamespace(gdn_snap_copy_fn=lambda k: ("c", k))
    run = _SnapRun(model, cache, "TOK", _points([128, 384, 8192], low=[8192], end=[384]))
    assert run.inside(0, 8192) == [128, 384]
    run.after(0, 8192, [128, 384])
    assert log == [
        (128, False, False, {"copy_fn": ("c", 0)}),
        (384, False, True, {"copy_fn": ("c", 1)}),
        (8192, True, False, {}),
    ]
    assert run.saved == [128, 384, 8192] and run.n_inside == 2


# --------------------------------------------------------------------------- #
# GdnPrefixStateCache.save with a copy function; Qwen36Model._snap_stage
# --------------------------------------------------------------------------- #
def test_cache_save_with_copy_fn(fake_ttnn):
    cache = _cache(num_slots=2)
    got = []
    toks = torch.tensor(_prompt(512))
    assert cache.save(toks, 256, copy_fn=lambda dn, rec, carry: got.append((rec, carry)))
    assert got == [(("rec", 1), ("carry", 1))] and fake_ttnn["copy"] == []  # nothing copied from the bound state
    assert cache.lookup(toks, 256) == (256, 1)
    cache.save(toks, 384)  # no copy_fn: the bound state, as before
    assert fake_ttnn["copy"] == [("bound_rec", ("rec", 0)), ("bound_carry", ("carry", 0))]


def test_snap_stage_copies_only_on_change(monkeypatch):
    staged = []
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **k: ("host", tuple(t.reshape(-1).tolist())))
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda h, d: staged.append((d, h[1])))
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: "rep")
    m = types.SimpleNamespace(_gdn_snap={"rec_idx": "REC", "conv_idx": "CONV"}, _snap_staged=(), mesh_device=None)
    assert Qwen36Model._snap_stage(m, []) == []  # nothing staged and nothing wanted: no copy
    refs = Qwen36Model._snap_stage(m, [256])
    assert len(refs) == 2 and [d for d, _ in staged] == ["REC", "CONV"]
    assert staged[0][1][:2] == (7, 0x7FFFFFFF) and staged[1][1][:3] == (253, 254, 255)
    assert Qwen36Model._snap_stage(m, [256]) == []  # unchanged
    Qwen36Model._snap_stage(m, [])  # cleared for the next replay
    assert staged[-2][1] == (0x7FFFFFFF,) * 8 and m._snap_staged == ()
    m_off = types.SimpleNamespace(_gdn_snap=None)
    assert Qwen36Model._snap_stage(m_off, [256]) == []

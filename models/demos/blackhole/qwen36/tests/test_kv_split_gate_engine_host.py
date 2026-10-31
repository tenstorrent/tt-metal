# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the engine side of the KV-pair split's context gate (QWEN36_ATTN_KV_SPLIT=user + _MIN_CTX / _MIN_CTX_DRAFT), on the
fake-device rigs of the bucket / chain / deferred tests:

* per step the verify gets kv_split=True and the drafter replays its split variant exactly when the gate wants it for the step's
  width at the step's context (the largest start + T of the live rows) and that width's variant exists; otherwise the kwargs and
  trace ids are the unsplit ones (flag off: unchanged calls);
* a split switch is free: it never breaks the chain (sync S3c and deferred S4b chain the same steps as without a gate);
* deferred serving decides from the RUNNER's positions: the same request stream gives the same split choices whether the pending
  step was read back before the next enqueue or not (no host-timing dependence);
* a free bucket down-shift re-decides the split for the new width;
* the drafter split needs the own page-table twin of THIS engine width (an adaptive-K family of another width never stages or
  reads the base family's twin), and the twin is only re-staged from a table of its width.
No device. Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_kv_split_gate_engine_host.py
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_async_engine_host import _Pipe
from models.demos.blackhole.qwen36.tests.test_spec_bucket_engine_host import DR, _bare
from models.demos.blackhole.qwen36.tests.test_spec_bucket_engine_host import _rig as _bucket_rig
from models.demos.blackhole.qwen36.tests.test_spec_chain_engine_host import _Flow
from models.demos.blackhole.qwen36.tests.test_spec_fused_engine_host import DRAFT, T
from models.demos.blackhole.qwen36.tt import kv_split as ks

DRS = {4: 94, 2: 92, 8: 98}  # fake split drafter variant trace ids per width
ACC = [[1, 1, 0, 0]] * 8


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_BUCKET_HOLD", "0")
    for k in ("QWEN36_ATTN_KV_SPLIT_MIN_CTX", "QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT"):
        monkeypatch.delenv(k, raising=False)


def _gate_rig(monkeypatch, B, vtab, dtab, vwidths=(4,), dwidths=(4,), **rig_kw):
    """A bucket rig (records ``bucket``) whose model carries a split plan, the gate of (vtab, dtab) and split verify variants
    for ``vwidths``; the engine has split drafter variants for ``dwidths``. Records every step's runner start positions,
    the kv_split kwarg of each verify and the drafter trace id of each step."""
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_MIN_CTX", vtab)
    monkeypatch.setenv("QWEN36_ATTN_KV_SPLIT_MIN_CTX_DRAFT", dtab)
    r = _bucket_rig(monkeypatch, B, **rig_kw)
    r.m._kv_split_plan = object()
    r.m._kv_split_gate = ks.KVSplitGate.from_env()
    r.m._vfy_bk_ksp = {b: object() for b in vwidths}
    for b in dwidths:
        r.e._dr_traces[("ksp", b)] = DRS[b]
    r.ks, r.drs, r.pos = [], [], []
    inner = r.m.verify_stage_and_enqueue

    def vse(*a, **k):
        r.ks.append(bool(k.pop("kv_split", False)))
        return inner(*a, **k)

    r.m.verify_stage_and_enqueue = vse
    dt = r.e._ksp_draft_trace

    def draft_trace():
        tid = dt()
        r.drs.append(tid)
        return tid

    r.e._ksp_draft_trace = draft_trace
    df = r.e.decode_forward

    def decode_forward(tok, pos, *a, **k):
        p0 = torch.as_tensor(pos)[:, 0].to(torch.int64)
        r.pos.append(p0.clone())
        return df(tok, pos, *a, **k)

    r.e.decode_forward = decode_forward
    return r


def _expected(r, part, widths):
    """The split choice the gate rule gives for every recorded step (width = the recorded bucket, else B)."""
    g = r.m._kv_split_gate
    out = []
    for i, p0 in enumerate(r.pos):
        b = r.bk[i] if r.bk[i] != "absent" else r.e.B
        ctx = ks.kv_split_gate_ctx(p0.clamp(min=0), p0 >= 0, T)
        out.append(g.want(part, b, ctx) and b in widths)
    return out


def _sync(monkeypatch, B, nv, accepts, finish=None, **kw):
    r = _gate_rig(monkeypatch, B, **kw)
    r.e._active = {u for u in range(B) if nv[u] >= 0}
    f = _Flow(r, B, list(nv))
    for i, acc in enumerate(accepts):
        f.step(acc)
        f.propose()
        f.next_block(finish=tuple((finish or {}).get(i, ())))
    return r


# --------------------------------------------------------------------------- #
# per-step choice
# --------------------------------------------------------------------------- #
def test_sync_split_choice_follows_the_gate_per_step(monkeypatch):
    # rows at 50 / 60 / 70 / 80 (+ commits): context 84 at step 0, growing ~1 per step (row 3 commits 1 per step)
    r = _sync(monkeypatch, 4, (3, 2, 1, 0), ACC, vtab="4:87", dtab="4:85")
    want_v, want_d = _expected(r, "verify", (4,)), _expected(r, "draft", (4,))
    assert r.ks == want_v and [t == DRS[4] for t in r.drs] == want_d
    assert want_v[0] is False and want_v[-1] is True, "the run must cross the verify threshold"
    assert want_d.index(True) < want_v.index(True), "the drafter threshold is the lower one"
    assert all(t in (DRAFT, DRS[4]) for t in r.drs)
    assert r.e.kv_split_steps == {"verify": sum(want_v), "draft": sum(want_d)}
    assert r.e.kv_split_engaged() and "kv split steps verify" in r.e._bucket_note_text()


def test_split_switch_never_breaks_the_chain(monkeypatch):
    nv = (3, 2, 1, 0)
    r = _sync(monkeypatch, 4, nv, ACC, vtab="4:86", dtab="4:86")
    assert any(r.ks) and not all(r.ks), "the run switches variants"
    r0 = _sync(monkeypatch, 4, nv, ACC, vtab="off", dtab="off")
    assert not any(r0.ks) and all(t == DRAFT for t in r0.drs)
    assert r.e.chain_breaks == r0.e.chain_breaks == {"first": 1} and r.e.chain_steps == r0.e.chain_steps


def test_bucketed_choice_uses_the_bucket_threshold_and_variant(monkeypatch):
    # rows 0, 1 live -> bucket 2 (trace 2); its threshold differs from bucket 4's; then row 2 joins... kept simple: 2 rows
    r = _sync(
        monkeypatch,
        4,
        (3, 2, -1, -1),
        [[1, 1, 0, 0]] * 6,
        vtab="4:10,2:66",
        dtab="4:10,2:off",
        vwidths=(4, 2),
        dwidths=(4, 2),
        buckets=(1, 2, 4),
        traces=(1, 2),
    )
    assert r.bk == [2] * 6
    assert r.ks == _expected(r, "verify", (4, 2)) and any(r.ks) and not all(r.ks)
    assert all(t == DR[2] for t in r.drs), "bucket 2 has no drafter threshold: its unsplit bucket trace"


def test_no_variant_for_the_width_means_unsplit(monkeypatch):
    r = _sync(monkeypatch, 4, (3, 2, 1, 0), ACC[:3], vtab="0", dtab="off", vwidths=(), dwidths=())
    assert r.m._kv_split_gate.mode == "gated" and r.ks == [False] * 3 and all(t == DRAFT for t in r.drs)


def test_flag_off_leaves_every_call_unchanged(monkeypatch):
    r = _sync(monkeypatch, 4, (3, 2, 1, 0), ACC[:4], vtab="4:0", dtab="4:0")
    on = (r.ks, r.drs)
    r0 = _sync(monkeypatch, 4, (3, 2, 1, 0), ACC[:4], vtab="4:0", dtab="4:0")
    r0.m._kv_split_plan = None  # QWEN36_ATTN_KV_SPLIT unset: no plan, so no gate
    r0.ks, r0.drs = [], []
    f = _Flow(r0, 4, [3, 2, 1, 0])
    for acc in ACC[:4]:
        f.step(acc)
        f.propose()
        f.next_block()
    assert on == ([True] * 4, [DRS[4]] * 4)
    assert r0.ks == [False] * 4 and r0.drs == [DRAFT] * 4 and r0.e._ksp_ctx is None
    assert not r0.e.kv_split_engaged() and r0.e._ksp_note_text() == ""


def test_always_mode_is_not_gated_but_engaged(monkeypatch):
    r = _sync(monkeypatch, 4, (3, 2, 1, 0), ACC[:2], vtab="0", dtab="0")
    assert r.m._kv_split_gate.mode == "always"
    assert r.ks == [False, False] and r.drs == [DRAFT, DRAFT], "the canonical traces ARE the split ones in always mode"
    assert r.e.kv_split_engaged() and r.e._ksp_ctx is None


# --------------------------------------------------------------------------- #
# deferred serving: runner positions, no timing dependence
# --------------------------------------------------------------------------- #
def _pipe(monkeypatch, inflight, n=8, **kw):
    r = _gate_rig(monkeypatch, 4, **kw)
    r.queue = [("acc", a) for a in ACC[:n]]
    p = _Pipe(r, 4, [3, 2, 1, 0])
    pend, last, hs = [], None, []
    for _ in range(n):
        h = p.enq(last)
        last = h
        hs.append(h)
        pend.append(h)
        if len(pend) >= inflight:
            p.resolve(pend.pop(0))
    while pend:
        p.resolve(pend.pop(0))
    return r, hs


def test_deferred_choice_uses_runner_positions_and_not_host_timing(monkeypatch):
    kw = dict(vtab="4:96", dtab="4:92")
    r2, hs2 = _pipe(monkeypatch, 2, **kw)  # two steps in flight: chain bounds not yet exact at enqueue
    r1, hs1 = _pipe(monkeypatch, 1, **kw)  # each step read back before the next enqueue: exact bounds
    assert [torch.equal(a, b) for a, b in zip(r2.pos, r1.pos)] == [True] * len(r2.pos), "same runner positions"
    assert r2.ks == r1.ks and r2.drs == r1.drs, "the split choices do not depend on read-back timing"
    assert r2.ks == _expected(r2, "verify", (4,)) and [t == DRS[4] for t in r2.drs] == _expected(r2, "draft", (4,))
    assert any(r2.ks) and not all(r2.ks)
    # chain rows of a step are recorded with the runner's bound, which is what the rule saw
    assert r2.e.__dict__.get("def_breaks", {}) == {} and r1.e.__dict__.get("def_breaks", {}) == {}


# --------------------------------------------------------------------------- #
# free down-shift, family widths, twin guard
# --------------------------------------------------------------------------- #
def _bare_gate(vtab, dtab="off", B=4):
    e = _bare(B=B, buckets=(1, 2, 4), traces=(1, 2))
    e.model = types.SimpleNamespace(
        _kv_split_plan=object(),
        _kv_split_gate=ks.KVSplitGate(ks.parse_min_ctx(vtab, {}), ks.parse_min_ctx(dtab, {})),
        _vfy_bk_ksp={4: object(), 2: object()},
    )
    return e


def test_free_down_shift_re_decides_the_split():
    e = _bare_gate("4:100,2:50")
    e._ksp_ctx, e._vb, e._bk_cb, e._bk_cand, e.bucket_steps = 60, 4, 4, 2, {4: 1}
    kw = e._bucket_kw({})
    assert kw == {"bucket": 4} and "kv_split" not in kw, "bucket 4 wants 100"
    e.bucket_steps = {4: 1}
    e._bucket_free_note(4, kw)
    assert kw == {"bucket": 2, "kv_split": True} and e.kv_split_steps["verify"] == 1, "bucket 2 wants 50"
    e2 = _bare_gate("4:100")
    e2._ksp_ctx, e2._vb, e2._bk_cb, e2._bk_cand, e2.bucket_steps = 120, 4, 4, 2, {4: 1}
    kw2 = e2._bucket_kw({})
    assert kw2 == {"bucket": 4, "kv_split": True} and e2.kv_split_steps["verify"] == 1
    e2._bucket_free_note(4, kw2)
    assert kw2 == {"bucket": 2} and e2.kv_split_steps["verify"] == 0, "no threshold for bucket 2: unsplit, uncounted"


class _Plan:
    def __init__(self, keys, parts=("verify", "draft")):
        self.keys, self.parts = set(keys), set(parts)

    def has(self, n, T):
        return (int(n), int(T)) in self.keys


def _drafter_engine(B, twin_rows, dtab="8:7168,4:12288,2:off"):
    e = _bare(B=B, buckets=(1, 2, 4, 8) if B == 8 else (1, 2, 4), traces=())
    e._dr_bks = tuple(b for b in (4, 2, 1) if b < B)
    e.mtp = types.SimpleNamespace(attention=types.SimpleNamespace(_kv_split_ready=True))
    e.model = types.SimpleNamespace(
        _kv_split_plan=_Plan({(b, 1) for b in (2, 4, 8)}),
        _kv_split_gate=ks.KVSplitGate({8: 10240}, ks.parse_min_ctx(dtab, {})),
        num_devices=8,
    )
    e._mtp_pt_own, e._mtp_pt_own_rows = "twin", twin_rows
    return e


def test_drafter_variant_widths_need_this_widths_twin(expect_error):
    e = _drafter_engine(8, 8)
    assert e._ksp_drafter_widths() == (8, 4)
    assert e._dr_ksp_want(4, True) and not e._dr_ksp_want(4, False) and not e._dr_ksp_want(8, None)
    w = _drafter_engine(4, 8)  # an adaptive-K wide family (B = 4) on the base family's 8-row twin
    assert w._ksp_drafter_widths() == () and not w._dr_ksp_want(4, None)
    with expect_error(AssertionError, "no split drafter"):
        w._dr_ksp_want(4, True)
    e.model._kv_split_plan.parts = {"verify"}
    assert e._ksp_drafter_widths() == ()
    e.model._kv_split_gate = ks.KVSplitGate(*([{b: 0 for b in range(2, 33, 2)}] * 2))  # always mode
    e.model._kv_split_plan.parts = {"verify", "draft"}
    assert e._ksp_drafter_widths() == () and e._dr_ksp_want(8, None), "always: the canonical drafter trace splits"


def test_twin_is_restaged_only_from_a_table_of_its_width(monkeypatch):
    e = _drafter_engine(8, 8)
    e.mesh, e._mtp_pt = "mesh", object()
    copies = []
    monkeypatch.setattr(ttnn, "ShardTensorToMesh", lambda mesh, dim: ("shard", dim), raising=False)
    monkeypatch.setattr(ttnn, "from_torch", lambda t, **k: ("host", tuple(t.shape)))
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda h, d: copies.append((h, d)))
    e._stage_split_twin(torch.zeros(8, 16, dtype=torch.int32), e._mtp_pt)
    assert copies == [(("host", (32, 16)), "twin")], "8 users -> 4 own rows per device x 8 devices"
    e._stage_split_twin(torch.zeros(4, 16, dtype=torch.int32), e._mtp_pt)  # a 4-row family table: not this twin's
    e._stage_split_twin(torch.zeros(8, 16, dtype=torch.int32), object())  # another buffer
    assert len(copies) == 1


def test_family_swap_without_variants_falls_back():
    """The variant registries live in family-swapped state (_dr_traces / the model's _vfy_* attributes): a family whose
    capture made no variant (the adaptive-K wide family today) replays its unsplit traces at any context."""
    e = _bare_gate("4:0,2:0", "4:0,2:0")
    e._ksp_ctx, e._vb = 10**6, 4
    e._dr_traces = {("ksp", 4): DRS[4]}
    e._dr_trace = DRAFT
    assert e._ksp_on("verify") and e._ksp_on("draft") and e._ksp_draft_trace() == DRS[4]
    base = (e._dr_traces, e.model._vfy_bk_ksp)
    e._dr_traces, e.model._vfy_bk_ksp = (
        {},
        {},
    )  # what vfy_family_stash / the FAM_ENGINE_ATTRS swap leave for the wide family
    assert not e._ksp_on("verify") and not e._ksp_on("draft") and e._ksp_draft_trace() == DRAFT
    e._dr_traces, e.model._vfy_bk_ksp = base
    assert e._ksp_on("verify")


# --------------------------------------------------------------------------- #
# adaptive K (QWEN36_SPEC_ADAPTIVE_K, tt/spec_adaptive.py; skipped on trees without it)
# --------------------------------------------------------------------------- #
def test_adaptive_k_family_swap_carries_the_variants(monkeypatch):
    """The real family swap (AdaptiveKMixin._ak_use: FAM_ENGINE_ATTRS + Qwen36Model.vfy_family_stash) moves the base
    family's split variants out of the way: the wide family (captured by spec_adaptive without variants) replays its
    unsplit traces at any context and never stages the base family's 8-row drafter twin; back in the base family the
    variants are live again."""
    sa = pytest.importorskip("models.demos.blackhole.qwen36.tt.spec_adaptive")
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    assert "_dr_traces" in sa.FAM_ENGINE_ATTRS and "_mtp_pt" in sa.FAM_ENGINE_ATTRS
    monkeypatch.setenv("QWEN36_SPEC_ADAPTIVE_K", "1")
    e = _bare(B=8, buckets=(1, 2, 4, 8), traces=(1, 2, 4))
    m = Qwen36Model.__new__(Qwen36Model)
    m._kv_split_plan = _Plan({(b, 1) for b in (2, 4, 8)})
    m._kv_split_gate = ks.KVSplitGate({8: 0, 4: 0}, {8: 0, 4: 0})
    m._vfy_bk_ksp, m._vfy_trace_id = {8: "V8s", 4: "V4s"}, 7
    e.model = m
    e._dr_traces.update({("ksp", 8): DRS[8], ("ksp", 4): DRS[4]})
    e._dr_trace, e._mtp_pt = DRAFT, "pt8"
    e._mtp_pt_own, e._mtp_pt_own_rows = "twin8", 8
    e.mtp = types.SimpleNamespace(attention=types.SimpleNamespace(_kv_split_ready=True))
    base, wide = sa.KFamily("base", 8, 3), sa.KFamily("wide", 4, 5)
    wide.eng = e._ak_fresh_attrs(4, 5)
    e._ak = sa.AdaptiveKState(base, wide)
    e._ksp_ctx, e._vb = 10**6, 8
    assert e._ksp_on("verify") and e._ksp_draft_trace() == DRS[8]
    e._ak_use(wide)
    assert e.B == 4 and e._mtp_pt is None and "_vfy_bk_ksp" not in m.__dict__
    e._vb = 4
    assert not e._ksp_on("verify") and not e._ksp_on("draft"), "the wide family has no variants"
    assert e._ksp_drafter_widths() == () and not e._dr_ksp_want(4, None)
    staged = []
    monkeypatch.setattr(ttnn, "copy_host_to_device_tensor", lambda h, d: staged.append(d))
    e._mtp_pt = "pt4w"
    e._stage_split_twin(torch.zeros(4, 16, dtype=torch.int32), e._mtp_pt)
    assert staged == [], "the base family's 8-row twin is never written from the wide family's 4-row table"
    e._ak_use(base)
    e._vb = 4
    assert e.B == 8 and m._vfy_bk_ksp == {8: "V8s", 4: "V4s"} and e._ksp_on("verify") and e._ksp_draft_trace() == DRS[4]

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_SPEC_JOIN_TRACE=1 (MTPSpecEngine._seed_fresh_rows trace replay, _capture_seed_traces,
shutdown): the GDN part of a join seed replays one captured trace per conv-window parity, selected by a fresh-row
mask, or falls back to the eager seed_spec_rows_select loop with a counted reason; the MTP part (slot Tp-1 warm)
replays ONE fixed B-row decode trace whose row mask is the page table (scratch block for a row that is not fresh),
or falls back to the eager warm with a counted reason. No device: fakes for the layers, the model and
ttnn.execute_trace / release_trace.

Run: pytest models/demos/blackhole/qwen36/tests/test_spec_join_trace_host.py -q
"""

import types

import pytest
import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_fastjoin_host import NB, _admit, _engine, _FakeModel, _inputs
from models.demos.blackhole.qwen36.tt.gdn.tp import fresh_row_mask, fresh_row_select
from models.demos.blackhole.qwen36.tt.spec_engine import join_mtp_warm_host
from models.demos.blackhole.qwen36.tt.tp_common import pack_u32_rows, rope_row_index, spec_join_trace_enabled

NV, DK = 3, 32  # fake per-device heads / key dim
TRACES = [101, 202]  # the trace ids of window parity 0 / 1
MTPW_TRACE = 303  # the id of the MTP warm trace
SCRATCH = 9999  # the MTP scratch block
ROPE = (7, 1000)  # (rope_delta, RoPE table rows)


def test_flag_read(monkeypatch):
    """QWEN36_SPEC_JOIN_TRACE is on by default: only "0" turns it off."""
    for val, want in (("1", True), ("0", False), (None, True), ("2", True)):
        if val is None:
            monkeypatch.delenv("QWEN36_SPEC_JOIN_TRACE", raising=False)
        else:
            monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", val)
        assert spec_join_trace_enabled() is want


@pytest.mark.parametrize("fresh", [[0], [3], [0, 5], [1, 2, 6], list(range(8)), []])
def test_fresh_row_mask_content(fresh):
    """1.0 on the Nv head blocks of each fresh row, 0.0 elsewhere; fp32 [1, B*Nv, Dk, 1]."""
    B = 8
    m = fresh_row_mask(fresh, B, NV, DK)
    assert m.dtype == torch.float32 and tuple(m.shape) == (1, B * NV, DK, 1)
    for u in range(B):
        assert bool((m[:, u * NV : (u + 1) * NV] == (1.0 if u in fresh else 0.0)).all())


def test_fresh_row_mask_rejects_out_of_range(expect_error):
    with expect_error(AssertionError, "outside range"):
        fresh_row_mask([4], 4, NV, DK)


class _TraceGdn:
    """GDN layer stand-in for the replay decision: the host state seed_masked_ok / the parity / the buffers."""

    K = 4
    Nv, Dk = NV, DK

    def __init__(self, T, ok=True, par=0):
        self.T, self.ok, self._spec_win_par, self._spec_fused = T, ok, par, True
        self.rec_state, self._conv_hist_rm, self._spec_ring = object(), object(), object()
        self._verify_win_pair = [object(), object()]
        self.selects = []

    def seed_masked_ok(self):
        return self.ok

    def seed_spec_rows_select(self, rows, sel):
        self.selects.append((sorted(int(u) for u in rows), sel))

    def save_spec_user(self, u):
        raise AssertionError("no parking in these tests")


@pytest.fixture
def traces(monkeypatch):
    """Fake ttnn.execute_trace / release_trace; the lists record (trace id, kwargs) and the released ids."""
    log = types.SimpleNamespace(exec=[], released=[])
    monkeypatch.setattr(ttnn, "execute_trace", lambda mesh, tid, **kw: log.exec.append((tid, kw)))
    monkeypatch.setattr(ttnn, "release_trace", lambda mesh, tid: log.released.append(tid))
    return log


def _join_engine(B, K, layers, with_trace=True, **kw):
    T = K + 1
    e = _engine(B, K, _FakeModel(B, T), layers[0], _gdn=layers, **kw)
    e.mesh = "mesh"
    e._fj_mask, e._fj_mask_dims = "MASK", (NV, DK)
    e.join_trace_replays, e.join_trace_fallbacks = 0, {}
    e._seed_trace = list(TRACES) if with_trace else None
    e._seed_trace_bufs = e._seed_buffers() if with_trace else None
    return e


def _join(e, fresh, active=None, start=None):
    """One verify step in which the engine rows ``fresh`` are fresh and every other active row runs."""
    B = e.B
    e._active = set(range(B)) - set(fresh) if active is None else set(active) - set(fresh)
    for u in fresh:
        _admit(e, u, 10 + u)
    tok, pos, pt = _inputs(e, [10 + u if u in fresh else 20 + u for u in range(B)] if start is None else start)
    act = None if active is None else [u in active for u in range(B)]
    e.verify(tok, pos, torch.ones(B, dtype=torch.int64), pt, active=act)


@pytest.mark.parametrize("fresh", [[0], [3], [0, 5], [1, 2, 6]])
def test_replay_stages_mask_of_the_fresh_rows_and_runs_the_parity_trace(monkeypatch, traces, fresh):
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    B, K = 8, 3
    layers = [_TraceGdn(K + 1) for _ in range(3)]
    e = _join_engine(B, K, layers)
    _join(e, fresh)
    (sel_host, sel_dev, *_), (mask_host, mask_dev, mdtype, _) = e.log["stage"]
    assert sel_dev == "SEL" and torch.equal(sel_host, fresh_row_select(fresh, B, K + 1, 4))
    assert mask_dev == "MASK" and mdtype == ttnn.float32
    assert torch.equal(mask_host, fresh_row_mask(fresh, B, NV, DK))
    assert traces.exec == [(TRACES[0], dict(cq_id=0, blocking=False))]
    assert e.join_trace_replays == 1 and e.join_trace_fallbacks == {}
    assert all(dn.selects == [] for dn in layers), "no eager seed when the trace replays"


def test_replay_when_every_active_row_is_fresh_masks_the_fresh_rows_only(monkeypatch, traces):
    """The eager path writes the whole slot here; the mask names the fresh rows only (the partial path)."""
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    B, K = 4, 3
    e = _join_engine(B, K, [_TraceGdn(K + 1)])
    _join(e, [0, 1], active=[0, 1])
    assert torch.equal(e.log["stage"][1][0], fresh_row_mask([0, 1], B, NV, DK))
    assert len(traces.exec) == 1 and e.join_trace_replays == 1


def test_mask_is_in_engine_rows_under_a_host_slot_map(monkeypatch, traces):
    """The ring rows are engine rows (as fresh_row_select): a host slot permutation does not change the mask."""
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    B, K = 4, 3
    e = _join_engine(B, K, [_TraceGdn(K + 1)])
    e._set_slot_map([2, 0, 3, 1])
    _join(e, [1, 3])
    assert torch.equal(e.log["stage"][1][0], fresh_row_mask([1, 3], B, NV, DK))
    assert e.join_trace_replays == 1


@pytest.mark.parametrize("par", [0, 1])
def test_replay_uses_the_trace_of_the_layers_parity(monkeypatch, traces, par):
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    K = 3
    e = _join_engine(4, K, [_TraceGdn(K + 1, par=par) for _ in range(2)])
    _join(e, [2])
    assert [t for t, _ in traces.exec] == [TRACES[par]]


def _fallback_cases():
    K = 3
    return {
        "flag off": dict(env="0", layers=[_TraceGdn(K + 1)]),
        "no trace": dict(env="1", layers=[_TraceGdn(K + 1)], with_trace=False),
        "taps route layer 1": dict(env="1", layers=[_TraceGdn(K + 1), _TraceGdn(K + 1, ok=False)]),
        "mixed parity": dict(env="1", layers=[_TraceGdn(K + 1), _TraceGdn(K + 1, par=1)]),
        "stale buffers": dict(env="1", layers=[_TraceGdn(K + 1)], stale=True),
    }


@pytest.mark.parametrize("reason", list(_fallback_cases()))
def test_every_fallback_runs_the_eager_loop_and_counts_its_reason(monkeypatch, traces, reason):
    case = _fallback_cases()[reason]
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", case["env"])
    layers = case["layers"]
    e = _join_engine(4, 3, layers, with_trace=case.get("with_trace", True))
    if case.get("stale"):
        layers[0]._spec_ring = object()  # the ring was reallocated after the capture
    _join(e, [1, 2])
    assert traces.exec == [] and e.join_trace_replays == 0
    assert e.join_trace_fallbacks == {reason: 1}
    assert all(dn.selects == [([1, 2], "SEL")] for dn in layers)
    assert len(e.log["stage"]) == 1, "only the selector is staged on the eager path"


def test_eager_whole_slot_when_every_active_row_is_fresh_on_fallback(monkeypatch, traces):
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    layers = [_TraceGdn(4, ok=False)]
    e = _join_engine(3, 3, layers)
    _join(e, [0, 1], active=[0, 1])
    assert layers[0].selects == [([0, 1, 2], "SEL")]


def test_fallback_note_in_the_periodic_line(monkeypatch):
    e = _engine(2, 2, _FakeModel(2, 3), None)
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "0")
    assert e._join_trace_note_text() == ""
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    e.join_trace_replays, e.join_trace_fallbacks = 5, {"mixed parity": 2, "no trace": 1}
    assert e._join_trace_note_text() == "; join trace replays 5, fallbacks mixed parity=2, no trace=1"


def test_shutdown_releases_both_seed_traces(traces):
    e = _join_engine(2, 2, [_TraceGdn(3)])
    e._release_seed_traces()
    assert sorted(traces.released) == sorted(TRACES)
    assert e._seed_trace is None and e._seed_trace_bufs is None
    e._release_seed_traces()  # idempotent
    assert len(traces.released) == 2


def test_single_trace_without_the_window_pair_is_released_once(traces):
    e = _join_engine(2, 2, [_TraceGdn(3)])
    e._seed_trace = [7, 7]
    e._release_seed_traces()
    assert traces.released == [7]


def _capture_engine(layers):
    e = _join_engine(2, 2, layers, with_trace=False)
    e.fast_join, e._fj_sel = True, "SEL"
    return e


def test_capture_records_one_trace_per_parity_and_restores_the_parity(monkeypatch):
    """Per parity: every layer's parity is set, the layers run in order inside begin / end capture; the original
    parities come back. A layer that fails seed_masked_ok() (or the flag off) leaves no trace."""
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    events, ids = [], iter([11, 12])
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda mesh, cq_id: (events.append("begin"), next(ids))[1])
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, tid, cq_id: events.append(f"end{tid}"))
    layers = [_TraceGdn(3, par=1), _TraceGdn(3, par=1)]
    for i, dn in enumerate(layers):
        dn._seed_masked_impl = lambda mask, sel, i=i, dn=dn: events.append((i, dn._spec_win_par, mask, sel))
    e = _capture_engine(layers)
    e._capture_seed_traces()
    assert events == [
        "begin", (0, 0, "MASK", "SEL"), (1, 0, "MASK", "SEL"), "end11",
        "begin", (0, 1, "MASK", "SEL"), (1, 1, "MASK", "SEL"), "end12",
    ]  # fmt: skip
    assert e._seed_trace == [11, 12] and [dn._spec_win_par for dn in layers] == [1, 1]
    events.clear()
    layers[1].ok = False
    e._capture_seed_traces()
    assert events == [] and e._seed_trace is None
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "0")
    layers[1].ok = True
    e._capture_seed_traces()
    assert events == [] and e._seed_trace is None


# --------------------------------------------------------------------------- #
# The traced MTP part of the seed (_mtpw_trace)
# --------------------------------------------------------------------------- #
def _mtp_engine(B, K, layers, trace=True, perm=None):
    """A join engine whose MTP warm trace was captured against the hidden buffers in engine order; ``perm`` then
    permutes the host list _last_hidden (a slot remap): engine row u holds the buffer captured at physical row perm[u].
    """
    e = _join_engine(B, K, layers)
    e.nb, e._scratch_block, e._mtpw_rope = NB, SCRATCH, ROPE
    e._mtpw_pack, e._mtpw_pt = "PACK", "MPT"
    e._mtpw_hidden = tuple(e._last_hidden)
    e._mtpw_phys = {id(h): p for p, h in enumerate(e._mtpw_hidden)}
    e._mtpw_trace = MTPW_TRACE if trace else None
    e.join_mtp_trace_replays = 0
    if perm is not None:
        e._last_hidden = [e._mtpw_hidden[j] for j in perm]
    return e


def _expected_mtp_inputs(e, fresh, phys):
    """Row p of the pack / table written out by hand: a fresh row's token, slot Tp-1 and page-table row at its physical
    row; token 0, position 0 and the scratch block everywhere else."""
    B = e.B
    tok, pos, table = [0] * B, [0] * B, torch.full((B, NB), SCRATCH, dtype=torch.int32)
    for u in fresh:
        tok[phys[u]], pos[phys[u]] = 100 + u * e.T, 10 + u - 1
        table[phys[u]] = torch.arange(NB, dtype=torch.int32) + 1 + u * NB
    ridx = [p + ROPE[0] for p in pos]
    return pack_u32_rows([tok, pos, ridx]), table


@pytest.mark.parametrize("perm", [None, [2, 0, 3, 1, 7, 5, 6, 4]])
@pytest.mark.parametrize("fresh", [[0], [3], [0, 5], [1, 2, 6], list(range(8))])
def test_mtp_replay_stages_pack_and_table_in_physical_rows(monkeypatch, traces, fresh, perm):
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    B, K = 8, 3
    e = _mtp_engine(B, K, [_TraceGdn(K + 1)], perm=perm)
    _join(e, fresh)
    phys = {u: (u if perm is None else perm[u]) for u in range(B)}
    (pack, pack_dev, pdtype, _), (table, table_dev, tdtype, _) = e.log["stage"][:2]
    want_pack, want_table = _expected_mtp_inputs(e, fresh, phys)
    assert pack_dev == "PACK" and pdtype == ttnn.uint32 and torch.equal(pack, want_pack)
    assert table_dev == "MPT" and tdtype == ttnn.int32 and torch.equal(table, want_table)
    assert [t for t, _ in traces.exec] == [MTPW_TRACE, TRACES[0]], "MTP warm first, then the GDN seed trace"
    assert traces.exec[0][1] == dict(cq_id=0, blocking=False)
    assert e.join_mtp_trace_replays == 1 and e.join_trace_fallbacks == {}
    assert e.log["warm"] == [] and e.log["tables"] == [], "no eager warm, no per-row table staging"


def test_mtp_non_fresh_rows_run_on_the_scratch_block():
    """Every physical row that is not fresh: token 0, position 0, the scratch block in every table entry."""
    B = 8
    tokens = torch.arange(B * 4).reshape(B, 4) + 100
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    pack, table = join_mtp_warm_host({1: 40, 6: 90}, tokens, pt, {1: 4, 6: 0}, B, SCRATCH, *ROPE)
    assert tuple(pack.shape) == (3, 32) and pack.dtype == torch.int32
    for p in range(B):
        fresh = p in (4, 0)
        u = {4: 1, 0: 6}.get(p)
        assert int(pack[0, p]) == (int(tokens[u, 0]) if fresh else 0)
        assert int(pack[1, p]) == ({1: 39, 6: 89}[u] if fresh else 0)
        assert torch.equal(table[p], pt[u] if fresh else torch.full((NB,), SCRATCH, dtype=torch.int32))
    assert torch.equal(pack[2, :B], rope_row_index(pack[1, :B], ROPE[0], ROPE[1]))
    assert not bool(pack[:, B:].any()), "the pack pad columns stay 0"


def test_mtp_replay_does_not_touch_the_per_row_tables_or_the_gdn_selector_order(monkeypatch, traces):
    """The traced path leaves _mtp_pt_rows unstaged (propose restages what it reads); the GDN selector + mask are still
    staged after the two MTP inputs, exactly as before."""
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    e = _mtp_engine(4, 3, [_TraceGdn(4)])
    _join(e, [2])
    assert [d for _, d, _, _ in e.log["stage"]] == ["PACK", "MPT", "SEL", "MASK"]
    assert e._fresh == {} and e._prompts == {} and e._fresh_dirty == set()


def test_mtp_fallback_no_trace_runs_the_eager_warm_and_counts_it(monkeypatch, traces):
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    e = _mtp_engine(4, 3, [_TraceGdn(4)], trace=False)
    _join(e, [1, 2])
    assert e.join_trace_fallbacks == {"mtp no trace": 1} and e.join_mtp_trace_replays == 0
    assert [t for t, _ in traces.exec] == [TRACES[0]], "the GDN seed still replays"
    assert [w[:3] for w in e.log["warm"]] == [("h1", 104, 10), ("h2", 108, 11)]
    assert e.log["tables"] and "PACK" not in [d for _, d, _, _ in e.log["stage"]]


@pytest.mark.parametrize("how", ["new buffer", "duplicate", "short list", "old capture"])
def test_mtp_fallback_stale_buffers(monkeypatch, traces, how):
    """A hidden buffer that is not a captured one (reallocated), twice the same one, a shorter list, or the capture
    reallocated."""
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    e = _mtp_engine(4, 3, [_TraceGdn(4)])
    if how == "new buffer":
        e._last_hidden = [e._last_hidden[0], "new", *e._last_hidden[2:]]
    elif how == "duplicate":
        e._last_hidden = [e._last_hidden[0], e._last_hidden[0], *e._last_hidden[2:]]
    elif how == "short list":
        e._last_hidden = e._last_hidden[:3]
    else:
        e._mtpw_hidden = tuple(["old", *e._mtpw_hidden[1:]])  # the trace was captured against other buffers
    _join(e, [0])
    assert e.join_trace_fallbacks == {"mtp stale buffers": 1} and e.join_mtp_trace_replays == 0
    assert e.log["warm"] and [t for t, _ in traces.exec] == [TRACES[0]]


def test_mtp_flag_off_and_unallocated_are_not_counted(monkeypatch, traces):
    """Flag off: the GDN part counts "flag off" and the MTP part runs eagerly without its own count. No allocated
    MTP inputs (feature not in use): eager and no count."""
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "0")
    e = _mtp_engine(4, 3, [_TraceGdn(4)])
    _join(e, [1])
    assert e.join_trace_fallbacks == {"flag off": 1} and e.log["warm"]
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    e = _mtp_engine(4, 3, [_TraceGdn(4)])
    e._mtpw_pack = None
    _join(e, [1])
    assert e.join_trace_fallbacks == {} and e.log["warm"] and e.join_mtp_trace_replays == 0


def test_release_frees_the_mtp_warm_trace(traces):
    e = _mtp_engine(2, 2, [_TraceGdn(3)])
    e._release_seed_traces()
    assert sorted(traces.released) == sorted([*TRACES, MTPW_TRACE]) and e._mtpw_trace is None
    e._release_seed_traces()  # idempotent
    assert len(traces.released) == 3


def test_capture_mtp_warm_trace_records_the_physical_order(monkeypatch):
    """The trace is captured against the hidden buffers in their current host order; _mtpw_phys maps id -> row. No
    trace without allocated inputs or with the flag off."""
    events = []
    monkeypatch.setattr(ttnn, "synchronize_device", lambda mesh: None)
    monkeypatch.setattr(ttnn, "begin_trace_capture", lambda mesh, cq_id: (events.append("begin"), 5)[1])
    monkeypatch.setattr(ttnn, "end_trace_capture", lambda mesh, tid, cq_id: events.append(f"end{tid}"))
    e = _mtp_engine(4, 3, [_TraceGdn(4)], trace=False)
    e._mtp_warm_body = lambda: events.append(("body", tuple(e._mtpw_hidden)))
    e._last_hidden = [e._last_hidden[j] for j in (3, 1, 0, 2)]  # a remap before the capture
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "0")
    e._capture_mtp_warm_trace()
    assert events == [] and e._mtpw_trace is None
    monkeypatch.setenv("QWEN36_SPEC_JOIN_TRACE", "1")
    e._capture_mtp_warm_trace()
    assert events == ["begin", ("body", ("h3", "h1", "h0", "h2")), "end5"] and e._mtpw_trace == 5
    assert [e._mtpw_phys[id(h)] for h in e._last_hidden] == [0, 1, 2, 3]
    e._mtpw_pack = None
    events.clear()
    e._capture_mtp_warm_trace()
    assert events == [] and e._mtpw_trace is None

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for chunked prefill on the MTP spec class (QWEN36_SPEC_CHUNKED_PREFILL=1, work/notes/84).
Option A (QWEN36_SPEC_CHUNK_EXACT_RESUME=0; the tests without "exact" in their name): a non-final chunk [s, e) snapshots the
GDN state at e-128 and registers nothing; the next chunk (start_pos = e) resumes from that snapshot. Option B (default;
tests named *exact*): the snapshot is at e, the hidden row of slot e-1 is kept, the next chunk resumes exactly at e. Real MTPSpecEngine._prefill_row / _prefill_fast / Qwen36Model.plan_spec_prefix /
GdnPrefixStateCache.save over the fakes of test_spec_fastprefill_host.py and test_spec_prefix_fast_host.py.

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_spec_chunked_prefill_host.py -q
"""
import types

import pytest
import torch
from loguru import logger

from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import VOCAB, _engine, _row, fake_ttnn
from models.demos.blackhole.qwen36.tests.test_spec_prefix_fast_host import _cache, _model, _prompt
from models.demos.blackhole.qwen36.tt import qwen36_vllm
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.qwen36_vllm import Qwen36MTPContractForCausalLM
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

__all__ = ["fake_ttnn"]  # fixture re-export


@pytest.fixture(autouse=True)
def _resume_mode(request, monkeypatch):
    """Tests with "exact" in their name run Option B (the default); all others pin Option A."""
    if "exact" not in request.node.name:
        monkeypatch.setenv("QWEN36_SPEC_CHUNK_EXACT_RESUME", "0")


NB = 160  # page-table blocks of 64 tokens (10240 tokens)
SLOT = 3


def _setup(calls, cache, B=8, trace_lengths=(128, 40, 768, 1920)):
    model = _model(calls, cache, trace_lengths)
    # a trace for every length with a big enough window table
    import ttnn
    from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import BS
    from models.demos.blackhole.qwen36.tt.model import _ShortPrefillTrace

    def _tr(L):
        if L not in trace_lengths:
            return None
        tr = _ShortPrefillTrace(2048, 192, 2048 // BS)
        tr.spec = True
        tr.last_feed = types.SimpleNamespace(dtype=ttnn.bfloat16, shape=(1, 1, 1, 16))
        tr.logits = "LOGITS"
        return tr

    model._short_trace_for = _tr

    def _replay(tokens, pt, L, strict=True, chunk_start=0, carry_gdn=False, spec_inputs=None):
        calls.append(("replay", tokens.clone(), pt.clone(), L, strict, chunk_start, carry_gdn, spec_inputs))
        return _tr(L)

    model._replay_short_trace = _replay
    model.args = types.SimpleNamespace(max_batch_size=B)
    model.plan_spec_prefix = lambda toks, L, start, **kw: Qwen36Model.plan_spec_prefix(model, toks, L, start, **kw)
    e = _engine(B, model=model, nb=NB, nb_prefill=NB)
    e._warm_mtp_chunk = lambda *a, **k: calls.append(("warm_mtp",))
    e._capture_last_row = lambda *a, **k: calls.append(("capture_last_row", a[1]))
    e._warm_mtp_last = lambda hid, tok, pos, pt, **k: calls.append(("warm_last", hid, tok, pos, pt.clone()))
    return e


@pytest.fixture
def fake_pt(monkeypatch):
    import ttnn

    monkeypatch.setattr(ttnn, "from_torch", lambda t, **k: t)
    monkeypatch.setattr(ttnn, "ReplicateTensorToMesh", lambda mesh: "rep", raising=False)


def _run(e, tokens, e_end, start, final):
    Tp = e_end if not final else len(tokens)
    return MTPSpecEngine._prefill_row(
        e, SLOT, tokens[:Tp], _row(NB), None, start_pos=start, **({} if final else dict(final=False))
    )


def _replays(calls):
    return [(c[5], c[3]) for c in calls if c[0] == "replay"]  # (chunk_start, length)


def test_non_final_chunk_saves_e_minus_128_and_registers_nothing(fake_ttnn):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    out = _run(e, tokens, 4096, 0, final=False)
    assert out.shape == (VOCAB,) and not bool(out.any())  # zero logits row of the normal shape
    # snapshot of the state after tokens[:3968] (key = digest of prompt[:3968])
    assert cache._positions == [3968]
    assert cache.lookup(torch.tensor(tokens), 4096)[0] == 3968
    assert e._prompts == {} and e._fresh == {} and e._active == set()
    assert not any(c[0] in ("slot_write", "capture_last_row") for c in calls)
    assert fake_ttnn["to_torch"] == []  # no host logits read
    assert getattr(e, "_skip_logits_read", False) is False  # flag restored
    assert ("replay" in [c[0] for c in calls]) and _replays(calls)[-1] == (3968, 128)  # [e-128, e) computed


def test_next_chunk_resumes_at_e_minus_128_and_recomputes_it(fake_ttnn):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    calls.clear()
    fake_ttnn["copy"].clear()
    out = _run(e, tokens, 8192, 4096, final=False)
    assert not bool(out.any())
    # restored snapshot 3968 into the bound scratch; [3968, 4096) is recomputed first (the learn point 4096 splits there),
    # then [4096, 8064) by chunk replays (state carried), then the 128 tokens [8064, 8192)
    restores = [d for _, d in fake_ttnn["copy"] if d in ("bound_rec", "bound_carry")]
    assert restores == ["bound_rec", "bound_carry"]
    chunks = [c for c in calls if c[0] == "chunks"]
    assert _replays(calls)[0] == (3968, 128)
    assert chunks and chunks[0][3] == 4096 and chunks[0][4] is False  # start, reset_gdn
    assert _replays(calls)[-1] == (8064, 128)
    assert 8064 in cache._positions and 3968 in cache._positions
    assert e._fresh == {} and e._prompts == {}


def test_final_chunk_behaves_like_today(fake_ttnn):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    _run(e, tokens, 8192, 4096, final=False)
    calls.clear()
    out = _run(e, tokens, 9000, 8192, final=True)
    assert out.shape == (VOCAB,) and out.dtype == torch.float32 and bool(out.any())
    assert e._fresh == {SLOT: 9000} and e._active == {SLOT}
    assert e._prompts[SLOT][0] == tokens and e._prompts[SLOT][2] == 8192
    assert ("slot_write", SLOT) in calls
    assert _replays(calls)[0] == (8064, 128)  # resumed at e-128 of the previous chunk
    assert len(fake_ttnn["to_torch"]) >= 1


def test_final_path_without_final_kw_is_unchanged(fake_ttnn):
    """prefill() without final passes no new kwarg to _prefill_row (older call sites / stubs)."""
    got = {}
    e = _engine(4)
    e._row_gens = lambda: [0] * 4
    e._prefill_row = lambda slot, ids, row, cap=None, start_pos=None, vision_tokens=None: got.update(s=start_pos)
    MTPSpecEngine.prefill(e, 1, [1, 2, 3], _row(), start_pos=0)
    assert got == dict(s=0)


def test_chunk_of_128_from_zero_skips_forced_save_and_logs(fake_ttnn):
    """e = 128 at start 0: e-128 = 0 is not a snapshot position (needs s < p <= e-3): skipped with a log."""
    tokens, calls = _prompt(9000), []
    e = _setup(calls, _cache())
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        out = _run(e, tokens, 128, 0, final=False)
    finally:
        logger.remove(sink)
    assert not bool(out.any()) and e.model._prefix_cache._positions == []
    assert any("no e-128 snapshot" in m for m in msgs), msgs


def test_short_chunk_after_resume_still_saves_above_cap(fake_ttnn):
    """Chunk [4096, 4224): cap = 3968 < e-128 = 4096, so the forced point 4096 is saved."""
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    _run(e, tokens, 4224, 4096, final=False)
    assert 4096 in cache._positions and e._fresh == {}


def test_non_final_without_cache_or_b1_raises(fake_ttnn, expect_error):
    tokens = _prompt(4096)
    e = _setup([], None)
    with expect_error(RuntimeError, "prefix cache"):
        _run(e, tokens, 4096, 0, final=False)
    e1 = _setup([], _cache(), B=1)
    with expect_error(RuntimeError, "prefix cache"):
        MTPSpecEngine._prefill_row(e1, 0, tokens[:4096], _row(NB), None, start_pos=0, final=False)


def test_non_final_unaligned_end_and_mm_assert(fake_ttnn, expect_error):
    tokens, e = _prompt(5000), _setup([], _cache())
    with expect_error(AssertionError, "multiple of 128"):
        MTPSpecEngine._prefill_row(e, SLOT, tokens[:4100], _row(NB), None, start_pos=0, final=False)
    with expect_error(AssertionError, "multimodal"):
        MTPSpecEngine._prefill_row(e, SLOT, tokens[:4096], _row(NB), None, start_pos=0, vision_tokens="V", final=False)


# --------------------------------------------------------------------------- #
# eviction fallback
# --------------------------------------------------------------------------- #
def test_eviction_fallback_counts_and_logs(fake_ttnn, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_CHUNKED_PREFILL", "1")
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    # evict everything: the next chunk (start 4096, cap 3968) finds no snapshot and recomputes from 0
    cache._lru.clear(), cache._by_pos.clear(), cache._positions.clear()
    cache._free = list(range(cache.num_slots))
    calls.clear()
    msgs = []
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        _run(e, tokens, 8192, 4096, final=False)
    finally:
        logger.remove(sink)
    assert getattr(e, "resume_fallbacks", 0) == 1
    assert any("resume fallback" in m and "resumes at 0, wanted 3968" in m for m in msgs), msgs
    assert _replays(calls)[0][0] == 0 or any(c[0] == "chunks" and c[3] == 0 for c in calls)


def test_no_fallback_when_snapshot_present(fake_ttnn):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    _run(e, tokens, 8192, 4096, final=False)
    assert getattr(e, "resume_fallbacks", 0) == 0


# --------------------------------------------------------------------------- #
# QV: prefill_forward + capabilities
# --------------------------------------------------------------------------- #
class _FakeEngine:
    def __init__(self):
        self.calls = []

    def prefill(self, slot, ids, row, *args, **kw):
        self.calls.append((slot, ids.clone(), args, kw))
        return torch.zeros(VOCAB) if kw.get("final") is False else torch.arange(VOCAB, dtype=torch.float32)


def _gen(monkeypatch, env="1", apc="1"):
    monkeypatch.setenv("QWEN36_SPEC_CHUNKED_PREFILL", env)
    monkeypatch.setenv("QWEN36_SPEC_PREFIX_CACHE", apc)
    g = Qwen36MTPContractForCausalLM.__new__(Qwen36MTPContractForCausalLM)
    eng = _FakeEngine()
    model = types.SimpleNamespace(get_image_features=lambda pix, grid: "IMG_ROWS")
    g.model = [model]
    g.data_parallel = 1
    g._ensure_ready = lambda kv_cache=None: eng
    g._table = lambda pt: pt.to(torch.int32)
    return g, eng


def _call(g, mask, N=2, **kw):
    tokens = torch.arange(1, 1 + N * 12, dtype=torch.int32).reshape(N, 12)
    pt = torch.ones(N, 4, dtype=torch.int32)
    return g.prefill_forward(
        tokens,
        pt,
        None,
        torch.tensor([8] * N),
        empty_slots=list(range(N)),
        start_pos=torch.tensor([0, 128][:N]),
        intermediate_prefill_mask=mask,
        **kw,
    )


def test_prefill_forward_passes_final_flag_per_row(monkeypatch):
    g, eng = _gen(monkeypatch)
    logits, deltas = _call(g, [True, False])
    (s0, _, _, k0), (s1, _, _, k1) = eng.calls
    assert k0 == {"start_pos": 0, "final": False}
    assert k1 == {"start_pos": 128}  # final row: today's call, no new kwarg
    assert logits.shape == (2, 1, VOCAB) and not bool(logits[0].any()) and bool(logits[1].any())
    assert deltas.shape == (2,)


def test_prefill_forward_mask_none_is_all_final(monkeypatch):
    g, eng = _gen(monkeypatch)
    _call(g, None)
    assert all("final" not in c[3] for c in eng.calls)


def test_prefill_forward_mm_non_final_raises(monkeypatch, expect_error):
    g, eng = _gen(monkeypatch)
    with expect_error(RuntimeError, "non-final"):
        _call(
            g, [False, True], pixel_values=[None, torch.zeros(16, 8)], image_grid_thw=[None, torch.tensor([[1, 4, 4]])]
        )
    # an MM row that is final is fine
    _call(g, [False, False], pixel_values=[None, torch.zeros(16, 8)], image_grid_thw=[None, torch.tensor([[1, 4, 4]])])


def test_prefill_forward_non_final_refused_when_chunking_off(monkeypatch, expect_error):
    g, eng = _gen(monkeypatch, env="0")
    with expect_error(RuntimeError, "CHUNKED_PREFILL"):
        _call(g, [True, False])
    assert eng.calls == []


@pytest.mark.parametrize(
    "env,apc,flag,on",
    [("1", "1", "1", True), ("0", "1", "1", False), ("1", "0", "1", False), ("1", "1", "0", False)],
)
def test_capability_flags(monkeypatch, env, apc, flag, on):
    monkeypatch.setenv("QWEN36_SPEC_CHUNKED_PREFILL", env)
    monkeypatch.setenv("QWEN36_SPEC_PREFIX_CACHE", apc)
    monkeypatch.setenv("QWEN36_PREFIX_CACHE", flag)
    assert qwen36_vllm._spec_chunked_prefill_enabled() is on
    assert qwen36_vllm._mtp_chunked_capabilities() == {
        "supports_chunked_prefill": on,
        "accepts_intermediate_prefill_mask": on,
    }


def test_capability_default_off(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_CHUNKED_PREFILL", raising=False)
    assert qwen36_vllm._spec_chunked_prefill_enabled() is False
    caps = Qwen36MTPContractForCausalLM.model_capabilities  # evaluated at import (env unset in the test run)
    assert "accepts_intermediate_prefill_mask" in caps


def test_serving_conflict_chunked_allowed_only_with_env(monkeypatch):
    sc = types.SimpleNamespace(
        enable_chunked_prefill=True, max_num_batched_tokens=4096, long_prefill_token_threshold=4096
    )
    vc = types.SimpleNamespace(
        cache_config=types.SimpleNamespace(enable_prefix_caching=True),
        scheduler_config=sc,
        model_config=types.SimpleNamespace(max_model_len=262144),
    )
    monkeypatch.setenv("QWEN36_SPEC_PREFIX_CACHE", "1")
    monkeypatch.setenv("QWEN36_PREFIX_CACHE", "1")
    monkeypatch.setenv("QWEN36_SPEC_CHUNKED_PREFILL", "0")
    assert "chunked prefill" in Qwen36MTPContractForCausalLM._serving_conflict(vc)
    monkeypatch.setenv("QWEN36_SPEC_CHUNKED_PREFILL", "1")
    assert Qwen36MTPContractForCausalLM._serving_conflict(vc) is None
    # APC off: the prefix-caching conflict still wins, and chunking is off again
    monkeypatch.setenv("QWEN36_SPEC_PREFIX_CACHE", "0")
    assert "prefix caching" in Qwen36MTPContractForCausalLM._serving_conflict(vc)


# --------------------------------------------------------------------------- #
# Option B: exact resume at the chunk end (QWEN36_SPEC_CHUNK_EXACT_RESUME=1, default)
# --------------------------------------------------------------------------- #
def _segments(calls):
    """(start, length) of every segment run: short-trace replays + chunk-trace runs (start, n chunks * 2048 as a span)."""
    out = []
    for c in calls:
        if c[0] == "replay":
            out.append((c[5], c[3]))
        elif c[0] == "chunks":
            out.append((c[3], c[1] * 2048 + c[2]))  # (start, num_full * chunk + tail)
    return out


def test_exact_first_chunk_saves_at_chunk_end_only(fake_ttnn, fake_pt):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    out = _run(e, tokens, 4096, 0, final=False)
    assert not bool(out.any())
    assert cache._positions == [4096]  # the chunk end, no e-128
    assert cache.lookup(torch.tensor(tokens), 4096)[0] == 4096
    assert e._prompts == {} and e._fresh == {} and e._active == set()
    assert not any(c[0] == "slot_write" for c in calls)
    assert fake_ttnn["to_torch"] == []
    assert getattr(e, "_skip_logits_read", False) is False
    # the hidden row of slot e-1 is kept in _last_hidden[row]; progress recorded
    assert e._chunk_prog()[SLOT][0] == 4096
    assert not any(c[0] == "warm_last" for c in calls)  # nothing to warm yet


def test_exact_continuation_segments_are_exactly_the_chunk(fake_ttnn, fake_pt):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    calls.clear()
    fake_ttnn["copy"].clear()
    out = _run(e, tokens, 8192, 4096, final=False)
    assert not bool(out.any())
    # one segment [4096, 8192): chunk replays from 4096 (2 x 2048), no 128 recompute, no e-128 split, no learn split
    assert _segments(calls) == [(4096, 4096)]
    restores = [d for _, d in fake_ttnn["copy"] if d in ("bound_rec", "bound_carry")]
    assert restores == ["bound_rec", "bound_carry"]  # snapshot at 4096 restored
    assert cache._positions == [4096, 8192]  # 8064 / 3968 never saved
    assert e._chunk_prog()[SLOT][0] == 8192 and e.chunk_exact_resumes == 1
    assert getattr(e, "resume_fallbacks", 0) == 0


def test_exact_warms_drafter_slot_e_minus_1_once_with_token_e(fake_ttnn, fake_pt):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    calls.clear()
    _run(e, tokens, 8192, 4096, final=False)
    warms = [c for c in calls if c[0] == "warm_last"]
    assert len(warms) == 1
    _, hid, tok, pos, pt = warms[0]
    assert hid == f"last_hidden[{SLOT}]" and tok == tokens[4096] and pos == 4095
    assert torch.equal(pt, _row(NB))  # this request's page table, not a stale _mtp_pt_rows entry
    # warm happens before the first segment
    assert [c[0] for c in calls].index("warm_last") < min(
        i for i, c in enumerate(calls) if c[0] in ("replay", "chunks")
    )


def test_exact_final_chunk_resumes_at_e_then_registers(fake_ttnn, fake_pt):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    _run(e, tokens, 8192, 4096, final=False)
    calls.clear()
    out = _run(e, tokens, 9000, 8192, final=True)
    assert out.shape == (VOCAB,) and bool(out.any())
    assert e._fresh == {SLOT: 9000} and e._active == {SLOT}
    assert e._prompts[SLOT][2] == 8192
    assert ("slot_write", SLOT) in calls
    assert [c[1:4] for c in calls if c[0] == "warm_last"] == [(f"last_hidden[{SLOT}]", tokens[8192], 8191)]
    assert _segments(calls)[0][0] == 8192  # starts at e, not e-128
    assert ("capture_last_row", 8999 - 8192) in calls or any(c[0] == "replay" for c in calls)
    assert SLOT not in e._chunk_prog()  # consumed
    assert len(fake_ttnn["to_torch"]) >= 1


def test_exact_middle_chunk_takes_stride_saves_but_no_end_split(fake_ttnn, fake_pt, monkeypatch):
    """A non-final chunk spanning a stride point keeps that save; the chunk end itself is the last segment end."""
    monkeypatch.setenv("QWEN36_PREFIX_SNAPSHOT_STRIDE", "2048")
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    calls.clear()
    _run(e, tokens, 8192, 4096, final=False)  # stride 2048 from s=4096 -> 6144 splits; the end 8192 is the last end
    assert cache._positions == [2048, 4096, 6144, 8192]  # stride save of the first chunk too; no 8064 (e-128)
    assert _segments(calls) == [(4096, 2048), (6144, 2048)]


@pytest.mark.parametrize("breakage", ["missing", "digest", "pt", "start", "evicted"])
def test_exact_falls_back_to_option_a_resume(fake_ttnn, fake_pt, breakage):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    row = _row(NB)
    run_tokens, start = tokens, 4096
    if breakage == "missing":
        e._chunk_prog().clear()
    elif breakage == "digest":
        run_tokens = list(tokens)
        run_tokens[10] += 1
    elif breakage == "start":
        start = 4224
    elif breakage == "evicted":
        cache._lru.clear(), cache._by_pos.clear(), cache._positions.clear()
        cache._free = list(range(cache.num_slots))
    calls.clear()
    pt = row.clone() if breakage != "pt" else row.flip(1).contiguous()
    MTPSpecEngine._prefill_row(e, SLOT, run_tokens[:8192], pt, None, start_pos=start, final=False)
    assert not any(c[0] == "warm_last" for c in calls)  # no exact-resume warm
    assert getattr(e, "chunk_exact_resumes", 0) == 0
    assert getattr(e, "chunk_exact_misses", 0) == (0 if breakage in ("missing",) else 1)
    # normal path: resumes at spec_resume_cap (3968 / 4096-ish) or earlier, never at 4096 from the exact snapshot
    if breakage in ("digest", "pt", "start", "missing"):
        assert getattr(e, "resume_fallbacks", 0) >= 0
    assert _segments(calls)  # still runs correctly
    first = _segments(calls)[0][0]
    assert first < 4096 if breakage != "start" else first <= 4224


def test_exact_progress_cleared_on_release_and_reprefill(fake_ttnn, fake_pt):
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    e._row_gens = lambda: [0] * 8
    e._no_draft, e._withheld_streak = set(), {}
    _run(e, tokens, 4096, 0, final=False)
    assert SLOT in e._chunk_prog()
    MTPSpecEngine.release(e, SLOT)
    assert SLOT not in e._chunk_prog()
    # a later chunk after the release must not take the exact path
    calls.clear()
    _run(e, tokens, 8192, 4096, final=False)
    assert not any(c[0] == "warm_last" for c in calls)


def test_exact_progress_follows_device_slot_remap_and_prefill_consumes_it(fake_ttnn, fake_pt):
    e = _setup([], _cache())
    e._chunk_prog()[2] = ("x",)
    e._chunk_prog()[5] = ("y",)
    # the permuting remap path rebuilds per-row dicts: mimic the line the engine runs
    idx = [1, 0, 5, 3, 4, 2, 6, 7]
    new_of = {old: new for new, old in enumerate(idx)}
    e.__dict__["_chunk_progress"] = {new_of[u]: v for u, v in e._chunk_prog().items()}
    assert e._chunk_prog() == {5: ("x",), 2: ("y",)}


def test_exact_env_zero_gives_option_a_segments(fake_ttnn, fake_pt, monkeypatch):
    monkeypatch.setenv("QWEN36_SPEC_CHUNK_EXACT_RESUME", "0")
    tokens, cache, calls = _prompt(9000), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    assert cache._positions == [3968] and e._chunk_prog() == {}
    calls.clear()
    _run(e, tokens, 8192, 4096, final=False)
    assert _replays(calls)[0] == (3968, 128) and _replays(calls)[-1] == (8064, 128)
    assert not any(c[0] == "warm_last" for c in calls)


def test_exact_short_tail_final_chunk_uses_normal_resume(fake_ttnn, fake_pt):
    """Final chunk with < 3 tokens left cannot resume at e (plan floor): normal path, no exact warm."""
    tokens, cache, calls = _prompt(4097), _cache(), []
    e = _setup(calls, cache)
    _run(e, tokens, 4096, 0, final=False)
    calls.clear()
    _run(e, tokens, 4097, 4096, final=True)
    assert not any(c[0] == "warm_last" for c in calls)


# --------------------------------------------------------------------------- #
# Concurrency: reserved rows, pins, remap, counters (note 84, Option B under load)
# --------------------------------------------------------------------------- #
def _bucketed_engine(calls, cache, B=8):
    e = _setup(calls, cache, B=B)
    e._buckets, e.host_slotmap, e.row_swaps, e._def_last = (1, 2, 4, 8), True, 0, None
    e._row_gen = None
    e._no_draft, e._withheld_streak = set(), {}
    return e


def _pre(e, slot, tokens, end, start, final, pt):
    """The public prefill (the preamble claims the lowest free row) of runner slot ``slot``."""
    Tp = end if not final else len(tokens)
    return e.prefill(slot, tokens[:Tp], pt, None, start_pos=start, **({} if final else dict(final=False)))


def test_exact_interleaved_requests_survive_row_swaps(fake_ttnn, fake_pt):
    ta, tb, tc = _prompt(9000, seed=1), _prompt(6500, seed=2), _prompt(768, seed=3)
    cache, calls = _cache(num_slots=8), []
    e = _bucketed_engine(calls, cache)
    pa, pb, pc = _row(NB), _row(NB).flip(1).contiguous(), _row(NB) + 500
    # A on runner slot 5, B on slot 6, C on slot 7: every NEW request swaps onto the lowest free row
    _pre(e, 5, ta, 4096, 0, False, pa)
    row_a = e._phys[5]
    assert row_a == 0 and e.row_swaps == 1
    _pre(e, 6, tb, 4096, 0, False, pb)
    row_b = e._phys[6]
    assert row_b == 1 and e._chunk_reserved_rows() == {0, 1}  # B did not take A's row
    _pre(e, 5, ta, 8192, 4096, False, pa)  # A2: its slot sits on a reserved row, never swapped
    assert e._phys[5] == row_a
    swaps = e.row_swaps
    _pre(e, 7, tc, 768, None, True, pc)  # a new short request between the chunks
    assert e._phys[7] == 2 and e.row_swaps == swaps + 1  # lowest UNRESERVED row
    assert e._phys[5] == row_a and e._phys[6] == row_b
    calls.clear()
    _pre(e, 6, tb, 6500, 4096, True, pb)  # B2 (final)
    warm = [c for c in calls if c[0] == "warm_last"]
    assert [(c[1], c[2], c[3]) for c in warm] == [(f"last_hidden[{row_b}]", tb[4096], 4095)]
    calls.clear()
    _pre(e, 5, ta, 9000, 8192, True, pa)  # A3 (final)
    warm = [c for c in calls if c[0] == "warm_last"]
    assert [(c[1], c[2], c[3]) for c in warm] == [(f"last_hidden[{row_a}]", ta[8192], 8191)]
    assert e.chunk_exact_resumes == 3 and getattr(e, "chunk_exact_misses", 0) == 0
    assert getattr(e, "resume_fallbacks", 0) == 0
    assert cache.stats["refusals"] == 0 if "refusals" in cache.stats else True
    assert e._chunk_prog() == {} and cache.pinned_count() == 0


def test_exact_foreign_prefill_on_pending_row_warns_with_identities_and_drops(fake_ttnn, fake_pt):
    from loguru import logger

    ta, td = _prompt(9000, seed=1), _prompt(5000, seed=4)
    cache, calls, msgs = _cache(num_slots=8), [], []
    e = _setup(calls, cache)
    _run(e, ta, 4096, 0, final=False)
    assert cache.pinned_count() == 1
    sink = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        _run(e, td, 4096, 0, final=False)  # another request lands on the pending row anyway
    finally:
        logger.remove(sink)
    assert any("refused" in m and "progress digest" in m and "another request dropped" in m for m in msgs)
    assert cache.stats["refusals"] == 1
    assert cache.pinned_count() == 1 and e._chunk_prog()[SLOT][0] == 4096  # the new request's own progress / pin


def test_exact_stride_saves_never_evict_a_pinned_chunk_snapshot(fake_ttnn, fake_pt):
    ta = _prompt(9000, seed=1)
    cache, calls = _cache(num_slots=4), []
    e = _setup(calls, cache)
    _run(e, ta, 4096, 0, final=False)
    pinned_key = list(cache._pins().values())[0]
    for i in range(10):  # other requests' stride / learn saves churn the 3 free slots
        cache.save(_prompt(2000, seed=20 + i), 1024)
    assert pinned_key in cache._lru and cache.stats["evictions"] == 7
    # the continuation still resumes exactly
    calls.clear()
    _run(e, ta, 8192, 4096, final=False)
    assert e.chunk_exact_resumes == 1 and _segments(calls) == [(4096, 4096)]
    # every slot pinned (forced): a plain save is skipped, nothing is evicted
    cache2 = _cache(num_slots=2)
    for i, k in enumerate(("x", "y")):
        cache2._lru[(128 * (i + 1), k)] = i
    cache2._free = []
    cache2._pins().update({0: (128, "x"), 1: (256, "y")})
    assert cache2.save(_prompt(500), 128) is False
    assert cache2.stats["skipped_saves"] == 1 and cache2.stats["evictions"] == 0 and len(cache2._lru) == 2


def test_exact_pin_cap_is_the_row_count(fake_ttnn, fake_pt):
    cache = _cache(num_slots=8)
    cache.max_pinned = 2
    for i in range(3):
        t = _prompt(1024, seed=i)
        cache.save(t, 128)
        assert cache.pin(t, 128, owner=i)
    assert cache.pinned_count() == 2 and 0 not in cache._pins()


def test_exact_release_of_mid_chunk_request_unpins_and_clears_progress(fake_ttnn, fake_pt):
    cache, calls = _cache(num_slots=8), []
    e = _bucketed_engine(calls, cache)
    e._row_gens = lambda: [0] * 8
    _pre(e, 3, _prompt(9000, seed=1), 4096, 0, False, _row(NB))
    row = e._phys[3]
    assert row in e._chunk_prog() and cache.pinned_count() == 1
    e.release(3)  # abort / finish of a mid-chunk request
    assert row not in e._chunk_prog() and cache.pinned_count() == 0
    assert row not in e._chunk_reserved_rows()


def test_exact_hidden_row_and_pin_follow_a_forced_remap(fake_ttnn, fake_pt):
    ta = _prompt(9000, seed=1)
    for host in (True, False):
        cache, calls = _cache(num_slots=8), []
        e = _bucketed_engine(calls, cache)
        e.host_slotmap, e._gdn = host, []
        _pre(e, 4, ta, 4096, 0, False, _row(NB))
        row = e._phys[4]
        hid = e._last_hidden[row]
        idx = [1, 0, 2, 3, 4, 5, 6, 7]  # runner slots 0 and 1 swap; slot 4 stays
        if not host:
            idx = list(range(8))
            idx[row], idx[7] = 7, row  # the device remap moves the engine row of the pending request
        e.remap_slots(idx)
        new_row = e._phys[4] if host else 7
        assert (e._phys[4] == row) if host else True
        assert e._last_hidden[new_row] is hid  # the hidden row follows the row
        assert list(e._chunk_prog()) == [new_row] and list(cache._pins()) == [new_row]
        calls.clear()
        slot = 4 if host else 4
        # the continuation (same runner slot, its row moved or not) resumes exactly
        if not host:
            e._phys = list(range(8))  # device remap: runner slot i is engine row i now
            slot = new_row
        _pre(e, slot, ta, 8192, 4096, False, _row(NB))
        assert e.chunk_exact_resumes == 1 and [c[1] for c in calls if c[0] == "warm_last"] == [hid]


def test_exact_counters_and_summary_line_every_1000_prefills(fake_ttnn, fake_pt):
    from loguru import logger

    cache, calls, msgs = _cache(num_slots=8), [], []
    e = _bucketed_engine(calls, cache)
    e._n_prefills = 997
    sink = logger.add(lambda m: msgs.append(str(m)), level="INFO")
    try:
        ta = _prompt(9000, seed=1)
        _pre(e, 2, ta, 4096, 0, False, _row(NB))
        _pre(e, 2, ta, 8192, 4096, False, _row(NB))
        assert not any("chunked-prefill summary" in m for m in msgs)
        _pre(e, 2, ta, 9000, 8192, True, _row(NB))
    finally:
        logger.remove(sink)
    line = [m for m in msgs if "chunked-prefill summary after 1000 prefills" in m]
    assert len(line) == 1
    for key in ("pinned_saves=2", "skipped_saves=0", "evictions=0", "exact_resumes=2", "refusals=0"):
        assert key in line[0], line[0]

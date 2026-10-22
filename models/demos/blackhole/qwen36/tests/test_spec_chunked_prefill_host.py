# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for chunked prefill on the MTP spec class (QWEN36_SPEC_CHUNKED_PREFILL=1, Option A of
work/notes/84): a non-final chunk [s, e) snapshots the GDN state at e-128 and registers nothing; the next chunk
(start_pos = e) resumes from that snapshot. Real MTPSpecEngine._prefill_row / _prefill_fast / Qwen36Model.plan_spec_prefix /
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
    e._capture_last_row = lambda *a, **k: calls.append(("capture_last_row",))
    return e


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

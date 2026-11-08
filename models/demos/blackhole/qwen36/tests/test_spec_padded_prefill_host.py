# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests (no device) for MTPSpecEngine._reconcile_fresh_rows: a fresh row whose first verify starts BELOW its
prefilled length was prefilled past its prompt and is prefilled again at the true length.

The sequence vLLM 0.30 + vllm-tt-plugin produce for it (server crash of 2026-10-19, note 56): a new request with a
1409-token prompt hits 22 full prefix-cache blocks (1408 tokens), so one token is left; the scheduler pads such a
request to 1 + K scheduled tokens with K placeholder drafts ([-1] * K, scheduler.py "Pad new decode requests to uniform
spec decoding size") and the runner hands the model prompt_lens = 1408 + 1 + K = 1412; after the step vLLM rolls the K
rejected placeholders back to num_computed_tokens = 1409, so the first verify of the row comes with start_pos 1409
while the engine prefilled 1412 tokens (`slot 1: start_pos 1409 != prompt length 1412`).

Run (worktree first on PYTHONPATH, ttnn from the built checkout):
  PYTHONPATH=<worktree>:$TT_METAL_HOME/ttnn:$TT_METAL_HOME/tools $TT_METAL_HOME/python_env/bin/python -m pytest \
      models/demos/blackhole/qwen36/tests/test_spec_padded_prefill_host.py -q
"""

import types

import torch

import ttnn
from models.demos.blackhole.qwen36.tests.test_spec_fastjoin_host import _engine, _FakeModel, _GdnStub
from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import SCRATCH, VOCAB
from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import _row as _pf_row
from models.demos.blackhole.qwen36.tests.test_spec_fastprefill_host import fake_ttnn
from models.demos.blackhole.qwen36.tests.test_spec_prefix_fast_host import _cache, _model, _prompt
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

__all__ = ["fake_ttnn"]  # fixture re-export

L = 1409  # prompt length: 22 full 64-token blocks + 1 token (the one-uncached-token case)
HIT = L - 1  # vLLM num_computed_tokens at the prefill (all full blocks cached)
K = 3  # QWEN36_SPEC_K of the crashed server
NB = 32  # page-table width in blocks of 64: 2048 tokens, room for the 1409-token prompt + verify window


def _mk(B, fast_join=True):
    """The fast-join test engine with a page table wide enough for the 1409-token prompt."""
    T = K + 1
    model, gdn = _FakeModel(B, T), _GdnStub(T)
    e = _engine(B, K, model, gdn, fast_join=fast_join)
    e.nb, e._last_cap_blocks = NB, NB
    return e, model, gdn


def _inputs(e, start):
    """tokens [B, T] (row u: 100 + u, 200 + u, ...), start_pos [B] from ``start``, a page table with distinct blocks."""
    B, T = e.B, e.T
    tok = torch.arange(B * T, dtype=torch.int64).reshape(B, T) + 100
    pos = torch.tensor(start, dtype=torch.int64)
    pt = torch.arange(B * NB, dtype=torch.int32).reshape(B, NB) + 1
    return tok, pos, pt


def _stub_prefill_row(e):
    """_prefill_row stand-in with the real bookkeeping side effects; records (engine row, prompt length, start_pos)."""
    calls = []

    def _prefill_row(slot, prompt_ids, page_table_row, _cap_blocks=None, start_pos=None):
        prompt = [int(t) for t in torch.as_tensor(prompt_ids).reshape(-1)]
        calls.append((slot, len(prompt), None if start_pos is None else int(start_pos)))
        e._fresh_dirty.discard(slot)
        e._prompts[slot] = (prompt, torch.as_tensor(page_table_row).clone()) + (
            () if start_pos is None else (int(start_pos),)
        )
        e._fresh[slot] = len(prompt)
        e._active.add(slot)
        e._last_hidden[slot] = f"h{slot}@{len(prompt)}"

    e._prefill_row = _prefill_row
    return calls


def _row():
    return torch.arange(1, NB + 1, dtype=torch.int32)


def _padded_prefill(e, slot, prompt_len=L, start=HIT, k=K):
    """What the runner's padded prefill leaves behind: the row prefilled over prompt_len + k tokens at vLLM start."""
    prompt = [1000 + i for i in range(prompt_len)] + [0] * k  # the token buffer past the prompt (zeros)
    e._prefill_row(slot, prompt, _row(), start_pos=start)
    assert e._fresh[slot] == prompt_len + k


def test_padded_prefill_then_first_verify_re_prefills_at_the_prompt_length():
    """Fast join: the crashed sequence. The row is re-prefilled with the first start_pos tokens of the stored prompt at
    the same vLLM start_pos, the MTP slot Tp-1 warm and the GDN seed use the corrected length, one replay runs."""
    B = 4
    e, model, gdn = _mk(B)
    calls = _stub_prefill_row(e)
    e._active = {0, 2}
    _padded_prefill(e, 1)  # Tp = 1412, vLLM start_pos 1408
    assert e._prompts[1][2] == HIT and len(e._prompts[1][0]) == L + K
    tok, pos, pt = _inputs(e, [20, L, 30, 0])
    ids, lt = e.verify(tok, pos, torch.tensor([2, 1, 3, 1]), pt, active=[True, True, True, False])
    assert calls == [(1, L + K, HIT), (1, L, HIT)], calls  # the padded prefill, then the re-prefill at 1409
    assert e._prompts.get(1) is None
    assert e.log["warm"] == [("h1@1409", int(tok[1, 0]), L - 1, "pt1")]  # slot Tp-1 of the TRUE prompt
    assert gdn.selects == [([1], "SEL")]
    assert len(model.calls) == 1 and model.calls[0]["positions"][1] == L and model.calls[0]["mi_prev"][1] == 0
    assert e._fresh == {} and e._fresh_dirty == set() and e._active == {0, 1, 2}
    assert torch.equal(ids, (tok * 10 + 1).to(torch.int32)) and lt is None


def test_re_prefill_uses_the_stored_prompt_cut_at_start_pos():
    """The re-prefill gets prompt[:start_pos] of the stored (over-long) prompt and its page-table row."""
    B = 2
    e, _, _ = _mk(B)
    seen = []
    real_stub = _stub_prefill_row(e)

    def _rec(slot, prompt_ids, page_table_row, _cap_blocks=None, start_pos=None):
        seen.append((list(prompt_ids), page_table_row.clone(), start_pos))
        return stub(slot, prompt_ids, page_table_row, _cap_blocks, start_pos)

    stub, e._prefill_row = e._prefill_row, _rec
    _padded_prefill(e, 0)
    tok, pos, pt = _inputs(e, [L, 0])
    e.verify(tok, pos, torch.tensor([1, 1]), pt, active=[True, False])
    assert real_stub == [(0, L + K, HIT), (0, L, HIT)]
    assert seen[1][0] == [1000 + i for i in range(L)] and torch.equal(seen[1][1], _row()) and seen[1][2] == HIT


def test_padded_prefill_eager_seed_path_re_prefills_too():
    """QWEN36_SPEC_FAST_JOIN=0: the eager seed asserts the same invariant; the reconcile runs before it."""
    B = 2
    e, model, gdn = _mk(B, fast_join=False)
    calls = _stub_prefill_row(e)
    hits = []
    e._seed = lambda *a: (hits.append("seed"), (torch.zeros(B, e.T, dtype=torch.int32), None))[1]
    e._join = lambda *a: (hits.append("join"), (torch.zeros(B, e.T, dtype=torch.int32), None))[1]
    e._active = {0}
    _padded_prefill(e, 1)
    tok, pos, pt = _inputs(e, [7, L])
    e.verify(tok, pos, torch.tensor([1, 1]), pt)
    assert calls == [(1, L + K, HIT), (1, L, HIT)]
    assert hits == ["join"] and len(e.log["tables"]) == 1
    assert e.log["warm"] == [("h1@1409", int(tok[1, 0]), L - 1, "pt1")]
    assert e._fresh == {} and 1 not in e._prompts and gdn.selects == [] and model.calls == []


def test_two_padded_rows_in_one_step_and_a_plain_fresh_row():
    """Two padded rows join with an ordinary fresh row: only the padded rows are re-prefilled, in engine-row order."""
    B = 4
    e, model, gdn = _mk(B)
    calls = _stub_prefill_row(e)
    _padded_prefill(e, 3, prompt_len=L, start=HIT)
    e._prefill_row(0, [5] * 70, torch.zeros(NB, dtype=torch.int32), start_pos=None)  # plain fresh row, Tp 70
    _padded_prefill(e, 2, prompt_len=2 * 64 + 1, start=2 * 64)  # Tp = 132, verify at 129
    calls.clear()
    tok, pos, pt = _inputs(e, [70, 0, 129, L])
    e.verify(tok, pos, torch.ones(B, dtype=torch.int64), pt, active=[True, False, True, True])
    assert calls == [(2, 129, 128), (3, L, HIT)]
    assert [w[2] for w in e.log["warm"]] == [69, 128, L - 1]
    assert gdn.selects == [([0, 1, 2, 3], "SEL")]  # every active row fresh -> whole-slot write
    assert e._fresh == {} and e._prompts == {} and len(model.calls) == 1


def test_fresh_row_at_its_prompt_length_is_not_re_prefilled():
    """The ordinary case (start_pos == prefilled length) runs as before: no re-prefill, no extra bookkeeping."""
    B = 2
    e, _, _ = _mk(B)
    calls = _stub_prefill_row(e)
    e._prefill_row(1, [3] * 50, torch.zeros(NB, dtype=torch.int32), start_pos=0)
    e._active.add(0)
    calls.clear()
    tok, pos, pt = _inputs(e, [9, 50])
    e.verify(tok, pos, torch.tensor([1, 1]), pt)
    assert calls == [] and e.log["warm"] == [("h1@50", int(tok[1, 0]), 49, "pt1")] and e._fresh == {}


def test_verify_above_the_prefilled_length_raises(expect_error):
    """A prompt longer than what was prefilled (a partial prompt) cannot be repaired: fail loudly, no re-prefill."""
    B = 2
    e, model, _ = _mk(B)
    calls = _stub_prefill_row(e)
    e._prefill_row(1, [3] * 50, torch.zeros(NB, dtype=torch.int32), start_pos=0)
    calls.clear()
    tok, pos, pt = _inputs(e, [0, 53])
    with expect_error(ValueError, r"never saw tokens \[50, 53\)"):
        e.verify(tok, pos, torch.tensor([1, 1]), pt, active=[False, True])
    assert calls == [] and e.log["warm"] == [] and model.calls == [] and e._fresh == {1: 50}


def test_padded_prefill_without_a_stored_prompt_raises(expect_error):
    """The re-prefill needs the stored prompt: a missing entry is a bookkeeping bug, not something to paper over."""
    B = 2
    e, _, _ = _mk(B)
    _stub_prefill_row(e)
    _padded_prefill(e, 1)
    del e._prompts[1]
    tok, pos, pt = _inputs(e, [0, L])
    with expect_error(RuntimeError, "no stored prompt"):
        e.verify(tok, pos, torch.tensor([1, 1]), pt, active=[False, True])


def test_reconcile_rejects_a_vllm_start_pos_at_or_past_the_prompt(expect_error):
    """The vLLM start_pos of the padded prefill must lie below the prompt length (it is prompt_len - 1 for the pad)."""
    B = 1
    e, _, _ = _mk(B)
    _stub_prefill_row(e)
    e._prefill_row(0, [1] * 10, torch.zeros(NB, dtype=torch.int32), start_pos=8)
    with expect_error(ValueError, "start_pos 8 of the prefill is not below the prompt length 8"):
        MTPSpecEngine._reconcile_fresh_rows(e, {0: 10}, torch.tensor([8]))


def test_end_to_end_re_prefill_resumes_from_the_snapshot_the_padded_prefill_saved(fake_ttnn):
    """The real _prefill_row / _prefill_fast with the fast-prefill fakes and a real GdnPrefixStateCache (device copies
    faked). The padded prefill (1412 tokens, vLLM start 1408, no snapshot yet) runs segments [0,1280) [1280,1408)
    [1408,1412) and saves the scratch at 1280 and 1408 -- both keyed by real prompt tokens. The first verify at 1409
    re-prefills 1409 tokens: cap(1408, 1409) = 1280 hits the 1280 snapshot, so only [1280, 1409) is recomputed."""
    B = 2
    e, fake, gdn = _mk(B)
    calls = []
    cache = _cache(num_slots=8)
    m = _model(calls, cache, trace_lengths=(1280, 128, 4, 129))
    m.verify_traced = fake.verify_traced
    m.args = types.SimpleNamespace(max_batch_size=B)
    m.plan_spec_prefix = lambda toks, Tp, sp: Qwen36Model.plan_spec_prefix(m, toks, Tp, sp)
    e.model = m
    e.nb, e.nb_prefill, e.vocab, e.block_size = 64, 64, VOCAB, 64
    e._scratch_block, e._feed_dtype, e._dim_frac, e.mesh = SCRATCH, ttnn.bfloat16, 16, None
    e.fast_prefill = e.prefill_recorded = e.recorded_prefill = True
    e._warm_mtp_chunk = lambda *a, **k: None
    tokens = _prompt(L) + [0] * K  # the runner's tokens[:1412]: the prompt, then the token buffer past it
    out = e._prefill_row(0, tokens, _pf_row(), start_pos=HIT)
    assert out.shape == (VOCAB,) and e._fresh == {0: L + K} and e._prompts[0][2] == HIT
    replays = [(c[3], c[5], c[6]) for c in calls if c[0] == "replay"]  # (segment length, chunk_start, carry)
    assert replays == [(1280, 0, False), (128, 1280, True), (4, 1408, True)], replays
    assert sorted(p for p, _ in cache._lru) == [1280, 1408]
    calls.clear()
    tok, pos, pt = _inputs(e, [L, 0])
    ids, _ = e.verify(tok, pos, torch.tensor([1, 1]), pt, active=[True, False])
    replays = [(c[3], c[5], c[6]) for c in calls if c[0] == "replay"]
    assert replays == [(129, 1280, True)], replays  # resumed at the 1280 snapshot, 129 tokens recomputed
    pos1280, slot1280 = cache.lookup(torch.tensor(tokens[:L]), 1280)  # the snapshot keyed by the REAL tokens[:1280]
    assert pos1280 == 1280 and slot1280 is not None
    restores = [src for src, dst in fake_ttnn["copy"] if dst == "bound_rec"]
    assert restores[-1] == ("rec", slot1280), restores  # the re-prefill restored that snapshot into the scratch
    assert e._fresh == {} and e._prompts == {} and e.log["warm"] == [("h0", int(tok[0, 0]), L - 1, "pt0")]
    assert gdn.selects == [([0, 1], "SEL")] and fake.calls[0]["positions"][0] == L and fake.calls[0]["mi_prev"][0] == 0
    assert torch.equal(ids[0], (tok[0] * 10 + 1).to(torch.int32))

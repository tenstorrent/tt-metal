# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the Gemma4 MTP decode block loop.

One MTP iteration commits an accepted prefix + bonus, which is VARIABLE (1..N)
because acceptance is content-dependent. The step's row must still be exactly N
wide, so iterations run back-to-back until the row is full.

Why this matters: a short row has to be padded and every pad value is wrong.
``-1`` puts an invalid token id on the wire; EOS terminates the request at the
first pad, because upstream trims committed tokens at the first stop token. That
regression capped solo generation at about one block (osl=128 returned 6 tokens)
while the batched baseline path, which never pads, ran to full length -- so a
throughput sweep did not catch it.

Host-only: the loop is driven through a stub session, no device.
"""

import pytest
import torch

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt.generator_vllm import Gemma4MTPForCausalLM as MTP


class _StubSession:
    """Returns a scripted (committed, accepted_m) per ``serving_step`` call."""

    def __init__(self, scripted):
        self.scripted = list(scripted)
        self.calls = []
        self.released = False
        self.restaged = []

    def serving_step(self, cur_token, cur_pos):
        self.calls.append((cur_token, cur_pos))
        committed = self.scripted.pop(0)
        return committed, len(committed) - 1  # m = accepted, committed = m+1

    def serving_release(self):
        self.released = True

    def refresh_page_tables(self, page_table):
        """The decode path re-stages the captured page tables whenever vLLM's
        block table for the request changes; record the rows it was given."""
        self.restaged.append(page_table.clone())


class _Harness:
    """Minimal stand-in exposing only what ``decode_forward`` touches on the
    solo path with a live session."""

    _SPEC_N = 6
    _spec_budget_end = 10_000
    _spec_horizon = 2048  # only read by the horizon-exhausted log line

    def __init__(self, scripted, start_pos=100):
        self._spec = _StubSession(scripted)
        self._spec_cur = (11, start_pos)
        self._spec_pending = None  # no bootstrap on this path
        self._spec_pending_owner = None
        # No page table is passed on this path, so ownership is unknown on both
        # sides and the live session is served -- see _spec_active_is_mine.
        self._spec_active_owner = None
        self._eos = 1

    def _eos_fill_id(self):
        return self._eos

    # real implementations, so the tests exercise shipped logic
    _spec_pt_identity = staticmethod(MTP._spec_pt_identity)
    # The adaptive gate counts REAL requests, not the padded decode wire width
    # (the runner pads positions with -1); the stub drives it with one row.
    _spec_real_batch = staticmethod(MTP._spec_real_batch)
    _spec_pending_is_mine = MTP._spec_pending_is_mine
    _spec_active_is_mine = MTP._spec_active_is_mine
    run = MTP.decode_forward


def _row(h, tokens=None):
    """Drive just the block loop by calling decode_forward with a B=1 input."""
    toks = torch.tensor([[11]], dtype=torch.int32) if tokens is None else tokens
    return h.run(tokens=toks, start_pos=torch.tensor([[100]], dtype=torch.int32))


def test_short_iterations_are_concatenated_to_a_full_row():
    """Three 2-token iterations must produce ONE 6-wide row, not a padded 2."""
    h = _Harness([[21, 22], [23, 24], [25, 26]])
    out = _row(h)
    assert out.shape == (1, 6)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    assert len(h._spec.calls) == 3  # looped rather than padding after one


def test_position_advances_by_tokens_emitted():
    """The session must resume from what was EMITTED. Iterations are driven from
    (token, pos), so a wrong position re-drafts the wrong prefix."""
    h = _Harness([[21, 22], [23, 24], [25, 26]], start_pos=100)
    _row(h)
    tok, pos = h._spec_cur
    assert pos == 100 + 6
    assert tok == 26
    # each iteration was driven from the previous one's last token / next pos
    assert h._spec.calls == [(11, 100), (22, 102), (24, 104)]


def test_overshoot_truncates_and_resumes_from_emitted():
    """A final iteration may overshoot N. The extra tokens are dropped and the
    position reflects the row, so the next step re-drafts and overwrites them."""
    h = _Harness([[21, 22, 23, 24], [25, 26, 27, 28]], start_pos=200)
    out = _row(h)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    tok, pos = h._spec_cur
    assert pos == 200 + 6 and tok == 26


def test_full_iteration_needs_no_loop():
    """High acceptance fills the row in one call; do not spin further."""
    h = _Harness([[21, 22, 23, 24, 25, 26]])
    out = _row(h)
    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    assert len(h._spec.calls) == 1


def test_eos_stops_the_loop_and_fills_the_tail():
    """A genuine stop is the ONLY case that may leave a short row; the tail is
    EOS so upstream trims there."""
    h = _Harness([[21, 1], [99, 99]])  # eos id is 1
    out = _row(h)
    assert out[0].tolist()[:2] == [21, 1]
    assert out[0].tolist()[2:] == [1, 1, 1, 1]  # EOS-filled tail
    assert len(h._spec.calls) == 1  # stopped, did not consume the next script


def test_horizon_exhaustion_emits_eos_without_stepping():
    """Past the horizon the packed verify would attend past its capture, so the
    request ends cleanly instead of stepping again."""
    h = _Harness([[21, 22]])
    h._spec_budget_end = 100  # cur_pos starts at 100
    out = _row(h)
    assert out[0].tolist() == [1, 1, 1, 1, 1, 1]
    assert h._spec.calls == []


def _live_session_instance(monkeypatch, scripted, owner, baseline="baseline-out"):
    """A real Gemma4MTPForCausalLM with a live session and no device.

    ``__new__`` skips the device-bound ``__init__``; the attributes below are
    everything ``decode_forward`` reads on the solo path. The BASE class's
    ``decode_forward`` is the plain-baseline fallback the ownership check hands
    off to, so it is stubbed to a sentinel -- the stub harness above cannot
    reach it at all (zero-arg ``super()`` needs a real instance).
    """
    from types import SimpleNamespace

    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    monkeypatch.setattr(Gemma4ForCausalLM, "decode_forward", lambda self, *a, **k: baseline)
    h = MTP.__new__(MTP)
    h._spec = _StubSession(scripted)
    h._spec_cur = (11, 100)
    h._spec_pending = None
    h._spec_pending_owner = None
    h._spec_active_owner = owner
    h._spec_budget_end = 10_000
    h._spec_horizon = 2048
    h._spec_first_step = False
    h._spec_last_pt = None  # nothing staged yet -> the first step stages
    h._bounded_sliding_kv_cache = False
    h._spec_warm = False  # this harness drives a session directly, not via warmup
    h.model = [SimpleNamespace(hf_config=SimpleNamespace(eos_token_id=1))]
    return h


def test_live_session_is_refused_for_a_request_that_does_not_own_it(monkeypatch):
    """A solo decode step can be scheduled for a request that is NOT the live
    session's owner: async scheduling skips a request that has reached
    max_tokens (upstream guards that skip on num_output_placeholders), so the
    owner can drop out of a step while its session is still armed. Serving the
    block loop here emits the OWNER's speculation for this request, against the
    single placeholder the scheduler reserved for a non-owner. The session must
    be released and the step served as plain baseline instead.
    """
    pt = torch.tensor([[3, 4, 5]], dtype=torch.int32)
    other = ("not-this-request",)
    h = _live_session_instance(monkeypatch, [[21, 22, 23, 24, 25, 26]], other)
    session = h._spec

    out = h.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[100]], dtype=torch.int32),
        page_table=pt,
    )

    assert out == "baseline-out"
    assert h._spec is None  # released
    assert session.calls == []  # never speculated for the non-owner
    assert h._spec_active_owner is None


def test_live_session_is_served_for_its_own_owner(monkeypatch):
    """The companion: a matching identity keeps the session and blocks."""
    pt = torch.tensor([[3, 4, 5]], dtype=torch.int32)
    h = _live_session_instance(monkeypatch, [[21, 22, 23, 24, 25, 26]], MTP._spec_pt_identity(pt))

    out = h.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[100]], dtype=torch.int32),
        page_table=pt,
    )

    assert out[0].tolist() == [21, 22, 23, 24, 25, 26]
    assert h._spec is not None


def test_page_tables_are_restaged_when_the_block_table_changes(monkeypatch):
    """vLLM allocates a KV block only every block_size tokens, so a session
    captured at prefill holds the prompt's blocks and zeros past them. The
    fused verify binds its page tables at capture, so without a re-stage it
    reads AND writes the null block once generation crosses out of the
    prompt's last block (tt-metal#55548 D2 on the qwen36 MTP trace; here it
    also reaches the host map behind the per-iteration hot-block uploads).
    """
    pt = torch.tensor([[3, 4, 5, 0]], dtype=torch.int32)
    h = _live_session_instance(monkeypatch, [[21, 22, 23, 24, 25, 26]] * 4, None)
    session = h._spec

    h.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[100]], dtype=torch.int32),
        page_table=pt,
    )
    assert len(session.restaged) == 1  # first step stages the current table

    # Same table on the next step: nothing to do.
    h.decode_forward(
        tokens=torch.tensor([[26]], dtype=torch.int32),
        start_pos=torch.tensor([[106]], dtype=torch.int32),
        page_table=pt,
    )
    assert len(session.restaged) == 1

    # A newly allocated block appears -> re-stage.
    grown = torch.tensor([[3, 4, 5, 9]], dtype=torch.int32)
    h.decode_forward(
        tokens=torch.tensor([[26]], dtype=torch.int32),
        start_pos=torch.tensor([[112]], dtype=torch.int32),
        page_table=grown,
    )
    assert len(session.restaged) == 2
    assert session.restaged[-1].reshape(-1).tolist() == [3, 4, 5, 9]


def test_a_warm_capture_with_no_active_request_serves_baseline(monkeypatch):
    """The batch-to-solo transition with _spec_warm=True.

    A warm release deliberately KEEPS the capture -- the warmup traces are a
    process artifact, and freeing them would leave no captured widths and no
    way to recapture -- and nulls the active request only. Gating on `_spec`
    alone therefore fell through to `cur_token, cur_pos = self._spec_cur` and
    raised TypeError on an ordinary two-requests-to-one transition.
    """
    pt = torch.tensor([[3, 4, 5]], dtype=torch.int32)
    owner = tuple(pt.reshape(-1).tolist())
    h = _live_session_instance(monkeypatch, [[21, 22, 23]], owner)
    h._spec_warm = True

    # Step 1: two real requests -> release the session, keep the warm capture.
    batched = h.decode_forward(
        tokens=torch.tensor([[11], [12]], dtype=torch.int32),
        start_pos=torch.tensor([[100], [100]], dtype=torch.int32),
        page_table=pt,
    )
    assert batched == "baseline-out"
    assert h._spec is not None, "a warm capture must survive the release"
    assert h._spec_cur is None, "but no request owns it any more"

    # Step 2: back to one request, no new prefill -> baseline, not TypeError.
    solo = h.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[101]], dtype=torch.int32),
        page_table=pt,
    )
    assert solo == "baseline-out"


# ── Missing drafter must fail at STARTUP, not on the first solo request ──────


def _warmup_stub(monkeypatch, tmp_path, assistant_env=None, cache_hit=False):
    """An MTP instance whose warmup can be driven host-only.

    ``__new__`` skips the device-bound ``__init__``. ``_spec_warmup_session`` is
    stubbed to succeed, so anything that raises came from the pre-flight and not
    from a capture failure.
    """
    monkeypatch.delenv("GEMMA4_ASSISTANT_MODEL", raising=False)
    if assistant_env is not None:
        monkeypatch.setenv("GEMMA4_ASSISTANT_MODEL", assistant_env)
    # HF_MODEL drives the fallback candidate path f"{HF_MODEL}-assistant".
    monkeypatch.setenv("HF_MODEL", str(tmp_path / "gemma-4-12B-it"))
    # This box HAS the assistant in its real HF cache, so without pinning the
    # resolver the "missing" case silently resolves and the test proves nothing.
    import models.demos.gemma4.tt.generator_vllm as gv

    monkeypatch.setattr(
        gv, "_hf_resolve_repo", lambda repo, cache_dir_name: (str(tmp_path / "cached-assistant") if cache_hit else None)
    )
    if cache_hit:
        (tmp_path / "cached-assistant").mkdir(exist_ok=True)
    h = MTP.__new__(MTP)
    h._spec = None
    h._spec_warm = False
    h._decode_warmup_complete = False
    h._mtp_warmup_capture = True  # the default path, and the one CI uses
    monkeypatch.setattr(MTP, "_spec_warmup_session", lambda self, kv, nb: None)
    monkeypatch.setattr(MTP, "_spec_release_session", lambda self, force=False: None)
    return h


def test_a_missing_assistant_fails_warmup_instead_of_the_first_request(monkeypatch, tmp_path, expect_error):
    """The batched benchmark never touches the drafter, so a server that boots
    without one passes every throughput check and then dies on the first SOLO
    request (EngineDeadError 500 at the coherence guard). Fail while starting."""
    h = _warmup_stub(monkeypatch, tmp_path)
    with expect_error(RuntimeError, "snapshot not found -- resolved"):
        h.warmup_model_decode(kv_cache=[[object()]], num_blocks=8, enable_trace=True)


def test_an_operator_supplied_assistant_path_is_left_alone(monkeypatch, tmp_path):
    """An explicit GEMMA4_ASSISTANT_MODEL is the operator's business: the
    pre-flight must not second-guess it (it may be a repo id, not a dir)."""
    h = _warmup_stub(monkeypatch, tmp_path, assistant_env="google/gemma-4-12B-it-assistant")
    h.warmup_model_decode(kv_cache=[[object()]], num_blocks=8, enable_trace=True)


def test_an_existing_assistant_dir_passes_the_preflight(monkeypatch, tmp_path):
    """The resolver finds a cached snapshot, so warmup proceeds to capture."""
    h = _warmup_stub(monkeypatch, tmp_path, cache_hit=True)
    h.warmup_model_decode(kv_cache=[[object()]], num_blocks=8, enable_trace=True)

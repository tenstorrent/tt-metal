# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Gemma4 dFlash on the plugin CONTRACT rail (vllm-tt-plugin#110).

The review on tt-metal#56048 asks for a first contract adapter at B=1, greedy,
synchronous, one K: the runner supplies the candidate block and owns the accept
walk, ``output_tokens_per_step`` is 1, and the adapter actually executes
verification rather than falling back to the width-one baseline branch.

These are host-only tests of the CONTRACT SURFACE -- declarations, shapes,
dtypes, modes, the proposed/verified pairing -- with a stub fused decoder. The
device behaviour (that the posterior is the right argmax for the block) is not
something a host test can attest; that needs the paired correctness run.
"""

import pytest
import torch

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashContractForCausalLM as CT


def _plugin_has_num_valid():
    """DraftOutput.num_valid arrives with the plugin's contract stack.

    That stack lands separately from this adapter, so a checkout can have the
    adapter and not the field. The adapter feature-detects it; these
    assertions cannot, so they skip rather than fail against a plugin that
    predates it.
    """
    from vllm_tt_plugin.spec_decode import DraftOutput

    return "num_valid" in getattr(DraftOutput, "__dataclass_fields__", {})


needs_num_valid = pytest.mark.skipif(not _plugin_has_num_valid(), reason="plugin DraftOutput has no num_valid yet")


class _Dec:
    """Fused decoder stub: one replay yields (drafts, posterior)."""

    def __init__(self, script):
        self.script = list(script)
        self.commits = []
        self.replays = 0

    def contract_replay(self, first=False):
        self.replays += 1
        return self.script.pop(0)

    def contract_commit(self, produced, anchor):
        self.commits.append((int(produced), int(anchor)))
        return 0


def _model(script, active=True):
    m = CT.__new__(CT)
    m._SPEC_CONTRACT_K = 5
    m._spec_decoder = _Dec(script) if active else None
    m._spec_active = active
    m._spec_first_step = False
    m._spec_pending = None  # prefill already armed and bootstrapped
    m._spec_width_set = False  # per-test: the width-selection test turns it on
    m._ct_posterior = None
    m._ct_drafts = None
    return m


def _bootstrap_step(m):
    """The contract's FIRST step: a verify with no drafts, which commits exactly
    one token, followed by the first proposal."""
    out = m.decode_forward(
        tokens=torch.tensor([[11]], dtype=torch.int32),
        start_pos=torch.tensor([[100]], dtype=torch.int32),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.tensor([0], dtype=torch.int32),
        accepted_counts=torch.tensor([1], dtype=torch.int32),
    )
    return out


def _counts(n):
    return torch.tensor([n], dtype=torch.int32)


def _positions(rows, live=1, start=100):
    """Committed positions as the runner builds them: padded rows carry -1."""
    p = torch.full((rows, 6), -1, dtype=torch.int32)
    for r in range(live):
        p[r] = torch.arange(start, start + 6, dtype=torch.int32)
    return p


# ── declarations ────────────────────────────────────────────────────────────


def test_declares_the_contract_rail_not_the_block_rail():
    caps = CT.model_capabilities
    assert caps["output_tokens_per_step"] == 1
    assert caps["supports_spec_decode"] is True
    assert "tt_adaptive_block_output" not in caps
    assert set(caps["spec_requirements"]) == {"device_propose", "hidden_feed"}
    assert caps["spec_hidden_handoff"] == ("on_device",)
    # A batched draft-less step decodes plain baseline and must answer with
    # ids, so the device sampler has to be available; declaring it False makes
    # the platform refuse sample_on_device_mode at config validation, and the
    # step then argmaxes [B, vocab] logits on host every time.
    assert caps["supports_sample_on_device"] is True
    # Synchronous by default, per the scope the review set for a first adapter.
    assert caps["supports_async_decode"] is False


def test_the_plugin_admits_this_declaration():
    """Run the PLUGIN's own admission over our capabilities, so the test fails
    if either side of the contract moves."""
    spec_admission = pytest.importorskip("vllm_tt_plugin.spec_admission")
    from types import SimpleNamespace

    spec_cfg = SimpleNamespace(
        method="custom_class",
        model="vllm_tt_plugin.model_owned_drafter",
        num_speculative_tokens=5,
    )
    plan = spec_admission.resolve_speculative_plan(
        SimpleNamespace(speculative_config=spec_cfg),
        CT,
        CT.model_capabilities,
        1,
    )
    assert plan.effective_k == 5
    assert "argmax_ids" in plan.accept_modes
    assert plan.lanes_per_request == 1
    # the per-session byte accounting is inherited, not re-derived here
    assert plan.extra_bytes_per_seq > 0


def test_a_wrong_sentinel_is_refused_by_the_plugin(expect_error):
    spec_admission = pytest.importorskip("vllm_tt_plugin.spec_admission")
    from types import SimpleNamespace

    spec_cfg = SimpleNamespace(method="custom_class", model="some.other:Proposer", num_speculative_tokens=5)
    with expect_error(ValueError, "model_owned_drafter"):
        spec_admission.resolve_speculative_plan(
            SimpleNamespace(speculative_config=spec_cfg), CT, CT.model_capabilities, 1
        )


# ── propose ─────────────────────────────────────────────────────────────────


def test_propose_returns_int32_drafts_of_width_k():
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    out = m.propose_draft_tokens(5, torch.tensor([[11]], dtype=torch.int32), None, _counts(1))
    assert out.draft_token_ids.dtype == torch.int32
    assert tuple(out.draft_token_ids.shape) == (1, 5)
    assert out.draft_token_ids[0].tolist() == [21, 22, 23, 24, 25]
    assert out.draft_scores is None


@needs_num_valid
def test_propose_without_a_session_proposes_nothing():
    """A declined row says so with num_valid, not with its ids.

    The contract's ids are always [rows, K] and every column has to be a real
    in-vocabulary id, so the shape cannot carry "nothing this step"; a count of
    0 is what makes the runner record nothing, and the next verify then sends
    no drafts for that row.
    """
    m = _model([], active=False)
    out = m.propose_draft_tokens(5, torch.tensor([[11]], dtype=torch.int32), None, _counts(1))
    assert tuple(out.draft_token_ids.shape) == (1, 5)
    assert out.num_valid.tolist() == [0]


def test_the_runners_accepted_count_drives_the_commit():
    """The model must commit what the RUNNER accepted, not its own walk: the
    count and the anchor both come from the accept walk just performed."""
    m = _model(
        [
            ([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36]),
            ([41, 42, 43, 44, 45], [51, 52, 53, 54, 55, 56]),
        ]
    )
    committed = torch.tensor([[21, 22, 23, 99, 0, 0]], dtype=torch.int32)
    m.propose_draft_tokens(5, committed, None, _counts(3))
    assert m._spec_decoder.commits == [(3, 23)]  # 3 rows, anchor = 3rd token


# ── verify ──────────────────────────────────────────────────────────────────


def _verify(m, block, valid=None):
    return m.decode_forward(
        tokens=torch.tensor([block], dtype=torch.int32),
        start_pos=torch.tensor([list(range(len(block)))], dtype=torch.int32),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.tensor([valid if valid is not None else len(block) - 1], dtype=torch.int32),
        accepted_counts=_counts(1),
    )


def test_verify_returns_the_posterior_for_the_block_it_was_sent():
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    m.propose_draft_tokens(5, torch.tensor([[11]], dtype=torch.int32), None, _counts(1))
    out = _verify(m, [11, 21, 22, 23, 24, 25])
    assert out.spec_mode == "argmax_ids"
    assert out.argmax_ids.dtype == torch.int32
    assert tuple(out.argmax_ids.shape) == (1, 6)
    assert out.argmax_ids[0].tolist() == [31, 32, 33, 34, 35, 36]
    assert out.hidden is None  # on-device handoff


def test_verify_refuses_a_block_carrying_drafts_we_did_not_propose(expect_error):
    """Answering would report a device posterior for tokens the device never
    evaluated -- silently wrong tokens, so it raises."""
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    m.propose_draft_tokens(5, torch.tensor([[11]], dtype=torch.int32), None, _counts(1))
    with expect_error(ValueError, "did not propose"):
        _verify(m, [11, 21, 77, 23, 24, 25])


def test_verify_tolerates_a_row_truncated_to_num_valid_drafts():
    """Grammar truncation is legitimate: fewer real drafts, same answer."""
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    m.propose_draft_tokens(5, torch.tensor([[11]], dtype=torch.int32), None, _counts(1))
    out = _verify(m, [11, 21, 22, -1, -1, -1], valid=2)
    assert out.argmax_ids[0].tolist()[:3] == [31, 32, 33]


def test_a_block_carrying_drafts_with_no_proposal_behind_it_raises(expect_error):
    """The runner sent drafts, but this model holds no posterior for them, so
    the two have diverged. Answering would invent a verify result."""
    m = _model([])
    with expect_error(RuntimeError, "holds no device"):
        _verify(m, [11, 21, 22, 23, 24, 25])


def test_a_non_speculative_step_is_not_a_verify():
    """Without spec_mode the runner is not asking for verification, so the
    adapter must not answer with a VerifyOutput."""
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    m = _model([])
    called = {}
    orig = Gemma4ForCausalLM.decode_forward
    Gemma4ForCausalLM.decode_forward = lambda self, *a, **k: called.setdefault("plain", True)
    try:
        m.decode_forward(tokens=torch.tensor([[11]], dtype=torch.int32))
    finally:
        Gemma4ForCausalLM.decode_forward = orig
    assert called == {"plain": True}


# ── row dimension: the contract is [rows, ...], padding rows included ───────


def test_propose_returns_one_row_per_verified_row():
    """The runner checks ``draft_token_ids.shape == (rows, K)`` where rows is
    the VERIFIED row count, padding included -- it drops padding rows only
    after its accept walk. B=1 fills row 0; the shape follows the block."""
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    committed = torch.zeros((4, 6), dtype=torch.int32)  # wire padded to 4 rows
    committed[0, 0] = 11
    out = m.propose_draft_tokens(5, committed, _positions(4, live=1), torch.tensor([1, 1, 1, 1], dtype=torch.int32))
    assert tuple(out.draft_token_ids.shape) == (4, 5)
    assert out.draft_token_ids[0].tolist() == [21, 22, 23, 24, 25]
    assert int(out.draft_token_ids[1:].sum()) == 0  # padding rows carry no draft


def test_verify_returns_one_row_per_block_row():
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    committed = torch.zeros((4, 6), dtype=torch.int32)
    committed[0, 0] = 11
    m.propose_draft_tokens(5, committed, _positions(4, live=1), torch.tensor([1, 1, 1, 1], dtype=torch.int32))
    block = torch.zeros((4, 6), dtype=torch.int32)
    block[0] = torch.tensor([11, 21, 22, 23, 24, 25], dtype=torch.int32)
    out = m.decode_forward(
        tokens=block,
        # One live row and three of the runner's padding rows: the block's row
        # dimension is the wire bucket, not the number of requests.
        start_pos=_positions(4, live=1),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.tensor([5, 0, 0, 0], dtype=torch.int32),
        accepted_counts=torch.tensor([1, 1, 1, 1], dtype=torch.int32),
    )
    assert tuple(out.argmax_ids.shape) == (4, 6)
    assert out.argmax_ids[0].tolist() == [31, 32, 33, 34, 35, 36]


def test_propose_selects_the_width_for_the_new_position():
    """The block loop calls select_width every iteration because the position
    moves; the contract rail must too, or a session stays on the width its
    PROMPT selected and eventually replays a trace whose mask and page table
    stop short of the live top."""
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    m._spec_width_set = True
    seen = []
    m._spec_decoder.select_width = lambda pos: (seen.append(pos), 4096)[1]
    m._spec_decoder.start = 777
    m.propose_draft_tokens(5, torch.tensor([[11]], dtype=torch.int32), None, _counts(1))
    assert seen == [777]


# ── adaptive on the contract rail: solo speculates, batched does not ────────


@needs_num_valid
def test_a_batched_step_proposes_nothing_and_drops_the_session():
    """The fused verify packs its candidate positions into ONE batch row, so it
    cannot speculate for several requests. Declining every row is how this rail
    gives up speculation for the step, with no scheduler reservation involved
    -- and the session goes too, because its taps belong to one request's
    prompt."""
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    released = {}
    m._spec_release_decoder = lambda *a, **k: released.setdefault("yes", True)
    committed = torch.zeros((4, 6), dtype=torch.int32)
    out = m.propose_draft_tokens(5, committed, _positions(4, live=3), torch.tensor([1, 1, 1, 1], dtype=torch.int32))
    assert tuple(out.draft_token_ids.shape) == (4, 5)
    assert out.num_valid.tolist() == [0, 0, 0, 0]  # every row declined
    assert released == {"yes": True}
    assert m._spec_decoder.replays == 0  # nothing drafted


def test_a_padded_solo_step_still_speculates():
    """The runner pads a decode batch to a wire bucket; counting rows instead
    of live requests would hand the drafter's own request a plain decode and
    give up speculation entirely."""
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    committed = torch.zeros((8, 6), dtype=torch.int32)
    committed[0, 0] = 11
    out = m.propose_draft_tokens(5, committed, _positions(8, live=1), torch.tensor([1] * 8, dtype=torch.int32))
    assert tuple(out.draft_token_ids.shape) == (8, 5)
    assert out.draft_token_ids[0].tolist() == [21, 22, 23, 24, 25]
    assert m._spec_decoder.replays == 1


def test_live_row_counting_uses_the_padding_sentinel():
    from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashContractForCausalLM as C

    assert C._contract_live_rows(_positions(8, live=1), 8) == 1
    assert C._contract_live_rows(_positions(8, live=5), 8) == 5
    assert C._contract_live_rows(_positions(4, live=4), 4) == 4
    # unknown positions: assume batched rather than speculate wrongly
    assert C._contract_live_rows(None, 4) == 4


# ── batched steps: the block is not the batch ───────────────────────────────


def _stub_plain_decode(monkeypatch, seen):
    """Stand in for the baseline decode, recording what it was handed.

    The real one reads its batch as ``tokens.reshape(-1).shape[0]`` and refuses
    a batch wider than the token feedback width, so what this records IS the
    thing that broke on device: a 32-row contract block arriving as 192 rows.
    """
    from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

    def decode_forward(self, *args, page_tables_per_layer=None, **kwargs):
        tokens = kwargs.get("tokens", args[0] if args else None)
        pos = kwargs.get("start_pos", args[1] if len(args) > 1 else None)
        seen["tokens"] = tokens
        seen["start_pos"] = pos
        seen["batch"] = int(tokens.reshape(-1).shape[0])
        return "tt_out"

    def read_decode_output(self, tt_out, async_read=False, *_, **__):
        return ["host_tensors_per_dp_group"]

    def process_decode_output_host(self, host, is_tokens=False):
        # Logits, as a launch that samples on host produces them: [B, S, vocab]
        # concatenated across data-parallel ranks, argmax row r -> 500+r.
        b = seen["batch"]
        logits = torch.full((b, 1, 600), -1.0)
        for r in range(b):
            logits[r, 0, 500 + r] = 1.0
        return logits, None

    monkeypatch.setattr(Gemma4ForCausalLM, "decode_forward", decode_forward)
    monkeypatch.setattr(Gemma4ForCausalLM, "read_decode_output", read_decode_output)
    monkeypatch.setattr(Gemma4ForCausalLM, "process_decode_output_host", process_decode_output_host)


def test_a_batched_step_decodes_the_committed_column_not_the_whole_block(monkeypatch):
    seen = {}
    _stub_plain_decode(monkeypatch, seen)
    m = _model([], active=False)
    rows = 32
    block = torch.arange(rows * 6, dtype=torch.int32).reshape(rows, 6)
    out = m.decode_forward(
        tokens=block,
        start_pos=_positions(rows, live=rows),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.zeros(rows, dtype=torch.int32),
        accepted_counts=torch.ones(rows, dtype=torch.int32),
    )
    # One row per request, NOT one per candidate column.
    assert seen["batch"] == rows
    assert seen["tokens"].reshape(-1).tolist() == block[:, 0].tolist()
    assert seen["start_pos"].reshape(-1).tolist() == [100] * rows
    # Column 0 carries each row's argmax; the draft columns stay unanswered.
    assert tuple(out.argmax_ids.shape) == (rows, 6)
    assert out.argmax_ids[:, 0].tolist() == list(range(500, 500 + rows))


def test_a_batched_step_keeps_the_padding_sentinel_on_padded_rows(monkeypatch):
    seen = {}
    _stub_plain_decode(monkeypatch, seen)
    m = _model([], active=False)
    block = torch.arange(4 * 6, dtype=torch.int32).reshape(4, 6)
    m.decode_forward(
        tokens=block,
        start_pos=_positions(4, live=2),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.zeros(4, dtype=torch.int32),
        accepted_counts=torch.ones(4, dtype=torch.int32),
    )
    # The baseline decode gets exactly what it gets on any other padded step.
    assert seen["start_pos"].reshape(-1).tolist() == [100, 100, -1, -1]


def test_a_straddle_step_reports_the_posterior_for_the_drafts_it_proposed(monkeypatch):
    """A peer joined between our proposal and its verify.

    The device has already evaluated THOSE drafts, so the kept posterior is the
    honest answer for row 0. Answering column 0 alone would let the walk accept
    draft 0 -- it may well be right -- and then read its bonus out of a column
    nothing answered.
    """
    seen = {}
    _stub_plain_decode(monkeypatch, seen)
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    committed = torch.zeros((1, 6), dtype=torch.int32)
    committed[0, 0] = 11
    m.propose_draft_tokens(5, committed, _positions(1, live=1), _counts(1))
    block = torch.zeros((2, 6), dtype=torch.int32)
    block[0] = torch.tensor([11, 21, 22, 23, 24, 25], dtype=torch.int32)
    out = m.decode_forward(
        tokens=block,
        start_pos=_positions(2, live=2),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.tensor([5, 0], dtype=torch.int32),
        accepted_counts=torch.tensor([1, 1], dtype=torch.int32),
    )
    assert out.argmax_ids[0].tolist() == [31, 32, 33, 34, 35, 36]
    assert int(out.argmax_ids[1, 0]) == 501
    # The next proposal is the batched one that drops the session, so nothing
    # here may leave a posterior behind for it to answer with.
    assert m._ct_posterior is None and m._ct_drafts is None


def test_a_straddle_step_ignores_a_posterior_for_other_drafts(monkeypatch):
    seen = {}
    _stub_plain_decode(monkeypatch, seen)
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    committed = torch.zeros((1, 6), dtype=torch.int32)
    committed[0, 0] = 11
    m.propose_draft_tokens(5, committed, _positions(1, live=1), _counts(1))
    block = torch.zeros((2, 6), dtype=torch.int32)
    block[0] = torch.tensor([11, 99, 98, 97, 96, 95], dtype=torch.int32)  # not ours
    out = m.decode_forward(
        tokens=block,
        start_pos=_positions(2, live=2),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.tensor([5, 0], dtype=torch.int32),
        accepted_counts=torch.tensor([1, 1], dtype=torch.int32),
    )
    # Column 0 from the plain decode, and no posterior for tokens the device
    # never evaluated.
    assert int(out.argmax_ids[0, 0]) == 500
    assert out.argmax_ids[0, 1:].tolist() != [32, 33, 34, 35, 36]


def test_column_zero_narrowing_leaves_an_already_narrow_step_alone():
    args, kwargs = CT._contract_col0((), {"tokens": torch.tensor([7, 8]), "start_pos": torch.tensor([1, 2])}, 2)
    assert kwargs["tokens"].tolist() == [7, 8]
    assert kwargs["start_pos"].tolist() == [1, 2]


@needs_num_valid
def test_a_solo_proposal_declines_every_other_row():
    """The block's rows are the wire bucket, and only row 0 has a drafter.

    Without the per-row count the runner would record row 0's ids for the
    padding rows' requests too, and the next verify would be sent drafts this
    drafter cannot speak for.
    """
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    committed = torch.zeros((4, 6), dtype=torch.int32)
    committed[0, 0] = 11
    out = m.propose_draft_tokens(5, committed, _positions(4, live=1), torch.tensor([1, 1, 1, 1], dtype=torch.int32))
    assert out.num_valid.tolist() == [5, 0, 0, 0]
    assert out.draft_token_ids[0].tolist() == [21, 22, 23, 24, 25]


def test_the_rail_declares_the_narrow_decode_it_serves():
    """A batched step IS the plain decode's shape, so the runner may keep it
    narrow rather than widening every host tensor to 1+K."""
    from types import SimpleNamespace

    from vllm_tt_plugin.spec_decode import SpecPlan

    spec_cfg = SimpleNamespace(
        method="custom_class",
        model="vllm_tt_plugin.model_owned_drafter",
        num_speculative_tokens=5,
    )
    plan = CT.spec_plan(SimpleNamespace(speculative_config=spec_cfg), 32, 5)
    assert isinstance(plan, SpecPlan)
    assert plan.supports_narrow_decode is True

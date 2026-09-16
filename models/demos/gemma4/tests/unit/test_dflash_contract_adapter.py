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

from models.demos.gemma4.tt.generator_vllm import Gemma4DFlashContractForCausalLM as CT


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


# ── declarations ────────────────────────────────────────────────────────────


def test_declares_the_contract_rail_not_the_block_rail():
    caps = CT.model_capabilities
    assert caps["output_tokens_per_step"] == 1
    assert caps["supports_spec_decode"] is True
    assert "tt_adaptive_block_output" not in caps
    assert set(caps["spec_requirements"]) == {"device_propose", "hidden_feed"}
    assert caps["spec_hidden_handoff"] == ("on_device",)


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


def test_propose_without_a_session_proposes_nothing():
    """Zero width is the contract's "no drafts"; inventing ids would have the
    runner verify tokens no drafter produced."""
    m = _model([], active=False)
    out = m.propose_draft_tokens(5, torch.tensor([[11]], dtype=torch.int32), None, _counts(1))
    assert tuple(out.draft_token_ids.shape) == (1, 0)


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
    out = m.propose_draft_tokens(5, committed, None, torch.tensor([1, 1, 1, 1], dtype=torch.int32))
    assert tuple(out.draft_token_ids.shape) == (4, 5)
    assert out.draft_token_ids[0].tolist() == [21, 22, 23, 24, 25]
    assert int(out.draft_token_ids[1:].sum()) == 0  # padding rows carry no draft


def test_verify_returns_one_row_per_block_row():
    m = _model([([21, 22, 23, 24, 25], [31, 32, 33, 34, 35, 36])])
    committed = torch.zeros((4, 6), dtype=torch.int32)
    committed[0, 0] = 11
    m.propose_draft_tokens(5, committed, None, torch.tensor([1, 1, 1, 1], dtype=torch.int32))
    block = torch.zeros((4, 6), dtype=torch.int32)
    block[0] = torch.tensor([11, 21, 22, 23, 24, 25], dtype=torch.int32)
    out = m.decode_forward(
        tokens=block,
        start_pos=torch.zeros((4, 6), dtype=torch.int32),
        spec_mode="argmax_ids",
        num_valid_drafts=torch.tensor([5, 0, 0, 0], dtype=torch.int32),
        accepted_counts=torch.tensor([1, 1, 1, 1], dtype=torch.int32),
    )
    assert tuple(out.argmax_ids.shape) == (4, 6)
    assert out.argmax_ids[0].tolist() == [31, 32, 33, 34, 35, 36]

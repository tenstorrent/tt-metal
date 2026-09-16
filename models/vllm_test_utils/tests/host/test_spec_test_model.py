# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host tests for the speculative test model's verify and draft arithmetic.

`DummySpecDecodeModel._verified_ids` decides what the model claims at each
candidate position, and `DummySpecDecodeModel.propose_draft_tokens` decides
what it drafts next. Both are pure PyTorch and neither needs TTNN or a device,
so they can be tested directly here.

These tests do need `vllm_tt_plugin` on the path, because the drafter validates
its inputs against the contract module's own validator and returns the
contract's `DraftOutput`: a witness that checked a weaker domain than the
contract states would be worth little. The verify tests alone need nothing
beyond PyTorch.

Worth testing rather than assuming, because several properties here are easy to
get wrong in a way that nothing downstream reports. A bonus written to a column
fixed for the batch instead of to each row's own draft count silently commits
the wrong token on a row that carries fewer drafts than its neighbour. An
accept depth applied across the batch instead of per row lets one row's short
draft list shorten another's speculation. And a drafter reading a fixed column
of the committed block instead of each row's `accepted_counts - 1` entry
continues a row from padding or from another row's token.

The layout under test: column `j` is the choice draft `j` has to match, for `j`
below the row's count, and the column at the row's count carries the bonus that
follows a fully accepted row.
"""

import torch

from models.vllm_test_utils.spec_test.test_model import DummySpecDecodeModel

VOCAB = 1024

# Distinguishes "pass the model's own handle" from "pass None on purpose".
_UNSET = object()


def _model(monkeypatch, accept_depth=None):
    """A model with no device, at the given accept depth.

    The depth is read from the environment when the instance is built, so it is
    set before construction rather than assigned afterwards.
    """
    if accept_depth is None:
        monkeypatch.delenv("TT_SPEC_ACCEPT_DEPTH", raising=False)
    else:
        monkeypatch.setenv("TT_SPEC_ACCEPT_DEPTH", str(accept_depth))
    return DummySpecDecodeModel(mesh_device=None, max_batch_size=8, vocab_size=VOCAB)


def _block(rows):
    """One `[rows, 4]` candidate block: a committed token and three drafts."""
    return torch.tensor([[100 + 10 * r, 11, 12, 13] for r in range(rows)], dtype=torch.int32)


def _bonus_for(tokens, row, num_valid):
    """The bonus the model's own arithmetic produces for that row."""
    return (int(tokens[row, 0]) + num_valid + 1) % VOCAB


def test_every_draft_is_accepted_by_default(monkeypatch):
    model = _model(monkeypatch)
    tokens = _block(1)

    ids = model._verified_ids(tokens, torch.tensor([3], dtype=torch.int32))

    # Columns 0..2 return the drafts unchanged, which is what accepting means.
    assert ids[0, :3].tolist() == tokens[0, 1:].tolist()
    assert int(ids[0, 3]) == _bonus_for(tokens, 0, 3)


def test_accept_depth_zero_matches_no_draft(monkeypatch):
    """The measurement mode: every row commits one token and no more."""
    model = _model(monkeypatch, accept_depth=0)
    tokens = _block(1)

    ids = model._verified_ids(tokens, torch.tensor([3], dtype=torch.int32))

    for column in range(3):
        assert int(ids[0, column]) != int(tokens[0, 1 + column])


def test_accept_depth_matches_exactly_that_many_drafts(monkeypatch):
    model = _model(monkeypatch, accept_depth=2)
    tokens = _block(1)

    ids = model._verified_ids(tokens, torch.tensor([3], dtype=torch.int32))

    assert ids[0, :2].tolist() == tokens[0, 1:3].tolist()
    assert int(ids[0, 2]) != int(tokens[0, 3])


def test_the_bonus_sits_at_each_row_s_own_draft_count(monkeypatch):
    """A row carrying fewer drafts finds its bonus earlier in the block.

    A bonus written to a column fixed for the batch would leave the column the
    accept walk actually reads holding draft arithmetic instead, and the row
    would commit a token the model never chose.
    """
    model = _model(monkeypatch)
    tokens = _block(3)
    num_valid = torch.tensor([3, 1, 0], dtype=torch.int32)

    ids = model._verified_ids(tokens, num_valid)

    for row, count in enumerate(num_valid.tolist()):
        assert int(ids[row, count]) == _bonus_for(
            tokens, row, count
        ), f"row {row} carries {count} draft(s), so its bonus belongs at column {count}"


def test_a_draftless_row_carries_its_bonus_in_column_zero(monkeypatch):
    """The narrow case, and the one a fixed bonus column gets wrong."""
    model = _model(monkeypatch)
    tokens = _block(1)

    ids = model._verified_ids(tokens, torch.tensor([0], dtype=torch.int32))

    assert int(ids[0, 0]) == _bonus_for(tokens, 0, 0)


def test_the_accept_depth_is_capped_per_row_not_across_the_batch(monkeypatch):
    """One row's short draft list must not shorten another's.

    Both rows are offered the same three drafts and the model accepts two of
    them, but the second row only has one real draft. The first row still
    matches two.
    """
    model = _model(monkeypatch, accept_depth=2)
    tokens = _block(2)

    ids = model._verified_ids(tokens, torch.tensor([3, 1], dtype=torch.int32))

    assert ids[0, :2].tolist() == tokens[0, 1:3].tolist()
    assert int(ids[1, 0]) == int(tokens[1, 1])
    assert int(ids[1, 1]) == _bonus_for(tokens, 1, 1)


def test_the_answer_is_int32_and_block_shaped(monkeypatch):
    """The accept walk compares it against an int32 draft block."""
    model = _model(monkeypatch)
    tokens = _block(4)

    ids = model._verified_ids(tokens, torch.tensor([3, 2, 1, 0], dtype=torch.int32))

    assert ids.shape == tokens.shape
    assert ids.dtype == torch.int32


def test_every_claimed_id_is_inside_the_vocabulary(monkeypatch):
    """Divergence and the bonus both wrap, so neither can leave the vocabulary.

    An out-of-vocabulary id would be committed as an output token, and the
    plugin range-checks drafts but not what a verify claims.
    """
    model = _model(monkeypatch, accept_depth=0)
    # A committed token and drafts at the top of the vocabulary, where the
    # arithmetic has to wrap.
    tokens = torch.tensor([[VOCAB - 1, VOCAB - 1, VOCAB - 2, VOCAB - 3]], dtype=torch.int32)

    ids = model._verified_ids(tokens, torch.tensor([3], dtype=torch.int32))

    assert int(ids.min()) >= 0
    assert int(ids.max()) < VOCAB


def _propose(model, committed, counts, num_drafts=3, hidden=_UNSET):
    """Call the drafter with the handle the model's own verify produced."""
    rows = committed.shape[0]
    positions = torch.arange(committed.shape[1], dtype=torch.int32).repeat(rows, 1)
    return model.propose_draft_tokens(
        num_drafts,
        committed,
        positions,
        counts,
        hidden=model._verify_hidden if hidden is _UNSET else hidden,
    )


def test_the_drafter_continues_from_each_row_s_own_last_committed_token(monkeypatch):
    """``accepted_counts - 1`` selects the row's last token, not a fixed column.

    Two rows commit different lengths of the same fixed-width block. A drafter
    reading a fixed column would continue the shorter row from padding, or from
    the longer row's token, and the whole row's output would be wrong from
    there on with nothing reporting it.
    """
    model = _model(monkeypatch)
    model._verify_hidden = object()
    committed = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.int32)
    counts = torch.tensor([4, 2], dtype=torch.int32)

    drafts = _propose(model, committed, counts).draft_token_ids

    # Row 0's last committed token is 13 (count 4), row 1's is 21 (count 2).
    assert drafts[0].tolist() == [14, 15, 16]
    assert drafts[1].tolist() == [22, 23, 24]


def test_the_drafts_are_what_this_model_s_own_verify_accepts(monkeypatch):
    """The property that makes a fixed committed width measurable.

    The drafter proposes ``last + 1 + j`` and the verify returns each draft
    unchanged up to its accept depth, so at full depth every draft is accepted
    and the step commits ``1+K``. If these two drifted apart, the acceptance
    rate would become a property of arithmetic coincidence.
    """
    model = _model(monkeypatch)
    model._verify_hidden = object()
    committed = torch.tensor([[100, 0, 0, 0]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    drafts = _propose(model, committed, counts).draft_token_ids
    # The verify's own input block for the next step: the row's last committed
    # token, then the drafts it was given.
    tokens = torch.cat([committed[:, :1], drafts], dim=1)
    verified = model._verified_ids(tokens, torch.tensor([3], dtype=torch.int32))

    assert verified[0, :3].tolist() == drafts[0].tolist()


def test_a_draft_id_stays_inside_the_vocabulary(monkeypatch):
    """The arithmetic wraps, because the plugin range-checks every draft."""
    model = _model(monkeypatch)
    model._verify_hidden = object()
    committed = torch.tensor([[VOCAB - 1, 0, 0, 0]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    drafts = _propose(model, committed, counts).draft_token_ids

    assert int(drafts.min()) >= 0
    assert int(drafts.max()) < VOCAB
    assert drafts.dtype == torch.int32


def test_a_foreign_hidden_handle_is_refused(monkeypatch, expect_error):
    """The handoff check: the runner must carry the handle back unchanged.

    A runner that dropped or replaced it would leave a real drafter running
    against another step's hidden state, which produces worse drafts and never
    an error. Here it produces an error.
    """
    model = _model(monkeypatch)
    model._verify_hidden = object()
    committed = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    with expect_error(ValueError, "hidden handle"):
        _propose(model, committed, counts, hidden=object())

    with expect_error(ValueError, "hidden handle"):
        _propose(model, committed, counts, hidden=None)


def test_a_committed_block_of_the_wrong_width_is_refused(monkeypatch, expect_error):
    model = _model(monkeypatch)
    model._verify_hidden = object()
    committed = torch.tensor([[10, 11]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    with expect_error(ValueError, "1\\+K wide"):
        _propose(model, committed, counts, num_drafts=3)


def test_an_accepted_count_outside_its_range_is_refused(monkeypatch, expect_error):
    """Validated against the contract module, not a private copy of its rules."""
    model = _model(monkeypatch)
    model._verify_hidden = object()
    committed = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)

    # The validator names the offending tensor and its range.
    with expect_error(ValueError, "accepted_counts"):
        _propose(model, committed, torch.tensor([0], dtype=torch.int32))
    with expect_error(ValueError, "accepted_counts"):
        _propose(model, committed, torch.tensor([5], dtype=torch.int32))


def test_the_verify_hands_out_a_fresh_handle_each_step(monkeypatch):
    """A drafter that cached a handle would be running on stale hidden state."""
    model = _model(monkeypatch)
    tokens = _block(1)
    num_valid = torch.tensor([3], dtype=torch.int32)

    first = model.decode_forward(
        tokens=tokens,
        start_pos=torch.zeros(1, 4, dtype=torch.int32),
        num_valid_drafts=num_valid,
        accepted_counts=torch.ones(1, dtype=torch.int32),
        spec_mode="argmax_ids",
    )
    second = model.decode_forward(
        tokens=tokens,
        start_pos=torch.zeros(1, 4, dtype=torch.int32),
        num_valid_drafts=num_valid,
        accepted_counts=torch.ones(1, dtype=torch.int32),
        spec_mode="argmax_ids",
    )

    assert first.hidden is not None
    assert second.hidden is not None
    assert first.hidden is not second.hidden


def test_a_draft_length_past_the_context_is_refused(monkeypatch):
    """The one ceiling this model has, and it is not an invented one.

    A candidate block of `1+K` positions cannot be verified past the context,
    so a draft length that needs more is refused with the length that fits.
    Nothing upstream refuses it instead: `MAX_SPEC_LEN` is asserted inside
    vLLM's `RejectionSampler`, which the TT path never calls, and
    `SpeculativeConfig` checks only that the draft length is positive.
    """
    from types import SimpleNamespace

    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=64))

    plan = DummySpecDecodeModel.spec_plan(config, max_num_seqs=8, requested_k=63)
    assert plan.effective_k == 63

    reject = DummySpecDecodeModel.spec_plan(config, max_num_seqs=8, requested_k=64)
    assert reject.supported_k == (63,)
    assert "max_model_len" in reject.reason


def test_a_draft_length_of_zero_is_refused():
    """Speculating nothing is a configuration error, not a quiet plain decode."""
    from types import SimpleNamespace

    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=2048))

    reject = DummySpecDecodeModel.spec_plan(config, max_num_seqs=8, requested_k=0)

    assert "speculates nothing" in reject.reason

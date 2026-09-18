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


def _fixed_model(monkeypatch, accept_depth=None):
    monkeypatch.setenv("TT_SPEC_TARGET", "fixed")
    return _model(monkeypatch, accept_depth=accept_depth)


def test_the_fixed_target_ignores_what_was_drafted(monkeypatch):
    """The property the `depth` target cannot have.

    Two blocks that differ only in their draft columns must produce the same
    answer at every column that is not one of the differing ones, because the
    rule reads the token and its position and nothing else. A target that
    echoed the drafts would answer differently at every column.
    """
    model = _fixed_model(monkeypatch)
    positions = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)
    first = torch.tensor([[100, 11, 12, 13]], dtype=torch.int32)
    second = torch.tensor([[100, 99, 98, 97]], dtype=torch.int32)

    a = model._fixed_choice(first, positions)
    b = model._fixed_choice(second, positions)

    # Column 0 holds the same committed token in both, so the same answer.
    assert int(a[0, 0]) == int(b[0, 0])
    # The rest differ because their inputs differ, and neither equals its input.
    assert a[0, 1:].tolist() != b[0, 1:].tolist()
    for column in range(1, 4):
        assert int(a[0, column]) != int(first[0, column])


def test_the_fixed_target_answers_a_plain_decode_by_the_same_rule(monkeypatch):
    """An unspeculated arm has to follow the rule, or it emits token 0 forever.

    The base class answers a decode with zero logits, so this mode overrides
    the plain call as well. The argmax of what it returns is what the ordinary
    host-sampling tail commits.
    """
    model = _fixed_model(monkeypatch)
    tokens = torch.tensor([[100], [200]], dtype=torch.int32)
    positions = torch.tensor([7, 9], dtype=torch.int32)

    logits = model.decode_forward(tokens=tokens, start_pos=positions)

    assert logits.shape == (2, 1, VOCAB)
    argmax = logits.argmax(dim=2).reshape(2)
    expected = model._fixed_choice(tokens, positions.reshape(2, 1)).reshape(2)
    assert argmax.tolist() == expected.tolist()


def test_the_fixed_target_s_drafter_proposes_what_the_verify_will_choose(monkeypatch):
    """Acceptance still happens, because the drafter walks the same rule.

    The drafter is where wrongness lives in this mode: it follows the rule for
    the first `accept_depth` drafts and bends the rest, so a partial-acceptance
    run stays reachable without the target ever looking at a draft.
    """
    model = _fixed_model(monkeypatch, accept_depth=2)
    model._verify_hidden = object()
    committed = torch.tensor([[100, 0, 0, 0]], dtype=torch.int32)
    positions = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    drafts = model.propose_draft_tokens(3, committed, positions, counts, hidden=model._verify_hidden).draft_token_ids

    # Walk the rule by hand from the row's last committed token at position 5.
    token, position = 100, 5
    truth = []
    for _ in range(3):
        token = (token * 31 + position * 7 + 11) % VOCAB
        position += 1
        truth.append(token)

    assert drafts[0, 0].item() == truth[0]
    assert drafts[0, 1].item() == truth[1]
    # Past the accept depth the drafter offers something the verify will refuse.
    assert drafts[0, 2].item() != truth[2]


def test_the_fixed_target_accepts_its_own_drafts_up_to_the_depth(monkeypatch):
    """The two halves agree, which is what makes the depth knob still work."""
    model = _fixed_model(monkeypatch, accept_depth=2)
    model._verify_hidden = object()
    committed = torch.tensor([[100, 0, 0, 0]], dtype=torch.int32)
    positions = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    drafts = model.propose_draft_tokens(3, committed, positions, counts, hidden=model._verify_hidden).draft_token_ids
    # The next verify's block: the row's last committed token, then the drafts.
    block = torch.cat([committed[:, :1], drafts], dim=1)
    verified = model._fixed_choice(block, positions)

    # Columns 0 and 1 agree with the drafts offered there, column 2 does not.
    assert int(verified[0, 0]) == int(drafts[0, 0])
    assert int(verified[0, 1]) == int(drafts[0, 1])
    assert int(verified[0, 2]) != int(drafts[0, 2])


def test_an_unknown_target_is_refused(monkeypatch, expect_error):
    """A typo in the mode must not silently select the default."""
    monkeypatch.setenv("TT_SPEC_TARGET", "fxied")

    with expect_error(ValueError, "TT_SPEC_TARGET"):
        DummySpecDecodeModel(mesh_device=None, max_batch_size=8, vocab_size=VOCAB)


def test_a_verify_answer_does_not_depend_on_the_step_before_it(monkeypatch):
    """The checkable half of `model_capabilities['supports_async_spec_decode']`.

    That declaration says this model can be driven by the plugin's deferred
    speculative path, where the readback and the commit of a step run after the
    step's submission has returned. Its buffer-lifetime half is vacuous for
    this model, which answers with host tensors and a host hidden object. Its
    remaining content is that the model keeps nothing between steps: every
    answer is a function of the step's own `tokens`, `start_pos` and
    `accepted_counts`, so running a different step first cannot change it. A
    model that selected a candidate state slot from whatever it ran last would
    answer differently here.
    """
    tokens = _block(2)
    positions = torch.tensor([[4, 5, 6, 7], [9, 10, 11, 12]], dtype=torch.int32)
    num_valid = torch.tensor([3, 2], dtype=torch.int32)
    counts = torch.tensor([2, 1], dtype=torch.int32)

    def verify(model, **overrides):
        call = {
            "tokens": tokens,
            "start_pos": positions,
            "num_valid_drafts": num_valid,
            "accepted_counts": counts,
            "spec_mode": "argmax_ids",
        }
        call.update(overrides)
        return model.decode_forward(**call).argmax_ids

    alone = verify(_model(monkeypatch))

    # A different step first: other tokens, other positions, other counts.
    model = _model(monkeypatch)
    verify(
        model,
        tokens=_block(2) + 7,
        start_pos=positions + 32,
        accepted_counts=torch.tensor([1, 3], dtype=torch.int32),
    )
    after_another_step = verify(model)

    assert after_another_step.tolist() == alone.tolist()


def test_the_fixed_target_s_drafter_does_not_depend_on_the_step_before_it(monkeypatch):
    """The same property for the proposal half, on the losslessness target.

    On the deferred path the proposal runs at the next step's drain, after the
    readback of the verify whose hidden handle it is given, so it must continue
    from the committed block it is handed rather than from anything it kept.
    """
    committed = torch.tensor([[100, 0, 0, 0]], dtype=torch.int32)
    positions = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    def propose(model):
        return model.propose_draft_tokens(3, committed, positions, counts, hidden=model._verify_hidden).draft_token_ids

    alone = _fixed_model(monkeypatch)
    alone._verify_hidden = object()
    expected = propose(alone)

    model = _fixed_model(monkeypatch)
    model._verify_hidden = object()
    model.propose_draft_tokens(
        3,
        committed + 5,
        positions + 16,
        counts,
        hidden=model._verify_hidden,
    )
    after_another_step = propose(model)

    assert after_another_step.tolist() == expected.tolist()


def _solo_model(monkeypatch, accept_depth=None):
    monkeypatch.setenv("TT_SPEC_DRAFT_POLICY", "solo")
    return _model(monkeypatch, accept_depth=accept_depth)


def test_the_solo_policy_offers_the_full_draft_length_to_a_lone_request(monkeypatch):
    """One live request, so speculation is worth trying: offer everything."""
    model = _solo_model(monkeypatch)
    committed = torch.tensor([[100, 0, 0, 0]], dtype=torch.int32)
    positions = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    out = model.propose_draft_tokens(3, committed, positions, counts, hidden=None)

    assert out.num_valid.tolist() == [3]
    assert out.draft_token_ids.shape == (1, 3)


def test_the_solo_policy_offers_nothing_once_a_peer_is_live(monkeypatch):
    """Two live requests, so the batch is doing the work: offer nothing.

    `num_valid` at 0 is how that is said. Returning fewer ids, or ids the
    runner is meant to recognise as empty, would both be unreadable: the ids
    are `[B, K]` whatever happens, because a device graph has one shape.
    """
    model = _solo_model(monkeypatch)
    committed = torch.tensor([[100, 0, 0, 0], [200, 0, 0, 0]], dtype=torch.int32)
    positions = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=torch.int32)
    counts = torch.tensor([1, 1], dtype=torch.int32)

    out = model.propose_draft_tokens(3, committed, positions, counts, hidden=None)

    assert out.num_valid.tolist() == [0, 0]
    assert out.draft_token_ids.shape == (2, 3)


def test_the_solo_policy_counts_live_requests_and_not_padded_rows(monkeypatch):
    """A padding row is not a peer.

    The rows reaching the drafter are padded to the wire batch size, so a lone
    request arrives in a block of eight rows on a server launched for eight. A
    policy counting rows would see a full batch and never offer anything. What
    separates them is the position: the runner leaves a padding row's committed
    positions negative for exactly this.
    """
    model = _solo_model(monkeypatch)
    rows = 8
    committed = torch.zeros((rows, 4), dtype=torch.int32)
    committed[0, 0] = 100
    positions = torch.full((rows, 4), -1, dtype=torch.int32)
    positions[0] = torch.tensor([5, 6, 7, 8], dtype=torch.int32)
    counts = torch.ones(rows, dtype=torch.int32)

    out = model.propose_draft_tokens(3, committed, positions, counts, hidden=None)

    assert out.num_valid.tolist() == [3] * rows


def test_the_always_policy_says_nothing_about_how_much_it_offers(monkeypatch):
    """`None` means every row is offered the full length.

    The policy that measures a fixed committed width always drafts, so it has
    nothing to declare, and a drafter written before `num_valid` existed keeps
    working for the same reason.
    """
    model = _model(monkeypatch)
    committed = torch.tensor([[100, 0, 0, 0]], dtype=torch.int32)
    positions = torch.tensor([[5, 6, 7, 8]], dtype=torch.int32)
    counts = torch.tensor([1], dtype=torch.int32)

    out = model.propose_draft_tokens(3, committed, positions, counts, hidden=model._verify_hidden)

    assert out.num_valid is None


def test_the_solo_policy_declares_what_the_plugin_must_know(monkeypatch):
    """Two declarations move with the policy, and both have to.

    The plugin reads these off the class at configuration time. It sends this
    model ordinary decode steps only if the plan says it serves them, and it
    keeps a drafter that requires a fed hidden state off those steps, because
    they produce no hidden handle. A policy asking to be drafted for on those
    steps must therefore declare the first and not the second.
    """
    from types import SimpleNamespace

    config = SimpleNamespace(model_config=SimpleNamespace(max_model_len=2048))

    monkeypatch.setenv("TT_SPEC_DRAFT_POLICY", "solo")
    import importlib

    import models.vllm_test_utils.spec_test.test_model as module

    importlib.reload(module)
    assert module.DummySpecDecodeModel.spec_plan(config, 8, 5).supports_narrow_decode
    assert "hidden_feed" not in module.DummySpecDecodeModel.model_capabilities["spec_requirements"]
    assert module.DummySpecDecodeModel.model_capabilities["spec_hidden_handoff"] == []

    monkeypatch.delenv("TT_SPEC_DRAFT_POLICY")
    importlib.reload(module)
    assert not module.DummySpecDecodeModel.spec_plan(config, 8, 5).supports_narrow_decode
    assert "hidden_feed" in module.DummySpecDecodeModel.model_capabilities["spec_requirements"]
    assert module.DummySpecDecodeModel.model_capabilities["spec_hidden_handoff"] == ["roundtrip"]


def test_an_unknown_draft_policy_is_refused(monkeypatch, expect_error):
    """A typo in the policy must not silently select the default."""
    monkeypatch.setenv("TT_SPEC_DRAFT_POLICY", "sollo")

    with expect_error(ValueError, "TT_SPEC_DRAFT_POLICY"):
        DummySpecDecodeModel(mesh_device=None, max_batch_size=8, vocab_size=VOCAB)


# The resident forward inputs behind `supports_async_decode`. A step submitted
# before the previous step's token came back cannot have been handed that token
# on the host, so the model holds it; these tests drive the commands that say
# when the host's copy is authoritative and when it is stale.


def _decode(model, tokens, positions, **commands):
    call = {
        "tokens": tokens,
        "start_pos": positions,
        "page_table": commands.pop("page_table", None),
        "reload_inputs": True,
        "reload_page_table": False,
        "reload_sampling_params": False,
        "reset_sampling_state": False,
    }
    call.update(commands)
    return model.decode_forward(**call)


def _ids(answer, rows):
    return answer.reshape(rows, 1, -1).argmax(dim=-1).reshape(rows).tolist()


def test_a_reloading_step_decodes_from_the_host_inputs(monkeypatch):
    """`reload_inputs` makes the host's token and position authoritative."""
    model = _fixed_model(monkeypatch)
    tokens = torch.tensor([[100]], dtype=torch.int32)
    positions = torch.tensor([5], dtype=torch.int32)

    answer = _decode(model, tokens, positions)

    assert _ids(answer, 1) == [int(model._fixed_choice(tokens, positions.reshape(1, 1))[0, 0])]


def test_a_resident_step_ignores_the_host_inputs(monkeypatch):
    """The contract's host-input authority rule, which is the whole point.

    With `reload_inputs` false the host's copies are stale by design: the token
    this step decodes from is the one the previous step chose, which the host
    had not seen when this step was submitted. A model reading the host's
    tokens there would decode from a token one step behind and nothing would
    report it.
    """
    model = _fixed_model(monkeypatch)
    first = _decode(model, torch.tensor([[100]], dtype=torch.int32), torch.tensor([5], dtype=torch.int32))
    chosen = _ids(first, 1)[0]

    # Deliberately wrong host inputs: a model honoring the command never reads
    # them, and one that reads them answers for token 999 at position 0.
    answer = _decode(
        model,
        torch.tensor([[999]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        reload_inputs=False,
    )

    expected = int(model._fixed_choice(torch.tensor([[chosen]]), torch.tensor([[6]]))[0, 0])
    assert _ids(answer, 1) == [expected]


def test_each_forward_advances_the_resident_position_once(monkeypatch):
    """Once per forward, and not once per readback.

    The position is what the next step decodes at, so a double advance skips a
    position and a missing one repeats it, and neither shows up in a single
    step's output.
    """
    model = _fixed_model(monkeypatch)
    _decode(model, torch.tensor([[100]], dtype=torch.int32), torch.tensor([5], dtype=torch.int32))
    assert model._position_advances == 1
    assert int(model._resident_positions[0]) == 6

    out = _decode(
        model,
        torch.tensor([[0]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        reload_inputs=False,
    )
    model.read_decode_output(out, async_read=True)
    model.read_decode_output(out, async_read=True)

    assert model._position_advances == 2
    assert int(model._resident_positions[0]) == 7


def test_the_readback_is_split_from_the_submission(monkeypatch):
    """What `supports_async_decode` asks for first, and all it asks here.

    The readback hands over what the forward produced, samples nothing and
    advances nothing. It returns no completion events: there is no device to
    signal one, and the plugin treats an empty list as already complete, which
    is the truthful description of a forward that ran on the host.
    """
    model = _fixed_model(monkeypatch)
    submitted = _decode(
        model,
        torch.tensor([[100]], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        read_from_device=False,
    )
    advances = model._position_advances

    read, events = model.read_decode_output(submitted, async_read=True)

    assert events == []
    assert torch.equal(read, submitted)
    assert model._position_advances == advances


def test_a_page_table_only_refresh_leaves_the_tokens_resident(monkeypatch):
    """The middle mode: new page tables, same resident token and position."""
    model = _fixed_model(monkeypatch)
    _decode(
        model,
        torch.tensor([[100]], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        page_table=torch.tensor([[0]], dtype=torch.int32),
    )
    resident = model._resident_tokens.clone()

    _decode(
        model,
        torch.tensor([[999]], dtype=torch.int32),
        torch.tensor([0], dtype=torch.int32),
        reload_inputs=False,
        reload_page_table=True,
        page_table=torch.tensor([[7]], dtype=torch.int32),
    )

    assert int(model._resident_page_table[0, 0]) == 7
    # The token this step decoded from was the previous step's choice, not the
    # host's 999, and the page-table refresh did not disturb it.
    assert not torch.equal(model._resident_tokens, torch.tensor([[999]], dtype=torch.int32))
    assert not torch.equal(model._resident_tokens, resident)


def test_reloading_everything_and_the_page_table_alone_is_refused(monkeypatch, expect_error):
    """Commands, not hints: an illegal combination raises rather than guessing."""
    model = _fixed_model(monkeypatch)

    with expect_error(ValueError, "not a legal combination"):
        _decode(
            model,
            torch.tensor([[100]], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            reload_page_table=True,
        )


def test_resetting_sampling_state_without_reloading_inputs_is_refused(monkeypatch, expect_error):
    """A sampler aligns its seed counters from authoritative host positions."""
    model = _fixed_model(monkeypatch)

    with expect_error(ValueError, "only authoritative"):
        _decode(
            model,
            torch.tensor([[100]], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            reload_inputs=False,
            reset_sampling_state=True,
        )


def test_decoding_from_resident_inputs_before_any_reload_is_refused(monkeypatch, expect_error):
    """The first decode of a chain reloads; nothing else can be resident yet."""
    model = _fixed_model(monkeypatch)

    with expect_error(ValueError, "before any step reloaded them"):
        _decode(
            model,
            torch.tensor([[100]], dtype=torch.int32),
            torch.tensor([5], dtype=torch.int32),
            reload_inputs=False,
        )


def test_the_sampling_commands_are_recorded_when_they_arrive(monkeypatch):
    """Nothing to upload here, and the command still has to be honored."""
    model = _fixed_model(monkeypatch)

    _decode(
        model,
        torch.tensor([[100]], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        reload_sampling_params=True,
        reset_sampling_state=True,
    )

    assert model._sampling_param_uploads == 1
    assert model._sampling_state_resets == 1


# The `fixed` target's prefill, which follows the same rule as its decode. A
# resumed request replays its history through a prefill, so a prefill outside
# the rule puts a token in the middle of a response that the rule cannot
# explain, and the output stops being a property of the rule.


def test_the_fixed_prefill_follows_the_same_rule_as_the_decode(monkeypatch):
    """The token a prefill chooses is the rule applied to the prompt tail."""
    model = _fixed_model(monkeypatch)
    prompt = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)

    logits = model.prefill_forward(
        tokens=prompt, prompt_lens=torch.tensor([4]), start_pos=torch.tensor([0])
    )

    chosen = int(logits.reshape(1, -1).argmax(dim=-1))
    assert chosen == int(model._fixed_choice(torch.tensor([[14]]), torch.tensor([[3]]))[0, 0])


def test_the_fixed_prefill_reads_each_row_s_own_length(monkeypatch):
    """Rows are padded to the longest prompt, and each has its own tail.

    A prefill that read a fixed column would answer for padding on a shorter
    row, which is what a resumed request looks like beside a fresh one.
    """
    model = _fixed_model(monkeypatch)
    tokens = torch.tensor([[11, 12, 13, 14], [21, 22, 0, 0]], dtype=torch.int32)

    logits = model.prefill_forward(
        tokens=tokens,
        prompt_lens=torch.tensor([4, 2]),
        start_pos=torch.tensor([0, 0]),
    )

    chosen = logits.reshape(2, -1).argmax(dim=-1)
    assert int(chosen[0]) == int(model._fixed_choice(torch.tensor([[14]]), torch.tensor([[3]]))[0, 0])
    assert int(chosen[1]) == int(model._fixed_choice(torch.tensor([[22]]), torch.tensor([[1]]))[0, 0])


def test_the_fixed_prefill_answers_ids_when_the_device_samples(monkeypatch):
    """Device sampling asks for the chosen id rather than logits."""
    model = _fixed_model(monkeypatch)
    prompt = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)

    ids = model.prefill_forward(
        tokens=prompt,
        prompt_lens=torch.tensor([4]),
        start_pos=torch.tensor([0]),
        sampling_params=object(),
    )

    assert ids.tolist() == [
        int(model._fixed_choice(torch.tensor([[14]]), torch.tensor([[3]]))[0, 0])
    ]


def test_the_depth_target_prefill_is_untouched(monkeypatch):
    """The measurement target keeps the base class's answer.

    Its whole output is a function of what was drafted, and the acceptance
    accounting the device suite does with it counts from a prefill that commits
    token 0. Changing that would move every expectation built on it.
    """
    model = _model(monkeypatch)
    prompt = torch.tensor([[11, 12, 13, 14]], dtype=torch.int32)

    logits = model.prefill_forward(
        tokens=prompt, prompt_lens=torch.tensor([4]), start_pos=torch.tensor([0])
    )

    assert int(logits.reshape(1, -1).argmax(dim=-1)) == 0

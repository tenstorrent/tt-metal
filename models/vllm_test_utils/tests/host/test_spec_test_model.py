# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PyTorch-only tests for the speculative test model's verify arithmetic.

`DummySpecDecodeModel._verified_ids` decides what the model claims at each
candidate position, and that answer is what the plugin's accept walk turns into
a committed prefix. Its arithmetic is pure PyTorch: no TTNN, no device, and no
`vllm_tt_plugin`, so it can be tested directly here.

Worth testing rather than assuming, because two properties of it are easy to
get wrong in a way that nothing downstream reports. A bonus written to a column
fixed for the batch instead of to each row's own draft count silently commits
the wrong token on a row that carries fewer drafts than its neighbour. And an
accept depth applied across the batch instead of per row lets one row's short
draft list shorten another's speculation.

The layout under test: column `j` is the choice draft `j` has to match, for `j`
below the row's count, and the column at the row's count carries the bonus that
follows a fully accepted row.
"""

import torch

from models.vllm_test_utils.spec_test.test_model import DummySpecDecodeModel

VOCAB = 1024


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

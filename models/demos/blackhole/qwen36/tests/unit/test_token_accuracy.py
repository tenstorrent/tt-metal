# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for the teacher-forcing scorer behind ``text_demo.py -k accuracy_512``.

The scoring itself is index arithmetic (which top-5 row predicts which decoded position), and a
wrong offset would not raise — it would just report a wrong accuracy after a full device run.
"""
import pytest
import torch

from models.demos.blackhole.qwen36.demo.text_demo import _TokenAccuracy


def _reference(total_length=8):
    """Reference in the committed ``.refpt`` shape: row i is the top-5 prediction of token i+1."""
    reference_tokens = torch.arange(10, 10 + total_length)
    top5_tokens = torch.stack([torch.arange(t, t + 5) for t in reference_tokens + 1])
    return reference_tokens, top5_tokens


def test_splits_reference_in_half():
    acc = _TokenAccuracy(*_reference(total_length=8))
    assert acc.prompt_tokens.tolist() == [[10, 11, 12, 13]]
    assert acc.reference_tokens.tolist() == [14, 15, 16, 17]
    # Row 3 (= split - 1) predicts token 4 (= split), the first decoded position.
    assert acc.top5_tokens[0, 0].item() == 14


def test_feed_returns_reference_tokens_in_order():
    acc = _TokenAccuracy(*_reference(total_length=8))
    # Whatever the model predicted, the next input is the reference token for that position.
    assert [acc.feed(p) for p in (99, 99, 99, 99)] == [14, 15, 16, 17]
    assert acc.predicted == [99, 99, 99, 99]


def test_feed_clamps_past_the_last_scored_position():
    acc = _TokenAccuracy(*_reference(total_length=8))
    for _ in range(4):
        acc.feed(0)
    # Decoding beyond the reference keeps feeding its last token instead of indexing out of range.
    assert acc.feed(0) == 17


def test_compute_scores_top1_and_top5():
    reference_tokens, top5_tokens = _reference(total_length=8)
    acc = _TokenAccuracy(reference_tokens, top5_tokens)
    # Positions: exact top-1 / inside top-5 / outside top-5 / exact top-1.
    for predicted in (14, 17, 0, 17):
        acc.feed(predicted)
    assert acc.compute() == (50.0, pytest.approx(75.0))


def test_compute_uses_only_scored_positions():
    acc = _TokenAccuracy(*_reference(total_length=8))
    acc.feed(14)
    # A partial run is scored over what it produced, not over the full reference.
    assert acc.compute() == (100.0, 100.0)

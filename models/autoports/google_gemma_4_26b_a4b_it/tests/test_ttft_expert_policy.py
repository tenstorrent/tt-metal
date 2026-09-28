# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only bounds and row-order contract for short-prefill expert batching."""

from unittest.mock import Mock

import pytest
import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _ExpertParallelExperts


@pytest.mark.parametrize(
    "rows,expected_chunks",
    [
        (1, [1]),
        (32, [32]),
        (64, [32, 32]),
        (96, [32, 32, 32]),
        (128, [128]),
        (160, [128, 32]),
        (192, [128, 64]),
        (224, [128, 96]),
        (256, [128, 128]),
        (288, [32] * 9),
        (1024, [32] * 32),
    ],
)
def test_short_prefill_policy_preserves_rows_and_bounds(monkeypatch, rows, expected_chunks):
    experts = _ExpertParallelExperts.__new__(_ExpertParallelExperts)
    experts.prefill_batch_tokens = 32
    experts.short_prefill_batch_tokens = 128
    x = torch.arange(rows, dtype=torch.float32).reshape(1, 1, rows, 1)
    routes = x + 1000
    observed = []

    def chunk(values, routing, decode):
        observed.append((values.shape[-2], decode))
        assert torch.equal(routing, values + 1000), "Expert inputs and routing rows became misaligned"
        return values + 7

    experts._chunk = chunk
    concat = Mock(side_effect=lambda values, dim: torch.cat(values, dim=dim))
    monkeypatch.setattr(ttnn, "concat", concat)
    output = experts(x, routes)
    assert observed == [(count, rows == 1) for count in expected_chunks]
    assert output.shape == x.shape
    assert torch.equal(output, x + 7), "Chunk assembly lost or reordered rows"
    if len(expected_chunks) == 1:
        concat.assert_not_called()
    else:
        concat.assert_called_once()
        assert concat.call_args.kwargs == {"dim": 2}
    assert experts.prefill_batch_tokens == 32
    assert experts.short_prefill_batch_tokens == 128

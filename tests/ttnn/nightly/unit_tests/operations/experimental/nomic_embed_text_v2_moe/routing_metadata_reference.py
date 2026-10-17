# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Torch oracle for ttnn.experimental.nomic_embed_text_v2_moe.routing_metadata.

Shared by the op's tests and the nomic model's router and expert tests.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

import ttnn

BLOCK_ROWS = 32


class RoutingMetadata(NamedTuple):
    """The op's four outputs on the host: ids as int64, weights as float32, flattened to 1D or 2D."""

    gather_index: torch.Tensor  # (gathered_rows,)
    combine_index: torch.Tensor  # (top_k, padded_tokens)
    combine_weights: torch.Tensor  # (top_k, padded_tokens)
    expert_counts: torch.Tensor  # (num_experts,)


def routing_reference(
    indices: torch.Tensor,
    values: torch.Tensor,
    num_experts: int,
    token_start: int = 0,
    num_tokens: int | None = None,
) -> RoutingMetadata:
    """The op's outputs computed slot by slot, in the order the kernel visits the slots.

    Args:
        indices: (tokens, top_k) integer expert ids of the whole batch.
        values: (tokens, top_k) float32 weights.
        num_experts, token_start, num_tokens: as the op takes them.
    """
    total_tokens, top_k = indices.shape
    if num_tokens is None:
        num_tokens = total_tokens - token_start
    padded_tokens = math.ceil(num_tokens / BLOCK_ROWS) * BLOCK_ROWS
    num_blocks = math.ceil(top_k * num_tokens / BLOCK_ROWS) + num_experts - 1
    window = indices[token_start : token_start + num_tokens].long().clamp(max=num_experts - 1).tolist()

    counts = [0] * num_experts
    for choices in window:
        for expert in choices:
            counts[expert] += 1

    # Each expert's region starts on a block: the exclusive scan of the counts rounded up to 32.
    next_row = []
    region_start = 0
    for count in counts:
        next_row.append(region_start)
        region_start += math.ceil(count / BLOCK_ROWS) * BLOCK_ROWS

    gather_index = [token_start] * (num_blocks * BLOCK_ROWS)
    combine_index = [[0] * padded_tokens for _ in range(top_k)]
    for token, choices in enumerate(window):
        for k, expert in enumerate(choices):
            row = next_row[expert]
            next_row[expert] += 1
            gather_index[row] = token_start + token
            combine_index[k][token] = row

    combine_weights = torch.zeros(top_k, padded_tokens, dtype=torch.float32)
    combine_weights[:, :num_tokens] = values[token_start : token_start + num_tokens].T
    return RoutingMetadata(
        torch.tensor(gather_index), torch.tensor(combine_index), combine_weights, torch.tensor(counts)
    )


def region_starts(expert_counts: torch.Tensor) -> torch.Tensor:
    """Each expert's first gathered row: the exclusive scan of the counts, each rounded up to a block."""
    region_rows = (expert_counts + BLOCK_ROWS - 1) // BLOCK_ROWS * BLOCK_ROWS
    return torch.cumsum(region_rows, 0) - region_rows


def block_experts(expert_counts: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """(num_blocks,) the expert each block of gathered rows runs through, derived from the counts alone.

    A block belongs to the last expert whose region starts at or before it. An empty region starts
    where the next one does, so it owns no block, and the blocks past the last region, which hold no
    slot, take the last expert.
    """
    first_blocks = region_starts(expert_counts) // BLOCK_ROWS
    return (first_blocks.unsqueeze(0) <= torch.arange(num_blocks).unsqueeze(1)).sum(1) - 1


def to_host(outputs) -> RoutingMetadata:
    """The op's four device tensors as a RoutingMetadata."""
    gather_index, combine_index, combine_weights, expert_counts = outputs
    top_k, padded_tokens = combine_weights.shape[1], combine_weights.shape[2]
    return RoutingMetadata(
        ttnn.to_torch(gather_index).long().reshape(-1),
        ttnn.to_torch(combine_index).long(),
        ttnn.to_torch(combine_weights).reshape(top_k, padded_tokens),
        ttnn.to_torch(expert_counts).long().reshape(-1),
    )


def assert_matches_reference(
    outputs,
    indices: torch.Tensor,
    values: torch.Tensor,
    num_experts: int,
    token_start: int = 0,
    num_tokens: int | None = None,
) -> RoutingMetadata:
    """Every output equal to the oracle, the weights bit for bit, and the contract checked on its own.

    The contract checks do not use the oracle: each real slot's row gathers its own token, lies in
    its expert's region and belongs to no other slot.
    """
    if num_tokens is None:
        num_tokens = indices.shape[0] - token_start
    expected = routing_reference(indices, values, num_experts, token_start, num_tokens)
    got = to_host(outputs)

    for name in ("gather_index", "combine_index", "expert_counts"):
        expected_tensor, got_tensor = getattr(expected, name), getattr(got, name)
        assert (
            got_tensor.shape == expected_tensor.shape
        ), f"{name}: shape {got_tensor.shape}, expected {expected_tensor.shape}"
        mismatches = int((got_tensor != expected_tensor).sum())
        assert mismatches == 0, f"{name}: {mismatches} of {expected_tensor.numel()} entries differ"
    assert got.combine_weights.shape == expected.combine_weights.shape
    assert torch.equal(
        got.combine_weights.view(torch.int32), expected.combine_weights.view(torch.int32)
    ), "combine_weights: not bit-identical to the input values"

    window = indices[token_start : token_start + num_tokens].long().clamp(max=num_experts - 1)
    region_start = region_starts(got.expert_counts)
    rows = got.combine_index[:, :num_tokens].T
    tokens = torch.arange(token_start, token_start + num_tokens).unsqueeze(1).expand_as(rows)
    assert torch.equal(got.gather_index[rows], tokens), "a slot's row does not gather its own token"
    first_row = region_start[window]
    assert bool(
        ((rows >= first_row) & (rows < first_row + got.expert_counts[window])).all()
    ), "a slot's row lies outside its expert's region"
    assert rows.unique().numel() == rows.numel(), "two slots share a row"
    return got

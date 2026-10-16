# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""One-round, explicitly nonqualifying Qwen3.8 MTP mechanics exercise.

This is the narrow handoff from ordinary decode to the already implemented
shifted-seed proposer and fixed-five target verifier.  It deliberately reuses
the live hybrid session rather than constructing another model, cache, or
runtime owner.  A successful return proves only that draft, verify, prefix
acceptance, recurrent-state alignment, and cache publication executed; it is
not a numerical qualification result.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from models.demos.blackhole.qwen38_flash_next.ttnn.hybrid_decode import (
    Qwen38HybridCachePolicy,
    Qwen38HybridMode,
    Qwen38HybridSessionStatus,
    Qwen38HybridStep,
    Qwen38TTNNHybridDecodeSession,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.mtp_decode import (
    DRAFT_EXTENSION_CALLS,
    VERIFY_POSITIONS,
    Qwen38SpeculativeRound,
)
from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38SamplingParameters


def execute_one_mtp_round(
    session: Qwen38TTNNHybridDecodeSession,
    input_ids: torch.Tensor | Sequence[int],
    *,
    cache_policy: Qwen38HybridCachePolicy | str = Qwen38HybridCachePolicy.MATCH,
) -> tuple[Qwen38HybridStep, Qwen38HybridStep]:
    """Emit one ordinary bootstrap token, then execute one real MTP round.

    The output budget is exactly one ordinary token plus the five-token
    verifier window.  This prevents the hybrid session's normal short-budget
    fallback from silently replacing the requested MTP mechanics exercise.
    Existing session ownership and cache rules remain authoritative.
    """

    if type(session) is not Qwen38TTNNHybridDecodeSession:
        raise TypeError("MTP mechanics requires the exact live Qwen38 TTNN hybrid session")

    bootstrap = session.begin(
        input_ids,
        max_new_tokens=VERIFY_POSITIONS + 1,
        mode=Qwen38HybridMode.MTP,
        cache_policy=cache_policy,
        sampling=Qwen38SamplingParameters.greedy(),
    )
    if (
        bootstrap.requested_mode is not Qwen38HybridMode.MTP
        or bootstrap.executed_mode is not Qwen38HybridMode.ORDINARY
        or len(bootstrap.tokens) != 1
        or bootstrap.committed_input_count <= 0
    ):
        raise RuntimeError("MTP mechanics bootstrap did not execute one ordinary target emission")
    if session.status is not Qwen38HybridSessionStatus.ACTIVE:
        raise RuntimeError("MTP mechanics bootstrap stopped before fixed-five verification")

    speculative = session.step()
    round_result = speculative.speculative_round
    if (
        speculative.requested_mode is not Qwen38HybridMode.MTP
        or speculative.executed_mode is not Qwen38HybridMode.MTP
        or type(round_result) is not Qwen38SpeculativeRound
        or speculative.fallback_reason is not None
    ):
        raise RuntimeError("MTP mechanics request fell back instead of executing fixed-five verification")
    if (
        round_result.mtp_extension_count != DRAFT_EXTENSION_CALLS
        or speculative.timing.mtp_extension_calls != DRAFT_EXTENSION_CALLS
        or speculative.timing.target_rows != VERIFY_POSITIONS
        or round_result.mtp_alignment_count != round_result.committed_input_count
        or speculative.timing.mtp_alignment_rows != round_result.committed_input_count
    ):
        raise RuntimeError("MTP draft, verify, or authoritative alignment counts differ from the exact contract")

    committed_inputs = round_result.input_token_ids[: round_result.committed_input_count]
    consumed = session.consumed_token_ids
    emitted = tuple(token.token_id for token in speculative.tokens)
    if (
        round_result.input_token_ids[0] != bootstrap.tokens[-1].token_id
        or consumed[-len(committed_inputs) :] != committed_inputs
        or emitted != tuple(item.token_id for item in round_result.emissions)
        or session.pending_token_id != round_result.pending_token_id
        or session.state_position != round_result.next_position
        or speculative.next_position != round_result.next_position
    ):
        raise RuntimeError("MTP accepted-prefix publication did not preserve target/MTP cache alignment")
    return bootstrap, speculative


__all__ = ["execute_one_mtp_round"]

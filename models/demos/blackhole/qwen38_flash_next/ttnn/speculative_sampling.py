# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact speculative sampling over the MTP verify rows: argmax drafts, point-mass acceptance.

Row j's target distribution ``p_j`` accepts the draft ``d_{j+1}`` iff ``u_j < p_j(d_{j+1})``. The first rejection
emits ``x ~ p_j`` with ``d_{j+1}`` removed and renormalised; full acceptance emits ``x ~ p_k``. The emitted stream
therefore has plain sampling's distribution. ``uniform`` is consumed in the order ``u_0, u_1, ..., v``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38RowDistribution


@dataclass(frozen=True)
class Qwen38SpeculativeAcceptance:
    """One pass's verdict: accepted draft count, emitted token, per-row acceptance probabilities and draws used."""

    accepted: int
    token: int
    acceptance_probabilities: tuple[float, ...]
    draws: int
    resampled: bool

    def __post_init__(self) -> None:
        drawn = len(self.acceptance_probabilities)
        if self.accepted < 0 or drawn not in (self.accepted, self.accepted + 1):
            raise ValueError(f"{self.accepted} accepted drafts with {drawn} acceptance probabilities")
        if self.draws != (self.accepted + 2 if self.resampled else self.accepted + 1):
            raise ValueError(f"{self.draws} draws for {self.accepted} accepted drafts (resampled {self.resampled})")


def accept_point_mass(
    distribution: Callable[[int], Qwen38RowDistribution],
    drafts: Sequence[int],
    uniform: Callable[[], float],
) -> Qwen38SpeculativeAcceptance:
    """Point-mass acceptance over a pass's rows; ``distribution(j)`` is built lazily, up to the first rejection."""

    drafts = [int(draft) for draft in drafts]
    if not drafts:
        raise ValueError("a pass proposes at least one draft")
    probabilities: list[float] = []
    for row, draft in enumerate(drafts):
        target = distribution(row)
        probability = target.probability(draft)
        probabilities.append(probability)
        if float(uniform()) < probability:
            continue
        token = target.without(draft).draw(float(uniform()))
        return Qwen38SpeculativeAcceptance(row, token, tuple(probabilities), row + 2, True)
    token = distribution(len(drafts)).draw(float(uniform()))
    return Qwen38SpeculativeAcceptance(len(drafts), token, tuple(probabilities), len(drafts) + 1, False)


__all__ = ["Qwen38SpeculativeAcceptance", "accept_point_mass"]

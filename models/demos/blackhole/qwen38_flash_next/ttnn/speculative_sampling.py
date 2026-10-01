# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Exact speculative sampling over the MTP verify rows: argmax drafts, point-mass acceptance.

Verify row j of a pass (row 0 fed the pending token ``t_P``, row j the draft ``d_j``) gives the target's
post-processor distribution ``p_j`` at that position, a :class:`Qwen38RowDistribution` built by the same penalties
and filters plain sampling applies (``ttnn/sampling.py``); the draft the pass proposes there is ``d_{j+1}``.  With
the deterministic proposal ``q_j = delta(d_{j+1})`` the Leviathan / Chen acceptance ``min(1, p / q)`` is
``p_j(d_{j+1})`` and the rejection distribution ``norm(max(0, p - q))`` is ``p_j`` with ``d_{j+1}`` removed and
renormalised.  So for j = 0, 1, ...: draw ``u_j`` and accept the draft iff ``u_j < p_j(d_{j+1})``; at the first
rejection draw ``v`` and emit ``x ~ p_j'``; when every draft is accepted draw ``v`` and emit ``x ~ p_k``.

Per row ``P(out = y) = p(d) 1[y = d] + (1 - p(d)) p'(y) 1[y != d] = p(y)``: the emitted token follows the target's
conditional at its position whatever the draft, the accepted drafts are the prefix the later rows were computed
on, so the stream's joint law is plain sampling's; the draft quality changes only the accepted length.  A kept set
of one token (``p(d) = 1``) always accepts (``u < 1``), so no residual is ever formed from an empty remainder.

``uniform`` is the request's draw source, consumed in the fixed order ``u_0, u_1, ..., v``: ``a* + 2`` draws when
``a* < k``, ``k + 1`` when every draft was accepted.  Nothing here touches a device.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from models.demos.blackhole.qwen38_flash_next.ttnn.sampling import Qwen38RowDistribution


@dataclass(frozen=True)
class Qwen38SpeculativeAcceptance:
    """One pass's verdict: ``accepted`` drafts (``a*``), the ``token`` emitted after them (``x*``), the acceptance
    probabilities ``p_j(d_{j+1})`` of the rows drawn on (j = 0 .. min(a*, k - 1)), the draws consumed, and whether
    the token came from a rejection's residual."""

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
    """The point-mass acceptance over the pass's rows: ``distribution(j)`` is ``p_j`` (asked for row by row, so a
    row past the first rejection is never built), ``drafts`` ``[d_1 .. d_k]``, ``uniform`` the next draw."""

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

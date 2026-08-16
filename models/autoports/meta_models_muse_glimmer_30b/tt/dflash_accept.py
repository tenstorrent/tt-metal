# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The DFlash accept/reject rule, isolated from any device code.

This is the part of speculative decoding that determines *correctness*, so it is
kept pure and unit-tested rather than buried in the generator loop.

The rule, from ``transformers.generation.utils._assisted_decoding``::

    selected_tokens = new_logits.argmax(dim=-1)          # target's own argmax
    n_matches = ((~(candidates == selected_tokens[:, :-1])).cumsum(-1) < 1).sum()
    valid_tokens = selected_tokens[:, : n_matches + 1]

Two properties matter and are easy to lose in a re-implementation:

**Emitted tokens always come from the target, never from the drafter.**  Even at
positions where the draft was accepted, the token appended is
``selected_tokens[i]``, not ``candidates[i]``.  They are equal there by
definition of a match, so this looks like a distinction without a difference -
until a position ties differently, at which point taking the candidate would
silently make the output depend on the drafter.  It must not.

**One correction token is always emitted.**  ``n_matches + 1`` tokens are
committed: the accepted prefix plus the target's own token at the first
mismatch.  That token is free - the verify forward already computed it - and it
is what guarantees forward progress even when every draft is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class AcceptResult:
    """Outcome of verifying one drafted block."""

    #: Tokens to commit, all taken from the target's argmax.
    tokens: tuple[int, ...]
    #: How many drafted tokens matched, i.e. the speculation win for this block.
    n_matches: int
    #: How many drafted tokens were proposed.
    n_candidates: int

    @property
    def n_committed(self) -> int:
        return len(self.tokens)

    @property
    def acceptance_ratio(self) -> float:
        return self.n_matches / self.n_candidates if self.n_candidates else 0.0


def accept_block(
    candidate_ids: Sequence[int],
    target_argmax: Sequence[int],
    *,
    eos_token_ids: Sequence[int] = (),
    max_new_tokens: int | None = None,
) -> AcceptResult:
    """Apply the acceptance rule to one drafted block.

    Args:
        candidate_ids: the ``block_size - 1`` tokens the drafter proposed.
        target_argmax: the target's argmax at each verify position.  Must be one
            longer than ``candidate_ids`` - the extra entry is the bonus token
            predicted from the position after the last candidate.
        eos_token_ids: committing stops at the first EOS, since tokens after it
            would be generated from a sequence that never existed.
        max_new_tokens: hard cap on committed tokens.

    Returns:
        An :class:`AcceptResult`.  ``tokens`` is never empty: even a fully
        rejected block commits the target's correction token, which is what
        makes the loop guaranteed to terminate.
    """
    if len(target_argmax) != len(candidate_ids) + 1:
        raise ValueError(
            f"target_argmax has {len(target_argmax)} entries; expected len(candidate_ids) + 1 = {len(candidate_ids) + 1}"
        )

    n_matches = 0
    for candidate, selected in zip(candidate_ids, target_argmax):
        if candidate != selected:
            break
        n_matches += 1

    tokens = list(target_argmax[: n_matches + 1])

    if eos_token_ids:
        eos = set(int(t) for t in eos_token_ids)
        for position, token in enumerate(tokens):
            if int(token) in eos:
                tokens = tokens[: position + 1]
                break

    if max_new_tokens is not None:
        tokens = tokens[:max_new_tokens]

    return AcceptResult(
        tokens=tuple(int(t) for t in tokens),
        n_matches=min(n_matches, max(len(tokens) - 1, 0)),
        n_candidates=len(candidate_ids),
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Opt-in cross-step reuse for the denoise loop. Off unless one of the two variables is set; eager path only.

Both knobs trade output quality for time (the reused quantities are one step stale); nothing here is bit-identical to
the plain schedule, so they stay off by default and are meant for measurement first.

MINIMAX_H3_STEP_SKIP
    Forwards on which the whole block stack is skipped: the stack's delta (hidden_out - hidden_in) from the last
    computed forward is added to the new hidden state instead. Spec: comma-separated ranges ``first-last/N`` (skip the
    forwards in [first, last] whose offset from ``first`` is not a multiple of N) or single forward indices.
    ``4-42/2`` skips forwards 5, 7, ..., 41. The first forward is never skipped.

MINIMAX_H3_ATTN_CACHE
    ``a-b:N[:first-last]``: blocks a..b reuse their cached attention-branch delta (x_after_attn - x_before) on the
    forwards in [first, last] whose offset from ``first`` is not a multiple of N; the FFN and the residual still run.
    The window defaults to forwards 4 .. total-6 (the first and last forwards move the sample most).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _parse_range(text: str) -> tuple[int, int]:
    lo, _, hi = text.partition("-")
    lo_i = int(lo)
    hi_i = int(hi) if hi else lo_i
    if hi_i < lo_i:
        raise ValueError(f"empty range {text!r}")
    return lo_i, hi_i


def parse_step_mask(spec: str | None, total: int) -> frozenset[int]:
    """Forward indices selected by a MINIMAX_H3_STEP_SKIP spec, clipped to [1, total)."""
    if not spec:
        return frozenset()
    chosen: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        rng, _, stride = part.partition("/")
        lo, hi = _parse_range(rng)
        n = int(stride) if stride else 1
        if n < 1:
            raise ValueError(f"stride must be >= 1 in {part!r}")
        for i in range(lo, hi + 1):
            if n == 1 or (i - lo) % n != 0:
                chosen.add(i)
    return frozenset(i for i in chosen if 1 <= i < total)


def parse_attn_spec(spec: str | None, total: int) -> tuple[frozenset[int], frozenset[int]]:
    """(blocks, forwards) selected by a MINIMAX_H3_ATTN_CACHE spec ``a-b:N[:first-last]``."""
    if not spec:
        return frozenset(), frozenset()
    fields = spec.split(":")
    if len(fields) not in (2, 3):
        raise ValueError(f"MINIMAX_H3_ATTN_CACHE={spec!r}: expected 'a-b:N[:first-last]'")
    a, b = _parse_range(fields[0])
    n = int(fields[1])
    if n < 2:
        raise ValueError("MINIMAX_H3_ATTN_CACHE: N must be >= 2 (N=1 would never compute)")
    first, last = _parse_range(fields[2]) if len(fields) == 3 else (4, total - 6)
    forwards = frozenset(i for i in range(first, last + 1) if (i - first) % n != 0 and 1 <= i < total)
    return frozenset(range(a, b + 1)), forwards


@dataclass(frozen=True)
class StepReusePlan:
    """Which forwards reuse what; `kwargs(i)` is what the transformer's forward takes for forward `i`."""

    skip_forwards: frozenset[int] = field(default_factory=frozenset)
    attn_blocks: frozenset[int] = field(default_factory=frozenset)
    attn_forwards: frozenset[int] = field(default_factory=frozenset)

    @classmethod
    def from_env(cls, total_forwards: int) -> "StepReusePlan":
        skip = parse_step_mask(os.environ.get("MINIMAX_H3_STEP_SKIP"), total_forwards)
        blocks, forwards = parse_attn_spec(os.environ.get("MINIMAX_H3_ATTN_CACHE"), total_forwards)
        return cls(skip, blocks, forwards)

    @property
    def active(self) -> bool:
        return bool(self.skip_forwards or (self.attn_blocks and self.attn_forwards))

    def kwargs(self, i: int) -> dict:
        out: dict = {}
        if i in self.skip_forwards:
            out["reuse_stack"] = True
        elif self.attn_blocks and i in self.attn_forwards:
            out["reuse_attn_blocks"] = self.attn_blocks
        return out

    def describe(self, total_forwards: int) -> str:
        parts = []
        if self.skip_forwards:
            parts.append(f"skip {len(self.skip_forwards)}/{total_forwards} forwards {sorted(self.skip_forwards)}")
        if self.attn_blocks and self.attn_forwards:
            parts.append(
                f"reuse attention of blocks {min(self.attn_blocks)}-{max(self.attn_blocks)} on "
                f"{len(self.attn_forwards)}/{total_forwards} forwards {sorted(self.attn_forwards)}"
            )
        return "; ".join(parts) if parts else "off"

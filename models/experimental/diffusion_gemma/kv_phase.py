# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Reference/spec KV-cache phase mapping helpers for DiffusionGemma.

This module documents the logical positions and bounded-sliding slot math used
by the device tests. Runtime Gemma4 cache updates use the shared
``models.demos.gemma4.tt.attention`` phase enum plus paged page-table/modulo
plumbing, not this helper directly.
"""

from __future__ import annotations

from dataclasses import dataclass


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


@dataclass(frozen=True)
class KVPhaseMapping:
    """Reference logical-to-physical KV positions for one denoise/commit canvas."""

    prompt_len: int
    canvas_len: int = 256
    sliding_window: int = 1024

    def __post_init__(self) -> None:
        if self.prompt_len < 0:
            raise ValueError(f"prompt_len must be >= 0, got {self.prompt_len}")
        _validate_positive("canvas_len", self.canvas_len)
        _validate_positive("sliding_window", self.sliding_window)

    @property
    def commit_positions(self) -> tuple[int, ...]:
        """Absolute positions written by commit-append."""

        return tuple(range(self.prompt_len, self.prompt_len + self.canvas_len))

    @property
    def canvas_scratch_positions(self) -> tuple[int, ...]:
        """Local positions inside the per-step denoise scratch zone."""

        return tuple(range(self.canvas_len))

    @property
    def full_attention_frozen_positions(self) -> tuple[int, ...]:
        """Full-attention frozen cache positions visible before commit."""

        return tuple(range(self.prompt_len))

    @property
    def sliding_frozen_positions(self) -> tuple[int, ...]:
        """Absolute positions retained in a bounded sliding cache before commit."""

        start = max(0, self.prompt_len - self.sliding_window)
        return tuple(range(start, self.prompt_len))

    def sliding_slots(self, positions: tuple[int, ...]) -> tuple[int, ...]:
        """Map absolute positions to bounded-sliding physical cache slots."""

        return tuple(position % self.sliding_window for position in positions)

    @property
    def sliding_frozen_slots(self) -> tuple[int, ...]:
        return self.sliding_slots(self.sliding_frozen_positions)

    @property
    def sliding_commit_slots(self) -> tuple[int, ...]:
        return self.sliding_slots(self.commit_positions)

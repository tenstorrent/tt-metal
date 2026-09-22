# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chronological segment topology for KDA prefill over MLA's block-cyclic layout.

MLA places a ``G = P * C`` token prefill chunk starting at absolute position ``S``
across ``P`` sequence-parallel chips so that chip ``c`` holds ``C`` rows, but the
chronological order of those rows is rotated by ``S``. Exactly one chip -- the
boundary chip -- can hold two causally non-adjacent pieces: the chronologically
first rows and the chronologically last rows.

This independent host oracle is test-only. Production derives chronology on the
device through the shared native helper; tests retain this formulation to check
causal ordering independently of device implementation.

Ground truth for the placement is MLA's own position oracle,
``tt/mla/utils.py::rotated_chip_positions``, which mirrors the KV writer kernel.
On the boundary chip the head occupies rows ``0:h`` and the tail rows ``h:C``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChronologicalTopology:
    """Chronological segment topology for one prefill chunk.

    ``first_rank`` owns the head segment (rows ``0:head_rows``) and, when
    ``tail_rows`` is nonzero, also the tail segment (rows ``head_rows:C``). Every
    other chip owns one full segment of ``C`` rows.
    """

    sp_size: int
    local_rows: int
    first_rank: int
    head_rows: int

    @property
    def tail_rows(self) -> int:
        return self.local_rows - self.head_rows

    @property
    def is_split(self) -> bool:
        """Whether the boundary chip holds two causally non-adjacent segments.

        With a single partition the wrapped tail immediately follows the head in
        absolute position, so the rows stay contiguous and nothing is split.
        Non-adjacency needs at least one other chip between the two pieces.
        """
        return self.tail_rows > 0 and self.sp_size > 1

    @property
    def chip_order(self) -> tuple[int, ...]:
        """Physical chips in chronological order of their first-owned segment."""
        return tuple((self.first_rank + step) % self.sp_size for step in range(self.sp_size))

    def predecessor_chip(self, chip: int) -> int:
        """Physical chip immediately preceding ``chip`` in chronological order."""
        return (chip - 1) % self.sp_size


def chronological_topology(actual_start: int, sp_size: int, local_rows: int) -> ChronologicalTopology:
    """Derive topology from layer-validated inputs for ``actual_start`` on an SP ring.

    ``actual_start`` is the absolute global position of the chunk's first token.
    Only ``actual_start`` modulo the global chunk size affects the topology, so
    the result is bounded and safe as a program-cache key.
    """
    cycle_offset = actual_start % (sp_size * local_rows)
    tail_rows = cycle_offset % local_rows
    head_rows = local_rows - tail_rows
    return ChronologicalTopology(
        sp_size=sp_size,
        local_rows=local_rows,
        first_rank=(cycle_offset // local_rows) % sp_size,
        head_rows=head_rows,
    )

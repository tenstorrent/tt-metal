# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Chronological segment topology for KDA prefill over MLA's block-cyclic layout.

MLA places a ``G = P * C`` token prefill chunk starting at absolute position ``S``
across ``P`` sequence-parallel chips so that chip ``c`` holds ``C`` rows, but the
chronological order of those rows is rotated by ``S``. Exactly one chip -- the
boundary chip -- can hold two causally non-adjacent pieces: the chronologically
first rows and the chronologically last rows.

This module is the single source of that ordering. Both order-sensitive KDA
stages (convolution carry routing and recurrent affine-prefix composition) must
derive their order from here rather than from physical rank, and must not
re-derive it independently.

Ground truth for the placement is MLA's own position oracle,
``tt/mla/utils.py::rotated_chip_positions``, which mirrors the KV writer kernel.
On the boundary chip the head occupies rows ``0:h`` and the tail rows ``h:C``.
"""

from __future__ import annotations

from dataclasses import dataclass

from models.demos.deepseek_v3_d_p.tt.kda.config import KDA_CHUNK_SIZE


@dataclass(frozen=True)
class OffsetTopology:
    """Chronological segment topology for one prefill chunk.

    ``boundary_chip`` owns the head segment (rows ``0:head_rows``) and, when
    ``tail_rows`` is nonzero, also the tail segment (rows ``head_rows:C``). Every
    other chip owns one full segment of ``C`` rows.
    """

    sp_size: int
    local_rows: int
    boundary_chip: int
    head_rows: int
    tail_rows: int

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
        return tuple((self.boundary_chip + step) % self.sp_size for step in range(self.sp_size))

    def predecessor_chip(self, chip: int) -> int:
        """Physical chip immediately preceding ``chip`` in chronological order."""
        return (chip - 1) % self.sp_size


def offset_topology(actual_start: int, sp_size: int, local_rows: int) -> OffsetTopology:
    """Derive the chronological topology for ``actual_start`` on an SP ring.

    ``actual_start`` is the absolute global position of the chunk's first token.
    Only ``actual_start`` modulo the global chunk size affects the topology, so
    the result is bounded and safe as a program-cache key.
    """
    if sp_size <= 0:
        raise ValueError(f"sp_size must be positive, got {sp_size}")
    if local_rows <= 0:
        raise ValueError(f"local_rows must be positive, got {local_rows}")
    if actual_start < 0:
        raise ValueError(f"actual_start must be non-negative, got {actual_start}")
    if actual_start % KDA_CHUNK_SIZE:
        raise ValueError(f"actual_start must be a multiple of {KDA_CHUNK_SIZE}, got {actual_start}")

    start = actual_start % (sp_size * local_rows)
    tail_rows = start % local_rows
    head_rows = local_rows - tail_rows
    # Only a split constrains the geometry: each segment must hold whole KDA
    # chunks, so the boundary chip's two pieces must both be chunk-aligned.
    if tail_rows and (head_rows % KDA_CHUNK_SIZE or tail_rows % KDA_CHUNK_SIZE):
        raise ValueError(
            f"actual_start {actual_start} splits chip {(start // local_rows) % sp_size} into "
            f"{head_rows}+{tail_rows} rows, which are not both multiples of {KDA_CHUNK_SIZE}"
        )
    return OffsetTopology(
        sp_size=sp_size,
        local_rows=local_rows,
        boundary_chip=(start // local_rows) % sp_size,
        head_rows=head_rows,
        tail_rows=tail_rows,
    )


def segment_owners(topology: OffsetTopology) -> tuple[tuple[int, int, int], ...]:
    """Return the chronological segments as ``(chip, row_start, row_end)`` triples.

    The first segment consumes the caller's entry state and the last segment
    produces the replacement state. When the topology is not split there are
    ``sp_size`` segments; otherwise there are ``sp_size + 1`` and the boundary
    chip appears first and last.
    """
    rows = topology.local_rows
    order = topology.chip_order
    if not topology.is_split:
        return tuple((chip, 0, rows) for chip in order)
    head = (order[0], 0, topology.head_rows)
    middle = tuple((chip, 0, rows) for chip in order[1:])
    tail = (order[0], topology.head_rows, rows)
    return (head, *middle, tail)


# Fragment indices for the uniform split every chip performs in the segment-aware
# prototype: 0 is ``rows[0:head_rows]``, 1 is ``rows[head_rows:local_rows]``.
HEAD_FRAGMENT = 0
TAIL_FRAGMENT = 1


def fragment_order(topology: OffsetTopology) -> tuple[tuple[int, int], ...]:
    """Chronological ``(chip, fragment)`` order when every chip splits at ``head_rows``.

    Splitting all chips keeps mesh shapes uniform even though only the boundary
    chip's two pieces are causally non-adjacent. For every other chip the two
    fragments are adjacent, so composing them in sequence is equivalent to one
    whole-chip segment.

    The boundary chip's head opens the stream and its tail closes it, giving
    ``2 * sp_size`` fragments.
    """
    if not topology.is_split:
        raise ValueError("fragment order is only defined for a split topology")
    order = topology.chip_order
    middle = [(chip, fragment) for chip in order[1:] for fragment in (HEAD_FRAGMENT, TAIL_FRAGMENT)]
    return ((order[0], HEAD_FRAGMENT), *middle, (order[0], TAIL_FRAGMENT))

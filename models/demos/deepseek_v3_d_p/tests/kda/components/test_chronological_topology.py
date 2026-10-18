# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests for KDA's chronological offset topology.

The topology is checked against MLA's own position oracle rather than against a
re-derivation of the same equations, so a divergence between KDA's notion of
order and MLA's actual row placement fails here.
"""

import pytest

from models.demos.deepseek_v3_d_p.tests.kda.chronology_oracle import ChronologicalTopology, chronological_topology
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions

SP_SIZE = 8
LOCAL_ROWS = 640
GLOBAL_ROWS = SP_SIZE * LOCAL_ROWS
TILE = 32
ALL_STARTS = tuple(range(0, GLOBAL_ROWS, TILE))


def segment_owners(topology: ChronologicalTopology) -> tuple[tuple[int, int, int], ...]:
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


def _chronological_positions(topology_start: int) -> list[int]:
    """Absolute positions in chronological order, as KDA's topology claims them."""
    topology = chronological_topology(topology_start, SP_SIZE, LOCAL_ROWS)
    oracle = rotated_chip_positions(topology_start, SP_SIZE, LOCAL_ROWS)
    ordered: list[int] = []
    for chip, row_start, row_end in segment_owners(topology):
        ordered.extend(oracle[chip][row_start:row_end])
    return ordered


@pytest.mark.parametrize("start", ALL_STARTS)
def test_topology_orders_every_position_exactly_once(start):
    """Segments tile the requested interval in strictly increasing order."""
    ordered = _chronological_positions(start)
    assert ordered == list(range(start, start + GLOBAL_ROWS))


@pytest.mark.parametrize("start", ALL_STARTS)
def test_first_rank_owns_the_first_and_last_positions(start):
    """The boundary chip is where the wrap lands, and it holds both ends when split."""
    topology = chronological_topology(start, SP_SIZE, LOCAL_ROWS)
    oracle = rotated_chip_positions(start, SP_SIZE, LOCAL_ROWS)
    segments = segment_owners(topology)

    assert oracle[topology.first_rank][0] == start
    assert segments[0][0] == topology.first_rank
    if topology.is_split:
        assert len(segments) == SP_SIZE + 1
        assert segments[-1][0] == topology.first_rank
        assert oracle[topology.first_rank][topology.head_rows] == start + GLOBAL_ROWS - topology.tail_rows
    else:
        assert len(segments) == SP_SIZE
        assert len({chip for chip, _, _ in segments}) == SP_SIZE


@pytest.mark.parametrize("start", ALL_STARTS)
def test_segment_rows_partition_each_chip(start):
    """Every chip's C rows are covered exactly once across all segments."""
    topology = chronological_topology(start, SP_SIZE, LOCAL_ROWS)
    covered: dict[int, list[int]] = {chip: [] for chip in range(SP_SIZE)}
    for chip, row_start, row_end in segment_owners(topology):
        assert 0 <= row_start < row_end <= LOCAL_ROWS
        assert row_start % TILE == 0 and row_end % TILE == 0
        covered[chip].extend(range(row_start, row_end))
    for chip in range(SP_SIZE):
        assert sorted(covered[chip]) == list(range(LOCAL_ROWS))


def test_every_first_rank_is_reachable():
    """All eight chips occur as the boundary chip across the tile-aligned starts."""
    boundaries = {chronological_topology(start, SP_SIZE, LOCAL_ROWS).first_rank for start in ALL_STARTS}
    assert boundaries == set(range(SP_SIZE))


def test_device_boundary_offsets_are_unsplit():
    """Offsets that land between chips rotate order without creating a tail."""
    for chip in range(SP_SIZE):
        topology = chronological_topology(chip * LOCAL_ROWS, SP_SIZE, LOCAL_ROWS)
        assert not topology.is_split
        assert topology.first_rank == chip
        assert topology.head_rows == LOCAL_ROWS
        assert topology.chip_order == tuple((chip + step) % SP_SIZE for step in range(SP_SIZE))


def test_zero_offset_preserves_physical_rank_order():
    """Offset zero must be indistinguishable from the pre-offset behavior."""
    topology = chronological_topology(0, SP_SIZE, LOCAL_ROWS)
    assert not topology.is_split
    assert topology.chip_order == tuple(range(SP_SIZE))
    assert segment_owners(topology) == tuple((chip, 0, LOCAL_ROWS) for chip in range(SP_SIZE))


def test_topology_depends_only_on_start_modulo_global_chunk():
    """Absolute position must not create an unbounded program-cache key."""
    for start in (0, 32, 960, 4480):
        assert chronological_topology(start, SP_SIZE, LOCAL_ROWS) == chronological_topology(
            start + 5 * GLOBAL_ROWS, SP_SIZE, LOCAL_ROWS
        )


def test_unsplit_topologies_do_not_require_chunk_aligned_local_rows():
    """Rank rotation alone is valid for any local row count, as no segment splits.

    Component-level callers exercise small synthetic partitions; only a split
    imposes the chunk-alignment requirement.
    """
    topology = chronological_topology(32, SP_SIZE, 8)
    assert not topology.is_split
    assert topology.first_rank == 4


@pytest.mark.parametrize("start", [32, 960, GLOBAL_ROWS - 32])
def test_single_partition_is_never_split(start):
    """One partition keeps rows contiguous, so no offset splits it.

    The wrapped tail immediately follows the head in absolute position, which
    MLA's oracle confirms: the single chip carries the whole interval in order.
    """
    topology = chronological_topology(start, 1, GLOBAL_ROWS)
    assert not topology.is_split
    assert rotated_chip_positions(start, 1, GLOBAL_ROWS)[0] == list(range(start, start + GLOBAL_ROWS))

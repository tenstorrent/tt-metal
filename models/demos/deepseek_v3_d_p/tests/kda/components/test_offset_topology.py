# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host tests for KDA's chronological offset topology.

The topology is checked against MLA's own position oracle rather than against a
re-derivation of the same equations, so a divergence between KDA's notion of
order and MLA's actual row placement fails here.
"""

import pytest

from models.demos.deepseek_v3_d_p.tt.kda.offset import offset_topology, segment_owners
from models.demos.deepseek_v3_d_p.tt.mla.utils import rotated_chip_positions

SP_SIZE = 8
LOCAL_ROWS = 640
GLOBAL_ROWS = SP_SIZE * LOCAL_ROWS
TILE = 32
ALL_STARTS = tuple(range(0, GLOBAL_ROWS, TILE))


def _chronological_positions(topology_start: int) -> list[int]:
    """Absolute positions in chronological order, as KDA's topology claims them."""
    topology = offset_topology(topology_start, SP_SIZE, LOCAL_ROWS)
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
def test_boundary_chip_owns_the_first_and_last_positions(start):
    """The boundary chip is where the wrap lands, and it holds both ends when split."""
    topology = offset_topology(start, SP_SIZE, LOCAL_ROWS)
    oracle = rotated_chip_positions(start, SP_SIZE, LOCAL_ROWS)
    segments = segment_owners(topology)

    assert oracle[topology.boundary_chip][0] == start
    assert segments[0][0] == topology.boundary_chip
    if topology.is_split:
        assert len(segments) == SP_SIZE + 1
        assert segments[-1][0] == topology.boundary_chip
        assert oracle[topology.boundary_chip][topology.head_rows] == start + GLOBAL_ROWS - topology.tail_rows
    else:
        assert len(segments) == SP_SIZE
        assert len({chip for chip, _, _ in segments}) == SP_SIZE


@pytest.mark.parametrize("start", ALL_STARTS)
def test_segment_rows_partition_each_chip(start):
    """Every chip's C rows are covered exactly once across all segments."""
    topology = offset_topology(start, SP_SIZE, LOCAL_ROWS)
    covered: dict[int, list[int]] = {chip: [] for chip in range(SP_SIZE)}
    for chip, row_start, row_end in segment_owners(topology):
        assert 0 <= row_start < row_end <= LOCAL_ROWS
        assert row_start % TILE == 0 and row_end % TILE == 0
        covered[chip].extend(range(row_start, row_end))
    for chip in range(SP_SIZE):
        assert sorted(covered[chip]) == list(range(LOCAL_ROWS))


def test_every_boundary_chip_is_reachable():
    """All eight chips occur as the boundary chip across the tile-aligned starts."""
    boundaries = {offset_topology(start, SP_SIZE, LOCAL_ROWS).boundary_chip for start in ALL_STARTS}
    assert boundaries == set(range(SP_SIZE))


def test_device_boundary_offsets_are_unsplit():
    """Offsets that land between chips rotate order without creating a tail."""
    for chip in range(SP_SIZE):
        topology = offset_topology(chip * LOCAL_ROWS, SP_SIZE, LOCAL_ROWS)
        assert not topology.is_split
        assert topology.boundary_chip == chip
        assert topology.head_rows == LOCAL_ROWS
        assert topology.chip_order == tuple((chip + step) % SP_SIZE for step in range(SP_SIZE))


def test_zero_offset_preserves_physical_rank_order():
    """Offset zero must be indistinguishable from the pre-offset behavior."""
    topology = offset_topology(0, SP_SIZE, LOCAL_ROWS)
    assert not topology.is_split
    assert topology.chip_order == tuple(range(SP_SIZE))
    assert segment_owners(topology) == tuple((chip, 0, LOCAL_ROWS) for chip in range(SP_SIZE))


def test_topology_depends_only_on_start_modulo_global_chunk():
    """Absolute position must not create an unbounded program-cache key."""
    for start in (0, 32, 960, 4480):
        assert offset_topology(start, SP_SIZE, LOCAL_ROWS) == offset_topology(
            start + 5 * GLOBAL_ROWS, SP_SIZE, LOCAL_ROWS
        )


@pytest.mark.parametrize(
    "bad_start,message",
    [(-32, "non-negative"), (16, "multiple of"), (1000, "multiple of")],
)
def test_misaligned_or_negative_starts_are_rejected(bad_start, message, expect_error):
    with expect_error(ValueError, message):
        offset_topology(bad_start, SP_SIZE, LOCAL_ROWS)


def test_unsplit_topologies_do_not_require_chunk_aligned_local_rows():
    """Rank rotation alone is valid for any local row count, as no segment splits.

    Component-level callers exercise small synthetic partitions; only a split
    imposes the chunk-alignment requirement.
    """
    topology = offset_topology(32, SP_SIZE, 8)
    assert not topology.is_split
    assert topology.boundary_chip == 4


def test_split_that_would_straddle_a_chunk_is_rejected(expect_error):
    """A split must leave whole KDA chunks on both sides of the boundary."""
    with expect_error(ValueError, "not both multiples"):
        offset_topology(32, SP_SIZE, 48)

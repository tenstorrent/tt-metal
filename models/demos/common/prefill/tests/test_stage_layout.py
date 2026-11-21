# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Rank -> stage mapping, host row ownership and D2D boundary pairing for stages that span two hosts.
ttnn-free."""

import pytest

from models.demos.common.prefill.runners.stage_layout import (
    BoundaryPair,
    StageLayout,
    chip_host_tag,
    collapse_to_stages,
    single_peer,
)

SC4 = StageLayout((1, 2, 1, 2, 1, 2, 1, 2))
TSR = StageLayout((1, 2, 1))


def test_single_host_layout_is_rank_equals_stage():
    layout = StageLayout.single_host(4)
    assert [layout.stage_of(r) for r in range(4)] == [0, 1, 2, 3]
    assert all(layout.is_leader(r) and layout.host_rank_of(r) == 0 for r in range(4))
    assert layout.is_single_host and layout.leader_ranks() == [0, 1, 2, 3]


def test_unset_env_gives_one_host_per_stage():
    assert StageLayout.from_env(3, {}) == StageLayout.single_host(3)


def test_env_describes_ranks_mesh_major_host_minor():
    layout = StageLayout.from_env(4, {"PREFILL_STAGE_HOSTS": "1,2,1"})
    assert [(layout.stage_of(r), layout.host_rank_of(r)) for r in range(4)] == [(0, 0), (1, 0), (1, 1), (2, 0)]
    assert layout.leader_ranks() == [0, 1, 3]
    assert list(layout.ranks_of(1)) == [1, 2]


def test_sc4_carve_is_eight_stages_over_twelve_ranks():
    assert (SC4.num_stages, SC4.num_ranks) == (8, 12)
    assert SC4.leader_ranks() == [0, 1, 3, 4, 6, 7, 9, 10]
    assert [SC4.stage_of(r) for r in range(12)] == [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7]


def test_env_rank_count_must_match_the_run(expect_error):
    with expect_error(ValueError, "4 ranks but the run has 3"):
        StageLayout.from_env(3, {"PREFILL_STAGE_HOSTS": "1,2,1"})


@pytest.mark.parametrize("hosts", [(), (1, 0), (0,)])
def test_every_stage_needs_a_host(hosts, expect_error):
    with expect_error(ValueError, "at least one host"):
        StageLayout(hosts)


def test_rank_out_of_range_is_rejected(expect_error):
    with expect_error(ValueError, "outside"):
        TSR.stage_of(4)


def test_host_rows_split_axis_zero_into_blocks():
    assert [TSR.host_rows(r, 4) for r in range(4)] == [(0, 4), (0, 2), (2, 4), (0, 4)]


def test_rows_must_divide_over_hosts(expect_error):
    with expect_error(ValueError, "do not split"):
        TSR.host_rows(1, 3)


def test_equal_host_boundaries_pair_rank_to_rank_over_the_whole_mesh():
    layout = StageLayout((2, 2))
    assert layout.boundary_pairs(0, 4) == [BoundaryPair(0, 2, (0, 2)), BoundaryPair(1, 3, (2, 4))]


def test_torus_to_seam_boundary_splits_the_torus_rank_over_both_seam_hosts():
    assert TSR.boundary_pairs(0, 4) == [BoundaryPair(0, 1, (0, 2)), BoundaryPair(0, 2, (2, 4))]


def test_seam_to_torus_boundary_pairs_each_seam_host_with_the_torus_rank():
    assert TSR.boundary_pairs(1, 4) == [BoundaryPair(1, 3, (0, 2)), BoundaryPair(2, 3, (2, 4))]


def test_boundary_pairs_cover_every_row_exactly_once():
    for stage in range(SC4.num_stages - 1):
        rows = sorted(r for p in SC4.boundary_pairs(stage, 4) for r in range(*p.rows))
        assert rows == [0, 1, 2, 3]


def test_per_rank_views_of_a_boundary():
    assert TSR.inbound_pairs(0, 4) == []
    assert TSR.outbound_pairs(3, 4) == []
    assert [p.receiver_rank for p in TSR.outbound_pairs(0, 4)] == [1, 2]
    assert [p.sender_rank for p in TSR.inbound_pairs(3, 4)] == [1, 2]
    assert TSR.inbound_pairs(2, 4) == [BoundaryPair(0, 2, (2, 4))]
    assert TSR.outbound_pairs(2, 4) == [BoundaryPair(2, 3, (2, 4))]


def test_the_last_stage_has_no_outbound_boundary(expect_error):
    with expect_error(ValueError, "last stage"):
        TSR.boundary_pairs(2, 4)


def _gathered(ranks, cols=4, rows=4):
    return [
        [
            {
                "rank": r,
                "first_layer": 0,
                "host_tag": 100 + r,
                "fnids": [[(r, row, c) for c in range(cols)] for row in range(rows)],
            }
            for r in ranks
        ]
    ]


def test_gathered_entries_collapse_to_one_per_stage():
    collapsed = collapse_to_stages(_gathered(range(4)), TSR, 4)
    assert [e["rank"] for e in collapsed[0]] == [0, 1, 3]


def test_a_seam_stage_records_the_host_of_every_chip():
    seam = collapse_to_stages(_gathered(range(4)), TSR, 4)[0][1]
    assert [[chip_host_tag(seam, r, c) for c in range(4)] for r in range(4)] == [
        [101] * 4,
        [101] * 4,
        [102] * 4,
        [102] * 4,
    ]


def test_a_single_host_stage_has_one_host_for_every_chip():
    torus = collapse_to_stages(_gathered(range(4)), TSR, 4)[0][0]
    assert {chip_host_tag(torus, r, c) for r in range(4) for c in range(4)} == {100}


def test_an_entry_without_per_chip_hosts_uses_its_rank_host():
    assert chip_host_tag({"host_tag": 5}, 3, 3) == 5


def test_without_a_layout_every_entry_stays():
    gathered = _gathered(range(3))
    assert collapse_to_stages(gathered, None, 4) == gathered


def test_equal_host_stages_have_one_peer_per_side():
    layout = StageLayout((2, 2))
    assert single_peer([BoundaryPair(0, 1, (0, 4))], 4, "outbound", 0) == 1
    assert single_peer([BoundaryPair(0, 1, (0, 4))], 4, "inbound", 1) == 0
    assert single_peer([], 4, "inbound", 0) is None


@pytest.mark.parametrize("rank, direction", [(0, "outbound"), (3, "inbound")])
def test_a_boundary_split_over_two_peers_is_refused_until_ttnn_can_serve_it(rank, direction, expect_error):
    pairs = TSR.outbound_pairs(rank, 4) if direction == "outbound" else TSR.inbound_pairs(rank, 4)
    with expect_error(NotImplementedError, "one service per host row block"):
        single_peer(pairs, 4, direction, rank)


def test_a_seam_host_owning_half_the_rows_is_refused_too(expect_error):
    with expect_error(NotImplementedError, "one service per host row block"):
        single_peer(TSR.outbound_pairs(1, 4), 4, "outbound", 1)

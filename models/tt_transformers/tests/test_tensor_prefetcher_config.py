# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

from models.tt_transformers.tt.prefetcher_config import allocate_tensor_prefetcher_receiver_layout

_RING32_RECEIVERS = (
    (0, 0),
    (1, 0),
    (2, 0),
    (6, 0),
    (0, 1),
    (1, 1),
    (3, 1),
    (6, 1),
    (0, 2),
    (1, 2),
    (2, 2),
    (4, 2),
    (0, 4),
    (1, 4),
    (4, 4),
    (5, 4),
    (7, 5),
    (8, 5),
    (9, 5),
    (10, 5),
    (2, 6),
    (3, 6),
    (7, 6),
    (8, 6),
    (0, 7),
    (7, 7),
    (9, 7),
    (10, 7),
    (0, 8),
    (6, 8),
    (7, 8),
    (8, 8),
)

_RING8_RECEIVERS = ((1, 0), (0, 1), (0, 2), (0, 4), (8, 5), (7, 6), (7, 7), (6, 8))

_RING16_RECEIVERS = (
    (1, 0),
    (2, 0),
    (0, 1),
    (3, 1),
    (0, 2),
    (1, 2),
    (0, 4),
    (4, 4),
    (8, 5),
    (9, 5),
    (2, 6),
    (7, 6),
    (7, 7),
    (10, 7),
    (6, 8),
    (8, 8),
)


def _anchors(num_dram_banks, right_start):
    left_banks = num_dram_banks // 2
    return tuple(
        [(0, 9 - bank) for bank in range(left_banks)]
        + [(right_start, bank) for bank in range(num_dram_banks - left_banks)]
    )


@pytest.mark.parametrize("receivers_per_bank", [1, 2, 4, 8])
def test_tensor_prefetcher_receiver_profiles_are_complete(receivers_per_bank):
    layout = allocate_tensor_prefetcher_receiver_layout(
        (11, 10),
        _anchors(8, right_start=6),
        num_dram_banks=8,
        receivers_per_bank=receivers_per_bank,
    )

    assert layout is not None
    assert len(layout.receiver_coords) == 8 * receivers_per_bank
    assert len(set(layout.receiver_coords)) == 8 * receivers_per_bank
    assert layout.ring_cols == 8
    assert layout.ring_rows == receivers_per_bank
    assert not layout.used_fallback


def test_tensor_prefetcher_ring32_profile_matches_verified_placement():
    layout = allocate_tensor_prefetcher_receiver_layout(
        (11, 10),
        _anchors(8, right_start=6),
        num_dram_banks=8,
        receivers_per_bank=4,
    )

    assert layout is not None
    assert layout.receiver_coords == _RING32_RECEIVERS


def test_tensor_prefetcher_lower_count_profiles_are_optimized_nested_subsets():
    layouts = {
        receivers_per_bank: allocate_tensor_prefetcher_receiver_layout(
            (11, 10),
            _anchors(8, right_start=6),
            num_dram_banks=8,
            receivers_per_bank=receivers_per_bank,
        )
        for receivers_per_bank in (1, 2, 4, 8)
    }

    assert layouts[1].receiver_coords == _RING8_RECEIVERS
    assert layouts[2].receiver_coords == _RING16_RECEIVERS
    for lower_count, higher_count in ((1, 2), (2, 4), (4, 8)):
        for bank in range(8):
            lower = layouts[lower_count].receiver_coords[bank * lower_count : (bank + 1) * lower_count]
            higher = layouts[higher_count].receiver_coords[bank * higher_count : (bank + 1) * higher_count]
            assert set(lower) < set(higher)


@pytest.mark.parametrize(
    "grid_x,right_start,expected_side_widths",
    [
        (10, 6, (6, 4)),
        (10, 7, (7, 3)),
        (9, 5, (5, 4)),
        (9, 7, (7, 2)),
    ],
)
def test_tensor_prefetcher_receiver_layout_adapts_to_harvested_columns(grid_x, right_start, expected_side_widths):
    layout = allocate_tensor_prefetcher_receiver_layout(
        (grid_x, 10),
        _anchors(8, right_start),
        num_dram_banks=8,
        receivers_per_bank=4,
    )

    assert layout is not None
    assert layout.used_fallback
    assert (layout.left_columns, layout.right_columns) == expected_side_widths
    assert layout.ambiguous_columns == 0
    assert len(set(layout.receiver_coords)) == 32
    assert all(0 <= x < grid_x and 0 <= y < 10 for x, y in layout.receiver_coords)


def test_tensor_prefetcher_receiver_layout_uses_shared_regions_for_different_harvesting():
    layout = allocate_tensor_prefetcher_receiver_layout(
        (11, 10),
        _anchors(8, right_start=6),
        num_dram_banks=8,
        receivers_per_bank=4,
        device_right_starts=(6, 7),
    )

    assert layout is not None
    assert layout.used_fallback
    assert (layout.left_columns, layout.ambiguous_columns, layout.right_columns) == (6, 1, 4)
    assert len(set(layout.receiver_coords)) == 32
    assert all(0 <= x < 11 and 0 <= y < 10 for x, y in layout.receiver_coords)


def test_tensor_prefetcher_receiver_layout_uses_ambiguous_column_before_opposite_side():
    tensor_config = {
        "fallback_profile_dram_banks": 3,
        "ring_grid": {"columns": "dram_banks", "rows": "receivers_per_bank"},
        "fallback": {"spill_rows": [0], "side_order": ["preferred", "opposite"]},
        "profiles": {
            3: {
                "banks": [
                    [["left", 0, 0]],
                    [["right", 0, 0]],
                    [["right", 0, 0]],
                ],
                "selections": {
                    1: {
                        "name": "ambiguous_fallback",
                        "bank_slot_indices": [[0], [0], [0]],
                    }
                },
            }
        },
    }
    layout = allocate_tensor_prefetcher_receiver_layout(
        (3, 1),
        _anchors(3, right_start=1),
        num_dram_banks=3,
        receivers_per_bank=1,
        tensor_config=tensor_config,
        device_right_starts=(1, 2),
    )

    assert layout is not None
    assert layout.receiver_coords == ((0, 0), (2, 0), (1, 0))
    assert (layout.left_columns, layout.ambiguous_columns, layout.right_columns) == (1, 1, 1)


def test_tensor_prefetcher_receiver_layout_spills_when_a_row_is_too_narrow():
    layout = allocate_tensor_prefetcher_receiver_layout(
        (7, 10),
        _anchors(8, right_start=2),
        num_dram_banks=8,
        receivers_per_bank=8,
    )

    assert layout is not None
    assert layout.used_fallback
    assert len(set(layout.receiver_coords)) == 64
    assert len({y for _, y in layout.receiver_coords[:8]}) > 1


def test_tensor_prefetcher_receiver_layout_rejects_insufficient_worker_grid():
    layout = allocate_tensor_prefetcher_receiver_layout(
        (6, 10),
        _anchors(8, right_start=2),
        num_dram_banks=8,
        receivers_per_bank=8,
    )

    assert layout is None


def test_tensor_prefetcher_receiver_layout_remaps_dram_harvest_by_side():
    layout = allocate_tensor_prefetcher_receiver_layout(
        (10, 10),
        _anchors(7, right_start=6),
        num_dram_banks=7,
        receivers_per_bank=1,
    )

    assert layout is not None
    assert layout.used_fallback
    assert layout.receiver_coords == ((1, 0), (0, 1), (0, 2), (8, 5), (7, 6), (7, 7), (6, 8))
    assert layout.ring_cols == 7
    assert layout.ring_rows == 1


def test_tensor_prefetcher_receiver_layout_is_deterministic():
    args = ((9, 10), _anchors(8, right_start=5), 8, 8)
    first = allocate_tensor_prefetcher_receiver_layout(*args)
    second = allocate_tensor_prefetcher_receiver_layout(*args)

    assert first == second

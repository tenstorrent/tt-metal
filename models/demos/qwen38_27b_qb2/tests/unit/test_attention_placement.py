# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.qwen38_27b_qb2.tests.attention_placement import LAYOUTS, placement, useful_kv_bytes
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry


@pytest.mark.parametrize("batch", [4, 8, 16, 32])
@pytest.mark.parametrize("grid", [(11, 10), (12, 10)])
def test_core_sets_fit_blackhole_and_match_declared_work(batch, grid):
    for name in LAYOUTS:
        item = placement(name, batch, grid)
        points = [tuple(p) for p in item["logical_cores"]]
        assert len(points) == len(set(points)) == item["grid"][0] * item["grid"][1]
        assert all(0 <= x < grid[0] and 0 <= y < grid[1] for x, y in points)
        assert item["active_cores"] == batch * item["cores_per_user"] <= len(points)
        assert points == sorted(points, key=lambda p: (p[1], p[0]))
        assert len(points[:batch]) == batch
    for left, right in (("row_major_64", "flash_mla_64"), ("row_major_80", "outer_columns_80")):
        a, b = placement(left, batch, grid), placement(right, batch, grid)
        assert a["active_cores"] == b["active_cores"]
        assert a["output_layout"] == b["output_layout"]
        assert a["logical_cores"] != b["logical_cores"]


@pytest.mark.parametrize("grid", [(11, 10), (12, 10)])
def test_reference_and_outer_column_candidates_balance_both_sides(grid):
    for name, count in (("flash_mla_64", 32), ("outer_columns_80", 40)):
        points = placement(name, 8, grid)["logical_cores"]
        assert sum(x < 4 for x, _ in points) == count
        assert sum(x >= 7 for x, _ in points) == count


def test_native_grid_uses_all_reported_columns_and_correct_reader_count(expect_error):
    assert placement("native", 8, (11, 10))["active_cores"] == 104
    native = placement("native", 8, (12, 10))
    assert native["active_cores"] == 120
    assert native["cores_per_user"] == 15
    assert len(native["logical_cores"]) == 120
    assert ((512 // native["active_cores"]) * 1152) // 2048 == 2
    outer = placement("outer_columns_80", 8, (12, 10))
    assert {x for x, _ in outer["logical_cores"]} == {0, 1, 2, 3, 8, 9, 10, 11}
    with expect_error(ValueError, "Unsupported placement worker grid"):
        placement("native", 8, (8, 8))


def test_bandwidth_accounting_counts_each_users_causal_kv_once():
    assert useful_kv_bytes([31, 63]) == 96 * 544
    assert useful_kv_bytes([131071] * 8) == 570425344


def test_wider_attention_batch_preserves_independent_causal_limits():
    case = geometry(32768, 32)
    assert len(case["positions"]) == len(set(case["positions"])) == 32
    assert 0 <= min(case["positions"]) <= max(case["positions"]) < case["aligned_capacity"]
    assert case["pool_tokens"] == 32 * case["aligned_capacity"]

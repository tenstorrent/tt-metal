# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.qwen38_27b_qb2.tests.attention_placement import LAYOUTS, placement, useful_kv_bytes
from models.demos.qwen38_27b_qb2.tests.attention_tuning import geometry


@pytest.mark.parametrize("batch", [4, 8, 16, 32])
def test_core_sets_fit_blackhole_and_match_declared_work(batch):
    for name in LAYOUTS:
        item = placement(name, batch)
        points = [tuple(p) for p in item["logical_cores"]]
        assert len(points) == len(set(points)) == item["grid"][0] * item["grid"][1]
        assert all(0 <= x < 11 and 0 <= y < 10 for x, y in points)
        assert item["active_cores"] == batch * item["cores_per_user"] <= len(points)
        assert points == sorted(points, key=lambda p: (p[1], p[0]))
        assert len(points[:batch]) == batch
    for left, right in (("row_major_64", "bank_proximity_64"), ("row_major_80", "outer_columns_80")):
        a, b = placement(left, batch), placement(right, batch)
        assert a["active_cores"] == b["active_cores"]
        assert a["output_layout"] == b["output_layout"]
        assert a["logical_cores"] != b["logical_cores"]


def test_bank_proximity_candidates_balance_both_dram_sides():
    for name, count in (("bank_proximity_64", 32), ("outer_columns_80", 40)):
        points = placement(name, 8)["logical_cores"]
        assert sum(x < 4 for x, _ in points) == count
        assert sum(x >= 7 for x, _ in points) == count


def test_bandwidth_accounting_counts_each_users_causal_kv_once():
    assert useful_kv_bytes([31, 63]) == 96 * 544
    assert useful_kv_bytes([131071] * 8) == 570425344


def test_wider_attention_batch_preserves_independent_causal_limits():
    case = geometry(32768, 32)
    assert len(case["positions"]) == len(set(case["positions"])) == 32
    assert 0 <= min(case["positions"]) <= max(case["positions"]) < case["aligned_capacity"]
    assert case["pool_tokens"] == 32 * case["aligned_capacity"]

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for lm_head multi-split planning helpers."""

from __future__ import annotations

from models.demos.gemma4.tt.dram_sharded import find_k_core_grid, pad_n_for_cores, plan_column_splits


def test_find_k_core_grid_gemma4_hidden():
    rows, cols = find_k_core_grid(5376)
    assert rows * cols == 56
    assert 5376 % (32 * 56) == 0


def test_plan_column_splits_sums_to_n():
    sizes = plan_column_splits(65536, 8192)
    assert sum(sizes) == 65536
    assert sizes[:-1] == [8192] * (len(sizes) - 1)
    assert sizes[-1] == 65536 - 8192 * (len(sizes) - 1)
    assert len(sizes) == 8


def test_plan_column_splits_tile_aligns_max():
    # 4000 → tile-aligned down to 3968
    sizes = plan_column_splits(65536, 4000)
    assert all(s % 32 == 0 or i == len(sizes) - 1 for i, s in enumerate(sizes[:-1]))
    assert sum(sizes) == 65536


def test_plan_column_splits_rejects_tiny_max(expect_error):
    with expect_error(ValueError, "max_columns must be >= 32"):
        plan_column_splits(65536, 16)


def test_pad_n_for_cores_8192_on_56():
    # Bug that scrambled lm_head logits: 8192 → effective 8960 without real pad.
    assert pad_n_for_cores(8192, 56) == 8960
    assert pad_n_for_cores(8960, 56) == 8960
    assert pad_n_for_cores(5376, 56) == 5376


def test_pad_n_for_cores_last_split_1024():
    # Default max_columns=7168 leaves a 1024-wide last split; pad to 56-core step.
    assert pad_n_for_cores(1024, 56) == 1792
    assert pad_n_for_cores(7168, 56) == 7168

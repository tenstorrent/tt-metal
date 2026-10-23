# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the fused norm op's work planner (custom_ops/fused_norm_decode): the core count per row tile,
the row-tile groups per pass and the passes are derived from the grid -- Wormhole 8x8, Blackhole 13x10, a 1-core
degenerate grid -- never from a constant. No device, no ttnn import beyond the module's.

    pytest models/demos/gemma4/tests/unit/custom_ops/test_fused_norm_plan_host.py --noconftest -p no:cacheprovider
"""

import pytest

from models.demos.gemma4.custom_ops.fused_norm_decode.op import FusedNormDecode

WH, BH, ONE = 8 * 8, 13 * 10, 1
COLS_26B = 2816 // 32  # 88 column tiles


@pytest.fixture(autouse=True)
def _default_cap(monkeypatch):
    monkeypatch.delenv("GEMMA4_FUSED_NORM_CORES", raising=False)


def test_decode_row_keeps_the_22_core_split():
    assert FusedNormDecode.plan(COLS_26B, 1, 1, BH) == (22, 4, 1, 1)
    assert FusedNormDecode.plan(COLS_26B, 1, 1, WH) == (22, 4, 1, 1)
    assert FusedNormDecode.plan(COLS_26B, 2, 1, WH) == (22, 4, 1, 1)
    assert FusedNormDecode.num_cores(COLS_26B, 1) == 22


def test_prefill_rows_derive_from_the_grid():
    # 128 rows (4 row tiles): Blackhole 22 cores x 4 groups in one pass, Wormhole 11 x 4 in one pass
    assert FusedNormDecode.plan(COLS_26B, 1, 4, BH) == (22, 4, 4, 1)
    assert FusedNormDecode.plan(COLS_26B, 1, 4, WH) == (11, 8, 4, 1)
    # the two-input form needs <= 4 tiles per core: 22 cores -> 2 groups per pass on Wormhole, 2 passes
    assert FusedNormDecode.plan(COLS_26B, 2, 4, WH) == (22, 4, 2, 2)
    assert FusedNormDecode.plan(COLS_26B, 2, 4, BH) == (22, 4, 4, 1)
    # 256 rows (8 row tiles): Wormhole 11 x 5 groups -> 2 passes; Blackhole 11 x 8 groups in one pass (88 cores)
    assert FusedNormDecode.plan(COLS_26B, 1, 8, WH) == (11, 8, 5, 2)
    assert FusedNormDecode.plan(COLS_26B, 1, 8, BH) == (11, 8, 8, 1)
    # 1024 rows (32 row tiles) on Blackhole: 3 passes of 11 groups x 11 cores
    assert FusedNormDecode.plan(COLS_26B, 1, 32, BH) == (11, 8, 11, 3)


def test_degenerate_grids():
    # one core cannot hold 88 column tiles of one row in a dest window: no plan (the callers fall back)
    assert not FusedNormDecode.plan_ok(COLS_26B, 1, 4, ONE)
    raised = False  # this host test runs without the repo conftest, so no expect_error fixture
    try:
        FusedNormDecode.plan(COLS_26B, 1, 1, ONE)
    except ValueError:
        raised = True
    assert raised
    # a narrow width on one core: one row tile per pass, four passes
    assert FusedNormDecode.plan(8, 1, 4, ONE) == (1, 8, 1, 4)
    # the cap knob bounds the preferred count (fewest passes first), the grid still decides the groups
    import os

    os.environ["GEMMA4_FUSED_NORM_CORES"] = "11"
    try:
        assert FusedNormDecode.plan(COLS_26B, 1, 4, BH) == (11, 8, 4, 1)
        assert FusedNormDecode.plan(COLS_26B, 1, 1, BH) == (11, 8, 1, 1)
    finally:
        del os.environ["GEMMA4_FUSED_NORM_CORES"]
    # nothing under the cap fits: the split goes above it and, as uncapped, takes the most cores at the pass count
    os.environ["GEMMA4_FUSED_NORM_CORES"] = "4"
    try:
        assert FusedNormDecode.plan(COLS_26B, 1, 4, BH) == (22, 4, 4, 1)
    finally:
        del os.environ["GEMMA4_FUSED_NORM_CORES"]

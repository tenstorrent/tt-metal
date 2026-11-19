# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Host check of ``prefill_linear_row_block``: the TP4 stem widths keep the requested row block (the validated
mesh configs are unchanged) and the single-chip full-width q_b stem (16384 wide, per_core_N 64 tiles, 2.3 MB of
CBs at 1024 rows: the L1 overflow of 2026-10-26 on chip 0) is split into row blocks that fit the L1 budget."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

ttnn = pytest.importorskip("ttnn")

from models.demos.glm53_flash.tt.matmul_configs import (  # noqa: E402
    PREFILL_L1_BUDGET,
    PREFILL_ROW_BLOCK,
    TILE,
    _tile_bytes,
    prefill_linear_l1_bytes,
    prefill_linear_row_block,
)

G13 = SimpleNamespace(x=13, y=10)
G8 = SimpleNamespace(x=8, y=8)


@pytest.mark.parametrize("grid", [G13, G8])
def test_single_chip_qb_stem_fits(grid):
    bf8 = _tile_bytes(ttnn.bfloat8_b)
    full = prefill_linear_l1_bytes(1024, 1536, 16384, grid, 4, bf8)
    assert full > 1_500_000, full  # the overflowing configuration
    rb = prefill_linear_row_block(2112, 1536, 16384, grid, PREFILL_ROW_BLOCK // 2, bf8)
    assert rb % TILE == 0 and rb < PREFILL_ROW_BLOCK // 2
    assert prefill_linear_l1_bytes(rb, 1536, 16384, grid, 4, bf8) <= PREFILL_L1_BUDGET


@pytest.mark.parametrize(
    "M,K,N,rb",
    [
        (2048, 4096, 1536, PREFILL_ROW_BLOCK),
        (2048, 1536, 4096, PREFILL_ROW_BLOCK // 2),
        (16384, 4096, 512, PREFILL_ROW_BLOCK),
    ],
)
def test_mesh_widths_unchanged(M, K, N, rb):
    for dt in (ttnn.bfloat16, ttnn.bfloat8_b):
        assert prefill_linear_row_block(M, K, N, G13, rb, _tile_bytes(dt)) == rb


def test_row_block_bounds():
    assert prefill_linear_row_block(64, 4096, 1536, G13, PREFILL_ROW_BLOCK) == 64
    assert prefill_linear_row_block(32, 4096, 1536, G13, PREFILL_ROW_BLOCK) == TILE
    assert _tile_bytes(ttnn.float32) == 4096 and _tile_bytes(ttnn.bfloat16) == 2048

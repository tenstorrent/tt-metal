# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import ttnn
from models.tt_dit.blocks.attention_opt import Attention
from models.tt_dit.models.transformers.flux3_klein import transformer_flux3_klein  # noqa: F401  (registers the entries)
from models.tt_dit.utils import matmul
from models.tt_dit.utils.matmul import get_agmm_config, get_matmul_config

FULL_GRID = ttnn.CoreCoord(12, 10)


def blocking(m, n):
    c = get_matmul_config(m, 3072, n, ttnn.CoreCoord(12, 9), use_heuristic=True)
    return c.M_block_size, c.K_block_size, c.N_block_size


@pytest.mark.parametrize(
    ("m", "n", "expected"),
    [
        (5120, 2304, (11, 8, 8)),  # 10 references, early stage (226340: 14x8x8 clashed)
        (3200, 4608, (4, 8, 16)),  # 2048 + 2 references (226340: 9x8x16 past L1)
        (9344, 4608, (4, 8, 16)),  # 4096 + 2 references (226340: 16x8x16)
        (9344, 2304, (11, 8, 8)),
    ],
)
def test_overflowing_shapes_are_cut(m, n, expected):
    assert blocking(m, n) == expected


@pytest.mark.parametrize(
    ("m", "n", "expected"),
    [(2048, 4608, (6, 8, 16)), (2816, 2304, (8, 8, 8)), (1664, 4608, (5, 8, 16)), (5120, 4608, (6, 8, 16))],
)
def test_shapes_that_run_keep_the_heuristic(m, n, expected):  # 226340 ran these
    assert blocking(m, n) == expected


def test_tabled_shapes_keep_their_entry():
    assert blocking(8192, 2304) == (11, 8, 8)
    assert blocking(4224, 4608) == (4, 8, 16)


@pytest.mark.parametrize("n", [2304, 4608, 9216])
def test_every_m_fits(n):
    for m in range(128, 32768 + 1, 128):
        mb, kb, nb = blocking(m, n)
        assert mb * kb + kb * nb + mb * nb <= 272, (m, n, mb, kb, nb)


def agmm(m, n, tp):  # as the model calls it: QKV (N 9216 / TP) and MLP-in (N 18432 / TP, fused SwiGLU)
    grid, c, workers = get_agmm_config(
        m,
        3072,
        n,
        FULL_GRID,
        tp,
        2,
        core_grid=Attention.get_core_grid(m, FULL_GRID),
        use_heuristic=True,
        fuse_swiglu=n == 18432 // tp,
    )
    return (grid.x, grid.y), (c.M_block_size, c.K_block_size, c.N_block_size, (c.subblock_h, c.subblock_w)), workers


def mm(m, k, n, grid):
    c = get_matmul_config(m, k, n, ttnn.CoreCoord(*grid))
    return c.M_block_size, c.K_block_size, c.N_block_size, (c.subblock_h, c.subblock_w)


def test_swept_shapes_resolve_to_their_entry():
    for (m, k, n, tp), (grid, blocking, buffers) in matmul.agmm_shape_configs.items():
        assert agmm(m, n, tp) == (grid, blocking, -(-grid[0] // 2))
        assert matmul.agmm_shape_buffers(m, k, n, tp) == buffers
    for (m, k, n, *grid), blocking in matmul.matmul_shape_configs.items():
        assert mm(m, k, n, grid) == blocking


def test_every_other_shape_resolves_as_without_the_entries(monkeypatch):
    rows = range(128, 32768 + 1, 128)
    agmm_shapes = [(m, n, tp) for m in rows for tp in (2, 4) for n in (9216 // tp, 18432 // tp)]
    mm_shapes = [(m, k, 3072, grid) for m in rows for k in (3072, 6144) for grid in ((12, 10), (11, 10))]
    swept_agmm = {(m, n, tp) for m, _, n, tp in matmul.agmm_shape_configs}
    swept_mm = {(m, k, n, tuple(grid)) for m, k, n, *grid in matmul.matmul_shape_configs}
    now = {s: agmm(*s) for s in agmm_shapes if s not in swept_agmm}
    now_mm = {s: mm(*s) for s in mm_shapes if s not in swept_mm}
    monkeypatch.setattr(matmul, "agmm_shape_configs", {})
    monkeypatch.setattr(matmul, "matmul_shape_configs", {})
    matmul._clear_config_caches()
    try:
        assert {s: agmm(*s) for s in now} == now
        assert {s: mm(*s) for s in now_mm} == now_mm
    finally:
        monkeypatch.undo()
        matmul._clear_config_caches()

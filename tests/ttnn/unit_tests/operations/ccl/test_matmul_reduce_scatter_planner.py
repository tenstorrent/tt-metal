# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the matmul_reduce_scatter grid planner (no device): balanced n-lines."""

import pytest

from ttnn.operations.matmul_reduce_scatter import matmul_reduce_scatter_program_descriptor as planner


@pytest.mark.parametrize("block_n", [1, 7, 43, 56, 100, 224])
@pytest.mark.parametrize("grid_n", [1, 5, 9, 12, 16])
def test_balanced_line_columns_partition_every_block(block_n, grid_n):
    """Each block's lines tile its columns exactly, and no line is wide more than ceil(G * extra / grid_n) times."""
    if grid_n > block_n:
        pytest.skip("a balanced plan has at most block_n lines")
    base, extra = divmod(block_n, grid_n)
    for G in (2, 4, 8):
        widths = [[0] * grid_n for _ in range(G)]
        for b in range(G):
            cols = planner.balanced_line_columns(block_n, grid_n, b)
            origin = 0
            for gn, (col, width) in enumerate(cols):
                assert col == origin and width in (base, base + 1)
                origin += width
                widths[b][gn] = width
            assert origin == block_n
        busiest = max(sum(widths[b][gn] for b in range(G)) for gn in range(grid_n))
        assert busiest == G * base + -(-G * extra // grid_n)


def test_kimi_k3_output_projection_grid():
    """Kimi-K3 o_proj on Galaxy (12 x 10 grid, row port strip, block 20 x 56 tiles, TP 4): 80 -> 90 cores."""
    uniform = planner.choose_grid(12, 10, 20, 56, 2, G=4)
    balanced = planner.choose_grid(12, 10, 20, 56, 2, G=4, balance_lines=True)
    assert (uniform["strip"], uniform["grid_m"], uniform["grid_n"], uniform["core_n"]) == ("row", 10, 8, 7)
    assert not uniform["balanced"]
    assert (balanced["strip"], balanced["grid_m"], balanced["grid_n"], balanced["core_n"]) == ("row", 10, 9, 7)
    assert balanced["balanced"]
    # the busiest core: 2 rows x (7 + 6 + 6 + 6) columns over the 4 blocks, against 2 x 7 x 4
    widths = [planner.balanced_line_columns(56, 9, b) for b in range(4)]
    assert max(2 * sum(widths[b][gn][1] for b in range(4)) for gn in range(9)) == 50


@pytest.mark.parametrize("gx,gy", [(12, 10), (11, 10), (8, 8)])
def test_balance_lines_never_adds_work(gx, gy):
    """The balanced candidate wins only at less or equal busiest-core work, and leaves the default plan unchanged."""
    for block_m in range(1, 64, 5):
        for block_n in range(1, 260, 9):
            for G in (2, 4, 8):
                uniform = planner.choose_grid(gx, gy, block_m, block_n, 2, G=G)
                balanced = planner.choose_grid(gx, gy, block_m, block_n, 2, G=G, balance_lines=True)
                assert not uniform["balanced"]
                work = uniform["core_m"] * G * uniform["core_n"]
                if balanced["balanced"]:
                    base, extra = divmod(block_n, balanced["grid_n"])
                    work_b = balanced["core_m"] * (G * base + -(-G * extra // balanced["grid_n"]))
                    assert work_b <= work
                else:
                    assert balanced == uniform

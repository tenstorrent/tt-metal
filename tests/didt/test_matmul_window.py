# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Host-only tests for matmul_window (no device needed). Expected sizes are worked out by hand from the matmul
# factories' block ownership (see the module docstring):
#   pytest tests/didt/test_matmul_window.py

import ttnn

from tt_py_test_utils_common.matmul_window import window_matmul


def config_2d(grid, pcM, pcN):
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=pcM,
        per_core_N=pcN,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def config_1d_mcast_in1(grid, pcM, pcN):
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(*grid),
        in0_block_w=4,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=pcM,
        per_core_N=pcN,
        fuse_batch=False,
        fused_activation=None,
        mcast_in0=False,
    )


def test_2d_left_window_keeps_full_blocks():
    # DeepSeek x_wq_a on 11x10: 4096x1536 output, per core 13x5 tiles. Window columns 0..4: 5 full column blocks
    M, N, cfg = window_matmul(config_2d((11, 10), 13, 5), 4096, 1536, (5, 10), (0, 0))
    assert (M, N) == (4096, 5 * 5 * 32)
    assert (cfg.per_core_M, cfg.per_core_N) == (13, 5)
    assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (5, 10)
    r = cfg.allowed_worker_cores.ranges()[0]
    assert (r.start.x, r.start.y, r.end.x, r.end.y) == (0, 0, 4, 9)


def test_2d_right_window_keeps_partial_last_block_and_idle_column():
    # columns 6..10: blocks 6, 7, 8 (5 tiles), block 9 partial (48 - 45 = 3 tiles), column 10 idle on the full grid
    M, N, cfg = window_matmul(config_2d((11, 10), 13, 5), 4096, 1536, (5, 10), (6, 0))
    assert (M, N) == (4096, (3 * 5 + 3) * 32)
    r = cfg.allowed_worker_cores.ranges()[0]
    assert (r.start.x, r.start.y, r.end.x, r.end.y) == (6, 0, 10, 9)


def test_2d_window_without_blocks_is_refused(expect_error):
    with expect_error(ValueError, "holds no output blocks"):
        window_matmul(config_2d((11, 10), 13, 5), 4096, 1536, (1, 10), (10, 0))


def test_window_outside_grid_is_refused(expect_error):
    with expect_error(ValueError, "not inside the config grid"):
        window_matmul(config_2d((11, 10), 13, 5), 4096, 1536, (5, 10), (7, 0))


def test_1d_mcast_in1_window_owns_full_grid_blocks():
    # mcast-ack on a 10x10 grid: Mt = 160, per core 2 tiles -> 80 row blocks on cores 0..79 (rows y = 0..7).
    # Window columns 7..9: cores y*10 + x with x in 7..9 and index < 80 -> 3 per row for y = 0..7 = 24 blocks
    M, N, cfg = window_matmul(config_1d_mcast_in1((10, 10), 2, 16), 5120, 512, (3, 10), (7, 0), batch=64)
    assert (M, N) == (24 * 2 * 32, 512)
    assert cfg.mcast_in0 is False
    r = cfg.allowed_worker_cores.ranges()[0]
    assert (r.start.x, r.start.y, r.end.x, r.end.y) == (7, 0, 9, 9)


def test_2d_transpose_mcast_swaps_rows_and_columns():
    # transpose_mcast: in0 (rows) follows core.x. 11x10 grid, Mt = 110 (11 row blocks of 10), Nt = 50 (10 col
    # blocks of 5). Window columns 0..4 -> row blocks 0..4 (50 tiles), all 10 rows of cores -> all 10 col blocks
    cfg = config_2d((11, 10), 10, 5)
    cfg.transpose_mcast = True
    M, N, w = window_matmul(cfg, 110 * 32, 50 * 32, (5, 10), (0, 0))
    assert (M, N) == (50 * 32, 50 * 32)
    assert w.transpose_mcast


def test_1d_mcast_in0_owns_column_blocks():
    # mcast_in0: core i owns column block i. 4x2 grid, Nt = 24, per core 3 -> 8 blocks on cores 0..7.
    # Window 2x2 at (2, 0): cores 2, 3, 6, 7 -> 4 blocks of 3 tiles
    cfg = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(4, 2),
        in0_block_w=1,
        out_subblock_h=1,
        out_subblock_w=1,
        per_core_M=2,
        per_core_N=3,
        fuse_batch=True,
        fused_activation=None,
        mcast_in0=True,
    )
    M, N, w = window_matmul(cfg, 64, 24 * 32, (2, 2), (2, 0))
    assert (M, N) == (64, 4 * 3 * 32)


def test_refusals(expect_error):
    from tt_py_test_utils_common.matmul_window import NoBlocksInWindow, NotWindowable

    fused = config_2d((11, 10), 13, 5)
    fused.fuse_batch = True
    with expect_error(NotWindowable, "fuse_batch"):
        window_matmul(fused, 4096, 1536, (5, 10), (0, 0), batch=2)
    placed = config_2d((11, 10), 13, 5)
    placed.allowed_worker_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(10, 9))])
    with expect_error(NotWindowable, "already placed"):
        window_matmul(placed, 4096, 1536, (5, 10), (0, 0))
    # 1D mcast_in1 with 80 blocks on a 10x10 grid: rows 8 and 9 own nothing
    with expect_error(NoBlocksInWindow, "holds no output blocks"):
        window_matmul(config_1d_mcast_in1((10, 10), 2, 16), 5120, 512, (10, 2), (0, 8), batch=64)

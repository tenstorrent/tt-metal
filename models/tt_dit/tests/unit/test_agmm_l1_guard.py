# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only: the heuristic all-gather-matmul blocking is shrunk to fit Blackhole L1 (no device).

The Klein DiT under SP8 x TP4 at 4K hands ``get_agmm_config`` tall-M, wide-N shapes whose heuristic
blocking the ``minimal_matmul`` factory throws on ("Statically allocated circular buffers ... beyond
max L1 size"). The guard keeps every blocking that already fits byte-identical (the SP4 x TP8 shapes
below are the ones measured in docs/STAGE_TIMING.md) and only shrinks the ones that would throw.
"""
import pytest

import ttnn
from models.tt_dit.utils.matmul import AGMM_CB_BUDGET_BYTES, agmm_cb_bytes, get_agmm_config

FULL = ttnn.CoreCoord(12, 10)
GRID = ttnn.CoreCoord(12, 9)  # Attention.get_core_grid for M > 1152


def _cfg(M, N, tp, swiglu):
    _, cfg, _ = get_agmm_config(
        M, 3072, N, full_grid=FULL, cluster_size=tp, num_links=2, core_grid=GRID, use_heuristic=True, fuse_swiglu=swiglu
    )
    return cfg


def test_cb_bytes_matches_factory_formula():
    # in0 2*16*6 + in1 2*6*8 + out 2*(16*8/2) (SwiGLU) in bf16 tiles, interm 16*8 in fp32, bias 8 tiles.
    assert (
        agmm_cb_bytes(16, 6, 8, fuse_swiglu=True)
        == 2 * 96 * 2048 + 2 * 48 * 2048 + 2 * 64 * 2048 + 128 * 4096 + 8 * 2048
    )
    assert agmm_cb_bytes(16, 8, 8) == 1851392  # the SP8xTP4 4K QKV blocking the factory threw on
    assert agmm_cb_bytes(2, 2, 2, use_addcmul=True) == agmm_cb_bytes(2, 2, 2) + 4 * 2048 + 2 * 2048


@pytest.mark.parametrize(
    "M, N, swiglu, expected",
    [  # SP4 x TP8 (K tiles per device 12 -> K block 6), measured blockings stay as they are
        (4096, 1152, False, (11, 6, 4)),
        (4096, 2304, True, (11, 6, 8)),
        (16384, 1152, False, (16, 6, 4)),
        (16384, 2304, True, (16, 6, 8)),  # 1392640 B: the largest blocking proven on device
        (18528, 2304, True, (16, 6, 8)),
    ],
)
def test_fitting_heuristic_blockings_are_unchanged(M, N, swiglu, expected):
    cfg = _cfg(M, N, 8, swiglu)
    assert (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size) == expected
    assert agmm_cb_bytes(*expected, fuse_swiglu=swiglu) <= AGMM_CB_BUDGET_BYTES


@pytest.mark.parametrize(
    "M, N, swiglu, heuristic",
    [  # SP8 x TP4 (K block 8): the heuristic's blockings exceed L1
        (4608, 2304, False, (12, 8, 8)),
        (4608, 4608, True, (12, 8, 16)),
        (8192, 2304, False, (16, 8, 8)),
        (8192, 4608, True, (16, 8, 16)),
        (9264, 4608, True, (16, 8, 16)),  # 4K + 10 references
    ],
)
def test_overflowing_heuristic_blockings_are_shrunk_to_fit(M, N, swiglu, heuristic):
    assert agmm_cb_bytes(*heuristic, fuse_swiglu=swiglu) > AGMM_CB_BUDGET_BYTES
    cfg = _cfg(M, N, 4, swiglu)
    blocks = (cfg.M_block_size, cfg.K_block_size, cfg.N_block_size)
    assert agmm_cb_bytes(*blocks, fuse_swiglu=swiglu) <= AGMM_CB_BUDGET_BYTES
    assert cfg.K_block_size == 8 and cfg.N_block_size == heuristic[2]  # only M shrinks for these
    assert cfg.M_block_size < heuristic[0]
    assert cfg.M_block_size % cfg.subblock_h == 0 and cfg.N_block_size % cfg.subblock_w == 0
    assert (cfg.compute_with_storage_grid_size.x, cfg.compute_with_storage_grid_size.y) == (GRID.x, GRID.y)

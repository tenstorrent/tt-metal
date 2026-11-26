# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
L1 BLOCK_SHARDED, ragged shards (Refinement 3): the geometry classes the kernels branch on, out of place and in
place, bf16, gamma/beta bf16 RM.

  * padded last core column (TILE: whole padding tiles skipped; RM: lanes masked after the column collapse);
  * padded last core row (TILE: valid tile rows only; RM: the partial tail tile row staged k-filled);
  * shards straddling images (a core holds two image segments: two statistic sets, the band root finalizes both);
  * groups straddling core columns (the statistic unit is the band's whole grid width, RowMajor mcast groups).
"""

import pytest
import torch
import ttnn

from eval.golden_tests.groupnorm_sc_N_1_HW_C_b.helpers import run_groupnorm_sc_N_1_HW_C

AXES = dict(
    dtype=ttnn.bfloat16,
    affine="gamma_beta",
    affine_dtype=ttnn.bfloat16,
    affine_layout=ttnn.ROW_MAJOR_LAYOUT,
)

CASES = [
    # TILE, padded last column (model shapes 8x8)
    pytest.param(((1, 1, 4096, 960), 32, (8, 8)), ttnn.TILE_LAYOUT, id="tile_4096x960_8x8_padcol"),
    pytest.param(((1, 1, 1024, 1920), 32, (8, 8)), ttnn.TILE_LAYOUT, id="tile_1024x1920_8x8_padcol"),
    pytest.param(((1, 1, 4096, 1152), 32, (8, 8)), ttnn.TILE_LAYOUT, id="tile_4096x1152_8x8_padcol"),
    # padded last row
    pytest.param(((1, 1, 1024, 1280), 32, (8, 7)), ttnn.TILE_LAYOUT, id="tile_1024x1280_8x7_padrow"),
    pytest.param(((1, 1, 1024, 1280), 32, (8, 7)), ttnn.ROW_MAJOR_LAYOUT, id="rm_1024x1280_8x7_padrow"),
    pytest.param(((1, 1, 4096, 320), 32, (8, 7)), ttnn.ROW_MAJOR_LAYOUT, id="rm_4096x320_8x7_k4_padrow"),
    # shards straddling images
    pytest.param(((2, 1, 256, 1280), 32, (8, 3)), ttnn.TILE_LAYOUT, id="tile_2x256x1280_8x3_straddle"),
    pytest.param(((2, 1, 256, 1280), 32, (8, 3)), ttnn.ROW_MAJOR_LAYOUT, id="rm_2x256x1280_8x3_straddle"),
    # groups straddling core columns
    pytest.param(((1, 1, 1024, 1280), 32, (7, 8)), ttnn.TILE_LAYOUT, id="tile_1024x1280_7x8_grpcol"),
    pytest.param(((1, 1, 1024, 1280), 32, (7, 8)), ttnn.ROW_MAJOR_LAYOUT, id="rm_1024x1280_7x8_grpcol"),
    pytest.param(((1, 1, 4096, 640), 32, (6, 8)), ttnn.ROW_MAJOR_LAYOUT, id="rm_4096x640_6x8_padcol"),
]


@pytest.mark.parametrize("inputs, layout", CASES)
def test_ragged_oop(device, inputs, layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, layout=layout, in_place=False, **AXES)


@pytest.mark.parametrize("inputs, layout", CASES)
def test_ragged_inplace(device, inputs, layout):
    torch.manual_seed(0)
    run_groupnorm_sc_N_1_HW_C(inputs, device=device, layout=layout, in_place=True, **AXES)

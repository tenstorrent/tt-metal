# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""L1 BLOCK_SHARDED placement + in_place output for groupnorm_sc_N_1_HW_C.

The input shard is each core's block: TILE shards back the input/output CBs directly (zero-copy),
ROW_MAJOR shards are consumed in place through the row-major direct view where it is exact and
staged stick-by-stick otherwise. Two shard families are pinned here:

  * the SDXL model geometry (8x8 grid, [HW/8, C/8] shards — RM widths 40/80/120/160/240/320
    channels, i.e. per-core channel counts that are NOT multiples of 32; TILE where tile-aligned)
  * auto-shard-like geometry on the live grid (ceil-split with a padded last shard row / column,
    RM shard heights that are not multiples of 32, N > 1 shards straddling images)
"""

import pytest
import torch
import ttnn

from tests.ttnn.unit_tests.operations.groupnorm_sc_N_1_HW_C.test_groupnorm_sc_N_1_HW_C import (
    compute_pcc,
    torch_groupnorm_n_1_hw_c,
)
from ttnn.operations.groupnorm_sc_N_1_HW_C import groupnorm_sc_N_1_HW_C

PCC_THRESHOLD = 0.995
RMS_THRESHOLD = 0.02  # bf16 rms gate


def block_shard_config(shard_shape, grid_xy):
    """Explicit L1 block shard: [shard_h, shard_w] per core on an (x, y) rectangle from (0,0)."""
    gx, gy = grid_xy
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))})
    spec = ttnn.ShardSpec(grid, [int(shard_shape[0]), int(shard_shape[1])], ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, spec)


def run_sharded(
    device, shape, num_groups, *, layout, shard_shape, grid, in_place, affine="gamma_beta", mean_offset=0.0
):
    torch.manual_seed(42)
    N, _, HW, C = shape
    x = (torch.randn(shape, dtype=torch.float32) + mean_offset).to(torch.bfloat16)
    gamma = beta = None
    if affine in ("gamma_beta", "gamma_only"):
        gamma = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)
    if affine == "gamma_beta":
        beta = torch.randn(1, 1, 1, C, dtype=torch.float32).to(torch.bfloat16)

    mc = block_shard_config(shard_shape, grid)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=mc)
    kwargs = {}
    if gamma is not None:
        kwargs["gamma"] = ttnn.from_torch(
            gamma,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    if beta is not None:
        kwargs["beta"] = ttnn.from_torch(
            beta,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
    tt_y = groupnorm_sc_N_1_HW_C(tt_x, num_groups, in_place=in_place, **kwargs)

    assert list(tt_y.shape) == list(shape)
    assert tt_y.layout == layout
    assert tt_y.memory_config().memory_layout == ttnn.TensorMemoryLayout.BLOCK_SHARDED
    if in_place:
        assert tt_y.buffer_address() == tt_x.buffer_address(), "in_place must return the input's own shard"

    expected = torch_groupnorm_n_1_hw_c(x, num_groups, gamma=gamma, beta=beta)
    actual = ttnn.to_torch(tt_y)
    pcc = compute_pcc(actual.float(), expected.float())
    assert torch.isfinite(actual.float()).all(), "non-finite values in output"
    # RMS gate at the bf16 tolerance: a stale-stats race on N > 1 shards (the next
    # image's mean landing mid pass 3) kept PCC at 0.9995 but pushed RMS to 0.03-0.045.
    rms = float(((actual.float() - expected.float()) ** 2).mean().sqrt())
    assert (
        rms <= RMS_THRESHOLD
    ), f"RMS {rms:.4f} > {RMS_THRESHOLD} for {shape} G={num_groups} {layout} shard={shard_shape} grid={grid}"
    assert (
        pcc >= PCC_THRESHOLD
    ), f"PCC {pcc:.6f} < {PCC_THRESHOLD} for {shape} G={num_groups} {layout} shard={shard_shape} grid={grid}"
    return pcc


IN_PLACE = [pytest.param(False, id="out"), pytest.param(True, id="in_place")]

# --- TILE shards ---------------------------------------------------------------------------
TILE_CASES = [
    # shape, G, shard [h, w], grid (x, y)
    pytest.param((1, 1, 256, 128), 4, [32, 32], (4, 8), id="tile_1x1_shards_4x8"),
    pytest.param((1, 1, 1024, 1280), 32, [128, 160], (8, 8), id="tile_sdxl_model_1024x1280"),
    pytest.param((1, 1, 4096, 1280), 32, [512, 160], (8, 8), id="tile_sdxl_model_4096x1280"),
    pytest.param((1, 1, 2048, 128), 4, [224, 32], (4, 10), id="tile_ragged_last_row_shard"),
    pytest.param((1, 1, 64, 4096), 32, [32, 384], (11, 2), id="tile_ragged_last_col_shard"),
    pytest.param((2, 1, 64, 128), 4, [32, 32], (4, 4), id="tile_batch2_one_row_per_image"),
    pytest.param((2, 1, 1024, 640), 32, [224, 64], (10, 10), id="tile_batch2_shards_straddle_images"),
    pytest.param((8, 1, 64, 64), 2, [64, 32], (2, 8), id="tile_batch8"),
]


@pytest.mark.parametrize("shape,num_groups,shard_shape,grid", TILE_CASES)
@pytest.mark.parametrize("in_place", IN_PLACE)
def test_block_sharded_tile(device, shape, num_groups, shard_shape, grid, in_place):
    run_sharded(
        device, shape, num_groups, layout=ttnn.TILE_LAYOUT, shard_shape=shard_shape, grid=grid, in_place=in_place
    )


# --- ROW_MAJOR shards ------------------------------------------------------------------------
RM_CASES = [
    # SDXL model geometry: 8x8 grid, shard widths that are not multiples of 32 (40 = 1.25 tiles)
    pytest.param((1, 1, 1024, 320), 32, [128, 40], (8, 8), id="rm_model_w40"),
    pytest.param((1, 1, 16384, 320), 32, [2048, 40], (8, 8), id="rm_model_sdxl_16384x320"),  # perf_cases.py anchor
    pytest.param((1, 1, 4096, 640), 32, [512, 80], (8, 8), id="rm_model_sdxl_4096x640"),
    pytest.param((1, 1, 1024, 640), 32, [128, 80], (8, 8), id="rm_model_w80"),
    pytest.param((1, 1, 1024, 960), 32, [128, 120], (8, 8), id="rm_model_w120"),
    pytest.param((1, 1, 1024, 1280), 32, [128, 160], (8, 8), id="rm_model_w160"),
    pytest.param((1, 1, 1024, 1920), 32, [128, 240], (8, 8), id="rm_model_w240"),
    pytest.param((1, 1, 1024, 2560), 32, [128, 320], (8, 8), id="rm_model_w320"),
    pytest.param((1, 1, 4096, 640), 32, [512, 160], (4, 8), id="rm_model_attn_4x8"),
    # auto-shard-like: shard heights that are not multiples of 32 (partial tile-rows -> pass-2 masks)
    pytest.param((1, 1, 64, 64), 2, [7, 8], (8, 10), id="rm_ragged_h7_w8"),
    pytest.param((1, 1, 32, 320), 32, [4, 32], (10, 8), id="rm_ragged_h4"),
    pytest.param((1, 1, 2048, 128), 4, [205, 16], (8, 10), id="rm_ragged_h205"),
    pytest.param((1, 1, 1024, 640), 32, [103, 64], (10, 10), id="rm_ragged_h103_w64"),
    # N > 1: shards straddle images mid tile-row
    pytest.param((2, 1, 64, 128), 4, [13, 16], (8, 10), id="rm_batch2_straddle_h13"),
    pytest.param((2, 1, 256, 1280), 32, [52, 128], (10, 10), id="rm_batch2_straddle_h52"),
]


@pytest.mark.parametrize("shape,num_groups,shard_shape,grid", RM_CASES)
@pytest.mark.parametrize("in_place", IN_PLACE)
def test_block_sharded_rm(device, shape, num_groups, shard_shape, grid, in_place):
    run_sharded(
        device, shape, num_groups, layout=ttnn.ROW_MAJOR_LAYOUT, shard_shape=shard_shape, grid=grid, in_place=in_place
    )


@pytest.mark.parametrize(
    "layout", [pytest.param(ttnn.TILE_LAYOUT, id="tile"), pytest.param(ttnn.ROW_MAJOR_LAYOUT, id="rm")]
)
@pytest.mark.parametrize("affine", ["gamma_only", "no_affine"])
def test_block_sharded_affine_variants(device, layout, affine):
    run_sharded(
        device, (1, 1, 256, 256), 8, layout=layout, shard_shape=[32, 32], grid=(8, 8), in_place=True, affine=affine
    )


def test_block_sharded_rm_large_mean_stable_variance(device):
    # Partial tile-rows (shard height 4 of 32) with |mean| ~ 10 sigma: the pass-2 row mask must
    # remove the zero-staged pad sticks BEFORE squaring, or the variance is off by (0 - mean)^2.
    run_sharded(
        device,
        (1, 1, 32, 320),
        32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_shape=[4, 32],
        grid=(10, 8),
        in_place=False,
        mean_offset=10.0,
    )

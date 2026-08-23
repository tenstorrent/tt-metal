# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn
import math

from tests.ttnn.utils_for_testing import assert_numeric_metrics
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from models.common.utility_functions import is_blackhole, is_watcher_enabled, run_for_blackhole, skip_for_blackhole


DEVICE_PARAMS_L1_SMALL_SIZE = [{"l1_small_size": 0}]
# atol for the non-tile-aligned regressions, matching the tile-aligned specify_grid cases.
NON_TILE_ALIGNED_ATOL = 0.08
STATISTICS_MODES = ("tile_reduction", "online_welford", "online_welford_reciprocal_lut")

GROUP_NORM_DRAM_SHAPES = [
    (9, 768, 1, 512, 32, 2, 8, 8),  # test batch size 9 (uneven batch sizes)
    (1, 480, 1, 64, 8, 1, 1, 1),  # test last group ends less than max tile span
    (1, 2560, 1, 512, 32, 2, 8, 8),  # test mcast num_out_blocks 2
    (1, 2560, 1, 1024, 32, 4, 8, 8),  # test mcast num_out_blocks 4
    (1, 768, 1, 512, 32, 2, 8, 8),  # test group channel count is less than tile size
    (2, 768, 1, 512, 32, 2, 8, 8),  # test batch size 2 (still multicast)
    (8, 768, 1, 512, 32, 2, 8, 8),  # test batch size 8 (no multicast)
    (8, 768, 1, 512, 32, 3, 8, 8),  # test batch size 8 (no multicast), but uneven num_out_blocks divisor
    (
        1,
        128,
        1,
        512,
        32,
        2,
        4,
        4,
    ),  # test all groups on core fit in less than one tile, so need to reduce col core count
    # SDXL/sd35 test cases. Additional slower test cases in nightly test.
    # SDXL Base
    (1, 1920, 16, 16, 32, 1, 4, 4),
    # SDXL Refiner
    (1, 1536, 8, 8, 32, 1, 2, 8),
    (1, 1152, 128, 128, 32, 2, 8, 4),
    (1, 512, 64, 64, 32, 1, 8, 8),  # SD 1.4 VAE
    (1, 512, 128, 128, 32, 1, 8, 8),  # SD 1.4 VAE
    (1, 256, 256, 256, 32, 8, 8, 8),  # SD 1.4 VAE
    # sd35. 4 indicates the number of device.
    (1, 256 // 4, 256, 256, 32 // 4, 1, 8, 8),
    (1, 512 // 4, 128, 128, 32 // 4, 1, 8, 8),
    (1, 512 // 4, 256, 256, 32 // 4, 2, 8, 8),
    # mochi
    # (21, 128, 480, 848, 32, 140, 8, 8), Failing on single device CI.
]


GROUP_NORM_NO_INPUT_MASK_DRAM_SHAPES = [
    (8, 768, 1, 512, 32, 2, 8, 8),  # base case
    (1, 768, 1, 512, 32, 2, 8, 8),  # test group channel count is less than tile size
    (1, 480, 1, 64, 8, 1, 1, 1),  # test last group ends less than max tile span
]

# Non-tile-aligned flattened height, i.e. N*H*W % 32 != 0 (tt-metal #50682). PCC stays ~1.0 with
# the bug present, so max abs error vs torch is the discriminator.
# (N, C, H, W, num_groups)
GROUP_NORM_NON_TILE_ALIGNED_DRAM_SHAPES = [
    (1, 1024, 1, 200, 32),  # issue repro (H*W=200 -> padded 224, 10.7% padding)
    (1, 1024, 1, 269, 32),  # XTTS-v2 conditioning encoder (~269 mel frames)
    (1, 512, 1, 100, 32),  # larger padding fraction (H*W=100 -> padded 128, 21.9%)
]

# N > 1 with non-tile-aligned PER-SAMPLE H*W: padded per batch slice, so these are non-aligned
# even though N*H*W is a multiple of 32 -- the case tt-mlir #8935 calls out.
# (N, C, H, W, num_groups)
GROUP_NORM_NON_TILE_ALIGNED_PER_SAMPLE_DRAM_SHAPES = [
    (2, 1024, 1, 80, 32),  # H*W=80 -> padded 96 (K=0.20), N*H*W=160 is a multiple of 32
    (4, 512, 1, 40, 32),  # H*W=40 -> padded 64 (K=0.60), N*H*W=160 is a multiple of 32
]

# Explicit grid, to reach paths auto-grid does not (see the test for which). Grids are constrained
# -- validate_dram_grid in groupnorm.cpp requires, with num_virtual_cols = 8 for every row below:
#   nvr = (grid_x / num_virtual_cols) * grid_y,  Ht >= nvr,  Ht % nvr == 0,
#   (nvr < num_batches or nvr % num_batches == 0),  and num_out_blocks <= Ht / nvr.
# (N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x)
GROUP_NORM_NON_TILE_ALIGNED_GRID_DRAM_CASES = [
    # XTTS length, mcast + chunked: Ht=9, nvr=3, block_ht=3
    (1, 1024, 1, 269, 32, 3, 3, 8),
    # issue repro shape, no_mcast + chunked: Ht=7, nvr=1, block_ht=7
    (1, 1024, 1, 200, 32, 2, 1, 8),
    # no_mcast first core group + chunked: H*W=500 -> 512, Ht=16, nvr=8, batch 8 == nvr
    (8, 768, 1, 500, 32, 2, 8, 8),
    # no_mcast SECOND core group (9 batches over nvr=8 is uneven) + chunked
    (9, 768, 1, 500, 32, 2, 8, 8),
]

# Accuracy vs the padding fraction K = padded/logical - 1: the correction subtracts K*E[x]^2 off a
# bfloat16 variance, so the residual grows with K. Thresholds are MEASURED on Blackhole with ~30%
# headroom, not derived -- they pin current behaviour so a future exact fix reads as an improvement.
# (N, C, H, W, num_groups, K, max_abs_err)
GROUP_NORM_NON_TILE_ALIGNED_PADDING_FRACTION_CASES = [
    (1, 1024, 1, 200, 32, 0.12, 0.08),  # 10.7% padding -- measured 0.040
    (1, 512, 1, 40, 32, 0.60, 0.08),  # 37.5% padding (issue's worst case) -- measured 0.048
    (2, 1024, 1, 16, 32, 1.00, 0.13),  # 50.0% padding -- measured 0.093, over the 0.08 tolerance
    (4, 512, 1, 8, 32, 3.00, 0.21),  # 75.0% padding -- measured 0.157, over the 0.08 tolerance
]

# Accuracy vs the mean-to-spread ratio at the XTTS-scale padding fraction (H*W=200, K=0.12).
# `aligned_ref` is the TILE-ALIGNED H*W=224 control at the same shift: the fused bf16 kernel degrades
# with a large mean even unpadded, so the correction's real cost is the gap. `max_ratio` bounds that
# gap and is the portable half of the assertion -- both columns move together with arch and
# precision, so the ratio survives what absolute numbers do not.
# (input_shift, max_abs_err, aligned_ref, max_ratio)
GROUP_NORM_NON_TILE_ALIGNED_SHIFT_CASES = [
    (0.0, 0.08, 0.042, 2.0),  # measured 0.040 vs 0.042 aligned (ratio 0.95) -- no measurable cost
    (1.0, 0.16, None, None),  # measured 0.114; no aligned control measured at this shift
    (4.0, 0.70, 0.158, 6.0),  # measured 0.539 vs 0.158 aligned (ratio 3.4) -- the real cost
]

SDXL_BASE_GROUP_NORM_SPLIT_SHAPES = [
    # (1, 256, 1024, 1024, 32, 32), # does not fit -> input is [16384, 8] per core (~260kB) gets tilized internally to [16384, 32] which is ~1MB, and 2 buffers are of that size (cb_x and cb_in)
    (
        1,
        512,
        128,
        128,
        32,
        1,
    ),  # Can fit in 1 slice, i2s = 0.09ms GN = 1.5ms, s2i = 0.135ms = 1.725ms against 1.57ms of dram GN. Is block shardable as well, in that case it GN takes 0.35ms
    (
        1,
        512,
        256,
        256,
        32,
        4,
    ),  # Can fit in 4 slice, split= 0.3ms i2s = 0.1ms GN = 0.6ms, s2i = 0.421ms = 5.6ms + 1ms for concat = 6.6ms (original is 6.1ms)
    # (1, 128, 1024, 1024, 32, 32), # does not fit -> input is [16384, 4] per core (~130kB) gets tilized internally to [16384, 32] which is ~1MB, and 2 buffers are of that size (cb_x and cb_in). in addition to that, RM stick of size 4 is not L1 aligned
]

GROUP_NORM_DRAM_OFT_PARAMS = [
    ### oft
    (1, 256, 159, 159, 16, 3, 4, 4, 1e-5),
    (1, 64, 192, 640, 16, 10, 2, 4, 1e-5),
]


# perf_test_mode is used to skip the torch execution and pcc comparison, and always runs the operation once
def run_group_norm_DRAM(
    device,
    N,
    C,
    H,
    W,
    num_groups,
    num_out_blocks,
    cores_y,
    cores_x,
    statistics_mode,
    use_input_mask,
    perf_test_mode=False,
    specify_grid=True,
    input_layout=ttnn.TILE_LAYOUT,
    output_layout=ttnn.TILE_LAYOUT,
):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)
    if specify_grid:
        #   Use the explicit user-chosen grid.
        grid_for_params = grid_size
    else:
        # Exercises the C++ automatic grid selection.
        # We must prepare gamma/beta/mask/reciprocals for that *same* auto-selected
        # grid; otherwise their shapes (driven by num_virtual_cols/num_virtual_rows)
        # would not match the grid the op actually picks at runtime.
        grid_for_params = ttnn.determine_expected_group_norm_dram_grid_size(
            device=device,
            num_channels=C,
            num_groups=num_groups,
            input_nhw=N * H * W,
            num_batches=N,
        )

    # Determine welford and reciprocals settings
    use_welford = statistics_mode in ("online_welford", "online_welford_reciprocal_lut")
    use_reciprocals = statistics_mode == "online_welford_reciprocal_lut"

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    if not perf_test_mode:
        torch_output_tensor = torch.nn.functional.group_norm(
            torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=1e-12
        )
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    if input_layout == ttnn.TILE_LAYOUT:
        input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    gn_input_tensor = input_tensor_tilized if input_layout == ttnn.TILE_LAYOUT else input_tensor_row_major

    # Create dram group norm params
    [gamma_t, beta_t], input_mask_tensor = ttnn.dram_group_norm_params_from_torch(
        [torch_weight, torch_bias], C, num_groups, device, core_grid=grid_for_params, return_mask=True
    )

    # Create reciprocals tensor if needed
    reciprocals_tensor = None
    if use_reciprocals:
        # Generate reciprocals tensor
        torch_reciprocals = ttnn.create_group_norm_reciprocals(N, C, H, W, num_groups, grid_for_params)
        reciprocals_tensor = ttnn.from_torch(
            torch_reciprocals,
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(0, 0),
                                ttnn.CoreCoord(grid_for_params.x - 1, grid_for_params.y - 1),
                            )
                        }
                    ),
                    (
                        torch_reciprocals.shape[0] // (grid_for_params.x * grid_for_params.y),
                        torch_reciprocals.shape[1],
                    ),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )

    # groupnorm

    num_itr = 2  # second iteration to help catch potential runtime args issue.

    if C > 512 or N > 2 or perf_test_mode:
        num_itr = 1  # one iter if it is too slow
    for _ in range(num_itr):
        output_tensor = ttnn.group_norm(
            gn_input_tensor,
            num_groups=num_groups,
            input_mask=input_mask_tensor if use_input_mask else None,
            weight=gamma_t,
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=output_layout,
            core_grid=grid_size if specify_grid else None,
            inplace=False,
            num_out_blocks=num_out_blocks if specify_grid else None,
            use_welford=use_welford,
            reciprocals=reciprocals_tensor,
        )
        ttnn.synchronize_device(device)

    if not perf_test_mode:
        output_tensor = ttnn.from_device(output_tensor)
        output_tensor = ttnn.to_torch(output_tensor)

        pcc_threshold = 0.999
        rtol = 0.060

        if use_welford:
            atol = 0.043
            frobenius_threshold = 0.01
        else:
            if specify_grid:
                atol = 0.069
                frobenius_threshold = 0.025
            else:
                # Not using Welford + auto-grid: the op also picks num_out_blocks via the
                # heuristic in the program factory, which generally differs from
                # the explicit num_out_blocks used in the specify_grid=True branch.
                # Different num_out_blocks chunks the per-block partial reductions
                # differently, producing visible bfloat16 rounding drift on the
                # tile-reduction mean/variance path.
                atol = 0.085
                frobenius_threshold = 0.030

        assert_numeric_metrics(
            torch_output_tensor,
            output_tensor,
            pcc_threshold=pcc_threshold,
            rtol=rtol,
            atol=atol,
            frobenius_threshold=frobenius_threshold,
        )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x", GROUP_NORM_DRAM_SHAPES)
@pytest.mark.parametrize("statistics_mode", STATISTICS_MODES)
def test_group_norm_DRAM(
    device,
    N,
    C,
    H,
    W,
    num_groups,
    num_out_blocks,
    cores_y,
    cores_x,
    statistics_mode,
    specify_grid=True,
    perf_test_mode=False,
):
    run_group_norm_DRAM(
        device,
        N,
        C,
        H,
        W,
        num_groups,
        num_out_blocks,
        cores_y,
        cores_x,
        statistics_mode,
        use_input_mask=True,
        perf_test_mode=perf_test_mode,
        specify_grid=specify_grid,
    )


# Post-commit smoke coverage for the ROW_MAJOR path; the full sweep over GROUP_NORM_ROW_MAJOR_SHAPES
# lives in nightly. Tests across all three layout combinations that involve ROW_MAJOR.
@skip_for_blackhole("interleaved ROW_MAJOR group_norm is Wormhole-only, see #52279")
@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "input_layout, output_layout",
    [
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT),
        (ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
        (ttnn.ROW_MAJOR_LAYOUT, ttnn.ROW_MAJOR_LAYOUT),
    ],
    ids=["RM_IN_TILE_OUT", "TILE_IN_RM_OUT", "RM_IN_RM_OUT"],
)
def test_group_norm_DRAM_row_major_smoke(device, input_layout, output_layout):
    # Issue #26594: N=1, C=480, H=1, W=64, num_groups=8 on a 1x1 grid.
    run_group_norm_DRAM(
        device,
        1,
        480,
        1,
        64,
        8,
        1,
        1,
        1,
        "tile_reduction",
        use_input_mask=True,
        input_layout=input_layout,
        output_layout=output_layout,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x", GROUP_NORM_NO_INPUT_MASK_DRAM_SHAPES
)
@pytest.mark.parametrize("statistics_mode", STATISTICS_MODES)
@pytest.mark.parametrize("specify_grid", [True])
def test_group_norm_no_input_mask_DRAM(
    device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, statistics_mode, specify_grid
):
    run_group_norm_DRAM(
        device,
        N,
        C,
        H,
        W,
        num_groups,
        num_out_blocks,
        cores_y,
        cores_x,
        statistics_mode,
        use_input_mask=False,
        specify_grid=specify_grid,
    )


def measure_group_norm_non_tile_aligned_DRAM(device, *args, **kwargs):
    # The two characterization tests below deliberately do NOT use assert_numeric_metrics: their
    # purpose is to pin a single measured max-abs number and show how it moves with the padding
    # fraction / input mean, which a multi-metric assert would obscure.
    expected, actual = run_group_norm_non_tile_aligned_DRAM(device, *args, **kwargs)
    return (actual.float() - expected.float()).abs().max().item()


def run_group_norm_non_tile_aligned_DRAM(
    device,
    N,
    C,
    H,
    W,
    num_groups,
    use_welford,
    use_input_mask=True,
    input_shift=0.0,
    cores_y=None,
    cores_x=None,
    num_out_blocks=None,
):
    # Shared body for the non-tile-aligned regressions on the fused interleaved path.
    #   input_shift  leaves the reference unchanged (shift-invariance) but scales mean^2/variance.
    #   cores_y/_x   pin the grid (mcast vs no_mcast via batch >= num_virtual_rows) and allow an
    #                explicit num_out_blocks; omit to exercise the C++ heuristics.
    # Also used with an aligned shape as a control, so alignment is asserted by callers.
    torch.manual_seed(0)

    specify_grid = cores_y is not None and cores_x is not None
    if specify_grid:
        grid_for_params = ttnn.CoreGrid(y=cores_y, x=cores_x)
    else:
        grid_for_params = ttnn.determine_expected_group_norm_dram_grid_size(
            device=device,
            num_channels=C,
            num_groups=num_groups,
            input_nhw=N * H * W,
            num_batches=N,
        )

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16) + input_shift
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor.float(), num_groups, weight=torch_weight.float(), bias=torch_bias.float(), eps=1e-12
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # tilize_with_zero_padding is what guarantees the precondition the correction relies on:
    # the P - L tile-padding rows hold exact zeros.
    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_row_major, use_multicore=True)

    if use_input_mask:
        [gamma_t, beta_t], input_mask_tensor = ttnn.dram_group_norm_params_from_torch(
            [torch_weight, torch_bias], C, num_groups, device, core_grid=grid_for_params, return_mask=True
        )
    else:
        [gamma_t, beta_t] = ttnn.dram_group_norm_params_from_torch(
            [torch_weight, torch_bias], C, num_groups, device, core_grid=grid_for_params, return_mask=False
        )
        input_mask_tensor = None

    output_tensor = ttnn.group_norm(
        input_tensor_tilized,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_for_params if specify_grid else None,
        inplace=False,
        num_out_blocks=num_out_blocks,
        use_welford=use_welford,
    )
    ttnn.synchronize_device(device)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))

    return torch_output_tensor, output_tensor


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", GROUP_NORM_NON_TILE_ALIGNED_DRAM_SHAPES)
@pytest.mark.parametrize("use_welford", [False, True], ids=["tile_reduction", "online_welford"])
def test_group_norm_non_tile_aligned_DRAM(device, N, C, H, W, num_groups, use_welford):
    # Fused interleaved group_norm must match torch even when N*H*W is not a multiple of the tile
    # height. use_welford=True is routed to the two-pass path; auto grid selection.
    if device.core_grid.y == 7:
        pytest.skip()

    expected, actual = run_group_norm_non_tile_aligned_DRAM(device, N, C, H, W, num_groups, use_welford)
    # rtol is left at its default rather than the 0.060 used for the tile-aligned cases: this bug
    # was a per-group affine drift, so PCC cannot see it and a value-proportional tolerance would
    # partly absorb it. atol is the discriminator here (pre-fix this shape measured ~0.36).
    assert_numeric_metrics(expected, actual, atol=NON_TILE_ALIGNED_ATOL, frobenius_threshold=0.03)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", GROUP_NORM_NON_TILE_ALIGNED_PER_SAMPLE_DRAM_SHAPES)
def test_group_norm_non_tile_aligned_per_sample_DRAM(device, N, C, H, W, num_groups):
    # N > 1: the correction keys off the PER-SAMPLE height (logical_shape()[2]), not N*H*W, so
    # H*W=80 pads to 96 even though N*H*W=160 is aligned. tt-mlir #8935 gates on the same quantity.
    if device.core_grid.y == 7:
        pytest.skip()

    assert (H * W) % 32 != 0, "per-sample H*W must be non-tile-aligned"
    assert (N * H * W) % 32 == 0, "N*H*W should look aligned -- that is the point of this case"

    expected, actual = run_group_norm_non_tile_aligned_DRAM(device, N, C, H, W, num_groups, use_welford=False)
    assert_numeric_metrics(expected, actual, atol=NON_TILE_ALIGNED_ATOL, frobenius_threshold=0.03)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
def test_group_norm_non_tile_aligned_program_cache_aliasing_DRAM(device):
    # The interleaved paths ship the scaler and K as compile-time args, so H*W=200 and H*W=224 --
    # same padded 224, different logical -- must not share a cached program, or one of them runs the
    # other's divisor. Safe today because the default program hash keys on logical_shape. The sharded
    # path needs no equivalent: runtime args, and its own tests already cover a cache hit.
    if device.core_grid.y == 7:
        pytest.skip()

    C, num_groups = 1024, 32
    non_aligned_hw, aligned_hw = 200, 224  # both pad to 224
    assert non_aligned_hw % 32 != 0 and aligned_hw % 32 == 0
    assert ((non_aligned_hw + 31) // 32) * 32 == aligned_hw, "the two shapes must share a padded_hw"

    # Non-aligned, aligned, non-aligned. Only the change across calls is meaningful: the helper's
    # tilize and param-prep ops add entries of their own.
    entries = []
    for call_index, hw in enumerate((non_aligned_hw, aligned_hw, non_aligned_hw)):
        expected, actual = run_group_norm_non_tile_aligned_DRAM(device, 1, C, 1, hw, num_groups, use_welford=False)
        passed, message = assert_numeric_metrics(
            expected,
            actual,
            atol=NON_TILE_ALIGNED_ATOL,
            frobenius_threshold=0.03,
            assert_on_fail=False,
        )
        assert passed, f"call {call_index} (H*W={hw}) wrong -- a shared cache entry does exactly this: {message}"
        entries.append(device.num_program_cache_entries())

    assert entries[1] > entries[0], (
        f"H*W={aligned_hw} added no program-cache entry over H*W={non_aligned_hw} "
        f"({entries[0]} -> {entries[1]}); the two logical shapes aliased onto one program"
    )
    # A hit here, plus call 3's numeric check above, is the proof: it reused call 1's entry and the
    # aligned call between them did not poison it.
    assert entries[2] == entries[1], (
        f"repeating H*W={non_aligned_hw} added {entries[2] - entries[1]} entries "
        f"({entries[1]} -> {entries[2]}); expected a program-cache hit"
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x", GROUP_NORM_NON_TILE_ALIGNED_GRID_DRAM_CASES
)
def test_group_norm_non_tile_aligned_explicit_grid_DRAM(
    device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x
):
    # Paths the auto-grid tests never reach:
    #  * num_out_blocks > 1 -- the padding tail lands in the LAST chunk (extra_out_block /
    #    out_block_h_last). The auto heuristic picks 1 here, so padding never meets the chunking.
    #  * batch >= num_virtual_rows -- selects no_mcast over mcast. no_mcast also has a SECOND core
    #    group (when num_batches does not divide the shard rows evenly) with its own scaler and
    #    compute args; N=9 on an 8-row grid is the uneven-batch case that reaches it.
    if device.core_grid.y == 7:
        pytest.skip()
    if device.core_grid.x < cores_x or device.core_grid.y < cores_y:
        pytest.skip(f"device grid too small for {cores_x}x{cores_y}")

    assert (H * W) % 32 != 0, "per-sample H*W must be non-tile-aligned"

    expected, actual = run_group_norm_non_tile_aligned_DRAM(
        device,
        N,
        C,
        H,
        W,
        num_groups,
        use_welford=False,
        cores_y=cores_y,
        cores_x=cores_x,
        num_out_blocks=num_out_blocks,
    )
    # Chunking into num_out_blocks changes how the per-block partial sums round, so use the same
    # relaxed atol run_group_norm_DRAM already allows for that on the aligned path.
    assert_numeric_metrics(expected, actual, atol=0.085, frobenius_threshold=0.03)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups, K, max_err", GROUP_NORM_NON_TILE_ALIGNED_PADDING_FRACTION_CASES)
def test_group_norm_non_tile_aligned_padding_fraction_DRAM(device, N, C, H, W, num_groups, K, max_err):
    # How the correction degrades as the padding fraction grows: it is exact in exact
    # arithmetic, and what degrades is the bfloat16 cancellation in Var - K*E[x]^2, which scales
    # with K. Real shapes sit at K <= 0.6 (XTTS-v2 H*W=269, K=0.07) and stay inside 0.08. K >= 1
    # happens only for H*W <= 16 and is recorded as a known limit, not a target.
    if device.core_grid.y == 7:
        pytest.skip()

    max_abs_err = measure_group_norm_non_tile_aligned_DRAM(device, N, C, H, W, num_groups, use_welford=False)
    logger.info(f"padding-fraction K={K} H*W={W * H} max_abs_err={max_abs_err}")
    assert max_abs_err < max_err, (
        f"max abs error {max_abs_err} exceeds the recorded {max_err} at K={K} (H*W={W * H}); the "
        f"analytical padding correction has degraded further than measured (see #50682)"
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", GROUP_NORM_NON_TILE_ALIGNED_DRAM_SHAPES[:1])
def test_group_norm_non_tile_aligned_no_input_mask_DRAM(device, N, C, H, W, num_groups):
    # No-input-mask variant: without the channel mask the compute kernel skips the "zero out
    # garbage by mult mask again" step, so this checks the padding-row correction is independent
    # of channel masking.
    if device.core_grid.y == 7:
        pytest.skip()

    expected, actual = run_group_norm_non_tile_aligned_DRAM(
        device, N, C, H, W, num_groups, use_welford=False, use_input_mask=False
    )
    assert_numeric_metrics(expected, actual, atol=NON_TILE_ALIGNED_ATOL, frobenius_threshold=0.03)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("input_shift, max_err, aligned_ref, max_ratio", GROUP_NORM_NON_TILE_ALIGNED_SHIFT_CASES)
def test_group_norm_non_tile_aligned_shifted_input_DRAM(device, input_shift, max_err, aligned_ref, max_ratio):
    # Accuracy limit of the correction vs the input mean. group_norm is shift-invariant so
    # the output must not change, but the kernel's internals do: subtracting K*E[x]^2 from a bf16
    # variance makes the relative error grow like 2^-8 * (1 + 2*K*r) with r = mean^2/variance.
    # The TILE-ALIGNED H*W=224 control is measured alongside because the fused bf16 kernel also
    # degrades with a large mean unpadded (~0.158 at shift=4), so the real cost is the gap.
    if device.core_grid.y == 7:
        pytest.skip()

    C, num_groups = 1024, 32
    measured = measure_group_norm_non_tile_aligned_DRAM(
        device, 1, C, 1, 200, num_groups, use_welford=False, input_shift=input_shift
    )
    if aligned_ref is not None:
        # H*W=224 is tile-aligned, so has_pad_correction is false and this is the op's own
        # bfloat16 baseline at the same mean-to-spread ratio.
        aligned = measure_group_norm_non_tile_aligned_DRAM(
            device, 1, C, 1, 224, num_groups, use_welford=False, input_shift=input_shift
        )
        logger.info(
            f"input_shift={input_shift} non-aligned(H*W=200)={measured} "
            f"tile-aligned control(H*W=224)={aligned} recorded_control={aligned_ref} "
            f"ratio={measured / aligned if aligned > 0 else float('nan')}"
        )
        # If the control has drifted far from what was recorded, the ratio below stops measuring
        # the correction and starts measuring a change in the fused kernel -- re-take this table.
        assert aligned < 2.0 * aligned_ref, (
            f"tile-aligned control {aligned} at input_shift={input_shift} has regressed past 2x the "
            f"recorded {aligned_ref}; the baseline moved, so re-measure this table (see #50682)"
        )
        # The point of this test: bound the correction's cost against that baseline, not in absolute
        # terms. Unlike max_err this does not have to be re-tuned per arch.
        assert measured < max_ratio * aligned, (
            f"non-aligned error {measured} is more than {max_ratio}x the tile-aligned control "
            f"{aligned} at input_shift={input_shift}; the analytical padding correction has become "
            f"more expensive relative to the op's own bfloat16 baseline (see #50682)"
        )
    else:
        logger.info(f"input_shift={input_shift} non-aligned(H*W=200) max_abs_err={measured}")

    assert measured < max_err, (
        f"max abs error {measured} exceeds the recorded {max_err} at input_shift={input_shift}; the "
        f"analytical padding correction has degraded further than measured (see #50682)"
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
def test_group_norm_non_tile_aligned_garbage_padding_DRAM(device):
    # Was a strict xfail while group_norm reduced over the tile padding (measured ~1.92 vs ~0.045);
    # the padding rows are masked out now, so it is a plain test.
    if device.core_grid.y == 7:
        pytest.skip()

    torch.manual_seed(0)
    N, C, HW, G, padded = 1, 1024, 200, 32, 224

    real = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)
    ref = torch.nn.functional.group_norm(real.view(N, HW, C).permute(0, 2, 1).reshape(N, C, 1, HW).float(), G)
    ref = ref.permute(0, 2, 3, 1).reshape(N, 1, HW, C)

    buf = torch.zeros((N, 1, padded, C), dtype=torch.bfloat16)
    buf[:, :, :HW, :] = real
    buf[:, :, HW:, :] = 7.0  # anything non-zero

    tt = ttnn.from_torch(
        buf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    # Reinterpret as 200 logical rows over 224 padded rows, preserving the padding content.
    tt = ttnn.reshape(tt, ttnn.Shape([N, 1, HW, C]), ttnn.Shape([N, 1, padded, C]))
    out = ttnn.group_norm(tt, num_groups=G, inplace=False, memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=None)
    out = ttnn.to_torch(ttnn.from_device(out)).float()[:, :, :HW, :]

    max_abs_err = (out - ref).abs().max().item()
    logger.info(f"garbage tile-padding rows: max_abs_err={max_abs_err}")
    assert max_abs_err < 0.08, f"max abs error {max_abs_err} with garbage padding rows (see #50682)"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "N, cores_y, num_out_blocks",
    [
        # The grid fixes num_virtual_rows, hence whether a batch is split across core rows and so
        # which cores hold the padding row-tile.
        pytest.param(1, 1, None, id="N1_grid1x8_whole_batch_per_core"),
        pytest.param(1, 2, None, id="N1_grid2x8_batch_split_2"),
        pytest.param(1, 4, None, id="N1_grid4x8_batch_split_4_block_h_1"),
        pytest.param(2, 1, None, id="N2_grid1x8_two_batches_per_core"),
        pytest.param(2, 4, None, id="N2_grid4x8_batch_split_2"),
        # num_out_blocks=3 over block_h=4 leaves the last out-block empty -- the case that breaks a
        # naive "last block, last row" test.
        pytest.param(1, 1, 3, id="N1_grid1x8_num_out_blocks_3_empty_last_block"),
        pytest.param(1, 1, 4, id="N1_grid1x8_num_out_blocks_4"),
    ],
)
def test_group_norm_non_tile_aligned_dirty_padding_grids_DRAM(device, N, cores_y, num_out_blocks):
    # Which core applies the row mask depends on how the grid splits H*W, and with N > 1 the padding
    # recurs once per batch on the same core -- a single auto-selected grid reaches neither. Also
    # pins the out-block indexing: num_out_blocks not dividing block_h can leave the last out-block
    # with zero rows, so the final row-tile is found by its global index within the batch.
    cores_x = 8
    if device.core_grid.y < cores_y or device.core_grid.x < cores_x:
        pytest.skip(f"device grid too small for {cores_x}x{cores_y}")

    torch.manual_seed(0)
    C, HW, G, padded = 512, 100, 32, 128

    real = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)
    ref = torch.nn.functional.group_norm(real.view(N, HW, C).permute(0, 2, 1).reshape(N, C, 1, HW).float(), G)
    ref = ref.permute(0, 2, 3, 1).reshape(N, 1, HW, C)

    buf = torch.zeros((N, 1, padded, C), dtype=torch.bfloat16)
    buf[:, :, :HW, :] = real
    buf[:, :, HW:, :] = 7.0

    tt = ttnn.from_torch(
        buf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt = ttnn.reshape(tt, ttnn.Shape([N, 1, HW, C]), ttnn.Shape([N, 1, padded, C]))
    out = ttnn.group_norm(
        tt,
        num_groups=G,
        inplace=False,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=ttnn.CoreGrid(y=cores_y, x=cores_x),
        num_out_blocks=num_out_blocks,
    )
    out = ttnn.to_torch(ttnn.from_device(out)).float()[:, :, :HW, :]

    max_abs_err = (out - ref).abs().max().item()
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    logger.info(f"N={N} grid={cores_x}x{cores_y} num_out_blocks={num_out_blocks}: max_abs_err={max_abs_err} pcc={pcc}")
    assert max_abs_err < 0.08, (
        f"max abs error {max_abs_err} with dirty tile padding at N={N} grid={cores_x}x{cores_y} "
        f"num_out_blocks={num_out_blocks}; group_norm must be independent of its padding (see #52685)"
    )
    assert pcc > 0.999, f"pcc {pcc} with dirty tile padding (see #52685)"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
def test_group_norm_non_tile_aligned_dirty_padding_is_ignored_DRAM(device):
    # The same logical rows with different padding bytes must give bit-identical output -- unlike an
    # error threshold, this cannot be satisfied by merely keeping the damage small.
    torch.manual_seed(0)
    N, C, HW, G, padded = 1, 512, 100, 32, 128

    real = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)

    def run(padding_value):
        buf = torch.zeros((N, 1, padded, C), dtype=torch.bfloat16)
        buf[:, :, :HW, :] = real
        buf[:, :, HW:, :] = padding_value
        tt = ttnn.from_torch(
            buf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        tt = ttnn.reshape(tt, ttnn.Shape([N, 1, HW, C]), ttnn.Shape([N, 1, padded, C]))
        out = ttnn.group_norm(tt, num_groups=G, inplace=False, memory_config=ttnn.DRAM_MEMORY_CONFIG, core_grid=None)
        return ttnn.to_torch(ttnn.from_device(out)).float()[:, :, :HW, :]

    baseline = run(0.0)
    for padding_value in (7.0, 1.0, -3.5, 0.5, 100.0):
        other = run(padding_value)
        assert torch.equal(baseline, other), (
            f"group_norm output changed when the tile padding changed from 0.0 to {padding_value} "
            f"(max delta {(baseline - other).abs().max().item()}); the padding rows must not reach "
            f"either accumulation pass (see #52685)"
        )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups, num_splits", SDXL_BASE_GROUP_NORM_SPLIT_SHAPES)
@pytest.mark.parametrize("specify_grid", [True])
def test_sdxl_base_group_norm_split(device, N, C, H, W, num_groups, num_splits, specify_grid):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    if (
        is_blackhole()
        and is_watcher_enabled()
        and N == 1
        and H == 512
        and W == 512
        and num_groups == 32
        and (C, num_splits) in [(256, 8), (512, 16)]
    ):
        pytest.skip("Skipping test on Blackhole with watcher enabled, see issue #37645")

    grid_size = ttnn.CoreGrid(y=8, x=8)

    # Generate torch tensor
    torch_input_tensor = torch.rand([N, C, H, W], dtype=torch.bfloat16)

    # Execute torch group_norm
    torch_output_tensor = torch.nn.functional.group_norm(torch_input_tensor, num_groups)
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate ttnn tensor
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        tt_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Generate input mask
    num_groups_per_split = num_groups // num_splits  # 16
    C_per_split = C // num_splits
    input_mask_tensor = ttnn.create_group_norm_input_mask(C_per_split, num_groups_per_split, 1, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    input_negative_mask_tensor = ttnn.create_group_norm_input_negative_mask(
        C_per_split, num_groups_per_split, 1, ttnn.DataType.BFLOAT8_B
    )
    input_negative_mask_tensor = ttnn.to_device(input_negative_mask_tensor, device)

    tt_input_tensor = ttnn.to_device(tt_input_tensor, device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # Generate shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // (grid_size.y * grid_size.x), C_per_split
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config_per_split = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    per_group_results = []
    for i in range(num_splits):
        tt_input_tensor_per_group_split = tt_input_tensor[:, :, :, i * C_per_split : (i + 1) * C_per_split]
        tt_input_tensor_per_group_split_sharded = ttnn.to_memory_config(
            tt_input_tensor_per_group_split, sharded_mem_config_per_split
        )
        tt_output_tensor_per_group_split = ttnn.group_norm(
            tt_input_tensor_per_group_split_sharded,
            num_groups=num_groups_per_split,
            input_mask=input_mask_tensor,
            memory_config=sharded_mem_config_per_split,
            core_grid=grid_size if specify_grid else None,
            inplace=False,
            negative_mask=input_negative_mask_tensor,
        )
        tt_output_tensor_per_group_split = ttnn.to_memory_config(
            tt_output_tensor_per_group_split, ttnn.DRAM_MEMORY_CONFIG
        )
        per_group_results.append(tt_output_tensor_per_group_split)

    tt_output_tensor = ttnn.concat(per_group_results, dim=-1)

    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        tt_output_tensor,
        pcc_threshold=0.999,
        rtol=10.519,
        atol=0.086,
        frobenius_threshold=0.043,
    )


def _nearest_32_per_core(x, core):
    return math.ceil(x / core / 32) * 32 * core


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, eps", GROUP_NORM_DRAM_OFT_PARAMS)
@pytest.mark.parametrize("specify_grid", [True])
@run_for_blackhole("blackhole specific tests")
def test_group_norm_DRAM_oft(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, eps, specify_grid):
    skip_if_not_blackhole_20_cores(device)
    torch.manual_seed(0)
    grid_size = ttnn.CoreGrid(y=cores_y, x=cores_x)
    if specify_grid:
        # Use the explicit user-chosen grid.
        grid_for_params = grid_size
    else:
        # Exercises the C++ automatic grid selection. Input padding, mask, and
        # gamma/beta must be prepared for that *same* auto-selected grid;
        # otherwise their shapes would not match the grid the op picks at runtime.
        grid_for_params = ttnn.determine_expected_group_norm_dram_grid_size(
            device=device,
            num_channels=C,
            num_groups=num_groups,
            input_nhw=N * H * W,
            num_batches=N,
        )
    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=eps
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_row_major = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    unpadded_shape = input_tensor_row_major.shape
    out_shape = [
        unpadded_shape[0],
        unpadded_shape[1],
        _nearest_32_per_core(unpadded_shape[2], grid_for_params.x),
        _nearest_32_per_core(unpadded_shape[3], grid_for_params.y),
    ]

    input_tensor_tilized = ttnn.tilize_with_val_padding(
        input_tensor_row_major, output_tensor_shape=out_shape, pad_value=0, use_multicore=True
    )

    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_for_params.y, ttnn.DataType.BFLOAT16)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)
    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_for_params.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_for_params.y)
    gamma_t = ttnn.from_torch(
        gamma,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    beta_t = ttnn.from_torch(
        beta,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    output_tensor = ttnn.group_norm(
        input_tensor_tilized,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        output_layout=ttnn.TILE_LAYOUT,
        core_grid=grid_size if specify_grid else None,
        inplace=False,
        # num_out_blocks must be omitted when core_grid is auto-selected; the op
        # picks a heuristic value internally in that case.
        num_out_blocks=num_out_blocks if specify_grid else None,
        epsilon=eps,
    )

    ttnn.synchronize_device(device)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor[:, :, : H * W, :C],
        pcc_threshold=0.9994,
        rtol=0.09,
        atol=0.09,
        frobenius_threshold=0.03,
    )


GN_INTERLEAVED_SHAPES = [
    (1, 320, 32, 32, 16, 1, 1, 8),  # base config (original single-shape test)
    (1, 480, 1, 64, 8, 1, 1, 1),  # single core, last group ends less than max tile span
    (1, 768, 1, 512, 32, 2, 8, 8),  # group channel count less than tile size
    (2, 768, 1, 512, 32, 2, 8, 8),  # batch 2 (still multicast), num_out_blocks 2
    (1, 2560, 1, 512, 32, 2, 8, 8),  # mcast num_out_blocks 2
    (1, 128, 1, 512, 32, 2, 4, 4),  # all groups on core fit in less than one tile
    (8, 768, 1, 512, 32, 3, 8, 8),  # batch 8 (no multicast), uneven num_out_blocks divisor
]


@pytest.mark.parametrize("N, C, H, W, num_groups, num_out_blocks, grid_y, grid_x", GN_INTERLEAVED_SHAPES)
# Cover every (statistics mode x input dtype) pair. Input dtype affects the accuracy thresholds and
# FP32 Welford buffer configuration; gamma/beta dtype only selects the parameter CB format, so it is
# varied across the matrix instead of introducing another Cartesian-product axis.
@pytest.mark.parametrize(
    "statistics_mode, in_dtype, gb_dtype",
    [
        ("tile_reduction", ttnn.bfloat16, ttnn.bfloat16),
        ("tile_reduction", ttnn.bfloat16, ttnn.float32),
        ("tile_reduction", ttnn.float32, ttnn.float32),
        ("tile_reduction", ttnn.float32, ttnn.bfloat16),
        ("online_welford", ttnn.bfloat16, ttnn.float32),
        ("online_welford", ttnn.float32, ttnn.bfloat16),
        ("online_welford_reciprocal_lut", ttnn.bfloat16, ttnn.bfloat16),
        ("online_welford_reciprocal_lut", ttnn.float32, ttnn.float32),
    ],
    ids=[
        "tile_reduction-bf16-gb_bf16",
        "tile_reduction-bf16-gb_fp32",
        "tile_reduction-fp32-gb_fp32",
        "tile_reduction-fp32-gb_bf16",
        "online_welford-bf16-gb_fp32",
        "online_welford-fp32-gb_bf16",
        "online_welford_reciprocal_lut-bf16-gb_bf16",
        "online_welford_reciprocal_lut-fp32-gb_fp32",
    ],
)
def test_group_norm_interleaved_all_config(
    device, N, C, H, W, num_groups, num_out_blocks, grid_y, grid_x, in_dtype, gb_dtype, statistics_mode
):
    # Interleaved (DRAM) group_norm across all statistics modes. The modes differ only in
    # use_welford/use_reciprocals and the accuracy thresholds; everything else (fp32/bf16 input,
    # fp32/bf16 gamma-beta, gamma/beta/mask prep via dram_group_norm_params_from_torch) is identical.
    # Interleaved input/output is TILE-only (ROW_MAJOR is rejected by the op for non-sharded tensors).
    grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
    torch.manual_seed(0)

    # Determine welford and reciprocals settings
    use_welford = statistics_mode in ("online_welford", "online_welford_reciprocal_lut")
    use_reciprocals = statistics_mode == "online_welford_reciprocal_lut"

    x = torch.rand((N, C, H, W), dtype=torch.float32)
    w = torch.rand((C,), dtype=torch.float32)
    b = torch.rand((C,), dtype=torch.float32)
    ref = torch.nn.functional.group_norm(x, num_groups, weight=w, bias=b).permute(0, 2, 3, 1).view(N, 1, W * H, C)

    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # required for FP32 on either statistics backend
        packer_l1_acc=False,
    )

    xt = x.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    xt = ttnn.from_torch(
        xt, dtype=in_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    [gt, bt], mask = ttnn.dram_group_norm_params_from_torch(
        [w, b], C, num_groups, device, core_grid=grid, return_mask=True, dtype=gb_dtype
    )

    # Create reciprocals tensor if needed (host-precomputed 1/count fed via the reciprocals= arg)
    reciprocals_tensor = None
    if use_reciprocals:
        torch_reciprocals = ttnn.create_group_norm_reciprocals(N, C, H, W, num_groups, grid)
        reciprocals_tensor = ttnn.from_torch(
            torch_reciprocals,
            dtype=ttnn.DataType.FLOAT32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                buffer_type=ttnn.BufferType.L1,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))}),
                    (torch_reciprocals.shape[0] // (grid.x * grid.y), torch_reciprocals.shape[1]),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
        )

    out = ttnn.group_norm(
        xt,
        num_groups=num_groups,
        input_mask=mask,
        weight=gt,
        bias=bt,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=grid,
        dtype=in_dtype,
        compute_kernel_config=ck,
        use_welford=use_welford,
        num_out_blocks=num_out_blocks,
        inplace=False,
        reciprocals=reciprocals_tensor,
    )
    out = ttnn.to_torch(ttnn.from_device(out)).float().reshape(ref.shape)

    # Thresholds branch on the reduction path and the input dtype (bf16 input is the dominant error
    # source); each bound sits ~1.4x above the worst observed value across the shape/gamma-beta matrix.
    if use_welford:
        if in_dtype == ttnn.bfloat16:
            pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.01, 0.06, 0.015
        else:
            pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.008, 0.02, 0.004
    else:
        if in_dtype == ttnn.bfloat16:
            pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.01, 0.10, 0.03
        else:
            pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.008, 0.03, 0.01
    assert_numeric_metrics(
        ref,
        out,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )

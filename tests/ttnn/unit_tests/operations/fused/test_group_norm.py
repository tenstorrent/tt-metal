# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from models.common.utility_functions import run_for_blackhole, is_wormhole_b0
from tests.ttnn.unit_tests.base_functionality.test_bh_20_cores_sharding import skip_if_not_blackhole_20_cores
from tests.ttnn.utils_for_testing import assert_numeric_metrics


welford_flavors, welford_ids = (True, False), ("welford", "legacy")

TEST_PADDING_VALUE = -42

DEVICE_PARAMS_L1_SMALL_SIZE = [{"l1_small_size": 0}]
DEVICE_PARAMS_L1_SMALL_SIZE_SDXL_BG_N_MASK = [{"l1_small_size": 47000}]

HEIGHT_SHARDED_SHAPES = [
    (1, 320, 32, 32, 16),
]

# Non-tile-aligned N*H*W on the sharded two-pass path (#50682). Single-core height sharding keeps
# the whole padded height, and its padding tail, on one core.
# (N, C, H, W, num_groups)
HEIGHT_SHARDED_NON_TILE_ALIGNED_SHAPES = [
    (1, 128, 1, 200, 32),  # H*W=200 -> padded 224 (10.7% padding)
    (1, 256, 1, 100, 32),  # H*W=100 -> padded 128 (21.9% padding)
]

# Block-sharded (v2) non-tile-aligned cases. grid_x splits the padded H*W tile count and
# must divide it; grid_y splits channels. H*W=100 -> padded 128 = 4 tiles, so grid_x in {1, 2, 4}
# keeps each M-shard tile-aligned. grid_x > 1 exercises the multi-core M-split, where the padding
# tail lands on the last M-core only.
# (N, C, H, W, num_groups, grid_y, grid_x)
BLOCK_SHARDED_NON_TILE_ALIGNED_CASES = [
    (1, 256, 1, 100, 32, 2, 1),  # channel split only, single M-core
    (1, 256, 1, 100, 32, 2, 2),  # multi-core M-split: 4 tiles across 2 x-cores, padding on last
]

# GroupNorm coverage shapes for the fp32/bf16 sharded all-config tests. Shapes are
# (N, C, H, W, num_groups, grid_y, grid_x) where grid_y == 1 is height-sharded and grid_y > 1 is
# block-sharded. Chosen to cover single-core, sub-tile group widths, and batch > 1.
# (Interleaved/DRAM coverage lives in test_group_norm_DRAM.py::GN_INTERLEAVED_SHAPES.)
GN_SHARDED_SHAPES = [
    (1, 320, 32, 32, 16, 1, 8),  # base config (original single-shape test), height-sharded
    (1, 256, 1, 256, 16, 1, 1),  # single core height-sharded, sub-tile group width (16 ch/group)
    (1, 128, 1, 512, 16, 1, 4),  # height-sharded, groups on core fit in less than one tile
    #   (num_groups <= 16 per core is required by the welford sharded path)
    (1, 1280, 1, 512, 32, 8, 8),  # block-sharded 8x8
    (2, 512, 32, 32, 32, 8, 8),  # block-sharded 8x8, batch 2 (C/grid_y = 64, tile-aligned)
    (1, 1280, 16, 16, 32, 4, 8),  # block-sharded 8x4
]

BLOCK_SHARDED_V2_8X4_SHAPES = [
    (1, 1280, 16, 16, 32),
    (1, 320, 1, 8192, 32),
    (1, 960, 1, 1024, 32),
    # not fit in L1 for GS
    # (1, 960, 1, 4096, 32),
]

BLOCK_SHARDED_V2_8X8_SHAPES = [
    (2, 320, 64, 64, 32),
    (1, 640, 1, 2048, 32),
    (1, 640, 1, 4096, 32),
    (1, 960, 1, 2048, 32),
    (1, 960, 1, 4096, 32),
    (1, 1280, 1, 512, 32),
    (1, 1280, 1, 2048, 32),
    (1, 1920, 1, 512, 32),
    (1, 1920, 1, 2048, 32),
    (1, 2560, 1, 512, 32),
    # not fit in L1 for GS
    # (2, 960, 64, 64, 32),
    # (1, 640, 1, 8192, 32),
]

BLOCK_SHARDED_V2_8X8_TILE_LAYOUT_SHAPES = [
    (1, 1280, 1, 512, 32),
    (1, 1280, 1, 2048, 32),
    (1, 2560, 1, 512, 32),
]

SDXL_BASE_GROUP_NORM_BH_SHAPES = [
    # UNet
    (1, 1280, 64, 64),
    (1, 1280, 32, 32),
    (1, 1920, 64, 64),
    (1, 1920, 32, 32),
    (1, 2560, 32, 32),
    (1, 320, 128, 128),
    (1, 320, 64, 64),
    (1, 640, 64, 64),
    (1, 640, 32, 32),
    (1, 960, 64, 64),
    # VAE
    (1, 512, 128, 128),
]

COMPUTE_CONFIG_SHAPES = [
    (1, 1920, 64, 64, 32),
]

GROUP_NORM_OFT_PARAMS = [
    (1, 256, 12, 40, 16, "BS", 1e-5, False),
    (1, 256, 24, 80, 16, "HS", 1e-5, False),
    (1, 256, 48, 160, 16, "HS", 1e-5, False),
    (1, 512, 12, 40, 16, "BS", 1e-5, False),
    (1, 64, 96, 320, 16, "HS", 1e-5, False),
    (1, 32, 192, 640, 8, "HS", 1e-5, True),  # half of (1, 64, 192, 640, 16, 10, 2, 4, 1e-5),
]

NO_INPUT_MASK_SHAPES = [
    (1, 256, 64, 64, 32),
]

DRAM_GRID_SIZE_SHAPES = [
    (1, 480, 8, 8, 16),
    (1, 320, 32, 32, 32),
    (1, 1280, 16, 16, 32),
]

OPTIONAL_WEIGHT_BIAS_SHAPES = [
    (1, 128, 64, 1, 32),
]

OPTIONAL_WEIGHT_BIAS_AFFINE_PARAMS = [
    (False, False),
    (True, False),
    (False, True),
]
OPTIONAL_WEIGHT_BIAS_AFFINE_IDS = ["no_affine", "weight_only", "bias_only"]

# ttnn.empty produces a ROW_MAJOR interleaved input. On Blackhole the non-sharded path
# rejects that up front; on Wormhole ROW_MAJOR is allowed, so the same shape reaches
# the device-op tile-height check instead.
_NEGATIVE_TEST_MSG = (
    "must be a multiple of the tile height"
    if is_wormhole_b0()
    else "interleaved \\(non-sharded\\) input must be in TILE layout"
)

NEGATIVE_TESTS_PARAMS = [
    ((2, 1, 16, 32), 8, _NEGATIVE_TEST_MSG),
]


# for debug purpose
def manual_group_norm(input_tensor, num_groups, eps=1e-2):
    N, C, H, W = input_tensor.shape
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"

    # Reshape into groups
    group_channels = C // num_groups
    input_tensor = input_tensor.view(N, num_groups, group_channels, H, W)

    # Calculate mean and variance
    mean = input_tensor.mean(dim=(2, 3, 4), keepdim=True)
    var = input_tensor.var(dim=(2, 3, 4), keepdim=True)

    # Normalize
    input_tensor = (input_tensor - mean) / torch.sqrt(var + eps)

    # Reshape back to original dimensions
    input_tensor = input_tensor.view(N, C, H, W)
    return input_tensor


@run_for_blackhole("low-variance stable-statistics regression is calibrated on Blackhole")
@pytest.mark.parametrize("base, amplitude", [(1.0, 0.01), (10.0, 0.05)])
@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
def test_group_norm_stable_stats_translation_stability(device, base, amplitude):
    torch.manual_seed(7)
    N, C, H, W, num_groups = 1, 1280, 32, 32, 32
    grid = ttnn.CoreGrid(y=8, x=8)

    torch_input = (base + amplitude * torch.randn((N, C, H, W))).to(torch.bfloat16)
    reference = torch.nn.functional.group_norm(torch_input.float(), num_groups, eps=1e-12)
    reference = reference.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)
    nhwc = torch_input.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)
    memory_config = ttnn.create_sharded_memory_config(
        shape=nhwc.shape,
        core_grid=grid,
        strategy=ttnn.ShardStrategy.BLOCK,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
    )
    input_tensor = ttnn.from_torch(
        nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
        device=device,
    )
    input_mask = ttnn.create_group_norm_input_mask(C, num_groups, grid.x, ttnn.bfloat8_b)
    input_mask = ttnn.to_device(input_mask, device)

    output = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        epsilon=1e-12,
        input_mask=input_mask,
        memory_config=memory_config,
        core_grid=grid,
        inplace=False,
        use_welford=True,
    )
    output = ttnn.to_torch(ttnn.from_device(output)).float()

    assert torch.isfinite(output).all()
    rmse = torch.mean((output - reference) ** 2).sqrt().item()
    pcc = torch.corrcoef(torch.stack((output.flatten(), reference.flatten())))[0, 1].item()
    assert rmse < 0.06
    assert pcc > 0.9995


@run_for_blackhole("SFPU global GroupNorm combine is enabled on Blackhole")
@pytest.mark.parametrize(
    "grid_size, spatial_shape",
    [
        (ttnn.CoreGrid(y=1, x=3), (24, 32)),
        (ttnn.CoreGrid(y=1, x=8), (32, 32)),
        (ttnn.CoreGrid(y=2, x=8), (32, 32)),
        (ttnn.CoreGrid(y=4, x=8), (32, 32)),
    ],
    ids=["3-cores", "8-cores", "16-cores", "32-cores"],
)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
def test_group_norm_sfpu_global_combine_core_counts(device, grid_size, spatial_shape):
    torch.manual_seed(11)
    N, C, num_groups = 1, 320, 16
    H, W = spatial_shape
    num_cores = grid_size.x * grid_size.y

    torch_input = torch.randn((N, C, H, W), dtype=torch.bfloat16)
    reference = torch.nn.functional.group_norm(torch_input.float(), num_groups)
    reference = reference.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)
    nhwc = torch_input.permute(0, 2, 3, 1).reshape(N, 1, H * W, C)

    shard_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))}
    )
    shard_spec = ttnn.ShardSpec(
        shard_grid,
        [N * H * W // num_cores, C],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    input_tensor = ttnn.from_torch(
        nhwc,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=memory_config,
        device=device,
    )
    input_mask = ttnn.create_group_norm_input_mask(C, num_groups, 1, ttnn.bfloat8_b)
    input_mask = ttnn.to_device(input_mask, device)

    output = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask,
        memory_config=memory_config,
        core_grid=grid_size,
        use_welford=True,
    )
    output = ttnn.to_torch(ttnn.from_device(output)).float()

    assert torch.isfinite(output).all()
    assert_numeric_metrics(
        reference,
        output,
        pcc_threshold=0.9995,
        rtol=0.14,
        atol=0.085,
        frobenius_threshold=0.04,
    )


@pytest.mark.parametrize("N, C, H, W, num_groups", HEIGHT_SHARDED_SHAPES)
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
@pytest.mark.parametrize("specify_grid", [True])
def test_group_norm_with_height_sharded(device, N, C, H, W, num_groups, use_welford, specify_grid):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=1, x=8)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

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

    # shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size if specify_grid else None,
        use_welford=use_welford,
    )

    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if use_welford:
        pcc_threshold = 0.99975
        rtol = 0.14
        atol = 0.085
        frobenius_threshold = 0.02
    else:
        pcc_threshold = 0.9999
        rtol = 0.065
        atol = 0.065
        frobenius_threshold = 0.015
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", HEIGHT_SHARDED_NON_TILE_ALIGNED_SHAPES)
def test_group_norm_height_sharded_non_tile_aligned(device, N, C, H, W, num_groups):
    # Single-core height-sharded group_norm with a non-tile-aligned flattened height must match
    # torch within bf16 tolerance.
    torch.manual_seed(0)

    assert (N * H * W) % 32 != 0, "shape must be non-tile-aligned to exercise the fix"
    grid_size = ttnn.CoreGrid(y=1, x=1)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=1e-12
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Tile-pad the flattened height to the tile boundary (logical H*W stays non-aligned; the
    # padded height is a multiple of 32), matching how a non-aligned tensor reaches the op.
    padded_hw = ((N * H * W + 31) // 32) * 32
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor = ttnn.tilize_with_zero_padding(input_tensor, use_multicore=True)

    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)
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

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = padded_hw // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    # Twice, like run_group_norm_DRAM: the sharded path ships the corrected reduce scaler and K as
    # RUNTIME args, so a program-cache hit on the second call must not lose or stale them.
    for _ in range(2):
        output_tensor = ttnn.group_norm(
            input_tensor,
            num_groups=num_groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
            output_layout=ttnn.TILE_LAYOUT,
            inplace=False,
            use_welford=False,
        )
        ttnn.synchronize_device(device)
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))

    # rtol is left at its default: this bug was a per-group affine drift, which PCC cannot see and
    # a value-proportional tolerance would partly absorb, so atol is the discriminator.
    assert_numeric_metrics(torch_output_tensor, output_tensor, atol=0.08, frobenius_threshold=0.03)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups, grid_y, grid_x", BLOCK_SHARDED_NON_TILE_ALIGNED_CASES)
@pytest.mark.parametrize(
    "use_welford, out_row_major",
    [(False, False), (True, False), (False, True)],
    ids=["legacy", "welford_routed", "row_major_out"],
)
def test_group_norm_block_sharded_non_tile_aligned(
    device, N, C, H, W, num_groups, grid_y, grid_x, use_welford, out_row_major
):
    # Block-sharded two-pass path. grid_x > 1 splits the padded H*W across M-cores
    # (padding tail on the last), exercising the multi-core correction; use_welford=True must be
    # routed to the two-pass path; out_row_major selects UNTILIZE_OUT, which runs after the
    # corrected rsqrt and so must not change the result. negative_mask is not covered: it requires
    # ROW_MAJOR, where padded_shape[2] == logical_shape[2], so the correction never engages.
    torch.manual_seed(0)
    if device.core_grid.x < grid_x or device.core_grid.y < grid_y:
        pytest.skip(f"device grid too small for {grid_x}x{grid_y}")

    assert (N * H * W) % 32 != 0, "shape must be non-tile-aligned to exercise the fix"
    padded_hw = ((N * H * W + 31) // 32) * 32
    assert (padded_hw // 32) % grid_x == 0, "padded H*W tiles must be divisible by grid_x"
    grid_size = ttnn.CoreGrid(y=grid_y, x=grid_x)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=1e-12
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor = ttnn.tilize_with_zero_padding(input_tensor, use_multicore=True)

    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)
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

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = padded_hw // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    # Twice: the sharded path ships the corrected reduce scaler and K as RUNTIME args, so a
    # program-cache hit on the second call must not lose or stale them.
    for _ in range(2):
        output_tensor = ttnn.group_norm(
            input_tensor,
            num_groups=num_groups,
            input_mask=input_mask_tensor,
            weight=gamma_t,
            bias=beta_t,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
            output_layout=ttnn.ROW_MAJOR_LAYOUT if out_row_major else ttnn.TILE_LAYOUT,
            inplace=False,
            use_welford=use_welford,
        )
        ttnn.synchronize_device(device)
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.DRAM_MEMORY_CONFIG)
    output_tensor = ttnn.to_torch(ttnn.from_device(output_tensor))
    output_tensor = output_tensor.reshape(N, 1, -1, C)[:, :, : W * H, :]

    assert_numeric_metrics(torch_output_tensor, output_tensor, atol=0.08, frobenius_threshold=0.03)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", BLOCK_SHARDED_V2_8X4_SHAPES)
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
@pytest.mark.parametrize("specify_grid", [True])
def test_group_norm_with_block_sharded_v2_8x4_grid(device, N, C, H, W, num_groups, use_welford, specify_grid):
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=4, x=8)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

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

    # shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size if specify_grid else None,
        use_welford=use_welford,
    )

    # output tensor
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if use_welford:
        pcc_threshold = 0.99975
        rtol = 0.14
        atol = 0.085
        frobenius_threshold = 0.02
    else:
        pcc_threshold = 0.9999
        rtol = 0.065
        atol = 0.065
        frobenius_threshold = 0.015
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


OFFSET_SHARD_GRID_SHAPE_CASES = [
    # (N, C, H, W, num_groups), (core_grid_x, core_grid_y)
    ((1, 1280, 16, 16, 32), (4, 4)),
    ((1, 960, 1, 1024, 32), (8, 4)),
]
OFFSET_SHARD_GRID_OFFSETS = [
    ttnn.CoreCoord(0, 0),
    ttnn.CoreCoord(2, 0),
    ttnn.CoreCoord(0, 2),
    ttnn.CoreCoord(1, 1),
    ttnn.CoreCoord(4, 4),
]
OFFSET_SHARD_GRID_ORIENTATIONS = [ttnn.ShardOrientation.COL_MAJOR, ttnn.ShardOrientation.ROW_MAJOR]


def _offset_grid_fits_device(device, core_grid, offset):
    dev = device.compute_with_storage_grid_size()
    return offset.x + core_grid[0] <= dev.x and offset.y + core_grid[1] <= dev.y


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("shape, core_grid", OFFSET_SHARD_GRID_SHAPE_CASES)
@pytest.mark.parametrize("grid_offset", OFFSET_SHARD_GRID_OFFSETS)
@pytest.mark.parametrize("orientation", OFFSET_SHARD_GRID_ORIENTATIONS)
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
def test_group_norm_with_offset_shard_grid(device, shape, core_grid, grid_offset, orientation, use_welford):
    """Sharded groupnorm must work when the shard grid does not start at core (0, 0), for both orientations."""
    if not _offset_grid_fits_device(device, core_grid, grid_offset):
        pytest.skip(f"core grid {core_grid} at offset ({grid_offset.x}, {grid_offset.y}) does not fit on this device")

    N, C, H, W, num_groups = shape
    torch.manual_seed(0)

    grid_size = ttnn.CoreGrid(y=core_grid[1], x=core_grid[0])

    # For BLOCK sharding the channel dim C is split across grid.y for COL_MAJOR and across
    # grid.x for ROW_MAJOR; the input mask / gamma / beta are laid out per that core count.
    col_major = orientation == ttnn.ShardOrientation.COL_MAJOR
    channel_cores = grid_size.y if col_major else grid_size.x

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, channel_cores, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, channel_cores)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, channel_cores)

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

    shard_end = ttnn.CoreCoord(core_grid[0] + grid_offset.x - 1, core_grid[1] + grid_offset.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(grid_offset, shard_end)})
    # COL_MAJOR splits height across grid.x and C across grid.y; ROW_MAJOR is transposed.
    if col_major:
        shard_shape = N * H * W // grid_size.x, C // grid_size.y
    else:
        shard_shape = N * H * W // grid_size.y, C // grid_size.x
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, orientation)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
        use_welford=use_welford,
    )

    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=0.9999,
        rtol=0.05,
        atol=0.065,
        frobenius_threshold=0.015,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", BLOCK_SHARDED_V2_8X8_SHAPES)
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
@pytest.mark.parametrize("specify_grid", [True])
def test_group_norm_with_block_sharded_v2_8x8_grid(device, N, C, H, W, num_groups, use_welford, specify_grid):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    grid_size = ttnn.CoreGrid(y=8, x=8)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

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

    # shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.interleaved_to_sharded(input_tensor, sharded_mem_config, keep_l1_aligned=True)

    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size if specify_grid else None,
        use_welford=use_welford,
    )

    # output tensor
    output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if use_welford:
        pcc_threshold = 0.99975
        rtol = 0.14
        atol = 0.085
        frobenius_threshold = 0.02
    else:
        pcc_threshold = 0.9999
        rtol = 0.065
        atol = 0.065
        frobenius_threshold = 0.02

    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", BLOCK_SHARDED_V2_8X8_TILE_LAYOUT_SHAPES)
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
@pytest.mark.parametrize("specify_grid", [True])
def test_group_norm_with_block_sharded_v2_8x8_grid_tile_layout(
    device, N, C, H, W, num_groups, use_welford, specify_grid
):
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    grid_size = ttnn.CoreGrid(y=8, x=8)

    # torch input tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # input tensor
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    # input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # gamma/beta
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.y)

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

    # shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, sharded_mem_config)

    # groupnorm
    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size if specify_grid else None,
        inplace=False,
        use_welford=use_welford,
    )

    # output tensor
    output_tensor = ttnn.to_memory_config(output_tensor, ttnn.L1_MEMORY_CONFIG)
    output_tensor = ttnn.from_device(output_tensor)
    output_tensor = ttnn.to_torch(output_tensor)

    if use_welford:
        pcc_threshold = 0.99975
        rtol = 0.14
        atol = 0.085
        frobenius_threshold = 0.02
    else:
        pcc_threshold = 0.9999
        rtol = 0.065
        atol = 0.065
        frobenius_threshold = 0.015
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


def generate_sdxl_test_inputs():
    inputs = []

    ##### START: 1024x1024 resolution #####
    # UNet inputs
    inputs.append((1, 1280, 64, 64))
    inputs.append((1, 1280, 32, 32))
    inputs.append((1, 1920, 64, 64))
    inputs.append((1, 1920, 32, 32))
    inputs.append((1, 2560, 32, 32))
    inputs.append((1, 320, 128, 128))
    inputs.append((1, 320, 64, 64))
    inputs.append((1, 640, 64, 64))
    inputs.append((1, 640, 32, 32))
    inputs.append((1, 960, 64, 64))

    # VAE inputs
    inputs.append((1, 512, 128, 128))

    # Refiner UNet inputs
    inputs.append((1, 1152, 64, 64))
    inputs.append((1, 1536, 16, 16))
    inputs.append((1, 1536, 32, 32))
    inputs.append((1, 1536, 64, 64))
    inputs.append((1, 2304, 32, 32))
    inputs.append((1, 2304, 64, 64))
    inputs.append((1, 3072, 16, 16))
    inputs.append((1, 3072, 32, 32))
    inputs.append((1, 384, 128, 128))
    inputs.append((1, 384, 64, 64))
    inputs.append((1, 768, 32, 32))
    inputs.append((1, 768, 64, 64))
    ###### END: 1024x1024 resolution ######

    ##### START: 512x512 resolution #####
    # UNet inputs
    inputs.append((1, 320, 64, 64))
    inputs.append((1, 320, 32, 32))
    inputs.append((1, 640, 32, 32))
    inputs.append((1, 640, 16, 16))
    inputs.append((1, 1280, 16, 16))
    inputs.append((1, 2560, 16, 16))
    inputs.append((1, 1920, 16, 16))
    inputs.append((1, 1920, 32, 32))
    inputs.append((1, 1280, 32, 32))
    inputs.append((1, 960, 32, 32))
    inputs.append((1, 960, 64, 64))
    inputs.append((1, 640, 64, 64))
    ###### END: 512x512 resolution ######

    return inputs


def run_sdxl_base_group_norm_test(
    device, N, C, H, W, use_welford, layout, inplace, specify_grid=True, perf_test_mode=False
):
    num_groups = 32  #  always 32 for SDXL Base
    if layout == ttnn.TILE_LAYOUT and inplace:
        pytest.skip("Tile layout requires non-inplace tensors.")
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    core_grid = ttnn.CoreGrid(y=8, x=8)

    # Generate torch tensor
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)

    if not perf_test_mode:
        # Execute torch group_norm
        torch_output_tensor = torch.nn.functional.group_norm(torch_input_tensor, num_groups)
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate ttnn tensor
    dummy_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        dummy_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=layout,
        memory_config=ttnn.create_sharded_memory_config(
            shape=dummy_tensor.shape,
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        device=device,
    )

    # Generate input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, core_grid.x, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # Execute ttnn group_norm
    tt_output_tensor = ttnn.group_norm(
        tt_input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        memory_config=tt_input_tensor.memory_config(),
        core_grid=core_grid if specify_grid else None,
        inplace=inplace,
        use_welford=use_welford,
    )
    ttnn.synchronize_device(device)

    if not perf_test_mode:
        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)

        if use_welford:
            pcc_threshold = 0.9995
            rtol = 0.14
            atol = 0.085
            frobenius_threshold = 0.04
        else:
            pcc_threshold = 0.9999
            rtol = 0.065
            atol = 0.065
            frobenius_threshold = 0.04
        assert_numeric_metrics(
            torch_output_tensor,
            tt_output_tensor,
            pcc_threshold=pcc_threshold,
            rtol=rtol,
            atol=atol,
            frobenius_threshold=frobenius_threshold,
        )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("input_shape", generate_sdxl_test_inputs())
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
# Paramemeters need to stay consistent with usage in
# models/demos/stable_diffusion_xl_base/tests/test_sdxl_op_unit_test_perf.py::test_block_sharded_group_norm_sdxl_performance
def test_sdxl_base_group_norm(device, input_shape, use_welford, specify_grid=True, perf_test_mode=False):
    # Only one test case has C == 512, which has TILE_LAYOUT and inplace False
    # ALL other inputs have ROW_MAJOR_LAYOUT and inplace True
    N, C, H, W = input_shape
    layout = ttnn.TILE_LAYOUT if C == 512 else ttnn.ROW_MAJOR_LAYOUT
    inplace = layout != ttnn.TILE_LAYOUT
    run_sdxl_base_group_norm_test(device, N, C, H, W, use_welford, layout, inplace, specify_grid, perf_test_mode)


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("input_shape", generate_sdxl_test_inputs())
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
@pytest.mark.parametrize("specify_grid", [True])
# Oppositive of previous test in terms of inplace, for full coverage purposes.
def test_sdxl_group_norm_reverse_inplace(device, input_shape, use_welford, specify_grid, perf_test_mode=False):
    # Only one test case has C == 512, which has TILE_LAYOUT and inplace True
    # ALL other inputs have ROW_MAJOR_LAYOUT and inplace False
    N, C, H, W = input_shape
    layout = ttnn.TILE_LAYOUT if C == 512 else ttnn.ROW_MAJOR_LAYOUT
    inplace = layout != ttnn.TILE_LAYOUT
    run_sdxl_base_group_norm_test(device, N, C, H, W, use_welford, layout, inplace, specify_grid, perf_test_mode)


@pytest.mark.parametrize("input_shape", SDXL_BASE_GROUP_NORM_BH_SHAPES)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("specify_grid", [True])
@run_for_blackhole("blackhole specific tests")
def test_sdxl_base_group_norm_bh(device, input_shape, specify_grid, perf_test_mode=False):
    torch.manual_seed(0)

    num_groups = 32  #  always 32 for SDXL Base
    N, C, H, W = input_shape

    core_grid = ttnn.CoreGrid(y=8, x=8)
    layout = ttnn.TILE_LAYOUT if C == 512 else ttnn.ROW_MAJOR_LAYOUT
    inplace = layout != ttnn.TILE_LAYOUT

    # Generate torch tensor
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)

    if not perf_test_mode:
        # Execute torch group_norm
        torch_output_tensor = torch.nn.functional.group_norm(torch_input_tensor, num_groups)
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate ttnn tensor
    dummy_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        dummy_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=layout,
        memory_config=ttnn.create_sharded_memory_config(
            shape=dummy_tensor.shape,
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.BLOCK,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
        ),
        device=device,
    )

    # Generate input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, core_grid.x, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # Execute ttnn group_norm
    tt_output_tensor = ttnn.group_norm(
        tt_input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        memory_config=tt_input_tensor.memory_config(),
        core_grid=core_grid if specify_grid else None,
        inplace=inplace,
        use_welford=False,
    )
    ttnn.synchronize_device(device)

    if not perf_test_mode:
        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)

        pcc_threshold = 0.9999
        rtol = 0.065
        atol = 0.065
        frobenius_threshold = 0.036
        assert_numeric_metrics(
            torch_output_tensor,
            tt_output_tensor,
            pcc_threshold=pcc_threshold,
            rtol=rtol,
            atol=atol,
            frobenius_threshold=frobenius_threshold,
        )


def generate_sdxl_test_inputs_neg_mask():
    inputs = []
    inputs.append((1, 640, 128, 128))
    inputs.append((1, 960, 128, 128))
    inputs.append((1, 768, 128, 128))
    return inputs


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE_SDXL_BG_N_MASK, indirect=True)
@pytest.mark.parametrize("input_shape", generate_sdxl_test_inputs_neg_mask())
def test_sdxl_base_group_norm_negative_mask(device, input_shape, specify_grid=True, perf_test_mode=False):
    num_groups = 32  #  always 32 for SDXL Base 1024x1024
    N, C, H, W = input_shape
    torch.manual_seed(0)
    if device.core_grid.y == 7:
        pytest.skip()

    core_x = 8
    core_y = 8
    grid_size = ttnn.CoreGrid(y=core_y, x=core_x)

    # Generate torch tensor
    torch_input_tensor = torch.rand(input_shape, dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    if not perf_test_mode:
        # Execute torch group_norm
        torch_output_tensor = torch.nn.functional.group_norm(
            torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
        )
        torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate ttnn tensor
    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        tt_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Generate input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.x, ttnn.DataType.BFLOAT8_B)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    input_negative_mask_tensor = ttnn.create_group_norm_input_negative_mask(
        C, num_groups, grid_size.x, ttnn.DataType.BFLOAT8_B
    )
    input_negative_mask_tensor = ttnn.to_device(input_negative_mask_tensor, device)

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.x)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.x)

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

    # Generate shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.y, C // grid_size.x
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_device(tt_input_tensor, device, memory_config=sharded_mem_config)

    # Execute ttnn group_norm
    tt_output_tensor = ttnn.group_norm(
        tt_input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        negative_mask=input_negative_mask_tensor,
        memory_config=sharded_mem_config,
        core_grid=grid_size if specify_grid else None,
        weight=gamma_t,
        bias=beta_t,
    )
    ttnn.synchronize_device(device)

    if not perf_test_mode:
        tt_output_tensor = ttnn.from_device(tt_output_tensor)
        tt_output_tensor = ttnn.to_torch(tt_output_tensor)

        pcc_threshold = 0.9999
        rtol = 0.065
        atol = 0.065
        frobenius_threshold = 0.016
        assert_numeric_metrics(
            torch_output_tensor,
            tt_output_tensor,
            pcc_threshold=pcc_threshold,
            rtol=rtol,
            atol=atol,
            frobenius_threshold=frobenius_threshold,
        )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", COMPUTE_CONFIG_SHAPES)
@pytest.mark.parametrize("specify_grid", [True])
def test_group_norm_compute_config(device, N, C, H, W, num_groups, specify_grid):
    """
    Test that a high-accuracy compute kernel config produces a higher PCC with torch
    than a lower-accuracy compute kernel config.
    """

    if device.core_grid.y == 7:
        pytest.skip()

    torch.manual_seed(0)
    input_shape = (N, C, H, W)
    grid_size = ttnn.CoreGrid(y=8, x=8)

    # Execute torch group_norm
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.nn.functional.group_norm(torch_input_tensor, num_groups)
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate input mask
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.y, ttnn.DataType.BFLOAT16)
    tt_input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    # Generate shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    # Helper function to execute group_norm for a given compute config
    def do_group_norm_for_config(compute_config):
        tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
        tt_input_tensor = ttnn.from_torch(
            tt_input_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=sharded_mem_config,
        )

        tt_output_tensor = ttnn.group_norm(
            tt_input_tensor,
            num_groups=num_groups,
            input_mask=tt_input_mask_tensor,
            memory_config=sharded_mem_config,
            core_grid=grid_size if specify_grid else None,
            compute_kernel_config=compute_config,
        )
        tt_output_tensor_host = ttnn.from_device(tt_output_tensor)
        tt_output_tensor_host = ttnn.to_torch(tt_output_tensor_host)

        ttnn.deallocate(tt_input_tensor)
        ttnn.deallocate(tt_output_tensor)

        return tt_output_tensor_host

    # Execute low-accuracy groupnorm
    config_low = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_output_low = do_group_norm_for_config(config_low)
    ref_f = torch_output_tensor.float()
    frobenius_low = (ref_f - tt_output_low.float()).norm() / (ref_f.norm() + 1e-8)

    # Execute high-accuracy groupnorm
    config_high = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tt_output_high = do_group_norm_for_config(config_high)
    frobenius_high = (ref_f - tt_output_high.float()).norm() / (ref_f.norm() + 1e-8)

    # Verify that the higher-accuracy config is closer to torch
    assert (
        frobenius_high <= frobenius_low
    ), "High-accuracy config should have lower Frobenius error than low-accuracy config"


@pytest.mark.parametrize("N, C, H, W, num_groups, shard, eps, use_negative_mask", GROUP_NORM_OFT_PARAMS)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("specify_grid", [True])
@run_for_blackhole("blackhole specific tests")
def test_group_norm_oft(device, N, C, H, W, num_groups, shard, eps, use_negative_mask, specify_grid):
    assert C % num_groups == 0, "Number of channels must be divisible by number of groups"

    skip_if_not_blackhole_20_cores(device)
    compute_grid = device.compute_with_storage_grid_size()
    grid_size = ttnn.CoreGrid(y=compute_grid.y, x=compute_grid.x)
    # Generate torch tensor
    torch.manual_seed(0)
    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)
    # Execute torch group_norm
    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias, eps=eps
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    # Generate input mask
    if shard == "HS":
        grid_x = grid_size.x * grid_size.y
        grid_y = 1
    else:
        grid_x = grid_size.x
        grid_y = grid_size.y
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_y, ttnn.DataType.BFLOAT16)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)
    if use_negative_mask:
        input_nmask_tensor = ttnn.create_group_norm_input_negative_mask(C, num_groups, grid_y, ttnn.DataType.BFLOAT16)
        input_nmask_tensor = ttnn.to_device(input_nmask_tensor, device)
    else:
        input_nmask_tensor = None
    # Generate gamma/beta tensors
    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_y)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_y)

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

    # Generate shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = (H * W) // grid_x, C // grid_y
    if shard == "HS":
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
    elif shard == "BS":
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
        sharded_mem_config = ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
        )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

    output_tensor = ttnn.group_norm(
        input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        negative_mask=input_nmask_tensor,
        weight=gamma_t,
        bias=beta_t,
        memory_config=sharded_mem_config,
        core_grid=grid_size if specify_grid else None,
        epsilon=eps,
    )
    output_tensor = ttnn.to_torch(output_tensor)
    pcc_threshold = 0.9999
    rtol = 0.065
    atol = 0.065
    frobenius_threshold = 0.014
    assert_numeric_metrics(
        torch_output_tensor,
        output_tensor,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", NO_INPUT_MASK_SHAPES)
@pytest.mark.parametrize("specify_grid", [True])
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
def test_group_norm_no_input_mask(device, N, C, H, W, num_groups, use_welford, specify_grid):
    """
    Test that a group norm without an input mask produces the same result as torch.

    Exercises the writer-kernel mask-synthesis path on the sharded factory.
    """
    torch.manual_seed(0)
    input_shape = (N, C, H, W)
    grid_size = ttnn.CoreGrid(y=4, x=4)

    # Execute torch group_norm
    torch_input_tensor = torch.rand(input_shape, dtype=torch.float32)
    torch_output_tensor = torch.nn.functional.group_norm(torch_input_tensor, num_groups)
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    # Generate shard config
    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.x, C // grid_size.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )

    # Helper function to execute group_norm for a given compute config
    def do_group_norm_for_config(compute_config):
        tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
        tt_input_tensor = ttnn.from_torch(
            tt_input_tensor,
            dtype=ttnn.DataType.BFLOAT16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=sharded_mem_config,
        )

        tt_output_tensor = ttnn.group_norm(
            tt_input_tensor,
            num_groups=num_groups,
            memory_config=sharded_mem_config,
            core_grid=grid_size if specify_grid else None,
            compute_kernel_config=compute_config,
            use_welford=use_welford,
        )
        tt_output_tensor_host = ttnn.from_device(tt_output_tensor)
        tt_output_tensor_host = ttnn.to_torch(tt_output_tensor_host)

        ttnn.deallocate(tt_input_tensor)
        ttnn.deallocate(tt_output_tensor)

        return tt_output_tensor_host

    # Execute low-accuracy groupnorm
    config_low = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )
    tt_output_low = do_group_norm_for_config(config_low)
    ref_f2 = torch_output_tensor.float()
    frobenius_low2 = (ref_f2 - tt_output_low.float()).norm() / (ref_f2.norm() + 1e-8)

    # Execute high-accuracy groupnorm
    config_high = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )
    tt_output_high = do_group_norm_for_config(config_high)
    frobenius_high2 = (ref_f2 - tt_output_high.float()).norm() / (ref_f2.norm() + 1e-8)

    assert_numeric_metrics(
        torch_output_tensor,
        tt_output_high,
        pcc_threshold=0.9999,
        rtol=0.065,
        atol=0.075,
        frobenius_threshold=0.015,
    )

    assert_numeric_metrics(
        torch_output_tensor,
        tt_output_low,
        pcc_threshold=0.999,
        rtol=0.15,
        atol=0.30,
        frobenius_threshold=0.12,
    )

    # Verify that the higher-accuracy config is closer to torch
    assert (
        frobenius_high2 <= frobenius_low2
    ), "High-accuracy config should have lower Frobenius error than low-accuracy config"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE_SDXL_BG_N_MASK, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", [(1, 640, 128, 128, 32)])
def test_group_norm_bf16_negative_mask(device, N, C, H, W, num_groups):
    """
    Caller-supplied bf16 positive + negative masks on the sharded factory.
    The existing SDXL negative-mask coverage passes BFP8 masks, so this is the
    only exercise of the bf16 mask dtype through the writer's DRAM read path.
    """
    torch.manual_seed(0)
    grid_size = ttnn.CoreGrid(y=8, x=8)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        tt_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.x, ttnn.DataType.BFLOAT16)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    input_negative_mask_tensor = ttnn.create_group_norm_input_negative_mask(
        C, num_groups, grid_size.x, ttnn.DataType.BFLOAT16
    )
    input_negative_mask_tensor = ttnn.to_device(input_negative_mask_tensor, device)

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.x)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.x)

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

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.y, C // grid_size.x
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt_input_tensor = ttnn.to_device(tt_input_tensor, device, memory_config=sharded_mem_config)

    tt_output_tensor = ttnn.group_norm(
        tt_input_tensor,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        negative_mask=input_negative_mask_tensor,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
        weight=gamma_t,
        bias=beta_t,
    )
    ttnn.synchronize_device(device)

    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        tt_output_tensor,
        pcc_threshold=0.9999,
        rtol=0.065,
        atol=0.065,
        frobenius_threshold=0.016,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE_SDXL_BG_N_MASK, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups", [(1, 640, 128, 128, 32)])
def test_group_norm_auto_negative_mask_synthesis(device, N, C, H, W, num_groups):
    """
    Exercises NEGATIVE_MASK_SYNTHESIZE with no mask tensors passed at all. The L1 ballast
    below leaves the overlap as the only layout that fits, so the call completing is the
    proof that the op enabled it by itself. Compared against the caller-supplied bf16
    mask path for bit-equivalence.
    """
    torch.manual_seed(0)
    grid_size = ttnn.CoreGrid(y=8, x=8)

    torch_input_tensor = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    torch_output_tensor = torch.nn.functional.group_norm(
        torch_input_tensor, num_groups, weight=torch_weight, bias=torch_bias
    )
    torch_output_tensor = torch_output_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)

    tt_input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    tt_input_tensor = ttnn.from_torch(
        tt_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    gamma = ttnn.create_group_norm_weight_bias_rm(torch_weight, C, grid_size.x)
    beta = ttnn.create_group_norm_weight_bias_rm(torch_bias, C, grid_size.x)
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

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_shape = N * H * W // grid_size.y, C // grid_size.x
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    # Each path gets its own freshly uploaded copy of the identical input: the op writes
    # its output over the input's L1 shard, so reusing one device tensor would feed the
    # second run already-normalized data.
    tt_input_synth = ttnn.to_device(tt_input_tensor, device, memory_config=sharded_mem_config)

    # 64 KB/core: more than the margin by which the non-overlapping layout fits, less
    # than the ~385 KB the overlap frees. Held live across the call below.
    ballast_rows_per_core, ballast_cols_per_core = 64, 512
    ballast_shard = ttnn.ShardSpec(
        shard_grid, (ballast_rows_per_core, ballast_cols_per_core), ttnn.ShardOrientation.ROW_MAJOR
    )
    ballast = ttnn.from_torch(
        torch.zeros(
            (1, 1, ballast_rows_per_core * grid_size.y, ballast_cols_per_core * grid_size.x), dtype=torch.bfloat16
        ),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.MemoryConfig(
            ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, ballast_shard
        ),
    )

    tt_output_tensor = ttnn.group_norm(
        tt_input_synth,
        num_groups=num_groups,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
        weight=gamma_t,
        bias=beta_t,
    )
    ttnn.synchronize_device(device)
    ttnn.deallocate(ballast)

    tt_output_tensor = ttnn.from_device(tt_output_tensor)
    tt_output_tensor = ttnn.to_torch(tt_output_tensor)

    assert_numeric_metrics(
        torch_output_tensor,
        tt_output_tensor,
        pcc_threshold=0.9999,
        rtol=0.065,
        atol=0.065,
        frobenius_threshold=0.016,
    )

    # Holding tt_input_synth leaves two 320 KB/core shards across the second call,
    # which does not fit on Wormhole's L1.
    ttnn.deallocate(tt_input_synth)

    # The path this replaces: the same computation driven by caller-supplied bf16
    # positive and negative mask tensors read from DRAM. Run it from the identical
    # input and compare the two device outputs directly, since agreeing with torch
    # to within a tolerance does not by itself establish that the two paths agree
    # with each other.
    input_mask_tensor = ttnn.create_group_norm_input_mask(C, num_groups, grid_size.x, ttnn.DataType.BFLOAT16)
    input_mask_tensor = ttnn.to_device(input_mask_tensor, device)

    input_negative_mask_tensor = ttnn.create_group_norm_input_negative_mask(
        C, num_groups, grid_size.x, ttnn.DataType.BFLOAT16
    )
    input_negative_mask_tensor = ttnn.to_device(input_negative_mask_tensor, device)

    tt_input_supplied = ttnn.to_device(tt_input_tensor, device, memory_config=sharded_mem_config)

    tt_output_supplied_mask = ttnn.group_norm(
        tt_input_supplied,
        num_groups=num_groups,
        input_mask=input_mask_tensor,
        negative_mask=input_negative_mask_tensor,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
        weight=gamma_t,
        bias=beta_t,
    )
    ttnn.synchronize_device(device)

    tt_output_supplied_mask = ttnn.from_device(tt_output_supplied_mask)
    tt_output_supplied_mask = ttnn.to_torch(tt_output_supplied_mask)

    assert torch.equal(tt_output_tensor, tt_output_supplied_mask), (
        "Synthesized masks must reproduce the caller-supplied bf16 mask path bit-exactly; "
        f"max abs difference {(tt_output_tensor.float() - tt_output_supplied_mask.float()).abs().max()}"
    )


@pytest.mark.parametrize("input_shape, num_groups, msg_pattern", NEGATIVE_TESTS_PARAMS)
def test_group_norm_negative_tests(
    input_shape,
    num_groups,
    msg_pattern,
    device,
    expect_error,
):
    input_tensor = ttnn.empty(input_shape, device=device)
    with expect_error(RuntimeError, msg_pattern):
        ttnn.group_norm(
            input_tensor,
            num_groups=num_groups,
            core_grid=ttnn.CoreGrid(y=1, x=1),
            inplace=False,
        )


def test_group_norm_rejects_non_tile_aligned_spatial(device, expect_error):
    # group_norm reduces over the flattened spatial dimension (N*H*W) in 32-row
    # tiles, so that dimension must be a whole number of tiles -- otherwise the
    # trailing partial tile would be silently dropped from the mean/variance,
    # producing wrong results. Here N*H*W = 16, which is not a multiple of 32.
    #
    # A TILE input cannot exercise this (TILE pads the row dim up to 32), and a
    # ROW_MAJOR interleaved input is rejected earlier as unsupported on the
    # non-sharded path. A sharded input keeps the unpadded row dim and reaches the
    # invariant check, so use that to cover it.
    C, HW, num_groups = 320, 16, 32
    torch_input_tensor = torch.rand((1, 1, HW, C), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (HW, C), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

    with expect_error(RuntimeError, "must be divisible by the tile size"):
        ttnn.group_norm(
            input_tensor,
            num_groups=num_groups,
            memory_config=sharded_mem_config,
            core_grid=ttnn.CoreGrid(y=1, x=1),
        )


def test_group_norm_rejects_per_sample_non_tile_aligned_spatial(device, expect_error):
    # Scope boundary for #50682, which only engages for TILE inputs with logical_shape[2] <
    # padded_shape[2]. The N*H*W check above misses a PER-SAMPLE H*W that is not a whole number of
    # tiles -- N=2, H*W=80 gives N*H*W=160, a multiple of 32 -- so the device op's own H*W check is
    # what rejects it, and such shapes fail loudly rather than reducing over a partial tile.
    N, C, HW, num_groups = 2, 320, 80, 32
    assert (N * HW) % 32 == 0, "N*H*W must look aligned, so only the per-sample check can reject"
    assert HW % 32 != 0, "per-sample H*W must not be tile-aligned"

    torch_input_tensor = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (N * HW, C), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

    with expect_error(RuntimeError, "must be a multiple of the tile height"):
        ttnn.group_norm(
            input_tensor,
            num_groups=num_groups,
            memory_config=sharded_mem_config,
            core_grid=ttnn.CoreGrid(y=1, x=1),
        )


def test_group_norm_rejects_tile_input_with_inplace(device, expect_error):
    # Scope boundary, same argument as negative_mask above: inplace is only
    # allowed for ROW_MAJOR inputs, and a ROW_MAJOR tensor has padded_shape[2] == logical_shape[2],
    # so inplace and the tile-padding correction can never be active at the same time. Note the
    # binding defaults inplace to True, which is why every TILE-input test passes inplace=False.
    C, HW, num_groups = 320, 128, 32
    input_tensor = ttnn.from_torch(
        torch.rand((1, 1, HW, C), dtype=torch.bfloat16),
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    with expect_error(RuntimeError, "Tile layout requires non-inplace tensors"):
        ttnn.group_norm(input_tensor, num_groups=num_groups, inplace=True)


def test_group_norm_rejects_host_input_mask(device, expect_error):
    # TILE layout: a ROW_MAJOR interleaved input is rejected earlier (it is unsupported
    # on the non-sharded path), which would pre-empt the host-input-mask check this test
    # targets.
    input_tensor = ttnn.empty((1, 1, 32, 320), device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    input_mask = ttnn.create_group_norm_input_mask(320, 32, 1, ttnn.DataType.BFLOAT16)

    with expect_error(RuntimeError, "Input mask must be on device"):
        ttnn.group_norm(
            input_tensor,
            num_groups=32,
            input_mask=input_mask,
            core_grid=ttnn.CoreGrid(y=1, x=1),
            inplace=False,
        )


def test_group_norm_rejects_host_negative_mask(device, expect_error):
    grid_size = ttnn.CoreGrid(y=1, x=1)
    torch_input_tensor = torch.rand((1, 320, 32, 32), dtype=torch.bfloat16)
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(1, 1, 32 * 32, 320)
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    input_mask = ttnn.create_group_norm_input_mask(320, 32, grid_size.x, ttnn.DataType.BFLOAT16)
    input_mask = ttnn.to_device(input_mask, device)
    negative_mask = ttnn.create_group_norm_input_negative_mask(320, 32, grid_size.x, ttnn.DataType.BFLOAT16)

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (32 * 32, 320), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.to_memory_config(input_tensor, memory_config=sharded_mem_config)

    with expect_error(RuntimeError, "Negative mask must be on device"):
        ttnn.group_norm(
            input_tensor,
            num_groups=32,
            input_mask=input_mask,
            negative_mask=negative_mask,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
        )


def _block_sharded_320ch_input(device, grid_size, layout, spatial=32):
    # Single core, so `spatial` sets how much L1 the program needs. The default is too big
    # to fit; pass a smaller one when the op is expected to run to completion.
    hw = spatial * spatial
    torch_input_tensor = torch.rand((1, 320, spatial, spatial), dtype=torch.bfloat16)
    input_tensor = torch_input_tensor.permute(0, 2, 3, 1).view(1, 1, hw, 320)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
    shard_spec = ttnn.ShardSpec(shard_grid, (hw, 320), ttnn.ShardOrientation.ROW_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    input_tensor = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=layout,
        device=device,
        memory_config=sharded_mem_config,
    )
    return torch_input_tensor, input_tensor, sharded_mem_config


def test_group_norm_rejects_negative_mask_with_welford(device, expect_error):
    # Only a caller-supplied negative mask can reach the Welford kernels; the op's own
    # decision declines for use_welford=True.
    grid_size = ttnn.CoreGrid(y=1, x=1)
    _, input_tensor, sharded_mem_config = _block_sharded_320ch_input(device, grid_size, ttnn.ROW_MAJOR_LAYOUT)

    input_mask = ttnn.to_device(ttnn.create_group_norm_input_mask(320, 32, grid_size.x, ttnn.DataType.BFLOAT16), device)
    negative_mask = ttnn.to_device(
        ttnn.create_group_norm_input_negative_mask(320, 32, grid_size.x, ttnn.DataType.BFLOAT16), device
    )
    with expect_error(RuntimeError, "Negative mask is not supported with use_welford=True"):
        ttnn.group_norm(
            input_tensor,
            num_groups=32,
            input_mask=input_mask,
            negative_mask=negative_mask,
            memory_config=sharded_mem_config,
            core_grid=grid_size,
            use_welford=True,
        )


def test_group_norm_tile_layout_declines_negative_mask(device):
    # A TILE input has no negative-mask code path, so the op must decline the overlap.
    # Guards the layout check in needs_negative_mask_overlap.
    grid_size = ttnn.CoreGrid(y=1, x=1)
    torch.manual_seed(0)
    torch_input, input_tensor, sharded_mem_config = _block_sharded_320ch_input(
        device, grid_size, ttnn.TILE_LAYOUT, spatial=8
    )

    output = ttnn.group_norm(
        input_tensor,
        num_groups=32,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
        inplace=False,
    )
    ttnn.synchronize_device(device)
    assert output.layout == ttnn.TILE_LAYOUT

    torch_output = torch.nn.functional.group_norm(torch_input, 32)
    torch_output = torch_output.permute(0, 2, 3, 1).view(1, 1, 8 * 8, 320)
    assert_numeric_metrics(
        torch_output,
        ttnn.to_torch(ttnn.from_device(output)),
        pcc_threshold=0.9999,
        rtol=0.065,
        atol=0.065,
        frobenius_threshold=0.016,
    )


@pytest.mark.parametrize("N, C, H, W, num_groups", DRAM_GRID_SIZE_SHAPES)
@pytest.mark.parametrize("specify_grid", [True])
def test_group_norm_dram_grid_size(device, N, C, H, W, num_groups, specify_grid):
    """Use determine_expected_group_norm_dram_grid_size to pick a grid, then
    run DRAM-interleaved group norm and compare against torch."""
    torch.manual_seed(0)

    grid_size = ttnn.determine_expected_group_norm_dram_grid_size(
        device=device,
        num_channels=C,
        num_groups=num_groups,
        input_nhw=N * H * W,
    )

    torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    torch_output = torch.nn.functional.group_norm(torch_input, num_groups, weight=torch_weight, bias=torch_bias)
    torch_output = torch_output.permute(0, 2, 3, 1).view(N, 1, H * W, C)

    [gamma_t, beta_t], input_mask = ttnn.dram_group_norm_params_from_torch(
        [torch_weight.float(), torch_bias.float()],
        C,
        num_groups,
        device,
        core_grid=grid_size,
        return_mask=True,
    )

    tt_input = torch_input.permute(0, 2, 3, 1).view(N, 1, H * W, C)
    tt_input = ttnn.from_torch(
        tt_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_input = ttnn.fill_implicit_tile_padding(tt_input, TEST_PADDING_VALUE)

    tt_output = ttnn.group_norm(
        tt_input,
        num_groups=num_groups,
        input_mask=input_mask,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=grid_size if specify_grid else None,
        inplace=False,
        num_out_blocks=1 if specify_grid else None,
        use_welford=True,
    )

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    pcc_threshold = 0.99975
    rtol = 0.14
    atol = 0.085
    frobenius_threshold = 0.02
    assert_numeric_metrics(
        torch_output,
        tt_output,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("N, C, H, W, num_groups", OPTIONAL_WEIGHT_BIAS_SHAPES)
@pytest.mark.parametrize("use_welford", welford_flavors, ids=welford_ids)
@pytest.mark.parametrize(
    "has_weight, has_bias", OPTIONAL_WEIGHT_BIAS_AFFINE_PARAMS, ids=OPTIONAL_WEIGHT_BIAS_AFFINE_IDS
)
@pytest.mark.parametrize("specify_grid", [True])
def test_group_norm_optional_weight_bias(
    device, N, C, H, W, num_groups, use_welford, has_weight, has_bias, specify_grid
):
    """Verify group_norm with all combinations of optional weight/bias, for both welford and legacy."""
    torch.manual_seed(0)

    grid_size = ttnn.determine_expected_group_norm_dram_grid_size(
        device=device,
        num_channels=C,
        num_groups=num_groups,
        input_nhw=N * H * W,
    )

    num_virtual_cols = ttnn.operations.normalization.dram_group_norm_virtual_columns(grid_size, C, num_groups)
    input_mask = ttnn.create_group_norm_input_mask(C, num_groups, num_virtual_cols, ttnn.bfloat16)
    input_mask = ttnn.to_device(input_mask, device)

    torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16) if has_weight else None
    torch_bias = torch.rand((C,), dtype=torch.bfloat16) if has_bias else None
    epsilon = 1e-5

    torch_output = torch.nn.functional.group_norm(
        torch_input.float(),
        num_groups,
        weight=torch_weight.float() if torch_weight is not None else None,
        bias=torch_bias.float() if torch_bias is not None else None,
        eps=epsilon,
    )
    torch_output = torch_output.to(torch.bfloat16)
    torch_output = torch_output.permute(0, 2, 3, 1).view(N, 1, H * W, C)

    gamma_t, beta_t = None, None
    if has_weight or has_bias:
        params_list = []
        if has_weight:
            params_list.append(torch_weight.float())
        if has_bias:
            params_list.append(torch_bias.float())

        tt_params = ttnn.dram_group_norm_params_from_torch(
            params_list if len(params_list) > 1 else params_list[0],
            C,
            num_groups,
            device,
            core_grid=grid_size,
            return_mask=False,
        )
        if has_weight and has_bias:
            gamma_t, beta_t = tt_params
        elif has_weight:
            gamma_t = tt_params
        else:
            beta_t = tt_params

    tt_input = torch_input.permute(0, 2, 3, 1).view(N, 1, H * W, C)
    tt_input = ttnn.from_torch(
        tt_input,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_output = ttnn.group_norm(
        tt_input,
        num_groups=num_groups,
        epsilon=epsilon,
        input_mask=input_mask,
        weight=gamma_t,
        bias=beta_t,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        core_grid=grid_size if specify_grid else None,
        inplace=False,
        use_welford=use_welford,
    )

    tt_output = ttnn.from_device(tt_output)
    tt_output = ttnn.to_torch(tt_output)

    if use_welford:
        pcc_threshold = 0.99
        rtol = 0.14
        atol = 0.3
        if specify_grid:
            frobenius_threshold = 0.06
        else:
            # Automatically chosen grid results in a slightly higher overall error.
            frobenius_threshold = 0.065
    else:
        pcc_threshold = 0.9999
        rtol = 0.065
        atol = 0.065
        frobenius_threshold = 0.016
    assert_numeric_metrics(
        torch_output,
        tt_output,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("N, C, H, W, num_groups, grid_y, grid_x", GN_SHARDED_SHAPES)
@pytest.mark.parametrize("gb_dtype", [ttnn.bfloat16, ttnn.float32], ids=["gb_bf16", "gb_fp32"])
@pytest.mark.parametrize("in_dtype", [ttnn.float32, ttnn.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT], ids=["row_major", "tile"])
@pytest.mark.parametrize("use_welford", [True, False], ids=["welford", "legacy"])
def test_group_norm_sharded_all_config(
    device, use_welford, layout, in_dtype, gb_dtype, N, C, H, W, num_groups, grid_y, grid_x
):
    # Sharded group_norm across both reduction paths (welford / legacy two-pass) for the fp32/bf16
    # input x fp32/bf16 gamma-beta matrix. Sharded supports ROW_MAJOR and TILE in both directions
    # (TILIZE_IN/UNTILIZE_OUT are gated on layout, not on welford). The welford_reciprocal mode is
    # DRAM-only (the sharded program factory never consumes a reciprocals tensor), so it is not
    # exercised here.
    grid = ttnn.CoreGrid(y=grid_y, x=grid_x)
    torch.manual_seed(0)
    x = torch.rand((N, C, H, W), dtype=torch.float32)
    w = torch.rand((C,), dtype=torch.float32)
    b = torch.rand((C,), dtype=torch.float32)
    ref = torch.nn.functional.group_norm(x, num_groups, weight=w, bias=b).permute(0, 2, 3, 1).view(N, 1, W * H, C)

    ck = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,  # required for FP32 (Welford path, or legacy fp32 DEST accumulation)
        packer_l1_acc=False,
    )

    xt = x.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    xt = ttnn.from_torch(xt, dtype=in_dtype, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    mask = ttnn.to_device(ttnn.create_group_norm_input_mask(C, num_groups, grid.y, ttnn.DataType.BFLOAT8_B), device)
    gamma = ttnn.create_group_norm_weight_bias_rm(w, C, grid.y)
    beta = ttnn.create_group_norm_weight_bias_rm(b, C, grid.y)
    gt = ttnn.from_torch(
        gamma, dtype=gb_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    bt = ttnn.from_torch(
        beta, dtype=gb_dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
    shard_shape = N * H * W // grid.x, C // grid.y
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.COL_MAJOR)
    tensor_memory_layout = (
        ttnn.types.TensorMemoryLayout.HEIGHT_SHARDED if grid.y == 1 else ttnn.types.TensorMemoryLayout.BLOCK_SHARDED
    )
    mem = ttnn.MemoryConfig(tensor_memory_layout, ttnn.types.BufferType.L1, shard_spec)
    xt = ttnn.to_memory_config(xt, mem)

    out = ttnn.group_norm(
        xt,
        num_groups=num_groups,
        input_mask=mask,
        weight=gt,
        bias=bt,
        memory_config=mem,
        core_grid=grid,
        dtype=in_dtype,
        compute_kernel_config=ck,
        use_welford=use_welford,
        output_layout=layout,
        inplace=(layout == ttnn.ROW_MAJOR_LAYOUT),  # in-place only valid for sharded ROW_MAJOR
    )
    out = (
        ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))).float().reshape(ref.shape)
    )

    # Thresholds branch on the reduction path and the input dtype (bf16 input is the dominant error
    # source); each bound sits ~1.4x above the worst observed value across the shape/gamma-beta/layout matrix.
    if use_welford:
        if in_dtype == ttnn.bfloat16:
            pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.01, 0.06, 0.015
        else:
            pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.008, 0.02, 0.004
    else:
        if in_dtype == ttnn.bfloat16:
            pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.01, 0.09, 0.035
        else:
            pcc_threshold, rtol, atol, frobenius_threshold = 0.999, 0.008, 0.08, 0.035
    assert_numeric_metrics(
        ref,
        out,
        pcc_threshold=pcc_threshold,
        rtol=rtol,
        atol=atol,
        frobenius_threshold=frobenius_threshold,
    )


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    # 1.0 is the exp() case, and the worst case for the back-correction this replaced (it drove the
    # variance negative there, and rsqrt to 786430).
    "padding_value",
    [7.0, 1.0, -3.5, 0.5],
)
@pytest.mark.parametrize(
    # grid_x is the M split. With grid_x=2 the batch spans two cores and only the second holds the
    # padding row-tile; with grid_x=1 every core holds its own batch tail.
    "grid_y, grid_x",
    [(2, 1), (2, 2), (2, 4), (4, 2)],
)
@pytest.mark.parametrize(
    # How the {0,1} column selector reaches the kernel, for each way group_norm accepts it:
    #   "doubled"     -- caller ships both mask sets (create_group_norm_input_mask with
    #                    rows_in_last_tile).
    #   "single_set"  -- caller ships the column selector only.
    #   "synthesized" -- no mask at all; the writer builds one row-0-only selector set directly
    #                    in L1, and compute composes the row-masked final row-tile on device.
    "mask_mode",
    ["doubled", "single_set", "synthesized"],
)
def test_group_norm_sharded_dirty_padding(device, grid_y, grid_x, padding_value, mask_mode):
    # Mirror of test_group_norm_non_tile_aligned_garbage_padding_DRAM for the BLOCK-SHARDED path:
    # supply non-zero tile padding and require the result to be independent of it.
    #
    # A host-side zero-fill of the input only worked interleaved, since fill_implicit_tile_padding
    # corrupts a block-sharded tensor whose padding tail sits on the last M-core. In-kernel masking
    # never touches the input, so it covers block-sharded too.
    if device.core_grid.x < grid_x or device.core_grid.y < grid_y:
        pytest.skip(f"device grid too small for {grid_x}x{grid_y}")

    torch.manual_seed(0)
    N, C, HW, G, padded = 1, 256, 100, 32, 128
    grid_size = ttnn.CoreGrid(y=grid_y, x=grid_x)

    real = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)
    ref = torch.nn.functional.group_norm(real.view(N, HW, C).permute(0, 2, 1).reshape(N, C, 1, HW).float(), G)
    ref = ref.permute(0, 2, 3, 1).reshape(N, 1, HW, C)

    buf = torch.zeros((N, 1, padded, C), dtype=torch.bfloat16)
    buf[:, :, :HW, :] = real
    buf[:, :, HW:, :] = padding_value  # dirty padding, as reshape / slice / exp would leave

    tt = ttnn.from_torch(
        buf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt = ttnn.reshape(tt, ttnn.Shape([N, 1, HW, C]), ttnn.Shape([N, 1, padded, C]))

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded // grid_size.x, C // grid_size.y), ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt = ttnn.to_memory_config(tt, sharded_mem_config)

    if mask_mode == "synthesized":
        mask = None
    else:
        mask = ttnn.to_device(
            ttnn.create_group_norm_input_mask(
                C,
                G,
                grid_size.y,
                ttnn.DataType.BFLOAT8_B,
                rows_in_last_tile=(HW % 32) if mask_mode == "doubled" else 0,
            ),
            device,
        )
    out = ttnn.group_norm(
        tt,
        num_groups=G,
        input_mask=mask,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
        inplace=False,
    )
    out = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))).float()[:, :, :HW, :]

    max_abs_err = (out - ref).abs().max().item()
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    logger.info(
        f"block-sharded dirty padding grid={grid_x}x{grid_y} padding={padding_value} "
        f"mask_mode={mask_mode}: max_abs_err={max_abs_err} pcc={pcc}"
    )
    assert max_abs_err < 0.08, (
        f"max abs error {max_abs_err} with tile padding = {padding_value} on a {grid_x}x{grid_y} "
        f"block-sharded grid; group_norm must be independent of its padding (see #52685)"
    )
    assert pcc > 0.999, f"pcc {pcc} with tile padding = {padding_value} (see #52685)"


@pytest.mark.parametrize("device_params", DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("padding_value", [7.0, -3.5])
@pytest.mark.parametrize("grid_x", [1, 2, 4])
@pytest.mark.parametrize(
    # Group sizes that are a WHOLE number of tiles. On grid_y=8, C=1024 gives 128 channels
    # (4 tiles) per core:
    #   G=32 -> group 32 ch = 1 tile   (block_wt == 1)
    #   G=16 -> group 64 ch = 2 tiles  (block_wt == 2)
    #   G=64 -> group 16 ch, sub-tile  (control, matches the regime already covered)
    "C, G, whole_tile_groups",
    [(1024, 32, True), (1024, 16, True), (1024, 64, False)],
    ids=["blockwt1", "blockwt2", "subtile_control"],
)
@pytest.mark.parametrize("mask_mode", ["doubled", "single_set", "synthesized"])
def test_group_norm_sharded_dirty_padding_tile_aligned_groups(
    device, grid_x, C, G, whole_tile_groups, padding_value, mask_mode
):
    # Same contract as test_group_norm_sharded_dirty_padding -- the result must not depend on what
    # sits in the tile padding -- but at group sizes that are a whole number of tiles, where the
    # per-group column selector is all-ones and only the composed row exclusion does any work.
    grid_y = 8
    if device.core_grid.x < grid_x or device.core_grid.y < grid_y:
        pytest.skip(f"device grid too small for {grid_x}x{grid_y}")

    torch.manual_seed(0)
    N, HW, padded = 1, 100, 128
    assert (padded // 32) % grid_x == 0, "padded H*W tiles must split evenly across the M cores"
    grid_size = ttnn.CoreGrid(y=grid_y, x=grid_x)

    real = torch.rand((N, 1, HW, C), dtype=torch.bfloat16)
    ref = torch.nn.functional.group_norm(real.view(N, HW, C).permute(0, 2, 1).reshape(N, C, 1, HW).float(), G)
    ref = ref.permute(0, 2, 3, 1).reshape(N, 1, HW, C)

    buf = torch.zeros((N, 1, padded, C), dtype=torch.bfloat16)
    buf[:, :, :HW, :] = real
    buf[:, :, HW:, :] = padding_value

    tt = ttnn.from_torch(
        buf, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    tt = ttnn.reshape(tt, ttnn.Shape([N, 1, HW, C]), ttnn.Shape([N, 1, padded, C]))

    grid_coord = ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), grid_coord)})
    shard_spec = ttnn.ShardSpec(shard_grid, (padded // grid_size.x, C // grid_size.y), ttnn.ShardOrientation.COL_MAJOR)
    sharded_mem_config = ttnn.MemoryConfig(
        ttnn.types.TensorMemoryLayout.BLOCK_SHARDED, ttnn.types.BufferType.L1, shard_spec
    )
    tt = ttnn.to_memory_config(tt, sharded_mem_config)

    if mask_mode == "synthesized":
        mask = None
    else:
        mask = ttnn.to_device(
            ttnn.create_group_norm_input_mask(
                C,
                G,
                grid_size.y,
                ttnn.DataType.BFLOAT8_B,
                rows_in_last_tile=(HW % 32) if mask_mode == "doubled" else 0,
            ),
            device,
        )
    out = ttnn.group_norm(
        tt,
        num_groups=G,
        input_mask=mask,
        memory_config=sharded_mem_config,
        core_grid=grid_size,
        inplace=False,
    )
    out = ttnn.to_torch(ttnn.from_device(ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG))).float()[:, :, :HW, :]

    max_abs_err = (out - ref).abs().max().item()
    pcc = torch.corrcoef(torch.stack([out.flatten(), ref.flatten()]))[0, 1].item()
    logger.info(
        f"whole-tile groups={whole_tile_groups} C={C} G={G} grid={grid_x}x{grid_y} "
        f"padding={padding_value} mask_mode={mask_mode}: max_abs_err={max_abs_err} pcc={pcc}"
    )
    assert max_abs_err < 0.08, (
        f"max abs error {max_abs_err} with tile padding = {padding_value} at C={C} G={G} "
        f"(whole-tile groups={whole_tile_groups}); the composed row mask must still apply"
    )
    assert pcc > 0.999, f"pcc {pcc} at C={C} G={G} with tile padding = {padding_value}"

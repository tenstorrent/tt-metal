# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole, run_for_blackhole

import tests.ttnn.unit_tests.operations.fused.test_group_norm_DRAM as base


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        # Only SDXL/sd35 tests with 512x512 or larger sizes moved to nightly
        #  SDXL VAE
        (1, 128, 1024, 1024, 32, 32, 8, 8),
        (1, 128, 512, 512, 32, 8, 8, 8),
        (1, 256, 1024, 1024, 32, 48, 8, 8),
        (1, 256, 515, 512, 32, 12, 8, 8),
        (1, 512, 512, 512, 32, 12, 8, 8),
        # SDXL Refiner
        (1, 256, 512, 512, 32, 16, 8, 8),  # SD 1.4 VAE
        (1, 128, 512, 512, 32, 22, 4, 4),  # SD 1.4 VAE
        # sd35. 4 indicates the number of device.
        (1, 512 // 4, 512, 512, 32 // 4, 8, 8, 8),
        (1, 256 // 4, 512, 512, 32 // 4, 4, 8, 8),
        (1, 256 // 4, 1024, 1024, 32 // 4, 16, 8, 8),
        # for test below, PCC drops to 0.9565977077851433 of welford_normal and welford_reciprocal
        (1, 128 // 4, 1024, 1024, 32 // 4, 8, 8, 8),
        # mochi
        # (21, 128, 480, 848, 32, 140, 8, 8), Failing on single device CI.
    ],
)
@pytest.mark.parametrize("welford_mode", base.WELFORD_MODES)
@pytest.mark.parametrize("specify_grid", [True, False])
def test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, specify_grid):
    base.test_group_norm_DRAM(
        device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, specify_grid
    )


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
def test_group_norm_DRAM_rejects_non_uniform_mcast_groups(device):
    """Regression test for GH#40912: N=2 with grid (8,5) creates num_virtual_rows=5
    which is not divisible by num_batches=2, producing non-uniform mcast groups
    that deadlock due to exact-equality semaphore waits in the sender kernel."""
    torch.manual_seed(0)
    N, C, H, W = 2, 768, 12, 12
    num_groups = 32
    bad_grid = ttnn.CoreGrid(y=5, x=8)

    torch_input = torch.rand((N, C, H, W), dtype=torch.bfloat16)
    torch_weight = torch.rand((C,), dtype=torch.bfloat16)
    torch_bias = torch.rand((C,), dtype=torch.bfloat16)

    input_tensor = torch_input.permute(0, 2, 3, 1).view(N, 1, W * H, C)
    input_tensor_rm = ttnn.from_torch(
        input_tensor,
        dtype=ttnn.DataType.BFLOAT16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    input_tensor_tilized = ttnn.tilize_with_zero_padding(input_tensor_rm, use_multicore=True)

    [gamma_t, beta_t], input_mask = ttnn.dram_group_norm_params_from_torch(
        [torch_weight, torch_bias], C, num_groups, device, core_grid=bad_grid, return_mask=True
    )

    with pytest.raises(RuntimeError, match="core_grid"):
        ttnn.group_norm(
            input_tensor_tilized,
            num_groups=num_groups,
            input_mask=input_mask,
            weight=gamma_t,
            bias=beta_t,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            output_layout=ttnn.TILE_LAYOUT,
            core_grid=bad_grid,
            inplace=False,
            num_out_blocks=1,
        )


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x",
    [
        (9, 768, 1, 512, 32, 2, 8, 8),  # test batch size 9 (uneven batch sizes)
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
        # SDXL Base
        (1, 1920, 16, 16, 32, 1, 4, 4),
        #  SDXL VAE
        (1, 256, 256, 256, 32, 4, 8, 8),
        (1, 512, 256, 256, 32, 4, 8, 8),
        # SDXL Refiner
        (1, 1152, 128, 128, 32, 2, 8, 4),
        (1, 512, 64, 64, 32, 1, 8, 8),  # SD 1.4 VAE
        (1, 512, 128, 128, 32, 1, 8, 8),  # SD 1.4 VAE
        (1, 512, 256, 256, 32, 4, 8, 8),  # SD 1.4 VAE
        (1, 256, 256, 256, 32, 8, 8, 8),  # SD 1.4 VAE
        # sd35. 4 indicates the number of device.
        (1, 256 // 4, 256, 256, 32 // 4, 1, 8, 8),
        (1, 512 // 4, 128, 128, 32 // 4, 1, 8, 8),
        (1, 512 // 4, 256, 256, 32 // 4, 2, 8, 8),
        # mochi
        # (21, 128, 480, 848, 32, 140, 8, 8), Failing on single device CI.
    ],
)
@pytest.mark.parametrize("welford_mode", base.WELFORD_MODES)
@pytest.mark.parametrize("specify_grid", [True, False])
def test_group_norm_no_input_mask_DRAM(
    device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, specify_grid
):
    base.test_group_norm_no_input_mask_DRAM(
        device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, specify_grid
    )


# ---------------------------------------------------------------------------
# Nightly wrappers: cover unit-test shapes with specify_grid=False so the
# auto-grid path is exercised without bloating the regular CI pipeline.
# All parametrize data is sourced from base.<CONST> so nightly stays in sync
# when unit-test parameters change.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x", base.GROUP_NORM_DRAM_SHAPES)
@pytest.mark.parametrize("welford_mode", base.WELFORD_MODES)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_DRAM_unit_shapes(
    device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, specify_grid
):
    base.test_group_norm_DRAM(
        device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, specify_grid
    )


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x", base.GROUP_NORM_NO_INPUT_MASK_DRAM_SHAPES
)
@pytest.mark.parametrize("welford_mode", base.WELFORD_MODES)
@pytest.mark.parametrize("specify_grid", [False])
def test_group_norm_no_input_mask_DRAM_unit_shapes(
    device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, specify_grid
):
    base.test_group_norm_no_input_mask_DRAM(
        device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, specify_grid
    )


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize("N, C, H, W, num_groups, num_splits", base.SDXL_BASE_GROUP_NORM_SPLIT_SHAPES)
@pytest.mark.parametrize("specify_grid", [False])
def test_sdxl_base_group_norm_split_unit_shapes(device, N, C, H, W, num_groups, num_splits, specify_grid):
    base.test_sdxl_base_group_norm_split(device, N, C, H, W, num_groups, num_splits, specify_grid)


@pytest.mark.parametrize("device_params", base.DEVICE_PARAMS_L1_SMALL_SIZE, indirect=True)
@pytest.mark.parametrize(
    "N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, eps", base.GROUP_NORM_DRAM_OFT_PARAMS
)
@pytest.mark.parametrize("specify_grid", [False])
@run_for_blackhole("blackhole specific tests")
def test_group_norm_DRAM_oft_unit_shapes(
    device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, eps, specify_grid
):
    base.test_group_norm_DRAM_oft(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, eps, specify_grid)

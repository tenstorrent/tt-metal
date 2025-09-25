# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole

import tests.ttnn.unit_tests.operations.fused.test_group_norm_DRAM as base

welford_flavors = base.get_welford_params()


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
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
        (1, 128 // 4, 1024, 1024, 32 // 4, 8, 8, 8),
        # mochi
        # (21, 128, 480, 848, 32, 140, 8, 8), Failing on single device CI.
    ],
)
@pytest.mark.parametrize("welford_mode", welford_flavors)
def test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode):
    base.test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
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
@pytest.mark.parametrize("welford_mode", welford_flavors)
def test_group_norm_no_input_mask_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode):
    base.run_group_norm_DRAM(
        device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode, use_input_mask=False
    )

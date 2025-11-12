# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

from loguru import logger

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.common.utility_functions import is_blackhole

import tests.ttnn.unit_tests.operations.fused.test_group_norm_DRAM as base


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
        # for test below, PCC drops to 0.9565977077851433 of welford_normal and welford_reciprocal
        (1, 128 // 4, 1024, 1024, 32 // 4, 8, 8, 8),
        # mochi
        # (21, 128, 480, 848, 32, 140, 8, 8), Failing on single device CI.
    ],
)
@pytest.mark.parametrize("welford_mode", ("legacy", "welford_normal", "welford_reciprocal"))
def test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode):
    base.test_group_norm_DRAM(device, N, C, H, W, num_groups, num_out_blocks, cores_y, cores_x, welford_mode)

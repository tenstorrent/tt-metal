# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple
import ttnn
from tests.sweep_framework.sweep_utils.adaptive_pool2d_common import (
    run_adaptive_avg_pool2d,
    run_adaptive_max_pool2d,
    invalidate_vector,
)

parameters = {
    "adaptive_pool2d_short_sweep_suite": {
        "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
        "pool_type": ["avg", "max"],
        "input_specs": [
            # [batch_size, input_channels, input_height, input_width, output_height, output_width]
            [1, 256, 56, 56, 1, 1],  # Global pooling deeper network
            [1, 512, 28, 28, 1, 1],  # Global pooling very deep
            [1, 1024, 14, 14, 1, 1],  # Global pooling final layers
            [1, 512, 7, 7, 7, 7],  # No-op case (input == output size)
            [1, 2048, 7, 7, 1, 1],  # ResNet-style global pooling
            [1, 64, 224, 224, 7, 7],  # Standard classifier head
            [1, 256, 32, 32, 4, 4],  # 8x downsampling
            [1, 128, 64, 64, 8, 8],  # 8x downsampling different size
            [1, 160, 75, 75, 3, 3],  # Small odd dimensions
            [1, 320, 28, 28, 14, 14],  # 2x downsampling
            [1, 480, 14, 14, 7, 7],  # 2x downsampling small
            [1, 512, 32, 32, 2, 4],  # Another asymmetric case
            [1, 256, 64, 64, 7, 7],  # Target case for dilation testing
            [1, 256, 64, 64, 3, 5],  # Asymmetric pooling
            [1, 64, 17, 17, 4, 3],  # Small asymmetric edge case
            [1, 256, 80, 80, 3, 3],  # Real model cases
            [1, 256, 20, 20, 3, 3],  # Real model cases
            [1, 256, 40, 40, 3, 3],  # Real model cases
            [1, 256, 60, 80, 3, 3],  # Real model cases
            [1, 256, 30, 40, 3, 3],  # Real model cases
            [1, 256, 15, 20, 3, 3],  # Real model cases
        ],
    },
}


def run(
    input_specs,
    pool_type,
    dtype,
    *,
    device,
):
    (in_n, in_c, in_h, in_w, out_h, out_w) = input_specs

    if pool_type == "avg":
        return run_adaptive_avg_pool2d(in_n, in_c, in_h, in_w, out_h, out_w, dtype, device)
    else:  # max
        return run_adaptive_max_pool2d(in_n, in_c, in_h, in_w, out_h, out_w, dtype, device)


import pytest


@pytest.mark.parametrize("input_spec", parameters["adaptive_pool2d_short_sweep_suite"]["input_specs"])
@pytest.mark.parametrize("pool_type", parameters["adaptive_pool2d_short_sweep_suite"]["pool_type"])
@pytest.mark.parametrize("dtype", parameters["adaptive_pool2d_short_sweep_suite"]["dtype"])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 16384}], indirect=True)
def test_adaptive_pool2d_localrun(device, dtype, pool_type, input_spec):
    (batch_size, input_channels, input_height, input_width, output_height, output_width) = input_spec

    if pool_type == "avg":
        run_adaptive_avg_pool2d(
            batch_size, input_channels, input_height, input_width, output_height, output_width, dtype, device
        )
    else:  # max
        run_adaptive_max_pool2d(
            batch_size, input_channels, input_height, input_width, output_height, output_width, dtype, device
        )

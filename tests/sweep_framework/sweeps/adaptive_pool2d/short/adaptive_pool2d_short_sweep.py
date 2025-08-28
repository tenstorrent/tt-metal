# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.sweep_framework.sweep_utils.adaptive_pool2d_common import run_adaptive_pool2d

parameters = {
    "input_specs": [
        # [[batch_size, input_channels, input_height, input_width, output_height, output_width]
        [1, 256, 56, 56, 1, 1],
        [1, 512, 28, 28, 1, 1],
        [1, 1024, 14, 14, 1, 1],
        [1, 512, 7, 7, 7, 7],
        [1, 2048, 7, 7, 1, 1],
        [1, 64, 224, 224, 7, 7],
        [1, 256, 32, 32, 4, 4],
        [1, 128, 64, 64, 8, 8],
        [1, 320, 28, 28, 14, 14],
        [1, 480, 14, 14, 7, 7],
        [1, 512, 32, 32, 2, 4],
        [1, 256, 64, 64, 7, 7],
        [1, 256, 64, 64, 3, 5],
        [1, 64, 17, 17, 4, 3],
        [1, 256, 80, 80, 3, 3],
        [1, 256, 20, 20, 3, 3],
        [1, 256, 40, 40, 3, 3],
        [1, 256, 60, 80, 3, 3],
        [1, 256, 30, 40, 3, 3],
        [1, 256, 15, 20, 3, 3],
        # Max pool variants
        [1, 256, 56, 56, 1, 1],
        [1, 512, 28, 28, 1, 1],
        [1, 1024, 14, 14, 1, 1],
        [1, 512, 7, 7, 7, 7],
        [1, 2048, 7, 7, 1, 1],
        [1, 64, 224, 224, 7, 7],
        [1, 256, 32, 32, 4, 4],
        [1, 128, 64, 64, 8, 8],
        [1, 320, 28, 28, 14, 14],
        [1, 480, 14, 14, 7, 7],
        # bfloat8_b variants (subset)
        [1, 256, 56, 56, 1, 1],
        [1, 512, 28, 28, 1, 1],
        [1, 1024, 14, 14, 1, 1],
        [1, 256, 56, 56, 1, 1],
        [1, 512, 28, 28, 1, 1],
        [1, 1024, 14, 14, 1, 1],
    ]
}


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}
    return tensor_map


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("input_spec", parameters["input_specs"])
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "pool_type",
    ["max", "avg"],
)
def test_adaptive_pool2d_short_sweep(
    device,
    tensor_map,
    input_spec,
    dtype,
    pool_type,
):
    (
        in_n,
        in_c,
        in_h,
        in_w,
        out_h,
        out_w,
    ) = input_spec
    run_adaptive_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=(in_n, in_c, in_h, in_w),
        output_size=(out_h, out_w),
        dtype=dtype,
        pool_type=pool_type,
    )

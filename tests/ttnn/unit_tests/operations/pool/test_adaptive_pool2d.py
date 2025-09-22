# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.sweep_framework.sweep_utils.adaptive_pool2d_common import run_adaptive_pool2d


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}
    return tensor_map


failing_parameters = [
    # [batch_size, input_channels, input_height, input_width, output_height, output_width]
    [1, 16, 23, 24, 7, 7],
    [1, 16, 23, 24, 14, 14],
    [1, 8, 19, 21, 7, 7],
    [1, 8, 19, 21, 14, 14],
]


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 512, 28, 28),
        (1, 224, 42, 42),
        (2, 512, 28, 28),
        (1, 256, 56, 56),
        (1, 16, 23, 24),
        (1, 8, 19, 21),
    ],
)
@pytest.mark.parametrize(
    "output_size",
    (
        (1, 1),
        (7, 7),
        (14, 14),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "pool_type",
    ["max", "avg"],
)
def test_adaptive_avg_pool2d(
    device,
    tensor_map,
    input_shape,
    output_size,
    dtype,
    pool_type,
):
    if list(input_shape) + list(output_size) in failing_parameters:
        pytest.skip(
            f"Skipping failing cases due to non correctable patterns in kernels or strides: {input_shape} -> {output_size}"
        )

    run_adaptive_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        output_size=output_size,
        dtype=dtype,
        pool_type=pool_type,
    )

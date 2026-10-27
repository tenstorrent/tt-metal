# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
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
def test_adaptive_pool2d(
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


# Kernels look uniform but lowered pool2d does not match pytorch adaptive windows
@pytest.mark.parametrize(
    "input_hw, output_hw",
    [
        (6, 4),
        (10, 4),
    ],
)
def test_adaptive_pool2d_rejects_mismatched_windows(device, input_hw, output_hw, expect_error):
    batch, channels = 1, 32
    ttnn_input = ttnn.from_torch(
        torch.randn(1, 1, batch * input_hw * input_hw, channels, dtype=torch.bfloat16),
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    with expect_error(RuntimeError, "do not produce uniform pooling behavior"):
        ttnn.adaptive_avg_pool2d(
            input_tensor=ttnn_input,
            batch_size=batch,
            input_h=input_hw,
            input_w=input_hw,
            channels=channels,
            output_size=[output_hw, output_hw],
        )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape, output_size, num_slices",
    [
        ((1, 64, 256, 256), (64, 64), 16),
        ((1, 224, 128, 128), (128, 128), 8),
        ((2, 128, 384, 384), (64, 64), 16),
        ((1, 16, 1024, 1024), (128, 128), 8),
        ((1, 8, 384, 384), (64, 64), 8),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "pool_type",
    ["max", "avg"],
)
def test_adaptive_dram_pool2d(
    device,
    tensor_map,
    input_shape,
    num_slices,
    output_size,
    dtype,
    pool_type,
):
    dram_slice_config = ttnn.Op2DSliceConfig(num_slices=num_slices, slice_type=ttnn.Op2DDRAMSliceHeight)

    run_adaptive_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        output_size=output_size,
        dtype=dtype,
        pool_type=pool_type,
        dram_slice_config=dram_slice_config,
    )

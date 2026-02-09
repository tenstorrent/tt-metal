# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from models.common.utility_functions import skip_with_watcher
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
        (1, 512, 350, 70),
        (1, 256, 350, 280),
        (1, 256, 280, 350),
        (1, 256, 700, 140),
        (1, 256, 140, 700),
    ],
)
@pytest.mark.parametrize(
    "output_size",
    (
        # (1, 1),
        (7, 7),
        (14, 14),
    ),
)
@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "pool_type",
    ["max"],
)
@skip_with_watcher("Skipping with watcher enabled due to github issue #37097")
def test_adaptive_pool2d(
    device,
    tensor_map,
    input_shape,
    output_size,
    dtype,
    pool_type,
):
    run_adaptive_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        output_size=output_size,
        dtype=dtype,
        pool_type=pool_type,
    )

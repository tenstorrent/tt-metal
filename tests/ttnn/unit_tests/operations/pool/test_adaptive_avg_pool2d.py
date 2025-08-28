# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import pytest
from tests.sweep_framework.sweep_utils.adaptive_pool2d_common import run_adaptive_pool2d


@pytest.fixture(scope="module")
def tensor_map():
    tensor_map = {}
    return tensor_map


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize(
    "input_shape",
    [
        (1, 512, 28, 28),
        (1, 224, 42, 42),
        (2, 512, 28, 28),
        (1, 256, 56, 56),
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
def test_adaptive_avg_pool2d(
    device,
    tensor_map,
    input_shape,
    output_size,
    dtype,
):
    run_adaptive_pool2d(
        device=device,
        tensor_map=tensor_map,
        input_shape=input_shape,
        output_size=output_size,
        dtype=dtype,
        pool_type="avg",
    )

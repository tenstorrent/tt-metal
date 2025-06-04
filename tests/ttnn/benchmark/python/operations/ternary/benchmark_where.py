# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from tests.ttnn.ttnn_utility_fuction import create_random_torch_tensors, convert_torch_to_ttnn_tensor

import pytest


SHAPE_LIST = [
    (32, 32),
    (64, 64),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]


@pytest.mark.parametrize("shape", SHAPE_LIST)
def test_benchmark_experimental_where(benchmark, shape):
    device = ttnn.open_device(device_id=0)
    dtype = ttnn.bfloat16

    input_tensors = create_random_torch_tensors(shape, dtype, 3)
    condition, true_values, false_values = convert_torch_to_ttnn_tensor(
        input_tensors, device, dtype, ttnn.TILE_LAYOUT, mem_config=None
    )

    def where_op():
        ttnn.experimental.where(condition, true_values, false_values)

    benchmark.pedantic(where_op, iterations=10, rounds=3, warmup_rounds=1)

    ttnn.close_device(device)


@pytest.mark.parametrize("shape", SHAPE_LIST)
def test_benchmark_where(benchmark, shape):
    device = ttnn.open_device(device_id=0)
    dtype = ttnn.bfloat16

    input_tensors = create_random_torch_tensors(shape, dtype, 3)
    condition, true_values, false_values = convert_torch_to_ttnn_tensor(
        input_tensors, device, dtype, ttnn.TILE_LAYOUT, mem_config=None
    )

    def where_op():
        ttnn.where(condition, true_values, false_values)

    benchmark.pedantic(where_op, iterations=10, rounds=3, warmup_rounds=1)

    ttnn.close_device(device)

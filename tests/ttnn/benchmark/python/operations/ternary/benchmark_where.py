# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest

from tests.ttnn.utils_for_testing import tt_dtype_to_torch_dtype

SHAPE_LIST = [(dim, dim) for dim in [2**i for i in range(5, 15)]]


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


def create_random_torch_tensors(tensor_shape: tuple, tt_dtype, num_tensors: int):
    torch.manual_seed(0)
    torch_dtype = tt_dtype_to_torch_dtype[tt_dtype]
    return [
        torch.rand(tensor_shape, dtype=torch_dtype)
        if is_ttnn_float_type(tt_dtype)
        else torch.randint(0, 100, tensor_shape, dtype=torch_dtype)
        for _ in range(3)
    ]


def convert_torch_to_ttnn_tensor(
    torch_tensors: tuple,
    device,
    tt_dtype,
    layout,
    mem_config,
):
    return [
        ttnn.from_torch(
            tensor,
            layout=layout,
            dtype=tt_dtype,
            memory_config=mem_config,
            device=device,
        )
        for tensor in torch_tensors
    ]


@pytest.mark.parametrize("shape", SHAPE_LIST)
def test_benchmark_experimental_where(benchmark, device, shape):
    dtype = ttnn.bfloat16

    condition_torch, true_torch, false_torch = create_random_torch_tensors(shape, dtype, 3)
    condition_torch = condition_torch.to(torch.bool)
    condition, true_values, false_values = convert_torch_to_ttnn_tensor(
        (condition_torch, true_torch, false_torch), device, dtype, ttnn.TILE_LAYOUT, mem_config=None
    )

    def where_op():
        ttnn.experimental.where(condition, true_values, false_values)
        ttnn.synchronize_device(device)

    benchmark.pedantic(where_op, iterations=10, rounds=3, warmup_rounds=1)


@pytest.mark.parametrize("shape", SHAPE_LIST)
def test_benchmark_where(benchmark, device, shape):
    dtype = ttnn.bfloat16

    condition_torch, true_torch, false_torch = create_random_torch_tensors(shape, dtype, 3)
    condition_torch = condition_torch.to(torch.bool)
    condition, true_values, false_values = convert_torch_to_ttnn_tensor(
        (condition_torch, true_torch, false_torch), device, dtype, ttnn.TILE_LAYOUT, mem_config=None
    )

    def where_op():
        ttnn.where(condition, true_values, false_values)
        ttnn.synchronize_device(device)

    benchmark.pedantic(where_op, iterations=10, rounds=3, warmup_rounds=1)

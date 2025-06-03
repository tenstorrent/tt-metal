# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

import pytest


SHAPE_LIST = [
    (32, 32),
    (64, 64),
    (256, 256),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]


# Share these helper functions with unit tests
def _is_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


def _get_torch_data_type(tt_dtype) -> torch.dtype:
    mapping = {
        ttnn.bfloat4_b: torch.bfloat16,  # approximate fallback
        ttnn.bfloat8_b: torch.bfloat16,  # approximate fallback
        ttnn.bfloat16: torch.bfloat16,
        ttnn.float32: torch.float32,
        ttnn.uint8: torch.int32,
        ttnn.uint16: torch.int32,  # torch has limited uint16 support
        ttnn.uint32: torch.int64,  # torch has no uint32, fallback
        ttnn.int32: torch.int32,
    }

    if tt_dtype == ttnn.DataType.INVALID:
        raise ValueError("INVALID data type provided.")

    torch_dtype = mapping.get(tt_dtype)
    if torch_dtype is None:
        raise ValueError(f"Unsupported or unknown TTNN data type: {tt_dtype}")

    return torch_dtype


def _create_random_torch_tensors(tensor_shape: tuple, tt_dtype, num_tensors: int):
    torch.manual_seed(0)
    torch_dtype = _get_torch_data_type(tt_dtype)

    results = []
    for _ in range(num_tensors):
        if _is_float_type(tt_dtype):
            t = torch.rand(tensor_shape, dtype=torch_dtype)
        else:
            t = torch.randint(0, 100, tensor_shape, dtype=torch_dtype)
        results.append(t)

    return tuple(results)


def _convert_torch_to_ttnn(
    torch_tensors: tuple,
    device,
    tt_dtype,
    layout,
    mem_config,
):
    ttnn_results = []
    for t in torch_tensors:
        tt_tensor = ttnn.from_torch(
            t,
            layout=layout,
            dtype=tt_dtype,
            memory_config=mem_config,
            device=device,
        )
        tt_tensor = ttnn.to_device(tt_tensor, device)
        ttnn_results.append(tt_tensor)

    return tuple(ttnn_results)


@pytest.mark.parametrize("shape", SHAPE_LIST)
def test_benchmark_experimental_where(benchmark, shape):
    device = ttnn.open_device(device_id=0)
    dtype = ttnn.bfloat16

    input_tensors = _create_random_torch_tensors(shape, dtype, 3)
    condition, true_values, false_values = _convert_torch_to_ttnn(
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

    input_tensors = _create_random_torch_tensors(shape, dtype, 3)
    condition, true_values, false_values = _convert_torch_to_ttnn(
        input_tensors, device, dtype, ttnn.TILE_LAYOUT, mem_config=None
    )

    def where_op():
        ttnn.where(condition, true_values, false_values)

    benchmark.pedantic(where_op, iterations=10, rounds=3, warmup_rounds=1)

    ttnn.close_device(device)

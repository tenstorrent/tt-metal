# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import contextlib
import os

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


@contextlib.contextmanager
def logging_context():
    old_types = os.environ.get("TT_METAL_LOGGER_TYPES", "")
    old_level = os.environ.get("TT_METAL_LOGGER_LEVEL", "")
    os.environ["TT_METAL_LOGGER_TYPES"] = "Op"
    os.environ["TT_METAL_LOGGER_LEVEL"] = "Debug"
    yield
    os.environ["TT_METAL_LOGGER_TYPES"] = old_types
    os.environ["TT_METAL_LOGGER_LEVEL"] = old_level


@pytest.mark.parametrize("scalar", [3])
@pytest.mark.parametrize("size", [64])
def test_add_1D_tensor_and_scalar(device, scalar, size):
    with logging_context():
        torch.manual_seed(0)

        torch_input_tensor = torch.rand((size,), dtype=torch.bfloat16)
        torch_output_tensor = torch_input_tensor + scalar

        input_tensor = ttnn.from_torch(torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        output_tensor = input_tensor + scalar
        output_tensor = ttnn.to_torch(output_tensor, torch_rank=1)

        assert ttnn.pearson_correlation_coefficient(torch_output_tensor, output_tensor) >= 0.99988
        assert output_tensor.shape == (size,)


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [64])
def test_add_2D_tensors(device, h, w):
    with logging_context():
        torch_input_tensor_a = torch.rand((h, w), dtype=torch.bfloat16)
        torch_input_tensor_b = torch.rand((h, w), dtype=torch.bfloat16)
        torch_output_tensor = torch.add(torch_input_tensor_a, torch_input_tensor_b)

        input_tensor_a = ttnn.from_torch(torch_input_tensor_a, device=device)
        input_tensor_b = ttnn.from_torch(torch_input_tensor_b, device=device)

        input_tensor_a = ttnn.to_layout(input_tensor_a, ttnn.TILE_LAYOUT)
        input_tensor_b = ttnn.to_layout(input_tensor_b, ttnn.TILE_LAYOUT)

        input_tensor_a = ttnn.to_memory_config(input_tensor_a, ttnn.L1_MEMORY_CONFIG)
        input_tensor_b = ttnn.to_memory_config(input_tensor_b, ttnn.L1_MEMORY_CONFIG)

        output = ttnn.add(input_tensor_a, input_tensor_b)
        output = ttnn.to_torch(output)

        assert_with_pcc(torch_output_tensor, output, 0.9999)

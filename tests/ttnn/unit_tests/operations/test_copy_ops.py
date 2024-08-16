# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull


def run_clone_test(N, C, H, W, memory_config, dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    tensor = ttnn.clone(input, memory_config=memory_config, dtype=dtype)
    assert tensor.shape == input.shape
    assert tensor.dtype == dtype
    assert tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(tensor), 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_clone(N, C, H, W, memory_config, dtype, device):
    run_clone_test(N, C, H, W, memory_config, dtype, device)


def run_copy_test(N, C, H, W, layout, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=layout, device=device)

    input_b = torch.zeros(shape, dtype=torch_dtype)
    input_b = ttnn.from_torch(input_b, ttnn.bfloat16, layout=layout, device=device)

    ttnn.copy(input, input_b)
    assert input_b.shape == input.shape
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(input_b), 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
def test_copy(N, C, H, W, layout, device):
    run_copy_test(N, C, H, W, layout, device)


def run_assign_test(N, C, H, W, memory_config, dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    tensor = ttnn.assign(input, memory_config=memory_config, dtype=dtype)
    assert tensor.shape == input.shape
    assert tensor.dtype == dtype
    assert tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(tensor), 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_assign(N, C, H, W, memory_config, dtype, device):
    run_assign_test(N, C, H, W, memory_config, dtype, device)


def run_binary_assign_test(N, C, H, W, layout, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=layout, device=device)

    input_b = torch.zeros(shape, dtype=torch_dtype)
    input_b = ttnn.from_torch(input_b, ttnn.bfloat16, layout=layout, device=device)

    ttnn.assign(input, input_b)
    assert input_b.shape == input.shape
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(input_b), 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("layout", [ttnn.Layout.TILE, ttnn.Layout.ROW_MAJOR])
def test_binary_assign(N, C, H, W, layout, device):
    run_binary_assign_test(N, C, H, W, layout, device)


def run_experimental_typecast_test(N, C, H, W, memory_config, dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device)
    tensor = ttnn.experimental.typecast(input, memory_config=memory_config, dtype=dtype)
    assert tensor.shape == input.shape
    assert tensor.dtype == dtype
    assert tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(tensor), 0.99)


@skip_for_grayskull()
@pytest.mark.parametrize(
    "dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_experimental_typecast(N, C, H, W, memory_config, dtype, device):
    run_experimental_typecast_test(N, C, H, W, memory_config, dtype, device)

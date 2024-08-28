# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest

import torch

import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc
from models.utility_functions import skip_for_grayskull
from models.utility_functions import is_grayskull


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


def run_assign_test_opt_tensor(N, C, H, W, memory_config, dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, ttnn.bfloat16, layout=ttnn.Layout.TILE, device=device, memory_config=memory_config)

    opt_tensor = torch.randn(shape, dtype=torch_dtype)
    opt_tensor = ttnn.from_torch(opt_tensor, dtype, layout=ttnn.Layout.TILE, device=device, memory_config=memory_config)
    ttnn.assign(input, memory_config=memory_config, dtype=dtype, optional_tensor=opt_tensor)
    assert opt_tensor.shape == input.shape
    assert opt_tensor.dtype == dtype
    assert opt_tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(opt_tensor), 0.99)


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
def test_assign_opt_tensor(N, C, H, W, memory_config, dtype, device):
    run_assign_test_opt_tensor(N, C, H, W, memory_config, dtype, device)


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


def run_experimental_typecast_test(N, C, H, W, memory_config, input_dtype, output_dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16

    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, input_dtype, layout=ttnn.Layout.TILE, device=device)
    tensor = ttnn.experimental.typecast(input, memory_config=memory_config, dtype=output_dtype)

    assert tensor.shape == input.shape
    assert tensor.dtype == output_dtype
    assert tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(tensor), 0.99)


@pytest.mark.parametrize(
    "input_dtype",
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
    "output_dtype",
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
def test_experimental_typecast(N, C, H, W, memory_config, input_dtype, output_dtype, device):
    if input_dtype == output_dtype:
        pytest.skip("Input and output data types are the same")
    run_experimental_typecast_test(N, C, H, W, memory_config, input_dtype, output_dtype, device)


def run_typecast_test(N, C, H, W, memory_config, input_dtype, output_dtype, device):
    torch.manual_seed(2005)
    shape = [N, C, H, W]
    torch_dtype = torch.bfloat16
    input = torch.randn(shape, dtype=torch_dtype)
    input = ttnn.from_torch(input, input_dtype, layout=ttnn.Layout.TILE, device=device)
    tensor = ttnn.typecast(input, memory_config=memory_config, dtype=output_dtype)

    assert tensor.shape == input.shape
    assert tensor.dtype == output_dtype
    assert tensor.memory_config() == memory_config
    assert_with_pcc(ttnn.to_torch(input), ttnn.to_torch(tensor), 0.99)


@pytest.mark.parametrize(
    "input_dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.float32,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "output_dtype",
    (
        ttnn.bfloat16,
        ttnn.bfloat8_b,
        ttnn.float32,
    ),
    ids=[
        "BFLOAT16_B",
        "BFLOAT8_B",
        "FLOAT32",
    ],
)
@pytest.mark.parametrize(
    "N, C, H, W,",
    ((1, 1, 32, 64),),
)
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_typecast(N, C, H, W, memory_config, input_dtype, output_dtype, device):
    if input_dtype == output_dtype:
        pytest.skip("Input and output data types are the same")
    run_typecast_test(N, C, H, W, memory_config, input_dtype, output_dtype, device)


# The idea of the test is to convert bfloat16 to uint32 into preallocated uint32 tensor
@pytest.mark.skipif(is_grayskull(), reason="GS does not support fp32/uint32/uint16 data types")
def test_typecast_output_tensor(device):
    torch.manual_seed(0)

    h = w = 32
    from_dtype = ttnn.bfloat16
    to_dtype = ttnn.uint32
    gold_tensor = ttnn.ones([h, w], to_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    bfloat16_tensor = ttnn.ones([h, w], from_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)
    uint32_preallocated = ttnn.empty([h, w], to_dtype, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG)

    output_ttnn = ttnn.typecast(bfloat16_tensor, ttnn.uint32, memory_config=ttnn.L1_MEMORY_CONFIG)

    pages_before = ttnn._ttnn.reports.get_buffer_pages()
    ttnn.typecast(bfloat16_tensor, to_dtype, memory_config=ttnn.L1_MEMORY_CONFIG, output_tensor=uint32_preallocated)
    assert len(pages_before) == len(ttnn._ttnn.reports.get_buffer_pages())

    torch_gold = ttnn.to_torch(gold_tensor)
    torch_output_ttnn = ttnn.to_torch(output_ttnn)
    torch_output_ttnn_preallocated = ttnn.to_torch(uint32_preallocated)
    torch.equal(torch_gold, torch_output_ttnn)
    torch.equal(torch_gold, torch_output_ttnn_preallocated)


def test_typecast_community(device):
    shape = (1, 1, 32, 32)
    x = ttnn.to_device(ttnn.to_layout(ttnn.zeros(shape), ttnn.TILE_LAYOUT), device)
    y = ttnn.typecast(x, ttnn.types.bfloat16)
    assert y.dtype == ttnn.types.bfloat16
    assert_with_pcc(ttnn.to_torch(x), ttnn.to_torch(y), 0.99)

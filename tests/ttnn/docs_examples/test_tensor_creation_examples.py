# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger


def test_arrange(device):
    # Create a TT-NN tensor with values ranging from 0 to 9
    tensor = ttnn.arange(start=0, end=10, step=2, dtype=ttnn.bfloat16, device=device)
    logger.info("TT-NN arange tensor:", tensor)


def test_empty(device):
    # Create an uninitialized TT-NN tensor with the specified shape and data type
    tensor = ttnn.empty(shape=[2, 3], dtype=ttnn.bfloat16, device=device)
    logger.info("TT-NN empty tensor shape:", tensor.shape)


def test_empty_like(device):
    # Create a TT-NN tensor with the same shape and data type as another tensor
    reference_tensor = ttnn.rand((4, 5), dtype=ttnn.bfloat16, device=device)
    tensor = ttnn.empty_like(reference_tensor, dtype=ttnn.float32, device=device)
    logger.info("TT-NN empty_like tensor shape:", tensor.shape)


def test_zeros(device):
    # Create a TT-NN tensor filled with zeros
    tensor = ttnn.zeros(shape=[2, 2], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    logger.info("TT-NN zeros tensor:", tensor)


def test_zeros_like(device):
    # Create a TT-NN tensor filled with zeros, based on the shape of another tensor
    reference_tensor = ttnn.rand((3, 4), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor = ttnn.zeros_like(tensor=reference_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    logger.info("TT-NN zeros_like tensor:", tensor)


def test_ones(device):
    # Create a TT-NN tensor filled with ones
    tensor = ttnn.ones(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    logger.info("TT-NN ones tensor:", tensor)


def test_ones_like(device):
    # Create a TT-NN tensor filled with ones, based on the shape of another tensor
    reference_tensor = ttnn.rand((3, 2), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor = ttnn.ones_like(tensor=reference_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    logger.info("TT-NN ones_like tensor:", tensor)


def test_full(device):
    # Create a TT-NN tensor filled with a specified value
    tensor = ttnn.full(shape=[2, 2], fill_value=7.0, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    logger.info("TT-NN full tensor:", tensor)


def test_full_like(device):
    # Create a TT-NN tensor filled with a specified value, based on the shape of another tensor
    reference_tensor = ttnn.rand((2, 3), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tensor = ttnn.full_like(
        tensor=reference_tensor, fill_value=3.14, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    logger.info("TT-NN full_like tensor:", tensor)


def test_rand(device):
    # Create a TT-NN tensor with random values uniformly distributed between 0 and 1
    tensor = ttnn.rand(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    logger.info("TT-NN rand tensor:", tensor)


def test_randn(device):
    # Create a TT-NN tensor with random values standard normal distributed
    tensor = ttnn.randn(shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    logger.info("TT-NN randn tensor:", tensor)


def test_from_buffer(device):
    # Create a TT-NN tensor from a Python buffer (list)
    buffer = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    tensor = ttnn.from_buffer(
        buffer=buffer, shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    logger.info("TT-NN from_buffer tensor:", tensor)


def test_bernoulli(device):
    # Create a TT-NN tensor with random values from a Bernoulli distribution
    # Initialize with ttnn.full to create probability values (0.5 = 50% chance of 1)
    input = ttnn.full([3, 3], fill_value=0.5, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    output = ttnn.bernoulli(input)
    logger.info("TT-NN bernoulli tensor:", output)


def test_complex_tensor(device):
    # Create a TT-NN complex tensor from real and imaginary parts using ttnn.Tensor constructor
    real = ttnn.Tensor([1.0, 2.0], [2], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    imag = ttnn.Tensor([1.0, 2.0], [2], ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device)
    complex_tensor = ttnn.complex_tensor(real, imag)
    logger.info("TT-NN complex tensor:", complex_tensor)


def test_index_fill(device):
    # Create a TT-NN tensor with values filled at the specified indices along the specified dimension
    tt_input = ttnn.rand([32, 32], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    tt_index = ttnn.Tensor([0, 31], [2], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, device)

    output = ttnn.index_fill(
        tt_input, 1, tt_index, 10.0
    )  # Need to ensure 10.0 is a float to match the bfloat16 dtype of the input tensor
    logger.info("TT-NN index_fill tensor:", output)


def test_uniform(device):
    # Create a TT-NN tensor with random values uniformly distributed between 0.0 and 1.0
    input = ttnn.ones([3, 3], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ttnn.uniform(input)
    logger.info("TT-NN uniform tensor:", input)

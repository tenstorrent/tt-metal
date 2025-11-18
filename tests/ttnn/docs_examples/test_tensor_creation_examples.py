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


def test_from_buffer(device):
    # Create a TT-NN tensor from a Python buffer (list)
    buffer = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    tensor = ttnn.from_buffer(
        buffer=buffer, shape=[2, 3], dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device
    )
    logger.info("TT-NN from_buffer tensor:", tensor)

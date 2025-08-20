# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import math

DEFAULT_SHAPE = (32, 32)
SHAPES = [tuple([32] * i) for i in range(6)]
ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


def check_uniform_distribution(data, value_range=(0, 1), is_discrete=False):
    n = data.numel()

    if n < 1000:
        print("[Warning] A meaningful analysis requires at least 1000 samples.")
        if n < 2:
            print("[Error] Cannot perform test with less than 2 data points.")
            return False

    start_value, end_value = value_range
    if is_discrete:
        min_val, max_val = start_value, end_value
    else:
        min_val, max_val = torch.aminmax(data)
        min_val = min_val.item()
        max_val = max_val.item()

    if min_val == max_val:
        return False

    # torch ops don't suport integer data types, convert to list
    data = data.detach().cpu().flatten().tolist()

    # Calculate sample statistics
    sample_mean = sum(data) / n
    sample_variance = sum([(x - sample_mean) ** 2 for x in data]) / n
    sample_std_dev = math.sqrt(sample_variance)

    # Calculate theoretical statistics
    if is_discrete:
        theoretical_mean = (start_value + end_value) / 2
        N = end_value - start_value + 1
        theoretical_std_dev = math.sqrt((N**2 - 1) / 12)
    else:
        theoretical_mean = (min_val + max_val) / 2
        theoretical_std_dev = (max_val - min_val) / math.sqrt(12)

    mean_diff = abs(sample_mean - theoretical_mean) / theoretical_mean * 100 if theoretical_mean != 0 else 0
    std_dev_diff = (
        abs(sample_std_dev - theoretical_std_dev) / theoretical_std_dev * 100 if theoretical_std_dev != 0 else 0
    )

    treshold_percentage = 4
    if mean_diff < treshold_percentage and std_dev_diff < treshold_percentage:
        return True

    return False


@pytest.mark.xfail(reason="BFLOAT4_B/UINT8 and `uint32/int32/BFLOAT8_B` for row major layout are not supported.")
@pytest.mark.parametrize("dtype", ALL_TYPES)
@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_tensor_dtype_and_value_range(device, dtype, layout):
    shape = (1024, 1024)
    if is_ttnn_float_type(dtype):
        tensor = ttnn.rand(shape, dtype=dtype, device=device, layout=layout)
        low = 0
        high = 1
    elif dtype == ttnn.int32:
        low = -100
        high = 100
        tensor = ttnn.rand(shape, low=low, high=high, dtype=dtype, device=device, layout=layout)
    else:
        low = 0
        high = 100
        tensor = ttnn.rand(shape, low=low, high=high, dtype=dtype, device=device, layout=layout)

    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(shape)

    torch_tensor = ttnn.to_torch(tensor)

    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
    assert check_uniform_distribution(
        torch_tensor, value_range=(low, high), is_discrete=not is_ttnn_float_type(dtype)
    ), "The distribution of random values is not uniform!"


def test_rand_defaults(device):
    tensor = ttnn.rand(DEFAULT_SHAPE, device=device)

    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert tensor.storage_type() == ttnn.StorageType.DEVICE
    assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


@pytest.mark.parametrize("shapes", SHAPES)
def test_rand_shapes(device, shapes):
    tensor = ttnn.rand(shapes, device=device)
    assert tuple(tensor.shape) == tuple(shapes)


@pytest.mark.parametrize("dim", [i for i in range(32)])
def test_rand_dims(dim, device):
    shape = (dim, dim)
    tensor = ttnn.rand(shape, device=device)
    assert tuple(tensor.shape) == tuple(shape)


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_rand_with_memory_config(device, mem_config):
    tensor = ttnn.rand(DEFAULT_SHAPE, device=device, memory_config=mem_config)
    assert tensor.memory_config() == mem_config
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


def test_rand_invalid_args(device):
    """
    Passing invalid args should raise TypeError.
    """

    with pytest.raises(TypeError):
        # expected list or tuple
        ttnn.rand(5, device=device)

    with pytest.raises(TypeError):
        # expected positive dim values
        ttnn.rand([2, -1], device=device)

    with pytest.raises(TypeError):
        # expected ttnn.LAYOUT type
        ttnn.rand([2, 2], device=device, layout="ROW_MAJOR")

    with pytest.raises(TypeError):
        # expected  ttnn.MemoryConfig type
        ttnn.rand([2, 2], device=device, memory_config="DRAM")

    with pytest.raises(TypeError):
        # expected  ttnn.Device type
        ttnn.rand([2, 2], device="WORMHOLE")

    with pytest.raises(TypeError):
        # expected  ttnn.DataType type
        ttnn.rand([2, 2], device=device, dtype="ttnn.bfloat16")

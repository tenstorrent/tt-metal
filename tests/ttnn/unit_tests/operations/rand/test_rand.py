# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


DEFAULT_SHAPE = (32, 32)
SHAPES = [tuple([32] * i) for i in range(6)]
ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]


@pytest.mark.xfail(reason="Integer data types are not yet supported. Float values could be nan")
@pytest.mark.parametrize("dtype", ALL_TYPES)
def test_tensor_dtype_and_value_range(device, dtype):
    tensor = ttnn.rand(DEFAULT_SHAPE, dtype=dtype, device=device)

    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)

    if is_ttnn_float_type(dtype):
        # TODO: Handle
        # assert (nan - nan) > 0.8 lead to test failure.
        torch_tensor = ttnn.to_torch(tensor)
        min_value = torch.min(torch_tensor).item()
        max_value = torch.max(torch_tensor).item()
        assert max_value - min_value > 0.8
    else:
        torch_tensor = ttnn.to_torch(tensor)
        assert torch.min(torch_tensor).item() == 0
        assert torch.max(torch_tensor).item() == 1


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


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_rand_with_layout(device, layout):
    size = DEFAULT_SHAPE
    tensor = ttnn.rand(size, device=device, layout=layout)

    assert tensor.layout == layout
    assert tuple(tensor.shape) == tuple(size)


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

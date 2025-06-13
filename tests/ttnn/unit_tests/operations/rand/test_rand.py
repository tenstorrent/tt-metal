# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn


DEFAULT_SHAPE = (32, 32)
SHAPES = [tuple([32] * i) for i in range(2, 6)]
ALL_TYPES = [dtype for dtype, _ in ttnn.DataType.__entries.values() if dtype != ttnn.DataType.INVALID]


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


@pytest.mark.xfail(reason="ttnn.bfloat8_b, ttnn.bfloat4_b data types are not yet supported.")
@pytest.mark.parametrize("dtype", ALL_TYPES)
def test_tensor_dtype_and_value_range(device, dtype):
    tensor = ttnn.rand(DEFAULT_SHAPE, dtype=dtype, device=device)

    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)

    if is_ttnn_float_type(dtype):
        assert ttnn.min(tensor) >= 0.0
        assert ttnn.max(tensor) < 1.0
    elif dtype == ttnn.int32:
        assert ttnn.min(tensor) == -1
        assert ttnn.max(tensor) == 1
    else:
        assert ttnn.min(tensor) == 0
        assert ttnn.max(tensor) == 1


def test_rand_defaults():
    tensor = ttnn.rand(DEFAULT_SHAPE)

    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert tensor.storage_type() == ttnn.StorageType.HOST
    assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


@pytest.mark.parametrize("shapes", SHAPES)
def test_rand_shapes(shapes):
    tensor = ttnn.rand(shapes)
    assert tuple(tensor.shape) == tuple(shapes)


@pytest.mark.parametrize("layout", [ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT])
def test_rand_with_layout(layout):
    size = DEFAULT_SHAPE
    tensor = ttnn.rand(size, layout=layout)

    assert tensor.layout == layout
    assert tuple(tensor.shape) == tuple(size)


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_rand_with_memory_config(device, mem_config):
    tensor = ttnn.rand(DEFAULT_SHAPE, device=device, memory_config=mem_config)
    assert tensor.memory_config() == mem_config
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


def test_rand_invalid_args():
    """
    Passing invalid args should raise TypeError.
    """

    with pytest.raises(TypeError):
        # expected list or tuple
        ttnn.rand(5)

    with pytest.raises(TypeError):
        # expected positive dim values
        ttnn.rand([2, -1])

    with pytest.raises(TypeError):
        # expected ttnn.LAYOUT type
        ttnn.rand([2, 2], layout="ROW_MAJOR")

    with pytest.raises(TypeError):
        # expected  ttnn.MemoryConfig type
        ttnn.rand([2, 2], memory_config="DRAM")

    with pytest.raises(TypeError):
        # expected  ttnn.Device type
        ttnn.rand([2, 2], device="WORMHOLE")

    with pytest.raises(TypeError):
        # expected  ttnn.DataType type
        ttnn.rand([2, 2], dtype="ttnn.bfloat16")

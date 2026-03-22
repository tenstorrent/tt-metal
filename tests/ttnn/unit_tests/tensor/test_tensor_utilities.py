# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn
from models.common.utility_functions import is_wormhole_b0, is_blackhole


@pytest.mark.parametrize(
    "dtype, expected_tile_size",
    [
        (ttnn.bfloat16, 2048),
        (ttnn.float32, 4096),
        (ttnn.bfloat8_b, 1088),  # (256 * 4) data + (16 * 4) exponents
        (ttnn.bfloat4_b, 576),  # (128 * 4) data + (16 * 4) exponents
        (ttnn.uint16, 2048),
        (ttnn.uint32, 4096),
    ],
)
def test_tile_size(dtype, expected_tile_size):
    assert ttnn.tile_size(dtype) == expected_tile_size


@pytest.mark.parametrize(
    "dtype, expected_element_size",
    [
        (ttnn.bfloat16, 2),
        (ttnn.float32, 4),
        (ttnn.uint16, 2),
        (ttnn.uint32, 4),
    ],
)
def test_element_size(dtype, expected_element_size):
    assert ttnn.element_size(dtype) == expected_element_size


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b])
def test_element_size_throws_for_block_floats(dtype):
    """Block float formats have shared exponents, so per-element size is undefined."""
    with pytest.raises(Exception):
        ttnn.element_size(dtype)


def test_dram_alignment(device):
    alignment = ttnn.get_dram_alignment()
    assert (alignment & (alignment - 1)) == 0, "DRAM alignment must be a power of 2"
    if is_wormhole_b0():
        assert alignment == 32
    elif is_blackhole():
        assert alignment == 64


def test_l1_alignment(device):
    alignment = ttnn.get_l1_alignment()
    assert (alignment & (alignment - 1)) == 0, "L1 alignment must be a power of 2"
    assert alignment == 16


@pytest.mark.parametrize(
    "shape, dtype, layout, memory_config",
    [
        ((1, 1, 32, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1, 32, 64), ttnn.float32, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1, 32, 64), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1, 32, 64), ttnn.bfloat4_b, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
        ((1, 1, 16, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, ttnn.DRAM_MEMORY_CONFIG),
    ],
)
def test_buffer_page_size(shape, dtype, layout, memory_config, device):
    torch_tensor = torch.randn(shape, dtype=torch.float32)
    t = ttnn.to_device(
        ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout), device=device, memory_config=memory_config
    )
    page_size = t.buffer_page_size()
    assert page_size > 0

    if layout == ttnn.TILE_LAYOUT:
        assert page_size == ttnn.tile_size(dtype)
    else:
        # Row-major: stick size = width * element_size
        assert page_size == shape[-1] * ttnn.element_size(dtype)


@pytest.mark.parametrize(
    "shape, dtype, layout, expected_pages",
    [
        ((1, 1, 64, 64), ttnn.bfloat16, ttnn.TILE_LAYOUT, 4),  # 2x2 tiles
        ((1, 1, 64, 64), ttnn.float32, ttnn.TILE_LAYOUT, 4),
        ((1, 1, 64, 64), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, 4),
        ((1, 1, 16, 64), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, 16),  # 16 sticks
    ],
)
def test_buffer_num_pages(shape, dtype, layout, expected_pages, device):
    torch_tensor = torch.randn(shape, dtype=torch.float32)
    t = ttnn.to_device(
        ttnn.from_torch(torch_tensor, dtype=dtype, layout=layout), device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    assert t.buffer_num_pages() == expected_pages


@pytest.mark.parametrize(
    "memory_config, alignment_fn",
    [
        (ttnn.DRAM_MEMORY_CONFIG, ttnn.get_dram_alignment),
        (ttnn.L1_MEMORY_CONFIG, ttnn.get_l1_alignment),
    ],
)
def test_buffer_aligned_page_size(memory_config, alignment_fn, device):
    torch_tensor = torch.randn((1, 1, 32, 32), dtype=torch.float32)
    t = ttnn.to_device(
        ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
        device=device,
        memory_config=memory_config,
    )
    aligned_size = t.buffer_aligned_page_size()
    page_size = t.buffer_page_size()
    assert aligned_size >= page_size
    assert aligned_size % alignment_fn() == 0


@pytest.mark.parametrize(
    "dtype, expected",
    [
        (ttnn.bfloat16, 2),
        (ttnn.float32, 4),
    ],
)
def test_tensor_element_size_instance_method(dtype, expected, device):
    torch_tensor = torch.randn((1, 1, 32, 32), dtype=torch.float32)
    t = ttnn.to_device(
        ttnn.from_torch(torch_tensor, dtype=dtype, layout=ttnn.TILE_LAYOUT),
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    assert t.element_size() == expected

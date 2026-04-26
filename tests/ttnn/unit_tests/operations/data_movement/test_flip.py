# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc


def random_torch_tensor(dtype, shape):
    if dtype == ttnn.int32:
        return torch.randint(-(2**31), 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.uint32:
        return torch.randint(0, 2**31, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        return torch.rand(shape, dtype=torch.float32)
    return torch.rand(shape, dtype=torch.bfloat16)


# ── basic correctness (4D, TILE_LAYOUT — tile-aligned shapes only) ────────────
# TILE_LAYOUT pads tensors to multiples of 32; non-tile-aligned shapes
# must use ROW_MAJOR_LAYOUT (see test_flip_non_tile_aligned below).


@pytest.mark.parametrize(
    "shape, dims",
    [
        ((1, 1, 32, 64), [3]),
        ((1, 1, 32, 64), [2]),
        ((1, 1, 32, 64), [2, 3]),
        ((2, 4, 32, 64), [1]),
        ((2, 4, 32, 64), [0]),
        ((1, 1, 32, 32), [2, 3]),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_flip(device, shape, dims, dtype):
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_input, dims)

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype)
    output_tensor = ttnn.flip(input_tensor, dims)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output, output_tensor)


# ── bfloat8_b (lossy dtype — pcc only) ───────────────────────────────────────


def test_flip_bfloat8(device):
    torch.manual_seed(2005)
    shape = (1, 1, 32, 64)
    torch_input = torch.rand(shape, dtype=torch.float32)
    torch_output = torch.flip(torch_input, [3])

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=ttnn.bfloat8_b)
    output_tensor = ttnn.flip(input_tensor, [3])
    output_tensor = ttnn.to_torch(output_tensor)

    assert_with_pcc(torch_output, output_tensor, 0.9999)


# ── non-tile-aligned shapes (ROW_MAJOR_LAYOUT) ────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims",
    [
        ((1, 1, 33, 65), [3]),
        ((2, 3, 17, 48), [2, 3]),
        ((1, 5, 40, 100), [2]),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32, ttnn.int32])
def test_flip_non_tile_aligned(device, shape, dims, dtype):
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_input, dims)

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
    output_tensor = ttnn.flip(input_tensor, dims)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output, output_tensor)


# ── negative dims ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims",
    [
        # tile-aligned
        ((1, 1, 32, 64), [-1]),
        ((1, 1, 32, 64), [-2]),
        ((1, 1, 32, 64), [-1, -2]),
        ((2, 4, 32, 64), [-3]),
        # non-tile-aligned — use ROW_MAJOR_LAYOUT
        ((1, 1, 33, 65), [-1]),
        ((2, 3, 17, 48), [-1, -2]),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_flip_negative_dims(device, shape, dims, dtype):
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_input, dims)

    # Use ROW_MAJOR_LAYOUT for non-tile-aligned shapes to avoid padding issues
    h, w = shape[-2], shape[-1]
    layout = ttnn.TILE_LAYOUT if (h % 32 == 0 and w % 32 == 0) else ttnn.ROW_MAJOR_LAYOUT

    input_tensor = ttnn.from_torch(torch_input, layout=layout, device=device, dtype=dtype)
    output_tensor = ttnn.flip(input_tensor, dims)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output, output_tensor)


# ── duplicate dims (NOT supported — kernel raises) ────────────────────────────


@pytest.mark.parametrize(
    "shape, dims",
    [
        ((1, 1, 32, 64), [3, 3]),
        ((2, 4, 32, 64), [2, 2]),
        ((1, 1, 32, 64), [-1, -1]),
    ],
)
def test_flip_duplicate_dims_raises(device, shape, dims):
    torch_input = torch.rand(shape, dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match="Duplicate dimension"):
        ttnn.flip(input_tensor, dims)


# ── 5D tensor ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape, dims",
    [
        # tile-aligned
        ((2, 3, 4, 32, 64), [4]),
        ((2, 3, 4, 32, 64), [3, 4]),
        ((1, 2, 3, 32, 64), [2, 3, 4]),
        # non-tile-aligned
        ((2, 3, 4, 33, 65), [4]),
        ((1, 1370, 1, 3, 32), [1]),
        ((1, 197, 1, 3, 48), [3, 4]),
    ],
)
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_flip_5d(device, shape, dims, dtype):
    torch.manual_seed(2005)
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_input, dims)

    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, dtype=dtype)
    output_tensor = ttnn.flip(input_tensor, dims)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output, output_tensor)


# ── memory configs (DRAM and L1 — sharded not supported by design) ────────────


@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.int32])
def test_flip_memory_config(device, memory_config, dtype):
    torch.manual_seed(2005)
    shape = (1, 1, 32, 64)
    torch_input = random_torch_tensor(dtype, shape)
    torch_output = torch.flip(torch_input, [3])

    input_tensor = ttnn.from_torch(
        torch_input, layout=ttnn.TILE_LAYOUT, device=device, dtype=dtype, memory_config=memory_config
    )
    output_tensor = ttnn.flip(input_tensor, [3], memory_config=memory_config)
    output_tensor = ttnn.to_torch(output_tensor)

    assert_equal(torch_output, output_tensor)


# ── error / validation ────────────────────────────────────────────────────────


def test_flip_empty_dims_raises(device):
    """Empty dims list must raise — the op has TT_FATAL(!dims.empty())."""
    torch_input = torch.rand((1, 1, 32, 64), dtype=torch.bfloat16)
    input_tensor = ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match="Flip dimensions cannot be empty"):
        ttnn.flip(input_tensor, [])


# ── program cache ─────────────────────────────────────────────────────────────


def test_flip_program_cache(device):
    torch.manual_seed(2005)
    shape = (1, 1, 32, 64)
    num_iters = 3

    # Prepare all inputs up front
    torch_results = []
    input_tensors = []
    for _ in range(num_iters):
        torch_input = torch.rand(shape, dtype=torch.bfloat16)
        torch_results.append(torch.flip(torch_input, [3]))
        input_tensors.append(ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT, device=device))

    base_count = None
    for i in range(num_iters):
        with device.cache_entries_counter.measure():
            output = ttnn.flip(input_tensors[i], [3])
        output = ttnn.to_torch(output)
        assert_equal(torch_results[i], output)
        if i == 0:
            base_count = device.cache_entries_counter.total
        else:
            assert device.cache_entries_counter.total == base_count, "program cache entries differ on same configs"

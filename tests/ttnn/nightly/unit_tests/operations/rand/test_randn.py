# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
from loguru import logger

from tests.ttnn.unit_tests.operations.test_utils import (
    get_compute_kernel_options,
    compute_kernel_options,
    compute_kernel_ids,
)

DEFAULT_SHAPE = (32, 32)
SHAPES = [tuple([32] * i) for i in range(6)]


def check_standard_normal_distribution(data: torch.Tensor, tt_dtype) -> bool:
    n = data.numel()

    if n < 10000:
        logger.warning("[Warning] A meaningful analysis requires at least 10000 samples.")
        if n < 2:
            logger.warning("[Error] Cannot perform test with less than 2 data points.")
            return False

    mean = data.mean().item()
    std = data.std(unbiased=False).item()

    if tt_dtype == ttnn.float32:
        mean_tol = 0.05
        std_tol = 0.05
    else:
        mean_tol = 0.08
        std_tol = 0.10

    if abs(mean) > mean_tol:
        logger.warning("Mean too far from 0: mean={}, tol={}", mean, mean_tol)
        return False
    if abs(std - 1.0) > std_tol:
        logger.warning("Std too far from 1: std={}, tol={}", std, std_tol)
        return False

    absx = data.abs()
    within1 = (absx <= 1.0).to(torch.float32).mean().item()
    within2 = (absx <= 2.0).to(torch.float32).mean().item()
    within3 = (absx <= 3.0).to(torch.float32).mean().item()

    if not (0.60 <= within1 <= 0.75):
        logger.warning("within1 out of range: {}", within1)
        return False
    if not (0.90 <= within2 <= 0.99):
        logger.warning("within2 out of range: {}", within2)
        return False
    if not (0.98 <= within3 <= 1.00):
        logger.warning("within3 out of range: {}", within3)
        return False

    return True


def test_randn_defaults(device):
    tensor = ttnn.randn(DEFAULT_SHAPE, device=device)

    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert tensor.storage_type() == ttnn.StorageType.DEVICE
    assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


@pytest.mark.parametrize("shapes", SHAPES)
def test_randn_shapes(device, shapes):
    tensor = ttnn.randn(shapes, device=device)
    assert tuple(tensor.shape) == tuple(shapes)


@pytest.mark.parametrize("dim", [i for i in range(32)])
def test_randn_dims(dim, device):
    shape = (dim, dim)
    tensor = ttnn.randn(shape, device=device)
    assert tuple(tensor.shape) == tuple(shape)


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_randn_with_memory_config(device, mem_config):
    tensor = ttnn.randn(DEFAULT_SHAPE, device=device, memory_config=mem_config)
    assert tensor.memory_config() == mem_config
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT])
def test_randn_dtype_layout_and_distribution(device, dtype, layout):
    shape = (1024, 1024)

    tensor = ttnn.randn(shape, device=device, dtype=dtype, layout=layout, seed=1234)

    assert tensor.layout == layout
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(shape)

    torch_tensor = ttnn.to_torch(tensor)
    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values!"
    assert check_standard_normal_distribution(torch_tensor, dtype), "Random values do not look standard normal!"


def test_randn_invalid_args(device):
    with pytest.raises(TypeError):
        ttnn.randn(5, device=device)

    with pytest.raises(TypeError):
        ttnn.randn([2, -1], device=device)

    with pytest.raises(TypeError):
        ttnn.randn([2, 2], device=device, layout="ROW_MAJOR")

    with pytest.raises(TypeError):
        ttnn.randn([2, 2], device=device, memory_config="DRAM")

    with pytest.raises(TypeError):
        ttnn.randn([2, 2], device="WORMHOLE")

    with pytest.raises(TypeError):
        ttnn.randn([2, 2], device=device, dtype="ttnn.bfloat16")


@pytest.mark.parametrize("shape", [[2, 32, 32, 16]])
@pytest.mark.parametrize("seed", [1234])
@pytest.mark.parametrize("dtype", [ttnn.float32])
def test_randn_callback(shape, seed, dtype, device):
    num_program_cache_entries_list = []
    for i in range(2):
        ttnn.randn(shape, device=device, dtype=dtype, layout=ttnn.TILE_LAYOUT, seed=seed + i)
        ttnn.empty([1, 1, 32, 32], ttnn.bfloat16, ttnn.TILE_LAYOUT, device)
        num_program_cache_entries_list.append(device.num_program_cache_entries())

    logger.info(f"num_program_cache_entries_list={num_program_cache_entries_list}")
    assert num_program_cache_entries_list[0] > 0
    assert num_program_cache_entries_list[0] == num_program_cache_entries_list[1]


@pytest.mark.parametrize("shape", [[512, 512], [5, 8, 70, 40]])
@pytest.mark.parametrize("dtype", [ttnn.float32])
@pytest.mark.parametrize("compute_kernel_options", compute_kernel_options, ids=compute_kernel_ids)
def test_randn_with_compute_kernel_options(shape, dtype, device, compute_kernel_options):
    compute_kernel_config = get_compute_kernel_options(compute_kernel_options)
    if compute_kernel_config is None:
        pytest.skip("Kernel option is not available")

    tensor = ttnn.randn(
        shape,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        seed=1234,
        compute_kernel_config=compute_kernel_config,
    )

    assert list(tensor.shape) == list(shape)
    assert tensor.dtype == dtype

    torch_tensor = ttnn.to_torch(tensor)
    assert check_standard_normal_distribution(torch_tensor, dtype), "Random values do not look standard normal!"

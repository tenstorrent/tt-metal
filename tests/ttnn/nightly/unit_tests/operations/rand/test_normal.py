# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import pytest
import ttnn
import torch
import math


DEFAULT_SHAPE = (32, 32)
# Use a large shape so statistical checks are meaningful
STAT_SHAPE = (1024, 1024)


def is_ttnn_float_type(tt_dtype) -> bool:
    match tt_dtype:
        case ttnn.bfloat16 | ttnn.float32 | ttnn.bfloat8_b | ttnn.bfloat4_b:
            return True
        case _:
            return False


def check_normal_distribution(data, expected_mean=0.0, expected_std=1.0, sigma_threshold=5.0, tolerance_pct=5.0):
    """
    Checks that sample mean and std are close to expected values.
    Mean is checked via z-test (within sigma_threshold standard errors).
    Std is checked via relative error (within tolerance_pct percent).
    """
    n = data.numel()
    assert n >= 1000, f"Need at least 1000 samples for a meaningful check, got {n}"

    flat = data.detach().cpu().float().flatten()
    sample_mean = flat.mean().item()
    sample_std = flat.std().item()

    if expected_std == 0.0:
        return (flat - expected_mean).abs().max().item() < 1e-3

    # z-test: sample mean should be within sigma_threshold standard errors
    se_mean = expected_std / math.sqrt(n)
    mean_ok = abs(sample_mean - expected_mean) < sigma_threshold * se_mean
    std_ok = abs(sample_std - expected_std) / expected_std * 100 < tolerance_pct

    return mean_ok and std_ok


# --- default behaviour ---


def test_normal_defaults(device):
    tensor = ttnn.normal(DEFAULT_SHAPE, device=device)

    assert tensor.dtype == ttnn.bfloat16
    assert tensor.layout == ttnn.TILE_LAYOUT
    assert tensor.storage_type() == ttnn.StorageType.DEVICE
    assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


# --- shape ---


@pytest.mark.parametrize("shape", [tuple([32] * i) for i in range(1, 6)])
def test_normal_shapes(device, shape):
    tensor = ttnn.normal(shape, device=device)
    assert tuple(tensor.shape) == tuple(shape)


# --- dtype ---


@pytest.mark.parametrize("dtype", [ttnn.bfloat16, ttnn.float32])
def test_normal_dtype(device, dtype):
    tensor = ttnn.normal(DEFAULT_SHAPE, device=device, dtype=dtype)
    assert tensor.dtype == dtype
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


# --- layout ---


@pytest.mark.parametrize("layout", [ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT])
def test_normal_layout(device, layout):
    tensor = ttnn.normal(DEFAULT_SHAPE, device=device, layout=layout)
    assert tensor.layout == layout


# --- memory config ---


@pytest.mark.parametrize("mem_config", [ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG])
def test_normal_memory_config(device, mem_config):
    tensor = ttnn.normal(DEFAULT_SHAPE, device=device, memory_config=mem_config)
    assert tensor.memory_config() == mem_config
    assert tuple(tensor.shape) == tuple(DEFAULT_SHAPE)


# --- distribution checks ---


@pytest.mark.parametrize("mean,stddev", [(0.0, 1.0), (5.0, 2.0), (-3.0, 0.5)])
def test_normal_distribution(device, mean, stddev):
    tensor = ttnn.normal(STAT_SHAPE, device=device, dtype=ttnn.float32, mean=mean, std=stddev, seed=42)
    torch_tensor = ttnn.to_torch(tensor)

    assert not torch.isnan(torch_tensor).any(), "Tensor contains NaN values"
    assert check_normal_distribution(
        torch_tensor, expected_mean=mean, expected_std=stddev
    ), f"Distribution check failed for mean={mean}, std={stddev}"


def test_normal_std_zero(device):
    """std=0 produces a constant tensor equal to mean."""
    mean = 3.5
    tensor = ttnn.normal(DEFAULT_SHAPE, device=device, dtype=ttnn.float32, mean=mean, std=0.0, seed=0)
    torch_tensor = ttnn.to_torch(tensor).float()
    assert torch.allclose(torch_tensor, torch.full(torch_tensor.shape, mean), atol=1e-2)


def test_normal_no_nan(device):
    tensor = ttnn.normal(STAT_SHAPE, device=device, dtype=ttnn.float32, seed=1)
    torch_tensor = ttnn.to_torch(tensor)
    assert not torch.isnan(torch_tensor).any()
    assert not torch.isinf(torch_tensor).any()


# --- seed reproducibility ---


def test_normal_seed_reproducibility(device):
    t1 = ttnn.normal(DEFAULT_SHAPE, device=device, dtype=ttnn.float32, seed=123)
    t2 = ttnn.normal(DEFAULT_SHAPE, device=device, dtype=ttnn.float32, seed=123)
    assert torch.equal(ttnn.to_torch(t1), ttnn.to_torch(t2)), "Same seed should produce identical tensors"


def test_normal_different_seeds_differ(device):
    t1 = ttnn.normal(DEFAULT_SHAPE, device=device, dtype=ttnn.float32, seed=1)
    t2 = ttnn.normal(DEFAULT_SHAPE, device=device, dtype=ttnn.float32, seed=2)
    assert not torch.equal(ttnn.to_torch(t1), ttnn.to_torch(t2)), "Different seeds should produce different tensors"


# --- invalid args ---


def test_normal_invalid_args(device):
    with pytest.raises(TypeError):
        # shape must be a list or tuple
        ttnn.normal(5, device=device)

    with pytest.raises(TypeError):
        # layout must be ttnn.Layout
        ttnn.normal(DEFAULT_SHAPE, device=device, layout="TILE")

    with pytest.raises(TypeError):
        # memory_config must be ttnn.MemoryConfig
        ttnn.normal(DEFAULT_SHAPE, device=device, memory_config="DRAM")

    with pytest.raises(TypeError):
        # dtype must be ttnn.DataType
        ttnn.normal(DEFAULT_SHAPE, device=device, dtype="bfloat16")

    with pytest.raises(TypeError):
        # device must be ttnn.Device / MeshDevice
        ttnn.normal(DEFAULT_SHAPE, device="WORMHOLE")

    with pytest.raises(Exception):
        # negative std must be rejected
        ttnn.normal(DEFAULT_SHAPE, device=device, std=-1.0)

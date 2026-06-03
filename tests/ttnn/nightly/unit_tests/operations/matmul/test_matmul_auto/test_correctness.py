# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for matmul_auto."""

from __future__ import annotations

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import check_with_pcc

# fmt: off
BASIC_SHAPES = [
    (1, 32, 32, 32),
    (1, 512, 512, 512),
    (1, 1024, 1024, 1024),
]

TALL_SHAPES = [
    (1, 2048, 1024, 32),
    (1, 4096, 512, 64),
]

WIDE_SHAPES = [
    (1, 32, 1024, 2048),
    (1, 64, 512, 4096),
]

LLM_SHAPES = [
    (1, 128, 4096, 4096),
    (1, 128, 4096, 11008),
    (1, 128, 11008, 4096),
    (1, 32, 4096, 4096),
    (1, 2048, 128, 2048),
    (1, 128, 2048, 128),
]

BATCHED_SHAPES = [
    (2, 128, 128, 128),
    (4, 256, 256, 256),
    (8, 512, 512, 512),
    (7, 384, 1024, 1024),
]

NON_POWER_OF_2_SHAPES = [
    (1, 384, 1024, 1024),
    (1, 768, 3072, 768),
]

EDGE_CASE_SHAPES = [
    (1, 32, 32, 32),
    (16, 32, 32, 32),
]

ALL_SHAPES = (
    BASIC_SHAPES
    + TALL_SHAPES
    + WIDE_SHAPES
    + LLM_SHAPES
    + BATCHED_SHAPES
    + NON_POWER_OF_2_SHAPES
    + EDGE_CASE_SHAPES
)
# fmt: on

pytestmark = pytest.mark.requires_wormhole_b0


def test_public_api_imports_are_callable():
    """Public imports should expose the matmul_auto function."""
    from ttnn._experimental.auto_config import matmul_auto

    assert callable(matmul_auto)
    assert callable(ttnn.matmul_auto)


def test_host_weight_with_device_argument(device):
    """Host input B should move to the requested device before execution."""
    from ttnn._experimental.auto_config import matmul_auto

    torch_a = torch.randn(1, 64, 128, dtype=torch.float32)
    torch_b = torch.randn(128, 64, dtype=torch.float32)
    torch_output = torch.matmul(torch_a, torch_b)

    input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    tt_output = matmul_auto(input_a, input_b, device=device)
    output = ttnn.to_torch(tt_output)

    passed, msg = check_with_pcc(torch_output, output, pcc=0.99)
    assert passed, f"Host-weight PCC check failed: {msg}"


@pytest.mark.parametrize("batch,m,k,n", ALL_SHAPES)
def test_output_correctness(device, batch, m, k, n):
    """Test that matmul_auto produces correct results for all shapes."""
    from ttnn._experimental.auto_config import matmul_auto

    torch_a = torch.randn(batch, m, k, dtype=torch.float32)
    torch_b = torch.randn(batch, k, n, dtype=torch.float32) if batch > 1 else torch.randn(k, n, dtype=torch.float32)
    torch_output = torch.matmul(torch_a, torch_b)

    input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = matmul_auto(input_a, input_b)
    output = ttnn.to_torch(tt_output)

    passed, msg = check_with_pcc(torch_output, output, pcc=0.99)
    assert passed, f"PCC check failed for shape ({batch}, {m}, {k}, {n}): {msg}"


@pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:2])
def test_output_with_bias(device, batch, m, k, n):
    """Test matmul_auto with bias."""
    from ttnn._experimental.auto_config import matmul_auto

    torch_a = torch.randn(batch, m, k, dtype=torch.float32)
    torch_b = torch.randn(k, n, dtype=torch.float32)
    torch_bias = torch.randn(1, n, dtype=torch.float32)
    torch_output = torch.matmul(torch_a, torch_b) + torch_bias

    input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    bias = ttnn.from_torch(torch_bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = matmul_auto(input_a, input_b, bias=bias)
    output = ttnn.to_torch(tt_output)

    passed, msg = check_with_pcc(torch_output, output, pcc=0.98)
    assert passed, f"PCC check failed for shape ({batch}, {m}, {k}, {n}) with bias: {msg}"


@pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:2])
def test_bfloat8_b_dtype(device, batch, m, k, n):
    """Test matmul_auto with bfloat8_b data type."""
    from ttnn._experimental.auto_config import matmul_auto

    torch_a = torch.randn(batch, m, k, dtype=torch.float32)
    torch_b = torch.randn(k, n, dtype=torch.float32)
    torch_output = torch.matmul(torch_a, torch_b)

    input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)

    tt_output = matmul_auto(input_a, input_b)
    output = ttnn.to_torch(tt_output)

    passed, msg = check_with_pcc(torch_output, output, pcc=0.97)
    assert passed, f"PCC check failed for bfloat8_b shape ({batch}, {m}, {k}, {n}): {msg}"


@pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:2])
def test_config_selection_no_errors(device, batch, m, k, n):
    """Test that config selection doesn't error."""
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    torch_a = torch.randn(batch, m, k, dtype=torch.float32)
    torch_b = torch.randn(k, n, dtype=torch.float32)

    input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    selector = MatmulAutoConfig()
    result = selector.select(input_a, input_b)

    assert result.selected_config is not None
    assert result.selected_config.is_valid
    assert result.selected_config.config_family in (
        "MultiCast1D",
        "MultiCast2D",
        "Reuse",
        "DRAMSharded",
        "BatchedDRAMSharded",
        "MinimalMatmul",
        "MultiCore",
    )
    assert result.selected_config.score > 0
    assert len(result.all_candidates) > 0


@pytest.mark.parametrize("batch,m,k,n", BASIC_SHAPES[:2])
def test_cache_hit_on_repeat(device, batch, m, k, n):
    """Test that repeated calls with same signature hit the cache."""
    from ttnn._experimental.auto_config.config_cache import ConfigCache
    from ttnn._experimental.auto_config.matmul_auto import MatmulAutoConfig

    torch_a = torch.randn(batch, m, k, dtype=torch.float32)
    torch_b = torch.randn(k, n, dtype=torch.float32)

    input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    cache = ConfigCache(cache_dir="/tmp/test_matmul_auto_cache")
    cache.clear()

    selector = MatmulAutoConfig(cache=cache)

    result1 = selector.select(input_a, input_b)
    assert not result1.cache_hit

    result2 = selector.select(input_a, input_b)
    assert result2.cache_hit

    cache.clear()


def test_features_contain_all_keys(device):
    """Test that features dict contains all expected keys."""
    from ttnn._experimental.auto_config.feature_extraction import extract_matmul_features

    torch_a = torch.randn(1, 128, 256, dtype=torch.float32)
    torch_b = torch.randn(256, 512, dtype=torch.float32)
    input_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    input_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    features = extract_matmul_features(input_a, input_b)

    required_keys = [
        "M",
        "K",
        "N",
        "batch_size_a",
        "batch_size_b",
        "M_tiles",
        "K_tiles",
        "N_tiles",
        "dtype_a",
        "dtype_b",
        "layout_a",
        "layout_b",
        "is_a_sharded",
        "is_b_sharded",
        "arch",
        "grid_x",
        "grid_y",
        "num_cores",
        "is_multi_device",
        "num_devices",
        "transpose_a",
        "transpose_b",
        "has_bias",
        "has_activation",
    ]
    for key in required_keys:
        assert key in features, f"Missing feature key: {key}"


def test_tile_alignment_validation():
    """Test tile alignment validation."""
    from ttnn._experimental.auto_config.constraint_validator import validate_tile_alignment

    is_valid, _ = validate_tile_alignment({"M": 128, "K": 256, "N": 512})
    assert is_valid

    is_valid, _ = validate_tile_alignment({"M": 100, "K": 256, "N": 512})
    assert not is_valid


def test_subblock_validation():
    """Test subblock parameter validation."""
    from ttnn._experimental.auto_config.constraint_validator import validate_subblock_params

    class MockConfig:
        per_core_M = 4
        per_core_N = 2
        out_subblock_h = 2
        out_subblock_w = 2
        in0_block_w = 2

    config = MockConfig()
    is_valid, _ = validate_subblock_params(config, "MultiCast1D")
    assert is_valid

    config.out_subblock_h = 8
    config.out_subblock_w = 2
    is_valid, _ = validate_subblock_params(config, "MultiCast1D")
    assert not is_valid

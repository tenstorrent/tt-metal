# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the RMSNorm1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. RMSNorm1D class matches PyTorch/HuggingFace reference model
3. RMSNorm1D correctly rejects TG/Galaxy devices
"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.rmsnorm.rmsnorm_1d import (
    RMSNorm1D,
    RMSNorm1DConfig,
    _compute_norm_core_grid,
    _create_sharded_norm_program_config,
)
from models.common.utility_functions import comp_allclose, comp_pcc

# ============================================================================
# Weight Caching - Avoid expensive weight loading per test
# ============================================================================

_CACHED_NORM_WEIGHTS: dict[str, torch.Tensor] = {}


def _get_or_init_norm_weights(model_name: str, reference_norm) -> None:
    """Initialize RMSNorm weights once per model, cache and reuse across tests."""
    if model_name not in _CACHED_NORM_WEIGHTS:
        logger.info(f"\033[33m[cache miss]\033[0m Initializing weights for {model_name}")
        with torch.no_grad():
            _CACHED_NORM_WEIGHTS[model_name] = torch.randn_like(reference_norm.weight)
    else:
        logger.info(f"\033[32m[cache hit]\033[0m Reusing cached weights for {model_name}")

    # Load cached weights into model
    with torch.no_grad():
        reference_norm.weight.copy_(_CACHED_NORM_WEIGHTS[model_name])


def _get_or_create_synthetic_weight(dim: int, seed: int = 1234) -> torch.Tensor:
    """Get or create synthetic RMSNorm weight, cached by dimension."""
    cache_key = f"synthetic_dim{dim}"
    if cache_key not in _CACHED_NORM_WEIGHTS:
        logger.info(f"\033[33m[cache miss]\033[0m Creating synthetic weight for dim={dim}")
        torch.manual_seed(seed)
        _CACHED_NORM_WEIGHTS[cache_key] = torch.randn(dim, dtype=torch.bfloat16)
    return _CACHED_NORM_WEIGHTS[cache_key]


# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_rmsnorm_1d_config_creation():
    """Test that RMSNorm1DConfig dataclass can be created with explicit values."""
    mock_mesh_device = MagicMock()
    mock_tt_ccl = MagicMock()
    mock_weight = MagicMock()

    config = RMSNorm1DConfig(
        weight=mock_weight,
        eps=1e-6,
        add_unit_offset=True,
        mesh_device=mock_mesh_device,
        tt_ccl=mock_tt_ccl,
        max_batch_size=64,
    )

    assert config.weight == mock_weight
    assert config.eps == 1e-6
    assert config.add_unit_offset is True
    assert config.mesh_device == mock_mesh_device
    assert config.tt_ccl == mock_tt_ccl
    assert config.max_batch_size == 64


def test_rmsnorm_1d_config_defaults():
    """Test that RMSNorm1DConfig has sensible defaults."""
    config = RMSNorm1DConfig(weight=MagicMock())

    # Check defaults
    assert config.eps == 1e-5
    assert config.add_unit_offset is False
    assert config.max_batch_size == 32
    assert config.decode_in_sharded is True
    assert config.decode_out_sharded is True

    # Optional fields default to None
    assert config.mesh_device is None
    assert config.tt_ccl is None
    assert config.prefill_distributed is None
    assert config.decode_program_config is None


def test_rmsnorm_1d_config_power_user_overrides():
    """Test that RMSNorm1DConfig accepts power-user overrides for program configs."""
    mock_prg_config = MagicMock()
    mock_mem_config = MagicMock()

    config = RMSNorm1DConfig(
        weight=MagicMock(),
        decode_program_config=mock_prg_config,
        decode_memory_config=mock_mem_config,
        decode_in_sharded=False,
    )

    assert config.decode_program_config == mock_prg_config
    assert config.decode_memory_config == mock_mem_config
    assert config.decode_in_sharded is False


def test_compute_norm_core_grid():
    """Test _compute_norm_core_grid helper function."""
    # dim=4096 -> 128 tiles -> should find a grid that divides 128
    grid = _compute_norm_core_grid(4096)
    assert grid.num_cores > 0
    assert 128 % grid.num_cores == 0

    # dim=8192 -> 256 tiles -> should find a grid that divides 256
    grid = _compute_norm_core_grid(8192)
    assert grid.num_cores > 0
    assert 256 % grid.num_cores == 0


def test_create_sharded_norm_program_config():
    """Test _create_sharded_norm_program_config helper function."""
    dim = 4096
    grid = ttnn.CoreGrid(x=8, y=4)  # 32 cores
    tile_padded_batch_rows = 32

    config = _create_sharded_norm_program_config(dim, grid, tile_padded_batch_rows)

    # Just verify the config is created successfully
    assert isinstance(config, ttnn.LayerNormShardedMultiCoreProgramConfig)


# ============================================================================
# Integration Tests - Device required
# ============================================================================


# HuggingFace model paths
LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_90B = "meta-llama/Llama-3.2-90B-Vision-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
MIXTRAL_MOE = "mistralai/Mixtral-8x7B-Instruct-v0.1"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN25_CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"
QWEN3_32B = "Qwen/Qwen3-32B"
DEEPSEEK_R1_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

_slow = pytest.mark.slow


def _list_rmsnorm_1d_test_cases() -> list[pytest.param]:
    """
    Test cases from rmsnorm_1d_testcases.csv.

    These cover various models across 1D topologies (1x1, 1x2, 1x8).
    Includes both decode and prefill modes with various seq_len values.

    Parameters:
    - mesh_shape: (cluster_shape_x, cluster_shape_y) tuple
    - input_shape: (x0, x1, x2, x3) - x1 > 1 for vision encoder norms
    - mode: "decode" or "prefill"
    - dim: hidden dimension
    - eps: epsilon for numerical stability
    - in_sharded: whether input is sharded
    - out_sharded: whether output is sharded
    - is_distributed: whether distributed path is used
    - model_name: HuggingFace model path
    - pcc: minimum PCC threshold
    """
    # fmt: off
    return [
        # === Fast tests (minimal coverage set) ===
        # Single device (1x1) - local paths
        pytest.param((1, 1), (1, 1, 128, 2048), "prefill", 2048, 1e-5, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-128-1B"),
        pytest.param((1, 1), (1, 1, 32, 2048), "decode", 2048, 1e-5, True, True, False, LLAMA_1B, 0.999, id="1x1-decode-32-1B"),
        pytest.param((1, 1), (1, 1, 128, 4096), "prefill", 4096, 1e-5, False, False, False, LLAMA_8B, 0.999, id="1x1-prefill-128-8B"),
        pytest.param((1, 1), (1, 1, 32, 4096), "decode", 4096, 1e-5, True, True, False, LLAMA_8B, 0.999, id="1x1-decode-32-8B"),
        # Multi-device (1x2) - local paths
        pytest.param((1, 2), (1, 1, 128, 4096), "prefill", 4096, 1e-5, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-128-8B"),
        pytest.param((1, 2), (1, 1, 32, 4096), "decode", 4096, 1e-5, True, True, False, LLAMA_8B, 0.999, id="1x2-decode-32-8B"),
        # Multi-device (1x8) - includes distributed path
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-5, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-128-8B"),
        pytest.param((1, 8), (1, 1, 32, 8192), "decode", 8192, 1e-5, True, True, False, LLAMA_70B, 0.999, id="1x8-decode-32-70B"),
        pytest.param((1, 8), (1, 1, 128, 8192), "prefill", 8192, 1e-5, False, False, True, LLAMA_70B, 0.999, id="1x8-prefill-128-70B-dist"),
        # Non-Llama models
        pytest.param((1, 2), (1, 1, 128, 3584), "prefill", 3584, 1e-6, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-128-Qwen2.5-7B"),
        pytest.param((1, 8), (1, 1, 128, 5120), "prefill", 5120, 1e-6, False, False, True, QWEN3_32B, 0.999, id="1x8-prefill-128-Qwen3-32B-dist"),
        # Vision encoder norms (x_shape_1 > 1, dim=128)
        pytest.param((1, 1), (1, 4, 6528, 128), "decode", 128, 1e-5, False, False, False, LLAMA_11B, 0.999, id="1x1-decode-6528x4-11B-vision"),
        pytest.param((1, 1), (1, 16, 128, 128), "prefill", 128, 1e-5, False, False, False, LLAMA_11B, 0.999, id="1x1-prefill-128x16-11B-vision"),
        # === Slow tests (full coverage from CSV) ===
        # Mesh 1x1
        pytest.param((1, 1), (1, 1, 128, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-128-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-32-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 1024, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-1024-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 2048, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-2048-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 4096, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-4096-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 8192, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-8192-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32, 2048), "decode", 2048, 1e-05, True, True, False, LLAMA_1B, 0.999, id="1x1-decode-32-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 16384, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-16384-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32768, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-32768-1B", marks=_slow),
        pytest.param((1, 1), (1, 1, 128, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x1-prefill-128-3B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x1-prefill-32-3B", marks=_slow),
        pytest.param((1, 1), (1, 1, 1024, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x1-prefill-1024-3B", marks=_slow),
        pytest.param((1, 1), (1, 1, 2048, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x1-prefill-2048-3B", marks=_slow),
        pytest.param((1, 1), (1, 1, 4096, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x1-prefill-4096-3B", marks=_slow),
        pytest.param((1, 1), (1, 1, 8192, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x1-prefill-8192-3B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32, 3072), "decode", 3072, 1e-05, True, True, False, LLAMA_3B, 0.999, id="1x1-decode-32-3B", marks=_slow),
        pytest.param((1, 1), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x1-prefill-128-8B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x1-prefill-32-8B", marks=_slow),
        pytest.param((1, 1), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x1-prefill-1024-8B", marks=_slow),
        pytest.param((1, 1), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x1-prefill-2048-8B", marks=_slow),
        pytest.param((1, 1), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x1-prefill-4096-8B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_8B, 0.999, id="1x1-decode-32-8B", marks=_slow),
        pytest.param((1, 1), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x1-prefill-128-7B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x1-prefill-32-7B", marks=_slow),
        pytest.param((1, 1), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x1-prefill-1024-7B", marks=_slow),
        pytest.param((1, 1), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x1-prefill-2048-7B", marks=_slow),
        pytest.param((1, 1), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x1-prefill-4096-7B", marks=_slow),
        pytest.param((1, 1), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, MISTRAL_7B, 0.999, id="1x1-decode-32-7B", marks=_slow),
        # Mesh 1x2
        pytest.param((1, 2), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-128-1B", marks=_slow),
        pytest.param((1, 2), (1, 4, 6528, 128), "decode", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-decode-6528x4-1B-vision", marks=_slow),
        pytest.param((1, 2), (1, 16, 128, 128), "prefill", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-128x16-1B-vision", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-32-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_11B, 0.999, id="1x2-decode-32-1B", marks=_slow),
        pytest.param((1, 2), (1, 16, 16, 128), "decode", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-decode-16x16-1B-vision", marks=_slow),
        pytest.param((1, 2), (1, 4, 16, 128), "decode", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-decode-16x4-1B-vision", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-128-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-32-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 1024, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-1024-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 2048, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-2048-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 4096, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-4096-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 8192, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-8192-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 2048), "decode", 2048, 1e-05, True, True, False, LLAMA_1B, 0.999, id="1x2-decode-32-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 16384, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-16384-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32768, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-32768-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-128-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-32-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 1024, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-1024-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 2048, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-2048-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 4096, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-4096-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 8192, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-8192-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 3072), "decode", 3072, 1e-05, True, True, False, LLAMA_3B, 0.999, id="1x2-decode-32-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 16384, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-16384-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32768, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-32768-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-128-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-32-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-1024-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-2048-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-4096-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 8192, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-8192-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_8B, 0.999, id="1x2-decode-32-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 16384, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-16384-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32768, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-32768-8B", marks=_slow),
        pytest.param((1, 2), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-1024-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-2048-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-4096-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 8192, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-8192-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 16384, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-16384-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32768, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-32768-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x2-prefill-128-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x2-prefill-32-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x2-prefill-1024-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x2-prefill-2048-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x2-prefill-4096-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, MISTRAL_7B, 0.999, id="1x2-decode-32-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN2_7B, 0.999, id="1x2-prefill-128-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN2_7B, 0.999, id="1x2-prefill-32-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 1024, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN2_7B, 0.999, id="1x2-prefill-1024-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 2048, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN2_7B, 0.999, id="1x2-prefill-2048-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 4096, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN2_7B, 0.999, id="1x2-prefill-4096-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 3584), "decode", 3584, 1e-06, True, True, False, QWEN2_7B, 0.999, id="1x2-decode-32-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 2560), "prefill", 5120, 1e-05, False, False, True, DEEPSEEK_R1_14B, 0.999, id="1x2-prefill-128-DeepSeek-dist", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 2560), "prefill", 5120, 1e-05, False, False, True, DEEPSEEK_R1_14B, 0.999, id="1x2-prefill-32-DeepSeek-dist", marks=_slow),
        pytest.param((1, 2), (1, 1, 1024, 2560), "prefill", 5120, 1e-05, False, False, True, DEEPSEEK_R1_14B, 0.999, id="1x2-prefill-1024-DeepSeek-dist", marks=_slow),
        pytest.param((1, 2), (1, 1, 2048, 2560), "prefill", 5120, 1e-05, False, False, True, DEEPSEEK_R1_14B, 0.999, id="1x2-prefill-2048-DeepSeek-dist", marks=_slow),
        pytest.param((1, 2), (1, 1, 4096, 2560), "prefill", 5120, 1e-05, False, False, True, DEEPSEEK_R1_14B, 0.999, id="1x2-prefill-4096-DeepSeek-dist", marks=_slow),
        pytest.param((1, 2), (1, 1, 8192, 2560), "prefill", 5120, 1e-05, False, False, True, DEEPSEEK_R1_14B, 0.999, id="1x2-prefill-8192-DeepSeek-dist", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 5120), "decode", 5120, 1e-05, True, True, False, DEEPSEEK_R1_14B, 0.999, id="1x2-decode-32-DeepSeek", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-128-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-32-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 1024, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-1024-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 2048, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-2048-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 4096, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-4096-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 8192, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-8192-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 3584), "decode", 3584, 1e-06, True, True, False, QWEN25_7B, 0.999, id="1x2-decode-32-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 16384, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-16384-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 32768, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-32768-7B", marks=_slow),
        # Mesh 1x8
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-128-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 6528, 128), "decode", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-decode-6528-1B", marks=_slow),
        pytest.param((1, 8), (1, 4, 128, 128), "prefill", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-128x4-1B-vision", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-32-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_11B, 0.999, id="1x8-decode-32-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 4, 128), "decode", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-decode-4-1B", marks=_slow),
        pytest.param((1, 8), (1, 32, 4, 128), "decode", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-decode-4x32-1B-vision", marks=_slow),
        pytest.param((1, 8), (1, 4, 4, 128), "decode", 128, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-decode-4x4-1B-vision", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 1024), "prefill", 8192, 1e-05, False, False, True, LLAMA_90B, 0.999, id="1x8-prefill-128-90B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 6528, 128), "decode", 128, 1e-05, False, False, False, LLAMA_90B, 0.999, id="1x8-decode-6528-90B", marks=_slow),
        pytest.param((1, 8), (1, 8, 128, 128), "prefill", 128, 1e-05, False, False, False, LLAMA_90B, 0.999, id="1x8-prefill-128x8-90B-vision", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 1024), "prefill", 8192, 1e-05, False, False, True, LLAMA_90B, 0.999, id="1x8-prefill-32-90B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 8192), "decode", 8192, 1e-05, True, True, False, LLAMA_90B, 0.999, id="1x8-decode-32-90B", marks=_slow),
        pytest.param((1, 8), (1, 1, 8, 128), "decode", 128, 1e-05, False, False, False, LLAMA_90B, 0.999, id="1x8-decode-8-90B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-128-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-32-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-1024-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-2048-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-4096-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 8192, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-8192-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 2048), "decode", 2048, 1e-05, True, True, False, LLAMA_1B, 0.999, id="1x8-decode-32-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 16384, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-16384-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32768, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-32768-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-128-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-32-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-1024-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-2048-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-4096-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 8192, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-8192-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 3072), "decode", 3072, 1e-05, True, True, False, LLAMA_3B, 0.999, id="1x8-decode-32-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 16384, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-16384-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32768, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-32768-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-128-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-32-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-1024-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-2048-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-4096-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 8192, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-8192-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_8B, 0.999, id="1x8-decode-32-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 16384, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-16384-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32768, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-32768-8B", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-1024-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-2048-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-4096-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 8192, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-8192-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 16384, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-16384-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32768, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-32768-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 1024), "prefill", 8192, 1e-05, False, False, True, LLAMA_70B, 0.999, id="1x8-prefill-128-70B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 1024), "prefill", 8192, 1e-05, False, False, True, LLAMA_70B, 0.999, id="1x8-prefill-32-70B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 1024), "prefill", 8192, 1e-05, False, False, True, LLAMA_70B, 0.999, id="1x8-prefill-1024-70B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 1024), "prefill", 8192, 1e-05, False, False, True, LLAMA_70B, 0.999, id="1x8-prefill-2048-70B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 1024), "prefill", 8192, 1e-05, False, False, True, LLAMA_70B, 0.999, id="1x8-prefill-4096-70B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 8192), "decode", 8192, 1e-05, True, True, False, LLAMA_70B, 0.999, id="1x8-decode-32-70B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 1024), "prefill", 8192, 1e-06, False, False, True, QWEN25_72B, 0.999, id="1x8-prefill-128-Qwen2.5-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 1024), "prefill", 8192, 1e-06, False, False, True, QWEN25_72B, 0.999, id="1x8-prefill-32-Qwen2.5-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 1024), "prefill", 8192, 1e-06, False, False, True, QWEN25_72B, 0.999, id="1x8-prefill-1024-Qwen2.5-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 1024), "prefill", 8192, 1e-06, False, False, True, QWEN25_72B, 0.999, id="1x8-prefill-2048-Qwen2.5-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 1024), "prefill", 8192, 1e-06, False, False, True, QWEN25_72B, 0.999, id="1x8-prefill-4096-Qwen2.5-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 8192, 1024), "prefill", 8192, 1e-06, False, False, True, QWEN25_72B, 0.999, id="1x8-prefill-8192-Qwen2.5-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 8192), "decode", 8192, 1e-06, True, True, False, QWEN25_72B, 0.999, id="1x8-decode-32-Qwen2.5", marks=_slow),
        pytest.param((1, 8), (1, 1, 16384, 1024), "prefill", 8192, 1e-06, False, False, True, QWEN25_72B, 0.999, id="1x8-prefill-16384-Qwen2.5-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 640), "prefill", 5120, 1e-06, False, False, True, QWEN25_CODER_32B, 0.999, id="1x8-prefill-128-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 640), "prefill", 5120, 1e-06, False, False, True, QWEN25_CODER_32B, 0.999, id="1x8-prefill-32-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 640), "prefill", 5120, 1e-06, False, False, True, QWEN25_CODER_32B, 0.999, id="1x8-prefill-1024-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 640), "prefill", 5120, 1e-06, False, False, True, QWEN25_CODER_32B, 0.999, id="1x8-prefill-2048-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 640), "prefill", 5120, 1e-06, False, False, True, QWEN25_CODER_32B, 0.999, id="1x8-prefill-4096-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 5120), "decode", 5120, 1e-06, True, True, False, QWEN25_CODER_32B, 0.999, id="1x8-decode-32-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 640), "prefill", 5120, 1e-06, False, False, True, QWEN3_32B, 0.999, id="1x8-prefill-128-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 8, 128, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-128x8-32B-vision", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-128-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 640), "prefill", 5120, 1e-06, False, False, True, QWEN3_32B, 0.999, id="1x8-prefill-32-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 640), "prefill", 5120, 1e-06, False, False, True, QWEN3_32B, 0.999, id="1x8-prefill-1024-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 8, 1024, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-1024x8-32B-vision", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-1024-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 640), "prefill", 5120, 1e-06, False, False, True, QWEN3_32B, 0.999, id="1x8-prefill-2048-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 8, 2048, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-2048x8-32B-vision", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-2048-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 640), "prefill", 5120, 1e-06, False, False, True, QWEN3_32B, 0.999, id="1x8-prefill-4096-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 8, 4096, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-4096x8-32B-vision", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-4096-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 5120), "decode", 5120, 1e-06, True, True, False, QWEN3_32B, 0.999, id="1x8-decode-32-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 8, 128), "decode", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-decode-8-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 1, 128), "decode", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-decode-1-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 16384, 640), "prefill", 5120, 1e-06, False, False, True, QWEN3_32B, 0.999, id="1x8-prefill-16384-32B-dist", marks=_slow),
        pytest.param((1, 8), (1, 8, 16384, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-16384x8-32B-vision", marks=_slow),
        pytest.param((1, 8), (1, 1, 16384, 128), "prefill", 128, 1e-06, False, False, False, QWEN3_32B, 0.999, id="1x8-prefill-16384-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x8-prefill-128-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x8-prefill-32-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x8-prefill-1024-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x8-prefill-2048-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x8-prefill-4096-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, MISTRAL_7B, 0.999, id="1x8-decode-32-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, MIXTRAL_MOE, 0.999, id="1x8-prefill-128-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "prefill", 4096, 1e-05, False, False, False, MIXTRAL_MOE, 0.999, id="1x8-prefill-32-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 1024, 4096), "prefill", 4096, 1e-05, False, False, False, MIXTRAL_MOE, 0.999, id="1x8-prefill-1024-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 2048, 4096), "prefill", 4096, 1e-05, False, False, False, MIXTRAL_MOE, 0.999, id="1x8-prefill-2048-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 4096, 4096), "prefill", 4096, 1e-05, False, False, False, MIXTRAL_MOE, 0.999, id="1x8-prefill-4096-7B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, MIXTRAL_MOE, 0.999, id="1x8-decode-32-7B", marks=_slow),
    ]
    # fmt: on


def _list_rmsnorm_2d_unique_test_cases() -> list[pytest.param]:
    """
    Unique test cases from rmsnorm_2d_testcases.csv (not duplicated in rmsnorm_1d_testcases.csv).

    These are the 1x4 cluster shape cases for Llama-3.1-8B that only exist in the 2D file.
    Note: Despite being in the "2D" file, these are still 1D topologies (cluster_shape_x=1).
    """
    # fmt: off
    return [
        # === Fast tests (1x4 minimal coverage) ===
        pytest.param((1, 4), (1, 1, 128, 4096), "prefill", 4096, 1e-5, False, False, False, LLAMA_8B, 0.999, id="1x4-prefill-128-8B"),
        pytest.param((1, 4), (1, 1, 32, 4096), "decode", 4096, 1e-5, True, True, False, LLAMA_8B, 0.999, id="1x4-decode-32-8B"),
        # === Slow tests (full 1x4 coverage) ===
        pytest.param((1, 4), (1, 1, 32, 4096), "prefill", 4096, 1e-5, False, False, False, LLAMA_8B, 0.999, id="1x4-prefill-32-8B", marks=_slow),
        pytest.param((1, 4), (1, 1, 1024, 4096), "prefill", 4096, 1e-5, False, False, False, LLAMA_8B, 0.999, id="1x4-prefill-1024-8B", marks=_slow),
        pytest.param((1, 4), (1, 1, 2048, 4096), "prefill", 4096, 1e-5, False, False, False, LLAMA_8B, 0.999, id="1x4-prefill-2048-8B", marks=_slow),
        pytest.param((1, 4), (1, 1, 4096, 4096), "prefill", 4096, 1e-5, False, False, False, LLAMA_8B, 0.999, id="1x4-prefill-4096-8B", marks=_slow),
    ]
    # fmt: on


def _list_rmsnorm_1d_test_cases_from_distnorm() -> list[pytest.param]:
    """
    Test cases derived from rmsnorm_*d_testcases_by_dist_norm.csv (de-duplicated).

    These are cases collected from distribute_norm.py runs, representing
    actual production usage patterns. Many overlap with existing tests but
    include some unique mesh/dim combinations.

    Generated by: models/common/tests/modules/rmsnorm/dedup_dist_rmsnorm.py
    """
    # fmt: off
    return [
        # === Fast tests (minimal coverage from dist_norm runs) ===
        # DeepSeek 1x2 distributed prefill
        pytest.param((1, 2), (1, 1, 128, 2560), "prefill", 5120, 1e-05, False, False, True, DEEPSEEK_R1_14B, 0.999, id="1x2-prefill-128-DeepSeek-14B-dist"),
        # Qwen 1x2 non-distributed (eps=1e-06)
        pytest.param((1, 2), (1, 1, 128, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN25_7B, 0.999, id="1x2-prefill-128-Qwen25-7B"),
        # Qwen 1x8 distributed prefill
        pytest.param((1, 8), (1, 1, 128, 1024), "prefill", 8192, 1e-06, False, False, True, QWEN25_72B, 0.999, id="1x8-prefill-128-Qwen25-72B-dist"),
        # Llama 1x8 decode
        pytest.param((1, 8), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_8B, 0.999, id="1x8-decode-32-8B-dn"),

        # === Slow tests (full coverage) ===
        # DEEPSEEK_R1_14B
        pytest.param((1, 2), (1, 1, 32, 5120), "decode", 5120, 1e-05, True, True, False, DEEPSEEK_R1_14B, 0.999, id="1x2-decode-32-DeepSeek-14B", marks=_slow),

        # LLAMA_11B
        pytest.param((1, 2), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_11B, 0.999, id="1x2-decode-32-11B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x2-prefill-128-11B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_11B, 0.999, id="1x8-decode-32-11B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_11B, 0.999, id="1x8-prefill-128-11B", marks=_slow),

        # LLAMA_1B
        pytest.param((1, 1), (1, 1, 32, 2048), "decode", 2048, 1e-05, True, True, False, LLAMA_1B, 0.999, id="1x1-decode-32-1B-dn", marks=_slow),
        pytest.param((1, 1), (1, 1, 128, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x1-prefill-128-1B-dn", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 2048), "decode", 2048, 1e-05, True, True, False, LLAMA_1B, 0.999, id="1x2-decode-32-1B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x2-prefill-128-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 2048), "decode", 2048, 1e-05, True, True, False, LLAMA_1B, 0.999, id="1x8-decode-32-1B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 2048), "prefill", 2048, 1e-05, False, False, False, LLAMA_1B, 0.999, id="1x8-prefill-128-1B", marks=_slow),

        # LLAMA_3B
        pytest.param((1, 1), (1, 1, 32, 3072), "decode", 3072, 1e-05, True, True, False, LLAMA_3B, 0.999, id="1x1-decode-32-3B-dn", marks=_slow),
        pytest.param((1, 1), (1, 1, 128, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x1-prefill-128-3B-dn", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 3072), "decode", 3072, 1e-05, True, True, False, LLAMA_3B, 0.999, id="1x2-decode-32-3B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x2-prefill-128-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 3072), "decode", 3072, 1e-05, True, True, False, LLAMA_3B, 0.999, id="1x8-decode-32-3B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 3072), "prefill", 3072, 1e-05, False, False, False, LLAMA_3B, 0.999, id="1x8-prefill-128-3B", marks=_slow),

        # LLAMA_70B
        pytest.param((1, 8), (1, 1, 32, 8192), "decode", 8192, 1e-05, True, True, False, LLAMA_70B, 0.999, id="1x8-decode-32-70B-dn", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 1024), "prefill", 8192, 1e-05, False, False, True, LLAMA_70B, 0.999, id="1x8-prefill-128-70B-dist-dn", marks=_slow),

        # LLAMA_8B
        pytest.param((1, 1), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_8B, 0.999, id="1x1-decode-32-8B-dn", marks=_slow),
        pytest.param((1, 1), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x1-prefill-128-8B-dn", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, LLAMA_8B, 0.999, id="1x2-decode-32-8B-dn", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x2-prefill-128-8B-dn", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, LLAMA_8B, 0.999, id="1x8-prefill-128-8B-dn", marks=_slow),

        # MISTRAL_7B
        pytest.param((1, 1), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, MISTRAL_7B, 0.999, id="1x1-decode-32-7B-dn", marks=_slow),
        pytest.param((1, 1), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x1-prefill-128-7B-dn", marks=_slow),
        pytest.param((1, 2), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, MISTRAL_7B, 0.999, id="1x2-decode-32-7B-dn", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x2-prefill-128-7B-dn", marks=_slow),
        pytest.param((1, 8), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, MISTRAL_7B, 0.999, id="1x8-decode-32-7B-dn", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, MISTRAL_7B, 0.999, id="1x8-prefill-128-7B-dn", marks=_slow),

        # MIXTRAL_MOE
        pytest.param((1, 8), (1, 1, 32, 4096), "decode", 4096, 1e-05, True, True, False, MIXTRAL_MOE, 0.999, id="1x8-decode-32-MOE", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 4096), "prefill", 4096, 1e-05, False, False, False, MIXTRAL_MOE, 0.999, id="1x8-prefill-128-MOE", marks=_slow),

        # QWEN25_72B
        pytest.param((1, 8), (1, 1, 32, 8192), "decode", 8192, 1e-06, True, True, False, QWEN25_72B, 0.999, id="1x8-decode-32-Qwen25-72B", marks=_slow),

        # QWEN25_7B
        pytest.param((1, 2), (1, 1, 32, 3584), "decode", 3584, 1e-06, True, True, False, QWEN25_7B, 0.999, id="1x2-decode-32-Qwen25-7B", marks=_slow),

        # QWEN25_CODER_32B
        pytest.param((1, 8), (1, 1, 32, 5120), "decode", 5120, 1e-06, True, True, False, QWEN25_CODER_32B, 0.999, id="1x8-decode-32-Qwen25-Coder-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 640), "prefill", 5120, 1e-06, False, False, True, QWEN25_CODER_32B, 0.999, id="1x8-prefill-128-Qwen25-Coder-32B-dist", marks=_slow),

        # QWEN2_7B
        pytest.param((1, 2), (1, 1, 32, 3584), "decode", 3584, 1e-06, True, True, False, QWEN2_7B, 0.999, id="1x2-decode-32-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), (1, 1, 128, 3584), "prefill", 3584, 1e-06, False, False, False, QWEN2_7B, 0.999, id="1x2-prefill-128-Qwen2-7B", marks=_slow),

        # QWEN3_32B
        pytest.param((1, 8), (1, 1, 32, 5120), "decode", 5120, 1e-06, True, True, False, QWEN3_32B, 0.999, id="1x8-decode-32-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), (1, 1, 128, 640), "prefill", 5120, 1e-06, False, False, True, QWEN3_32B, 0.999, id="1x8-prefill-128-Qwen3-32B-dist", marks=_slow),
    ]
    # fmt: on


# ============================================================================
# CSV-based Parametrized Test
# ============================================================================


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 4), (1, 8)],
    ids=["1x1", "1x2", "1x4", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape,input_shard_shape,mode,dim,eps,in_sharded,out_sharded,is_distributed,model_name,pcc",
    _list_rmsnorm_1d_test_cases() + _list_rmsnorm_2d_unique_test_cases() + _list_rmsnorm_1d_test_cases_from_distnorm(),
)
def test_rmsnorm_1d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    input_shard_shape: tuple[int, int, int, int],
    mode: str,
    dim: int,
    eps: float,
    in_sharded: bool,
    out_sharded: bool,
    is_distributed: bool,
    model_name: str,
    pcc: float,
):
    """
    Test RMSNorm1D matches PyTorch reference for test cases from CSV files.

    Test cases are derived from:
    - rmsnorm_1d_testcases.csv (all cases)
    - rmsnorm_2d_testcases.csv (unique 1x4 cases only, duplicates excluded)
    """
    # Skip if mesh_shape doesn't match the current device
    if ttnn_mesh_device.shape != ttnn.MeshShape(*mesh_shape):
        pytest.skip(f"Test requires {mesh_shape} mesh, got {ttnn_mesh_device.shape}")

    seed = 1234
    torch.manual_seed(seed)

    # Create synthetic weights for RMSNorm reference.
    # This approach is used for all cases because:
    # 1. The test validates RMSNorm1D implementation correctness, not HF weight loading
    # 2. Production code loads weights from state_dict, not HF AutoModel
    # 3. The math is identical regardless of weight values
    # 4. Avoids HF model loading overhead and config complexity (e.g., MllamaConfig)
    # Weights are cached by dim to avoid regeneration across tests.
    norm_weight = _get_or_create_synthetic_weight(dim, seed)
    reference_norm = torch.nn.RMSNorm(dim, eps=eps).to(torch.bfloat16)
    reference_norm.weight.data.copy_(norm_weight)

    # Create input tensor with full hidden dimension
    # input_shard_shape = (x0, x1, x2, x3) where:
    # - x0, x1 = batch dimensions (x1 > 1 for vision encoder norms)
    # - x2 = sequence length
    # - x3 = per-device hidden dim (equals dim for non-distributed, dim/num_devices for distributed)
    # We always create full input shape using dim, then let prepare_input_tensor handle sharding
    torch_input = torch.randn(*input_shard_shape[:-1], dim, dtype=torch.bfloat16)

    # Create LazyWeights
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/rmsnorm_1d"))

    lazy_weight = LazyWeight(
        source=norm_weight,  # Raw [dim] shape - module reshapes internally
        dtype=ttnn.bfloat16,
        cache_dir_weight_name=(cache_dir, f"norm_weight_{model_name}_dim{dim}"),
    )

    # Construct RMSNorm1D
    # Happy path: standard cases with default eps and sharded decode
    # Power path: vision encoder (interleaved decode) or non-default eps
    is_default_eps = eps == 1e-5
    is_default_sharding = in_sharded and out_sharded

    # 1x4 mesh special case: Ring topology all_gather is not supported on 1x4 on WH LB/QB because
    # fabric cannot route between non-adjacent devices (e.g., D0 -> D3). We must
    # explicitly set prefill_distributed from test params to override auto-detection,
    # which would otherwise enable distributed prefill based on num_devices and dim.
    needs_explicit_distributed = mesh_shape == (1, 4)

    if is_default_eps and is_default_sharding and not needs_explicit_distributed:
        tt_model = RMSNorm1D(weight=lazy_weight)
    else:
        config = RMSNorm1DConfig(
            weight=lazy_weight,
            eps=eps,
            decode_in_sharded=in_sharded,
            decode_out_sharded=out_sharded,
            prefill_distributed=is_distributed if needs_explicit_distributed else None,
        )
        tt_model = RMSNorm1D.from_config(config)

    # Verify config matches expected sharding behavior from CSV
    cfg = tt_model.config
    if mode == "prefill":
        # Prefill uses interleaved memory (not sharded)
        assert not in_sharded, f"Prefill should have in_sharded=False, got {in_sharded}"
        assert not out_sharded, f"Prefill should have out_sharded=False, got {out_sharded}"
    else:  # decode
        # Decode never uses distributed path
        assert not is_distributed, f"Decode should have is_distributed=False, got {is_distributed}"
        # Verify decode sharding config matches CSV
        assert cfg.decode_in_sharded == in_sharded, f"Expected decode_in_sharded={in_sharded}"
        assert cfg.decode_out_sharded == out_sharded, f"Expected decode_out_sharded={out_sharded}"
        # in_sharded/out_sharded should match (decode either shards both or neither externally)
        assert (
            in_sharded == out_sharded
        ), f"Decode in_sharded and out_sharded should match, got in={in_sharded}, out={out_sharded}"

    # Run TT model - wrap input in LazyWeight, forward() handles conversion
    tt_input = LazyWeight(source=torch_input, dtype=ttnn.bfloat16)
    tt_output = tt_model.forward(tt_input, mode=mode)
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Run reference model
    # RMSNorm operates on last dimension, so reshape to 2D, apply, reshape back
    # For distributed cases, torch_input already has full dim (not sharded)
    original_shape = torch_input.shape
    torch_input_2d = torch_input.reshape(-1, original_shape[-1])  # (batch*heads*seq, dim)
    with torch.no_grad():
        reference_output_2d = reference_norm(torch_input_2d)
    reference_output = reference_output_2d.reshape(original_shape)

    # For distributed cases with sharded output, we may need to adjust comparison
    # The TT output is auto-composed (gathered) so should match full reference

    # Compare
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"RMSNorm1D vs HF reference: {pcc_message}")

    assert passing, f"RMSNorm1D output does not meet PCC requirement {pcc}: {pcc_message}."
    logger.info(
        f"RMSNorm1D vs HF reference: PASSED for model={model_name}, mode={mode}, shard_shape={input_shard_shape}"
    )


# Get HF model name from environment variable or use default
HF_MODEL_NAME = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 8)],
    ids=["1x8"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [1, 128])
def test_rmsnorm_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len: int):
    """
    Test RMSNorm1D.from_model_args() factory method.
    """
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    seed = 1234
    torch.manual_seed(seed)
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    # Create ModelArgs
    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=batch_size, max_seq_len=512, cache_hf=True)
    model_args.n_layers = 1

    # Load HF model for reference
    hf_model_name = HF_MODEL_NAME
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_hidden_layers = 1

    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    first_layer = hf_model.model.layers[0]
    reference_norm = first_layer.input_layernorm
    _get_or_init_norm_weights(hf_model_name, reference_norm)

    # Get state_dict
    state_dict = hf_model.state_dict()

    # Create TT_CCL
    tt_ccl = TT_CCL(ttnn_mesh_device)

    # Get model config for sharded configs
    model_config = model_args.get_model_config()
    sharded_program_config = model_config.get("SHARDED_NORM_ATTN_PRGM_CFG")
    sharded_output_config = model_config.get("SHARDED_ATTN_INPUT_MEMCFG")

    # Build RMSNorm1D via from_model_args
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/rmsnorm_1d_from_args"))
    tt_model = RMSNorm1D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=cache_dir,
        layer_num=0,
        weight_key="input_layernorm",
        state_dict_prefix="model.layers.0.",
        sharded_program_config=sharded_program_config,
        sharded_output_config=sharded_output_config,
    )

    # Run TT model
    dim = config.hidden_size
    torch_input = torch.randn(batch_size, 1, seq_len, dim, dtype=torch.bfloat16)
    tt_input = LazyWeight(source=torch_input, dtype=ttnn.bfloat16)
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    tt_output = tt_model.forward(tt_input, mode=mode)
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Run reference
    torch_input_squeezed = torch_input.squeeze(1)
    with torch.no_grad():
        reference_output = reference_norm(torch_input_squeezed)
    reference_output = reference_output.unsqueeze(1)

    # Compare
    pcc = 0.999
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(f"RMSNorm1D (from_model_args) vs HF reference: {pcc_message}")
    assert passing, f"RMSNorm1D output does not meet PCC requirement {pcc}: {pcc_message}."


def test_rmsnorm_1d_rejects_galaxy():
    """Test that RMSNorm1D.from_model_args raises error for Galaxy devices."""
    mock_args = MagicMock()
    mock_args.is_galaxy = True

    with pytest.raises(ValueError, match="cannot be used for Galaxy devices"):
        RMSNorm1D.from_model_args(
            mesh_device=MagicMock(),
            tt_ccl=MagicMock(),
            args=mock_args,
            state_dict={},
            weight_cache_path=None,
            layer_num=0,
            weight_key="input_layernorm",
        )

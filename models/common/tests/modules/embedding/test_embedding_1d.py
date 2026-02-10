# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the Embedding1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. Embedding1D class matches torch.nn.Embedding reference
3. Embedding1D correctly rejects TG/Galaxy devices
4. from_model_args backward compatibility
"""

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.embedding.embedding_1d import Embedding1D, Embedding1DConfig
from models.common.modules.lazy_weight import LazyWeight
from models.common.utility_functions import comp_allclose, comp_pcc

# ============================================================================
# HF model name constants
# ============================================================================

LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN25_CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEEPSEEK_R1_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
QWEN3_32B = "Qwen/Qwen3-32B"
MIXTRAL_8X7B = "mistralai/Mixtral-8x7B-v0.1"


# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_embedding_1d_config_creation():
    """Test that Embedding1DConfig dataclass can be created with explicit values."""
    from unittest.mock import MagicMock

    mock_w = MagicMock()
    mock_device = MagicMock()

    config = Embedding1DConfig(
        weights=mock_w,
        mesh_device=mock_device,
        embed_scale=2.0,
        weights_dtype=ttnn.bfloat16,
    )

    assert config.weights is mock_w
    assert config.mesh_device is mock_device
    assert config.embed_scale == 2.0
    assert config.weights_dtype == ttnn.bfloat16


def test_embedding_1d_config_defaults():
    """Test that Embedding1DConfig has sensible defaults."""
    from unittest.mock import MagicMock

    config = Embedding1DConfig(weights=MagicMock())

    assert config.embed_scale == 1.0
    assert config.mesh_device is None
    assert config.weights_dtype is None
    assert config.weights_memcfg is None
    assert config.output_memcfg is None


def test_embedding_1d_config_embed_scale_override():
    """Test that embed_scale can be overridden for ScaledEmbedding use case."""
    from unittest.mock import MagicMock

    config = Embedding1DConfig(weights=MagicMock(), embed_scale=55.4256)
    assert config.embed_scale == 55.4256


# ============================================================================
# Weight caching
# ============================================================================

_CACHED_EMB_WEIGHTS: dict[str, torch.Tensor] = {}


def _get_or_init_embedding_weight(model_name: str, vocab_size: int, dim: int) -> torch.Tensor:
    """Initialize embedding weight once per model, cache and reuse across tests."""
    key = f"{model_name}_{vocab_size}_{dim}"
    if key not in _CACHED_EMB_WEIGHTS:
        logger.info(f"\033[33m[cache miss]\033[0m Initializing embedding weight for {key}")
        _CACHED_EMB_WEIGHTS[key] = torch.randn(vocab_size, dim, dtype=torch.bfloat16)
    else:
        logger.info(f"\033[32m[cache hit]\033[0m Reusing cached embedding weight for {key}")
    return _CACHED_EMB_WEIGHTS[key]


# ============================================================================
# Integration Tests - Require device
# ============================================================================

_slow = pytest.mark.slow


# Collected from embedding_1d_test_cases.csv (deduplicated, representative subset)
# Each case: (mesh_shape, vocab_size, dim, seq_len, embed_scale, hf_model_name)
# Vocab/dim derived from model, seq_len varies per collected case.
def _list_test_cases() -> list[pytest.param]:
    # fmt: off
    return [
        # === Fast tests (minimal coverage set) ===
        # Single device (1x1) - Llama 1B (vocab=128256, dim=2048)
        pytest.param((1, 1), 128256, 2048, 32, 1.0, LLAMA_1B, 0.999, id="1x1-32-1B"),
        pytest.param((1, 1), 128256, 2048, 128, 1.0, LLAMA_1B, 0.999, id="1x1-128-1B"),
        # N300 (1x2) - Llama 8B (vocab=128256, dim=4096)
        pytest.param((1, 2), 128256, 4096, 128, 1.0, LLAMA_8B, 0.999, id="1x2-128-8B"),
        pytest.param((1, 2), 128256, 4096, 32, 1.0, LLAMA_8B, 0.999, id="1x2-32-8B"),
        # T3K (1x8) - Llama 70B (vocab=128256, dim=8192)
        pytest.param((1, 8), 128256, 8192, 128, 1.0, LLAMA_70B, 0.999, id="1x8-128-70B"),
        pytest.param((1, 8), 128256, 8192, 32, 1.0, LLAMA_70B, 0.999, id="1x8-32-70B"),
        # Non-Llama models
        pytest.param((1, 1), 32768, 4096, 32, 1.0, MISTRAL_7B, 0.999, id="1x1-32-Mistral-7B"),
        pytest.param((1, 2), 152064, 3584, 128, 1.0, QWEN2_7B, 0.999, id="1x2-128-Qwen2-7B"),

        # === Slow tests (full coverage from CSV) ===
        # (1,1) Llama-3.2-1B
        pytest.param((1, 1), 128256, 2048, 1024, 1.0, LLAMA_1B, 0.999, id="1x1-1024-1B", marks=_slow),
        pytest.param((1, 1), 128256, 2048, 2048, 1.0, LLAMA_1B, 0.999, id="1x1-2048-1B", marks=_slow),
        pytest.param((1, 1), 128256, 2048, 4096, 1.0, LLAMA_1B, 0.999, id="1x1-4096-1B", marks=_slow),
        pytest.param((1, 1), 128256, 2048, 8192, 1.0, LLAMA_1B, 0.999, id="1x1-8192-1B", marks=_slow),
        # (1,1) Llama-3.2-3B
        pytest.param((1, 1), 128256, 3072, 32, 1.0, LLAMA_3B, 0.999, id="1x1-32-3B", marks=_slow),
        pytest.param((1, 1), 128256, 3072, 128, 1.0, LLAMA_3B, 0.999, id="1x1-128-3B", marks=_slow),
        pytest.param((1, 1), 128256, 3072, 1024, 1.0, LLAMA_3B, 0.999, id="1x1-1024-3B", marks=_slow),
        pytest.param((1, 1), 128256, 3072, 2048, 1.0, LLAMA_3B, 0.999, id="1x1-2048-3B", marks=_slow),
        pytest.param((1, 1), 128256, 3072, 4096, 1.0, LLAMA_3B, 0.999, id="1x1-4096-3B", marks=_slow),
        pytest.param((1, 1), 128256, 3072, 8192, 1.0, LLAMA_3B, 0.999, id="1x1-8192-3B", marks=_slow),
        # (1,1) Llama-3.1-8B
        pytest.param((1, 1), 128256, 4096, 32, 1.0, LLAMA_8B, 0.999, id="1x1-32-8B", marks=_slow),
        pytest.param((1, 1), 128256, 4096, 128, 1.0, LLAMA_8B, 0.999, id="1x1-128-8B", marks=_slow),
        pytest.param((1, 1), 128256, 4096, 1024, 1.0, LLAMA_8B, 0.999, id="1x1-1024-8B", marks=_slow),
        pytest.param((1, 1), 128256, 4096, 2048, 1.0, LLAMA_8B, 0.999, id="1x1-2048-8B", marks=_slow),
        pytest.param((1, 1), 128256, 4096, 4096, 1.0, LLAMA_8B, 0.999, id="1x1-4096-8B", marks=_slow),
        # (1,1) Mistral-7B
        pytest.param((1, 1), 32768, 4096, 128, 1.0, MISTRAL_7B, 0.999, id="1x1-128-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 32768, 4096, 1024, 1.0, MISTRAL_7B, 0.999, id="1x1-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 32768, 4096, 2048, 1.0, MISTRAL_7B, 0.999, id="1x1-2048-Mistral-7B", marks=_slow),
        pytest.param((1, 1), 32768, 4096, 4096, 1.0, MISTRAL_7B, 0.999, id="1x1-4096-Mistral-7B", marks=_slow),
        # (1,2) Llama-3.2-1B
        pytest.param((1, 2), 128256, 2048, 32, 1.0, LLAMA_1B, 0.999, id="1x2-32-1B", marks=_slow),
        pytest.param((1, 2), 128256, 2048, 128, 1.0, LLAMA_1B, 0.999, id="1x2-128-1B", marks=_slow),
        pytest.param((1, 2), 128256, 2048, 1024, 1.0, LLAMA_1B, 0.999, id="1x2-1024-1B", marks=_slow),
        pytest.param((1, 2), 128256, 2048, 2048, 1.0, LLAMA_1B, 0.999, id="1x2-2048-1B", marks=_slow),
        pytest.param((1, 2), 128256, 2048, 4096, 1.0, LLAMA_1B, 0.999, id="1x2-4096-1B", marks=_slow),
        pytest.param((1, 2), 128256, 2048, 8192, 1.0, LLAMA_1B, 0.999, id="1x2-8192-1B", marks=_slow),
        # (1,2) Llama-3.2-3B
        pytest.param((1, 2), 128256, 3072, 32, 1.0, LLAMA_3B, 0.999, id="1x2-32-3B", marks=_slow),
        pytest.param((1, 2), 128256, 3072, 128, 1.0, LLAMA_3B, 0.999, id="1x2-128-3B", marks=_slow),
        pytest.param((1, 2), 128256, 3072, 1024, 1.0, LLAMA_3B, 0.999, id="1x2-1024-3B", marks=_slow),
        pytest.param((1, 2), 128256, 3072, 2048, 1.0, LLAMA_3B, 0.999, id="1x2-2048-3B", marks=_slow),
        pytest.param((1, 2), 128256, 3072, 4096, 1.0, LLAMA_3B, 0.999, id="1x2-4096-3B", marks=_slow),
        pytest.param((1, 2), 128256, 3072, 8192, 1.0, LLAMA_3B, 0.999, id="1x2-8192-3B", marks=_slow),
        # (1,2) Llama-3.1-8B
        pytest.param((1, 2), 128256, 4096, 1024, 1.0, LLAMA_8B, 0.999, id="1x2-1024-8B", marks=_slow),
        pytest.param((1, 2), 128256, 4096, 2048, 1.0, LLAMA_8B, 0.999, id="1x2-2048-8B", marks=_slow),
        pytest.param((1, 2), 128256, 4096, 4096, 1.0, LLAMA_8B, 0.999, id="1x2-4096-8B", marks=_slow),
        pytest.param((1, 2), 128256, 4096, 8192, 1.0, LLAMA_8B, 0.999, id="1x2-8192-8B", marks=_slow),
        # (1,2) Llama-3.2-11B
        pytest.param((1, 2), 128256, 4096, 32, 1.0, LLAMA_11B, 0.999, id="1x2-32-11B", marks=_slow),
        pytest.param((1, 2), 128256, 4096, 128, 1.0, LLAMA_11B, 0.999, id="1x2-128-11B", marks=_slow),
        pytest.param((1, 2), 128256, 4096, 1024, 1.0, LLAMA_11B, 0.999, id="1x2-1024-11B", marks=_slow),
        pytest.param((1, 2), 128256, 4096, 2048, 1.0, LLAMA_11B, 0.999, id="1x2-2048-11B", marks=_slow),
        pytest.param((1, 2), 128256, 4096, 4096, 1.0, LLAMA_11B, 0.999, id="1x2-4096-11B", marks=_slow),
        pytest.param((1, 2), 128256, 4096, 8192, 1.0, LLAMA_11B, 0.999, id="1x2-8192-11B", marks=_slow),
        # (1,2) Mistral-7B
        pytest.param((1, 2), 32768, 4096, 32, 1.0, MISTRAL_7B, 0.999, id="1x2-32-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 32768, 4096, 128, 1.0, MISTRAL_7B, 0.999, id="1x2-128-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 32768, 4096, 1024, 1.0, MISTRAL_7B, 0.999, id="1x2-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 32768, 4096, 2048, 1.0, MISTRAL_7B, 0.999, id="1x2-2048-Mistral-7B", marks=_slow),
        pytest.param((1, 2), 32768, 4096, 4096, 1.0, MISTRAL_7B, 0.999, id="1x2-4096-Mistral-7B", marks=_slow),
        # (1,2) Qwen2-7B
        pytest.param((1, 2), 152064, 3584, 32, 1.0, QWEN2_7B, 0.999, id="1x2-32-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 152064, 3584, 1024, 1.0, QWEN2_7B, 0.999, id="1x2-1024-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 152064, 3584, 2048, 1.0, QWEN2_7B, 0.999, id="1x2-2048-Qwen2-7B", marks=_slow),
        pytest.param((1, 2), 152064, 3584, 4096, 1.0, QWEN2_7B, 0.999, id="1x2-4096-Qwen2-7B", marks=_slow),
        # (1,2) Qwen2.5-7B
        pytest.param((1, 2), 152064, 3584, 32, 1.0, QWEN25_7B, 0.999, id="1x2-32-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 152064, 3584, 128, 1.0, QWEN25_7B, 0.999, id="1x2-128-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 152064, 3584, 1024, 1.0, QWEN25_7B, 0.999, id="1x2-1024-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 152064, 3584, 2048, 1.0, QWEN25_7B, 0.999, id="1x2-2048-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 152064, 3584, 4096, 1.0, QWEN25_7B, 0.999, id="1x2-4096-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 152064, 3584, 8192, 1.0, QWEN25_7B, 0.999, id="1x2-8192-Qwen2.5-7B", marks=_slow),
        # (1,2) DeepSeek-R1-Distill-Qwen-14B
        pytest.param((1, 2), 152064, 5120, 32, 1.0, DEEPSEEK_R1_14B, 0.999, id="1x2-32-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 152064, 5120, 128, 1.0, DEEPSEEK_R1_14B, 0.999, id="1x2-128-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 152064, 5120, 1024, 1.0, DEEPSEEK_R1_14B, 0.999, id="1x2-1024-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 152064, 5120, 2048, 1.0, DEEPSEEK_R1_14B, 0.999, id="1x2-2048-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 152064, 5120, 4096, 1.0, DEEPSEEK_R1_14B, 0.999, id="1x2-4096-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 152064, 5120, 8192, 1.0, DEEPSEEK_R1_14B, 0.999, id="1x2-8192-DeepSeek-R1-14B", marks=_slow),
        # (1,8) Llama-3.2-1B
        pytest.param((1, 8), 128256, 2048, 32, 1.0, LLAMA_1B, 0.999, id="1x8-32-1B", marks=_slow),
        pytest.param((1, 8), 128256, 2048, 128, 1.0, LLAMA_1B, 0.999, id="1x8-128-1B", marks=_slow),
        pytest.param((1, 8), 128256, 2048, 1024, 1.0, LLAMA_1B, 0.999, id="1x8-1024-1B", marks=_slow),
        pytest.param((1, 8), 128256, 2048, 2048, 1.0, LLAMA_1B, 0.999, id="1x8-2048-1B", marks=_slow),
        pytest.param((1, 8), 128256, 2048, 4096, 1.0, LLAMA_1B, 0.999, id="1x8-4096-1B", marks=_slow),
        pytest.param((1, 8), 128256, 2048, 8192, 1.0, LLAMA_1B, 0.999, id="1x8-8192-1B", marks=_slow),
        # (1,8) Llama-3.2-3B
        pytest.param((1, 8), 128256, 3072, 32, 1.0, LLAMA_3B, 0.999, id="1x8-32-3B", marks=_slow),
        pytest.param((1, 8), 128256, 3072, 128, 1.0, LLAMA_3B, 0.999, id="1x8-128-3B", marks=_slow),
        pytest.param((1, 8), 128256, 3072, 1024, 1.0, LLAMA_3B, 0.999, id="1x8-1024-3B", marks=_slow),
        pytest.param((1, 8), 128256, 3072, 2048, 1.0, LLAMA_3B, 0.999, id="1x8-2048-3B", marks=_slow),
        pytest.param((1, 8), 128256, 3072, 4096, 1.0, LLAMA_3B, 0.999, id="1x8-4096-3B", marks=_slow),
        pytest.param((1, 8), 128256, 3072, 8192, 1.0, LLAMA_3B, 0.999, id="1x8-8192-3B", marks=_slow),
        # (1,8) Llama-3.1-8B
        pytest.param((1, 8), 128256, 4096, 32, 1.0, LLAMA_8B, 0.999, id="1x8-32-8B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 128, 1.0, LLAMA_8B, 0.999, id="1x8-128-8B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 1024, 1.0, LLAMA_8B, 0.999, id="1x8-1024-8B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 2048, 1.0, LLAMA_8B, 0.999, id="1x8-2048-8B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 4096, 1.0, LLAMA_8B, 0.999, id="1x8-4096-8B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 8192, 1.0, LLAMA_8B, 0.999, id="1x8-8192-8B", marks=_slow),
        # (1,8) Llama-3.2-11B
        pytest.param((1, 8), 128256, 4096, 32, 1.0, LLAMA_11B, 0.999, id="1x8-32-11B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 128, 1.0, LLAMA_11B, 0.999, id="1x8-128-11B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 1024, 1.0, LLAMA_11B, 0.999, id="1x8-1024-11B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 2048, 1.0, LLAMA_11B, 0.999, id="1x8-2048-11B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 4096, 1.0, LLAMA_11B, 0.999, id="1x8-4096-11B", marks=_slow),
        pytest.param((1, 8), 128256, 4096, 8192, 1.0, LLAMA_11B, 0.999, id="1x8-8192-11B", marks=_slow),
        # (1,8) Llama-3.3-70B
        pytest.param((1, 8), 128256, 8192, 1024, 1.0, LLAMA_70B, 0.999, id="1x8-1024-70B", marks=_slow),
        pytest.param((1, 8), 128256, 8192, 2048, 1.0, LLAMA_70B, 0.999, id="1x8-2048-70B", marks=_slow),
        pytest.param((1, 8), 128256, 8192, 4096, 1.0, LLAMA_70B, 0.999, id="1x8-4096-70B", marks=_slow),
        # (1,8) Mistral-7B
        pytest.param((1, 8), 32768, 4096, 32, 1.0, MISTRAL_7B, 0.999, id="1x8-32-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 32768, 4096, 128, 1.0, MISTRAL_7B, 0.999, id="1x8-128-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 32768, 4096, 1024, 1.0, MISTRAL_7B, 0.999, id="1x8-1024-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 32768, 4096, 2048, 1.0, MISTRAL_7B, 0.999, id="1x8-2048-Mistral-7B", marks=_slow),
        pytest.param((1, 8), 32768, 4096, 4096, 1.0, MISTRAL_7B, 0.999, id="1x8-4096-Mistral-7B", marks=_slow),
        # (1,8) Qwen2.5-72B
        pytest.param((1, 8), 152064, 8192, 32, 1.0, QWEN25_72B, 0.999, id="1x8-32-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 152064, 8192, 128, 1.0, QWEN25_72B, 0.999, id="1x8-128-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 152064, 8192, 1024, 1.0, QWEN25_72B, 0.999, id="1x8-1024-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 152064, 8192, 2048, 1.0, QWEN25_72B, 0.999, id="1x8-2048-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 152064, 8192, 4096, 1.0, QWEN25_72B, 0.999, id="1x8-4096-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 152064, 8192, 8192, 1.0, QWEN25_72B, 0.999, id="1x8-8192-Qwen2.5-72B", marks=_slow),
        # (1,8) Qwen2.5-Coder-32B
        pytest.param((1, 8), 152064, 5120, 32, 1.0, QWEN25_CODER_32B, 0.999, id="1x8-32-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 152064, 5120, 128, 1.0, QWEN25_CODER_32B, 0.999, id="1x8-128-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 152064, 5120, 1024, 1.0, QWEN25_CODER_32B, 0.999, id="1x8-1024-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 152064, 5120, 2048, 1.0, QWEN25_CODER_32B, 0.999, id="1x8-2048-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 152064, 5120, 4096, 1.0, QWEN25_CODER_32B, 0.999, id="1x8-4096-Qwen2.5-Coder-32B", marks=_slow),
        # (1,8) Qwen3-32B
        pytest.param((1, 8), 151936, 5120, 32, 1.0, QWEN3_32B, 0.999, id="1x8-32-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 151936, 5120, 128, 1.0, QWEN3_32B, 0.999, id="1x8-128-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 151936, 5120, 1024, 1.0, QWEN3_32B, 0.999, id="1x8-1024-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 151936, 5120, 2048, 1.0, QWEN3_32B, 0.999, id="1x8-2048-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 151936, 5120, 4096, 1.0, QWEN3_32B, 0.999, id="1x8-4096-Qwen3-32B", marks=_slow),
    ]
    # fmt: on


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape,vocab_size,dim,seq_len,embed_scale,hf_model_name,pcc",
    _list_test_cases(),
)
def test_embedding_1d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mesh_shape,
    vocab_size,
    dim,
    seq_len,
    embed_scale,
    hf_model_name,
    pcc,
):
    """
    Test Embedding1D constructed via direct APIs matches torch.nn.Embedding reference.

    Configs pulled from embedding_1d_test_cases.csv (deduplicated).
    """
    seed = 42
    torch.manual_seed(seed)

    # Get or create deterministic random embedding weight
    weight_torch = _get_or_init_embedding_weight(hf_model_name, vocab_size, dim)

    # Reference: torch.nn.Embedding
    ref_embedding = torch.nn.Embedding(vocab_size, dim)
    with torch.no_grad():
        ref_embedding.weight.copy_(weight_torch)

    # Input: random token IDs
    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int64)

    # Reference output
    with torch.no_grad():
        ref_output = ref_embedding(input_ids)  # [1, seq_len, dim]
        if embed_scale != 1.0:
            ref_output = ref_output * embed_scale

    # Build Embedding1D TT model
    # Weight shape for TTNN: [1, 1, vocab_size, dim] to match TTTv1 convention
    weight_4d = weight_torch.unsqueeze(0).unsqueeze(0)  # [1, 1, vocab_size, dim]

    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lazy_weights = LazyWeight(
        source=weight_4d,
        dtype=ttnn.bfloat16,
    )

    tt_model = Embedding1D(weights=lazy_weights, embed_scale=embed_scale)

    # Input: reshape to [1, 1, 1, seq_len] uint32 for TTNN, wrap in LazyWeight
    input_ids_4d = input_ids.reshape(1, 1, 1, seq_len).to(torch.int32)
    lazy_input = LazyWeight(
        source=input_ids_4d,
        dtype=ttnn.uint32,
    )

    tt_output = tt_model.forward(lazy_input)
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Reshape for comparison: tt output is [1, 1, seq_len, dim/num_devices] per shard
    # auto_compose concatenates shards -> [1, 1, seq_len, dim]
    # ref output is [1, seq_len, dim]
    tt_output_torch = tt_output_torch.squeeze(0)  # remove leading batch dim if present

    # Handle shape differences: ref is [1, seq_len, dim], tt might be [1, seq_len, padded_dim]
    if tt_output_torch.shape[-1] > ref_output.shape[-1]:
        tt_output_torch = tt_output_torch[..., : ref_output.shape[-1]]

    # Ensure shapes match
    if tt_output_torch.dim() == 3 and ref_output.dim() == 2:
        ref_output = ref_output.unsqueeze(0)
    elif tt_output_torch.dim() == 2 and ref_output.dim() == 3:
        tt_output_torch = tt_output_torch.unsqueeze(0)

    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc)
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"Embedding1D vs reference: {pcc_message}")

    assert passing, f"Embedding1D output does not meet PCC requirement {pcc}: {pcc_message}."
    logger.info(f"Embedding1D vs reference: PASSED for seq_len={seq_len}, vocab={vocab_size}, dim={dim}")


# ============================================================================
# from_model_args backward compatibility test
# ============================================================================


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),
        (1, 2),
        (1, 8),
    ],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [32, 128])
def test_embedding_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that Embedding1D.from_model_args matches torch.nn.Embedding reference.

    Uses HF_MODEL env var or defaults to Llama-3.1-8B-Instruct.
    """
    from models.tt_transformers.tt.model_config import ModelArgs

    hf_model_name = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    dtype = ttnn.bfloat16

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=1, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("Embedding1D test only runs on non-TG devices")

    state_dict = model_args.load_state_dict()

    # Get reference embedding weight from state dict
    base_name = model_args.get_state_dict_prefix("", None) + "tok_embeddings.weight"
    ref_weight = state_dict[base_name]
    vocab_size, dim = ref_weight.shape

    # Reference model
    ref_embedding = torch.nn.Embedding(vocab_size, dim)
    with torch.no_grad():
        ref_embedding.weight.copy_(ref_weight.to(torch.bfloat16))

    # Build TT model via from_model_args
    def topology_aware_cache_path():
        return model_args.model_cache_path / f"tensor_cache_bf16_{ttnn_mesh_device.shape}"

    tt_model = Embedding1D.from_model_args(
        mesh_device=ttnn_mesh_device,
        args=model_args,
        weight_cache_path=topology_aware_cache_path(),
        state_dict=state_dict,
        dtype=dtype,
    )

    # Input tokens
    torch.manual_seed(42)
    input_ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.int64)

    # Reference output
    with torch.no_grad():
        ref_output = ref_embedding(input_ids)  # [1, seq_len, dim]

    # TT input: [1, 1, 1, seq_len] uint32
    tt_input = ttnn.from_torch(
        input_ids.reshape(1, 1, 1, seq_len).to(torch.int32),
        device=ttnn_mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.replicate_tensor_to_mesh_mapper(ttnn_mesh_device),
    )

    tt_output = tt_model.forward(tt_input)
    tt_output_torch = to_torch_auto_compose(tt_output)

    # Shape: tt is [1, 1, seq_len, dim/N] per shard, composed to [1, 1, seq_len, dim]
    # Trim padding if needed
    if tt_output_torch.shape[-1] > dim:
        tt_output_torch = tt_output_torch[..., :dim]

    # Flatten to [1, seq_len, dim] for comparison
    tt_output_torch = tt_output_torch.view(1, seq_len, dim)

    pcc_required = 0.999
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_required)
    logger.info(comp_allclose(ref_output, tt_output_torch))
    logger.info(f"Embedding1D (from_model_args) vs reference: {pcc_message}")

    assert passing, f"Embedding1D output does not meet PCC requirement {pcc_required}: {pcc_message}."
    logger.info(f"Embedding1D (from_model_args) vs reference: PASSED for seq_len={seq_len}")

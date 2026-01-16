# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the MLP1D module (1D mesh topology: N150, N300, T3K).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. MLP1D class matches HuggingFace/Meta reference model
3. MLP1D correctly rejects TG/Galaxy devices
"""

import math
import os
from functools import lru_cache
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_utils import no_init_weights

import ttnn
from models.common.auto_compose import to_torch_auto_compose
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.mlp.mlp_1d import MLP1D, MLP1DConfig, _matmul_config
from models.common.tensor_utils import TILE_SIZE
from models.common.utility_functions import comp_allclose, comp_pcc


def get_mlp_weights_from_ref_model(reference_mlp) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract w1, w2, w3 weights from a reference MLP module in TTNN layout (transposed).

    Handles both standard LLaMA-style MLPs (gate_proj, up_proj, down_proj) and
    fused gate_up_proj models (Phi-3/Phi-4).

    Returns:
        (w1, w2, w3) tensors in TTNN layout: (in_features, out_features)
    """
    if hasattr(reference_mlp, "gate_proj"):
        w1_torch = reference_mlp.gate_proj.weight.T  # (dim, hidden_dim)
        w3_torch = reference_mlp.up_proj.weight.T  # (dim, hidden_dim)
    elif hasattr(reference_mlp, "gate_up_proj"):
        # Handle models like Phi-3/Phi-4 that use fused gate_up_proj
        gate_up_weight = reference_mlp.gate_up_proj.weight
        hidden_dim = gate_up_weight.shape[0] // 2
        w1_torch = gate_up_weight[:hidden_dim, :].T  # (dim, hidden_dim)
        w3_torch = gate_up_weight[hidden_dim:, :].T  # (dim, hidden_dim)
    else:
        raise AttributeError(f"Reference MLP {type(reference_mlp)} has no gate_proj or gate_up_proj")

    w2_torch = reference_mlp.down_proj.weight.T  # (hidden_dim, dim)
    return w1_torch, w2_torch, w3_torch


def _get_prefill_len_cutoff(hf_model_name: str, mesh_shape: tuple[int, int]) -> int | None:
    """
    Get model/device-specific prefill_len_cutoff override.

    Root cause:
        The matmul program config computes per_core_M = ceil(m / (tile_size * grid_height)),
        where m = min(seq_len, prefill_len_cutoff). Larger per_core_M requires more L1 memory
        for circular buffers. Combined with in0_block_w=8 and BFP8 weights, certain model/device
        combinations overflow L1.

    Symptom:
        RuntimeError: TT_FATAL ... "Statically allocated circular buffers ... grow to ...
        beyond max L1 size"

    Fix:
        Reduce prefill_len_cutoff from 1024 to 512 for affected models. This halves m,
        reducing per_core_M (e.g., from 4 to 2), which fits in L1.

    Matches tt_transformers/tt/model_config.py:577-584 logic:
    - Llama-3.1-8B, Llama-3.2-11B, Mistral-7B, gemma-3-4b on N150 (1x1) → 512
    - Qwen2.5-7B on N300 (1x2) → 512
    - Mixtral-8x7B on T3K (1x8) → 512
    - Others → None (use default)
    """
    # Extract base model name from HF model name
    base_name = hf_model_name.split("/")[-1].rsplit("-Instruct", 1)[0]

    # Map mesh_shape to device type
    if mesh_shape == (1, 1):
        device = "N150"
    elif mesh_shape == (1, 2):
        device = "N300"
    elif mesh_shape == (1, 8):
        device = "T3K"
    else:
        device = None

    # Apply model_config.py logic
    if base_name in ["Llama-3.1-8B", "Llama-3.2-11B", "Mistral-7B", "gemma-3-4b"] and device == "N150":
        return 512
    elif base_name in ["Qwen2.5-7B"] and device == "N300":
        return 512
    elif base_name in ["Mixtral-8x7B"] and device == "T3K":
        return 512

    return None  # Use default


# ============================================================================
# Weight Caching - Avoid expensive torch.randn_like() per test
# ============================================================================

_CACHED_MLP_WEIGHTS: dict[str, dict[str, torch.Tensor]] = {}


def _get_or_init_mlp_weights(model_name: str, reference_mlp) -> None:
    """Initialize MLP weights once per model, cache and reuse across tests.

    torch.randn_like() is very slow for large models (18s for 70B).
    This caches the random weights and reuses them across tests.
    """
    if model_name not in _CACHED_MLP_WEIGHTS:
        logger.info(f"\033[33m[cache miss]\033[0m Initializing weights for {model_name}")
        _CACHED_MLP_WEIGHTS[model_name] = {}
        with torch.no_grad():
            for name, param in reference_mlp.named_parameters():
                _CACHED_MLP_WEIGHTS[model_name][name] = torch.randn_like(param)
    else:
        logger.info(f"\033[32m[cache hit]\033[0m Reusing cached weights for {model_name}")

    # Load cached weights into model
    with torch.no_grad():
        for name, param in reference_mlp.named_parameters():
            param.copy_(_CACHED_MLP_WEIGHTS[model_name][name])


# ============================================================================
# Unit Tests - No device required
# ============================================================================
# Note: These tests only test the MLP1DConfig dataclass creation.
# The _resolve_mlp1d_config function is tested via integration tests
# (test_mlp_1d_vs_reference) since it requires real LazyWeight instances.


def test_mlp_1d_config_creation():
    """Test that MLP1DConfig dataclass can be created with explicit values."""
    from unittest.mock import MagicMock

    from models.common.modules.mlp.mlp_1d import MLP1DConfig

    mock_mesh_device = MagicMock()
    mock_tt_ccl = MagicMock()
    mock_w1 = MagicMock()
    mock_w2 = MagicMock()
    mock_w3 = MagicMock()

    # Create config with explicit values
    config = MLP1DConfig(
        w1=mock_w1,
        w2=mock_w2,
        w3=mock_w3,
        mesh_device=mock_mesh_device,
        tt_ccl=mock_tt_ccl,
        dim=4096,
        hidden_dim=14336,
        max_batch_size=64,
        topology=ttnn.Topology.Ring,
    )

    # Verify explicit values are preserved
    assert config.w1 is mock_w1
    assert config.w2 is mock_w2
    assert config.w3 is mock_w3
    assert config.mesh_device is mock_mesh_device
    assert config.tt_ccl is mock_tt_ccl
    assert config.dim == 4096
    assert config.hidden_dim == 14336
    assert config.max_batch_size == 64
    assert config.topology == ttnn.Topology.Ring


def test_mlp_1d_config_defaults():
    """Test that MLP1DConfig has sensible defaults."""
    from unittest.mock import MagicMock

    from models.common.modules.mlp.mlp_1d import MLP1DConfig

    # Minimal creation - only required fields
    config = MLP1DConfig(w1=MagicMock(), w2=MagicMock(), w3=MagicMock())

    # Check defaults
    assert config.max_batch_size == 32
    assert config.mlp_activation_type == ttnn.UnaryOpType.SILU
    assert config.num_reduce_scatter_links == 1

    # Optional fields default to None
    assert config.mesh_device is None
    assert config.tt_ccl is None
    assert config.dim is None
    assert config.hidden_dim is None


def test_mlp_1d_config_power_user_overrides():
    """Test that MLP1DConfig accepts power-user overrides for program configs."""
    from unittest.mock import MagicMock

    from models.common.modules.mlp.mlp_1d import MLP1DConfig

    mock_prg_config = MagicMock()
    mock_mem_config = MagicMock()

    config = MLP1DConfig(
        w1=MagicMock(),
        w2=MagicMock(),
        w3=MagicMock(),
        decode_w1_w3_prg_config=mock_prg_config,
        decode_w2_prg_config=mock_prg_config,
        decode_mlp2_input_memcfg=mock_mem_config,
        decode_residual_memcfg=mock_mem_config,
        activation_dtype=ttnn.bfloat16,
    )

    # User-provided overrides should be preserved
    assert config.decode_w1_w3_prg_config is mock_prg_config
    assert config.decode_w2_prg_config is mock_prg_config
    assert config.decode_mlp2_input_memcfg is mock_mem_config
    assert config.decode_residual_memcfg is mock_mem_config
    assert config.activation_dtype == ttnn.bfloat16


# Pulled from deduped perf sweep of existing model tests in CI
LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"
LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_3B = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_11B = "meta-llama/Llama-3.2-11B-Vision-Instruct"
LLAMA_90B = "meta-llama/Llama-3.2-90B-Vision-Instruct"
MISTRAL_7B = "mistralai/Mistral-7B-Instruct-v0.3"
QWEN2_7B = "Qwen/Qwen2-7B-Instruct"
QWEN25_7B = "Qwen/Qwen2.5-7B-Instruct"
QWEN25_72B = "Qwen/Qwen2.5-72B-Instruct"
QWEN25_CODER_32B = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEEPSEEK_R1_14B = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
PHI_4 = "microsoft/phi-4"
QWEN3_32B = "Qwen/Qwen3-32B"

_slow = pytest.mark.slow


# [INFO] Galaxy DP run multiple copies of the following on 1x1, 1x2, and 1x8 meshes.
def _list_glx_test_cases() -> list[pytest.param]:
    # fmt: off
    return [
        # === Fast tests (minimal coverage set) ===
        # Single device
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-prefill-128-mixed-8B"),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-decode-32-uniform-8B"),
        # Multi-device (1x8)
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-128-mixed-8B"),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-1024-mixed-8B"),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-decode-32-mixed-8B"),
        # 70B (larger dims)
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_70B, 0.98, id="1x8-prefill-1024-mixed-70B"),
        # === Slow tests (full coverage from models sweep) ===
        # (1,1) mesh - from DP-32 (8B)
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-128-uniform-8B", marks=_slow),
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-prefill-256-mixed-8B", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-prefill-512-mixed-8B", marks=_slow),
        pytest.param((1, 1), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-prefill-1024-mixed-8B", marks=_slow),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-decode-32-mixed-8B", marks=_slow),
        # (1,2) mesh - from DP-16 (8B)
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-128-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-128-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-256-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-512-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-1024-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-decode-32-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-decode-32-uniform-8B", marks=_slow),
        # (1,8) mesh - from DP-4 (8B)
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-128-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-256-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-512-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-decode-32-uniform-8B", marks=_slow),
        # (1,8) mesh - from DP-4_70B (70B, mixed dtype only)
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_70B, 0.98, id="1x8-prefill-128-mixed-70B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_70B, 0.98, id="1x8-prefill-256-mixed-70B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_70B, 0.98, id="1x8-prefill-512-mixed-70B", marks=_slow),
        pytest.param((1, 8), 1, 2048, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_70B, 0.98, id="1x8-prefill-2048-mixed-70B", marks=_slow),
        pytest.param((1, 8), 1, 4096, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_70B, 0.98, id="1x8-prefill-4096-mixed-70B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_70B, 0.98, id="1x8-decode-32-mixed-70B", marks=_slow),
    ]
    # fmt: on


# [INFO] Non-Galaxy test cases from N150/N300/T3K/BH runs.
def _list_non_glx_test_cases() -> list[pytest.param]:
    # fmt: off
    return [
        # === Fast tests (minimal coverage set) ===
        # Single device (1x1) - small model
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x1-prefill-128-mixed-1B"),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-decode-32-uniform-1B"),
        # Multi-device (1x2) - vision model 11B
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-prefill-128-uniform-11B"),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-decode-32-uniform-11B"),
        # Multi-device (1x8) - vision model 90B
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_90B, 0.98, id="1x8-prefill-128-mixed-90B"),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_90B, 0.98, id="1x8-decode-32-mixed-90B"),
        # Non-Llama model families
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, PHI_4, 0.99, id="1x2-prefill-128-uniform-phi-4"),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN3_32B, 0.98, id="1x8-prefill-128-mixed-Qwen3-32B"),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_7B, 0.99, id="1x2-prefill-128-uniform-Qwen2.5-7B"),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, DEEPSEEK_R1_14B, 0.98, id="1x2-prefill-128-mixed-DeepSeek-R1-14B"),
        # === Slow tests (full coverage) ===
        # (1,1) LLAMA_1B - remaining cases not in fast set
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x1-prefill-256-mixed-1B", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x1-prefill-512-mixed-1B", marks=_slow),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x1-decode-32-mixed-1B", marks=_slow),
        pytest.param((1, 1), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x1-prefill-1024-mixed-1B", marks=_slow),
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-128-uniform-1B", marks=_slow),
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-256-uniform-1B", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x1-prefill-512-uniform-1B", marks=_slow),
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x1-prefill-128-mixed-3B", marks=_slow),
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x1-prefill-256-mixed-3B", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x1-prefill-512-mixed-3B", marks=_slow),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x1-decode-32-mixed-3B", marks=_slow),
        pytest.param((1, 1), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x1-prefill-1024-mixed-3B", marks=_slow),
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-128-uniform-3B", marks=_slow),
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-256-uniform-3B", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-prefill-512-uniform-3B", marks=_slow),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x1-decode-32-uniform-3B", marks=_slow),
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-prefill-128-mixed-8B", marks=_slow),
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-128-uniform-8B", marks=_slow),
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-prefill-256-mixed-8B", marks=_slow),
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-256-uniform-8B", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-prefill-512-mixed-8B", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-512-uniform-8B", marks=_slow),
        pytest.param((1, 1), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-prefill-1024-mixed-8B", marks=_slow),
        pytest.param((1, 1), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-prefill-1024-uniform-8B", marks=_slow),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x1-decode-32-mixed-8B", marks=_slow),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x1-decode-32-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x2-prefill-128-mixed-1B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x2-prefill-256-mixed-1B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x2-prefill-512-mixed-1B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x2-prefill-1024-mixed-1B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x2-decode-32-mixed-1B", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-128-uniform-1B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-256-uniform-1B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-512-uniform-1B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-prefill-1024-uniform-1B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x2-decode-32-uniform-1B", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x2-prefill-128-mixed-3B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x2-prefill-256-mixed-3B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x2-prefill-512-mixed-3B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x2-prefill-1024-mixed-3B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x2-decode-32-mixed-3B", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-128-uniform-3B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-256-uniform-3B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-512-uniform-3B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-prefill-1024-uniform-3B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x2-decode-32-uniform-3B", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-128-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-128-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-256-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-256-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-512-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-512-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-1024-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-1024-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 2048, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-2048-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 2048, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-2048-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 4096, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-4096-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 4096, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-4096-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 8192, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-prefill-8192-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 8192, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-prefill-8192-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x2-decode-32-mixed-8B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x2-decode-32-uniform-8B", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x2-prefill-128-mixed-11B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x2-prefill-256-mixed-11B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x2-prefill-512-mixed-11B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x2-prefill-1024-mixed-11B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x2-decode-32-mixed-11B", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-prefill-256-uniform-11B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-prefill-512-uniform-11B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x2-prefill-1024-uniform-11B", marks=_slow),
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x1-prefill-128-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x1-prefill-256-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x1-prefill-512-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x1-decode-32-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 1), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x1-prefill-1024-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 1), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-128-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 1), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-256-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 1), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-prefill-512-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 1), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x1-decode-32-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x2-prefill-128-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x2-prefill-256-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x2-prefill-512-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x2-prefill-1024-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x2-decode-32-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-128-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-256-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-512-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-prefill-1024-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x2-decode-32-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 2), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN2_7B, 0.98, id="1x2-prefill-128-mixed-Qwen2-7B-Instruct", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN2_7B, 0.98, id="1x2-prefill-256-mixed-Qwen2-7B-Instruct", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN2_7B, 0.98, id="1x2-prefill-512-mixed-Qwen2-7B-Instruct", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN2_7B, 0.98, id="1x2-prefill-1024-mixed-Qwen2-7B-Instruct", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN2_7B, 0.98, id="1x2-decode-32-mixed-Qwen2-7B-Instruct", marks=_slow),
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, PHI_4, 0.99, id="1x2-prefill-256-uniform-phi-4", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, PHI_4, 0.99, id="1x2-prefill-512-uniform-phi-4", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, PHI_4, 0.99, id="1x2-prefill-1024-uniform-phi-4", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, PHI_4, 0.99, id="1x2-decode-32-uniform-phi-4", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x8-prefill-128-mixed-1B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x8-prefill-256-mixed-1B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x8-prefill-512-mixed-1B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x8-prefill-1024-mixed-1B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_1B, 0.98, id="1x8-decode-32-mixed-1B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-128-uniform-1B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-256-uniform-1B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-512-uniform-1B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-prefill-1024-uniform-1B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_1B, 0.99, id="1x8-decode-32-uniform-1B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x8-prefill-128-mixed-3B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x8-prefill-256-mixed-3B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x8-prefill-512-mixed-3B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x8-prefill-1024-mixed-3B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_3B, 0.98, id="1x8-decode-32-mixed-3B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-128-uniform-3B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-256-uniform-3B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-512-uniform-3B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-prefill-1024-uniform-3B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_3B, 0.99, id="1x8-decode-32-uniform-3B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-128-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-128-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-256-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-256-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-512-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-512-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-1024-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-1024-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 2048, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-2048-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 2048, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-2048-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 4096, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-4096-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 4096, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-4096-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 8192, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-prefill-8192-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 8192, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-prefill-8192-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_8B, 0.98, id="1x8-decode-32-mixed-8B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_8B, 0.99, id="1x8-decode-32-uniform-8B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x8-prefill-128-mixed-11B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x8-prefill-256-mixed-11B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x8-prefill-512-mixed-11B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x8-prefill-1024-mixed-11B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, LLAMA_11B, 0.98, id="1x8-decode-32-mixed-11B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-prefill-128-uniform-11B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-prefill-256-uniform-11B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-prefill-512-uniform-11B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-prefill-1024-uniform-11B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, LLAMA_11B, 0.99, id="1x8-decode-32-uniform-11B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN3_32B, 0.98, id="1x8-prefill-256-mixed-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN3_32B, 0.98, id="1x8-prefill-512-mixed-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN3_32B, 0.98, id="1x8-prefill-1024-mixed-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN3_32B, 0.98, id="1x8-decode-32-mixed-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-128-uniform-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-256-uniform-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-512-uniform-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-prefill-1024-uniform-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN3_32B, 0.99, id="1x8-decode-32-uniform-Qwen3-32B", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x8-prefill-128-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x8-prefill-256-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x8-prefill-512-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x8-prefill-1024-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, MISTRAL_7B, 0.98, id="1x8-decode-32-mixed-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-128-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-256-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-512-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-prefill-1024-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, MISTRAL_7B, 0.99, id="1x8-decode-32-uniform-Mistral-7B-Instruct-v0.3", marks=_slow),
        # --- New test cases from mlp_1d_performance.csv ---
        # Qwen2.5-7B on N300 (1x2) - uniform BF8
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_7B, 0.99, id="1x2-prefill-256-uniform-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_7B, 0.99, id="1x2-prefill-512-uniform-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_7B, 0.99, id="1x2-prefill-1024-uniform-Qwen2.5-7B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_7B, 0.99, id="1x2-decode-32-uniform-Qwen2.5-7B", marks=_slow),
        # Qwen2.5-72B on T3K (1x8) - mixed BF4/BF8
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_72B, 0.98, id="1x8-prefill-128-mixed-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_72B, 0.98, id="1x8-prefill-256-mixed-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_72B, 0.98, id="1x8-prefill-512-mixed-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_72B, 0.98, id="1x8-prefill-1024-mixed-Qwen2.5-72B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_72B, 0.98, id="1x8-decode-32-mixed-Qwen2.5-72B", marks=_slow),
        # DeepSeek-R1-Distill-Qwen-14B on N300 (1x2) - mixed BF4/BF8
        pytest.param((1, 2), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, DEEPSEEK_R1_14B, 0.98, id="1x2-prefill-256-mixed-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, DEEPSEEK_R1_14B, 0.98, id="1x2-prefill-512-mixed-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, DEEPSEEK_R1_14B, 0.98, id="1x2-prefill-1024-mixed-DeepSeek-R1-14B", marks=_slow),
        pytest.param((1, 2), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, DEEPSEEK_R1_14B, 0.98, id="1x2-decode-32-mixed-DeepSeek-R1-14B", marks=_slow),
        # Qwen2.5-Coder-32B on T3K (1x8) - mixed BF4/BF8
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_CODER_32B, 0.98, id="1x8-prefill-128-mixed-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_CODER_32B, 0.98, id="1x8-prefill-256-mixed-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_CODER_32B, 0.98, id="1x8-prefill-512-mixed-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_CODER_32B, 0.98, id="1x8-prefill-1024-mixed-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat4_b, ttnn.bfloat8_b, ttnn.bfloat4_b, QWEN25_CODER_32B, 0.98, id="1x8-decode-32-mixed-Qwen2.5-Coder-32B", marks=_slow),
        # Qwen2.5-Coder-32B on T3K (1x8) - uniform BF8
        pytest.param((1, 8), 1, 128, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-128-uniform-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1, 256, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-256-uniform-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1, 512, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-512-uniform-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1, 1024, "prefill", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-prefill-1024-uniform-Qwen2.5-Coder-32B", marks=_slow),
        pytest.param((1, 8), 1, 32, "decode", ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.bfloat8_b, QWEN25_CODER_32B, 0.99, id="1x8-decode-32-uniform-Qwen2.5-Coder-32B", marks=_slow),
    ]


# [INFO] generate random tensor for every test case is too expensive; cache weights and reuse them across test cases
# [INFO] separate out ttnn_mesh_device parameter allows for sharing the same mesh device across test cases
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_shape,batch_size,seq_len,mode,act_dtype,w1_dtype,w2_dtype,w3_dtype,hf_model_name,pcc",
    _list_non_glx_test_cases() + _list_glx_test_cases(),
)
def test_mlp_1d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mesh_shape,
    batch_size,
    seq_len,
    mode,
    act_dtype,
    w1_dtype,
    w2_dtype,
    w3_dtype,
    hf_model_name,
    pcc,
):
    """
    Test MLP1D constructed via direct APIs (MLP1DConfig) matches HF reference MLP.

    Configs pulled from perf sweep CSVs (b{batch_size}-DP-{dp}_{model}).
    """

    # get reference model; generate and load deterministic, random weights into the reference model
    seed = 1234
    torch.manual_seed(seed)

    # HF model (default small) for reference; skip global init to only seed MLP.
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_hidden_layers = 1

    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    first_layer = hf_model.model.layers[0]
    reference_mlp = first_layer.mlp

    # Initialize only the MLP submodule deterministically (cached for speed).
    _get_or_init_mlp_weights(hf_model_name, reference_mlp)

    # Build MLP1D TT model and load the same weights as the reference model
    w1_torch, w2_torch, w3_torch = get_mlp_weights_from_ref_model(reference_mlp)
    dim = w1_torch.shape[0]
    torch_input = torch.randn(batch_size, 1, seq_len, dim, dtype=torch.bfloat16)

    # Create LazyWeights
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/mlp_1d"))
    lazy_w1 = LazyWeight(source=w1_torch, dtype=w1_dtype, cache_dir_weight_name=(cache_dir, "w1"))
    lazy_w2 = LazyWeight(source=w2_torch, dtype=w2_dtype, cache_dir_weight_name=(cache_dir, "w2"))
    lazy_w3 = LazyWeight(source=w3_torch, dtype=w3_dtype, cache_dir_weight_name=(cache_dir, "w3"))

    # Get model/device-specific prefill_len_cutoff
    prefill_len_cutoff = _get_prefill_len_cutoff(hf_model_name, mesh_shape)

    # Construct the MLP1D model
    if prefill_len_cutoff is None:
        # Use default prefill_len_cutoff, take the happy path of MLP1D
        tt_model = MLP1D(w1=lazy_w1, w2=lazy_w2, w3=lazy_w3)
    else:
        # Use custom config with prefill_len_cutoff override
        mlp_config = MLP1DConfig(w1=lazy_w1, w2=lazy_w2, w3=lazy_w3, prefill_len_cutoff=prefill_len_cutoff)
        tt_model = MLP1D.from_config(mlp_config)

    # Run TT model with the same input -- torch_input -- converted to ttnn tensor lazily on the fly
    # [INFO] we use LazyWeight on input for the benefit of faster testing (cached input); in production, the input is already a ttnn tensor.
    tt_input = LazyWeight(
        source=torch_input,
        dtype=act_dtype,
        # cache_dir_weight_name=(cache_dir, "input"), # todo)) needs better fingerprinting for input tensor to enable
    )
    tt_output = tt_model.forward(tt_input, mode)
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Now both models are ready to go
    # Run reference model
    with torch.no_grad():
        reference_output = reference_mlp(torch_input)

    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"MLP1D (direct API) vs HF reference: {pcc_message}")

    assert passing, f"MLP1D output does not meet PCC requirement {pcc}: {pcc_message}."
    logger.info(f"MLP1D (direct API) vs HF reference: PASSED for mode={mode}, seq_len={seq_len}")


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 8)],
    ids=["1x8"],
    indirect=True,
)
def test_mlp_1d_config_prefill_override(ttnn_mesh_device: ttnn.MeshDevice):
    """
    Show how to override prefill_w2_prg_config with the MLP1DConfig API.

    Use MLP1D.from_config() for any customization beyond the simple 3-weight API.
    """
    from models.common.modules.mlp.mlp_1d import _find_prefill_grid

    # Use Llama 8B config
    hf_model_name = "meta-llama/Llama-3.1-8B-Instruct"
    hf_config = AutoConfig.from_pretrained(hf_model_name)
    seq_len = 128
    batch_size = 1

    # Load HF model for reference weights
    hf_config.num_hidden_layers = 1
    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(hf_config, torch_dtype=torch.bfloat16)
    reference_mlp = hf_model.model.layers[0].mlp

    # Generate random weights directly for this test
    # todo)) using _get_or_init_mlp_weights instead here would interfere with the test_mlp_1d_vs_reference test; this problem could be solved by provenance-based fingerprinting the torch.tensor inputs
    with torch.no_grad():
        for param in reference_mlp.parameters():
            param.copy_(torch.randn_like(param))

    # Prepare weights
    w1_torch, w2_torch, w3_torch = get_mlp_weights_from_ref_model(reference_mlp)

    # Create LazyWeights (no disk cache)
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    lazy_w1 = LazyWeight(source=w1_torch, dtype=ttnn.bfloat4_b)
    lazy_w2 = LazyWeight(source=w2_torch, dtype=ttnn.bfloat8_b)
    lazy_w3 = LazyWeight(source=w3_torch, dtype=ttnn.bfloat4_b)

    # Step 1: Create MLP1D with default config
    tt_model = MLP1D.from_config(MLP1DConfig(w1=lazy_w1, w2=lazy_w2, w3=lazy_w3))

    # Step 2: Define custom prefill w2 config using resolved values from tt_model.config
    cfg = tt_model.config
    dim = cfg.dim
    hidden_dim = cfg.hidden_dim
    tile_size = TILE_SIZE
    prefill_len_cutoff = cfg.prefill_len_cutoff

    @lru_cache
    def custom_prefill_w2_prg_config(seq_len: int):
        n_w2 = dim
        dram_shard_grid_width = 8
        prefill_rows = 8
        grid_size = _find_prefill_grid(prefill_rows, hidden_dim // tile_size)
        return _matmul_config(
            m=min(seq_len, prefill_len_cutoff),
            k=hidden_dim,
            n=n_w2,
            grid_size=grid_size,
            per_core_n=math.ceil(n_w2 / (tile_size * dram_shard_grid_width)),
        )

    # Step 3: Override the prefill config on the existing model
    tt_model.config.prefill_w2_prg_config = custom_prefill_w2_prg_config

    # Verify the override was applied
    assert tt_model.config.prefill_w2_prg_config is custom_prefill_w2_prg_config

    # Run prefill forward
    torch_input = torch.randn(batch_size, 1, seq_len, dim, dtype=torch.bfloat16)
    tt_input = LazyWeight(source=torch_input, dtype=ttnn.bfloat8_b)
    tt_output = tt_model.forward(tt_input, mode="prefill")
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Verify output shape matches input shape (MLP is dim -> dim)
    assert tt_output_torch.shape == torch_input.shape, f"Expected {torch_input.shape}, got {tt_output_torch.shape}"

    # Verify numerical correctness against reference
    with torch.no_grad():
        reference_output = reference_mlp(torch_input)
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, 0.98)
    assert passing, f"MLP1D with custom prefill config failed PCC: {pcc_message}"
    logger.info(f"test_mlp_1d_config_prefill_override: PASSED - {pcc_message}")


# ============================================================================
# Integration Tests - Require device
# ============================================================================


# [INFO] this test will retire once models/tt_transformers/tt/model_config.py retires
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [
        (1, 1),  # single device
        (1, 2),  # 1D mesh, 2 devices
        (1, 4),  # 1D mesh, 4 devices
        (1, 8),  # 1D mesh, 8 devices
    ],
    ids=["1x1", "1x2", "1x4", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", (512, 32))
def test_mlp_1d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len):
    """
    Test that MLP1D class matches the HuggingFace/Meta reference model.
    """
    from models.common.modules.mlp.mlp_1d import MLP1D
    from models.tt_transformers.tests.test_utils import get_ref_model_dype
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    dtype = ttnn.bfloat8_b
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    model_args = ModelArgs(ttnn_mesh_device, max_batch_size=batch_size, max_seq_len=128, cache_hf=True)
    model_args.n_layers = 1

    if model_args.is_galaxy:
        pytest.skip("MLP1D test only runs on non-TG devices")

    state_dict = model_args.load_state_dict()
    model_config = model_args.get_model_config()

    # Load reference model
    first_layer_prefix = model_args.get_state_dict_prefix("MLP", 0)
    partial_state_dict = {
        k[len(first_layer_prefix) + 1 :]: v for k, v in state_dict.items() if k.startswith(first_layer_prefix)
    }

    reference_model = model_args.reference_mlp()
    reference_model.load_state_dict(partial_state_dict)

    # Create MLP1D
    def topology_aware_cache_path(dtype):
        if model_args.instruct:
            return (
                model_args.model_cache_path
                / {
                    ttnn.bfloat16: f"tensor_cache_instruct_bf16_{ttnn_mesh_device.shape}",
                    ttnn.bfloat8_b: f"tensor_cache_instruct_bfp8_{ttnn_mesh_device.shape}",
                }[dtype]
            )
        else:
            return (
                model_args.model_cache_path
                / {
                    ttnn.bfloat16: f"tensor_cache_bf16_{ttnn_mesh_device.shape}",
                    ttnn.bfloat8_b: f"tensor_cache_bfp8_{ttnn_mesh_device.shape}",
                }[dtype]
            )

    tt_ccl = TT_CCL(ttnn_mesh_device)
    tt_model = MLP1D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=topology_aware_cache_path(dtype),
        layer_num=0,
    )

    # Create input
    torch_input = torch.randn(
        1, 1, seq_len, model_args.dim, dtype=get_ref_model_dype(reference_model, model_args.model_name)
    )

    # Run reference
    reference_output = reference_model(torch_input)

    # Run TT model
    input_mem_config = model_config["SHARDED_MLP_INPUT_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG

    tt_input = ttnn.from_torch(
        torch_input,
        device=ttnn_mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(ttnn_mesh_device, dims=(None, None), mesh_shape=model_args.cluster_shape),
        dtype=ttnn.bfloat8_b,
        memory_config=input_mem_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_output = tt_model.forward(tt_input, mode)

    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(ttnn_mesh_device, dims=(1, 3), mesh_shape=model_args.cluster_shape),
    )
    tt_output_torch = tt_output_torch[:, :1, :, :]

    # Compare
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"MLP1D vs reference: {pcc_message}")

    assert passing, f"MLP1D output does not meet PCC requirement {pcc_required}: {pcc_message}."
    logger.info(f"MLP1D vs reference: PASSED for mode={mode}, seq_len={seq_len}")

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
    SHARD_HEIGHT,
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


def get_norm_weight_from_ref_model(reference_norm, dim: int) -> torch.Tensor:
    """
    Extract weight from a reference RMSNorm module in TTNN layout.

    Returns:
        weight tensor in shape (1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT)
    """
    weight = reference_norm.weight.detach().clone()
    return weight.unsqueeze(0).view(1, 1, dim).reshape([1, 1, dim // SHARD_HEIGHT, SHARD_HEIGHT])


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
        dim=4096,
        max_batch_size=64,
    )

    assert config.weight == mock_weight
    assert config.eps == 1e-6
    assert config.add_unit_offset is True
    assert config.mesh_device == mock_mesh_device
    assert config.tt_ccl == mock_tt_ccl
    assert config.dim == 4096
    assert config.max_batch_size == 64


def test_rmsnorm_1d_config_defaults():
    """Test that RMSNorm1DConfig has sensible defaults."""
    config = RMSNorm1DConfig(weight=MagicMock())

    # Check defaults
    assert config.eps == 1e-5
    assert config.add_unit_offset is False
    assert config.max_batch_size == 32
    assert config.weight_dtype == ttnn.bfloat16
    assert config.fp32_dest_acc_en is True

    # Optional fields default to None
    assert config.mesh_device is None
    assert config.tt_ccl is None
    assert config.dim is None
    assert config.decode_sharded_program_config is None


def test_rmsnorm_1d_config_power_user_overrides():
    """Test that RMSNorm1DConfig accepts power-user overrides for program configs."""
    mock_prg_config = MagicMock()
    mock_mem_config = MagicMock()

    config = RMSNorm1DConfig(
        weight=MagicMock(),
        decode_sharded_program_config=mock_prg_config,
        decode_sharded_output_memcfg=mock_mem_config,
        fp32_dest_acc_en=False,
    )

    assert config.decode_sharded_program_config == mock_prg_config
    assert config.decode_sharded_output_memcfg == mock_mem_config
    assert config.fp32_dest_acc_en is False


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


# Get HF model name from environment variable or use default
HF_MODEL_NAME = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(1, 1), (1, 2), (1, 8)],
    ids=["1x1", "1x2", "1x8"],
    indirect=True,
)
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize(
    "batch_size,seq_len",
    [
        (1, 1),  # decode
        (32, 1),  # decode with max batch
        (1, 128),  # short prefill
        (1, 512),  # medium prefill
    ],
    ids=["b1_s1", "b32_s1", "b1_s128", "b1_s512"],
)
def test_rmsnorm_1d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mode: str,
    batch_size: int,
    seq_len: int,
):
    """
    Test RMSNorm1D.forward(x, mode) matches PyTorch reference.
    """
    # Decode mode is for single-token generation (seq_len=1)
    # Prefill mode handles variable-length sequences
    if mode == "decode" and seq_len > 1:
        pytest.skip("Decode mode is only valid for seq_len=1")

    seed = 1234
    torch.manual_seed(seed)

    # Load HF model for reference
    hf_model_name = HF_MODEL_NAME
    config = AutoConfig.from_pretrained(hf_model_name)
    config.num_hidden_layers = 1

    with no_init_weights():
        hf_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)

    # Get reference RMSNorm (input_layernorm from first layer)
    first_layer = hf_model.model.layers[0]
    reference_norm = first_layer.input_layernorm

    # Initialize deterministic weights
    _get_or_init_norm_weights(hf_model_name, reference_norm)

    # Get dimensions
    dim = config.hidden_size
    eps = config.rms_norm_eps

    # Build RMSNorm1D
    weight_torch = get_norm_weight_from_ref_model(reference_norm, dim)

    # Tensor shapes match real model usage:
    # - Decode: (1, 1, batch_size, dim) - each user has 1 token, height = batch_size
    # - Prefill: (1, 1, seq_len, dim) - single user with full sequence
    if mode == "decode":
        torch_input = torch.randn(1, 1, batch_size, dim, dtype=torch.bfloat16)
    else:
        torch_input = torch.randn(1, 1, seq_len, dim, dtype=torch.bfloat16)

    # Create LazyWeights
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/rmsnorm_1d"))

    lazy_weight = LazyWeight(
        source=weight_torch,
        dtype=ttnn.bfloat16,
        cache_dir_weight_name=(cache_dir, "norm_weight"),
    )

    # Construct RMSNorm1D
    tt_model = RMSNorm1D(weight=lazy_weight, eps=eps)

    # Run TT model
    tt_input = LazyWeight(source=torch_input, dtype=ttnn.bfloat16)
    tt_output = tt_model.forward(tt_input, mode=mode)
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Run reference model
    # PyTorch RMSNorm expects (*, dim), we pass (height, dim)
    torch_input_squeezed = torch_input.squeeze(0).squeeze(0)  # (height, dim)
    with torch.no_grad():
        reference_output = reference_norm(torch_input_squeezed)
    reference_output = reference_output.unsqueeze(0).unsqueeze(0)  # (1, 1, height, dim)

    # Compare
    pcc = 0.999
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"RMSNorm1D vs HF reference: {pcc_message}")

    assert passing, f"RMSNorm1D output does not meet PCC requirement {pcc}: {pcc_message}."
    logger.info(f"RMSNorm1D vs HF reference: PASSED for mode={mode}, seq_len={seq_len}")


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

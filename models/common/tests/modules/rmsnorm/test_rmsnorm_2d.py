# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for the RMSNorm2D module (2D mesh topology: TG/Galaxy 4x8 or 8x4).

This test suite verifies:
1. Unit tests for config dataclasses (no device needed)
2. RMSNorm2D class matches PyTorch/HuggingFace reference model
3. RMSNorm2D correctly rejects non-TG devices
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
from models.common.modules.rmsnorm.rmsnorm_2d import RMSNorm2D, RMSNorm2DConfig
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


# ============================================================================
# Unit Tests - No device required
# ============================================================================


def test_rmsnorm_2d_config_creation():
    """Test that RMSNorm2DConfig dataclass can be created with explicit values."""
    mock_mesh_device = MagicMock()
    mock_tt_ccl = MagicMock()
    mock_weight = MagicMock()

    config = RMSNorm2DConfig(
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


def test_rmsnorm_2d_config_defaults():
    """Test that RMSNorm2DConfig has sensible defaults."""
    config = RMSNorm2DConfig(weight=MagicMock())

    # Check defaults
    assert config.eps == 1e-5
    assert config.add_unit_offset is False
    assert config.max_batch_size == 32

    # Optional fields default to None
    assert config.mesh_device is None
    assert config.tt_ccl is None
    assert config.decode_input_memcfg is None
    assert config.decode_progcfg is None


def test_rmsnorm_2d_config_power_user_overrides():
    """Test that RMSNorm2DConfig accepts power-user overrides for program configs."""
    mock_prg_config = MagicMock()
    mock_mem_config = MagicMock()
    mock_kernel_config = MagicMock()

    config = RMSNorm2DConfig(
        weight=MagicMock(),
        decode_input_memcfg=mock_mem_config,
        decode_progcfg=mock_prg_config,
        compute_kernel_config_prefill=mock_kernel_config,
    )

    assert config.decode_input_memcfg == mock_mem_config
    assert config.decode_progcfg == mock_prg_config
    assert config.compute_kernel_config_prefill == mock_kernel_config


def test_rmsnorm_2d_rejects_non_tg():
    """Test that RMSNorm2D.from_model_args raises error for non-TG mesh shapes."""
    mock_args = MagicMock()
    mock_mesh_device = MagicMock()
    mock_mesh_device.shape = [1, 8]  # Not TG

    with pytest.raises(ValueError, match="requires Galaxy topology"):
        RMSNorm2D.from_model_args(
            mesh_device=mock_mesh_device,
            tt_ccl=MagicMock(),
            args=mock_args,
            state_dict={},
            weight_cache_path=None,
            layer_num=0,
            weight_key="input_layernorm",
        )


# ============================================================================
# Integration Tests - Device required (TG only)
# ============================================================================


# todo)) tttv1 code for TG has bit-rotten! --> so we start with a simple test case for now
@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(4, 8), (8, 4)],
    ids=["4x8", "8x4"],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,batch_size,seq_len",
    [
        ("decode", 1, 1),
        ("decode", 32, 1),
        ("prefill", 1, 128),
        ("prefill", 1, 512),
        ("prefill", 2, 128),  # Regression test: prefill with batch > 1
        ("prefill", 4, 64),  # Regression test: prefill with batch > 1
    ],
    ids=["decode-b1_s1", "decode-b32_s1", "prefill-b1_s128", "prefill-b1_s512", "prefill-b2_s128", "prefill-b4_s64"],
)
def test_rmsnorm_2d_vs_reference(
    ttnn_mesh_device: ttnn.MeshDevice,
    mode: str,
    batch_size: int,
    seq_len: int,
):
    """
    Test RMSNorm2D.forward(x, mode) matches PyTorch reference on TG.
    """
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

    # Build RMSNorm2D - pass raw weight, module handles reshaping
    weight_torch = reference_norm.weight.detach().clone()

    # TTNN expects different input shapes for decode vs prefill:
    # - decode: [1, 1, batch, dim] - batch in dim 2 (sharded across batch)
    # - prefill: [batch, 1, seq, dim] - standard layout
    if mode == "decode":
        torch_input = torch.randn(1, 1, batch_size * seq_len, dim, dtype=torch.bfloat16)
    else:
        torch_input = torch.randn(batch_size, 1, seq_len, dim, dtype=torch.bfloat16)

    # Create LazyWeights
    ttnn.SetDefaultDevice(ttnn_mesh_device)
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/rmsnorm_2d"))

    lazy_weight = LazyWeight(
        source=weight_torch,
        dtype=ttnn.bfloat16,
        cache_dir_weight_name=(cache_dir, "norm_weight_2d"),
    )

    # Construct RMSNorm2D (use from_config to set eps from HF model)
    tt_config = RMSNorm2DConfig(weight=lazy_weight, eps=eps)
    tt_model = RMSNorm2D.from_config(tt_config)

    # Run TT model
    tt_input = LazyWeight(source=torch_input, dtype=ttnn.bfloat16)
    tt_output = tt_model.forward(tt_input, mode=mode)
    tt_output_torch = to_torch_auto_compose(tt_output)
    ttnn.SetDefaultDevice(None)

    # Run reference model
    # PyTorch RMSNorm expects (..., dim) for the last dimension
    if mode == "decode":
        # decode input is [1, 1, batch, dim] -> squeeze to [batch, dim] for PyTorch
        torch_input_for_ref = torch_input.squeeze(0).squeeze(0)  # (batch, dim)
        with torch.no_grad():
            reference_output = reference_norm(torch_input_for_ref)
        reference_output = reference_output.unsqueeze(0).unsqueeze(0)  # back to [1, 1, batch, dim]
    else:
        # prefill input is [batch, 1, seq, dim] -> squeeze to [batch, seq, dim] for PyTorch
        torch_input_for_ref = torch_input.squeeze(1)  # (batch, seq, dim)
        with torch.no_grad():
            reference_output = reference_norm(torch_input_for_ref)
        reference_output = reference_output.unsqueeze(1)  # back to [batch, 1, seq, dim]

    # Compare - TG distributed norm may have slightly lower PCC due to distributed computation
    pcc = 0.998
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(comp_allclose(reference_output, tt_output_torch))
    logger.info(f"RMSNorm2D vs HF reference: {pcc_message}")

    assert passing, f"RMSNorm2D output does not meet PCC requirement {pcc}: {pcc_message}."
    logger.info(f"RMSNorm2D vs HF reference: PASSED for mode={mode}, seq_len={seq_len}")


# Get HF model name from environment variable or use default
HF_MODEL_NAME = os.environ.get("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


@pytest.mark.parametrize(
    "ttnn_mesh_device",
    [(4, 8)],
    ids=["4x8"],
    indirect=True,
)
@pytest.mark.parametrize("seq_len", [1, 128])
def test_rmsnorm_2d_vs_reference_from_model_args(ttnn_mesh_device: ttnn.MeshDevice, seq_len: int):
    """
    Test RMSNorm2D.from_model_args() factory method.
    """
    from models.tt_transformers.tt.ccl import TT_CCL
    from models.tt_transformers.tt.model_config import ModelArgs

    seed = 1234
    torch.manual_seed(seed)
    batch_size = 1
    mode = "decode" if seq_len <= 32 else "prefill"

    # Create ModelArgs (HF_MODEL set at module level via setdefault)
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

    # Build RMSNorm2D via from_model_args
    cache_dir = Path(os.getenv("TT_CACHE_PATH", "model_cache/rmsnorm_2d_from_args"))
    tt_model = RMSNorm2D.from_model_args(
        mesh_device=ttnn_mesh_device,
        tt_ccl=tt_ccl,
        args=model_args,
        state_dict=state_dict,
        weight_cache_path=cache_dir,
        layer_num=0,
        weight_key="input_layernorm",
        state_dict_prefix="model.layers.0.",
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
    pcc = 0.998
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc)

    logger.info(f"RMSNorm2D (from_model_args) vs HF reference: {pcc_message}")
    assert passing, f"RMSNorm2D output does not meet PCC requirement {pcc}: {pcc_message}."

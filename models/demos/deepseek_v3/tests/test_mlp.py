# models/demos/deepseek_v3/tests/test_mlp.py
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.reference.mlp import DeepseekV3MLP
from models.demos.deepseek_v3.tt.mlp import MLP
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.utility_functions import comp_pcc


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config():
    """Load DeepSeek config for testing"""
    config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
    config.num_hidden_layers = 1  # Reduce layers for testing
    return config


@pytest.fixture
def reference_model(config):
    """Get the actual DeepSeek MLP model using local implementation."""
    return DeepseekV3MLP(config)


@pytest.fixture
def mlp_module():
    """Create MLP module instance."""
    return MLP()


# Unit Tests
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_convert_weights(mlp_module, reference_model, config, temp_dir, mesh_device):
    """Test that weights are correctly converted to TTNN format."""
    output_path = temp_dir / "weights"

    # Convert weights - now handles HF names directly
    mlp_module.convert_weights(reference_model.state_dict(), output_path, mesh_device)

    # Verify files exist
    assert (output_path / "w1.weight").exists()
    assert (output_path / "w2.weight").exists()
    assert (output_path / "w3.weight").exists()

    # Load and verify a weight
    w1_ttnn = ttnn.load_tensor(str(output_path / "w1.weight"))
    w1_torch = ttnn.to_torch(w1_ttnn)

    # Weight should be transposed from PyTorch format
    expected_shape = (config.hidden_size, config.intermediate_size)
    assert w1_torch.shape[-2:] == expected_shape

    # Verify the values match (accounting for transpose and bfloat8 conversion)
    w1_ref_transposed = reference_model.state_dict()["gate_proj.weight"].T
    passing, pcc_msg = comp_pcc(w1_ref_transposed, w1_torch, 0.99)
    assert passing, f"Weight conversion PCC failed: {pcc_msg}"

    # Cleanup
    ttnn.deallocate(w1_ttnn)


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_prefill_config_generation(mlp_module, config, mesh_device):
    """Test prefill config generation."""
    hf_config = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "hidden_act": config.hidden_act,
    }

    model_config = mlp_module.prefill_model_config(hf_config, mesh_device)

    # Check required keys exist
    assert "w1.memory_config" in model_config
    assert "w1.program_config" in model_config
    assert "w1.compute_kernel_config" in model_config
    assert "max_rows" in model_config
    assert "all_reduce.topology" in model_config

    # Verify program config is a list for prefill (chunk-dependent)
    assert isinstance(model_config["w1.program_config"], list)
    assert len(model_config["w1.program_config"]) == 8  # 8 different chunk sizes


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_decode_config_generation(mlp_module, config, mesh_device):
    """Test decode config generation."""
    hf_config = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "hidden_act": config.hidden_act,
    }

    model_config = mlp_module.decode_model_config(hf_config, mesh_device)

    # Check required keys exist
    assert "w1.memory_config" in model_config
    assert "w1.program_config" in model_config
    assert "w1.compute_kernel_config" in model_config

    # Decode should use L1 sharding
    assert model_config["w1.memory_config"] == ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG

    # Program config should not be a list for decode
    assert not isinstance(model_config["w1.program_config"], list)


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
def test_run_config_creation(mlp_module, reference_model, config, temp_dir, mesh_device):
    """Test creating runtime config from ModelConfig and weights."""
    # Get state dict from actual model - pass directly to convert_weights
    hf_state_dict = reference_model.state_dict()

    # First convert weights
    weights_path = temp_dir / "weights"
    mlp_module.convert_weights(hf_state_dict, weights_path, mesh_device)

    # Generate model config
    hf_config = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "hidden_act": config.hidden_act,
    }
    model_config = mlp_module.decode_model_config(hf_config, mesh_device)
    model_config["mode"] = "decode"

    # Create RunConfig directly from dict (no JSON serialization)
    run_config = create_run_config(model_config, weights_path, mesh_device)

    # Verify we can access operation configs
    assert hasattr(run_config, "w1")
    assert hasattr(run_config, "w2")
    assert hasattr(run_config, "w3")
    assert hasattr(run_config, "all_reduce")

    # Verify mode is accessible
    assert run_config.mode == "decode"


# Integration Tests


@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids())
        )
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 32),
        ("prefill", 512),
        ("prefill", 2048),  # Test chunking
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    mlp_module,
    reference_model,
    config,
    temp_dir,
    mesh_device,
):
    """Test forward pass against reference model."""
    batch_size = 1

    # Get state dict from actual model - pass directly to convert_weights
    hf_state_dict = reference_model.state_dict()

    # Setup: Convert weights
    weights_path = temp_dir / "weights"
    mlp_module.convert_weights(hf_state_dict, weights_path, mesh_device)

    # Generate appropriate config
    hf_config = {
        "hidden_size": config.hidden_size,
        "intermediate_size": config.intermediate_size,
        "hidden_act": config.hidden_act,
    }

    if mode == "prefill":
        model_config = mlp_module.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = mlp_module.decode_model_config(hf_config, mesh_device)
    model_config["mode"] = mode

    # Create RunConfig directly from dict instead of saving/loading
    run_config = create_run_config(model_config, weights_path, mesh_device)

    # Create input tensor
    torch_input = torch.randn(batch_size, 1, seq_len, config.hidden_size)

    # Reference forward pass
    reference_output = reference_model(torch_input)

    # Convert input to TTNN
    if mode == "decode":
        # For decode, use DRAM for simplicity
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        # For prefill, use DRAM
        tt_input = ttnn.from_torch(
            torch_input,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    # TTNN forward pass
    tt_output = mlp_module.forward(tt_input, run_config, mesh_device)

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare outputs
    pcc_required = 0.98  # Slightly lower due to bfloat conversions
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"MLP output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)
    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    pytest.main([__file__])

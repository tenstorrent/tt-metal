# models/demos/deepseek_v3/tests/test_mlp.py
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
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
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP
from models.demos.deepseek_v3.tt.mlp_1d import MLP1D
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.utility_functions import comp_pcc


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def hf_config():
    """Load DeepSeek config for testing"""
    path = os.getenv("HF_MODEL", "/proj_sw/user_dev/deepseek-ai")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config.num_hidden_layers = 1  # Reduce layers for testing
    return config


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""
    return DeepseekV3MLP(hf_config)


# Unit Tests
@pytest.mark.parametrize(
    "mesh_device",
    [
        {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (8, 4)}.get(
            os.environ.get("MESH_DEVICE"), (1, ttnn.get_num_devices())
        )
    ],
    indirect=True,
)
def test_convert_weights(reference_model, hf_config, temp_dir, mesh_device):
    """Test that weights are correctly converted to TTNN format."""
    # Convert weights - now returns weight_config
    weight_config = MLP1D.convert_weights(hf_config, reference_model.state_dict(), temp_dir, mesh_device)

    # Verify weight_config structure
    assert "w1" in weight_config
    assert "w2" in weight_config
    assert "w3" in weight_config
    assert "input_tensor_b" in weight_config["w1"]
    assert "input_tensor_b" in weight_config["w2"]
    assert "input_tensor_b" in weight_config["w3"]

    # Verify files exist
    assert Path(weight_config["w1"]["input_tensor_b"]).exists()
    assert Path(weight_config["w2"]["input_tensor_b"]).exists()
    assert Path(weight_config["w3"]["input_tensor_b"]).exists()

    # Load and verify a weight
    w1_ttnn = ttnn.load_tensor(weight_config["w1"]["input_tensor_b"])
    w1_torch = ttnn.to_torch(w1_ttnn)

    # Weight should be transposed from PyTorch format
    expected_shape = (hf_config.hidden_size, hf_config.intermediate_size // mesh_device.get_num_devices())
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
            os.environ.get("MESH_DEVICE"), (1, ttnn.get_num_devices())
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
    reference_model,
    hf_config,
    temp_dir,
    mesh_device,
):
    """Test forward pass against reference model."""
    batch_size = 1

    # Get state dict from actual model - pass directly to convert_weights
    hf_state_dict = reference_model.state_dict()

    # Setup: Convert weights and get weight_config
    weight_config = MLP1D.convert_weights(hf_config, hf_state_dict, temp_dir, mesh_device)

    # Generate appropriate config
    if mode == "prefill":
        model_config = MLP1D.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = MLP1D.decode_model_config(hf_config, mesh_device)

    # Create a new model state
    model_state = MLP1D.create_state(hf_config, mesh_device=mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Create input tensor
    torch_input = torch.randn(batch_size, 1, seq_len, hf_config.hidden_size)

    # Reference forward pass
    reference_output = reference_model(torch_input)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass
    if mode == "prefill":
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = MLP1D.forward_prefill(tt_input, run_config)
        expected_output_memory_config = run_config["output_memory_config"]
    else:
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = MLP1D.forward_decode(tt_input, run_config)
        expected_output_memory_config = run_config["output_memory_config"]

    # Verify output memory config matches expected
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

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


if __name__ == "__main__":
    pytest.main([__file__])

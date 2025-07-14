# models/demos/deepseek_v3/tests/test_expert.py
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
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP as ReferenceExpert

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.tt.expert import Expert as TTExpert
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


mesh_device_shape = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8), "TG": (1, 8)}.get(
    os.environ.get("MESH_DEVICE"), (1, min(ttnn.get_num_devices(), 8))
)


# Unit Tests
@pytest.mark.parametrize(
    "mesh_device",
    [mesh_device_shape],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, seq_len",
    [
        ("decode", 32),
        ("prefill", 512),
        ("prefill", 2048),  # Test chunking
    ],
)
def test_forward_pass(mode, seq_len, mesh_device, temp_dir, hf_config):
    """Test forward pass against reference model."""

    batch_size = 1

    logger.info("Loading reference expert model..")
    reference_model = ReferenceExpert(
        hf_config, hidden_size=hf_config.hidden_size, intermediate_size=hf_config.moe_intermediate_size
    )
    breakpoint()  # Debugging point to inspect the reference model
    reference_model.eval()

    hf_state_dict = reference_model.state_dict()
    logger.info("Loading reference model state dict..done")
    # Setup: Convert weights and get weight_config
    weight_config = TTExpert.convert_weights(hf_config, hf_state_dict, temp_dir, mesh_device)

    # Generate appropriate config
    if mode == "prefill":
        model_config = TTExpert.prefill_model_config(hf_config, mesh_device)
    else:
        model_config = TTExpert.decode_model_config(hf_config, mesh_device)

    # Create a new model state
    model_state = TTExpert.create_state(hf_config, mesh_device=mesh_device)

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
        tt_output = TTExpert.forward_prefill(tt_input, run_config)
        expected_output_memory_config = run_config["output_memory_config"]
    elif mode == "decode":
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = TTExpert.forward_decode(tt_input, run_config)
        expected_output_memory_config = run_config["output_memory_config"]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Verify output memory config matches expected
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Compare outputs
    pcc_required = 0.98
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"The Expert output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)


if __name__ == "__main__":
    pytest.main([__file__])

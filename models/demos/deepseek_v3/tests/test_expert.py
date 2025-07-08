# models/demos/deepseek_v3/tests/test_expert.py
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP as ReferenceExpert

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.tt.moe import Expert as TTExpert
from models.utility_functions import comp_pcc


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.parametrize(
    "mesh_device",
    [
        (1, 1),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size",
    [
        ("decode", 1024, 64),
        ("prefill", 1, 1024 * 64),
    ],
)
def test_forward_pass(mode, seq_len, mesh_device, temp_dir, batch_size):
    """Test forward pass against reference model."""

    print(f"mesh_device: {mesh_device}")
    hf_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
    hf_config.num_hidden_layers = 1  # Reduce layers for testing

    logger.info("Loading reference expert model..")
    reference_model = ReferenceExpert(hf_config)
    reference_model.eval()

    hf_state_dict = reference_model.state_dict()
    logger.info("Loading reference model state dict..done")
    # Setup: Convert weights and get weight_config
    weight_config = TTExpert.convert_weights(hf_config, hf_state_dict, temp_dir, mesh_device)

    # Generate appropriate config
    model_decode_config = TTExpert.decode_model_config(hf_config, mesh_device)
    model_prefill_config = TTExpert.prefill_model_config(hf_config, mesh_device)

    # Create RunConfig using both weight_config and model_config
    run_prefill_config, run_decode_config = TTExpert.run_config(
        model_prefill_config, model_decode_config, weight_config, mesh_device
    )

    # Create input tensor
    if mode == "decode":
        torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size)
    elif mode == "prefill":
        torch_input = torch.randn(1, batch_size * seq_len, hf_config.hidden_size)
    else:
        raise ValueError(f"Invalid mode: {mode}")

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
    if mode == "decode":
        tt_input = ttnn.to_memory_config(tt_input, run_decode_config["input_memory_config"])
        tt_output = TTExpert.forward_decode(tt_input, run_decode_config, mesh_device)
        expected_output_memory_config = run_decode_config["output_memory_config"]
    elif mode == "prefill":
        tt_input = ttnn.to_memory_config(tt_input, run_prefill_config["input_memory_config"])
        tt_output = TTExpert.forward_prefill(tt_input, run_prefill_config, mesh_device)
        expected_output_memory_config = run_prefill_config["output_memory_config"]
    else:
        raise ValueError(f"Invalid mode: {mode}")

    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare outputs
    pcc_required = 0.99
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"The Expert output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)


if __name__ == "__main__":
    pytest.main([__file__])

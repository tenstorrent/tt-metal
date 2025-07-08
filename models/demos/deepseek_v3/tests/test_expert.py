# models/demos/deepseek_v3/tests/test_expert.py
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
from pathlib import Path

import pytest
import torch
from loguru import logger
from transformers import AutoConfig

import ttnn

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.tt.moe import Expert as TTExpert
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3_impl.model import Expert as ReferenceExpert
from models.demos.deepseek_v3_impl.model import ModelArgs
from models.utility_functions import comp_pcc, get_mesh_device


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.mark.parametrize(
    "mesh_device",
    [get_mesh_device()],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode, seq_len, batch_size",
    [
        ("decode", 1024, 32),
    ],
)
def test_forward_pass(mode, seq_len, mesh_device, temp_dir, batch_size):
    """Test forward pass against reference model."""

    hf_config = AutoConfig.from_pretrained("deepseek-ai/DeepSeek-R1-0528", trust_remote_code=True)
    hf_config.num_hidden_layers = 1  # Reduce layers for testing

    # Get state dict from actual model - pass directly to convert_weights
    print("Loading reference model state dict")
    config_path = "models/demos/deepseek_v3_impl/configs/config_671B.json"
    with open(config_path) as f:
        model_args = ModelArgs(**json.load(f))

    logger.info("Loading reference model expert")

    reference_model = ReferenceExpert(hf_config.hidden_size, hf_config.moe_intermediate_size)
    reference_model.init_weights_with_random()
    reference_model.eval()

    hf_state_dict = reference_model.state_dict()
    logger.info("Loading reference model state dict..done")
    # Setup: Convert weights and get weight_config
    weight_config = TTExpert.convert_weights(hf_config, hf_state_dict, temp_dir, mesh_device)

    # Generate appropriate config
    if mode == "prefill":
        model_config = TTExpert.prefill_model_config(hf_config, mesh_device)
    elif mode == "decode":
        model_config = TTExpert.decode_model_config(hf_config, mesh_device)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, mesh_device)

    # Instantiate the model
    tt_expert = TTExpert(hf_config, mesh_device)

    # Create input tensor
    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

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
    tt_output = tt_expert.forward(tt_input, run_config, mesh_device)

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(tt_output)

    # Compare outputs
    pcc_required = 0.9990
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"The Expert output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)


if __name__ == "__main__":
    pytest.main([__file__])

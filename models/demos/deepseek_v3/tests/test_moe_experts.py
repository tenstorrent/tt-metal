# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0


import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE_Experts

# Import from local reference files instead of HuggingFace
from models.demos.deepseek_v3.tt.expert import Expert as TTExpert
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.utility_functions import comp_pcc


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""
    return DeepseekV3MoE_Experts(hf_config)


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 32),
        # ("prefill", 512),
        # ("prefill", 2048),  # Test chunking
    ],
)
def test_forward_pass(
    mode,
    seq_len,
    reference_model,
    hf_config_single_layer,
    temp_dir,
    galaxy_or_t3k_mesh,
):
    """Test forward pass against reference model."""
    batch_size = 1

    # Get state dict from actual model - pass directly to convert_weights
    hf_state_dict = reference_model.state_dict()

    # Setup: Convert weights and get weight_config
    weight_config = TTExpert.convert_weights_moe(hf_config_single_layer, hf_state_dict, temp_dir, galaxy_or_t3k_mesh)

    # Generate appropriate config
    if mode == "prefill":
        model_config = TTExpert.prefill_model_config(hf_config_single_layer, galaxy_or_t3k_mesh)
    else:
        model_config = TTExpert.decode_model_config(hf_config_single_layer, galaxy_or_t3k_mesh)

    # Create a new model state
    model_state = TTExpert.create_state(hf_config_single_layer, mesh_device=galaxy_or_t3k_mesh)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state)

    # Create input tensor
    torch_input = torch.randn(batch_size, 1, seq_len, hf_config_single_layer.hidden_size)

    # Reference forward pass
    reference_output = reference_model(torch_input)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input,
        device=galaxy_or_t3k_mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(galaxy_or_t3k_mesh),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # TTNN forward pass
    if mode == "prefill":
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = TTExpert.forward_prefill(tt_input, run_config)
        expected_output_memory_config = run_config["output_memory_config"]
    else:
        tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
        tt_output = TTExpert.forward_decode(tt_input, run_config)
        expected_output_memory_config = run_config["output_memory_config"]

    # Verify output memory config matches expected
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert output back to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(
            galaxy_or_t3k_mesh, dims=(-2, -1), mesh_shape=tuple(galaxy_or_t3k_mesh.shape)
        ),
    )

    # Compare outputs
    pcc_required = 0.98  # Slightly lower due to bfloat conversions
    passing, pcc_message = comp_pcc(reference_output, tt_output_torch, pcc_required)

    logger.info(f"Mode: {mode}, Seq len: {seq_len}")
    logger.info(f"PCC: {pcc_message}")

    assert passing, f"TTExpert output does not meet PCC requirement {pcc_required}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_output)


if __name__ == "__main__":
    pytest.main([__file__])

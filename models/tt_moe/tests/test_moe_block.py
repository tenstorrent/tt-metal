# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test for MoEBlock that exactly matches the structure of test_moe.py
This ensures the test works the same way as the original.
"""

import pytest
import torch
from loguru import logger

import ttnn

# Import fixtures from DeepSeek conftest
from models.demos.deepseek_v3.conftest import *  # noqa: F401,F403
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.tt.moe import MoE  # This now points to MoEBlock
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    assert_hidden_dim_pcc,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)


@pytest.fixture
def reference_model(hf_config):
    """Get the actual DeepSeek MLP model using local implementation."""
    torch.use_deterministic_algorithms(True)
    # Note : Running Reference MoE without shared experts
    hf_config.n_shared_experts = None
    return DeepseekV3MoE(hf_config).eval()


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,num_tokens",
    [
        ("decode", 128),
        ("prefill", 128),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
def test_forward_pass(
    mode,
    num_tokens,
    set_deterministic_env,
    reference_model,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
):
    """Test forward pass against reference model - identical to original test_moe.py."""

    # Get state dict from actual model - pass directly to convert_weights
    state_dict = add_inv_scale_to_state_dict(
        reference_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Create input tensor
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Reference forward pass
    reference_model.eval()
    reference_model.to(torch.bfloat16)
    with torch.no_grad():
        reference_output = reference_model(torch_input)

    # For testing, use backend="deepseek" explicitly to ensure backward compatibility
    backend = "deepseek"  # Default backend for DeepSeek tests

    # Use the standard test utility to get weight config
    # Note: get_test_weight_config doesn't support backend parameter,
    # but since we're using backend="deepseek" (the default), it will work correctly
    weight_config = get_test_weight_config(
        MoE,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=False,
        test_name="test_moe_block_deepseek",
        real_weights=False,
    )

    # Generate appropriate config with backend parameter
    model_config = get_model_config(
        MoE,
        mode,
        hf_config,
        mesh_device,
        topk_fallback=topk_fallback,
        backend=backend,  # This gets passed through to decode_model_config
    )

    # Create a new model state with CCL
    model_state = MoE.create_state(hf_config, mesh_device, ccl)

    # Create a new model shared state with backend parameter
    model_shared_state = MoE.create_shared_state(hf_config, mesh_device, backend=backend)

    # Create RunConfig using both weight_config and model_config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Convert input to TTNN, DP=4 and Replicated
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Convert to expected memory config
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    # Run forward (pass handle_tensor_parallel=True to enable collective operations)
    tt_output = run_module_forward(MoE, mode, tt_input, run_config, handle_tensor_parallel=True)

    # Verify memory configuration
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()

    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # Convert back to PyTorch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Remove batch dimension and validate
    tt_output_torch = tt_output_torch.squeeze(1)

    logger.info(f"Mode: {mode}, Num tokens: {num_tokens}")

    assert_hidden_dim_pcc(
        reference_output,
        tt_output_torch,
    )

    # Also log the actual PCC value for debugging
    from models.common.utility_functions import comp_pcc

    _, pcc_value = comp_pcc(tt_output_torch, reference_output)
    logger.info(f"PCC: {pcc_value}")

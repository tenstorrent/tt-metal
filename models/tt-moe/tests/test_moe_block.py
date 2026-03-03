# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for MoEBlock - the complete MoE module with gate and experts.

This module tests the full MoE block implementation that combines:
- GroupedTopKRouter (gate/router)
- RoutedExperts (expert processing)
- All-to-all communication
"""

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# Add tt-moe to path for local imports
tt_moe_path = str(Path(__file__).parent.parent)
if tt_moe_path not in sys.path:
    sys.path.insert(0, tt_moe_path)

# Import MoEBlock
from moe_block import MoEBlock

from models.common.utility_functions import comp_pcc

# Import reference implementation - use the ORIGINAL one from demos/deepseek_v3
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE

# Import test utilities
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)

# Test configuration constants
DEFAULT_NUM_TOKENS_DECODE = 128
DEFAULT_NUM_TOKENS_PREFILL = 128


@pytest.mark.timeout(1200)
@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", DEFAULT_NUM_TOKENS_DECODE),
        ("prefill", DEFAULT_NUM_TOKENS_PREFILL),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
def test_moe_block(
    mode,
    seq_len,
    hf_config,
    cache_path,
    mesh_device,
    set_deterministic_env,
    ccl,  # CCL fixture for all-to-all communication
    topk_fallback,
):
    """Test MoEBlock forward pass against reference implementation.

    Args:
        mode: "decode" or "prefill" mode
        seq_len: Sequence length (number of tokens to process)
        hf_config: HuggingFace model configuration
        cache_path: Path to cache directory for weights
        mesh_device: TTNN mesh device
        set_deterministic_env: Fixture to set deterministic environment
        state_dict: Model state dict with real weights
        model_path: Path to model
        ccl: CCL fixture for collective communication
    """
    batch_size = 1

    # Check device support
    if mesh_device.shape[1] != 8:
        pytest.skip(f"Device shape {mesh_device.shape} not supported for MoEBlock")

    # Use the full number of experts from the config (256)
    num_experts = hf_config.n_routed_experts

    # Verify expert distribution is valid
    if hf_config.n_routed_experts % mesh_device.get_num_devices() != 0:
        pytest.skip(
            f"Number of experts ({hf_config.n_routed_experts}) must be divisible by "
            f"number of devices ({mesh_device.get_num_devices()})"
        )

    # 1. Create reference model (using DeepseekV3MoE from reference)
    # Disable shared experts for testing
    hf_config.n_shared_experts = None

    reference_model = DeepseekV3MoE(hf_config).eval()
    reference_model.to(torch.bfloat16)

    # 2. Get state dict from reference model - use RANDOM weights like test_moe.py
    state_dict_for_ttnn = add_inv_scale_to_state_dict(
        reference_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Ensure deterministic seed for input
    torch.manual_seed(5)

    # 3. Setup TTNN configs
    weight_config = get_test_weight_config(
        MoEBlock,
        hf_config,
        (state_dict_for_ttnn,),
        cache_path,
        mesh_device,
        force_recalculate=False,  # Use cache if available
        test_name="test_moe",  # Use same cache name as original test_moe.py
        real_weights=False,  # Using random weights like test_moe.py
    )

    model_config = get_model_config(MoEBlock, mode, hf_config, mesh_device, topk_fallback=topk_fallback)

    # Create model state with CCL
    model_state = MoEBlock.create_state(hf_config, mesh_device, ccl)

    # Create shared state
    model_shared_state = MoEBlock.create_shared_state(hf_config, mesh_device)

    # Create run config
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # 4. Generate test input
    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # 5. Run reference model
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(
        f"Reference output stats - mean: {ref_output.mean():.6f}, "
        f"std: {ref_output.std():.6f}, min: {ref_output.min():.6f}, max: {ref_output.max():.6f}"
    )

    # 6. Convert input to TTNN
    # MoE expects input sharded across devices for data parallel
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),  # Add DP dimension
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # 7. Move to expected memory config and run forward
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    # Pass handle_tensor_parallel=True to enable collective operations
    tt_output = run_module_forward(MoEBlock, mode, tt_input, run_config, handle_tensor_parallel=True)

    # 8. Verify memory configuration
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # 9. Convert TTNN output to torch
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Remove the DP dimension
    tt_output_torch = tt_output_torch.squeeze(1)

    logger.info(f"TTNN output shape: {tt_output_torch.shape}")
    logger.info(
        f"TTNN output stats - mean: {tt_output_torch.mean():.6f}, "
        f"std: {tt_output_torch.std():.6f}, min: {tt_output_torch.min():.6f}, max: {tt_output_torch.max():.6f}"
    )

    # 10. Validate with PCC
    min_pcc = 0.98
    passed, pcc_value = comp_pcc(tt_output_torch, ref_output, pcc=min_pcc)

    logger.info(f"PCC: {pcc_value:.6f}, passed: {passed}")

    # 11. Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    assert passed, f"PCC check failed! PCC: {pcc_value:.6f} < {min_pcc}"

    logger.info(f"✓ Test passed for mode={mode}, seq_len={seq_len}")


@pytest.mark.parametrize(
    "device_params",
    [
        {"fabric_config": ttnn.FabricConfig.FABRIC_1D},
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", DEFAULT_NUM_TOKENS_DECODE),
    ],
)
@pytest.mark.parametrize(
    "topk_fallback",
    [
        True,
    ],
)
def test_moe_block_simplified(
    mode,
    seq_len,
    hf_config,
    cache_path,
    mesh_device,
    set_deterministic_env,
    ccl,
    topk_fallback,
):
    """Test MoEBlock with simplified reference model using random weights.

    This test uses SimplifiedMoEBlock which is easier to debug.
    """
    batch_size = 1

    # Check device support
    if mesh_device.shape[1] != 8:
        pytest.skip(f"Device shape {mesh_device.shape} not supported for MoEBlock")

    # 1. Create reference model using DeepseekV3MoE
    # Disable shared experts for this test
    hf_config.n_shared_experts = None

    reference_model = DeepseekV3MoE(hf_config).eval()
    reference_model.to(torch.bfloat16)

    # 2. Create random weights with quantization
    state_dict_for_ttnn = add_inv_scale_to_state_dict(
        reference_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # 3. Setup TTNN configs
    weight_config = get_test_weight_config(
        MoEBlock,
        hf_config,
        (state_dict_for_ttnn,),
        cache_path,
        mesh_device,
        force_recalculate=False,  # Use cache if available
        test_name="test_moe",  # Use same cache name as original test_moe.py
        real_weights=False,  # Using random weights
    )

    model_config = get_model_config(MoEBlock, mode, hf_config, mesh_device, topk_fallback=topk_fallback)
    model_state = MoEBlock.create_state(hf_config, mesh_device, ccl)
    model_shared_state = MoEBlock.create_shared_state(hf_config, mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # 4. Generate test input
    torch_input = torch.randn(batch_size, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # 5. Run reference model
    with torch.no_grad():
        ref_output = reference_model(torch_input)

    logger.info(f"Reference output shape: {ref_output.shape}")
    logger.info(
        f"Reference output stats - mean: {ref_output.mean():.6f}, "
        f"std: {ref_output.std():.6f}, min: {ref_output.min():.6f}, max: {ref_output.max():.6f}"
    )

    # 6. Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # 7. Run TTNN forward
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(MoEBlock, mode, tt_input, run_config, handle_tensor_parallel=True)

    # 8. Convert output and compare
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )
    tt_output_torch = tt_output_torch.squeeze(1)

    logger.info(f"TTNN output shape: {tt_output_torch.shape}")
    logger.info(
        f"TTNN output stats - mean: {tt_output_torch.mean():.6f}, "
        f"std: {tt_output_torch.std():.6f}, min: {tt_output_torch.min():.6f}, max: {tt_output_torch.max():.6f}"
    )

    # 9. Validate with PCC
    min_pcc = 0.95  # Lower threshold for random weights
    passed, pcc_value = comp_pcc(tt_output_torch, ref_output, pcc=min_pcc)

    logger.info(f"PCC: {pcc_value:.6f}, passed: {passed}")

    # 10. Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    assert passed, f"PCC check failed! PCC: {pcc_value:.6f} < {min_pcc}"

    logger.info(f"✓ Simplified test passed for mode={mode}, seq_len={seq_len}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

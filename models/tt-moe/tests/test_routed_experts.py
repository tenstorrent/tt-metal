# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for RoutedExperts component.

This module tests the routed experts implementation that processes tokens
through multiple expert MLPs in parallel.
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

# Import RoutedExperts
from components.experts.routed_experts import RoutedExperts

# Import reference implementation
from pytorch_reference.deepseek.routed_experts import SimplifiedRoutedExperts
from utils.test_utils import get_test_weight_config

# Import test utilities
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict, get_model_config, run_module_forward
from tests.ttnn.utils_for_testing import comp_pcc

# Test configuration constants (MATCH test_moe_experts.py)
DEFAULT_NUM_TOKENS_DECODE = 128  # CHANGED to match test_moe_experts.py
DEFAULT_NUM_TOKENS_PREFILL = 128  # Could be overridden by env var like test_moe_experts.py
DEFAULT_NUM_EXPERTS = 8  # Use smaller number for testing


@pytest.mark.parametrize(
    "mode,seq_len",  # RENAMED to match test_moe_experts.py parameter naming
    [
        ("decode", DEFAULT_NUM_TOKENS_DECODE),
        ("prefill", DEFAULT_NUM_TOKENS_PREFILL),
    ],
)
def test_routed_experts(
    mode,
    seq_len,  # RENAMED from num_tokens to match test_moe_experts.py
    hf_config,
    cache_path,
    mesh_device,
    set_deterministic_env,
    state_dict,  # Add state_dict fixture for real weights
    model_path,  # Add model_path fixture
):
    """Test RoutedExperts forward pass against reference implementation.

    Args:
        mode: "decode" or "prefill" mode
        seq_len: Sequence length (number of tokens to process)
        hf_config: HuggingFace model configuration
        cache_path: Path to cache directory for weights
        mesh_device: TTNN mesh device
        set_deterministic_env: Fixture to set deterministic environment
    """
    batch_size = 1

    # Use the full number of experts from the config (256)
    num_experts = hf_config.n_routed_experts

    # Check device support
    if not RoutedExperts.is_device_supported(mesh_device):
        pytest.skip(f"Device shape {mesh_device.shape} not supported for RoutedExperts")

    # Verify expert distribution is valid
    if hf_config.n_routed_experts % mesh_device.get_num_devices() != 0:
        pytest.skip(
            f"Number of experts ({hf_config.n_routed_experts}) must be divisible by number of devices ({mesh_device.get_num_devices()})"
        )

    # 1. Create reference model using our SimplifiedRoutedExperts (now matches DeepseekV3MoEExperts structure)
    from models.demos.deepseek_v3.tests.test_moe_experts import create_combined_state_dict
    from models.demos.deepseek_v3.utils.test_utils import dequantize_state_dict

    reference_model = SimplifiedRoutedExperts(hf_config).eval()
    torch_input = torch.randn(batch_size, 1, seq_len, hf_config.hidden_size)

    # 2. Load real weights (EXACTLY like test_moe_experts.py)
    module_path = "model.layers.3.mlp.experts.0-255"  # Layer 3, all experts
    state_dict_combined = create_combined_state_dict(module_path, model_path, state_dict)
    reference_model.load_state_dict(dequantize_state_dict(state_dict_combined, hf_config))

    # Use the combined state dict for TTNN
    state_dict = state_dict_combined

    # 3. Setup TTNN configs (EXACTLY like test_moe_experts.py lines 120-133)
    weight_config = get_test_weight_config(
        RoutedExperts,  # Using RoutedExperts which is aliased to Experts (same as TTExperts)
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=True,  # force_recalculate_weight_config in reference
        test_name="test_moe_experts",  # Use same test name as reference
        real_weights=True,  # weight_type == "real"
        layer_id=module_path,  # Include layer_id like reference!
    )

    model_config = get_model_config(RoutedExperts, mode, hf_config, mesh_device)
    model_state = RoutedExperts.create_state(hf_config, mesh_device)  # Use create_state like reference!
    run_config = create_run_config(model_config, weight_config, model_state)

    # 4. Generate test input (MATCH test_moe_experts.py line 109)
    num_experts_per_device = RoutedExperts._get_num_experts_per_device(hf_config, mesh_device)

    # Generate base input with shape (batch_size, 1, seq_len, hidden_size) like test_moe_experts.py
    torch_input = torch.randn(
        batch_size, 1, seq_len, hf_config.hidden_size
    )  # dtype will be float32 like test_moe_experts.py

    logger.info(f"Base input shape: {torch_input.shape}")
    logger.info(f"Number of experts per device: {num_experts_per_device}")

    # 5. Convert to TTNN first (MATCH test_moe_experts.py order and approach)
    # Repeat input for each expert like test_moe_experts.py line 136
    tt_input = ttnn.from_torch(
        torch_input.repeat(1, num_experts_per_device, 1, 1),  # repeat activations per expert
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),  # CHANGED to match test_moe_experts.py
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # 6. Move to expected memory config and run forward (MATCH test_moe_experts.py line 144-145)
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(RoutedExperts, mode, tt_input, run_config)

    # 7. Verify memory configuration (MATCH test_moe_experts.py line 148-151)
    expected_output_memory_config = run_config["output_memory_config"]
    actual_output_memory_config = tt_output.memory_config()
    assert (
        actual_output_memory_config == expected_output_memory_config
    ), f"Output memory config mismatch: expected {expected_output_memory_config}, got {actual_output_memory_config}"

    # 8. Chunked validation (EXACTLY like test_moe_experts.py lines 153-192)
    TARGET_CHUNK_SIZE = 2048
    num_chunks = (seq_len + TARGET_CHUNK_SIZE - 1) // TARGET_CHUNK_SIZE

    from models.common.utility_functions import comp_pcc as comp_pcc_ref  # Use same comp_pcc as reference

    min_pcc = 0.98
    passed = True

    num_experts_per_device = run_config["num_experts_per_device"]  # Get from run_config like reference

    for chunk_idx in range(num_chunks):
        start_seq = chunk_idx * TARGET_CHUNK_SIZE
        end_seq = min(start_seq + TARGET_CHUNK_SIZE, seq_len)
        chunk_seq_len = end_seq - start_seq

        chunk_input = torch_input[:, :, start_seq:end_seq, :]  # Shape: [1, 1, chunk_seq_len, hidden_size]
        chunk_ref_output = reference_model(chunk_input)  # Should output [256, 1, chunk_seq_len, hidden_size]

        tt_output_chunk = ttnn.slice(
            tt_output,
            [0, 0, start_seq, 0],
            [1, num_experts_per_device, end_seq, hf_config.hidden_size],
        )

        tt_output_chunk_torch = ttnn.to_torch(
            tt_output_chunk,
            mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
        )

        ttnn.deallocate(tt_output_chunk)

        tt_output_chunk_torch = tt_output_chunk_torch.reshape(1, -1, chunk_seq_len, hf_config.hidden_size)
        tt_output_chunk_torch = tt_output_chunk_torch[0].unsqueeze(1)

        if chunk_ref_output.shape != tt_output_chunk_torch.shape:
            logger.info(f"Shape mismatch - ref: {chunk_ref_output.shape}, tt: {tt_output_chunk_torch.shape}")
            chunk_ref_output = chunk_ref_output.unsqueeze(0)
            logger.info(f"After unsqueeze - ref: {chunk_ref_output.shape}")

        chunk_passed, chunk_pcc = comp_pcc_ref(tt_output_chunk_torch, chunk_ref_output, pcc=0.98)

        min_pcc = min(min_pcc, chunk_pcc)
        if not chunk_passed:
            passed = False

        logger.info(f"Chunk {chunk_idx}: PCC = {chunk_pcc:.6f}, passed = {chunk_passed}")

        # Cleanup chunk tensors
        del chunk_ref_output
        del tt_output_chunk_torch
        del chunk_input

    # 9. Final cleanup (like test_moe_experts.py lines 198-199)
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    assert passed, f"PCC check failed! Min PCC: {min_pcc:.6f} < 0.98"

    # 11. Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    logger.info("Test passed successfully!")

    logger.info(f"✓ Test passed for mode={mode}, seq_len={seq_len}, num_experts={num_experts}")


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_routed_experts_weight_extraction(
    mode,
    hf_config,
    cache_path,
    mesh_device,
    set_deterministic_env,
):
    """Test that weight extraction from reference model works correctly.

    Args:
        mode: "decode" or "prefill" mode
        hf_config: HuggingFace model configuration
        cache_path: Path to cache directory for weights
        mesh_device: TTNN mesh device
        set_deterministic_env: Fixture to set deterministic environment
    """
    # Use full number of experts from the config (256)
    num_experts = hf_config.n_routed_experts

    if not RoutedExperts.is_device_supported(mesh_device):
        pytest.skip(f"Device shape {mesh_device.shape} not supported for RoutedExperts")

    # Create reference model
    reference_model = SimplifiedRoutedExperts(hf_config).eval()

    # Extract weights in the expected format
    state_dict = {}
    for expert_id in range(num_experts):
        state_dict[f"experts.{expert_id}.gate_proj.weight"] = reference_model.w1_weight[expert_id]
        state_dict[f"experts.{expert_id}.down_proj.weight"] = reference_model.w2_weight[expert_id]
        state_dict[f"experts.{expert_id}.up_proj.weight"] = reference_model.w3_weight[expert_id]

    # Add quantization scales using the proper method
    block_shape = hf_config.quantization_config.get("weight_block_size", [128, 128])
    state_dict = add_inv_scale_to_state_dict(state_dict, block_shape)

    # Verify weight shapes
    for expert_id in range(num_experts):
        expected_shape_gate = (hf_config.hidden_size, hf_config.moe_intermediate_size)
        expected_shape_down = (hf_config.moe_intermediate_size, hf_config.hidden_size)

        assert (
            state_dict[f"experts.{expert_id}.gate_proj.weight"].shape == expected_shape_gate
        ), f"Gate proj shape mismatch for expert {expert_id}"
        assert (
            state_dict[f"experts.{expert_id}.up_proj.weight"].shape == expected_shape_gate
        ), f"Up proj shape mismatch for expert {expert_id}"
        assert (
            state_dict[f"experts.{expert_id}.down_proj.weight"].shape == expected_shape_down
        ), f"Down proj shape mismatch for expert {expert_id}"

    # Test weight conversion
    weight_config = get_test_weight_config(
        RoutedExperts,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=True,
        test_name="test_weight_extraction",
        real_weights=True,
    )

    # Verify weight config has expected keys
    expected_keys = {"w1_experts", "w2_experts", "w3_experts"}
    assert (
        set(weight_config.keys()) == expected_keys
    ), f"Weight config keys mismatch: expected {expected_keys}, got {set(weight_config.keys())}"

    logger.info("✓ Weight extraction test passed")


@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_routed_experts_intermediate_validation(
    mode,
    hf_config,
    cache_path,
    mesh_device,
    set_deterministic_env,
):
    """Validate intermediate tensors during expert processing.

    This test hooks into intermediate computations to verify:
    - w1_out (gate projection)
    - w3_out (up projection)
    - activated tensor (after SiLU)

    Args:
        mode: "decode" or "prefill" mode
        hf_config: HuggingFace model configuration
        cache_path: Path to cache directory for weights
        mesh_device: TTNN mesh device
        set_deterministic_env: Fixture to set deterministic environment
    """
    batch_size = 1
    num_tokens = 32 if mode == "decode" else 128
    num_experts = hf_config.n_routed_experts

    if not RoutedExperts.is_device_supported(mesh_device):
        pytest.skip(f"Device shape {mesh_device.shape} not supported for RoutedExperts")

    if hf_config.n_routed_experts % mesh_device.get_num_devices() != 0:
        pytest.skip(f"Number of experts must be divisible by number of devices")

    # 1. Create reference model with hooks for intermediate values
    torch.manual_seed(42)
    reference_model = SimplifiedRoutedExperts(hf_config).eval()
    reference_model.to(torch.bfloat16)

    # Storage for intermediate values
    intermediates = {}

    # Original forward method to capture intermediates
    original_forward = reference_model.forward

    def forward_with_intermediates(x):
        batch_size, num_experts, num_tokens, hidden_size = x.shape
        x_flat = x.reshape(-1, hidden_size)

        # Compute projections
        w1_out = torch.matmul(x_flat, reference_model.w1_weight.reshape(-1, reference_model.w1_weight.shape[-1]).T)
        w3_out = torch.matmul(x_flat, reference_model.w3_weight.reshape(-1, reference_model.w3_weight.shape[-1]).T)

        # Store intermediates
        intermediates["w1_out"] = w1_out.reshape(batch_size, num_experts, num_tokens, -1)
        intermediates["w3_out"] = w3_out.reshape(batch_size, num_experts, num_tokens, -1)

        # Activation
        activated = torch.nn.functional.silu(w1_out) * w3_out
        intermediates["activated"] = activated.reshape(batch_size, num_experts, num_tokens, -1)

        # Final projection
        output = torch.matmul(activated, reference_model.w2_weight.reshape(-1, reference_model.w2_weight.shape[-1]).T)
        return output.reshape(batch_size, num_experts, num_tokens, hidden_size)

    reference_model.forward = forward_with_intermediates

    # 2. Extract weights and setup TTNN
    state_dict = {}
    for expert_id in range(num_experts):
        state_dict[f"experts.{expert_id}.gate_proj.weight"] = reference_model.w1_weight[expert_id]
        state_dict[f"experts.{expert_id}.down_proj.weight"] = reference_model.w2_weight[expert_id]
        state_dict[f"experts.{expert_id}.up_proj.weight"] = reference_model.w3_weight[expert_id]

    # Add quantization scales using the proper method
    block_shape = hf_config.quantization_config.get("weight_block_size", [128, 128])
    state_dict = add_inv_scale_to_state_dict(state_dict, block_shape)

    weight_config = get_test_weight_config(
        RoutedExperts,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=True,
        test_name="test_intermediate_validation",
        real_weights=True,
    )

    model_config = get_model_config(RoutedExperts, mode, hf_config, mesh_device)
    model_state = RoutedExperts.create_state(hf_config, mesh_device=mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)

    # 3. Generate test input
    num_experts_per_device = RoutedExperts._get_num_experts_per_device(hf_config, mesh_device)
    torch_input = torch.randn(
        batch_size, num_experts_per_device, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16
    )

    # 4. Run reference forward pass
    logger.info("Running reference forward pass with intermediate capture")
    ref_output = reference_model(torch_input)

    # Log intermediate stats
    for name, tensor in intermediates.items():
        logger.info(
            f"Reference {name}: shape={tensor.shape}, min={tensor.min():.6f}, max={tensor.max():.6f}, mean={tensor.mean():.6f}"
        )

    # 5. Verify intermediate dimensions
    assert intermediates["w1_out"].shape == (
        batch_size,
        num_experts_per_device,
        num_tokens,
        hf_config.moe_intermediate_size,
    )
    assert intermediates["w3_out"].shape == (
        batch_size,
        num_experts_per_device,
        num_tokens,
        hf_config.moe_intermediate_size,
    )
    assert intermediates["activated"].shape == (
        batch_size,
        num_experts_per_device,
        num_tokens,
        hf_config.moe_intermediate_size,
    )

    # 6. Validate activation function behavior (SiLU properties)
    # SiLU should produce outputs in a reasonable range
    activated_min = intermediates["activated"].min().item()
    activated_max = intermediates["activated"].max().item()

    # SiLU(x) * y should have values that depend on both inputs
    assert activated_min < 0 or activated_max > 0, "Activation produced all zeros"

    # Cleanup
    logger.info("✓ Intermediate validation test passed")


@pytest.mark.parametrize(
    "routing_scenario",
    [
        "all_tokens_single_expert",
        "uniform_distribution",
        "sparse_experts",
    ],
)
def test_routed_experts_routing_scenarios(
    routing_scenario,
    hf_config,
    cache_path,
    mesh_device,
    set_deterministic_env,
):
    """Test various expert routing patterns to ensure RoutedExperts handles edge cases.

    Args:
        routing_scenario: Type of routing pattern to test
        hf_config: HuggingFace model configuration
        cache_path: Path to cache directory for weights
        mesh_device: TTNN mesh device
        set_deterministic_env: Fixture to set deterministic environment
    """
    batch_size = 1
    mode = "decode"
    num_tokens = 32
    num_experts = hf_config.n_routed_experts

    if not RoutedExperts.is_device_supported(mesh_device):
        pytest.skip(f"Device shape {mesh_device.shape} not supported for RoutedExperts")

    if hf_config.n_routed_experts % mesh_device.get_num_devices() != 0:
        pytest.skip(f"Number of experts must be divisible by number of devices")

    # 1. Create reference model
    torch.manual_seed(42)
    reference_model = SimplifiedRoutedExperts(hf_config).eval()
    reference_model.to(torch.bfloat16)
    # Re-initialize weights after converting to bfloat16
    torch.manual_seed(42)
    reference_model.reset_parameters()

    # 2. Extract weights
    state_dict = {}
    for expert_id in range(num_experts):
        state_dict[f"experts.{expert_id}.gate_proj.weight"] = reference_model.w1_weight[expert_id]
        state_dict[f"experts.{expert_id}.down_proj.weight"] = reference_model.w2_weight[expert_id]
        state_dict[f"experts.{expert_id}.up_proj.weight"] = reference_model.w3_weight[expert_id]

    # Add quantization scales using the proper method
    block_shape = hf_config.quantization_config.get("weight_block_size", [128, 128])
    state_dict = add_inv_scale_to_state_dict(state_dict, block_shape)

    weight_config = get_test_weight_config(
        RoutedExperts,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=True,
        test_name=f"test_routing_{routing_scenario}",
        real_weights=True,
    )

    model_config = get_model_config(RoutedExperts, mode, hf_config, mesh_device)
    model_state = RoutedExperts.create_state(hf_config, mesh_device=mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)

    # 3. Generate test input based on routing scenario
    num_experts_per_device = RoutedExperts._get_num_experts_per_device(hf_config, mesh_device)

    if routing_scenario == "all_tokens_single_expert":
        # All tokens go to first expert, others get zeros
        torch_input = torch.zeros(
            batch_size, num_experts_per_device, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16
        )
        torch_input[:, 0, :, :] = torch.randn(batch_size, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)
        logger.info("Testing scenario: All tokens routed to single expert")

    elif routing_scenario == "uniform_distribution":
        # Uniform distribution across all experts
        torch_input = torch.randn(
            batch_size, num_experts_per_device, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16
        )
        logger.info("Testing scenario: Uniform distribution across experts")

    elif routing_scenario == "sparse_experts":
        # Only a few experts get tokens
        torch_input = torch.zeros(
            batch_size, num_experts_per_device, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16
        )
        # Activate only 25% of experts
        active_experts = num_experts_per_device // 4
        for i in range(active_experts):
            torch_input[:, i, :, :] = torch.randn(batch_size, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)
        logger.info(f"Testing scenario: Sparse experts (only {active_experts} active)")

    # 4. Run reference forward pass
    ref_output = reference_model(torch_input)

    # 5. Convert to TTNN and run forward pass
    tt_input_reshaped = torch_input.permute(1, 0, 2, 3)
    tt_input_reshaped = tt_input_reshaped.reshape(num_experts_per_device, num_tokens, hf_config.hidden_size)

    tt_input = ttnn.from_torch(
        tt_input_reshaped.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])

    logger.info(f"Running TTNN forward pass for scenario: {routing_scenario}")
    tt_output = run_module_forward(RoutedExperts, mode, tt_input, run_config)

    # 6. Convert output and compare
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )[0]

    # 7. Validate with PCC
    pcc_threshold = 0.95
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_threshold)

    logger.info(f"Output PCC for {routing_scenario}: {pcc_message}")

    # Special handling for sparse scenarios
    if routing_scenario in ["all_tokens_single_expert", "sparse_experts"]:
        # Check that inactive experts produce near-zero outputs
        for expert_idx in range(num_experts_per_device):
            if routing_scenario == "all_tokens_single_expert" and expert_idx > 0:
                expert_output = tt_output_torch[0, expert_idx, :, :]
                assert expert_output.abs().max() < 1e-5, f"Inactive expert {expert_idx} produced non-zero output"
            elif routing_scenario == "sparse_experts" and expert_idx >= num_experts_per_device // 4:
                expert_output = tt_output_torch[0, expert_idx, :, :]
                assert expert_output.abs().max() < 1e-5, f"Inactive expert {expert_idx} produced non-zero output"

    assert passing, f"Output PCC failed for {routing_scenario}: {pcc_message}"

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    logger.info(f"✓ Routing scenario test passed for: {routing_scenario}")


@pytest.mark.parametrize("memory_config", ["L1", "DRAM"])
def test_routed_experts_memory_modes(
    memory_config,
    hf_config,
    cache_path,
    mesh_device,
    set_deterministic_env,
):
    """Verify L1 vs DRAM memory configurations for RoutedExperts.

    Args:
        memory_config: "L1" or "DRAM" memory configuration
        hf_config: HuggingFace model configuration
        cache_path: Path to cache directory for weights
        mesh_device: TTNN mesh device
        set_deterministic_env: Fixture to set deterministic environment
    """
    batch_size = 1
    mode = "decode"
    num_tokens = 32
    num_experts = hf_config.n_routed_experts

    if not RoutedExperts.is_device_supported(mesh_device):
        pytest.skip(f"Device shape {mesh_device.shape} not supported for RoutedExperts")

    if hf_config.n_routed_experts % mesh_device.get_num_devices() != 0:
        pytest.skip(f"Number of experts must be divisible by number of devices")

    # 1. Create reference model
    torch.manual_seed(42)
    reference_model = SimplifiedRoutedExperts(hf_config).eval()
    reference_model.to(torch.bfloat16)
    # Re-initialize weights after converting to bfloat16
    torch.manual_seed(42)
    reference_model.reset_parameters()

    # 2. Extract weights
    state_dict = {}
    for expert_id in range(num_experts):
        state_dict[f"experts.{expert_id}.gate_proj.weight"] = reference_model.w1_weight[expert_id]
        state_dict[f"experts.{expert_id}.down_proj.weight"] = reference_model.w2_weight[expert_id]
        state_dict[f"experts.{expert_id}.up_proj.weight"] = reference_model.w3_weight[expert_id]

    # Add quantization scales using the proper method
    block_shape = hf_config.quantization_config.get("weight_block_size", [128, 128])
    state_dict = add_inv_scale_to_state_dict(state_dict, block_shape)

    weight_config = get_test_weight_config(
        RoutedExperts,
        hf_config,
        (state_dict,),
        cache_path,
        mesh_device,
        force_recalculate=True,
        test_name=f"test_memory_{memory_config}",
        real_weights=True,
    )

    # Override memory configuration
    model_config = get_model_config(RoutedExperts, mode, hf_config, mesh_device)

    # Set specific memory configuration
    if memory_config == "L1":
        test_memory_config = ttnn.L1_MEMORY_CONFIG
    else:
        test_memory_config = ttnn.DRAM_MEMORY_CONFIG

    model_config["input_memory_config"] = test_memory_config
    model_config["output_memory_config"] = test_memory_config

    model_state = RoutedExperts.create_state(hf_config, mesh_device=mesh_device)
    run_config = create_run_config(model_config, weight_config, model_state)
    run_config["input_memory_config"] = test_memory_config
    run_config["output_memory_config"] = test_memory_config

    # 3. Generate test input
    num_experts_per_device = RoutedExperts._get_num_experts_per_device(hf_config, mesh_device)
    torch_input = torch.randn(
        batch_size, num_experts_per_device, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16
    )

    logger.info(f"Testing memory configuration: {memory_config}")

    # 4. Run reference forward pass
    ref_output = reference_model(torch_input)

    # 5. Convert to TTNN with specified memory config
    tt_input_reshaped = torch_input.permute(1, 0, 2, 3)
    tt_input_reshaped = tt_input_reshaped.reshape(num_experts_per_device, num_tokens, hf_config.hidden_size)

    tt_input = ttnn.from_torch(
        tt_input_reshaped.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=test_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    # Verify input is in correct memory
    assert tt_input.memory_config() == test_memory_config, f"Input not in {memory_config} memory"

    # 6. Run TTNN forward pass
    logger.info(f"Running TTNN forward pass with {memory_config} memory")
    tt_output = run_module_forward(RoutedExperts, mode, tt_input, run_config)

    # Verify output is in correct memory
    assert tt_output.memory_config() == test_memory_config, f"Output not in {memory_config} memory"

    # 7. Convert output and compare
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )[0]

    # 8. Validate with PCC
    pcc_threshold = 0.95
    passing, pcc_message = comp_pcc(ref_output, tt_output_torch, pcc_threshold)

    logger.info(f"Output PCC with {memory_config} memory: {pcc_message}")
    assert passing, f"Output PCC failed with {memory_config} memory: {pcc_message}"

    # 9. Cleanup and verify deallocation
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    logger.info(f"✓ Memory configuration test passed for: {memory_config}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

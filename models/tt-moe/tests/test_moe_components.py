#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for individual MoE block components.
These tests isolate each operation to identify where the hang occurs.
"""

import json
import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
# Add demos directory for reference utilities
sys.path.append(str(Path(__file__).parent.parent.parent / "demos" / "deepseek_v3"))

from components.routers.grouped_topk_router import GroupedTopKRouter
from utils.ccl import CCL

# Test configuration constants
BATCH_SIZE = 32
HIDDEN_SIZE = 7168
NUM_EXPERTS_PER_TOK = 6
CONFIG_PATH = str(Path(__file__).parent.parent / "configs" / "deepseek_v3.json")


@pytest.fixture(scope="function")
def mesh_device():
    """Create and cleanup mesh device."""
    device_type = os.getenv("MESH_DEVICE", "TG").upper()

    mesh_shapes = {
        "TG": (4, 8),
        "T3K": (1, 8),
    }

    if device_type not in mesh_shapes:
        pytest.skip(f"Device type {device_type} not supported")

    mesh_shape = mesh_shapes[device_type]

    # Set fabric configuration for multi-device systems
    fabric_config = None
    if device_type in ["TG"]:
        fabric_config = ttnn.FabricConfig.FABRIC_1D
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,  # eth_buffer_size
            ttnn.FabricTensixConfig.DISABLED,
            ttnn.FabricUDMMode.DISABLED,
            ttnn.FabricManagerMode.DEFAULT,
        )

    # Open mesh device
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))

    logger.info(f"Created mesh device: {device_type} with shape {mesh_shape}")
    yield device

    # Cleanup
    ttnn.close_mesh_device(device)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="function")
def ccl(mesh_device):
    """Create CCL instance for collective operations."""
    return CCL(mesh_device)


# MoEBlock fixture removed - tests now use components directly with class methods


def test_01_all_gather(mesh_device, ccl):
    """Test 1: All-gather operation at the beginning of forward pass."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: ALL-GATHER OPERATION")
    logger.info("=" * 60)

    # Create input tensor (replicated across devices)
    torch_input = torch.randn(1, 1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )
    tt_input = ttnn.to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)

    logger.info(f"Input shape: {tt_input.shape}")

    # Test all-gather if tensor parallel is enabled
    tp_axis = 1  # From deepseek_v3.json config
    tp_enabled = True

    if tp_enabled:
        logger.info(f"Testing all-gather on axis {tp_axis}")

        # Get CCL parameters for all_gather
        all_gather_config = {
            "cluster_axis": tp_axis,
            "dim": -1,  # Last dimension for hidden states
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "topology": ttnn.Topology.Linear,
        }

        # Use CCL to populate runtime arguments (semaphores)
        all_gather_args = ccl.populate_all_gather_runtime_args(all_gather_config)

        logger.info("Calling all_gather_async...")
        output = ttnn.experimental.all_gather_async(tt_input, **all_gather_args)
        logger.info(f"All-gather output shape: {output.shape}")

        # Cleanup
        ttnn.deallocate(output)

    ttnn.deallocate(tt_input)
    logger.info("✅ All-gather test passed")


def test_02_router_forward(mesh_device):
    """Test 2: Router forward operation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: ROUTER FORWARD")
    logger.info("=" * 60)

    # Create router configuration
    router_config = {
        "hidden_size": HIDDEN_SIZE,
        "n_routed_experts": 256,
        "num_experts_per_tok": NUM_EXPERTS_PER_TOK,
        "num_experts": 256,  # Also add num_experts for compatibility
        "routed_scaling_factor": 1.0,
        "n_group": 1,
        "topk_group": 1,
        "score_correction_bias": True,  # Enable to test bias loading
        "memory_config": "L1_MEMORY_CONFIG",
    }

    # Create router instance
    router = GroupedTopKRouter(router_config, mesh_device)

    # Load mock weights for the router
    mock_weights = {
        "weight": torch.randn(router_config["n_routed_experts"], router_config["hidden_size"], dtype=torch.bfloat16),
        "e_score_correction_bias": torch.zeros(router_config["n_routed_experts"], dtype=torch.bfloat16),
    }
    router.load_weights(mock_weights)

    # Create input tensor
    torch_input = torch.randn(1, 1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Input shape: {tt_input.shape}")

    # Test router forward
    logger.info("Calling router.forward...")
    weights, indices = router.forward(tt_input, mode="decode")

    logger.info(f"Router weights shape: {weights.shape}")
    logger.info(f"Router indices shape: {indices.shape}")

    # Cleanup
    ttnn.deallocate(weights)
    ttnn.deallocate(indices)
    ttnn.deallocate(tt_input)
    logger.info("✅ Router forward test passed")


def test_03_prepare_expert_weights(mesh_device):
    """Test 3: Prepare expert weights operation for dispatch."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: PREPARE EXPERT WEIGHTS FOR DISPATCH")
    logger.info("=" * 60)

    # Create mock router output weights matching the actual router output shape
    # Router outputs: [1, 1, batch_size, num_experts_per_tok]
    torch_weights = torch.randn(1, 1, BATCH_SIZE, NUM_EXPERTS_PER_TOK, dtype=torch.bfloat16)

    # Use DRAM for initial tensor to avoid L1 memory issues with large repeat
    tt_weights = ttnn.from_torch(
        torch_weights,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Use DRAM for large operations
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Input weights shape: {tt_weights.shape}")

    # Test weight preparation by repeating for hidden size dimension
    # This simulates what would happen before all_to_all_dispatch
    logger.info("Preparing weights by repeating for hidden dimension...")

    # Repeat weights along hidden dimension to match hidden_size
    # ttnn.repeat takes repeat_dims parameter (how many times to repeat in each dimension)
    # We want to go from shape [1, 1, 32, 6] to [1, 1, 32, 7168]
    # So we need to repeat 7168/6 = 1194.67, which is not an integer
    # Instead, we'll use a different approach - tile the weights to the right size
    weights_prepared = ttnn.repeat(
        tt_weights, repeat_dims=[1, 1, 1, HIDDEN_SIZE // NUM_EXPERTS_PER_TOK], memory_config=ttnn.DRAM_MEMORY_CONFIG
    )

    logger.info(f"Prepared weights shape: {weights_prepared.shape}")

    # Cleanup
    ttnn.deallocate(weights_prepared)
    ttnn.deallocate(tt_weights)
    logger.info("✅ Prepare expert weights test passed")


@pytest.mark.parametrize(
    "arch_config",
    [
        {
            "name": "deepseek",
            "topology": ttnn.Topology.Linear,
            "num_experts": 256,
            "experts_per_device": 8,
            "experts_per_tok": 8,
            "cluster_axis": 0,
            "apply_all_reduce": False,
            "all_reduce_axis": None,
            "hidden_size": 7168,
            "batch_size": 32,
            "seq_len": 1,
            "memory_config": "L1_MEMORY_CONFIG",
        },
        {
            "name": "gpt_oss",
            "topology": ttnn.Topology.Linear,  # Changed from Ring to Linear to avoid routing issues
            "num_experts": 128,
            "experts_per_device": 4,
            "experts_per_tok": 4,
            "cluster_axis": 0,
            "apply_all_reduce": True,
            "all_reduce_axis": 1,
            "hidden_size": 2880,
            "batch_size": 1,
            "seq_len": 32,
            "memory_config": "DRAM_MEMORY_CONFIG",
        },
    ],
)
def test_04_all_to_all_dispatch_and_combine(mesh_device, arch_config):
    """Test 4: Combined all-to-all dispatch and combine for both architectures."""
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST 4: ALL-TO-ALL DISPATCH AND COMBINE - {arch_config['name'].upper()}")
    logger.info("=" * 60)

    # Import the all-to-all operations
    from components.collective.all_to_all_ops import AllToAllCombiner, AllToAllDispatcher

    # Extract architecture parameters
    num_experts = arch_config["num_experts"]
    experts_per_device = arch_config["experts_per_device"]
    experts_per_tok = arch_config["experts_per_tok"]
    topology = arch_config["topology"]
    cluster_axis = arch_config["cluster_axis"]
    apply_all_reduce = arch_config["apply_all_reduce"]
    all_reduce_axis = arch_config["all_reduce_axis"]
    hidden_size = arch_config["hidden_size"]
    batch_size = arch_config["batch_size"]
    seq_len = arch_config["seq_len"]
    memory_config = getattr(ttnn, arch_config["memory_config"])

    logger.info(f"Architecture: {arch_config['name']}")
    logger.info(f"Topology: {topology}")
    logger.info(f"Experts: {num_experts} total, {experts_per_device} per device")
    logger.info(f"Hidden size: {hidden_size}, Batch: {batch_size}, Seq: {seq_len}")

    # Create input tensors
    torch_x = torch.randn(1, 1, batch_size * seq_len, hidden_size, dtype=torch.bfloat16)
    torch_indices = torch.randint(0, num_experts, (batch_size * seq_len, 1, experts_per_tok), dtype=torch.int32)

    tt_x = ttnn.from_torch(
        torch_x,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_indices = ttnn.from_torch(
        torch_indices,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Input x shape: {tt_x.shape}")
    logger.info(f"Input indices shape: {tt_indices.shape}")

    # Convert to ROW_MAJOR and reshape as in _forward_moe
    x_rm = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    x_rm = ttnn.reshape(x_rm, shape=(batch_size * seq_len, 1, 1, hidden_size))

    indices_rm = ttnn.to_layout(tt_indices, ttnn.ROW_MAJOR_LAYOUT)
    indices_rm = ttnn.reshape(indices_rm, shape=(batch_size * seq_len, 1, 1, experts_per_tok))

    logger.info(f"Reshaped x_rm shape: {x_rm.shape}")
    logger.info(f"Reshaped indices_rm shape: {indices_rm.shape}")

    # Create expert mapping tensors
    num_devices = mesh_device.get_num_devices()
    assert experts_per_device == num_experts // num_devices, f"Mismatch in experts per device calculation"

    expert_mapping_tensors = ttnn.from_torch(
        torch.eye(num_devices, dtype=torch.int32)
        .repeat_interleave(experts_per_device, dim=0)
        .unsqueeze(0)
        .unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    try:
        # Test dispatch
        logger.info(f"Testing all_to_all_dispatch with {topology} topology...")
        dispatch_output, dispatch_metadata = AllToAllDispatcher.dispatch(
            x_rm,
            indices_rm,
            expert_mapping_tensors,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            topology=topology,
        )
        logger.info(f"Dispatch output shape: {dispatch_output.shape}")
        logger.info(f"Dispatch metadata shape: {dispatch_metadata.shape}")

        # Simulate expert processing (just pass through for this test)
        # For combine to work, the expert output needs to have the correct shape
        # Reshape to have experts_per_device in the first dimension
        dispatch_shape = dispatch_output.shape
        # The dispatch output is [1, total_tokens_across_experts, 1, hidden_size]
        # We need to reshape to [experts_per_device, 1, tokens_per_expert, hidden_size]
        total_tokens_across_experts = dispatch_shape[1]
        tokens_per_expert = total_tokens_across_experts // experts_per_device
        expert_output = ttnn.reshape(dispatch_output, (experts_per_device, 1, tokens_per_expert, hidden_size))
        logger.info(f"Expert output shape after reshape: {expert_output.shape}")

        # Test combine
        logger.info(f"Testing all_to_all_combine with {topology} topology...")
        if apply_all_reduce:
            logger.info(f"Will apply all-reduce on axis {all_reduce_axis}")

        combined_output = AllToAllCombiner.combine(
            expert_output,
            dispatch_metadata,
            expert_mapping_tensors,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            topology=topology,
            apply_all_reduce=apply_all_reduce,
            all_reduce_axis=all_reduce_axis,
        )
        logger.info(f"Combined output shape: {combined_output.shape}")

        # Cleanup
        ttnn.deallocate(expert_output)
        ttnn.deallocate(dispatch_output)
        ttnn.deallocate(dispatch_metadata)
        ttnn.deallocate(combined_output)
        ttnn.deallocate(expert_mapping_tensors)
        logger.info(f"✅ All-to-all dispatch and combine test passed for {arch_config['name']}")

    except Exception as e:
        logger.error(f"❌ All-to-all test FAILED for {arch_config['name']}: {e}")
        ttnn.deallocate(expert_mapping_tensors)
        raise
    finally:
        # Cleanup inputs
        ttnn.deallocate(x_rm)
        ttnn.deallocate(indices_rm)
        ttnn.deallocate(tt_x)
        ttnn.deallocate(tt_indices)


def test_05_reduce_scatter(mesh_device, ccl):
    """Test 5: Reduce-scatter operation at the end of forward pass."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: REDUCE-SCATTER OPERATION")
    logger.info("=" * 60)

    # Create input tensor (after MoE processing)
    torch_input = torch.randn(1, 1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Input shape: {tt_input.shape}")

    # Test reduce-scatter if tensor parallel is enabled
    tp_axis = 1  # From deepseek_v3.json config
    tp_enabled = True

    if tp_enabled:
        logger.info(f"Testing reduce-scatter on axis {tp_axis}")

        # Get CCL parameters for reduce_scatter
        reduce_scatter_config = {
            "cluster_axis": tp_axis,
            "dim": 3,  # Last dimension after batching
            "memory_config": ttnn.L1_MEMORY_CONFIG,
            "topology": ttnn.Topology.Linear,
        }

        # Use CCL to populate runtime arguments (semaphores)
        reduce_scatter_args = ccl.populate_reduce_scatter_runtime_args(reduce_scatter_config)

        logger.info("Calling reduce_scatter_minimal_async...")
        output = ttnn.experimental.reduce_scatter_minimal_async(tt_input, **reduce_scatter_args)
        logger.info(f"Reduce-scatter output shape: {output.shape}")

        # Cleanup
        ttnn.deallocate(output)

    ttnn.deallocate(tt_input)
    logger.info("✅ Reduce-scatter test passed")


def test_06_distributed_expert_with_reference_comparison(mesh_device):
    """Test 6: Distributed Expert with comparison against reference implementation."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: DISTRIBUTED EXPERT WITH REFERENCE COMPARISON")
    logger.info("=" * 60)

    # Import our new distributed expert
    from components.experts.distributed_expert import DistributedExpert as TTExperts

    from models.common.utility_functions import comp_pcc
    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP as ReferenceExpert

    # Import utilities for comparison
    from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict

    # Load configuration from JSON file
    config_path = Path(__file__).parent.parent / "configs" / "deepseek_v3.json"
    with open(config_path, "r") as f:
        config_json = json.load(f)["moe_block"]

    # Extract model parameters from simplified config
    model_params = config_json["model_params"]

    # Configuration based on DeepSeek-V3 from JSON file
    class MockConfig:
        n_routed_experts = model_params["num_experts"]
        hidden_size = model_params["hidden_size"]
        moe_intermediate_size = model_params["intermediate_size"]
        quantization_config = {"weight_block_size": [128, 128]}  # Default for DeepSeek-V3
        hidden_act = "silu"  # Maps to swiglu in the config

    hf_config = MockConfig()
    seq_len = 128

    # Calculate expected experts per device
    num_experts_per_device = hf_config.n_routed_experts // mesh_device.get_num_devices()
    logger.info(f"Total experts: {hf_config.n_routed_experts}")
    logger.info(f"Devices: {mesh_device.get_num_devices()}")
    logger.info(f"Experts per device: {num_experts_per_device}")

    # Create all experts reference model
    class AllExpertsReference(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.experts = torch.nn.ModuleList(
                [
                    ReferenceExpert(config, intermediate_size=config.moe_intermediate_size).eval()
                    for _ in range(config.n_routed_experts)
                ]
            )

        def forward(self, hidden_states):
            outputs = []
            for expert in self.experts:
                outputs.append(expert(hidden_states))
            return torch.cat(outputs, dim=0)

    # Create reference model and generate random weights
    reference_model = AllExpertsReference(hf_config).eval().to(torch.bfloat16)
    torch_input = torch.randn(1, 1, seq_len, hf_config.hidden_size, dtype=torch.bfloat16)

    # Create state dict with random weights
    state_dict = add_inv_scale_to_state_dict(
        reference_model.state_dict(), block_shape=hf_config.quantization_config["weight_block_size"]
    )

    # Convert weights using our DistributedExpert
    logger.info("Converting weights...")
    weight_config = TTExperts.convert_weights(
        hf_config,
        (state_dict,),
        Path("/tmp"),
        mesh_device,
    )

    # Create model config
    model_config = TTExperts.decode_model_config(hf_config, mesh_device)

    # Merge weight config into model config (weights are passed via config)
    for key, value in weight_config.items():
        if key in model_config:
            model_config[key].update(value)
        else:
            model_config[key] = value

    # Debug: Verify weights are in config
    logger.info(f"w1_experts keys in config: {model_config.get('w1_experts', {}).keys()}")
    if "input_tensor_b" in model_config.get("w1_experts", {}):
        w1_weight = model_config["w1_experts"]["input_tensor_b"]
        logger.info(f"w1_experts weight shape: {w1_weight.shape}")
    else:
        logger.error("w1_experts missing input_tensor_b!")

    # Get memory config from JSON (default to L1 for distributed experts)
    # In the simplified config, distributed is just a boolean, not a dict with config details

    # Create TTNN input (repeat for each expert on device)
    tt_input = ttnn.from_torch(
        torch_input.repeat(1, num_experts_per_device, 1, 1),  # repeat activations per expert
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Move to correct memory config from JSON
    tt_input = ttnn.to_memory_config(tt_input, model_config["input_memory_config"])

    # Run forward pass
    logger.info("Running TTNN forward pass...")
    # TODO: This test needs to be updated to provide routing information (indices, weights, expert_mapping)
    # as required by DistributedExpert.forward_decode(). For now, skip the forward pass.
    pytest.skip("Test needs to be updated to provide routing information for DistributedExpert.forward_decode")

    # The following code is kept for reference when the test is fixed:
    # tt_output = TTExperts.forward_decode(tt_input, tt_indices, tt_weights, model_config, expert_mapping, mesh_device)

    # Get reference output
    logger.info("Computing reference output...")
    reference_output = reference_model(torch_input[0])  # Remove batch dimension for reference

    # Convert TTNN output to torch for comparison
    # tt_output_torch = ttnn.to_torch(
    #     tt_output,
    #     mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    # )

    # Reshape output to match reference
    # tt_output_torch is [mesh_rows*batch, mesh_cols*num_experts_per_device, seq_len, hidden_size]
    # = [4*1, 8*8, 128, 7168] = [4, 64, 128, 7168]
    # We need to reshape to [1, 256, seq_len, hidden_size]
    batch = 1
    tt_output_torch = tt_output_torch.reshape(batch, -1, seq_len, hf_config.hidden_size)

    # tt_output_torch is now [1, 256, 128, 7168]
    # reference_output is [256, 128, 7168], need to add batch dim
    if len(reference_output.shape) == 3:
        reference_output = reference_output.unsqueeze(0)  # Now [1, 256, 128, 7168]

    logger.info(f"Reference output shape: {reference_output.shape}")
    logger.info(f"TTNN output shape: {tt_output_torch.shape}")

    # Debug: Check a few values to understand the mismatch
    logger.info(
        f"Reference output stats - min: {reference_output.min():.6f}, max: {reference_output.max():.6f}, mean: {reference_output.mean():.6f}, std: {reference_output.std():.6f}"
    )
    logger.info(
        f"TTNN output stats - min: {tt_output_torch.min():.6f}, max: {tt_output_torch.max():.6f}, mean: {tt_output_torch.mean():.6f}, std: {tt_output_torch.std():.6f}"
    )

    # Check first expert's output
    logger.info(f"First expert reference output sample: {reference_output[0, 0, 0, :5]}")
    logger.info(f"First expert TTNN output sample: {tt_output_torch[0, 0, 0, :5]}")

    # Compare outputs
    passed, pcc = comp_pcc(tt_output_torch, reference_output, pcc=0.98)

    if not passed:
        logger.error(f"PCC check failed: {pcc:.6f} < 0.98")
    else:
        logger.info(f"✅ Test passed! PCC: {pcc:.6f}")

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    assert passed, f"PCC check failed! PCC: {pcc:.6f} < 0.98"


def test_07_shared_expert_with_reference_comparison(mesh_device):
    """Test 7: SharedExpert with real weights against reference implementation."""
    logger.info("\n============================================================")
    logger.info("TEST 7: SHARED EXPERT WITH REFERENCE COMPARISON")
    logger.info("============================================================")

    # Load configuration from JSON file
    config_path = Path(__file__).parent.parent / "configs" / "deepseek_v3.json"
    with open(config_path, "r") as f:
        config_json = json.load(f)["moe_block"]

    # In simplified config, shared is just a boolean, construct the config from other fields
    has_shared_expert = config_json["experts"]["shared"]
    if not has_shared_expert:
        pytest.skip("Shared expert not enabled in config")

    # Load HuggingFace model config and weights
    model_path = os.getenv("DEEPSEEK_V3_HF_MODEL")
    if not model_path or not os.path.exists(model_path):
        pytest.skip("DEEPSEEK_V3_HF_MODEL not set or path doesn't exist")

    from transformers import AutoConfig

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP
    from models.demos.deepseek_v3.utils.test_utils import load_state_dict

    hf_config = AutoConfig.from_pretrained(model_path)

    # Create shared_config from model_params and experts config
    # DeepSeek shared experts use moe_intermediate_size (1408) which is different from intermediate_size (2048)
    shared_config = {
        "hidden_size": config_json["model_params"]["hidden_size"],
        "intermediate_size": 1408,  # DeepSeek shared expert specific size
        "weight_block_size": config_json["experts"].get("weight_block_size", [128, 128]),
        "memory_config": "L1_MEMORY_CONFIG",  # Default for shared experts
    }

    # Validate that our constructed config matches HF config
    assert (
        shared_config["hidden_size"] == hf_config.hidden_size
    ), f"JSON hidden_size {shared_config['hidden_size']} != HF {hf_config.hidden_size}"
    assert (
        shared_config["weight_block_size"] == hf_config.quantization_config["weight_block_size"]
    ), f"JSON weight_block_size {shared_config['weight_block_size']} != HF {hf_config.quantization_config['weight_block_size']}"

    # Note: shared expert has its own intermediate_size (1408) different from moe_intermediate_size (2048)
    # This is intentional - shared experts are smaller than distributed experts
    logger.info(f"Shared expert intermediate_size: {shared_config['intermediate_size']}")
    logger.info(f"MoE intermediate_size from HF: {hf_config.moe_intermediate_size}")
    logger.info("✅ Config validated successfully")

    # Load only the specific layer weights we need (to avoid incomplete safetensors issue)
    logger.info("Loading weights for shared expert from layer 3...")
    try:
        # Load just the specific file containing layer 3
        state_dict = load_state_dict(Path(model_path), "model.layers.3.mlp.shared_experts")
    except Exception as e:
        logger.warning(f"Could not load real weights: {e}")
        logger.info("Using random weights for testing...")

        # Create mock weights with quantization like test_06
        from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict

        # Create reference model and generate random weights (use shared expert intermediate_size)
        reference_model = DeepseekV3MLP(hf_config, intermediate_size=shared_config["intermediate_size"]).eval()
        state_dict = add_inv_scale_to_state_dict(
            reference_model.state_dict(), block_shape=hf_config.quantization_config["weight_block_size"]
        )
        # Add the shared_experts prefix
        sub_dict = {}
        for key, value in state_dict.items():
            sub_dict[key] = value
    else:
        # Extract shared expert weights from loaded state dict
        module_path = "model.layers.3.mlp.shared_experts"
        sub_dict = {}
        prefix = module_path + "."
        for key, value in state_dict.items():
            if key.startswith(prefix):
                new_key = key[len(prefix) :]
                sub_dict[new_key] = value

        if not sub_dict:
            logger.warning(f"No shared expert weights found at {module_path}, using random weights")
            # Create mock weights with quantization
            from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict

            reference_model = DeepseekV3MLP(hf_config, intermediate_size=shared_config["intermediate_size"]).eval()
            sub_dict = add_inv_scale_to_state_dict(
                reference_model.state_dict(), block_shape=hf_config.quantization_config["weight_block_size"]
            )

    logger.info(f"Hidden size: {hf_config.hidden_size}")
    logger.info(f"Shared expert intermediate size: {shared_config['intermediate_size']}")
    logger.info(f"Distributed expert intermediate size (HF moe_intermediate_size): {hf_config.moe_intermediate_size}")

    # Convert weights using our SharedExpert
    from components.experts.shared_expert import SharedExpert

    logger.info("Converting weights...")
    weight_config = SharedExpert.convert_weights(hf_config, (sub_dict,), Path("/tmp/shared_expert_test"), mesh_device)

    # Verify weight config structure
    assert "w1" in weight_config, "w1 weights missing"
    assert "w2" in weight_config, "w2 weights missing"
    assert "w3" in weight_config, "w3 weights missing"
    assert "input_tensor_b" in weight_config["w1"], "w1 input_tensor_b missing"
    assert "input_tensor_b" in weight_config["w2"], "w2 input_tensor_b missing"
    assert "input_tensor_b" in weight_config["w3"], "w3 input_tensor_b missing"

    # Create model configs
    decode_config = SharedExpert.decode_model_config(hf_config, mesh_device)

    # Merge weight config into decode config
    for key in ["w1", "w2", "w3"]:
        decode_config[key].update(weight_config[key])

    # Create test input (decode mode with 32 users)
    batch_size = 32
    seq_len = 1
    hidden_size = hf_config.hidden_size

    # Get memory config from JSON
    memory_config_str = shared_config.get("memory_config", "L1_MEMORY_CONFIG")
    initial_memory_config = getattr(ttnn, memory_config_str)

    # Create input tensor
    torch_input = torch.randn(1, seq_len, batch_size, hidden_size, dtype=torch.bfloat16)

    # Convert to TTNN using memory config from JSON
    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=initial_memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info("Running TTNN forward pass...")
    tt_output = SharedExpert.forward_decode(tt_input, decode_config)

    logger.info("Computing reference output...")
    # Create reference model with shared expert's intermediate_size from JSON config
    # Note: Shared experts use a different intermediate_size (1408) than distributed experts (2048)
    reference_model = DeepseekV3MLP(hf_config, intermediate_size=shared_config["intermediate_size"]).eval()

    # Manually set the weights from the state dict
    with torch.no_grad():
        # Dequantize weights if needed
        from models.demos.deepseek_v3.utils.config_helpers import dequantize

        gate_weight = sub_dict["gate_proj.weight"]
        down_weight = sub_dict["down_proj.weight"]
        up_weight = sub_dict["up_proj.weight"]

        # Check if weights are quantized
        if "gate_proj.weight_scale_inv" in sub_dict:
            gate_weight = dequantize(
                gate_weight, sub_dict["gate_proj.weight_scale_inv"], hf_config.quantization_config["weight_block_size"]
            )
            down_weight = dequantize(
                down_weight, sub_dict["down_proj.weight_scale_inv"], hf_config.quantization_config["weight_block_size"]
            )
            up_weight = dequantize(
                up_weight, sub_dict["up_proj.weight_scale_inv"], hf_config.quantization_config["weight_block_size"]
            )

        # Set the weights in the reference model
        reference_model.gate_proj.weight = torch.nn.Parameter(gate_weight.to(torch.float32))
        reference_model.down_proj.weight = torch.nn.Parameter(down_weight.to(torch.float32))
        reference_model.up_proj.weight = torch.nn.Parameter(up_weight.to(torch.float32))

    # Run reference forward pass
    reference_input = torch_input.squeeze(1).to(torch.float32)  # Remove seq_len=1 dimension for reference
    reference_output = reference_model(reference_input)
    reference_output = reference_output.unsqueeze(1)  # Add back seq_len dimension

    # Convert TTNN output to torch
    # SharedExpert uses ReplicateTensorToMesh, so all devices have the same output
    # We just need to extract from one device
    tt_output_full = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, -1), mesh_shape=mesh_device.shape),
    )
    logger.info(f"Full TTNN output shape after mesh composer: {tt_output_full.shape}")
    tt_output_torch = (
        tt_output_full[0].unsqueeze(0).to(torch.bfloat16)
    )  # Take first device output and restore batch dim

    # Compare shapes
    logger.info(f"Reference output shape: {reference_output.shape}")
    logger.info(f"TTNN output shape: {tt_output_torch.shape}")

    # Ensure shapes match exactly
    if tt_output_torch.shape != reference_output.shape:
        logger.warning(
            f"Shape mismatch, attempting to reshape TTNN output from {tt_output_torch.shape} to {reference_output.shape}"
        )
        # The TTNN output might have extra padding, extract the relevant part
        tt_output_torch = tt_output_torch[
            : reference_output.shape[0],
            : reference_output.shape[1],
            : reference_output.shape[2],
            : reference_output.shape[3],
        ]

    # Check PCC
    from models.common.utility_functions import comp_pcc

    ref_bf16 = reference_output.to(torch.bfloat16)
    passed, pcc = comp_pcc(ref_bf16, tt_output_torch, 0.98)

    # Log statistics for debugging
    logger.info(
        f"Reference output stats - min: {ref_bf16.min():.6f}, max: {ref_bf16.max():.6f}, mean: {ref_bf16.mean():.6f}, std: {ref_bf16.std():.6f}"
    )
    logger.info(
        f"TTNN output stats - min: {tt_output_torch.min():.6f}, max: {tt_output_torch.max():.6f}, mean: {tt_output_torch.mean():.6f}, std: {tt_output_torch.std():.6f}"
    )

    # Sample outputs for debugging
    logger.info(f"First sample reference output: {ref_bf16[0, 0, 0, :5]}")
    logger.info(f"First sample TTNN output: {tt_output_torch[0, 0, 0, :5]}")

    if passed:
        logger.info(f"✅ Test passed! PCC: {pcc:.6f}")
    else:
        logger.error(f"❌ Test failed! PCC: {pcc:.6f} < 0.98")

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    assert passed, f"PCC check failed! PCC: {pcc:.6f} < 0.98"


def test_08_gpt_oss_clamped_swiglu(mesh_device):
    """Test GPT-OSS clamped SwiGLU activation function."""
    logger.info("=" * 60)
    logger.info("Test 08: GPT-OSS Clamped SwiGLU Activation")
    logger.info("=" * 60)

    from components.experts.distributed_expert import DistributedExpert

    # Create tensors for testing the activation function
    intermediate_size = 32

    # Create test inputs with values that will be clamped
    torch.manual_seed(42)
    gate_input = torch.randn(1, 1, 16, intermediate_size) * 10  # Some values > 7.0
    up_input = torch.randn(1, 1, 16, intermediate_size) * 10  # Some values outside [-7.0, 7.0]

    # Convert to TTNN tensors and put on device
    # Use the mesh device with replication
    gate_tt = ttnn.from_torch(
        gate_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    up_tt = ttnn.from_torch(
        up_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    # Apply clamped SwiGLU activation
    activated = DistributedExpert._apply_clamped_swiglu(
        gate=gate_tt, up=up_tt, alpha=1.702, limit=7.0, memory_config=ttnn.L1_MEMORY_CONFIG
    )

    # Bring back to CPU
    activated = ttnn.from_device(activated)

    # Convert back to torch (handle mesh device)
    activated_torch = ttnn.to_torch(activated, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))

    # Compute reference implementation
    gate_clamped = torch.clamp(gate_input, max=7.0)
    up_clamped = torch.clamp(up_input, min=-7.0, max=7.0)
    gate_sigmoid = torch.sigmoid(gate_clamped * 1.702)
    reference = (up_clamped + 1.0) * (gate_clamped * gate_sigmoid)

    # Check that values match (allow for some precision loss)
    diff = torch.abs(activated_torch - reference)
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    logger.info(f"Max difference: {max_diff}")
    logger.info(f"Mean difference: {mean_diff}")

    # Check that clamping worked (relaxed tolerances for bfloat16 precision)
    assert activated_torch.max().item() < 150, "Output values should be bounded by clamping"
    assert max_diff < 0.5, f"Max difference {max_diff} exceeds tolerance for bfloat16"
    assert mean_diff < 0.05, f"Mean difference {mean_diff} exceeds tolerance for bfloat16"

    logger.info("✅ GPT-OSS clamped SwiGLU activation test passed!")

    # Create test input (batch=1, seq=32, hidden=2880)
    batch_size = 1
    seq_len = 32
    hidden_size = 2880
    num_experts = 128
    num_experts_per_tok = 4

    # ========================================
    # Step 1: Create mock weights for GPT-OSS
    # ========================================
    logger.info("Creating mock GPT-OSS weights...")
    state_dict = {}

    # Option 1: Create fused gate_up projection weights (GPT-OSS style)
    # Shape: [num_experts, hidden_size, 2 * intermediate_size]
    gate_up_weight = torch.randn(128, 2880, 2 * 2880, dtype=torch.bfloat16)
    state_dict["gate_up_proj"] = gate_up_weight

    # Down projection weights
    # Shape: [num_experts, intermediate_size, hidden_size]
    down_weight = torch.randn(128, 2880, 2880, dtype=torch.bfloat16)
    state_dict["down_proj"] = down_weight

    # Optional: Add biases (usually zeros)
    state_dict["gate_up_proj_bias"] = torch.zeros(128, 2 * 2880, dtype=torch.bfloat16)
    state_dict["down_proj_bias"] = torch.zeros(128, 2880, dtype=torch.bfloat16)

    # ========================================
    # Step 2: Convert weights using DistributedExpert
    # ========================================
    logger.info("Converting weights using DistributedExpert.convert_weights...")
    from pathlib import Path

    # Update config for DistributedExpert
    expert_config = {
        "num_experts": 128,
        "num_experts_per_device": 4,  # 128 experts / 32 devices
        "tp_size": 1,  # No tensor parallelism in test
        "ep_size": 32,  # Expert parallel across all devices
        "hidden_size": 2880,
        "intermediate_size": 2880,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "swiglu_alpha": 1.702,
        "swiglu_limit": 7.0,
        "memory_config": "L1_MEMORY_CONFIG",
        "output_memory_config": "L1_MEMORY_CONFIG",
    }

    converted_weights = DistributedExpert.convert_weights(
        expert_config, [state_dict], Path("/tmp"), mesh_device  # List of state dicts  # Weight cache dir
    )

    logger.info(f"Converted weights: TTNNDistributedExpertWeights object")

    # ========================================
    # Step 3: Create test data
    # ========================================
    # Create random input
    torch.manual_seed(42)
    torch_input = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Create random indices and weights
    indices = torch.randint(0, num_experts, (batch_size, seq_len, num_experts_per_tok))
    weights = torch.rand(batch_size, seq_len, num_experts_per_tok, dtype=torch.bfloat16)
    weights = weights / weights.sum(dim=-1, keepdim=True)  # Normalize weights

    # Reshape weights for ThroughputExpert: [batch, seq, K] -> [K, 1, batch*seq, 1]
    # This allows broadcasting across hidden_size dimension
    tokens_per_device = batch_size * seq_len
    weights_reshaped = weights.permute(2, 0, 1).reshape(num_experts_per_tok, 1, tokens_per_device, 1)
    # Expand to hidden size for proper broadcasting
    weights_reshaped = weights_reshaped.expand(-1, -1, -1, hidden_size)

    # ========================================
    # Step 4: Convert to TTNN tensors
    # ========================================
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_indices = ttnn.from_torch(
        indices.unsqueeze(0).to(torch.int16),  # Convert to int16 for uint16 dtype
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,  # Changed from uint32 to uint16 as required by all_to_all_dispatch
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    tt_weights = ttnn.from_torch(
        weights_reshaped,  # Use reshaped weights
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create expert mapping
    num_devices = mesh_device.get_num_devices()
    experts_per_device = 4  # 128 experts / 32 devices
    expert_mapping = ttnn.from_torch(
        torch.eye(num_devices, dtype=torch.int16)  # Changed to int16 for uint16
        .repeat_interleave(experts_per_device, dim=0)
        .unsqueeze(0)
        .unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,  # Changed from uint32 to uint16 as required
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # ========================================
    # Step 5: Create decode config with weights
    # ========================================
    # Update expert_config with runtime parameters
    expert_config["weights"] = converted_weights
    expert_config["cluster_axis"] = 0
    expert_config["dispatch_topology"] = "Linear"  # Use Linear to avoid ring issues
    expert_config["combine_topology"] = "Linear"
    expert_config["num_experts"] = num_experts
    expert_config["num_experts_per_tok"] = num_experts_per_tok

    # ========================================
    # Step 6: Run forward pass
    # ========================================
    logger.info(f"Running DistributedExpert forward_decode...")
    try:
        logger.info("Calling forward_decode...")
        tt_output = DistributedExpert.forward_decode(
            tt_input,  # hidden_states
            tt_indices,  # topk_expert_indices
            tt_weights,  # topk_expert_weights
            expert_config,  # config (includes all parameters and weights)
            expert_mapping,  # expert_mapping_tensors
            mesh_device,  # mesh_device
        )
        logger.info(f"forward_decode completed, output shape: {tt_output.shape}")

        # Convert to torch for verification
        # For mesh tensors, we may need to get from a specific device
        logger.info("Converting output to torch...")
        try:
            # Try direct conversion first
            output_torch = ttnn.to_torch(tt_output)
        except Exception:
            # If that fails, try getting from the first device
            logger.info("Direct conversion failed, trying to get from first device...")
            output_torch = ttnn.to_torch(tt_output, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
        logger.info(f"Output torch shape: {output_torch.shape}")
        logger.info(f"Output sample values: {output_torch[0, 0, :5, 0]}")

        # Basic validation
        assert (
            tt_output.shape[-1] == hidden_size
        ), f"Output hidden size mismatch: {tt_output.shape[-1]} != {hidden_size}"
        assert not torch.isnan(output_torch).any(), "Output contains NaN values"
        assert not torch.isinf(output_torch).any(), "Output contains Inf values"

        logger.info("✅ DistributedExpert forward_decode completed successfully")
    except Exception as e:
        logger.error(f"❌ DistributedExpert forward_decode failed: {e}")
        raise

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_indices)
    ttnn.deallocate(tt_weights)
    ttnn.deallocate(expert_mapping)
    ttnn.deallocate(tt_output)

    # Cleanup converted weights (TTNNDistributedExpertWeights object)
    if hasattr(converted_weights, "gate_up_proj_weights"):
        ttnn.deallocate(converted_weights.gate_up_proj_weights)
    if hasattr(converted_weights, "down_proj_weights"):
        ttnn.deallocate(converted_weights.down_proj_weights)
    if hasattr(converted_weights, "gate_up_proj_biases"):
        ttnn.deallocate(converted_weights.gate_up_proj_biases)
    if hasattr(converted_weights, "down_proj_biases"):
        ttnn.deallocate(converted_weights.down_proj_biases)


@pytest.mark.parametrize(
    "arch_config",
    [
        {
            "name": "deepseek",
            "router_type": "grouped_topk",
            "num_experts": 256,
            "experts_per_tok": 8,
            "hidden_size": 7168,
            "batch_size": 32,
            "seq_len": 1,
            "memory_config": "L1_MEMORY_CONFIG",
            "score_correction_bias": True,
        },
        {
            "name": "gpt_oss",
            "router_type": "topk",
            "num_experts": 128,
            "experts_per_tok": 4,
            "hidden_size": 2880,
            "batch_size": 1,
            "seq_len": 32,
            "memory_config": "DRAM_MEMORY_CONFIG",
            "score_correction_bias": False,
        },
    ],
)
def test_09_routers_comparison(mesh_device, arch_config):
    """Test 9: Compare routers for both DeepSeek and GPT-OSS architectures."""
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST 9: ROUTER COMPARISON - {arch_config['name'].upper()}")
    logger.info("=" * 60)

    # Extract parameters
    router_type = arch_config["router_type"]
    num_experts = arch_config["num_experts"]
    experts_per_tok = arch_config["experts_per_tok"]
    hidden_size = arch_config["hidden_size"]
    batch_size = arch_config["batch_size"]
    seq_len = arch_config["seq_len"]
    memory_config_str = arch_config["memory_config"]
    memory_config = getattr(ttnn, memory_config_str)
    score_correction_bias = arch_config["score_correction_bias"]

    logger.info(f"Router type: {router_type}")
    logger.info(f"Experts: {num_experts} total, {experts_per_tok} per token")
    logger.info(f"Input: batch={batch_size}, seq={seq_len}, hidden={hidden_size}")

    # Import the appropriate router
    if router_type == "grouped_topk":
        from components.routers.grouped_topk_router import GroupedTopKRouter

        router_config = {
            "hidden_size": hidden_size,
            "n_routed_experts": num_experts,
            "num_experts_per_tok": experts_per_tok,
            "num_experts": num_experts,
            "routed_scaling_factor": 1.0,
            "n_group": 1,
            "topk_group": 1,
            "score_correction_bias": score_correction_bias,
            "memory_config": memory_config_str,
        }
        router = GroupedTopKRouter(router_config, mesh_device)
    elif router_type == "topk":
        # Check if TopKRouter exists
        try:
            from components.routers.topk_router import TopKRouter
        except ImportError:
            pytest.skip(f"TopKRouter not implemented yet for {arch_config['name']}")

        router_config = {
            "hidden_size": hidden_size,
            "num_experts": num_experts,
            "num_experts_per_tok": experts_per_tok,
            "memory_config": memory_config_str,
        }
        router = TopKRouter(mesh_device, router_config)  # Fixed order: mesh_device first, then config
    else:
        raise ValueError(f"Unknown router type: {router_type}")

    # Load mock weights
    mock_weights = {
        "weight": torch.randn(num_experts, hidden_size, dtype=torch.bfloat16),
    }
    if router_type == "topk":
        # TopKRouter needs bias as well
        mock_weights["bias"] = torch.zeros(num_experts, dtype=torch.bfloat16)
    elif score_correction_bias and router_type == "grouped_topk":
        mock_weights["e_score_correction_bias"] = torch.zeros(num_experts, dtype=torch.bfloat16)

    router.load_weights(mock_weights)

    # Create input tensor
    torch_input = torch.randn(1, seq_len, batch_size, hidden_size, dtype=torch.bfloat16)

    tt_input = ttnn.from_torch(
        torch_input,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Input shape: {tt_input.shape}")

    # Test router forward
    logger.info(f"Calling {router_type} router.forward...")
    weights, indices = router.forward(tt_input, mode="decode")

    logger.info(f"Router weights shape: {weights.shape}")
    logger.info(f"Router indices shape: {indices.shape}")

    # Verify output shapes
    # Router outputs: weights and indices both in shape [1, 1, batch_size * seq_len, experts_per_tok]
    expected_weights_shape = (1, 1, batch_size * seq_len, experts_per_tok)
    expected_indices_shape = (1, 1, batch_size * seq_len, experts_per_tok)

    assert weights.shape == ttnn.Shape(
        expected_weights_shape
    ), f"Weights shape mismatch: {weights.shape} != {expected_weights_shape}"
    assert indices.shape == ttnn.Shape(
        expected_indices_shape
    ), f"Indices shape mismatch: {indices.shape} != {expected_indices_shape}"

    logger.info(f"✅ {router_type} router test passed for {arch_config['name']}")

    # Cleanup
    ttnn.deallocate(weights)
    ttnn.deallocate(indices)
    ttnn.deallocate(tt_input)


if __name__ == "__main__":
    # Run tests individually for debugging
    import sys

    # Set up environment
    os.environ["PYTHONPATH"] = os.getcwd()
    os.environ["TT_METAL_HOME"] = os.getcwd()
    os.environ["MESH_DEVICE"] = os.getenv("MESH_DEVICE", "TG")
    os.environ["DEEPSEEK_V3_HF_MODEL"] = os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52",
    )

    # Run with pytest
    pytest.main([__file__, "-v", "-s"])

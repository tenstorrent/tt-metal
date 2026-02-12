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

from components.routers.moe_gate import MoEGateRouter
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
    router = MoEGateRouter(router_config, mesh_device)

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


def test_04_forward_moe_dispatch_only(mesh_device):
    """Test 4: Forward MoE - test only the all_to_all_dispatch part."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: FORWARD MoE - ALL_TO_ALL_DISPATCH")
    logger.info("=" * 60)

    # Create input tensors
    torch_x = torch.randn(1, 1, BATCH_SIZE, HIDDEN_SIZE, dtype=torch.bfloat16)
    torch_indices = torch.randint(0, 256, (BATCH_SIZE, 1, NUM_EXPERTS_PER_TOK), dtype=torch.int32)

    tt_x = ttnn.from_torch(
        torch_x,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    tt_indices = ttnn.from_torch(
        torch_indices,
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    logger.info(f"Input x shape: {tt_x.shape}")
    logger.info(f"Input indices shape: {tt_indices.shape}")

    # Convert to ROW_MAJOR and reshape as in _forward_moe
    x_rm = ttnn.to_layout(tt_x, ttnn.ROW_MAJOR_LAYOUT)
    x_rm = ttnn.reshape(x_rm, shape=(BATCH_SIZE, 1, 1, HIDDEN_SIZE))

    indices_rm = ttnn.to_layout(tt_indices, ttnn.ROW_MAJOR_LAYOUT)
    indices_rm = ttnn.reshape(indices_rm, shape=(BATCH_SIZE, 1, 1, NUM_EXPERTS_PER_TOK))

    logger.info(f"Reshaped x_rm shape: {x_rm.shape}")
    logger.info(f"Reshaped indices_rm shape: {indices_rm.shape}")

    # Create expert mapping tensors
    num_devices = mesh_device.get_num_devices()
    num_experts_per_device = 256 // num_devices  # 256 total experts

    expert_mapping_tensors = ttnn.from_torch(
        torch.eye(num_devices, dtype=torch.int32)
        .repeat_interleave(num_experts_per_device, dim=0)
        .unsqueeze(0)
        .unsqueeze(0),
        device=mesh_device,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        dtype=ttnn.uint16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )

    # Test all_to_all_dispatch
    logger.info("Calling all_to_all_dispatch...")
    ep_axis = 0  # Expert parallel axis

    try:
        dispatch_output, dispatch_metadata = ttnn.all_to_all_dispatch(
            x_rm,
            indices_rm,
            expert_mapping_tensors,
            cluster_axis=ep_axis,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
        )
        logger.info(f"Dispatch output shape: {dispatch_output.shape}")
        logger.info(f"Dispatch metadata shape: {dispatch_metadata.shape}")

        # Cleanup
        ttnn.deallocate(dispatch_output)
        ttnn.deallocate(dispatch_metadata)
        ttnn.deallocate(expert_mapping_tensors)
        logger.info("✅ All-to-all dispatch test passed")

    except Exception as e:
        logger.error(f"❌ All-to-all dispatch FAILED: {e}")
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

    # Extract distributed expert config
    distributed_config = config_json["experts"]["distributed"]

    # Configuration based on DeepSeek-V3 from JSON file
    class MockConfig:
        n_routed_experts = distributed_config["n_routed_experts"]
        hidden_size = distributed_config["hidden_size"]
        moe_intermediate_size = distributed_config["intermediate_size"]
        quantization_config = {"weight_block_size": distributed_config["weight_block_size"]}
        hidden_act = "silu"  # Maps to swiglu in the config

    hf_config = MockConfig()
    mode = "decode"
    seq_len = 128

    # Calculate expected experts per device
    num_experts_per_device = hf_config.n_routed_experts // mesh_device.get_num_devices()
    logger.info(f"Total experts: {hf_config.n_routed_experts}")
    logger.info(f"Devices: {mesh_device.get_num_devices()}")
    logger.info(f"Experts per device: {num_experts_per_device}")

    # Create reference model for a single expert
    class SingleExpertReference(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.expert = ReferenceExpert(config, intermediate_size=config.moe_intermediate_size)

        def forward(self, hidden_states):
            return self.expert(hidden_states)

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

    # Get memory config from JSON
    memory_config_str = distributed_config.get("memory_config", "L1_MEMORY_CONFIG")
    initial_memory_config = getattr(ttnn, memory_config_str)

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
    tt_output = TTExperts.forward_decode(tt_input, model_config)

    # Get reference output
    logger.info("Computing reference output...")
    reference_output = reference_model(torch_input[0])  # Remove batch dimension for reference

    # Convert TTNN output to torch for comparison
    tt_output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(0, 1), mesh_shape=tuple(mesh_device.shape)),
    )

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

    # Extract shared expert config
    shared_config = config_json["experts"]["shared"]

    # Load HuggingFace model config and weights
    model_path = os.getenv("DEEPSEEK_V3_HF_MODEL")
    if not model_path or not os.path.exists(model_path):
        pytest.skip("DEEPSEEK_V3_HF_MODEL not set or path doesn't exist")

    from transformers import AutoConfig

    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP
    from models.demos.deepseek_v3.utils.test_utils import load_state_dict

    hf_config = AutoConfig.from_pretrained(model_path)

    # Validate that JSON config has required fields and correct hidden_size
    assert (
        shared_config["hidden_size"] == hf_config.hidden_size
    ), f"JSON hidden_size {shared_config['hidden_size']} != HF {hf_config.hidden_size}"
    assert (
        shared_config["weight_block_size"] == hf_config.quantization_config["weight_block_size"]
    ), f"JSON weight_block_size {shared_config['weight_block_size']} != HF {hf_config.quantization_config['weight_block_size']}"

    # Note: shared expert has its own intermediate_size (1408) different from moe_intermediate_size (2048)
    # This is intentional - shared experts are smaller than distributed experts
    logger.info(f"Shared expert intermediate_size from JSON: {shared_config['intermediate_size']}")
    logger.info(f"MoE intermediate_size from HF: {hf_config.moe_intermediate_size}")
    logger.info("✅ JSON config has all required fields")

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
    logger.info(f"Shared expert intermediate size (from JSON): {shared_config['intermediate_size']}")
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

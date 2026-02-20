# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test DeepSeek MoE Block against DeepSeek-V3 reference implementation.
Uses PCC (Pearson Correlation Coefficient) for accuracy checking.
"""

import os
import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

# Add necessary paths
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "demos" / "deepseek_v3"))

import ttnn.graph
from moe_block import MoEBlock

# Import CCL from our utils
from utils.ccl import CCL

import ttnn

# Import reference model and test utilities from DeepSeek
try:
    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
    from models.demos.deepseek_v3.utils.test_utils import assert_hidden_dim_pcc, dequantize_state_dict, load_state_dict
except ImportError as e:
    logger.warning(f"Could not import DeepSeek utilities: {e}")
    DeepseekV3MoE = None
    CCL = None
    assert_hidden_dim_pcc = None
    dequantize_state_dict = None
    load_state_dict = None


# Test configuration constants
USERS_PER_ROW = 32
PCC_REQUIRED = 0.98  # Required PCC threshold


@pytest.fixture(scope="session")
def hf_config():
    """Load HuggingFace configuration for DeepSeek-V3."""
    from transformers import AutoConfig

    model_path = os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52",
    )

    if not Path(model_path).exists():
        pytest.skip(f"Model not found at {model_path}")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    return config


@pytest.fixture(scope="function")
def mesh_device():
    """Create and cleanup mesh device."""
    device_type = os.getenv("MESH_DEVICE", "TG").upper()

    mesh_shapes = {
        "TG": (4, 8),
        "DUAL": (8, 8),
        "QUAD": (16, 8),
        "T3K": (1, 8),
    }

    if device_type not in mesh_shapes:
        pytest.skip(f"Device type {device_type} not supported")

    mesh_shape = mesh_shapes[device_type]

    # Set fabric configuration for multi-device systems
    # This is required for all_gather and reduce_scatter operations
    fabric_config = None
    if device_type in ["TG", "DUAL", "QUAD"]:
        # Enable fabric for CCL operations (all_gather, reduce_scatter)
        fabric_config = ttnn.FabricConfig.FABRIC_1D
        ttnn.set_fabric_config(fabric_config)

    # Open mesh device
    device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(*mesh_shape))

    logger.info(f"Created mesh device: {device_type} with shape {mesh_shape}")
    yield device

    # Close mesh device and disable fabric config
    ttnn.close_mesh_device(device)
    if fabric_config:
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


@pytest.fixture(scope="function")
def ccl(mesh_device):
    """Create CCL instance for collective operations."""
    return CCL(mesh_device)


@pytest.fixture(scope="session")
def model_path():
    """Get model path for DeepSeek-V3 weights."""
    model_path = os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52",
    )

    if not Path(model_path).exists():
        pytest.skip(f"Model not found at {model_path}")

    return Path(model_path)


@pytest.fixture(scope="session")
def state_dict(model_path):
    """Load state dict from HuggingFace model."""
    if load_state_dict is None:
        pytest.skip("load_state_dict not available")
    return load_state_dict(model_path, "")


@pytest.mark.parametrize(
    "mode,seq_len",
    [
        ("decode", 1),  # Current test configuration (batch_size=128, seq_len=1)
        ("decode", 128),  # Match reference test configuration (batch_size=1, seq_len=128)
        ("prefill", 128),  # Add prefill mode with seq_len=128
    ],
    ids=["mode_decode_seq_1", "mode_decode_seq_128", "mode_prefill_seq_128"],  # Clear test IDs
)
def test_deepseek_moe_against_reference(mesh_device, hf_config, ccl, state_dict, mode, seq_len):
    """Test DeepSeek MoE Block against reference implementation with PCC checking."""

    if DeepseekV3MoE is None:
        pytest.skip("DeepSeek reference model not available")

    if assert_hidden_dim_pcc is None:
        pytest.skip("PCC checking utilities not available")

    logger.info("\n" + "=" * 60)
    logger.info(f"Testing DeepSeek MoE Block Against Reference - Mode: {mode}, Seq Len: {seq_len}")
    logger.info("=" * 60)

    # Test parameters based on mode
    MODE = mode
    SEQ_LEN = seq_len

    # Adjust batch size based on mode and sequence length (following reference test pattern)
    # Note: DeepSeek reference tests use different batch sizes for different modes:
    # - Decode with seq_len=1: batch_size=32 per row (total 128 for 4 rows)
    # - Decode with seq_len=128: batch_size=1 (matches reference test_moe.py)
    # - Prefill: batch_size=1 per row (total 4 for 4 rows) with seq_len=128
    # This is due to memory constraints and computational efficiency considerations
    if MODE == "decode" and SEQ_LEN == 1:
        BATCH_SIZE_PER_ROW = 32  # USERS_PER_ROW
        BATCH_SIZE = BATCH_SIZE_PER_ROW * mesh_device.shape[0]  # 32 * 4 = 128 for decode mode
    elif MODE == "decode" and SEQ_LEN == 128:
        # Match reference test configuration exactly: batch_size=1 with seq_len=128
        BATCH_SIZE_PER_ROW = 1  # Set for consistency
        BATCH_SIZE = 1
    else:  # prefill mode
        BATCH_SIZE_PER_ROW = 1  # Use batch_size=1 for prefill to manage memory usage
        BATCH_SIZE = BATCH_SIZE_PER_ROW * mesh_device.shape[0]  # 1 * 4 = 4 for prefill mode

    HIDDEN_SIZE = hf_config.hidden_size

    # Select layer 3 (MoE layer) like reference test
    LAYER_IDX = 3
    MODULE_PATH = f"model.layers.{LAYER_IDX}"

    # Create reference model
    logger.info("Creating reference model...")
    torch.use_deterministic_algorithms(False)  # Required for loading real weights
    reference_model = DeepseekV3MoE(hf_config).eval().to(torch.bfloat16)

    # Load real weights from HF model
    logger.info("Loading real weights from HuggingFace model...")
    layer_state_dict = {}

    # Only load the specific weights we need for layer 3 MoE
    required_keys = []
    for i in range(256):  # 256 experts
        required_keys.extend(
            [
                f"{MODULE_PATH}.mlp.experts.{i}.gate_proj.weight",
                f"{MODULE_PATH}.mlp.experts.{i}.up_proj.weight",
                f"{MODULE_PATH}.mlp.experts.{i}.down_proj.weight",
            ]
        )
    required_keys.extend(
        [
            f"{MODULE_PATH}.mlp.gate.weight",
            f"{MODULE_PATH}.mlp.gate.e_score_correction_bias",
            f"{MODULE_PATH}.mlp.shared_experts.gate_proj.weight",
            f"{MODULE_PATH}.mlp.shared_experts.up_proj.weight",
            f"{MODULE_PATH}.mlp.shared_experts.down_proj.weight",
        ]
    )

    # Try to load each key individually to avoid iterating through all
    weights_loaded = 0
    for key in required_keys:
        try:
            if key in state_dict:
                value = state_dict[key]
                # Remove the module path prefix for reference model
                new_key = key.replace(f"{MODULE_PATH}.mlp.", "")
                layer_state_dict[new_key] = value
                weights_loaded += 1

                # Also load quantization scales if they exist
                scale_key = key + "_scale_inv"
                if scale_key in state_dict:
                    scale_value = state_dict[scale_key]
                    scale_new_key = new_key + "_scale_inv"
                    layer_state_dict[scale_new_key] = scale_value
                    weights_loaded += 1
        except Exception as e:
            logger.debug(f"Could not load weight {key}: {e}")
            continue

    if layer_state_dict:
        reference_model.load_state_dict(dequantize_state_dict(layer_state_dict, hf_config))
        logger.info(f"Loaded {weights_loaded} weights for layer {LAYER_IDX}")
    else:
        logger.warning(f"No weights found for layer {LAYER_IDX}, using random weights")

    # Generate test input
    if MODE == "prefill":
        # For prefill: (batch_size, seq_len, hidden_size)
        torch_input = torch.randn(BATCH_SIZE_PER_ROW, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
    else:
        # For decode: (batch_size, 1, hidden_size)
        torch_input = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)

    logger.info(f"Input shape: {torch_input.shape}")

    # Generate reference output
    logger.info("Generating reference output...")
    with torch.no_grad():
        reference_output = reference_model(torch_input)
    logger.info(f"Reference output shape: {reference_output.shape}")

    # Load DeepSeek MoEBlock
    # Create a temporary config without weight_path to prevent auto-loading
    config_path = str(Path(__file__).parent.parent / "configs" / "deepseek_v3.json")

    # Load the config and remove weight_path temporarily
    import json

    with open(config_path, "r") as f:
        config_data = json.load(f)

    # Remove weight_path to prevent auto-loading during init
    if "weight_path" in config_data["moe_block"]:
        del config_data["moe_block"]["weight_path"]

    # Write temporary config
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_config:
        json.dump(config_data, tmp_config)
        temp_config_path = tmp_config.name

    logger.info(f"Loading DeepSeek MoEBlock from config (without auto-loading weights)")
    logger.info(f"Mode: {MODE}, Seq Len: {SEQ_LEN}, Batch Size: {BATCH_SIZE}, Batch Size Per Row: {BATCH_SIZE_PER_ROW}")
    moe_block = MoEBlock(temp_config_path, mesh_device, ccl)

    # Clean up temp config
    Path(temp_config_path).unlink()

    # Load real weights from HF model into our MoEBlock
    logger.info("Loading weights into MoEBlock...")

    # Use the same required_keys list to load weights for MoEBlock
    moe_state_dict = {}
    weights_loaded_moe = 0
    for key in required_keys:
        try:
            if key in state_dict:
                value = state_dict[key]
                # Keep the mlp. prefix as our MoEBlock expects it
                new_key = key.replace(f"{MODULE_PATH}.", "")
                moe_state_dict[new_key] = value
                weights_loaded_moe += 1
                logger.debug(f"Mapping weight: {key} -> {new_key}, shape: {value.shape}")

                # Also load quantization scales if they exist
                scale_key = key + "_scale_inv"
                if scale_key in state_dict:
                    scale_value = state_dict[scale_key]
                    scale_new_key = new_key + "_scale_inv"
                    moe_state_dict[scale_new_key] = scale_value
                    weights_loaded_moe += 1
        except Exception as e:
            logger.debug(f"Could not load weight {key}: {e}")
            continue

    # Don't add quantization scales again if they were already loaded from the model
    # The scales are already in moe_state_dict from the loading loop above

    if moe_state_dict:
        moe_block.load_weights(moe_state_dict)
        logger.info(f"Loaded {weights_loaded_moe} weights into MoEBlock")
    else:
        logger.error("No weights found to load into MoEBlock!")

    # Convert input to TTNN format
    if MODE == "prefill":
        # For prefill: (batch, seq_len, hidden) -> (1, batch, seq_len, hidden)
        # Following the reference pattern, add batch dimension for TTNN
        torch_input_reshaped = torch_input.unsqueeze(0)  # Shape: (1, batch, seq_len, hidden)
        # Shard across devices on the sequence dimension
        tt_input = ttnn.from_torch(
            torch_input_reshaped,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        # For decode mode, we need to match the reference exactly
        # Reference creates (1, num_tokens, hidden_size) then does unsqueeze(1) -> (1, 1, num_tokens, hidden_size)
        # Our torch_input is already (BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE)
        # For seq_len=128: (1, 128, 7168) -> unsqueeze(1) -> (1, 1, 128, 7168)
        # For seq_len=1: (128, 1, 7168) -> needs different handling

        if SEQ_LEN == 128:
            # Match reference exactly: (1, 128, 7168) -> (1, 1, 128, 7168)
            torch_input_reshaped = torch_input.unsqueeze(1)  # Shape: (1, 1, 128, 7168)
        else:
            # Original logic for seq_len=1: (128, 1, 7168) -> (1, 1, 128, 7168)
            torch_input_reshaped = torch_input.permute(1, 0, 2).unsqueeze(0)  # Shape: (1, 1, 128, 7168)

        # Use ShardTensor2dMesh with dims=(-2, -1) like the reference test
        tt_input = ttnn.from_torch(
            torch_input_reshaped,
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

    logger.info("Running MoEBlock forward pass...")

    # Start graph capture to trace operations before the crash
    logger.info("Starting graph capture to trace operations...")
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)

    try:
        # Forward pass through DeepSeek MoE
        tt_output = moe_block.forward(tt_input, mode=MODE)

        # If we get here, capture the successful graph
        captured_graph = ttnn.graph.end_graph_capture()
        logger.info("Graph capture complete (test passed)!")

        # Save captured graph
        trace_dir = tempfile.mkdtemp(prefix="ttnn_infra_test_graph_")
        graph_json_path = Path(trace_dir) / "infra_test_captured_operations.json"
        with open(graph_json_path, "w") as f:
            json.dump(captured_graph, f, indent=2)
        logger.info(f"Captured graph saved to: {graph_json_path}")

    except Exception as e:
        # Capture whatever operations executed before the crash
        captured_graph = ttnn.graph.end_graph_capture()
        logger.error("Graph capture complete (test failed)!")

        # Save the partial graph
        trace_dir = tempfile.mkdtemp(prefix="ttnn_infra_test_failed_graph_")
        graph_json_path = Path(trace_dir) / "infra_test_failed_operations.json"
        with open(graph_json_path, "w") as f:
            json.dump(captured_graph, f, indent=2)
        logger.error(f"Failed test graph saved to: {graph_json_path}")
        logger.error(f"Graph contains {len(captured_graph) if captured_graph else 0} nodes before crash")

        # Re-raise the exception
        raise e

    # Convert output back to torch
    # Use ConcatMesh2dToTensor to gather sharded output like the reference test does
    tt_output_torch = ttnn.to_torch(
        tt_output, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape)
    )

    # Reshape output back to match reference format
    if MODE == "prefill":
        # Remove batch dimension added for TTNN: (1, batch, seq_len, hidden) -> (batch, seq_len, hidden)
        tt_output_torch = tt_output_torch.squeeze(0)
    else:
        # Decode mode: handle different seq_len cases
        if SEQ_LEN == 128:
            # For seq_len=128: output is [1, 1, 128, 7168], reference expects [1, 128, 7168]
            # The 128 represents sequence length, not batch, so just squeeze the second dim
            tt_output_torch = tt_output_torch.squeeze(1)  # [1, 1, 128, 7168] -> [1, 128, 7168]
        else:
            # For seq_len=1: output is [1, 1, 128, 7168], reference expects [128, 1, 7168]
            # The 128 represents batch size, so permute is needed
            tt_output_torch = tt_output_torch.squeeze(0).permute(1, 0, 2)  # [1, 1, 128, 7168] -> [128, 1, 7168]

    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)

    logger.info(f"✅ Test passed with PCC >= {PCC_REQUIRED}")

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

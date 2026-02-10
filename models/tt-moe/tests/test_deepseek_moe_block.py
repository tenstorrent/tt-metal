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

from moe_block import MoEBlock

# Import CCL from our utils
from utils.ccl import CCL

import ttnn

# Import reference model and test utilities from DeepSeek
try:
    from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
    from models.demos.deepseek_v3.utils.test_utils import add_inv_scale_to_state_dict, assert_hidden_dim_pcc
except ImportError as e:
    logger.warning(f"Could not import DeepSeek utilities: {e}")
    DeepseekV3MoE = None
    CCL = None
    assert_hidden_dim_pcc = None


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
        ttnn.set_fabric_config(
            fabric_config,
            ttnn.FabricReliabilityMode.STRICT_INIT,
            None,  # eth_buffer_size
            ttnn.FabricTensixConfig.DISABLED,  # fabric_tensix_config
            ttnn.FabricUDMMode.DISABLED,  # fabric_udm_mode
            ttnn.FabricManagerMode.DEFAULT,  # fabric_manager
        )

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


def test_deepseek_moe_against_reference(mesh_device, hf_config, ccl):
    """Test DeepSeek MoE Block against reference implementation with PCC checking."""

    if DeepseekV3MoE is None:
        pytest.skip("DeepSeek reference model not available")

    if assert_hidden_dim_pcc is None:
        pytest.skip("PCC checking utilities not available")

    logger.info("\n" + "=" * 60)
    logger.info("Testing DeepSeek MoE Block Against Reference")
    logger.info("=" * 60)

    # Test parameters
    MODE = "decode"
    SEQ_LEN = 1
    BATCH_SIZE = 32  # Same as working test (USERS_PER_ROW)
    HIDDEN_SIZE = hf_config.hidden_size

    # Create reference model
    logger.info("Creating reference model...")
    torch.use_deterministic_algorithms(True)
    reference_model = DeepseekV3MoE(hf_config).eval().to(torch.bfloat16)

    # Generate test input
    if MODE == "prefill":
        torch_input = torch.randn(1, SEQ_LEN, HIDDEN_SIZE, dtype=torch.bfloat16)
    else:
        torch_input = torch.randn(BATCH_SIZE, 1, HIDDEN_SIZE, dtype=torch.bfloat16)

    logger.info(f"Input shape: {torch_input.shape}")

    # Generate reference output
    logger.info("Generating reference output...")
    with torch.no_grad():
        reference_output = reference_model(torch_input)
    logger.info(f"Reference output shape: {reference_output.shape}")

    # Load DeepSeek MoEBlock
    config_path = str(Path(__file__).parent.parent / "configs" / "deepseek_v3.json")
    logger.info(f"Loading DeepSeek MoEBlock from config: {config_path}")
    moe_block = MoEBlock(config_path, mesh_device, ccl)

    # Load weights from reference model
    logger.info("Loading weights from reference model...")
    state_dict = reference_model.state_dict()

    # Add quantization scales if needed
    if hasattr(hf_config, "quantization_config"):
        state_dict = add_inv_scale_to_state_dict(
            state_dict, block_shape=hf_config.quantization_config.get("weight_block_size", [128, 128])
        )

    # Map weights to our format - our MoEBlock expects "mlp." prefix
    moe_state_dict = {}
    for key, value in state_dict.items():
        new_key = f"mlp.{key}"
        moe_state_dict[new_key] = value
        logger.debug(f"Mapping weight: {key} -> {new_key}, shape: {value.shape}")

    moe_block.load_weights(moe_state_dict)

    # Convert input to TTNN format
    if MODE == "prefill":
        # For prefill, add batch dimension and shard
        tt_input = ttnn.from_torch(
            torch_input.unsqueeze(0),
            device=mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=mesh_device.shape),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )
    else:
        # For decode, reshape [batch, 1, hidden] -> [1, 1, batch, hidden]
        torch_input_reshaped = torch_input.permute(1, 0, 2).unsqueeze(0)
        # Try using ReplicateTensorToMesh like the working test instead of ShardTensor2dMesh
        # Start with DRAM like working test, then move to L1
        tt_input = ttnn.from_torch(
            torch_input_reshaped,
            device=mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),  # Changed to replicate like working test
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Start in DRAM
            layout=ttnn.TILE_LAYOUT,
        )
        # Move to L1 after creation (like working test does)
        tt_input = ttnn.to_memory_config(tt_input, ttnn.L1_MEMORY_CONFIG)

    logger.info("Running MoEBlock forward pass...")

    # Forward pass through DeepSeek MoE
    tt_output = moe_block.forward(tt_input, mode=MODE)

    # Convert output back to torch
    # Since we're now replicating input, output might also be replicated
    tt_output_torch = ttnn.to_torch(tt_output)

    # Reshape output back to match reference format
    if MODE == "prefill":
        # Remove batch dimension added for TTNN
        tt_output_torch = tt_output_torch.squeeze(0)
    else:
        # Reshape back from [1, 1, batch, hidden] to [batch, 1, hidden]
        tt_output_torch = tt_output_torch.squeeze(0).permute(1, 0, 2)

    logger.info(f"TT output shape: {tt_output_torch.shape}")

    # Check PCC
    assert_hidden_dim_pcc(tt_output_torch, reference_output, pcc_required=PCC_REQUIRED)

    logger.info(f"✅ Test passed with PCC >= {PCC_REQUIRED}")

    # Cleanup
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

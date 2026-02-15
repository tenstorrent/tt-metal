# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test suite for GPT-OSS MoE integration with TT-MoE infrastructure.

This module tests:
1. Configuration loading for GPT-OSS
2. TopKRouter functionality
3. MoE block compatibility with GPT-OSS settings
4. Comparison with reference GPT-OSS implementation
"""

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn


@pytest.fixture(scope="module")
def mesh_device_fixture(request):
    """Create mesh device for testing."""
    device_params = {"l1_small_size": 24576}
    num_devices = ttnn.get_device_count()

    # Setup fabric for CCL operations
    ttnn.set_fabric_config()

    # Create mesh device
    mesh_device = ttnn.open_mesh_device(
        ttnn.MeshShape(4, 8),
        device_params=device_params,
    )

    yield mesh_device

    # Cleanup
    ttnn.close_mesh_device(mesh_device)
    del mesh_device


def test_gpt_oss_config_loading():
    """Test that GPT-OSS configuration loads correctly."""
    config_path = Path(__file__).parent.parent / "configs" / "gpt_oss.json"

    # Check config file exists
    assert config_path.exists(), f"Config file not found: {config_path}"

    # Load and validate config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Check required sections
    assert "moe_block" in config
    moe_config = config["moe_block"]

    # Validate model parameters
    assert moe_config["model_params"]["num_experts"] == 128
    assert moe_config["model_params"]["num_experts_per_tok"] == 4
    assert moe_config["model_params"]["hidden_size"] == 2880
    assert moe_config["model_params"]["intermediate_size"] == 360

    # Validate router configuration
    assert moe_config["router"]["type"] == "topk"
    assert moe_config["router"]["use_throughput_experts"] is True

    # Validate expert configuration
    assert moe_config["experts"]["distributed"] is True
    assert moe_config["experts"]["shared"] is False

    logger.info("✅ GPT-OSS configuration loaded successfully")


def test_topk_router_initialization(mesh_device_fixture):
    """Test TopKRouter initialization and basic functionality."""
    # Import here to avoid import errors if files don't exist
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from components.routers.topk_router import TopKRouter

    # Create router configuration
    router_config = {
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "hidden_size": 2880,
        "use_throughput_experts": True,
    }

    # Create router (without weights for now)
    router = TopKRouter(mesh_device_fixture, router_config)

    # Verify configuration
    assert router.num_experts == 128
    assert router.num_experts_per_tok == 4
    assert router.hidden_size == 2880
    assert router.use_throughput_experts is True

    logger.info("✅ TopKRouter initialized successfully")


def test_topk_router_weight_loading(mesh_device_fixture):
    """Test TopKRouter weight loading functionality."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from components.routers.topk_router import TopKRouter

    # Create router configuration
    router_config = {
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "hidden_size": 2880,
        "use_throughput_experts": True,
    }

    # Create dummy weights
    dummy_weights = {
        "weight": torch.randn(128, 2880),  # [num_experts, hidden_size]
        "bias": torch.randn(128),  # [num_experts]
    }

    # Create router and load weights
    router = TopKRouter(mesh_device_fixture, router_config, state_dict=dummy_weights)

    # Verify weights are loaded
    assert router.weight is not None
    assert router.bias is not None

    logger.info("✅ TopKRouter weights loaded successfully")


def test_topk_router_forward(mesh_device_fixture):
    """Test TopKRouter forward pass."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from components.routers.topk_router import TopKRouter

    # Create router configuration
    router_config = {
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "hidden_size": 2880,
        "use_throughput_experts": True,
    }

    # Create dummy weights
    dummy_weights = {
        "weight": torch.randn(128, 2880),
        "bias": torch.randn(128),
    }

    # Create router
    router = TopKRouter(mesh_device_fixture, router_config, state_dict=dummy_weights)

    # Create input tensor (decode mode: batch=32, seq_len=1)
    batch_size = 32
    seq_len = 1
    hidden_size = 2880
    input_tensor = torch.randn(batch_size, seq_len, hidden_size)

    # Convert to TTNN tensor
    tt_input = ttnn.from_torch(
        input_tensor.unsqueeze(0),
        device=mesh_device_fixture,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device_fixture),
    )

    # Run forward pass
    expert_indices, expert_weights = router.forward(tt_input, is_decode=True)

    # Verify output shapes
    assert expert_indices is not None
    assert expert_weights is not None

    # Convert back to torch for verification
    indices_torch = ttnn.to_torch(expert_indices)
    weights_torch = ttnn.to_torch(expert_weights)

    # Check shapes: [batch*seq_len, num_experts_per_tok]
    assert indices_torch.shape[-1] == 4  # num_experts_per_tok
    assert weights_torch.shape[-1] == 4  # Dense format for throughput experts

    # Check indices are valid
    assert torch.all(indices_torch >= 0)
    assert torch.all(indices_torch < 128)

    # Check weights sum to approximately 1 (softmax normalized)
    weight_sums = weights_torch.sum(dim=-1)
    assert torch.allclose(weight_sums, torch.ones_like(weight_sums), rtol=1e-3)

    logger.info("✅ TopKRouter forward pass successful")


@pytest.mark.parametrize("mode,seq_len", [("decode", 1), ("prefill", 128)])
def test_gpt_oss_moe_block(mesh_device_fixture, mode, seq_len):
    """Test full GPT-OSS MoE block functionality."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    from moe_block import MoEBlock

    # Skip if no HF model available
    hf_model_path = os.getenv("HF_MODEL")
    if not hf_model_path:
        pytest.skip("HF_MODEL environment variable not set")

    # Load configuration
    config_path = Path(__file__).parent.parent / "configs" / "gpt_oss.json"

    # Update config with actual model path
    with open(config_path, "r") as f:
        config = json.load(f)

    config["moe_block"]["weight_path"] = hf_model_path

    # Create temporary config file
    temp_config = Path("/tmp/gpt_oss_test_config.json")
    with open(temp_config, "w") as f:
        json.dump(config, f, indent=2)

    # Create MoE block
    try:
        moe_block = MoEBlock(temp_config, mesh_device_fixture)

        # Create input tensor
        batch_size = 32 if mode == "decode" else 1
        hidden_size = 2880
        input_tensor = torch.randn(batch_size, seq_len, hidden_size)

        # Convert to TTNN tensor
        tt_input = ttnn.from_torch(
            input_tensor.permute(1, 0, 2).unsqueeze(0),  # [1, seq_len, batch, hidden]
            device=mesh_device_fixture,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                mesh_device_fixture,
                dims=(-2, -1),
                mesh_shape=mesh_device_fixture.shape,
            ),
        )

        # Run forward pass
        output = moe_block.forward(tt_input, mode=mode)

        # Verify output
        assert output is not None
        output_torch = ttnn.to_torch(output)

        # Check shape preservation
        expected_shape = input_tensor.permute(1, 0, 2).unsqueeze(0).shape
        assert output_torch.shape == expected_shape

        logger.info(f"✅ GPT-OSS MoE block {mode} mode test successful")

    except Exception as e:
        if "not yet implemented" in str(e) or "NotImplementedError" in str(e):
            pytest.skip(f"Feature not yet implemented: {e}")
        else:
            raise


def test_gpt_oss_moe_against_reference(mesh_device_fixture):
    """Compare GPT-OSS MoE block output with reference implementation."""
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent))

    # Skip if no HF model available
    hf_model_path = os.getenv("HF_MODEL")
    if not hf_model_path or "gpt-oss" not in hf_model_path.lower():
        pytest.skip("GPT-OSS HF model not available")

    # Check if reference implementation is available
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError:
        pytest.skip("GPT-OSS reference implementation not available")

    # Load model configuration
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    # Load reference model weights
    reference_model = AutoModelForCausalLM.from_pretrained(
        hf_model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Get layer 0 MoE block from reference
    layer_index = 0
    reference_mlp = reference_model.model.layers[layer_index].mlp

    # Create input
    batch_size = 32
    seq_len = 1
    hidden_size = hf_config.hidden_size
    input_tensor = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Run reference forward pass
    with torch.no_grad():
        reference_output, _ = reference_mlp(input_tensor)

    # Now run our implementation
    from moe_block import MoEBlock

    # Create config for our implementation
    config_path = Path(__file__).parent.parent / "configs" / "gpt_oss.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    config["moe_block"]["weight_path"] = hf_model_path
    config["moe_block"]["layer_index"] = layer_index

    # Update dimensions from actual model
    config["moe_block"]["model_params"]["hidden_size"] = hidden_size
    config["moe_block"]["router"]["config"]["hidden_size"] = hidden_size
    config["moe_block"]["experts"]["distributed"]["hidden_size"] = hidden_size

    # Save temp config
    temp_config = Path("/tmp/gpt_oss_comparison_config.json")
    with open(temp_config, "w") as f:
        json.dump(config, f, indent=2)

    # Create our MoE block
    moe_block = MoEBlock(temp_config, mesh_device_fixture)

    # Convert input to TTNN format
    tt_input = ttnn.from_torch(
        input_tensor.permute(1, 0, 2).unsqueeze(0),
        device=mesh_device_fixture,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            mesh_device_fixture,
            dims=(-2, -1),
            mesh_shape=mesh_device_fixture.shape,
        ),
    )

    # Run our forward pass
    our_output = moe_block.forward(tt_input, mode="decode")
    our_output_torch = ttnn.to_torch(our_output)

    # Reshape to match reference
    our_output_torch = our_output_torch.squeeze(0).permute(1, 0, 2)

    # Calculate PCC
    reference_flat = reference_output.flatten()
    our_flat = our_output_torch.flatten()

    correlation = torch.corrcoef(torch.stack([reference_flat, our_flat]))[0, 1]
    pcc = correlation.item()

    logger.info(f"PCC between reference and our implementation: {pcc:.6f}")

    # Assert PCC meets threshold
    assert pcc >= 0.98, f"PCC {pcc:.6f} below threshold 0.98"

    logger.info("✅ GPT-OSS MoE matches reference implementation")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-xvs"])

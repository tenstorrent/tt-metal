# SPDX-FileCopyrightText: (c) 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Projector Tests - PCC Validation

Tests the MLP projector (1024 -> 4096 -> 4096) against PyTorch reference.

Test Strategy:
    1. Create PyTorch reference projector with random weights
    2. Create TTNN projector with same weights
    3. Run both on same input (simulated CLIP output)
    4. Compare outputs with PCC (Pearson Correlation Coefficient)
    5. Assert PCC >= 0.99 for projector (higher than vision encoder threshold)

Usage:
    # Run all projector tests
    scripts/run_safe_pytest.sh models/demos/llava/tests/test_projectors.py -v

    # Run only PCC test
    scripts/run_safe_pytest.sh models/demos/llava/tests/test_projectors.py::test_mlp_projector_pcc -v

    # With device config
    MESH_DEVICE=N300 scripts/run_safe_pytest.sh models/demos/llava/tests/test_projectors.py -v

Reference:
    - models/demos/llava/tests/test_vision_encoders.py (test patterns)
    - models/common/utility_functions.comp_pcc
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llava.reference.llava_reference import create_mlp_projector as create_mlp_projector_pytorch
from models.demos.llava.tt.model_config import VisionModelArgs
from models.demos.llava.tt.projectors.mlp_projector import create_mlp_projector as create_mlp_projector_ttnn


def _create_random_projector_state_dict(vision_hidden_size=1024, llm_hidden_size=4096, seed=42):
    """
    Create random projector weights for testing.

    Returns state_dict with keys matching LLaVA's mm_projector format:
        mm_projector.0.weight [4096, 1024]
        mm_projector.0.bias   [4096]
        mm_projector.2.weight [4096, 4096]
        mm_projector.2.bias   [4096]
    """
    torch.manual_seed(seed)
    state_dict = {
        "mm_projector.0.weight": torch.randn(llm_hidden_size, vision_hidden_size) * 0.02,
        "mm_projector.0.bias": torch.zeros(llm_hidden_size),
        "mm_projector.2.weight": torch.randn(llm_hidden_size, llm_hidden_size) * 0.02,
        "mm_projector.2.bias": torch.zeros(llm_hidden_size),
    }
    return state_dict


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_mlp_projector_pcc(mesh_device):
    """
    Test MLP projector PCC against PyTorch reference

    Validates:
        1. Weight loading from state_dict
        2. Forward pass correctness (Linear -> GELU -> Linear)
        3. PCC >= 0.99 (projector threshold, higher than vision encoder)

    Args:
        mesh_device: Fixture providing ttnn mesh device (1, 2) for n300
    """
    logger.info("=" * 80)
    logger.info("Testing MLP Projector PCC")
    logger.info("=" * 80)

    # Configuration
    batch_size = 1
    num_patches = 576  # CLIP output: 24x24 patches
    vision_hidden_size = 1024
    llm_hidden_size = 4096

    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Num patches: {num_patches}")
    logger.info(f"Vision hidden size: {vision_hidden_size}")
    logger.info(f"LLM hidden size: {llm_hidden_size}")

    # Create VisionModelArgs
    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=batch_size)

    # Create random weights (same for both PyTorch and TTNN)
    state_dict = _create_random_projector_state_dict(
        vision_hidden_size=vision_hidden_size,
        llm_hidden_size=llm_hidden_size,
        seed=42,
    )

    # Create PyTorch reference
    logger.info("Creating PyTorch reference projector...")
    pytorch_projector = create_mlp_projector_pytorch(
        vision_hidden_size=vision_hidden_size,
        llm_hidden_size=llm_hidden_size,
        state_dict=state_dict,
    )

    # Create TTNN projector
    logger.info("Creating TTNN projector...")
    ttnn_projector = create_mlp_projector_ttnn(
        args=args,
        dtype=ttnn.bfloat16,
        state_dict=state_dict,
    )

    # Create test input [B=1, 576, 1024] (simulated CLIP output)
    torch.manual_seed(123)
    test_input = torch.randn(batch_size, num_patches, vision_hidden_size)

    # Run PyTorch reference
    logger.info("Running PyTorch reference...")
    with torch.no_grad():
        pytorch_output = pytorch_projector(test_input)

    logger.info(f"PyTorch output shape: {pytorch_output.shape}")

    # Convert input to TTNN
    ttnn_input = ttnn.from_torch(
        test_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run TTNN model
    logger.info("Running TTNN model...")
    ttnn_output = ttnn_projector(ttnn_input)

    # Convert TTNN output back to torch
    ttnn_output_torch = ttnn.to_torch(ttnn.get_device_tensors(ttnn_output)[0])

    logger.info(f"TTNN output shape: {ttnn_output_torch.shape}")

    # Compute PCC
    logger.info("Computing PCC...")
    passing, pcc_value = comp_pcc(pytorch_output, ttnn_output_torch, pcc=0.99)

    logger.info("=" * 80)
    logger.info(f"PCC: {pcc_value:.6f}")
    logger.info(f"Threshold: 0.99")
    logger.info(f"Result: {'PASS' if passing else 'FAIL'}")
    logger.info("=" * 80)

    # Additional statistics
    diff = torch.abs(pytorch_output - ttnn_output_torch)
    logger.info(f"Max absolute difference: {diff.max().item():.6f}")
    logger.info(f"Mean absolute difference: {diff.mean().item():.6f}")

    # Assert PCC threshold
    assert passing, f"PCC {pcc_value:.6f} < 0.99 (threshold)"

    logger.info("MLP projector PCC test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_mlp_projector_output_shape(mesh_device):
    """
    Test MLP projector output shape

    Validates:
        1. Output shape is [B, 576, 4096]
        2. Input dimension 1024 is projected to 4096
    """
    logger.info("=" * 80)
    logger.info("Testing MLP Projector Output Shape")
    logger.info("=" * 80)

    batch_size = 2
    num_patches = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=batch_size)

    state_dict = _create_random_projector_state_dict(
        vision_hidden_size=vision_hidden_size,
        llm_hidden_size=llm_hidden_size,
    )

    ttnn_projector = create_mlp_projector_ttnn(
        args=args,
        dtype=ttnn.bfloat16,
        state_dict=state_dict,
    )

    # Create test input [B=2, 576, 1024]
    test_input = torch.randn(batch_size, num_patches, vision_hidden_size)
    ttnn_input = ttnn.from_torch(
        test_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run TTNN model
    ttnn_output = ttnn_projector(ttnn_input)
    ttnn_output_torch = ttnn.to_torch(ttnn.get_device_tensors(ttnn_output)[0])

    # Validate shape
    expected_shape = (batch_size, num_patches, llm_hidden_size)
    actual_shape = tuple(ttnn_output_torch.shape)

    logger.info(f"Expected shape: {expected_shape}")
    logger.info(f"Actual shape: {actual_shape}")

    assert actual_shape == expected_shape, f"Shape mismatch: {actual_shape} != {expected_shape}"

    logger.info("Output shape test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 2)],  # n300 dual-chip
    indirect=True,
)
def test_mlp_projector_deterministic(mesh_device):
    """
    Test MLP projector is deterministic

    Validates:
        1. Same input produces same output
        2. No randomness in forward pass
    """
    logger.info("=" * 80)
    logger.info("Testing MLP Projector Determinism")
    logger.info("=" * 80)

    batch_size = 1
    num_patches = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=batch_size)

    state_dict = _create_random_projector_state_dict(
        vision_hidden_size=vision_hidden_size,
        llm_hidden_size=llm_hidden_size,
    )

    ttnn_projector = create_mlp_projector_ttnn(
        args=args,
        dtype=ttnn.bfloat16,
        state_dict=state_dict,
    )

    # Create test input
    test_input = torch.randn(batch_size, num_patches, vision_hidden_size)
    ttnn_input = ttnn.from_torch(
        test_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run twice
    output1 = ttnn.to_torch(ttnn.get_device_tensors(ttnn_projector(ttnn_input))[0])
    output2 = ttnn.to_torch(ttnn.get_device_tensors(ttnn_projector(ttnn_input))[0])

    # Check outputs are identical
    diff = torch.abs(output1 - output2)
    max_diff = diff.max().item()

    logger.info(f"Max difference between runs: {max_diff:.10f}")

    assert max_diff < 1e-6, f"Outputs differ: max_diff={max_diff:.10f}"

    logger.info("Determinism test passed!")


@pytest.mark.parametrize(
    "mesh_device",
    [(1, 1)],  # n150 single-chip
    indirect=True,
)
def test_mlp_projector_single_chip(mesh_device):
    """
    Test MLP projector on n150 (single chip)

    Validates:
        1. Model works on single chip
        2. Tensor replication works correctly
    """
    logger.info("=" * 80)
    logger.info("Testing MLP Projector on Single Chip (n150)")
    logger.info("=" * 80)

    batch_size = 1
    num_patches = 576
    vision_hidden_size = 1024
    llm_hidden_size = 4096

    logger.info(f"Device: {mesh_device} (single chip)")

    args = VisionModelArgs(mesh_device=mesh_device, max_batch_size=batch_size)

    state_dict = _create_random_projector_state_dict(
        vision_hidden_size=vision_hidden_size,
        llm_hidden_size=llm_hidden_size,
    )

    ttnn_projector = create_mlp_projector_ttnn(
        args=args,
        dtype=ttnn.bfloat16,
        state_dict=state_dict,
    )

    # Create test input
    test_input = torch.randn(batch_size, num_patches, vision_hidden_size)
    ttnn_input = ttnn.from_torch(
        test_input,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Run TTNN model
    ttnn_output = ttnn_projector(ttnn_input)
    ttnn_output_torch = ttnn.to_torch(ttnn.get_device_tensors(ttnn_output)[0])

    # Validate shape
    expected_shape = (batch_size, num_patches, llm_hidden_size)
    actual_shape = tuple(ttnn_output_torch.shape)

    assert actual_shape == expected_shape, f"Shape mismatch: {actual_shape} != {expected_shape}"

    logger.info("Single chip test passed!")


if __name__ == "__main__":
    logger.info("Projector Tests - Direct execution mode")
    logger.info("For proper testing, use pytest:")
    logger.info("  scripts/run_safe_pytest.sh models/demos/llava/tests/test_projectors.py -v")

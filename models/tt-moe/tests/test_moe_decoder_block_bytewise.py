#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test to verify MoEDecoderBlock2D (MoE + SharedExpert) produces bytewise identical outputs
between reference and copied implementations using actual model weights.
"""

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

# Add tt-moe directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our copied implementation
from deepseek_reference.moe_decoder_block_2d import MoEDecoderBlock2D as CopiedMoEDecoderBlock2D

# Import reference implementation
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import (
    MoEDecoderBlock2D as ReferenceMoEDecoderBlock2D,
)
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config


def verify_bytewise_identical(tensor1, tensor2, name="tensor"):
    """Verify two tensors are bytewise identical."""
    # Convert to numpy
    if hasattr(tensor1, "cpu"):
        arr1 = tensor1.cpu().float().numpy()
    else:
        arr1 = tensor1.float().numpy() if hasattr(tensor1, "float") else tensor1.numpy()

    if hasattr(tensor2, "cpu"):
        arr2 = tensor2.cpu().float().numpy()
    else:
        arr2 = tensor2.float().numpy() if hasattr(tensor2, "float") else tensor2.numpy()

    # Get bytes
    bytes1 = arr1.tobytes()
    bytes2 = arr2.tobytes()

    # Compare hashes
    hash1 = hashlib.md5(bytes1).hexdigest()
    hash2 = hashlib.md5(bytes2).hexdigest()

    logger.info(f"[{name}] Hash1 (reference): {hash1}")
    logger.info(f"[{name}] Hash2 (copied):    {hash2}")

    if hash1 != hash2:
        # Show where they differ
        diff_mask = arr1 != arr2
        if diff_mask.any():
            logger.error(f"[{name}] Arrays differ at {diff_mask.sum()} positions")
            logger.error(f"[{name}] Max absolute difference: {np.abs(arr1 - arr2).max()}")
        else:
            logger.warning(f"[{name}] Arrays are equal but hashes differ (floating point precision?)")

    assert hash1 == hash2, f"{name} not bytewise identical! ref={hash1}, copy={hash2}"
    return True


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mode,num_tokens", [("decode", 128)])
def test_moe_decoder_block_bytewise(
    mode,
    num_tokens,
    set_deterministic_env,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    state_dict,
):
    """Test MoEDecoderBlock2D for bytewise identical outputs with actual weights."""

    logger.info("=" * 80)
    logger.info("Testing MoEDecoderBlock2D (MoE + SharedExpert) for bytewise identical outputs")
    logger.info("Using ACTUAL model weights from layer 3")
    logger.info("=" * 80)

    # Get state dict for layer 3 (which has MoE)
    layer_idx = 3
    layer_state_dict = {
        k.replace(f"model.layers.{layer_idx}.", ""): v
        for k, v in state_dict.items()
        if k.startswith(f"model.layers.{layer_idx}.")
    }

    # Filter to just MLP-related weights (MoE + SharedExpert)
    mlp_state_dict = {k: v for k, v in layer_state_dict.items() if k.startswith("mlp.")}

    logger.info(f"Loaded {len(mlp_state_dict)} MLP weights for layer {layer_idx}")

    # Create identical input tensor
    torch.manual_seed(5)
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Save input for verification
    input_path = Path("/tmp/test_moe_decoder_input.pt")
    torch.save(torch_input, input_path)
    logger.info(f"Saved test input to {input_path}, shape: {torch_input.shape}")

    # Enable output saving
    os.environ["SAVE_MOE_DECODER_OUTPUT"] = "1"

    # 1. Run REFERENCE MoEDecoderBlock2D implementation
    logger.info("-" * 40)
    logger.info("Running REFERENCE MoEDecoderBlock2D implementation...")
    logger.info("-" * 40)

    # Setup configurations for REFERENCE implementation
    ref_weight_config = get_test_weight_config(
        ReferenceMoEDecoderBlock2D,
        hf_config,
        (mlp_state_dict,),
        cache_path / "reference_moe_decoder",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    ref_model_config = get_model_config(ReferenceMoEDecoderBlock2D, mode, hf_config, mesh_device)
    ref_model_config.update({"topk_fallback": True})

    ref_model_state = ReferenceMoEDecoderBlock2D.create_mlp_state(hf_config, mesh_device, ccl)
    ref_model_shared_state = ReferenceMoEDecoderBlock2D.create_mlp_shared_state(hf_config, mesh_device)

    # Merge the MLP configs into the run config
    ref_run_config = create_run_config(ref_model_config, ref_weight_config, ref_model_state, ref_model_shared_state)

    # Convert input to TTNN for reference
    ref_tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run forward pass - calling forward_mlp_decode directly
    ref_tt_input = ttnn.to_memory_config(
        ref_tt_input, ref_run_config.get("input_memory_config", ttnn.DRAM_MEMORY_CONFIG)
    )

    # Call forward_mlp_decode which does MoE + SharedExpert
    ref_tt_output = ReferenceMoEDecoderBlock2D.forward_mlp_decode(ref_tt_input, ref_run_config)

    # Convert output back to torch
    reference_output = ttnn.to_torch(
        ref_tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Clean up reference tensors
    ttnn.deallocate(ref_tt_input)
    ttnn.deallocate(ref_tt_output)

    # Check if reference saved output
    ref_output_path = Path("/tmp/moe_decoder_reference_output/moe_decoder_output.npy")
    if ref_output_path.exists():
        ref_output_np = np.load(ref_output_path)
        logger.info(f"Loaded reference output from {ref_output_path}")
        logger.info(f"Reference output shape from file: {ref_output_np.shape}")

    # 2. Run COPIED MoEDecoderBlock2D implementation
    logger.info("-" * 40)
    logger.info("Running COPIED MoEDecoderBlock2D implementation...")
    logger.info("-" * 40)

    copy_weight_config = get_test_weight_config(
        CopiedMoEDecoderBlock2D,  # Use our copied class
        hf_config,
        (mlp_state_dict,),
        cache_path / "copied_moe_decoder",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    copy_model_config = get_model_config(CopiedMoEDecoderBlock2D, mode, hf_config, mesh_device)
    copy_model_config.update({"topk_fallback": True})

    copy_model_state = CopiedMoEDecoderBlock2D.create_mlp_state(hf_config, mesh_device, ccl)
    copy_model_shared_state = CopiedMoEDecoderBlock2D.create_mlp_shared_state(hf_config, mesh_device)

    copy_run_config = create_run_config(
        copy_model_config, copy_weight_config, copy_model_state, copy_model_shared_state
    )

    # Convert input to TTNN (use same input tensor)
    copy_tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run forward pass - calling forward_mlp_decode directly
    copy_tt_input = ttnn.to_memory_config(
        copy_tt_input, copy_run_config.get("input_memory_config", ttnn.DRAM_MEMORY_CONFIG)
    )

    # Call forward_mlp_decode which does MoE + SharedExpert
    copy_tt_output = CopiedMoEDecoderBlock2D.forward_mlp_decode(copy_tt_input, copy_run_config)

    # Convert output back to torch
    copied_output = ttnn.to_torch(
        copy_tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Clean up
    ttnn.deallocate(copy_tt_input)
    ttnn.deallocate(copy_tt_output)

    # Check if copied saved output
    copy_output_path = Path("/tmp/moe_decoder_copied_output/moe_decoder_output.npy")
    if copy_output_path.exists():
        copy_output_np = np.load(copy_output_path)
        logger.info(f"Loaded copied output from {copy_output_path}")
        logger.info(f"Copied output shape from file: {copy_output_np.shape}")

    # 3. Compare outputs
    logger.info("=" * 80)
    logger.info("Comparing MoEDecoderBlock2D outputs for bytewise identity...")
    logger.info("=" * 80)

    # Both should have the same shape from TTNN
    ref_shape = reference_output.shape
    copy_shape = copied_output.shape

    logger.info(f"Reference MoEDecoderBlock2D output shape: {ref_shape}")
    logger.info(f"Copied MoEDecoderBlock2D output shape:    {copy_shape}")
    assert ref_shape == copy_shape, f"Shape mismatch! ref={ref_shape}, copy={copy_shape}"

    # Verify bytewise identical
    verify_bytewise_identical(reference_output, copied_output, name="MoEDecoderBlock2D_output")

    # Also compare saved files if they exist
    if ref_output_path.exists() and copy_output_path.exists():
        logger.info("Comparing saved output files...")
        ref_hash = hashlib.md5(ref_output_np.tobytes()).hexdigest()
        copy_hash = hashlib.md5(copy_output_np.tobytes()).hexdigest()
        logger.info(f"Saved reference hash: {ref_hash}")
        logger.info(f"Saved copied hash:    {copy_hash}")
        assert ref_hash == copy_hash, f"Saved outputs not identical! ref={ref_hash}, copy={copy_hash}"

    logger.info("=" * 80)
    logger.info("✅ SUCCESS: MoEDecoderBlock2D (MoE + SharedExpert) produces bytewise identical outputs!")
    logger.info("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

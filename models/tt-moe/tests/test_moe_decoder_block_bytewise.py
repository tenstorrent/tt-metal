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
    """Test MoEDecoderBlock2D for bytewise identical outputs using actual model weights."""

    logger.info("=" * 80)
    logger.info("Testing MoEDecoderBlock2D (MoE + SharedExpert) for bytewise identical outputs")
    logger.info("Using layer 3 weights from actual model (same as Test B)")
    logger.info("=" * 80)

    # Create a view of the state_dict for just layer 3 to avoid loading all files
    # This avoids the corrupted file model-00101-of-000163.safetensors
    layer_idx = 3
    layer3_state_dict = state_dict.view_with_prefix(f"model.layers.{layer_idx}.")

    # Create identical input tensor
    # Use batch_size=32 to be divisible by the 32 devices (4x8 mesh)
    batch_size = 32  # Must be divisible by mesh_device.shape[0] * mesh_device.shape[1]
    torch.manual_seed(5)
    torch_input = torch.randn(batch_size, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

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

    # Use a unique cache directory for this test
    test_cache_path = Path("/tmp/deepseek_cache_test_d1")
    test_cache_path.mkdir(exist_ok=True, parents=True)

    # Setup configurations for REFERENCE implementation - use layer 3 view
    ref_weight_config = get_test_weight_config(
        ReferenceMoEDecoderBlock2D,
        hf_config,
        (layer3_state_dict,),  # Pass layer 3 view to avoid corrupted files
        test_cache_path / "reference_moe_decoder",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    # Get the base model config and then add the MLP-specific config
    ref_model_config = get_model_config(ReferenceMoEDecoderBlock2D, mode, hf_config, mesh_device)
    ref_model_config.update({"topk_fallback": True})

    # Add the MoE and SharedExpert configurations using the specific decode_mlp_config method
    mlp_config = ReferenceMoEDecoderBlock2D.decode_mlp_config(hf_config, mesh_device)
    ref_model_config.update(mlp_config)

    # Create paged config for decoder block (required for create_state)
    from models.demos.deepseek_v3.tt.mla.mla1d import MLA1D
    from models.demos.deepseek_v3.utils.test_utils import paged_cache_from_torch

    USERS_PER_ROW = 128
    paged_config = MLA1D.get_valid_paged_config(hf_config.max_position_embeddings, USERS_PER_ROW, mesh_device.shape[1])

    # Create dummy cache for MLA
    dim = hf_config.kv_lora_rank + hf_config.qk_rope_head_dim
    # Use batch_size=32 to match the input tensor and be divisible by 32 devices
    input_cache = torch.zeros((batch_size, 1, 0, dim), dtype=torch.bfloat16)
    paged_input_cache, _ = paged_cache_from_torch(input_cache, tuple(mesh_device.shape), paged_config, None)

    # Use create_state instead of create_mlp_state to get proper mesh_device in state
    ref_model_state = ReferenceMoEDecoderBlock2D.create_state(
        hf_config,
        paged_config,
        mesh_device,
        ccl,
        mla_cache=paged_input_cache,
    )
    ref_model_shared_state = ReferenceMoEDecoderBlock2D.create_shared_state(hf_config, mesh_device)

    # Merge the MLP configs into the run config
    ref_run_config = create_run_config(ref_model_config, ref_weight_config, ref_model_state, ref_model_shared_state)

    # CRITICAL: forward_mlp_decode expects cfg["mlp"] from the full config, not the full config itself
    # We need to extract just the MLP portion of the configuration
    # The full decoder config has: {"mla_norm": ..., "mla": ..., "mlp_norm": ..., "mlp": {...}}
    # We only need the "mlp" portion for forward_mlp_decode
    ref_mlp_config = ref_run_config.get("mlp", ref_run_config)

    # CRITICAL FIX: The MoE inside the MLP expects cfg["ccl"] to be the CCL object
    # But MoE.create_state doesn't store the CCL in the state (unlike DistributedRMSNorm which does)
    # We need to manually ensure the CCL is in the MLP config for MoE to access it
    if "moe" in ref_mlp_config and "ccl" not in ref_mlp_config["moe"]:
        ref_mlp_config["moe"]["ccl"] = ccl

    # Convert input to TTNN for reference
    ref_tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run forward pass - we need to preprocess the input like forward_decode does
    ref_tt_input = ttnn.to_memory_config(
        ref_tt_input, ref_run_config.get("input_memory_config", ttnn.DRAM_MEMORY_CONFIG)
    )

    # CRITICAL: forward_mlp_decode expects input that has been through RMS norm and resharding
    # In forward_decode, the input goes through:
    # 1. mlp_norm (RMS normalization)
    # 2. mlp_reshard (memory config transformation)
    # We need to apply these transformations before calling forward_mlp_decode

    # Import RMS norm
    from models.demos.deepseek_v3.tt.rms_norm.distributed_rms_norm import DistributedRMSNorm

    # Apply MLP norm (RMS normalization)
    if "mlp_norm_reshard" in ref_run_config:
        mlp_norm_in = ttnn.to_memory_config(ref_tt_input, **ref_run_config["mlp_norm_reshard"])
    else:
        mlp_norm_in = ref_tt_input

    mlp_norm_out = DistributedRMSNorm.forward_decode(mlp_norm_in, ref_run_config["mlp_norm"])

    if mlp_norm_in is not ref_tt_input:
        ttnn.deallocate(mlp_norm_in)

    # Apply MLP resharding
    if "mlp_reshard" in ref_run_config:
        mlp_ready_input = ttnn.to_memory_config(mlp_norm_out, **ref_run_config["mlp_reshard"])
        ttnn.deallocate(mlp_norm_out)
    else:
        mlp_ready_input = mlp_norm_out

    # Now call forward_mlp_decode with properly preprocessed input
    ref_tt_output = ReferenceMoEDecoderBlock2D.forward_mlp_decode(mlp_ready_input, ref_mlp_config)
    ttnn.deallocate(mlp_ready_input)

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
        (layer3_state_dict,),  # Pass layer 3 view to avoid corrupted files
        test_cache_path / "copied_moe_decoder",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    # Get the base model config and then add the MLP-specific config
    copy_model_config = get_model_config(CopiedMoEDecoderBlock2D, mode, hf_config, mesh_device)
    copy_model_config.update({"topk_fallback": True})

    # Add the MoE and SharedExpert configurations using the specific decode_mlp_config method
    mlp_config = CopiedMoEDecoderBlock2D.decode_mlp_config(hf_config, mesh_device)
    copy_model_config.update(mlp_config)

    # Use create_state for copied implementation too, to match reference
    copy_model_state = CopiedMoEDecoderBlock2D.create_state(
        hf_config,
        paged_config,
        mesh_device,
        ccl,
        mla_cache=paged_input_cache,
    )
    copy_model_shared_state = CopiedMoEDecoderBlock2D.create_shared_state(hf_config, mesh_device)

    copy_run_config = create_run_config(
        copy_model_config, copy_weight_config, copy_model_state, copy_model_shared_state
    )

    # CRITICAL: Extract just the MLP portion for forward_mlp_decode
    copy_mlp_config = copy_run_config.get("mlp", copy_run_config)

    # CRITICAL FIX: Ensure CCL is in the MoE config for the copied implementation too
    if "moe" in copy_mlp_config and "ccl" not in copy_mlp_config["moe"]:
        copy_mlp_config["moe"]["ccl"] = ccl

    # Convert input to TTNN (use same input tensor)
    copy_tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run forward pass - preprocess input like forward_decode does
    copy_tt_input = ttnn.to_memory_config(
        copy_tt_input, copy_run_config.get("input_memory_config", ttnn.DRAM_MEMORY_CONFIG)
    )

    # Apply the same preprocessing as for the reference implementation
    # Apply MLP norm (RMS normalization)
    if "mlp_norm_reshard" in copy_run_config:
        copy_mlp_norm_in = ttnn.to_memory_config(copy_tt_input, **copy_run_config["mlp_norm_reshard"])
    else:
        copy_mlp_norm_in = copy_tt_input

    copy_mlp_norm_out = DistributedRMSNorm.forward_decode(copy_mlp_norm_in, copy_run_config["mlp_norm"])

    if copy_mlp_norm_in is not copy_tt_input:
        ttnn.deallocate(copy_mlp_norm_in)

    # Apply MLP resharding
    if "mlp_reshard" in copy_run_config:
        copy_mlp_ready_input = ttnn.to_memory_config(copy_mlp_norm_out, **copy_run_config["mlp_reshard"])
        ttnn.deallocate(copy_mlp_norm_out)
    else:
        copy_mlp_ready_input = copy_mlp_norm_out

    # Now call forward_mlp_decode with properly preprocessed input
    copy_tt_output = CopiedMoEDecoderBlock2D.forward_mlp_decode(copy_mlp_ready_input, copy_mlp_config)
    ttnn.deallocate(copy_mlp_ready_input)

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

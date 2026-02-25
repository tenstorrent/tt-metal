# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test to verify our copied DeepSeek reference implementation produces
bytewise identical outputs to the original reference.
"""

import hashlib
import os

# Import from our COPIED files (not the originals)
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn

sys.path.insert(0, str(Path(__file__).parent.parent))  # Add tt-moe directory to path

sys.path.insert(0, str(Path(__file__).parent.parent))  # Already there - allows importing from tt-moe
from deepseek_reference.moe import MoE as CopiedMoE

from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE

# Also import originals for comparison
from models.demos.deepseek_v3.tt.moe import MoE as ReferenceMoE
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    get_model_config,
    get_test_weight_config,
    run_module_forward,
)


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


@pytest.fixture
def reference_model(hf_config):
    """Get the reference DeepSeek MoE model."""
    torch.use_deterministic_algorithms(True)
    hf_config.n_shared_experts = None  # No shared experts for this test
    return DeepseekV3MoE(hf_config).eval()


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("topk_fallback", [True])
@pytest.mark.parametrize("mode,num_tokens", [("decode", 128)])
def test_moe_only(
    mode,
    num_tokens,
    set_deterministic_env,
    reference_model,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
):
    """Test that our copied MoE produces bytewise identical outputs to reference."""

    logger.info("=" * 80)
    logger.info("Testing copied MoE implementation for bytewise identical outputs")
    logger.info("=" * 80)

    # Get state dict from reference model
    state_dict = add_inv_scale_to_state_dict(
        reference_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Create identical input tensor
    torch.manual_seed(5)
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Save input for verification
    input_path = Path("/tmp/test_deepseek_copy_input.pt")
    torch.save(torch_input, input_path)
    logger.info(f"Saved test input to {input_path}, shape: {torch_input.shape}")

    # Enable output saving for both implementations
    os.environ["SAVE_MOE_OUTPUT"] = "1"

    # 1. Run REFERENCE TTNN implementation
    logger.info("-" * 40)
    logger.info("Running REFERENCE TTNN implementation...")
    logger.info("-" * 40)

    # Setup configurations for REFERENCE TTNN implementation
    ref_weight_config = get_test_weight_config(
        ReferenceMoE,
        hf_config,
        (state_dict,),
        cache_path / "reference_moe",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    ref_model_config = get_model_config(ReferenceMoE, mode, hf_config, mesh_device)
    ref_model_config.update({"topk_fallback": topk_fallback})

    ref_model_state = ReferenceMoE.create_state(hf_config, mesh_device, ccl)
    ref_model_shared_state = ReferenceMoE.create_shared_state(hf_config, mesh_device)

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

    # Run forward pass using reference TTNN implementation
    ref_tt_input = ttnn.to_memory_config(ref_tt_input, ref_run_config["input_memory_config"])
    ref_tt_output = run_module_forward(ReferenceMoE, mode, ref_tt_input, ref_run_config)

    # Convert output back to torch
    reference_output = ttnn.to_torch(
        ref_tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Clean up reference tensors
    ttnn.deallocate(ref_tt_input)
    ttnn.deallocate(ref_tt_output)

    # Load reference output that was saved (for verification)
    ref_output_path = Path("/tmp/moe_reference_output/moe_output.npy")
    if ref_output_path.exists():
        ref_output_np = np.load(ref_output_path)
        logger.info(f"Loaded reference output from {ref_output_path}")
        logger.info(f"Reference output shape from file: {ref_output_np.shape}")

    # 2. Run COPIED implementation using TTNN
    logger.info("-" * 40)
    logger.info("Running COPIED TTNN implementation...")
    logger.info("-" * 40)

    # Run the exact same thing again but using the copied module
    # Since the copied files import from the originals, they should produce identical results
    copy_weight_config = get_test_weight_config(
        CopiedMoE,  # Use our copied class
        hf_config,
        (state_dict,),
        cache_path / "copied_moe",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    copy_model_config = get_model_config(CopiedMoE, mode, hf_config, mesh_device)
    copy_model_config.update({"topk_fallback": topk_fallback})

    copy_model_state = CopiedMoE.create_state(hf_config, mesh_device, ccl)
    copy_model_shared_state = CopiedMoE.create_shared_state(hf_config, mesh_device)

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

    # Run forward pass using COPIED implementation
    copy_tt_input = ttnn.to_memory_config(copy_tt_input, copy_run_config["input_memory_config"])
    copy_tt_output = run_module_forward(CopiedMoE, mode, copy_tt_input, copy_run_config)

    # Convert output back to torch
    copied_output = ttnn.to_torch(
        copy_tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Clean up
    ttnn.deallocate(copy_tt_input)
    ttnn.deallocate(copy_tt_output)

    # 3. Compare outputs
    logger.info("=" * 80)
    logger.info("Comparing TTNN outputs for bytewise identity...")
    logger.info("=" * 80)

    # Both should have the same shape from TTNN
    ref_shape = reference_output.shape
    copy_shape = copied_output.shape

    logger.info(f"Reference TTNN output shape: {ref_shape}")
    logger.info(f"Copied TTNN output shape:    {copy_shape}")
    assert ref_shape == copy_shape, f"Shape mismatch! ref={ref_shape}, copy={copy_shape}"

    # Verify bytewise identical - these should be EXACTLY the same
    # since the copied files just import from the originals
    verify_bytewise_identical(reference_output, copied_output, name="TTNN_MoE_output")

    # If we also saved the output from copied implementation, compare that too
    copied_output_path = Path("/tmp/moe_copied_output/moe_output.npy")
    if copied_output_path.exists():
        copied_output_np = np.load(copied_output_path)
        logger.info(f"Loaded copied output from {copied_output_path}")

        # Compare saved outputs
        ref_hash = hashlib.md5(ref_output_np.tobytes()).hexdigest()
        copy_hash = hashlib.md5(copied_output_np.tobytes()).hexdigest()
        logger.info(f"Saved reference hash: {ref_hash}")
        logger.info(f"Saved copied hash:    {copy_hash}")

        assert ref_hash == copy_hash, f"Saved outputs not identical! ref={ref_hash}, copy={copy_hash}"

    logger.info("=" * 80)
    logger.info("✅ SUCCESS: Copied implementation produces bytewise identical outputs!")
    logger.info("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

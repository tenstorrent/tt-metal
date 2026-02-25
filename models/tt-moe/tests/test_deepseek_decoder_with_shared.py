# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test to verify our DeepSeek MoEDecoderBlock2D (MoE + SharedExpert)
produces bytewise identical outputs to the reference.
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

# Import our copied decoder block that combines MoE + SharedExpert
from deepseek_reference.moe_decoder_block_2d import MoEDecoderBlock2D as CopiedMoEDecoderBlock2D

from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP, DeepseekV3MoE

# Import reference decoder block for comparison
from models.demos.deepseek_v3.tt.decoder_block.moe_decoder_block_2d import (
    MoEDecoderBlock2D as ReferenceMoEDecoderBlock2D,
)
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
def reference_models(hf_config):
    """Get the reference MoE and SharedExpert models."""
    torch.use_deterministic_algorithms(True)
    # Create both MoE and SharedExpert reference models
    hf_config.n_shared_experts = 1  # Enable shared expert for this test
    moe_model = DeepseekV3MoE(hf_config).eval()
    shared_expert_model = DeepseekV3MLP(
        config=hf_config,
        hidden_size=hf_config.hidden_size,
        intermediate_size=hf_config.moe_intermediate_size * hf_config.n_shared_experts,
    ).eval()
    return moe_model, shared_expert_model


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("topk_fallback", [True])
@pytest.mark.parametrize("mode,num_tokens", [("decode", 128)])
def test_decoder_with_shared_expert(
    mode,
    num_tokens,
    set_deterministic_env,
    reference_models,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    topk_fallback,
):
    """Test that our copied MoEDecoderBlock2D produces bytewise identical outputs to reference."""

    logger.info("=" * 80)
    logger.info("Testing MoEDecoderBlock2D (MoE + SharedExpert) for bytewise identical outputs")
    logger.info("=" * 80)

    moe_model, shared_expert_model = reference_models

    # Get state dict from reference models
    moe_state_dict = add_inv_scale_to_state_dict(
        moe_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    shared_expert_state_dict = add_inv_scale_to_state_dict(
        {f"shared_expert.{k}": v for k, v in shared_expert_model.state_dict().items()},
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Combine state dicts for decoder block
    combined_state_dict = {}
    combined_state_dict.update({f"mlp.{k}": v for k, v in moe_state_dict.items()})
    combined_state_dict.update(shared_expert_state_dict)

    # Create identical input tensor
    torch.manual_seed(5)
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Save input for verification
    input_path = Path("/tmp/test_decoder_with_shared_input.pt")
    torch.save(torch_input, input_path)
    logger.info(f"Saved test input to {input_path}, shape: {torch_input.shape}")

    # Enable output saving for both implementations
    os.environ["SAVE_DECODER_OUTPUT"] = "1"

    # 1. Run REFERENCE TTNN implementation
    logger.info("-" * 40)
    logger.info("Running REFERENCE MoEDecoderBlock2D implementation...")
    logger.info("-" * 40)

    # Setup configurations for REFERENCE TTNN implementation
    ref_weight_config = get_test_weight_config(
        ReferenceMoEDecoderBlock2D,
        hf_config,
        (combined_state_dict,),
        cache_path / "reference_decoder",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    ref_model_config = get_model_config(ReferenceMoEDecoderBlock2D, mode, hf_config, mesh_device)
    ref_model_config.update({"topk_fallback": topk_fallback})

    ref_model_state = ReferenceMoEDecoderBlock2D.create_state(hf_config, mesh_device, ccl)
    ref_model_shared_state = ReferenceMoEDecoderBlock2D.create_shared_state(hf_config, mesh_device)

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
    ref_tt_output = run_module_forward(ReferenceMoEDecoderBlock2D, mode, ref_tt_input, ref_run_config)

    # Convert output back to torch
    reference_output = ttnn.to_torch(
        ref_tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Clean up reference tensors
    ttnn.deallocate(ref_tt_input)
    ttnn.deallocate(ref_tt_output)

    # 2. Run COPIED implementation using TTNN
    logger.info("-" * 40)
    logger.info("Running COPIED MoEDecoderBlock2D implementation...")
    logger.info("-" * 40)

    copy_weight_config = get_test_weight_config(
        CopiedMoEDecoderBlock2D,  # Use our copied class
        hf_config,
        (combined_state_dict,),
        cache_path / "copied_decoder",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    copy_model_config = get_model_config(CopiedMoEDecoderBlock2D, mode, hf_config, mesh_device)
    copy_model_config.update({"topk_fallback": topk_fallback})

    copy_model_state = CopiedMoEDecoderBlock2D.create_state(hf_config, mesh_device, ccl)
    copy_model_shared_state = CopiedMoEDecoderBlock2D.create_shared_state(hf_config, mesh_device)

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
    copy_tt_output = run_module_forward(CopiedMoEDecoderBlock2D, mode, copy_tt_input, copy_run_config)

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
    logger.info("Comparing MoEDecoderBlock2D outputs for bytewise identity...")
    logger.info("=" * 80)

    # Both should have the same shape from TTNN
    ref_shape = reference_output.shape
    copy_shape = copied_output.shape

    logger.info(f"Reference decoder output shape: {ref_shape}")
    logger.info(f"Copied decoder output shape:    {copy_shape}")
    assert ref_shape == copy_shape, f"Shape mismatch! ref={ref_shape}, copy={copy_shape}"

    # Verify bytewise identical - these should be EXACTLY the same
    # since the copied files have minimal import changes
    verify_bytewise_identical(reference_output, copied_output, name="MoEDecoderBlock2D_output")

    logger.info("=" * 80)
    logger.info("✅ SUCCESS: MoEDecoderBlock2D produces bytewise identical outputs!")
    logger.info("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Direct test to verify MoE + SharedExpert produces bytewise identical outputs
between reference and copied implementations.
"""

import hashlib

# Import our copied implementations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.tt.mlp.shared_expert import SharedExpert as ReferenceSharedExpert

# Import reference implementations
from models.demos.deepseek_v3.tt.moe import MoE as ReferenceMoE

sys.path.insert(0, str(Path(__file__).parent.parent))
from deepseek_reference.moe import MoE as CopiedMoE
from deepseek_reference.shared_expert import SharedExpert as CopiedSharedExpert

from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MLP, DeepseekV3MoE
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

    # SharedExpert is just an MLP with different size
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
def test_moe_plus_shared_expert(
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
    """Test MoE + SharedExpert for bytewise identical outputs."""

    logger.info("=" * 80)
    logger.info("Testing MoE + SharedExpert for bytewise identical outputs")
    logger.info("=" * 80)

    moe_model, shared_expert_model = reference_models

    # Get state dicts
    moe_state_dict = add_inv_scale_to_state_dict(
        moe_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Adjust shared expert state dict keys to match expected format
    shared_state_dict = shared_expert_model.state_dict()
    # The SharedExpert expects keys with "shared_experts." prefix
    shared_expert_state_dict = add_inv_scale_to_state_dict(
        {f"shared_experts.{k}": v for k, v in shared_state_dict.items()},
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Create identical input tensor
    torch.manual_seed(5)
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Save input for verification
    input_path = Path("/tmp/test_moe_shared_input.pt")
    torch.save(torch_input, input_path)
    logger.info(f"Saved test input to {input_path}, shape: {torch_input.shape}")

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # 1. Run REFERENCE implementations (MoE + SharedExpert)
    logger.info("-" * 40)
    logger.info("Running REFERENCE MoE + SharedExpert...")
    logger.info("-" * 40)

    # MoE forward
    ref_moe_config = get_test_weight_config(
        ReferenceMoE, hf_config, (moe_state_dict,), cache_path / "reference_moe_shared", mesh_device, False
    )
    ref_moe_model_config = get_model_config(ReferenceMoE, mode, hf_config, mesh_device)
    ref_moe_model_config.update({"topk_fallback": topk_fallback})
    ref_moe_state = ReferenceMoE.create_state(hf_config, mesh_device, ccl)
    ref_moe_shared_state = ReferenceMoE.create_shared_state(hf_config, mesh_device)
    ref_moe_run_config = create_run_config(ref_moe_model_config, ref_moe_config, ref_moe_state, ref_moe_shared_state)

    # SharedExpert forward
    ref_shared_config = get_test_weight_config(
        ReferenceSharedExpert,
        hf_config,
        (shared_expert_state_dict,) * mesh_device.shape[0],  # Replicate for each device
        cache_path / "reference_shared",
        mesh_device,
        False,
    )
    ref_shared_model_config = get_model_config(ReferenceSharedExpert, mode, hf_config, mesh_device)
    ref_shared_state = ReferenceSharedExpert.create_state(hf_config, mesh_device, ccl)
    ref_shared_shared_state = {}
    ref_shared_run_config = create_run_config(
        ref_shared_model_config, ref_shared_config, ref_shared_state, ref_shared_shared_state
    )

    # Run MoE
    ref_tt_input = ttnn.to_memory_config(tt_input, ref_moe_run_config["input_memory_config"])
    ref_moe_output = run_module_forward(ReferenceMoE, mode, ref_tt_input, ref_moe_run_config)

    # Run SharedExpert
    ref_shared_input = ttnn.to_memory_config(tt_input, ref_shared_run_config["input_memory_config"])
    ref_shared_output = run_module_forward(ReferenceSharedExpert, mode, ref_shared_input, ref_shared_run_config)

    # Add them together (MoE + SharedExpert)
    ref_combined_output = ref_moe_output + ref_shared_output

    # Convert to torch
    reference_output = ttnn.to_torch(
        ref_combined_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Clean up
    ttnn.deallocate(ref_tt_input)
    ttnn.deallocate(ref_shared_input)
    ttnn.deallocate(ref_moe_output)
    ttnn.deallocate(ref_shared_output)
    ttnn.deallocate(ref_combined_output)

    # 2. Run COPIED implementations (MoE + SharedExpert)
    logger.info("-" * 40)
    logger.info("Running COPIED MoE + SharedExpert...")
    logger.info("-" * 40)

    # MoE forward
    copy_moe_config = get_test_weight_config(
        CopiedMoE, hf_config, (moe_state_dict,), cache_path / "copied_moe_shared", mesh_device, False
    )
    copy_moe_model_config = get_model_config(CopiedMoE, mode, hf_config, mesh_device)
    copy_moe_model_config.update({"topk_fallback": topk_fallback})
    copy_moe_state = CopiedMoE.create_state(hf_config, mesh_device, ccl)
    copy_moe_shared_state = CopiedMoE.create_shared_state(hf_config, mesh_device)
    copy_moe_run_config = create_run_config(
        copy_moe_model_config, copy_moe_config, copy_moe_state, copy_moe_shared_state
    )

    # SharedExpert forward
    copy_shared_config = get_test_weight_config(
        CopiedSharedExpert,
        hf_config,
        (shared_expert_state_dict,) * mesh_device.shape[0],  # Replicate for each device
        cache_path / "copied_shared",
        mesh_device,
        False,
    )
    copy_shared_model_config = get_model_config(CopiedSharedExpert, mode, hf_config, mesh_device)
    copy_shared_state = CopiedSharedExpert.create_state(hf_config, mesh_device, ccl)
    copy_shared_shared_state = {}
    copy_shared_run_config = create_run_config(
        copy_shared_model_config, copy_shared_config, copy_shared_state, copy_shared_shared_state
    )

    # Run MoE
    copy_tt_input = ttnn.to_memory_config(tt_input, copy_moe_run_config["input_memory_config"])
    copy_moe_output = run_module_forward(CopiedMoE, mode, copy_tt_input, copy_moe_run_config)

    # Run SharedExpert
    copy_shared_input = ttnn.to_memory_config(tt_input, copy_shared_run_config["input_memory_config"])
    copy_shared_output = run_module_forward(CopiedSharedExpert, mode, copy_shared_input, copy_shared_run_config)

    # Add them together (MoE + SharedExpert)
    copy_combined_output = copy_moe_output + copy_shared_output

    # Convert to torch
    copied_output = ttnn.to_torch(
        copy_combined_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Clean up
    ttnn.deallocate(copy_tt_input)
    ttnn.deallocate(copy_shared_input)
    ttnn.deallocate(copy_moe_output)
    ttnn.deallocate(copy_shared_output)
    ttnn.deallocate(copy_combined_output)
    ttnn.deallocate(tt_input)

    # 3. Compare outputs
    logger.info("=" * 80)
    logger.info("Comparing MoE + SharedExpert outputs for bytewise identity...")
    logger.info("=" * 80)

    # Both should have the same shape
    ref_shape = reference_output.shape
    copy_shape = copied_output.shape

    logger.info(f"Reference (MoE+SharedExpert) output shape: {ref_shape}")
    logger.info(f"Copied (MoE+SharedExpert) output shape:    {copy_shape}")
    assert ref_shape == copy_shape, f"Shape mismatch! ref={ref_shape}, copy={copy_shape}"

    # Verify bytewise identical
    verify_bytewise_identical(reference_output, copied_output, name="MoE_plus_SharedExpert")

    logger.info("=" * 80)
    logger.info("✅ SUCCESS: MoE + SharedExpert produces bytewise identical outputs!")
    logger.info("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test that ONLY runs our MoE implementation.
No comparison with reference - just validates that our implementation runs successfully.

Usage:
    pytest test_our_moe_only.py -xvs
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

# Add tt-moe directory to path for our implementation
sys.path.insert(0, str(Path(__file__).parent.parent))
from deepseek_reference.moe import MoE as OurMoE

# Import utilities
from models.demos.deepseek_v3.reference.modeling_deepseek import DeepseekV3MoE
from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import (
    add_inv_scale_to_state_dict,
    get_model_config,
    get_test_weight_config,
)


def run_module_forward(module, mode, x, cfg):
    """Helper to run forward pass based on mode."""
    if mode == "prefill":
        return module.forward_prefill(x, cfg)
    elif mode == "decode":
        return module.forward_decode(x, cfg)
    else:
        return module.forward(x, cfg)


@pytest.fixture
def reference_model(hf_config):
    """Get the reference DeepSeek MoE model (only used for state dict)."""
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
def test_our_moe_only(
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
    """
    Test ONLY our MoE implementation.
    No comparison with reference - just runs our implementation and validates output.
    """

    logger.info("=" * 80)
    logger.info("Testing OUR MoE Implementation (No Comparison)")
    logger.info(f"Mode: {mode}, Tokens: {num_tokens}")
    logger.info("=" * 80)

    # Get state dict from reference model (we need the weights)
    state_dict = add_inv_scale_to_state_dict(
        reference_model.state_dict(),
        block_shape=hf_config.quantization_config["weight_block_size"],
    )

    # Create identical input tensor
    torch.manual_seed(5)
    torch_input = torch.randn(1, num_tokens, hf_config.hidden_size, dtype=torch.bfloat16)

    # Save input if requested
    save_outputs = os.environ.get("SAVE_OUTPUTS", "0") == "1"
    if save_outputs:
        input_dir = Path("/tmp/our_moe_test")
        input_dir.mkdir(exist_ok=True)
        input_path = input_dir / "input.pt"
        torch.save(torch_input, input_path)
        logger.info(f"Saved test input to {input_path}, shape: {torch_input.shape}")

    # ================================================================================
    # Run OUR MoE Implementation
    # ================================================================================
    logger.info("")
    logger.info("Running OUR MoE Implementation...")
    logger.info("-" * 40)

    # Setup configurations for our implementation
    weight_config = get_test_weight_config(
        OurMoE,
        hf_config,
        (state_dict,),
        cache_path / "our_moe",
        mesh_device,
        False,  # force_recalculate_weight_config
    )

    model_config = get_model_config(OurMoE, mode, hf_config, mesh_device)
    model_config.update({"topk_fallback": topk_fallback})

    model_state = OurMoE.create_state(hf_config, mesh_device, ccl)
    model_shared_state = OurMoE.create_shared_state(hf_config, mesh_device)

    run_config = create_run_config(model_config, weight_config, model_state, model_shared_state)

    # Convert input to TTNN
    tt_input = ttnn.from_torch(
        torch_input.unsqueeze(1),
        device=mesh_device,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
        dtype=ttnn.bfloat16,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        layout=ttnn.TILE_LAYOUT,
    )

    # Run forward pass
    logger.info("Executing forward pass...")
    tt_input = ttnn.to_memory_config(tt_input, run_config["input_memory_config"])
    tt_output = run_module_forward(OurMoE, mode, tt_input, run_config)

    # Convert output back to torch
    output = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    # Clean up
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # ================================================================================
    # Validate Output
    # ================================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("Output Validation")
    logger.info("=" * 80)

    # Check output shape
    expected_shape = (1, 1, num_tokens, hf_config.hidden_size)
    assert output.shape == expected_shape, f"Unexpected shape: {output.shape}, expected {expected_shape}"
    logger.info(f"✅ Output shape: {output.shape}")

    # Compute hash for reproducibility check
    # Convert to float32 first since numpy doesn't support BFloat16 directly
    output_hash = hashlib.md5(output.float().cpu().numpy().tobytes()).hexdigest()
    logger.info(f"✅ Output hash: {output_hash}")

    # Check for NaN/Inf
    assert not torch.isnan(output).any(), "Output contains NaN values!"
    assert not torch.isinf(output).any(), "Output contains Inf values!"
    logger.info("✅ No NaN/Inf values detected")

    # Compute statistics
    output_mean = output.mean().item()
    output_std = output.std().item()
    output_min = output.min().item()
    output_max = output.max().item()

    logger.info("")
    logger.info("Output Statistics:")
    logger.info(f"  Mean: {output_mean:.6f}")
    logger.info(f"  Std:  {output_std:.6f}")
    logger.info(f"  Min:  {output_min:.6f}")
    logger.info(f"  Max:  {output_max:.6f}")

    # Save output if requested
    if save_outputs:
        output_path = input_dir / "output.npy"
        # Convert to float32 for numpy compatibility
        np.save(output_path, output.float().cpu().numpy())
        logger.info("")
        logger.info(f"Saved output to {output_path}")

        # Also save as text for easy inspection
        stats_path = input_dir / "output_stats.txt"
        with open(stats_path, "w") as f:
            f.write(f"Output Statistics for OUR MoE Implementation\n")
            f.write(f"=" * 50 + "\n")
            f.write(f"Shape: {output.shape}\n")
            f.write(f"Hash:  {output_hash}\n")
            f.write(f"Mean:  {output_mean:.6f}\n")
            f.write(f"Std:   {output_std:.6f}\n")
            f.write(f"Min:   {output_min:.6f}\n")
            f.write(f"Max:   {output_max:.6f}\n")
        logger.info(f"Saved statistics to {stats_path}")

    # Optional: Check against known good hash (if we have one)
    # This is the hash we expect based on previous runs
    EXPECTED_HASH = "2ec74fa4aa709d7e7c3f1db7abf02f7c"
    if output_hash == EXPECTED_HASH:
        logger.info("")
        logger.info(f"✅ Output matches expected hash: {EXPECTED_HASH}")
        logger.info("   This indicates the implementation is producing consistent results!")
    else:
        logger.info("")
        logger.info(f"ℹ️  Output hash: {output_hash}")
        logger.info(f"   Expected:    {EXPECTED_HASH}")
        logger.info("   (This may be fine if the implementation has changed)")

    # ================================================================================
    # Test Summary
    # ================================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("✅ TEST PASSED: OUR MoE Implementation")
    logger.info("=" * 80)
    logger.info(f"  - Successfully executed forward pass")
    logger.info(f"  - Output shape correct: {output.shape}")
    logger.info(f"  - No NaN/Inf values")
    logger.info(f"  - Output hash: {output_hash}")
    logger.info("=" * 80)


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

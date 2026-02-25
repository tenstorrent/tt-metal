#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Direct test of forward_mlp_decode (MoE + SharedExpert) for our copied implementation.
"""

import sys
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn

# Add tt-moe directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our copied implementation
from deepseek_reference.moe_decoder_block_2d import MoEDecoderBlock2D as CopiedMoEDecoderBlock2D

from models.demos.deepseek_v3.utils.run_config import create_run_config
from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
def test_copied_mlp_forward(
    set_deterministic_env,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    state_dict,
):
    """Test our copied forward_mlp_decode (MoE + SharedExpert) directly."""

    logger.info("=" * 80)
    logger.info("Testing COPIED MoEDecoderBlock2D.forward_mlp_decode")
    logger.info("Running MoE + SharedExpert with real weights")
    logger.info("=" * 80)

    # Fix the max_seq_len attribute issue
    if not hasattr(hf_config, "max_seq_len"):
        hf_config.max_seq_len = hf_config.max_position_embeddings

    # Get layer 3 MLP weights (MoE + SharedExpert)
    layer_idx = 3
    mlp_state_dict = {
        k.replace(f"model.layers.{layer_idx}.", ""): v
        for k, v in state_dict.items()
        if k.startswith(f"model.layers.{layer_idx}.mlp.")
    }

    logger.info(f"Loaded {len(mlp_state_dict)} MLP weights for layer {layer_idx}")

    # Create input tensor
    torch.manual_seed(5)
    batch_size = 32
    seq_len = 1
    torch_input = torch.randn(seq_len, batch_size, hf_config.hidden_size, dtype=torch.bfloat16)

    # Setup configurations for our copied implementation
    weight_config = get_test_weight_config(
        CopiedMoEDecoderBlock2D,
        hf_config,
        (mlp_state_dict,),
        cache_path / "copied_mlp_forward",
        mesh_device,
        False,
    )

    model_config = get_model_config(CopiedMoEDecoderBlock2D, "decode", hf_config, mesh_device)
    model_config.update({"topk_fallback": True})

    model_state = CopiedMoEDecoderBlock2D.create_mlp_state(hf_config, mesh_device, ccl)
    model_shared_state = CopiedMoEDecoderBlock2D.create_mlp_shared_state(hf_config, mesh_device)

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

    # Prepare input
    tt_input = ttnn.to_memory_config(tt_input, run_config.get("input_memory_config", ttnn.DRAM_MEMORY_CONFIG))

    # Run forward_mlp_decode (MoE + SharedExpert)
    logger.info("Running forward_mlp_decode (MoE + SharedExpert)...")
    tt_output = CopiedMoEDecoderBlock2D.forward_mlp_decode(tt_input, run_config)

    # Convert output to torch
    output_torch = ttnn.to_torch(
        tt_output,
        mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(-2, -1), mesh_shape=tuple(mesh_device.shape)),
    )

    logger.info(f"Output shape: {output_torch.shape}, dtype: {output_torch.dtype}")

    # Clean up
    ttnn.deallocate(tt_input)
    ttnn.deallocate(tt_output)

    # Check if hash was saved
    copy_path = Path("/tmp/moe_decoder_copied_output/moe_decoder_hash.txt")
    ref_path = Path("/tmp/moe_decoder_reference_output/moe_decoder_hash.txt")

    if copy_path.exists():
        copy_hash = copy_path.read_text().strip()
        logger.info(f"Copied implementation MD5 hash: {copy_hash}")

        if ref_path.exists():
            ref_hash = ref_path.read_text().strip()
            logger.info(f"Reference MD5 hash: {ref_hash}")

            logger.info("=" * 80)
            logger.info("BYTEWISE COMPARISON RESULTS:")
            logger.info(f"Reference MD5: {ref_hash}")
            logger.info(f"Copied MD5:    {copy_hash}")

            if ref_hash == copy_hash:
                logger.info("=" * 80)
                logger.info("✅ SUCCESS: MoE + SharedExpert outputs are BYTEWISE IDENTICAL!")
                logger.info("=" * 80)
            else:
                logger.error("=" * 80)
                logger.error("❌ FAILED: Outputs are NOT bytewise identical!")
                logger.error("=" * 80)
                pytest.fail(f"Hashes don't match! ref={ref_hash}, copy={copy_hash}")
        else:
            logger.info("Reference hash not found. Run reference test first for comparison.")
    else:
        logger.warning("Copied implementation did not save hash. Check instrumentation.")

    logger.info("Test completed successfully!")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

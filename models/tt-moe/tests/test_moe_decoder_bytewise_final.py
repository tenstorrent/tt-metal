#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Final bytewise comparison test for MoEDecoderBlock2D (MoE + SharedExpert)
This test uses the exact same setup as the reference test_decoder_block.py
"""

import sys
from pathlib import Path

import pytest
from loguru import logger

# Add tt-moe directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our copied implementation
from deepseek_reference.moe_decoder_block_2d import MoEDecoderBlock2D as CopiedMoEDecoderBlock2D

from models.demos.deepseek_v3.tests.test_decoder_block import run_test_forward_pass_decoder2d


def test_moe_decoder_bytewise_verification(
    set_deterministic_env,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    state_dict,
):
    """Test that our copied MoEDecoderBlock2D produces bytewise identical outputs."""

    logger.info("=" * 80)
    logger.info("Testing Copied MoEDecoderBlock2D for bytewise identity")
    logger.info("=" * 80)

    # Run the test with our copied implementation
    # Using the exact same parameters as the reference test
    run_test_forward_pass_decoder2d(
        DecoderBlockClass=CopiedMoEDecoderBlock2D,
        module_path="model.layers.3",  # Layer 3 weights
        reference_layer_idx=3,
        mode="decode",
        seq_len=1,
        batch_size_per_row=32,
        hf_config_short=hf_config,
        cache_path=cache_path,
        mesh_device=mesh_device,
        model_path=None,  # Not needed, we use state_dict
        ccl=ccl,
        force_recalculate_weight_config=False,
        state_dict=state_dict,
        decode_position_ids=None,  # Random positions
    )

    # Check if both outputs were saved
    ref_path = Path("/tmp/moe_decoder_reference_output/moe_decoder_hash.txt")
    copy_path = Path("/tmp/moe_decoder_copied_output/moe_decoder_hash.txt")

    if ref_path.exists() and copy_path.exists():
        ref_hash = ref_path.read_text().strip()
        copy_hash = copy_path.read_text().strip()

        logger.info("=" * 80)
        logger.info("BYTEWISE COMPARISON RESULTS:")
        logger.info(f"Reference MD5: {ref_hash}")
        logger.info(f"Copied MD5:    {copy_hash}")

        if ref_hash == copy_hash:
            logger.info("=" * 80)
            logger.info("✅ SUCCESS: MoEDecoderBlock2D outputs are BYTEWISE IDENTICAL!")
            logger.info("=" * 80)
        else:
            logger.info("=" * 80)
            logger.error("❌ FAILED: Outputs are NOT bytewise identical!")
            logger.info("=" * 80)
            pytest.fail(f"MD5 hashes do not match! ref={ref_hash}, copy={copy_hash}")
    else:
        pytest.fail(f"Could not find saved hashes. ref={ref_path.exists()}, copy={copy_path.exists()}")


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])

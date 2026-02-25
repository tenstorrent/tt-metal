#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
Test our copied MoEDecoderBlock2D implementation and compare hash with reference.
"""

import sys
from pathlib import Path

# Add tt-moe directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our copied implementation
from deepseek_reference.moe_decoder_block_2d import MoEDecoderBlock2D as CopiedMoEDecoderBlock2D

# Import the test runner from reference
from models.demos.deepseek_v3.tests.test_decoder_block import run_test_forward_pass_decoder2d


def test_copied_implementation(
    set_deterministic_env,
    hf_config,
    cache_path,
    mesh_device,
    ccl,
    state_dict,
):
    """Run our copied MoEDecoderBlock2D and save output for comparison."""

    from loguru import logger

    logger.info("=" * 80)
    logger.info("Running COPIED MoEDecoderBlock2D (MoE + SharedExpert) with real weights")
    logger.info("=" * 80)

    # Fix the max_seq_len attribute issue
    if not hasattr(hf_config, "max_seq_len"):
        hf_config.max_seq_len = hf_config.max_position_embeddings

    # Run with exact same parameters as reference test
    run_test_forward_pass_decoder2d(
        DecoderBlockClass=CopiedMoEDecoderBlock2D,
        module_path="model.layers.3",
        reference_layer_idx=3,
        mode="decode",
        seq_len=1,
        batch_size_per_row=32,
        hf_config_short=hf_config,
        cache_path=cache_path,
        mesh_device=mesh_device,
        model_path=None,
        ccl=ccl,
        force_recalculate_weight_config=False,
        state_dict=state_dict,
        decode_position_ids=None,
    )

    # Compare hashes
    ref_path = Path("/tmp/moe_decoder_reference_output/moe_decoder_hash.txt")
    copy_path = Path("/tmp/moe_decoder_copied_output/moe_decoder_hash.txt")

    logger.info("=" * 80)
    logger.info("BYTEWISE COMPARISON RESULTS:")
    logger.info("=" * 80)

    if ref_path.exists() and copy_path.exists():
        ref_hash = ref_path.read_text().strip()
        copy_hash = copy_path.read_text().strip()

        logger.info(f"Reference MD5: {ref_hash}")
        logger.info(f"Copied MD5:    {copy_hash}")

        if ref_hash == copy_hash:
            logger.info("=" * 80)
            logger.info("✅ SUCCESS: MoEDecoderBlock2D outputs are BYTEWISE IDENTICAL!")
            logger.info("=" * 80)
            return True
        else:
            logger.error("=" * 80)
            logger.error("❌ FAILED: Outputs are NOT bytewise identical!")
            logger.error("=" * 80)
            import pytest

            pytest.fail(f"Hashes don't match! ref={ref_hash}, copy={copy_hash}")
    else:
        import pytest

        pytest.fail(f"Hash files missing. ref={ref_path.exists()}, copy={copy_path.exists()}")


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-xvs"])

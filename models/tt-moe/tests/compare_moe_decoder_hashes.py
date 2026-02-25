#!/usr/bin/env python3
"""
Compare MD5 hashes between reference and copied MoEDecoderBlock2D implementations.
"""

import hashlib
from pathlib import Path

import numpy as np
from loguru import logger


def load_and_hash(npy_path):
    """Load numpy array and compute MD5 hash."""
    if not npy_path.exists():
        return None

    arr = np.load(npy_path)
    arr_bytes = arr.tobytes()
    hash_value = hashlib.md5(arr_bytes).hexdigest()
    return hash_value, arr.shape


def compare_implementations():
    """Compare reference and copied MoEDecoderBlock2D outputs."""

    logger.info("=" * 80)
    logger.info("Comparing MoEDecoderBlock2D (MoE + SharedExpert) Outputs")
    logger.info("=" * 80)

    # Reference output
    ref_dir = Path("/tmp/moe_decoder_reference_output")
    ref_npy = ref_dir / "moe_decoder_output.npy"
    ref_hash_file = ref_dir / "moe_decoder_hash.txt"

    # Copied output
    copy_dir = Path("/tmp/moe_decoder_copied_output")
    copy_npy = copy_dir / "moe_decoder_output.npy"
    copy_hash_file = copy_dir / "moe_decoder_hash.txt"

    # Load reference
    if ref_npy.exists():
        ref_hash, ref_shape = load_and_hash(ref_npy)
        logger.info(f"Reference output found:")
        logger.info(f"  Path: {ref_npy}")
        logger.info(f"  Shape: {ref_shape}")
        logger.info(f"  MD5 Hash: {ref_hash}")
    else:
        logger.error(f"Reference output not found at {ref_npy}")
        return False

    # Load copied
    if copy_npy.exists():
        copy_hash, copy_shape = load_and_hash(copy_npy)
        logger.info(f"Copied output found:")
        logger.info(f"  Path: {copy_npy}")
        logger.info(f"  Shape: {copy_shape}")
        logger.info(f"  MD5 Hash: {copy_hash}")
    else:
        logger.error(f"Copied output not found at {copy_npy}")
        return False

    # Compare
    logger.info("-" * 80)
    logger.info("COMPARISON RESULTS:")
    logger.info(f"Reference MD5: {ref_hash}")
    logger.info(f"Copied MD5:    {copy_hash}")

    if ref_hash == copy_hash:
        logger.info("=" * 80)
        logger.info("✅ SUCCESS: MoEDecoderBlock2D outputs are BYTEWISE IDENTICAL!")
        logger.info("=" * 80)
        return True
    else:
        # Load arrays for detailed comparison
        ref_arr = np.load(ref_npy)
        copy_arr = np.load(copy_npy)

        if ref_shape != copy_shape:
            logger.error(f"❌ Shape mismatch! ref={ref_shape}, copy={copy_shape}")
        else:
            diff_mask = ref_arr != copy_arr
            if diff_mask.any():
                logger.error(f"❌ Arrays differ at {diff_mask.sum()} positions")
                logger.error(f"❌ Max absolute difference: {np.abs(ref_arr - copy_arr).max()}")
            else:
                logger.warning("❌ Arrays are equal but hashes differ (floating point precision?)")

        logger.info("=" * 80)
        logger.error("❌ FAILED: MoEDecoderBlock2D outputs are NOT bytewise identical!")
        logger.info("=" * 80)
        return False


if __name__ == "__main__":
    import sys

    success = compare_implementations()
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Test script to verify parity between reference and copied MoE implementations.
Compares MD5 hashes of outputs to ensure bytewise identical results.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))  # tt-moe directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # tt-metal directory


def compute_hash(tensor):
    """Compute MD5 hash from tensor."""
    if isinstance(tensor, torch.Tensor):
        arr = tensor.cpu().float().numpy()
    else:
        arr = tensor

    # Ensure float32 for consistent hashing
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32)

    return hashlib.md5(arr.tobytes()).hexdigest()


def test_moe_implementations():
    """Test both MoE implementations and compare outputs."""

    logger.info("=" * 80)
    logger.info("Testing MoE Implementation Parity")
    logger.info("=" * 80)

    try:
        # Import both implementations
        from deepseek_reference.moe import MoE as CopiedMoE

        from models.demos.deepseek_v3.tt.moe import MoE as ReferenceMoE

        logger.info("✓ Successfully imported both MoE implementations")

        # Import helper functions
        from models.demos.deepseek_v3.utils.run_config import create_run_config
        from models.demos.deepseek_v3.utils.test_utils import get_model_config, get_test_weight_config

        # Setup test configuration
        layer_idx = 3
        model_config = get_model_config()
        mesh_device = None  # Will work with CPU tensors for hash comparison

        # Create minimal run config
        run_config = create_run_config(
            batch_size=32,
            seq_len=128,
            layer_start=layer_idx,
            layer_end=layer_idx + 1,
            mode="decode",
            shard_cache=False,
            num_users=1,
        )

        # Get weight config
        weight_config = get_test_weight_config(run_config)

        # Create test input
        torch.manual_seed(42)  # For reproducibility
        hidden_dim = model_config.hidden_size
        batch_size = 32
        seq_len = 1  # Decode mode

        test_input = torch.randn(1, batch_size, seq_len, hidden_dim, dtype=torch.bfloat16)

        logger.info(f"Test input shape: {test_input.shape}")

        # Test Reference MoE
        logger.info("\nTesting Reference MoE...")
        reference_moe = ReferenceMoE.from_state_dict(
            state_dict={},  # Empty for testing
            device=None,
            model_config=model_config,
            weight_config=weight_config,
            layer_idx=layer_idx,
            run_config=run_config,
            mesh_device=mesh_device,
        )

        # Note: We'll need to mock the forward pass since it requires TT hardware
        # For now, let's just verify the configurations match

        logger.info("Reference MoE configuration:")
        logger.info(
            f"  - num_links in all_to_all_dispatch: {reference_moe.ccl_config.get('all_to_all_dispatch', {}).get('num_links', 'Not set')}"
        )
        logger.info(
            f"  - num_links in all_to_all_combine: {reference_moe.ccl_config.get('all_to_all_combine', {}).get('num_links', 'Not set')}"
        )

        # Test Copied MoE
        logger.info("\nTesting Copied MoE...")
        copied_moe = CopiedMoE.from_state_dict(
            state_dict={},  # Empty for testing
            device=None,
            model_config=model_config,
            weight_config=weight_config,
            layer_idx=layer_idx,
            run_config=run_config,
            mesh_device=mesh_device,
        )

        logger.info("Copied MoE configuration:")
        logger.info(
            f"  - num_links in all_to_all_dispatch: {copied_moe.ccl_config.get('all_to_all_dispatch', {}).get('num_links', 'Not set')}"
        )
        logger.info(
            f"  - num_links in all_to_all_combine: {copied_moe.ccl_config.get('all_to_all_combine', {}).get('num_links', 'Not set')}"
        )

        # Compare configurations
        logger.info("\n" + "=" * 80)
        logger.info("Configuration Comparison Results:")

        ref_dispatch_links = reference_moe.ccl_config.get("all_to_all_dispatch", {}).get("num_links", None)
        copy_dispatch_links = copied_moe.ccl_config.get("all_to_all_dispatch", {}).get("num_links", None)

        ref_combine_links = reference_moe.ccl_config.get("all_to_all_combine", {}).get("num_links", None)
        copy_combine_links = copied_moe.ccl_config.get("all_to_all_combine", {}).get("num_links", None)

        config_match = ref_dispatch_links == copy_dispatch_links and ref_combine_links == copy_combine_links

        if config_match:
            logger.info("✅ SUCCESS: CCL configurations match!")
            logger.info(f"   Both use num_links={ref_dispatch_links} for dispatch and combine")
        else:
            logger.info("❌ FAILED: CCL configurations differ!")
            logger.info(f"   Reference: dispatch={ref_dispatch_links}, combine={ref_combine_links}")
            logger.info(f"   Copied:    dispatch={copy_dispatch_links}, combine={copy_combine_links}")

        logger.info("=" * 80)

        return config_match

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("\nNote: This test requires the TT-Metal environment to be properly set up.")
        return False
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False


def check_existing_outputs():
    """Check for existing test outputs and compare hashes."""

    logger.info("\n" + "=" * 80)
    logger.info("Checking for existing test outputs...")
    logger.info("=" * 80)

    # Check various output directories
    output_dirs = [
        ("/tmp/moe_reference_output", "/tmp/moe_copied_output", "moe_output.npy"),
        ("/tmp/moe_decoder_reference_output", "/tmp/moe_decoder_copied_output", "moe_decoder_output.npy"),
    ]

    for ref_dir, copy_dir, filename in output_dirs:
        ref_path = Path(ref_dir) / filename
        copy_path = Path(copy_dir) / filename

        if ref_path.exists() and copy_path.exists():
            logger.info(f"\nFound outputs in {ref_dir} and {copy_dir}")

            ref_arr = np.load(ref_path)
            copy_arr = np.load(copy_path)

            ref_hash = compute_hash(ref_arr)
            copy_hash = compute_hash(copy_arr)

            logger.info(f"Reference hash: {ref_hash}")
            logger.info(f"Copied hash:    {copy_hash}")

            if ref_hash == copy_hash:
                logger.info("✅ Outputs are BYTEWISE IDENTICAL!")
            else:
                logger.info("❌ Outputs differ!")

                # Check shapes
                if ref_arr.shape != copy_arr.shape:
                    logger.info(f"   Shape mismatch: {ref_arr.shape} vs {copy_arr.shape}")
                else:
                    # Check numerical difference
                    diff = np.abs(ref_arr - copy_arr)
                    logger.info(f"   Max difference: {np.max(diff)}")
                    logger.info(f"   Mean difference: {np.mean(diff)}")

                    # Check percentage of elements that differ
                    num_diff = np.sum(diff > 1e-7)
                    pct_diff = (num_diff / ref_arr.size) * 100
                    logger.info(f"   Elements that differ: {num_diff}/{ref_arr.size} ({pct_diff:.2f}%)")


if __name__ == "__main__":
    # Run configuration comparison
    success = test_moe_implementations()

    # Check for existing outputs
    check_existing_outputs()

    if success:
        logger.info("\n✅ Configuration parity test PASSED")
    else:
        logger.info("\n❌ Configuration parity test FAILED")
        sys.exit(1)

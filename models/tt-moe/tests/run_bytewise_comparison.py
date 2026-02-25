#!/usr/bin/env python3
"""
Simple script to run reference and copied implementations and compare MD5 hashes.
"""

import subprocess
import sys
from pathlib import Path


def run_reference():
    """Run the reference MoEDecoderBlock2D test."""
    print("=" * 80)
    print("Running REFERENCE MoEDecoderBlock2D...")
    print("=" * 80)

    cmd = """
    source python_env/bin/activate && \
    export MESH_DEVICE=TG && \
    export PYTHONPATH=$PWD && \
    export TT_METAL_HOME=$PWD && \
    export DEEPSEEK_V3_HF_MODEL=/data/MLPerf/huggingface/hub/models--deepseek-ai--DeepSeek-R1-0528/snapshots/4236a6af538feda4548eca9ab308586007567f52 && \
    export DEEPSEEK_V3_CACHE=/tmp/deepseek_cache && \
    export SAVE_MOE_DECODER_OUTPUT=1 && \
    rm -rf /tmp/moe_decoder_reference_output && \
    pytest "models/demos/deepseek_v3/tests/test_decoder_block.py::test_forward_pass[mode_decode_seq_1_batch_32_pos_random-MoEDecoderBlock2D-model.layers.3-3-run_test_forward_pass_decoder2d-device_params0]" -xvs 2>&1 | grep -E "MD5 hash|PCC|PASSED"
    """

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)

    # Get the hash
    hash_file = Path("/tmp/moe_decoder_reference_output/moe_decoder_hash.txt")
    if hash_file.exists():
        ref_hash = hash_file.read_text().strip()
        print(f"Reference MD5 hash: {ref_hash}")
        return ref_hash
    else:
        print("ERROR: Reference hash file not found!")
        return None


def compare_hashes():
    """Compare the saved hashes."""
    ref_file = Path("/tmp/moe_decoder_reference_output/moe_decoder_hash.txt")
    copy_file = Path("/tmp/moe_decoder_copied_output/moe_decoder_hash.txt")

    print("\n" + "=" * 80)
    print("BYTEWISE COMPARISON RESULTS:")
    print("=" * 80)

    if ref_file.exists() and copy_file.exists():
        ref_hash = ref_file.read_text().strip()
        copy_hash = copy_file.read_text().strip()

        print(f"Reference MD5: {ref_hash}")
        print(f"Copied MD5:    {copy_hash}")

        if ref_hash == copy_hash:
            print("=" * 80)
            print("✅ SUCCESS: MoEDecoderBlock2D (MoE + SharedExpert) outputs are BYTEWISE IDENTICAL!")
            print("=" * 80)
            return True
        else:
            print("=" * 80)
            print("❌ FAILED: Outputs are NOT bytewise identical!")
            print("=" * 80)
            return False
    else:
        print(f"ERROR: Hash files not found. ref={ref_file.exists()}, copy={copy_file.exists()}")
        return False


if __name__ == "__main__":
    # Run reference implementation
    ref_hash = run_reference()

    if ref_hash:
        print(f"\nReference implementation complete. Hash: {ref_hash}")
        print("\nNOTE: To complete the comparison, you need to:")
        print("1. Update the copied implementation imports if needed")
        print("2. Run the copied implementation test")
        print("3. Compare the hashes")

        # Check if we already have a copied hash to compare
        if Path("/tmp/moe_decoder_copied_output/moe_decoder_hash.txt").exists():
            print("\nFound existing copied implementation output. Comparing...")
            compare_hashes()
        else:
            print("\nNo copied implementation output found yet.")
            print("The reference hash has been saved for comparison.")
    else:
        print("\nERROR: Failed to get reference hash!")
        sys.exit(1)

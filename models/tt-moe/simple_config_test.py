#!/usr/bin/env python3
"""
Simple test to compare configurations between reference and copied MoE implementations.
"""

import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent))  # tt-moe directory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # tt-metal directory


def test_ccl_config():
    """Test CCL configuration parity."""

    print("=" * 80)
    print("Testing CCL Configuration Parity")
    print("=" * 80)

    # Import CCL classes
    from deepseek_reference.ccl import CCL as CopiedCCL

    from models.demos.deepseek_v3.tt.ccl import CCL as ReferenceCCL

    # Create instances
    # Using dummy values for mesh shape
    mesh_shape = (8, 4)  # Example TG mesh

    ref_ccl = ReferenceCCL(mesh_shape)
    copy_ccl = CopiedCCL(mesh_shape)

    print("\nTesting get_max_links() method:")
    print("-" * 40)

    for axis in [0, 1]:
        ref_links = ref_ccl.get_max_links(axis)
        copy_links = copy_ccl.get_max_links(axis)

        print(f"Axis {axis}:")
        print(f"  Reference: {ref_links}")
        print(f"  Copied:    {copy_links}")

        if ref_links == copy_links:
            print(f"  ✓ Match!")
        else:
            print(f"  ✗ Mismatch!")

    print("\n" + "=" * 80)


def test_moe_ccl_config():
    """Test MoE CCL configuration."""

    print("\nTesting MoE CCL Configuration")
    print("=" * 80)

    try:
        # Import MoE classes
        # Import CCL
        from deepseek_reference.ccl import CCL as CopiedCCL
        from deepseek_reference.moe import MoE as CopiedMoE

        from models.demos.deepseek_v3.tt.moe import MoE as ReferenceMoE

        # Create CCL instance
        mesh_shape = (8, 4)
        ccl = CopiedCCL(mesh_shape)

        # Get CCL config from both implementations
        ref_config = ReferenceMoE.create_ccl_config(ccl)
        copy_config = CopiedMoE.create_ccl_config(ccl)

        print("\nReference MoE CCL config:")
        print(
            f"  all_to_all_dispatch num_links: {ref_config.get('all_to_all_dispatch', {}).get('num_links', 'Not set')}"
        )
        print(f"  all_to_all_combine num_links: {ref_config.get('all_to_all_combine', {}).get('num_links', 'Not set')}")

        print("\nCopied MoE CCL config:")
        print(
            f"  all_to_all_dispatch num_links: {copy_config.get('all_to_all_dispatch', {}).get('num_links', 'Not set')}"
        )
        print(
            f"  all_to_all_combine num_links: {copy_config.get('all_to_all_combine', {}).get('num_links', 'Not set')}"
        )

        # Compare
        dispatch_match = ref_config.get("all_to_all_dispatch", {}).get("num_links") == copy_config.get(
            "all_to_all_dispatch", {}
        ).get("num_links")
        combine_match = ref_config.get("all_to_all_combine", {}).get("num_links") == copy_config.get(
            "all_to_all_combine", {}
        ).get("num_links")

        print("\n" + "-" * 40)
        if dispatch_match and combine_match:
            print("✅ SUCCESS: All num_links configurations match!")
        else:
            print("❌ FAILED: num_links configurations differ!")
            if not dispatch_match:
                print("  - all_to_all_dispatch mismatch")
            if not combine_match:
                print("  - all_to_all_combine mismatch")

    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Some modules may require TT-Metal environment setup.")

    print("=" * 80)


def check_existing_hash_files():
    """Check for existing hash files from previous test runs."""

    print("\n" + "=" * 80)
    print("Checking for existing hash files from test runs")
    print("=" * 80)

    hash_locations = [
        ("/tmp/moe_reference_output/moe_hash.txt", "MoE Reference"),
        ("/tmp/moe_copied_output/moe_hash.txt", "MoE Copied"),
        ("/tmp/moe_decoder_reference_output/moe_decoder_hash.txt", "MoEDecoderBlock Reference"),
        ("/tmp/moe_decoder_copied_output/moe_decoder_hash.txt", "MoEDecoderBlock Copied"),
    ]

    found_hashes = {}

    for hash_path, label in hash_locations:
        path = Path(hash_path)
        if path.exists():
            hash_value = path.read_text().strip()
            print(f"\n{label}:")
            print(f"  Path: {hash_path}")
            print(f"  Hash: {hash_value}")
            found_hashes[label] = hash_value
        else:
            print(f"\n{label}: Not found at {hash_path}")

    # Compare if we have pairs
    print("\n" + "-" * 40)
    print("Comparison Results:")

    if "MoE Reference" in found_hashes and "MoE Copied" in found_hashes:
        if found_hashes["MoE Reference"] == found_hashes["MoE Copied"]:
            print("✅ MoE outputs are BYTEWISE IDENTICAL!")
        else:
            print("❌ MoE outputs differ!")
            print(f"  Reference: {found_hashes['MoE Reference']}")
            print(f"  Copied:    {found_hashes['MoE Copied']}")

    if "MoEDecoderBlock Reference" in found_hashes and "MoEDecoderBlock Copied" in found_hashes:
        if found_hashes["MoEDecoderBlock Reference"] == found_hashes["MoEDecoderBlock Copied"]:
            print("✅ MoEDecoderBlock outputs are BYTEWISE IDENTICAL!")
        else:
            print("❌ MoEDecoderBlock outputs differ!")
            print(f"  Reference: {found_hashes['MoEDecoderBlock Reference']}")
            print(f"  Copied:    {found_hashes['MoEDecoderBlock Copied']}")

    print("=" * 80)


if __name__ == "__main__":
    # Test CCL configurations
    test_ccl_config()

    # Test MoE CCL configurations
    test_moe_ccl_config()

    # Check for existing hash files
    check_existing_hash_files()

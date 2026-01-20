#!/usr/bin/env python3
"""
Diagnostic script to check device availability and mesh connectivity.
"""

import sys
import ttnn


def check_devices():
    """Check basic device availability."""
    print("=" * 80)
    print("DEVICE DIAGNOSTICS")
    print("=" * 80)

    # Check number of devices
    num_pcie = ttnn.get_num_pcie_devices()
    print(f"\nPCIe devices detected: {num_pcie}")

    if num_pcie < 2:
        print("ERROR: Need at least 2 devices for multi-chip tests")
        return False

    # Get device IDs
    device_ids = ttnn.get_pcie_device_ids()
    print(f"Device IDs: {device_ids}")

    # Try opening individual devices
    print("\nTesting individual device access...")
    devices = []
    for dev_id in device_ids[:4]:  # Test first 4
        try:
            print(f"  Opening device {dev_id}...", end=" ")
            dev = ttnn.open_device(device_id=dev_id)
            devices.append(dev)
            print("✓")
        except Exception as e:
            print(f"✗ Failed: {e}")
            return False

    # Close devices
    print("\nClosing devices...")
    for dev in devices:
        ttnn.close_device(dev)
    print("✓ All devices closed")

    return True


def check_mesh_connectivity():
    """Check if mesh device can be created."""
    print("\n" + "=" * 80)
    print("MESH CONNECTIVITY TEST")
    print("=" * 80)

    num_pcie = ttnn.get_num_pcie_devices()

    # Try 2-device mesh first (simplest)
    if num_pcie >= 2:
        print("\nTrying 2-device mesh (1x2)...")
        try:
            mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
            print("✓ 2-device mesh created successfully")
            ttnn.close_mesh_device(mesh)
            print("✓ 2-device mesh closed")
            return True
        except Exception as e:
            print(f"✗ Failed to create 2-device mesh: {e}")
            return False

    return False


def main():
    """Run diagnostics."""
    print("\nRunning device diagnostics...\n")

    # Check basic devices
    devices_ok = check_devices()

    if not devices_ok:
        print("\n" + "=" * 80)
        print("DIAGNOSTICS FAILED")
        print("=" * 80)
        print("\nSuggested fixes:")
        print("1. Reset all devices:")
        print("   for i in {0..7}; do tt-smi -r $i; done && sleep 30")
        print("2. Check physical connections (Ethernet cables)")
        print("3. Verify cluster configuration")
        return 1

    # Check mesh connectivity
    mesh_ok = check_mesh_connectivity()

    if mesh_ok:
        print("\n" + "=" * 80)
        print("✓ ALL DIAGNOSTICS PASSED")
        print("=" * 80)
        print("\nYour system is ready for benchmarks!")
        return 0
    else:
        print("\n" + "=" * 80)
        print("MESH CONNECTIVITY FAILED")
        print("=" * 80)
        print("\nThe Ethernet training timeout suggests:")
        print("1. Physical Ethernet cables may be disconnected")
        print("2. Devices may need firmware reset")
        print("3. Cluster topology may be misconfigured")
        print("\nTry:")
        print("  1. Check 'tt-smi' output for all devices")
        print("  2. Reset devices: for i in {0..7}; do tt-smi -r $i; done")
        print("  3. Wait 60 seconds after reset")
        print("  4. Re-run this diagnostic")
        return 1


if __name__ == "__main__":
    sys.exit(main())

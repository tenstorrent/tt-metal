#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Simple script to verify fabric telemetry initialization.

This script reads telemetry from fabric routers and verifies that:
1. static_info fields are properly initialized (mesh_id, device_id, direction)
2. supported_stats bitmask is set to enable telemetry
3. Counters are zeroed (not garbage values)

Usage:
    python test_fabric_telemetry_init.py
"""

import sys
from pathlib import Path

# Add tt_metal to path
tt_metal_path = Path(__file__).parent
sys.path.insert(0, str(tt_metal_path))

try:
    import ttnn
    from tt_metal.api.tt_metalium.experimental.fabric import read_fabric_telemetry, ControlPlane
    from tt_metal import MeshDevice
except ImportError as e:
    print(f"Error importing tt_metal: {e}")
    print("Make sure tt_metal Python bindings are built and in PYTHONPATH")
    sys.exit(1)


def verify_telemetry_initialization():
    """Verify fabric telemetry is properly initialized."""
    print("=" * 80)
    print("Fabric Telemetry Initialization Verification")
    print("=" * 80)

    # Get devices
    device_ids = ttnn.get_device_ids()
    if not device_ids:
        print("ERROR: No devices found")
        return False

    print(f"\nFound {len(device_ids)} device(s): {device_ids}")

    # Create mesh device
    print("\nInitializing mesh device...")
    mesh_device = MeshDevice.create_unit_meshes(device_ids)

    # Get control plane
    control_plane = ControlPlane.get_instance()

    all_passed = True
    total_channels = 0

    # Read telemetry from each device
    for device_id in device_ids:
        print(f"\n{'=' * 80}")
        print(f"Device {device_id}")
        print(f"{'=' * 80}")

        # Get fabric node ID
        fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device_id)
        print(f"Fabric Node ID: mesh_id={fabric_node_id.mesh_id}, chip_id={fabric_node_id.chip_id}")

        # Read telemetry samples
        samples = read_fabric_telemetry(fabric_node_id)

        if not samples:
            print(f"WARNING: No telemetry samples read from device {device_id}")
            continue

        print(f"\nFound {len(samples)} ethernet channel(s) with telemetry")

        # Verify each sample
        for sample in samples:
            total_channels += 1
            channel_id = sample.channel_id
            static_info = sample.snapshot.static_info
            dynamic_info = sample.snapshot.dynamic_info

            print(f"\n  Channel {channel_id}:")
            print(f"    mesh_id:         {static_info.mesh_id}")
            print(f"    device_id:       {static_info.device_id}")
            print(f"    direction:       {static_info.direction} ", end="")

            # Map direction to human-readable
            direction_names = {0: "EAST", 1: "WEST", 2: "NORTH", 3: "SOUTH"}
            if static_info.direction in direction_names:
                print(f"({direction_names[static_info.direction]})")
            else:
                print(f"(INVALID)")

            print(f"    fabric_config:   0x{static_info.fabric_config:08x}")
            print(f"    supported_stats: 0x{static_info.supported_stats:02x} ", end="")

            # Decode supported_stats
            stats = []
            if static_info.supported_stats & 0x01:
                stats.append("ROUTER_STATE")
            if static_info.supported_stats & 0x02:
                stats.append("BANDWIDTH")
            if static_info.supported_stats & 0x04:
                stats.append("HEARTBEAT_TX")
            if static_info.supported_stats & 0x08:
                stats.append("HEARTBEAT_RX")
            print(f"({', '.join(stats) if stats else 'NONE'})")

            # Verify values
            channel_passed = True

            # Check mesh_id matches
            if static_info.mesh_id != fabric_node_id.mesh_id:
                print(f"    ❌ FAIL: mesh_id {static_info.mesh_id} != expected {fabric_node_id.mesh_id}")
                channel_passed = False

            # Check device_id matches
            if static_info.device_id != fabric_node_id.chip_id:
                print(f"    ❌ FAIL: device_id {static_info.device_id} != expected {fabric_node_id.chip_id}")
                channel_passed = False

            # Check direction is valid
            if static_info.direction > 3:
                print(f"    ❌ FAIL: direction {static_info.direction} is invalid (must be 0-3)")
                channel_passed = False

            # Check supported_stats is non-zero
            if static_info.supported_stats == 0:
                print(f"    ❌ FAIL: supported_stats is 0, telemetry disabled")
                channel_passed = False

            # Check BANDWIDTH bit is set
            if not (static_info.supported_stats & 0x02):
                print(f"    ❌ FAIL: BANDWIDTH telemetry not enabled")
                channel_passed = False

            # Check counters aren't garbage (> 10^12)
            GARBAGE_THRESHOLD = 1_000_000_000_000

            tx_bw = dynamic_info.tx_bandwidth
            rx_bw = dynamic_info.rx_bandwidth

            if tx_bw.elapsed_cycles > GARBAGE_THRESHOLD:
                print(f"    ❌ FAIL: TX elapsed_cycles={tx_bw.elapsed_cycles} appears uninitialized")
                channel_passed = False

            if rx_bw.elapsed_cycles > GARBAGE_THRESHOLD:
                print(f"    ❌ FAIL: RX elapsed_cycles={rx_bw.elapsed_cycles} appears uninitialized")
                channel_passed = False

            if channel_passed:
                print(f"    ✅ PASS: Telemetry properly initialized")
            else:
                all_passed = False

    # Summary
    print(f"\n{'=' * 80}")
    print("Summary")
    print(f"{'=' * 80}")
    print(f"Total channels tested: {total_channels}")

    if all_passed and total_channels > 0:
        print("✅ ALL TESTS PASSED")
        return True
    else:
        print("❌ SOME TESTS FAILED")
        return False


if __name__ == "__main__":
    try:
        success = verify_telemetry_initialization()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""Test script for Python UI."""

import sys
import time

try:
    from tt_smi import get_devices, cleanup_dead_processes, format_bytes
    from tt_smi.ui.dashboard import Dashboard
    from rich.console import Console
except ImportError as e:
    print(f"Error: {e}")
    print("\nInstall TT-SMI first:")
    print("  pip install -e .")
    sys.exit(1)


def test_basic_api():
    """Test basic API calls."""
    print("Testing basic API...")

    # Get devices
    devices = get_devices()
    print(f"Found {len(devices)} device(s)")

    for dev in devices:
        print(f"\nDevice {dev.display_id}:")
        print(f"  Architecture: {dev.arch_name}")
        print(f"  Remote: {dev.is_remote}")
        print(f"  Temperature: {dev.temperature}°C")
        print(f"  Power: {dev.power}W")
        print(f"  AICLK: {dev.aiclk_mhz} MHz")
        print(f"  DRAM: {format_bytes(dev.used_dram)} / {format_bytes(dev.total_dram)}")
        print(f"  L1: {format_bytes(dev.used_l1)} / {format_bytes(dev.total_l1)}")
        print(f"  Has SHM: {dev.has_shm}")
        print(f"  Processes: {len(dev.processes)}")

        for proc in dev.processes[:3]:  # Show first 3
            print(f"    - PID {proc['pid']}: {proc['name']} (DRAM: {format_bytes(proc['dram'])})")

    # Cleanup
    cleaned = cleanup_dead_processes()
    print(f"\nCleaned {cleaned} dead process(es)")

    return len(devices) > 0


def test_dashboard():
    """Test dashboard rendering."""
    print("\nTesting dashboard...")

    console = Console()
    dashboard = Dashboard(console)

    devices = get_devices()
    if not devices:
        print("No devices found!")
        return False

    # Single snapshot
    print("\nRendering snapshot...")
    dashboard.print_snapshot(devices)

    return True


def test_watch_mode():
    """Test watch mode (5 seconds)."""
    print("\nTesting watch mode (5 seconds)...")

    console = Console()
    dashboard = Dashboard(console)

    try:
        import threading

        def stop_after_5s():
            time.sleep(5)
            raise KeyboardInterrupt()

        timer = threading.Timer(5.0, stop_after_5s)
        timer.start()

        dashboard.watch(get_devices, refresh_ms=1000)

    except KeyboardInterrupt:
        print("\nWatch mode test completed")

    return True


if __name__ == "__main__":
    print("TT-SMI Python UI Test Suite\n")
    print("=" * 60)

    tests = [
        ("Basic API", test_basic_api),
        ("Dashboard", test_dashboard),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n[TEST] {name}")
        print("-" * 60)
        try:
            if test_func():
                print(f"[PASS] {name}")
                passed += 1
            else:
                print(f"[FAIL] {name}")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {name}: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} test(s) failed")
        sys.exit(1)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test script to verify allocation tracking works with real ttnn operations.

This script performs basic ttnn operations and allocations, which should be
visible in the allocation monitor if tracking is enabled.

Usage:
    # Terminal 1: Start server
    ./allocation_server_poc

    # Terminal 2: Start monitor
    ./allocation_monitor_client -r 500

    # Terminal 3: Run this script with tracking enabled
    export TT_ALLOC_TRACKING_ENABLED=1
    python test_ttnn_allocations.py
"""

import os
import time
import torch
import ttnn

# Color codes for pretty output
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_step(step_num, description):
    """Print a colored step header"""
    print(f"\n{BOLD}{CYAN}[Step {step_num}] {description}{RESET}")
    print(f"{CYAN}{'─' * 60}{RESET}")


def print_info(message):
    """Print info message"""
    print(f"{BLUE}ℹ {message}{RESET}")


def print_success(message):
    """Print success message"""
    print(f"{GREEN}✓ {message}{RESET}")


def print_warning(message):
    """Print warning message"""
    print(f"{YELLOW}⚠ {message}{RESET}")


def check_tracking_enabled():
    """Check if allocation tracking is enabled"""
    enabled = os.environ.get("TT_ALLOC_TRACKING_ENABLED", "0") == "1"
    if enabled:
        print_success("Allocation tracking is ENABLED")
    else:
        print_warning("Allocation tracking is DISABLED")
        print_info("Enable it with: export TT_ALLOC_TRACKING_ENABLED=1")
    return enabled


def main():
    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  TTNN Allocation Tracking Test{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")

    # Check if tracking is enabled
    tracking_enabled = check_tracking_enabled()

    if not tracking_enabled:
        print_warning("\nRunning without tracking. Allocations won't be visible in monitor.")
        print_info("Rerun with: TT_ALLOC_TRACKING_ENABLED=1 python test_ttnn_allocations.py")

    print_info("PID: {}".format(os.getpid()))
    print_info("Watch the allocation monitor to see real-time memory changes!")

    # Step 1: Open device
    print_step(1, "Opening TT Device")
    print_info("This will allocate initial device structures...")

    # Get the actual device ID from the mesh device
    device = ttnn.open_device(device_id=0)

    # Get the actual chip ID being used
    try:
        actual_device_id = device.get_devices()[0].id()
        print_success(f"Device opened: {device}")
        print_info(f"Actual device ID: {actual_device_id}")
        print_warning(f"⚠️  IMPORTANT: Monitor device {actual_device_id} with:")
        print(f"{YELLOW}   ./allocation_monitor_client -d {actual_device_id} -r 500{RESET}")
    except:
        print_success(f"Device opened: {device}")
        print_warning("⚠️  If monitor shows no changes, try different device IDs:")
        print(f"{YELLOW}   ./allocation_monitor_client -d 1 -r 500{RESET}")

    time.sleep(2)  # Give time to see initial allocations

    # Step 2: Create a large DRAM tensor (100MB)
    print_step(2, "Allocating Large Tensor (100MB DRAM)")
    print_info("Creating 512x512 tensor (~100MB)...")

    # Create tensor on host
    torch_tensor_large = torch.randn(1, 1, 512, 512, dtype=torch.bfloat16)

    # Move to device (allocates DRAM buffer)
    tt_tensor_large = ttnn.from_torch(
        torch_tensor_large,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print_success(f"Large tensor created on device")
    print_info(f"Shape: {tt_tensor_large.shape}")
    print_info(f"📊 Check monitor - should see ~100MB DRAM increase!")
    time.sleep(3)

    # Step 3: Create another huge tensor (200MB)
    print_step(3, "Allocating Huge Tensor (200MB DRAM)")
    print_info("Creating 1024x512 tensor (~200MB)...")

    torch_tensor_huge = torch.randn(1, 1, 1024, 512, dtype=torch.bfloat16)
    tt_tensor_huge = ttnn.from_torch(
        torch_tensor_huge,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print_success("Huge tensor created on device")
    print_info(f"Shape: {tt_tensor_huge.shape}")
    print_info(f"📊 Check monitor - should see ~300MB total DRAM now!")
    time.sleep(3)

    # Step 4: Create a massive tensor (500MB)
    print_step(4, "Allocating MASSIVE Tensor (500MB DRAM)")
    print_info("Creating 1024x1024 tensor (~500MB)...")

    torch_tensor_massive = torch.randn(1, 1, 1024, 1024, dtype=torch.bfloat16)
    tt_tensor_massive = ttnn.from_torch(
        torch_tensor_massive,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    print_success("Massive tensor created on device")
    print_info(f"Shape: {tt_tensor_massive.shape}")
    print_info(f"📊 Check monitor - should see ~800MB total DRAM now!")
    time.sleep(3)

    # Step 5: Create L1 tensor (if supported)
    print_step(5, "Allocating L1 Tensor (smaller, for L1)")
    print_info("Creating tensor in L1 memory...")

    tt_tensor_l1 = None
    try:
        torch_tensor_l1 = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
        tt_tensor_l1 = ttnn.from_torch(
            torch_tensor_l1,
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        print_success("L1 tensor created")
        print_info(f"Shape: {tt_tensor_l1.shape}")
        print_info(f"📊 Check monitor - should see L1 usage now!")
    except Exception as e:
        print_warning(f"L1 allocation failed: {e}")

    time.sleep(3)

    # Step 6: Perform operations (may create intermediate buffers)
    print_step(6, "Performing Operations")
    print_info("Adding large tensors together...")

    result = ttnn.add(tt_tensor_large, tt_tensor_large)
    print_success("Operation completed")
    print_info("📊 May see temporary buffer allocations")
    time.sleep(3)

    # Step 7: Start deallocating - watch memory DROP!
    print_step(7, "Deallocating Tensors - WATCH MEMORY DROP!")
    print_info("Monitor should show dramatic decreases...")
    time.sleep(2)

    print_info("Deallocating result tensor...")
    ttnn.deallocate(result)
    time.sleep(2)

    print_info("Deallocating MASSIVE tensor (500MB freed)...")
    ttnn.deallocate(tt_tensor_massive)
    print_success("📊 Monitor should drop by ~500MB!")
    time.sleep(3)

    print_info("Deallocating HUGE tensor (200MB freed)...")
    ttnn.deallocate(tt_tensor_huge)
    print_success("📊 Monitor should drop by ~200MB!")
    time.sleep(3)

    print_info("Deallocating LARGE tensor (100MB freed)...")
    ttnn.deallocate(tt_tensor_large)
    print_success("📊 Monitor should drop by ~100MB!")
    time.sleep(3)

    if tt_tensor_l1 is not None:
        print_info("Deallocating L1 tensor...")
        ttnn.deallocate(tt_tensor_l1)
        print_success("📊 L1 should be cleared!")
        time.sleep(2)

    print_success("All tensors deallocated - memory should be back to baseline!")

    # Step 7: Close device
    print_step(7, "Closing Device")
    print_info("Cleaning up device resources...")

    ttnn.close_device(device)
    print_success("Device closed")

    time.sleep(1)

    # Summary
    print(f"\n{BOLD}{GREEN}{'═' * 60}{RESET}")
    print(f"{BOLD}{GREEN}  Test Complete!{RESET}")
    print(f"{BOLD}{GREEN}{'═' * 60}{RESET}\n")

    if tracking_enabled:
        print_success("If allocation server is running, you should have seen:")
        print(f"{GREEN}  • Allocations when tensors were created{RESET}")
        print(f"{GREEN}  • Memory usage increasing in monitor{RESET}")
        print(f"{GREEN}  • Deallocations when tensors were freed{RESET}")
        print(f"{GREEN}  • Memory usage decreasing back to baseline{RESET}")
    else:
        print_warning("Tracking was disabled. To see allocations:")
        print(f"{YELLOW}  1. Start server: ./allocation_server_poc{RESET}")
        print(f"{YELLOW}  2. Start monitor: ./allocation_monitor_client -r 500{RESET}")
        print(f"{YELLOW}  3. Rerun: TT_ALLOC_TRACKING_ENABLED=1 python {__file__}{RESET}")

    print(f"\n{BLUE}Check the allocation server and monitor for detailed stats!{RESET}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{RED}Error: {e}{RESET}")
        import traceback

        traceback.print_exc()
        exit(1)

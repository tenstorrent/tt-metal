#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Test allocation tracking with distributed tensors across 8-device mesh (2x4).

This script:
1. Opens an 8-device mesh (2x4 topology)
2. Creates tensors distributed across all devices
3. Performs computations (add, matmul)
4. Validates that allocations are tracked on all devices

Usage:
    # Terminal 1: Start allocation server
    ./allocation_server_poc

    # Terminal 2: Start monitor for ALL devices
    ./allocation_monitor_client -a -r 500

    # Terminal 3: Run this test with tracking enabled
    export TT_ALLOC_TRACKING_ENABLED=1
    python test_mesh_allocation.py
"""

import os
import time
import torch
import ttnn
from ttnn.distributed import visualize_system_mesh, visualize_mesh_device, visualize_tensor

# Color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(title):
    """Print a styled header"""
    print(f"\n{BOLD}{CYAN}{'═' * 80}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 80}{RESET}\n")


def print_step(step_num, description):
    """Print a step header"""
    print(f"\n{BOLD}{BLUE}[Step {step_num}] {description}{RESET}")
    print(f"{BLUE}{'─' * 80}{RESET}")


def print_info(message):
    """Print info message"""
    print(f"{BLUE}ℹ  {message}{RESET}")


def print_success(message):
    """Print success message"""
    print(f"{GREEN}✓ {message}{RESET}")


def print_warning(message):
    """Print warning message"""
    print(f"{YELLOW}⚠  {message}{RESET}")


def print_allocation_tip(devices_str="0-7"):
    """Print tip about monitoring allocations"""
    print(f"\n{YELLOW}📊 CHECK MONITOR: Watch allocations on devices {devices_str}!{RESET}")
    print(f"{YELLOW}   Run: ./allocation_monitor_client -a -r 500{RESET}\n")


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
    print_header("8-Device Mesh Allocation Tracking Test")

    # Check tracking status
    tracking_enabled = check_tracking_enabled()
    if not tracking_enabled:
        print_warning("Running without tracking. Allocations won't be visible.")
        print_info("Rerun with: TT_ALLOC_TRACKING_ENABLED=1 python test_mesh_allocation.py")

    print_info(f"PID: {os.getpid()}")

    # Step 1: Open mesh device
    print_step(1, "Opening 8-Device Mesh (2x4 Topology)")
    print_info("Opening mesh with shape (2, 4) - 2 rows, 4 columns...")

    try:
        mesh_device = ttnn.open_mesh_device(ttnn.MeshShape(2, 4))
        print_success(f"Mesh device opened: {mesh_device}")

        # Visualize the mesh topology
        print_info("Mesh topology:")
        visualize_system_mesh()
        visualize_mesh_device(mesh_device)

        # Get individual device IDs
        # MeshDevice doesn't expose get_devices(), but we know it's 8 devices (0-7)
        print_info(f"Mesh contains 8 devices (IDs: 0-7)")
        print_allocation_tip("0-7")

        time.sleep(3)

    except Exception as e:
        print(f"{RED}❌ Error opening mesh device: {e}{RESET}")
        print_info("Make sure you have 8 devices available")
        return 1

    # Step 2: Create distributed tensor (sharded across mesh)
    print_step(2, "Creating Distributed Tensor (~4MB per device)")
    print_info("Creating tensor sharded across all 8 devices...")
    print_info("Shape: (8, 8, 512, 512) - about 4MB per device (32MB total)")

    # Create tensor on host
    torch_tensor = torch.randn(8, 8, 512, 512, dtype=torch.bfloat16)
    print_success(f"Host tensor created: {torch_tensor.shape}")

    # Convert to ttnn tensor first
    print_info("Converting to ttnn tensor...")
    ttnn_tensor = ttnn.from_torch(
        torch_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )

    # Create mesh mapper for 2D distribution
    print_info("Creating mesh mapper (shard dim 0 across rows, dim 1 across cols)...")
    mapper = ttnn.create_mesh_mapper(
        mesh_device,
        ttnn.MeshMapperConfig(
            [ttnn.PlacementShard(0), ttnn.PlacementShard(1)],  # Shard dims 0 and 1
            ttnn.MeshShape(2, 4),
        ),
    )

    # Distribute tensor across mesh
    print_info("Distributing tensor across mesh...")
    distributed_tensor = ttnn.distribute_tensor(ttnn_tensor, mapper, mesh_device)
    print_success("Tensor distributed across mesh!")

    # Visualize distribution
    print_info("Tensor distribution across mesh:")
    visualize_tensor(distributed_tensor)

    print_allocation_tip()
    print_info("Each device should show ~4MB DRAM allocation")
    time.sleep(5)

    # Step 3: Create another distributed tensor
    print_step(3, "Creating Second Distributed Tensor (~4MB per device)")
    print_info("Creating another tensor for computation...")

    torch_tensor2 = torch.randn(8, 8, 512, 512, dtype=torch.bfloat16)
    ttnn_tensor2 = ttnn.from_torch(torch_tensor2, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    distributed_tensor2 = ttnn.distribute_tensor(ttnn_tensor2, mapper, mesh_device)
    print_success("Second tensor distributed!")

    print_allocation_tip()
    print_info("Each device should now show ~8MB DRAM allocation (2 tensors × 4MB)")
    time.sleep(5)

    # Step 4: Perform distributed computation (element-wise add)
    print_step(4, "Performing Distributed Computation (Add)")
    print_info("Computing: result = tensor1 + tensor2")
    print_info("This operation happens IN PARALLEL on all 8 devices!")

    result_add = ttnn.add(distributed_tensor, distributed_tensor2)
    print_success("Distributed addition completed!")

    # Verify math correctness
    print_info("Verifying computation correctness...")
    # Need to provide a mesh_composer to concatenate shards from multiple devices
    # Since we sharded along dims 0 and 1, we need to concatenate along both
    mesh_composer = ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig([0, 1], ttnn.MeshShape(2, 4)))
    result_cpu = ttnn.to_torch(result_add, mesh_composer=mesh_composer)
    expected = torch_tensor + torch_tensor2
    if torch.allclose(result_cpu, expected, rtol=1e-2, atol=1e-2):
        print_success("✓ Math verification PASSED - results are correct!")
    else:
        print_warning("⚠ Math verification FAILED - results differ!")

    print_allocation_tip()
    print_info("May see temporary allocations during computation")
    time.sleep(2)

    # Step 5: Perform matrix multiplication (more complex)
    print_step(2, "Performing Distributed MatMul")
    print_info("Creating matrices for matmul...")

    # Create smaller tensors for matmul (to avoid OOM)
    torch_a = torch.randn(8, 8, 256, 128, dtype=torch.bfloat16)
    torch_b = torch.randn(8, 8, 128, 256, dtype=torch.bfloat16)

    ttnn_a = ttnn.from_torch(torch_a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    ttnn_b = ttnn.from_torch(torch_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    dist_a = ttnn.distribute_tensor(ttnn_a, mapper, mesh_device)
    dist_b = ttnn.distribute_tensor(ttnn_b, mapper, mesh_device)

    print_info("Computing: result = matmul(A, B)")
    print_info("Matrix shapes: (256, 128) @ (128, 256) = (256, 256)")

    result_matmul = ttnn.matmul(dist_a, dist_b)
    print_success("Distributed matmul completed!")

    # Verify matmul correctness
    print_info("Verifying matmul correctness...")
    result_matmul_cpu = ttnn.to_torch(result_matmul, mesh_composer=mesh_composer)
    expected_matmul = torch.matmul(torch_a, torch_b)
    if torch.allclose(result_matmul_cpu, expected_matmul, rtol=1e-1, atol=1e-1):
        print_success("✓ MatMul verification PASSED - results are correct!")
    else:
        print_warning("⚠ MatMul verification FAILED - results differ!")

    print_allocation_tip()
    print_info("Each device now has multiple tensors allocated")
    time.sleep(2)

    # Step 6: Deallocate tensors - watch memory drop!
    print_step(2, "Deallocating Tensors - WATCH MEMORY DROP!")
    print_info("Deallocating tensors one by one...")

    print_info("Deallocating result_matmul...")
    ttnn.deallocate(result_matmul)
    time.sleep(2)

    print_info("Deallocating dist_a and dist_b...")
    ttnn.deallocate(dist_a)
    ttnn.deallocate(dist_b)
    print_success("📊 MatMul tensors freed on all devices!")
    time.sleep(2)

    print_info("Deallocating result_add...")
    ttnn.deallocate(result_add)
    time.sleep(2)

    print_info("Deallocating distributed_tensor2...")
    ttnn.deallocate(distributed_tensor2)
    print_success("📊 Second tensor freed (~4MB per device)!")
    time.sleep(2)

    print_info("Deallocating distributed_tensor...")
    ttnn.deallocate(distributed_tensor)
    print_success("📊 First tensor freed (~4MB per device)!")
    time.sleep(2)

    print_success("All tensors deallocated - memory should be back to baseline!")
    print_info("Note: ~36KB cached program buffers + ~14-15MB system buffers remain")
    print_info("       (Cached programs = compiled kernels from add/matmul operations)")

    # Step 7: Clear program cache to free cached kernel buffers
    print_step(2, "Clearing Program Cache")
    print_info("Clearing cached programs (compiled kernels from add/matmul)...")
    print_allocation_tip()
    print_info("Watch the monitor - cached kernel buffers (~36KB per device) will be freed!")

    mesh_device.disable_and_clear_program_cache()
    print_success("Program cache cleared!")

    print_info("Waiting 3 seconds for cache cleanup to propagate to monitor...")
    time.sleep(2)

    # Step 8: Close mesh device
    print_step(8, "Closing Mesh Device")
    print_info("Closing all 8 devices...")
    print_allocation_tip()
    print_info("Watch the monitor - system buffers (~14MB + L1) will be freed!")

    ttnn.close_mesh_device(mesh_device)
    print_success("Mesh device closed")

    print_info("Waiting 5 seconds for cleanup to propagate to monitor...")
    time.sleep(5)

    # Summary
    print_header("Test Complete!")

    if tracking_enabled:
        print_success("If allocation server is running, you should have seen:")
        print(f"{GREEN}  • Allocations on ALL 8 devices when tensors were created (~4MB each){RESET}")
        print(f"{GREEN}  • Memory distributed evenly across the mesh{RESET}")
        print(f"{GREEN}  • Parallel computation on all devices{RESET}")
        print(f"{GREEN}  • Deallocations when tensors were freed{RESET}")
        print(f"{GREEN}  • Program cache cleared (freeing ~36KB cached kernels per device){RESET}")
        print(f"{GREEN}  • Memory returning to baseline (~14-15MB system buffers per device){RESET}")
    else:
        print_warning("Tracking was disabled. To see allocations:")
        print(f"{YELLOW}  1. Start server: ./allocation_server_poc{RESET}")
        print(f"{YELLOW}  2. Start monitor: ./allocation_monitor_client -a -r 500{RESET}")
        print(f"{YELLOW}  3. Rerun: TT_ALLOC_TRACKING_ENABLED=1 python test_mesh_allocation.py{RESET}")

    print(f"\n{BLUE}Check the allocation monitor for per-device stats!{RESET}")
    print(f"{BLUE}You should see allocations distributed across devices 0-7{RESET}")
    print(f"\n{CYAN}Note: System buffers (~14-15MB) persist until mesh device closes.{RESET}")
    print(f"{CYAN}Cached programs (~36KB) are freed when program cache is cleared.{RESET}\n")

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}⚠  Interrupted by user{RESET}")
        exit(1)
    except Exception as e:
        print(f"\n{RED}❌ Error: {e}{RESET}")
        import traceback

        traceback.print_exc()
        exit(1)

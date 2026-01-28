# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Test to reliably trigger the "allocation during active trace" warning.

This test is designed to reproduce the conditions that caused the intermittent
CI failure in the ViT performance test (test_demo_vit_ttnn_inference_perf_e2e_2cq_trace.py).

Background:
-----------
The original failure occurred when:
1. A trace was captured (operations recorded for later replay)
2. Buffer allocations happened after trace capture but before trace release
3. The allocator warned: "Allocating device buffers is unsafe due to the existence 
   of an active trace. These buffers may be corrupted once a trace is executed."
4. In rare cases, this led to memory corruption and device hangs

This test deliberately triggers this warning to:
1. Verify the warning mechanism works
2. Provide a benchmark for testing defensive mechanisms
3. Allow stress testing of the trace + allocation interaction

UMD/Firmware Version Verification:
----------------------------------
The original failure occurred with firmware bundle version 18.7.0.
To verify your environment matches CI:

    import ttnn
    # The firmware version is logged during device initialization:
    # "Established firmware bundle version: X.Y.Z"
    
    # To check UMD submodule version:
    # cd tt-metal && git submodule status tt_metal/third_party/umd

Usage:
------
Run with pytest:
    pytest tests/ttnn/tracy/test_trace_allocation_warning.py -v -s

To capture the warning output:
    pytest tests/ttnn/tracy/test_trace_allocation_warning.py -v -s 2>&1 | grep -i "unsafe"
"""

import pytest
import torch
from loguru import logger

import ttnn


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000, "l1_small_size": 32768}],
    indirect=True,
)
def test_allocation_during_active_trace_triggers_warning(device):
    """
    Test that allocating buffers while a trace is active triggers the expected warning.
    
    This test:
    1. Creates tensors and runs an operation to compile kernels
    2. Captures a trace of operations
    3. Allocates NEW buffers after trace capture (triggering the warning)
    4. Executes the trace
    5. Releases the trace
    
    The warning "Allocating device buffers is unsafe due to the existence of an 
    active trace" should be logged after step 3.
    """
    torch.manual_seed(42)
    
    # Step 1: Create initial tensors
    a = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    
    tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Warm up / compile kernels before trace capture
    tt_result = ttnn.add(tt_a, tt_b)
    ttnn.synchronize_device(device)
    
    logger.info("Step 1: Initial tensors created and operation compiled")
    
    # Step 2: Capture a trace
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    
    # Run operations inside trace capture
    for _ in range(5):
        tt_result = ttnn.add(tt_a, tt_b)
    
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    
    logger.info("Step 2: Trace captured (trace is now ACTIVE)")
    
    # Step 3: Allocate NEW buffers while trace is active
    # THIS SHOULD TRIGGER THE WARNING
    logger.info("Step 3: Allocating new buffers while trace is active...")
    logger.info("        >>> EXPECT WARNING: 'Allocating device buffers is unsafe...'")
    
    # Create multiple new tensors to increase chance of triggering the warning
    new_tensors = []
    for i in range(10):
        new_tensor = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
        tt_new = ttnn.from_torch(new_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        new_tensors.append(tt_new)
    
    logger.info(f"Step 3: Allocated {len(new_tensors)} new tensors while trace was active")
    
    # Step 4: Execute the trace
    logger.info("Step 4: Executing trace...")
    for _ in range(3):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    
    ttnn.synchronize_device(device)
    logger.info("Step 4: Trace executed successfully")
    
    # Step 5: Release the trace
    ttnn.release_trace(device, trace_id)
    logger.info("Step 5: Trace released (allocations are now safe again)")
    
    # Cleanup
    for tt in new_tensors:
        tt.deallocate()
    tt_a.deallocate()
    tt_b.deallocate()
    tt_result.deallocate()


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 200000, "l1_small_size": 32768, "num_command_queues": 2}],
    indirect=True,
)
def test_allocation_during_trace_execution_2cq(device):
    """
    Test allocation during trace execution with dual command queues.
    
    This more closely mimics the original ViT test failure scenario:
    - CQ0: Executes the trace (non-blocking)
    - CQ1: Handles input/output transfers (which may allocate buffers)
    
    The race condition between CQ0 trace execution and CQ1 allocations
    is what caused the original intermittent failure.
    """
    torch.manual_seed(42)
    
    # Create tensors
    a = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
    b = torch.randn(1, 1, 128, 128, dtype=torch.bfloat16)
    
    tt_a = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    tt_b = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    
    # Warm up
    tt_result = ttnn.matmul(tt_a, tt_b)
    ttnn.synchronize_device(device)
    
    logger.info("Dual CQ test: Tensors created, operation compiled")
    
    # Capture trace on CQ0
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(3):
        tt_result = ttnn.matmul(tt_a, tt_b)
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    
    logger.info("Dual CQ test: Trace captured on CQ0")
    
    # Initialize events for CQ synchronization
    event_cq0 = ttnn.record_event(device, 0)
    event_cq1 = ttnn.record_event(device, 1)
    
    # Simulate the ViT test pattern:
    # Execute trace on CQ0 (non-blocking) while doing allocations
    num_iterations = 10
    allocated_tensors = []
    
    logger.info(f"Dual CQ test: Running {num_iterations} iterations with concurrent allocations")
    
    for i in range(num_iterations):
        # Execute trace on CQ0 (non-blocking)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
        
        # While trace is executing, allocate on host and potentially transfer
        # This simulates what happens in the ViT test warmup/measurement loops
        new_tensor = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
        
        # This allocation may trigger the warning if it happens while trace is running
        tt_new = ttnn.from_torch(new_tensor, layout=ttnn.TILE_LAYOUT, device=device)
        allocated_tensors.append(tt_new)
        
        # Record event after allocation
        event_cq1 = ttnn.record_event(device, 1)
        
        # Wait for trace to complete before next iteration
        ttnn.wait_for_event(0, event_cq1)
        event_cq0 = ttnn.record_event(device, 0)
    
    ttnn.synchronize_device(device)
    logger.info(f"Dual CQ test: Completed {num_iterations} iterations")
    
    # Cleanup
    ttnn.release_trace(device, trace_id)
    for tt in allocated_tensors:
        tt.deallocate()
    tt_a.deallocate()
    tt_b.deallocate()
    tt_result.deallocate()
    
    logger.info("Dual CQ test: Cleanup complete")


@pytest.mark.parametrize(
    "device_params",
    [{"trace_region_size": 500000, "l1_small_size": 32768}],
    indirect=True,
)
def test_stress_allocation_during_trace(device):
    """
    Stress test: Perform many allocations while a trace is active.
    
    This test is designed to maximize the probability of triggering
    any latent race conditions or memory corruption issues.
    
    WARNING: This test intentionally pushes the system to trigger
    edge cases. It should NOT cause crashes but may log many warnings.
    """
    torch.manual_seed(42)
    
    # Create initial tensors for trace
    tensors = []
    for i in range(4):
        t = torch.randn(1, 1, 64, 64, dtype=torch.bfloat16)
        tt = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=device)
        tensors.append(tt)
    
    # Warm up with various operations
    result = ttnn.add(tensors[0], tensors[1])
    result = ttnn.mul(result, tensors[2])
    ttnn.synchronize_device(device)
    
    logger.info("Stress test: Initial setup complete")
    
    # Capture a complex trace
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    
    temp = tensors[0]
    for i in range(10):
        temp = ttnn.add(temp, tensors[i % 4])
        temp = ttnn.mul(temp, tensors[(i + 1) % 4])
    
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.synchronize_device(device)
    
    logger.info("Stress test: Complex trace captured")
    
    # Now stress the allocation system while trace is active
    stress_tensors = []
    num_stress_allocations = 50
    
    logger.info(f"Stress test: Performing {num_stress_allocations} allocations while trace is active...")
    
    for i in range(num_stress_allocations):
        # Vary tensor sizes to stress different allocation paths
        size = 32 * ((i % 4) + 1)
        new_t = torch.randn(1, 1, size, size, dtype=torch.bfloat16)
        tt_new = ttnn.from_torch(new_t, layout=ttnn.TILE_LAYOUT, device=device)
        stress_tensors.append(tt_new)
        
        # Periodically execute the trace
        if i % 10 == 0:
            ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    
    ttnn.synchronize_device(device)
    logger.info(f"Stress test: Completed {num_stress_allocations} allocations and periodic trace executions")
    
    # Final trace executions
    for _ in range(5):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    
    ttnn.synchronize_device(device)
    
    # Cleanup
    ttnn.release_trace(device, trace_id)
    for tt in stress_tensors:
        tt.deallocate()
    for tt in tensors:
        tt.deallocate()
    result.deallocate()
    temp.deallocate()
    
    logger.info("Stress test: Complete - check logs for any warnings or errors")


def test_verify_environment_info(device):
    """
    Utility test to print environment information for debugging.
    
    This helps verify that your test environment matches CI when
    investigating intermittent failures.
    """
    import os
    
    logger.info("=" * 60)
    logger.info("ENVIRONMENT INFORMATION")
    logger.info("=" * 60)
    
    # Device info
    logger.info(f"Device ID: {device.id()}")
    logger.info(f"Arch: {ttnn.get_arch_name()}")
    logger.info(f"Num devices: {ttnn.get_num_devices()}")
    
    # Cluster info
    try:
        cluster_type = ttnn.cluster.get_cluster_type()
        logger.info(f"Cluster type: {cluster_type}")
    except Exception as e:
        logger.info(f"Cluster type: Unable to determine ({e})")
    
    # Check for N300 specifically (relevant to original failure)
    try:
        num_pcie = ttnn.GetNumPCIeDevices()
        is_n300 = num_pcie == 1 and ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.N300
        logger.info(f"Is single-card N300: {is_n300}")
    except Exception as e:
        logger.info(f"N300 detection failed: {e}")
    
    # UMD/Firmware version is logged during device init
    # Look for: "Established firmware bundle version: X.Y.Z"
    logger.info("")
    logger.info("NOTE: Firmware bundle version is logged during device initialization.")
    logger.info("      Look for 'Established firmware bundle version' in the test output.")
    logger.info("")
    logger.info("To verify UMD submodule version matches CI:")
    logger.info("  cd tt-metal && git submodule status tt_metal/third_party/umd")
    logger.info("")
    logger.info("Original CI failure occurred with firmware bundle version: 18.7.0")
    logger.info("=" * 60)

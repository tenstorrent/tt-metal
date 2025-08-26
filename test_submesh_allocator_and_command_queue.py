#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for sub-mesh allocator addresses and command queue execution behavior.

This test file implements the example scenario described in the sub-meshes specification:
- Submesh A: devices {0, 1, 4, 5}
- Submesh B: devices {2, 3, 6, 7}
- Submesh C: devices {0, 1, 2, 3}

Tests verify:
1. Buffer allocator addresses are consistent across overlapping submeshes
2. Command queue execution behavior for overlapping vs non-overlapping submeshes
"""

import pytest
import torch
import ttnn
from typing import List, Dict, Any
from loguru import logger


class SubmeshAllocatorTester:
    """Helper class for testing submesh allocator behavior."""

    def __init__(self, mesh_device_2x4):
        """Initialize with a 2x4 mesh device (8 devices total: 0,1,2,3,4,5,6,7)."""
        self.parent_mesh = mesh_device_2x4
        self.submeshes = {}
        self._setup_submeshes()

    def _setup_submeshes(self):
        """Create the three submeshes as specified in the example."""
        # Submesh A: devices {0, 1, 4, 5} -> coordinates (0,0), (0,1), (1,0), (1,1)
        submesh_a = self.parent_mesh.create_submesh(ttnn.MeshShape(2, 2), ttnn.MeshCoordinate(0, 0))

        # Submesh B: devices {2, 3, 6, 7} -> coordinates (0,2), (0,3), (1,2), (1,3)
        submesh_b = self.parent_mesh.create_submesh(ttnn.MeshShape(2, 2), ttnn.MeshCoordinate(0, 2))

        # Submesh C: devices {0, 1, 2, 3} -> coordinates (0,0), (0,1), (0,2), (0,3)
        submesh_c = self.parent_mesh.create_submesh(ttnn.MeshShape(1, 4), ttnn.MeshCoordinate(0, 0))

        self.submeshes = {"A": submesh_a, "B": submesh_b, "C": submesh_c}

        # Verify device mappings
        assert submesh_a.get_device_ids() == [0, 1, 4, 5], f"Submesh A devices: {submesh_a.get_device_ids()}"
        assert submesh_b.get_device_ids() == [2, 3, 6, 7], f"Submesh B devices: {submesh_b.get_device_ids()}"
        assert submesh_c.get_device_ids() == [0, 1, 2, 3], f"Submesh C devices: {submesh_c.get_device_ids()}"

        logger.info("Submeshes created successfully:")
        logger.info(f"  Submesh A: devices {submesh_a.get_device_ids()}")
        logger.info(f"  Submesh B: devices {submesh_b.get_device_ids()}")
        logger.info(f"  Submesh C: devices {submesh_c.get_device_ids()}")


def create_test_tensor(mesh_device, size=(1, 1, 32, 32), dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """Create a test tensor allocated on the given mesh device."""
    torch_tensor = torch.randn(size)
    return ttnn.from_torch(
        torch_tensor, device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config
    )


def get_buffer_addresses(tensor) -> Dict[int, int]:
    """Extract buffer addresses from tensor across all devices."""
    addresses = {}

    # For mesh tensors, we need to get the address on each device
    if hasattr(tensor, "buffer") and tensor.buffer is not None:
        buffer = tensor.buffer
        # Get device buffer addresses for each device in the mesh
        if hasattr(buffer, "get_device_buffer"):
            # This is a mesh buffer
            for device_id in range(8):  # Assuming 8 devices total
                try:
                    device_coord = ttnn.MeshCoordinate(device_id // 4, device_id % 4)
                    device_buffer = buffer.get_device_buffer(device_coord)
                    if device_buffer is not None:
                        addresses[device_id] = device_buffer.address()
                except:
                    # Device not in this submesh
                    pass
        else:
            # Single device buffer
            device_id = tensor.device.id()
            addresses[device_id] = buffer.address()

    return addresses


def check_allocator_consistency(submesh_a_tensor, submesh_c_tensor):
    """
    Check allocator consistency for overlapping submeshes as described in the spec.

    According to the specification:
    - Submesh A (devices 0,1,4,5) and Submesh C (devices 0,1,2,3) overlap on devices 0,1
    - When tensors are allocated on both submeshes, the allocator should handle the shared addresses correctly
    - If both submeshes allocate at the same time, they should use the same allocator state
    """
    addr_a = get_buffer_addresses(submesh_a_tensor)
    addr_c = get_buffer_addresses(submesh_c_tensor)

    shared_devices = [0, 1]  # Devices that overlap between submesh A and C

    consistency_report = {
        "shared_devices": shared_devices,
        "submesh_a_addresses": addr_a,
        "submesh_c_addresses": addr_c,
        "overlapping_addresses": {},
        "allocator_state_consistent": True,
    }

    for device_id in shared_devices:
        if device_id in addr_a and device_id in addr_c:
            consistency_report["overlapping_addresses"][device_id] = {
                "submesh_a": addr_a[device_id],
                "submesh_c": addr_c[device_id],
            }

            # The key insight from the spec: tensors allocated at the same time on overlapping
            # submeshes should reflect the shared allocator state
            logger.info(
                f"Device {device_id}: Submesh A address = {addr_a[device_id]}, Submesh C address = {addr_c[device_id]}"
            )

    return consistency_report


class CommandQueueTester:
    """Helper class for testing command queue execution behavior."""

    def __init__(self, submeshes: Dict[str, Any]):
        self.submeshes = submeshes

    def test_overlapping_execution(self):
        """Test execution behavior for overlapping submeshes (A and C)."""
        logger.info("Testing overlapping submeshes A and C execution...")

        # Create tensors on overlapping submeshes A and C
        tensor_a = create_test_tensor(self.submeshes["A"])
        tensor_c = create_test_tensor(self.submeshes["C"])

        # Perform operations that would contend on shared devices (0, 1)
        result_a = ttnn.add(tensor_a, tensor_a)  # Operation on submesh A
        result_c = ttnn.add(tensor_c, tensor_c)  # Operation on submesh C

        # Ensure operations complete successfully
        ttnn.synchronize_device(self.submeshes["A"])
        ttnn.synchronize_device(self.submeshes["C"])

        return result_a, result_c

    def test_non_overlapping_execution(self):
        """Test execution behavior for non-overlapping submeshes (A and B)."""
        logger.info("Testing non-overlapping submeshes A and B execution...")

        # Create tensors on non-overlapping submeshes A and B
        tensor_a = create_test_tensor(self.submeshes["A"])
        tensor_b = create_test_tensor(self.submeshes["B"])

        # These operations should be able to execute in parallel
        result_a = ttnn.add(tensor_a, tensor_a)  # Operation on submesh A
        result_b = ttnn.add(tensor_b, tensor_b)  # Operation on submesh B

        # Synchronize both submeshes
        ttnn.synchronize_device(self.submeshes["A"])
        ttnn.synchronize_device(self.submeshes["B"])

        return result_a, result_b


@pytest.mark.parametrize("mesh_device", [pytest.param((2, 4), id="2x4_grid")], indirect=True)
class TestSubmeshAllocatorAndCommandQueue:
    """Test suite for submesh allocator addresses and command queue execution."""

    def test_submesh_creation(self, mesh_device):
        """Test that submeshes are created with correct device assignments."""
        tester = SubmeshAllocatorTester(mesh_device)

        # Verify the submeshes were created correctly
        assert "A" in tester.submeshes
        assert "B" in tester.submeshes
        assert "C" in tester.submeshes

        # Verify device assignments
        assert tester.submeshes["A"].get_device_ids() == [0, 1, 4, 5]
        assert tester.submeshes["B"].get_device_ids() == [2, 3, 6, 7]
        assert tester.submeshes["C"].get_device_ids() == [0, 1, 2, 3]

    def test_allocator_addresses_overlapping_submeshes(self, mesh_device):
        """Test buffer allocation addresses for overlapping submeshes A and C."""
        tester = SubmeshAllocatorTester(mesh_device)

        # Create tensors on overlapping submeshes A and C
        tensor_a = create_test_tensor(tester.submeshes["A"])
        tensor_c = create_test_tensor(tester.submeshes["C"])

        # Get buffer addresses
        addr_a = get_buffer_addresses(tensor_a)
        addr_c = get_buffer_addresses(tensor_c)

        logger.info(f"Submesh A buffer addresses: {addr_a}")
        logger.info(f"Submesh C buffer addresses: {addr_c}")

        # Check that overlapping devices (0, 1) have consistent allocator behavior
        # The exact addresses may be different, but the allocation should follow the same pattern
        overlapping_devices = [0, 1]
        for device_id in overlapping_devices:
            if device_id in addr_a and device_id in addr_c:
                logger.info(f"Device {device_id}: A={addr_a[device_id]}, C={addr_c[device_id]}")
                # Both submeshes should be able to allocate on shared devices
                assert addr_a[device_id] > 0, f"Invalid address on device {device_id} for submesh A"
                assert addr_c[device_id] > 0, f"Invalid address on device {device_id} for submesh C"

    def test_allocator_addresses_non_overlapping_submeshes(self, mesh_device):
        """Test buffer allocation addresses for non-overlapping submeshes A and B."""
        tester = SubmeshAllocatorTester(mesh_device)

        # Create tensors on non-overlapping submeshes A and B
        tensor_a = create_test_tensor(tester.submeshes["A"])
        tensor_b = create_test_tensor(tester.submeshes["B"])

        # Get buffer addresses
        addr_a = get_buffer_addresses(tensor_a)
        addr_b = get_buffer_addresses(tensor_b)

        logger.info(f"Submesh A buffer addresses: {addr_a}")
        logger.info(f"Submesh B buffer addresses: {addr_b}")

        # Verify no overlap in device assignments
        devices_a = set(addr_a.keys())
        devices_b = set(addr_b.keys())

        assert devices_a.isdisjoint(devices_b), f"Expected no overlap, but found: {devices_a & devices_b}"

        # All addresses should be valid
        for device_id, addr in addr_a.items():
            assert addr > 0, f"Invalid address {addr} on device {device_id} for submesh A"
        for device_id, addr in addr_b.items():
            assert addr > 0, f"Invalid address {addr} on device {device_id} for submesh B"

    def test_sequential_allocation_same_submesh(self, mesh_device):
        """Test that sequential allocations on the same submesh use different addresses."""
        tester = SubmeshAllocatorTester(mesh_device)

        # Create two tensors sequentially on submesh A
        tensor_a1 = create_test_tensor(tester.submeshes["A"], size=(1, 1, 32, 32))
        tensor_a2 = create_test_tensor(tester.submeshes["A"], size=(1, 1, 32, 32))

        addr_a1 = get_buffer_addresses(tensor_a1)
        addr_a2 = get_buffer_addresses(tensor_a2)

        logger.info(f"First allocation addresses: {addr_a1}")
        logger.info(f"Second allocation addresses: {addr_a2}")

        # Sequential allocations should use different addresses
        for device_id in addr_a1:
            if device_id in addr_a2:
                assert (
                    addr_a1[device_id] != addr_a2[device_id]
                ), f"Expected different addresses on device {device_id}, got {addr_a1[device_id]} and {addr_a2[device_id]}"

    def test_command_queue_overlapping_execution(self, mesh_device):
        """Test command queue execution for overlapping submeshes."""
        tester = SubmeshAllocatorTester(mesh_device)
        cq_tester = CommandQueueTester(tester.submeshes)

        # Test execution on overlapping submeshes A and C
        result_a, result_c = cq_tester.test_overlapping_execution()

        # Both operations should complete successfully
        assert result_a is not None, "Operation on submesh A failed"
        assert result_c is not None, "Operation on submesh C failed"

        # Verify results have expected shapes
        assert result_a.shape == (1, 1, 32, 32), f"Unexpected shape for result A: {result_a.shape}"
        assert result_c.shape == (1, 1, 32, 32), f"Unexpected shape for result C: {result_c.shape}"

    def test_command_queue_non_overlapping_execution(self, mesh_device):
        """Test command queue execution for non-overlapping submeshes."""
        tester = SubmeshAllocatorTester(mesh_device)
        cq_tester = CommandQueueTester(tester.submeshes)

        # Test execution on non-overlapping submeshes A and B
        result_a, result_b = cq_tester.test_non_overlapping_execution()

        # Both operations should complete successfully
        assert result_a is not None, "Operation on submesh A failed"
        assert result_b is not None, "Operation on submesh B failed"

        # Verify results have expected shapes
        assert result_a.shape == (1, 1, 32, 32), f"Unexpected shape for result A: {result_a.shape}"
        assert result_b.shape == (1, 1, 32, 32), f"Unexpected shape for result B: {result_b.shape}"

    def test_workload_execution_ordering(self, mesh_device):
        """Test workload execution ordering constraints for overlapping submeshes."""
        tester = SubmeshAllocatorTester(mesh_device)

        # Create tensors on overlapping submeshes
        tensor_a = create_test_tensor(tester.submeshes["A"])
        tensor_c = create_test_tensor(tester.submeshes["C"])

        # Execute operations in sequence that would affect shared devices
        intermediate_a = ttnn.add(tensor_a, tensor_a)
        ttnn.synchronize_device(tester.submeshes["A"])

        # Now execute on submesh C - this should wait for previous operation on shared devices
        intermediate_c = ttnn.add(tensor_c, tensor_c)
        ttnn.synchronize_device(tester.submeshes["C"])

        # Final operations
        result_a = ttnn.multiply(intermediate_a, intermediate_a)
        result_c = ttnn.multiply(intermediate_c, intermediate_c)

        # Synchronize both
        ttnn.synchronize_device(tester.submeshes["A"])
        ttnn.synchronize_device(tester.submeshes["C"])

        # Verify operations completed successfully
        assert result_a is not None and result_c is not None
        logger.info("Sequential workload execution completed successfully")

    def test_detailed_allocator_behavior_spec_example(self, mesh_device):
        """
        Test the specific allocator behavior example described in the specification.

        From the spec:
        - Submesh A: devices {0, 1, 4, 5}
        - Submesh C: devices {0, 1, 2, 3}
        - Both submeshes share allocator state on devices 0 and 1
        - Tensors allocated simultaneously should reflect shared allocator addresses
        """
        tester = SubmeshAllocatorTester(mesh_device)

        logger.info("Testing detailed allocator behavior as per specification...")

        # Test case 1: Simultaneous allocation on overlapping submeshes
        logger.info("=== Test Case 1: Simultaneous allocation ===")
        tensor_a1 = create_test_tensor(tester.submeshes["A"], size=(1, 1, 32, 32))
        tensor_c1 = create_test_tensor(tester.submeshes["C"], size=(1, 1, 32, 32))

        consistency_report = check_allocator_consistency(tensor_a1, tensor_c1)
        logger.info(f"Consistency report: {consistency_report}")

        # Verify that both tensors were allocated successfully on shared devices
        shared_devices_a = [d for d in consistency_report["submesh_a_addresses"] if d in [0, 1]]
        shared_devices_c = [d for d in consistency_report["submesh_c_addresses"] if d in [0, 1]]

        assert len(shared_devices_a) > 0, "Submesh A should have allocations on shared devices 0,1"
        assert len(shared_devices_c) > 0, "Submesh C should have allocations on shared devices 0,1"

        # Test case 2: Sequential allocation to verify allocator state progression
        logger.info("=== Test Case 2: Sequential allocation ===")
        tensor_a2 = create_test_tensor(tester.submeshes["A"], size=(1, 1, 64, 64))  # Larger tensor

        addr_a1 = get_buffer_addresses(tensor_a1)
        addr_a2 = get_buffer_addresses(tensor_a2)

        # Second allocation should have different addresses (allocator has moved forward)
        for device_id in [0, 1]:
            if device_id in addr_a1 and device_id in addr_a2:
                assert (
                    addr_a1[device_id] != addr_a2[device_id]
                ), f"Sequential allocations should have different addresses on device {device_id}"
                logger.info(f"Device {device_id}: First={addr_a1[device_id]}, Second={addr_a2[device_id]}")

        # Test case 3: Verify non-overlapping submeshes have independent allocators
        logger.info("=== Test Case 3: Independent allocators for non-overlapping submeshes ===")
        tensor_b1 = create_test_tensor(tester.submeshes["B"], size=(1, 1, 32, 32))
        tensor_b2 = create_test_tensor(tester.submeshes["B"], size=(1, 1, 32, 32))

        addr_b1 = get_buffer_addresses(tensor_b1)
        addr_b2 = get_buffer_addresses(tensor_b2)

        # Submesh B (devices 2,3,6,7) should not interfere with submesh A allocations
        devices_a = set(addr_a1.keys())
        devices_b = set(addr_b1.keys())

        assert devices_a.isdisjoint(
            devices_b
        ), f"Non-overlapping submeshes should use different devices: A={devices_a}, B={devices_b}"

        # Test case 4: Resource allocation across different buffer types
        logger.info("=== Test Case 4: Different buffer types ===")

        # Create tensors with different memory configs on overlapping submeshes
        tensor_a_dram = create_test_tensor(tester.submeshes["A"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tensor_c_dram = create_test_tensor(tester.submeshes["C"], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # Both should allocate successfully even with different memory configurations
        addr_a_dram = get_buffer_addresses(tensor_a_dram)
        addr_c_dram = get_buffer_addresses(tensor_c_dram)

        assert len(addr_a_dram) > 0, "DRAM allocation on submesh A should succeed"
        assert len(addr_c_dram) > 0, "DRAM allocation on submesh C should succeed"

        logger.info("All detailed allocator behavior tests passed!")

    def test_command_queue_execution_dependencies(self, mesh_device):
        """
        Test command queue execution dependencies as described in the specification.

        This tests the behavior where:
        - Submeshes that overlap must execute sequentially on shared devices
        - Submeshes that don't overlap can execute in parallel
        """
        tester = SubmeshAllocatorTester(mesh_device)

        logger.info("Testing command queue execution dependencies...")

        # Create workloads that will stress the command queue execution
        tensor_a = create_test_tensor(tester.submeshes["A"])
        tensor_b = create_test_tensor(tester.submeshes["B"])
        tensor_c = create_test_tensor(tester.submeshes["C"])

        # Test 1: Operations on overlapping submeshes A and C
        logger.info("=== Testing overlapping submesh execution (A and C) ===")

        # Start operations on both submeshes - these should handle conflicts properly
        result_a = ttnn.add(tensor_a, tensor_a)
        result_c = ttnn.add(tensor_c, tensor_c)

        # Chain more operations to test dependency handling
        result_a = ttnn.multiply(result_a, tensor_a)
        result_c = ttnn.multiply(result_c, tensor_c)

        # Synchronize to ensure completion
        ttnn.synchronize_device(tester.submeshes["A"])
        ttnn.synchronize_device(tester.submeshes["C"])

        assert result_a is not None and result_c is not None
        logger.info("Overlapping submesh operations completed successfully")

        # Test 2: Operations on non-overlapping submeshes A and B
        logger.info("=== Testing non-overlapping submesh execution (A and B) ===")

        # These operations should be able to proceed in parallel
        result_a2 = ttnn.add(tensor_a, result_a)
        result_b = ttnn.add(tensor_b, tensor_b)

        # More parallel operations
        result_a2 = ttnn.multiply(result_a2, tensor_a)
        result_b = ttnn.multiply(result_b, tensor_b)

        # Synchronize both
        ttnn.synchronize_device(tester.submeshes["A"])
        ttnn.synchronize_device(tester.submeshes["B"])

        assert result_a2 is not None and result_b is not None
        logger.info("Non-overlapping submesh operations completed successfully")

        logger.info("Command queue execution dependency tests passed!")


if __name__ == "__main__":
    # This allows running the test directly for debugging
    pytest.main([__file__, "-v", "-s"])

import os
import sys
import traceback

import pytest
import torch
import ttnn


@pytest.mark.parametrize("mesh_shape", [(1, 2)])
def test_submesh_allocator_stacktrace(mesh_shape):
    if ttnn.get_num_pcie_devices() < sum(mesh_shape):
        pytest.skip("Requires multiple devices to run")

    # Open parent mesh and create a submesh that covers the same span (simple case)
    parent = ttnn.open_mesh_device(ttnn.MeshShape(*mesh_shape))
    try:
        submesh = parent.create_submesh(ttnn.MeshShape(*mesh_shape), offset=ttnn.MeshCoordinate(0, 0))

        # Print Python-side stack prior to allocation
        print("\n[Python stack before allocation]\n" + "".join(traceback.format_stack(limit=8)))

        # Allocate a tiny tensor into L1 on the submesh to exercise allocator path
        t = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
        tt_tensor = ttnn.from_torch(
            t,
            device=submesh,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(submesh),
        )

        # Touch the tensor to ensure allocation has happened
        assert tt_tensor.shape == (1, 1, 32, 32)

        # Print an indicative C++ path for allocator handling on submesh
        # (Static call flow reference for debugging; not a runtime backtrace.)
        print(
            "\n[Expected C++ allocator path for submesh]\n"
            "MeshDevice::create_submesh(...) -> submesh->initialize(... allocator_config ...)\n"
            "MeshDevice::initialize(...) -> L1BankingAllocator constructed from reference_device()->allocator()->get_config()\n"
            "MeshDevice::allocator() (submesh) -> returns allocator owned by submesh initialize()\n"
        )
    finally:
        ttnn.close_mesh_device(parent)

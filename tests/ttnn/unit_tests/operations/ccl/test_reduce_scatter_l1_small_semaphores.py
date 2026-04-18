# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from loguru import logger


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 512}],
    indirect=True,
)
@pytest.mark.parametrize("use_l1_small", [False, True], ids=["l1_default", "l1_small"])
def test_reduce_scatter_l1_small_semaphores(mesh_device, device_params, use_l1_small):
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Test requires at least 2 devices")

    compute_grid = mesh_device.compute_with_storage_grid_size()
    num_worker_cores = compute_grid.x * compute_grid.y
    logger.info(f"Compute grid: {compute_grid.x}x{compute_grid.y} = {num_worker_cores} cores")

    # Tensor A: 0.8MB * num_worker_cores in L1
    tensor_a_bytes = int(0.8 * 1024 * 1024) * num_worker_cores
    tensor_a_elements = tensor_a_bytes // 2  # bfloat16
    tensor_a_cols = tensor_a_elements // 32
    logger.info(f"Tensor A: shape [1, 1, 32, {tensor_a_cols}], {tensor_a_bytes} bytes total")

    tensor_a = ttnn.from_torch(
        torch.randn(1, 1, 32, tensor_a_cols),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Tensor B: sharded across devices for reduce_scatter
    # dim=3 will be scattered, so input size must be divisible by num_devices
    tensor_b_cols = 1024  # must be divisible by num_devices
    tensor_b = ttnn.from_torch(
        torch.randn(1, 1, 512, tensor_b_cols),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Reduce-scatter — creates semaphores in L1 or L1_SMALL (cached in program cache)
    logger.info(f"Running reduce_scatter with use_l1_small_for_semaphores={use_l1_small}")
    output = ttnn.reduce_scatter(
        tensor_b,
        dim=3,
        topology=ttnn.Topology.Linear,
        use_l1_small_for_semaphores=use_l1_small,
    )
    ttnn.synchronize_device(mesh_device)

    # Free tensors A, B, and output
    tensor_a.deallocate(True)
    tensor_b.deallocate(True)
    output.deallocate(True)

    # Tensor C: 1.0MB * num_worker_cores in L1
    # With L1 semaphores: should fail (OOM — semaphores fragment L1)
    # With L1_SMALL semaphores: should succeed (semaphores in L1_SMALL, L1 is clean)
    tensor_c_bytes = int(1.0 * 1024 * 1024) * num_worker_cores
    tensor_c_elements = tensor_c_bytes // 2
    tensor_c_cols = tensor_c_elements // 32
    logger.info(f"Tensor C: shape [1, 1, 32, {tensor_c_cols}], {tensor_c_bytes} bytes total")

    if not use_l1_small:
        with pytest.raises(RuntimeError):
            tensor_c = ttnn.from_torch(
                torch.randn(1, 1, 32, tensor_c_cols),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
    else:
        tensor_c = ttnn.from_torch(
            torch.randn(1, 1, 32, tensor_c_cols),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1),
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        tensor_c.deallocate(True)

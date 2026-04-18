# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from loguru import logger


def print_buffers(device, name, buffer_type):
    buffers = ttnn._ttnn.reports.get_buffers(device)
    filtered_buffers = [buf for buf in buffers if buf.buffer_type == buffer_type]

    for i, buf in enumerate(filtered_buffers):
        logger.warning(
            f"{buffer_type} [{name}] Buffer {i}: addr={buf.address}, size={buf.max_size_per_bank}, layout={buf.buffer_layout}"
        )


def print_l1_buffers(device, name):
    print_buffers(device, name, ttnn.BufferType.L1)


def print_l1_small_buffers(device, name):
    print_buffers(device, name, ttnn.BufferType.L1_SMALL)


def run_ccl_op(ccl_op, tensor_b, use_l1_small):
    if ccl_op == "all_gather":
        return ttnn.all_gather(
            tensor_b,
            dim=3,
            topology=ttnn.Topology.Linear,
            use_l1_small_for_semaphores=use_l1_small,
        )
    elif ccl_op == "reduce_scatter":
        return ttnn.reduce_scatter(
            tensor_b,
            dim=3,
            topology=ttnn.Topology.Linear,
            use_l1_small_for_semaphores=use_l1_small,
        )
    else:
        raise ValueError(f"Unknown ccl_op: {ccl_op}")


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 512}],
    indirect=True,
)
@pytest.mark.parametrize("ccl_op", ["all_gather", "reduce_scatter"])
@pytest.mark.parametrize("use_l1_small", [False, True], ids=["l1_default", "l1_small"])
def test_ccl_l1_small_semaphores(mesh_device, device_params, ccl_op, use_l1_small):
    if mesh_device.get_num_devices() < 2:
        pytest.skip("Test requires at least 2 devices")

    compute_grid = mesh_device.compute_with_storage_grid_size()
    num_worker_cores = compute_grid.x * compute_grid.y
    logger.info(f"Compute grid: {compute_grid.x}x{compute_grid.y} = {num_worker_cores} cores")

    # Tensor A: 0.8MB * num_worker_cores in L1
    tensor_a_bytes = int(0.8 * 1024 * 1024) * num_worker_cores
    tensor_a_elements = tensor_a_bytes // 2  # bfloat16
    # Shape must be tile-aligned (multiples of 32)
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

    # Tensor B: [512, 1024] in DRAM
    # all_gather needs sharded input; reduce_scatter needs replicated input
    # (reduce_scatter with sharded [1,1,512,128] on dim=3 across 8 devices
    #  would produce [1,1,512,16] which is not tile-aligned)
    if ccl_op == "all_gather":
        mesh_mapper_b = ttnn.ShardTensorToMesh(mesh_device, dim=3)
    else:
        mesh_mapper_b = ttnn.ReplicateTensorToMesh(mesh_device)
    tensor_b = ttnn.from_torch(
        torch.randn(1, 1, 512, 1024),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM),
        mesh_mapper=mesh_mapper_b,
    )

    print_l1_buffers(mesh_device, f"before_{ccl_op}")
    print_l1_small_buffers(mesh_device, f"before_{ccl_op}")
    # CCL op — creates semaphores in L1 or L1_SMALL (cached in program cache)
    logger.info(f"Running {ccl_op} with use_l1_small_for_semaphores={use_l1_small}")
    output = run_ccl_op(ccl_op, tensor_b, use_l1_small)
    print_l1_buffers(mesh_device, f"after_{ccl_op}")
    print_l1_small_buffers(mesh_device, f"after_{ccl_op}")
    ttnn.synchronize_device(mesh_device)

    # Free tensors A, B, and output
    tensor_a.deallocate(True)
    tensor_b.deallocate(True)
    output.deallocate(True)

    print_l1_buffers(mesh_device, "before_tensor_c")
    print_l1_small_buffers(mesh_device, "before_tensor_c")

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
        print_l1_buffers(mesh_device, "after_tensor_c")
        print_l1_small_buffers(mesh_device, "after_tensor_c")

        tensor_c.deallocate(True)

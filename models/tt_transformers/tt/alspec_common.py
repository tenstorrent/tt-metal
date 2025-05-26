# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import ttnn
import torch

from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_global_semaphore_with_same_address,
)


def get_buffer_address(tensor):
    """
    Get the buffer address of a multi-device tensor
    """
    addr = []
    for i, ten in enumerate(ttnn.get_device_tensors(tensor)):
        addr.append(ten.buffer_address())
        if len(addr) > 0:
            assert addr[i - 1] == addr[i], f"Expected {addr[i-1]} == {addr[i]}"
    return addr[0]


class ALSpec:
    def __init__(
        self,
        mesh_device,
    ):
        """
        Class to manage Attention Level Speculation (ALSpec).
        """
        self.mesh_device = mesh_device
        self.cluster_shape = tuple(mesh_device.shape)

        ##################################
        ##### Set up fabric stuff
        ##################################
        logger.info("Set up fabric stuff for ALSpec")

        core_grid = mesh_device.compute_with_storage_grid_size()
        self.total_num_cores = core_grid.x * core_grid.y
        ALL_CORES = ttnn.num_cores_to_corerangeset(self.total_num_cores, core_grid, row_wise=True)

        self.swap_tensor_topology = ttnn.Topology.Linear
        self.worker_sub_device_id = ttnn.SubDeviceId(0)

        # create global semaphore handles
        self.num_sems = 2
        self.sem_idx = 0
        self.sem_handles = [
            create_global_semaphore_with_same_address(mesh_device, ALL_CORES, 0) for _ in range(self.num_sems)
        ]

        ##################################
        ##### Set up priority tensors
        ##################################
        logger.info("Set up priority tensors for ALSpec")

        priority_a = torch.ones((1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE))
        priority_b = torch.zeros((1, 1, ttnn.TILE_SIZE, ttnn.TILE_SIZE))

        # Priority tensor A from the perspective of the first device
        self.priority_a = ttnn.from_torch(
            torch.concat([priority_a, priority_b], dim=0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=self.cluster_shape),
        )

        # Priority tensor B from the perspective of the second device
        self.priority_b = ttnn.from_torch(
            torch.concat([priority_b, priority_a], dim=0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, None), mesh_shape=self.cluster_shape),
        )

        ##################################
        ##### Set up skip tensor
        ##################################
        logger.info("Set up skip tensor for ALSpec")

        skip_tensor = torch.ones((1, self.total_num_cores, ttnn.TILE_SIZE, ttnn.TILE_SIZE))
        skip_tensor_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=ALL_CORES,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        self.skip_tensor = ttnn.from_torch(
            skip_tensor,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.int32,
            memory_config=skip_tensor_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        self.skip_tensor_address = get_buffer_address(self.skip_tensor)

    def set_use_skip_tensor(self, state: bool):
        """
        Set the use of skip tensor for all devices in the mesh.
        """
        for d in self.skip_tensor.get_devices():
            d.set_speculation_mode(state, self.skip_tensor_address)

        logger.debug(f"Set speculation mode to {state}")

    def synchronize_tensor(self, tensor: ttnn.Tensor):
        """
        Synchronize a tensor across all devices in the mesh,
        based on the priority of each device.
        """
        # TODO: Implement persistent output tensor
        out = ttnn.experimental.swap_tensor_async(
            tensor,
            self.priority_a,
            self.priority_b,
            multi_device_global_semaphore=self.sem_handles[self.sem_idx],
            memory_config=tensor.memory_config(),
            topology=self.swap_tensor_topology,
            num_links=1,
            subdevice_id=self.worker_sub_device_id,
        )
        self.sem_idx = (self.sem_idx + 1) % self.num_sems

        return out

    def synchronize_priority(self):
        """
        Update each device such that they are aware of the priority of the other device.
        """
        # TODO: Implement persistent output tensor
        self.priority_b = ttnn.experimental.swap_tensor_async(
            self.priority_a,
            multi_device_global_semaphore=self.sem_handles[self.sem_idx],
            memory_config=self.priority_b.memory_config(),
            topology=self.swap_tensor_topology,
            num_links=1,
            subdevice_id=self.worker_sub_device_id,
        )
        self.sem_idx = (self.sem_idx + 1) % self.num_sems

    def update_skip_tensor(self):
        """
        Update the skip tensor using the priority of each device.

        1. Repeat the tensor for the number of cores, by using ttnn.concat
        2. Shard the tensor across the core grid
        3. Typecast to int32 and provide an output tensor (essentially a copy)
        """

        priority_ = ttnn.concat([self.priority_a] * (self.total_num_cores // 2), dim=1)
        priority_ = ttnn.concat([priority_] * 2, dim=1)

        priority_ = ttnn.to_memory_config(priority_, self.skip_tensor.memory_config())

        ttnn.typecast(priority_, ttnn.int32, output_tensor=self.skip_tensor)

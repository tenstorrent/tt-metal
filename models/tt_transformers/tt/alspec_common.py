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


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


class ALSpec:
    def __init__(
        self,
        mesh_device,
        head_dim,
        nh,
        k_chunk_size,
        lambda_=0.1,
        # lambda_=1000.0,
    ):
        """
        Class to manage Attention Level Speculation (ALSpec).
        """
        self.mesh_device = mesh_device
        self.cluster_shape = tuple(mesh_device.shape)

        ##################################
        ##### Set up speculative sdpa stuff
        ##################################
        self.lambda_ = lambda_
        self.scale = head_dim**-0.5

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

        sdpa_grid_size = (8, 7)
        padded_num_heads = padded_num_heads = nearest_pow_2(nearest_n(nh, n=32))
        self.program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=sdpa_grid_size,
            q_chunk_size=padded_num_heads,
            k_chunk_size=k_chunk_size,
            exp_approx_mode=False,
        )

        ##################################
        ##### Set up fabric stuff
        ##################################
        logger.info("Set up fabric stuff for ALSpec")

        core_grid = mesh_device.compute_with_storage_grid_size()
        self.total_num_cores = core_grid.x * core_grid.y
        ALL_CORES = ttnn.num_cores_to_corerangeset(self.total_num_cores, core_grid, row_wise=True)
        SINGLE_CORE = ttnn.num_cores_to_corerangeset(1, core_grid, row_wise=True)

        self.swap_tensor_topology = ttnn.Topology.Linear
        self.worker_sub_device = ttnn.SubDevice([ALL_CORES])
        self.worker_sub_device_id = ttnn.SubDeviceId(0)
        sub_device_manager_id = mesh_device.create_sub_device_manager([self.worker_sub_device], 0)
        mesh_device.load_sub_device_manager(sub_device_manager_id)

        # create global semaphore handles
        self.num_sems = 4
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
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=self.cluster_shape),
        )

        # Priority tensor B from the perspective of the second device
        self.priority_b = ttnn.from_torch(
            torch.concat([priority_b, priority_a], dim=0),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.int32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(None, 0), mesh_shape=self.cluster_shape),
        )

        self.priority_l1_mem_config = ttnn.create_sharded_memory_config(
            shape=(ttnn.TILE_SIZE, ttnn.TILE_SIZE),
            core_grid=SINGLE_CORE,
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
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
        self.reset_skip_tensor = ttnn.from_torch(
            skip_tensor,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.int32,
            memory_config=skip_tensor_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        self.skip_tensor_address = get_buffer_address(self.skip_tensor)
        self.reset_skip_tensor_address = get_buffer_address(self.reset_skip_tensor)

    def set_use_skip_tensor(self, state: bool):
        """
        Set the use of skip tensor for all devices in the mesh.
        """
        for d in self.skip_tensor.devices():
            d.set_speculation_mode(state, self.skip_tensor_address if state else self.reset_skip_tensor_address)

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
        priority_a_l1 = ttnn.to_memory_config(self.priority_a, self.priority_l1_mem_config)

        priority_b_l1 = ttnn.experimental.swap_tensor_async(
            priority_a_l1,
            multi_device_global_semaphore=self.sem_handles[self.sem_idx],
            memory_config=self.priority_l1_mem_config,
            topology=self.swap_tensor_topology,
            num_links=1,
            subdevice_id=self.worker_sub_device_id,
        )
        self.sem_idx = (self.sem_idx + 1) % self.num_sems

        priority_a_l1.deallocate(True)

        self.priority_b = ttnn.to_memory_config(priority_b_l1, self.priority_b.memory_config())
        priority_b_l1.deallocate(True)

    def update_skip_tensor(self, reset: bool = False):
        """
        Update the skip tensor using the priority of each device.

        1. Repeat the tensor for the number of cores, by using ttnn.concat
        2. Shard the tensor across the core grid
        3. Typecast to int32 and provide an output tensor (essentially a copy)
        """

        if reset:
            ttnn.bitwise_xor(self.reset_skip_tensor, 0, output_tensor=self.skip_tensor)
            return

        priority_ = ttnn.concat([self.priority_a] * (self.total_num_cores // 2), dim=1)
        priority_ = ttnn.concat([priority_] * 2, dim=1)

        priority_ = ttnn.to_memory_config(priority_, self.skip_tensor.memory_config())

        ttnn.typecast(priority_, ttnn.int32, output_tensor=self.skip_tensor)

    def speculative_sdpa_decode(
        self,
        Q,
        K,
        V,
        cur_pos_tensor,
    ):
        tt_Q = ttnn.to_memory_config(Q, ttnn.DRAM_MEMORY_CONFIG)
        tt_K = ttnn.to_memory_config(K, ttnn.DRAM_MEMORY_CONFIG)  # TODO: Check if no-op
        tt_V = ttnn.to_memory_config(V, ttnn.DRAM_MEMORY_CONFIG)  # TODO: Check if no-op

        # Run speculative flash decode
        outputs = ttnn.experimental.speculative_scaled_dot_product_attention_decode(
            tt_Q,
            tt_K,
            tt_V,
            lambda_=self.lambda_,
            cur_pos_tensor=cur_pos_tensor,
            scale=self.scale,
            program_config=self.program_config,
            compute_kernel_config=self.compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            priority_tensor=self.priority_a,
            other_priority_tensor=self.priority_b,
            ccl_enabled=True,
            multi_device_global_semaphore=self.sem_handles[self.sem_idx],
        )
        tt_back_gt_md, tt_back_spec_md, tt_back_spec_lp_distance_md, tt_back_lp_norm_x_md = outputs

        self.sem_idx = (self.sem_idx + 1) % self.num_sems

        return tt_back_gt_md

    def get_priority_on_host(self):
        """
        Get the priority tensors on the host.
        """

        priorities = ttnn.to_torch(
            self.priority_a,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 0), mesh_shape=self.cluster_shape),
        )
        priority_a_host = priorities[0, 0, 0, 0].item()
        priority_b_host = priorities[1, 0, 0, 0].item()

        assert priority_a_host != priority_b_host, "Priority tensors should not be equal"

        return (priority_a_host, priority_b_host)

    def get_correct_tensor(self, tensor: torch.Tensor, dim: int):
        """
        Get the correct tensor, depending on the priority of each device.
        """

        # Slice the tensor in half based on the dim
        tensor1, tensor2 = torch.split(tensor, tensor.shape[dim] // 2, dim=dim)

        priority1, priority2 = self.get_priority_on_host()

        return tensor1 if priority1 > priority2 else tensor2

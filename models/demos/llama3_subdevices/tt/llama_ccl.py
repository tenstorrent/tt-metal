# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
import torch

from tests.ttnn.unit_tests.operations.ccl.test_ccl_common import (
    create_and_load_sub_device_manager_with_fabric_interface,
    teardown_fabric_interface,
    create_global_semaphore_with_same_address,
)


class TT_CCL:
    def __init__(
        self,
        mesh_device,
        sub_device_crs,
        worker_sub_device_id,
        enable_persistent_fabric=True,
        create_persistent_fabric=True,
        teardown_persistent_fabric=True,
    ):
        self.mesh_device = mesh_device
        self.sub_device_crs = sub_device_crs
        self.worker_sub_device_id = worker_sub_device_id
        self.enable_persistent_fabric = enable_persistent_fabric
        self.create_persistent_fabric = create_persistent_fabric
        self.teardown_persistent_fabric = teardown_persistent_fabric

        if create_persistent_fabric:
            assert enable_persistent_fabric
        if teardown_persistent_fabric:
            assert enable_persistent_fabric

        self.num_cbs = 2
        self.from_remote_semaphore_handles = []
        self.to_remote_semaphore_handles = []

        # Double buffered on each axis
        self.gather_semaphore_handles = [[], []]
        for i in range(2):
            for _ in range(self.num_cbs):
                self.gather_semaphore_handles[i].append(
                    create_global_semaphore_with_same_address(self.mesh_device, self.sub_device_crs, 0)
                )

        self.gather_idx = [0, 0]
        self.buffer_idx = [0, 0]

        self.persistent_buffers = self.get_persistent_buffers()

    def get_persistent_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes.

        Creates double buffered persistent CCL buffers for each cluster axis.

        """

        persistent_buffers = [[], []]

        cluster_shape = (8, 4)
        M = 32
        num_cores = self.sub_device_crs.num_cores()

        # Create persistent buffers for cluster axis 0
        cluster_axis = 0
        N_per_shard = 2048 // 16 * cluster_shape[cluster_axis]  # FF2/DO
        buffer_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sub_device_crs,
                [M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        for _ in range(self.num_cbs):
            tt_buffer = ttnn.from_torch(
                torch.zeros((*cluster_shape, M, N_per_shard * num_cores)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=buffer_mem_cfg,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
            persistent_buffers[cluster_axis].append(tt_buffer)

        # Create persistent buffers for cluster axis 1
        cluster_axis = 1
        N_per_shard = 3840 // 24 * cluster_shape[cluster_axis]  # FF1/FF3/QKV
        buffer_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sub_device_crs,
                [M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        for _ in range(self.num_cbs):
            tt_buffer = ttnn.from_torch(
                torch.zeros((*cluster_shape, M, N_per_shard * num_cores)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=buffer_mem_cfg,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
            persistent_buffers[cluster_axis].append(tt_buffer)

        # Create persistent buffer for lm_head
        N_per_shard = (16 * 1024) // 16 * cluster_shape[cluster_axis]  # LM Head
        self.lm_head_buffer_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sub_device_crs,
                [M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        self.tt_lm_head_buffer = ttnn.from_torch(
            torch.zeros((*cluster_shape, M, N_per_shard * num_cores)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )

        return persistent_buffers

    def line_all_reduce(self, input_tensor_mesh, cluster_axis, num_links, memory_config, lm_head=False):
        if lm_head:
            persistent_buffer = ttnn.to_memory_config(self.tt_lm_head_buffer, self.lm_head_buffer_mem_cfg)
        else:
            persistent_buffer = self.persistent_buffers[cluster_axis][self.buffer_idx[cluster_axis]]

        output_tensor_mesh = ttnn.experimental.all_reduce_async(
            input_tensor_mesh,
            persistent_buffer,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            num_links=num_links,
            memory_config=memory_config,
            topology=ttnn.Topology.Linear,
            subdevice_id=self.worker_sub_device_id,
        )
        # ttnn.synchronize_devices(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])

        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        self.buffer_idx[cluster_axis] = (self.buffer_idx[cluster_axis] + 1) % self.num_cbs
        return output_tensor_mesh

    def line_reduce_scatter(
        self, input_tensor_mesh, memory_config, dim, cluster_axis, num_links=1, math_op=ttnn.ReduceType.Sum
    ):
        ttnn_tensor_out = ttnn.experimental.reduce_scatter_async(
            input_tensor_mesh,
            dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            from_remote_multi_device_global_semaphore=self.from_remote_semaphore_handles[self.from_sem_flag],
            to_remote_multi_device_global_semaphore=self.to_remote_semaphore_handles[self.to_sem_flag],
            math_op=math_op,
            memory_config=memory_config,
            topology=ttnn.Topology.Linear,
            num_links=num_links,
            subdevice_id=self.worker_sub_device_id,
        )
        self.from_sem_flag = (self.from_sem_flag + 1) % self.num_cbs
        self.to_sem_flag = (self.to_sem_flag + 1) % self.num_cbs
        # ttnn.synchronize_devices(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])
        return ttnn_tensor_out

    def line_all_gather(self, input_tensor_mesh, dim, cluster_axis, memory_config, num_links=1):
        ttnn_tensor_out = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Linear,
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            num_links=num_links,
            memory_config=memory_config,
            subdevice_id=self.worker_sub_device_id,
            enable_persistent_fabric_mode=self.enable_persistent_fabric,
        )
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        # ttnn.synchronize_devices(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])
        return ttnn_tensor_out

    def close(self):
        if self.enable_persistent_fabric and self.teardown_persistent_fabric:
            logger.info("Tearing down persistent fabric interface")
            self.mesh_device.reset_sub_device_stall_group()
            teardown_fabric_interface(self.mesh_device)
            logger.info("Done tearing down persistent fabric interface")


# def tt_all_reduce(input_tensor, mesh_device, cluster_axis=0, dim=0, num_links=2, memory_config=None, sharded=False):
def tt_all_reduce(
    input_tensor,
    mesh_device,
    cluster_axis=0,
    dim=0,
    num_reduce_scatter_links=1,
    num_all_gather_links=2,
    memory_config=None,
    sharded=False,
    dtype=ttnn.bfloat16,
    use_composite=False,
):
    # N150
    if list(mesh_device.shape) == [1, 1] or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Ensure dim 0 and 1 are 1
    original_shape = input_tensor.shape
    if original_shape[0] != 1 or original_shape[1] != 1:
        input_tensor = ttnn.reshape(
            input_tensor, (1, 1, original_shape[-4] * original_shape[-3] * original_shape[-2], original_shape[-1])
        )

    # N300 and T3K: reduce_scatter
    if 1 in list(mesh_device.shape):
        if input_tensor.is_sharded() and not sharded:
            input_tensor_sharded = input_tensor
            input_tensor = ttnn.sharded_to_interleaved(input_tensor_sharded, ttnn.L1_MEMORY_CONFIG)
            input_tensor_sharded.deallocate(True)
        reduced = ttnn.reduce_scatter(
            input_tensor,
            dim=dim,
            math_op=ttnn.ReduceType.Sum,
            num_links=num_reduce_scatter_links,
            memory_config=memory_config,
        )
        input_tensor.deallocate(True)
        return reduced

    # TG: all_reduce
    # Cast to CCL dtype
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)  # typecast and to interleaved
        if sharded and memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config, dtype)  # to sharded

    # Ensure the input tensor is in the correct memory configuration
    if not sharded:  # prefill
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    if not use_composite:
        gathered_tensor = ttnn.all_gather(
            input_tensor,
            dim,
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
        )

        if sharded:
            gathered_tensor = ttnn.to_memory_config(gathered_tensor, ttnn.L1_MEMORY_CONFIG)

        reduced_tensor = ttnn.experimental.fast_reduce_nc(
            gathered_tensor,
            dims=[dim],
            output=None,
            compute_kernel_config=None,
            memory_config=ttnn.L1_MEMORY_CONFIG if sharded else ttnn.DRAM_MEMORY_CONFIG,
        )
        gathered_tensor.deallocate(True)
    else:
        input_mem_cfg = input_tensor.memory_config()
        reduced_tensor = ttnn.reduce_scatter(
            input_tensor,
            dim=dim,
            num_links=num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            math_op=ttnn.ReduceType.Sum,
            topology=ttnn.Topology.Linear,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
        )

        reduced_tensor = ttnn.all_gather(
            reduced_tensor,
            dim,
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=ttnn.Topology.Linear,
            memory_config=input_mem_cfg,
        )

    # Reshape the reduced tensor to the original shape
    reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)

    return reduced_tensor


def tt_all_gather(
    input_tensor,
    mesh_device,
    cluster_axis,
    dim,
    num_links=2,
    memory_config=None,
    sharded=False,
    topology=ttnn.Topology.Linear,
    dtype=ttnn.bfloat16,
):
    # N150
    if list(mesh_device.shape) == (1, 1) or (cluster_axis == 1 and 1 in list(mesh_device.shape)):
        return input_tensor

    # Ensure the input tensor is in the correct memory configuration
    if not sharded:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.DRAM_MEMORY_CONFIG)

    # Cast to CCL dtype
    if input_tensor.dtype != dtype:
        input_tensor = ttnn.to_memory_config(input_tensor, ttnn.L1_MEMORY_CONFIG, dtype)  # typecast and to interleaved
        if sharded and memory_config is not None:
            input_tensor = ttnn.to_memory_config(input_tensor, memory_config, dtype)  # to sharded

    if cluster_axis is None:
        gathered = ttnn.all_gather(
            input_tensor,
            dim,
            num_links=num_links,
            topology=topology,
            memory_config=memory_config,
        )
    else:
        gathered = ttnn.all_gather(
            input_tensor,
            dim,
            num_links=num_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            memory_config=memory_config,
        )
    input_tensor.deallocate(True)
    return gathered


def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, compute_kernel_config):
    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
    padded_shape = (1, 1, inp.shape[-2], 32)
    tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape, padded_shape))  # TODO: Figure out why we need this
    tt_stats_gathered = tt_all_gather(
        tt_stats,
        mesh_device=mesh_device,
        dim=3,
        cluster_axis=1,
        num_links=1,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_stats.deallocate(True)

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats_gathered, epsilon=epsilon, weight=gamma, compute_kernel_config=compute_kernel_config
    )

    tt_stats_gathered.deallocate(True)
    # inp.deallocate(True)

    return tt_out


def tt_sharded_distributed_rmsnorm(
    inp,
    res,
    epsilon,
    gamma,
    mesh_device,
    ln_sharded_input_memcfg,
    ln_sharded_progcfg,
    ln_sharded_stats_memcfg,
    tt_ccl=None,
):
    # inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, residual_input_tensor=res, program_config=ln_sharded_progcfg)
    # print("tt_stats")
    # All gather stats
    # tt_stats = ttnn.all_gather(
    #     tt_stats,
    #     3,
    #     num_links=1,
    #     cluster_axis=1,
    #     mesh_device=mesh_device,
    #     memory_config=ln_sharded_stats_memcfg,
    #     topology=ttnn.Topology.Linear,
    # )
    # tt_stats_dram = ttnn.to_memory_config(tt_stats, ttnn.DRAM_MEMORY_CONFIG)
    grid_offset = ttnn.CoreCoord(1, 0)
    tt_stats_sharded_config = ttnn.create_sharded_memory_config(
        shape=(32, 128),
        core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(grid_offset, grid_offset)]),
        strategy=ttnn.ShardStrategy.WIDTH,
        use_height_and_width_as_shard_shape=True,
    )
    # ttnn.deallocate(tt_stats)
    # print("mem cfg")
    tt_global_stats_sharded = tt_ccl.line_all_gather(
        tt_stats, dim=3, cluster_axis=1, num_links=1, memory_config=tt_stats_sharded_config
    )
    # ttnn.synchronize_devices(tt_ccl.mesh_device, sub_device_ids=[tt_ccl.worker_sub_device_id])
    # ttnn.deallocate(tt_stats_dram)
    # print("all gather stats", tt_global_stats.shape)

    # tt_global_stats_sharded = ttnn.to_memory_config(tt_global_stats, memory_config=tt_stats_sharded_config)
    ttnn.deallocate(tt_stats)
    # print("sharded stats")

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp,
        epsilon=epsilon,
        weight=gamma,
        program_config=ln_sharded_progcfg,
        stats=tt_global_stats_sharded,
    )
    ttnn.deallocate(tt_global_stats_sharded)
    # print("rmsnorm post all gather", tt_out.shape)
    return tt_out, inp

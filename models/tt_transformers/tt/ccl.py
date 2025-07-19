# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device

        self.worker_sub_device_id = ttnn.SubDeviceId(0)
        self.sub_device_crs = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(
                        self.mesh_device.compute_with_storage_grid_size().x - 1,
                        self.mesh_device.compute_with_storage_grid_size().y - 1,
                    ),
                )
            }
        )

        self.ag_semaphores_idx = 0
        self.ag_semaphore_handles = [[], []]

        self.rs_semaphores_idx = 0
        self.rs_semaphore_handles = [[], []]

        for i in range(2):
            for _ in range(2):
                self.ag_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
            for _ in range(3):
                self.rs_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

        self.ag_persistent_buffers = self.create_ag_persistent_buffers()  # dict[PersistentBufferKey, ttnn_tensor]
        self.rs_persistent_buffers = (
            self.create_rs_persistent_buffers()
        )  # dict[tuple(PersistentBufferKey, PersistentBufferKey), ttnn_tensor] - intermediate_buffer, output_buffer

        # TODO: Is this setup correct?
        worker_sub_device = ttnn.SubDevice([self.sub_device_crs])
        sub_device_manager = self.mesh_device.create_sub_device_manager([worker_sub_device], 0)
        self.mesh_device.load_sub_device_manager(sub_device_manager)
        self.mesh_device.set_sub_device_stall_group([self.worker_sub_device_id])

    def get_and_cycle_ag_semaphore_handles(self):
        current_idx = self.ag_semaphores_idx
        self.ag_semaphores_idx = (self.ag_semaphores_idx + 1) % 2
        return self.ag_semaphore_handles[current_idx]

    def get_and_cycle_rs_semaphore_handles(self):
        current_idx = self.rs_semaphores_idx
        self.rs_semaphores_idx = (self.rs_semaphores_idx + 1) % 2
        return self.rs_semaphore_handles[current_idx]

    def close(self):
        self.mesh_device.reset_sub_device_stall_group()


def tt_all_reduce(
    input_tensor,
    mesh_device,
    tt_ccl,
    cluster_axis=0,
    dim=0,
    num_reduce_scatter_links=1,
    num_all_gather_links=2,
    topology=ttnn.Topology.Linear,
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
        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
            num_links=num_reduce_scatter_links,
            memory_config=memory_config,
            topology=topology,
            subdevice_id=tt_ccl.worker_sub_device_id,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
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
        gathered_tensor = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            subdevice_id=tt_ccl.worker_sub_device_id,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
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

        if not tt_ccl.is_using_preallocated_persistent_buffers():
            gathered_tensor.deallocate(True)
    else:
        input_mem_cfg = input_tensor.memory_config()
        reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
            num_links=num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            topology=topology,
            subdevice_id=tt_ccl.worker_sub_device_id,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        reduced_tensor = ttnn.experimental.all_gather_async(
            reduced_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            memory_config=input_mem_cfg,
            subdevice_id=tt_ccl.worker_sub_device_id,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

    # Reshape the reduced tensor to the original shape
    reduced_tensor = ttnn.reshape(reduced_tensor, original_shape)

    return reduced_tensor


def tt_all_gather(
    input_tensor,
    mesh_device,
    tt_ccl,
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
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=num_links,
            topology=topology,
            memory_config=memory_config,
            subdevice_id=tt_ccl.worker_sub_device_id,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
    else:
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=None,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=num_links,
            cluster_axis=cluster_axis,
            mesh_device=mesh_device,
            topology=topology,
            memory_config=memory_config,
            subdevice_id=tt_ccl.worker_sub_device_id,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
    input_tensor.deallocate(True)
    return gathered


def tt_distributed_rmsnorm(inp, epsilon, gamma, mesh_device, tt_ccl, compute_kernel_config):
    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
    padded_shape = (1, 1, inp.shape[-2], 32)
    tt_stats = ttnn.reshape(tt_stats, ttnn.Shape(padded_shape))  # TODO: Figure out why we need this
    tt_stats_gathered = tt_all_gather(
        tt_stats,
        mesh_device=mesh_device,
        tt_ccl=tt_ccl,
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

    if not tt_ccl.is_using_preallocated_persistent_buffers():
        tt_stats_gathered.deallocate(True)
    # inp.deallocate(True)

    return tt_out


def tt_sharded_distributed_rmsnorm(
    inp, epsilon, gamma, mesh_device, tt_ccl, ln_sharded_input_memcfg, ln_sharded_progcfg, ln_sharded_stats_memcfg
):
    inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, program_config=ln_sharded_progcfg)

    # All gather stats
    tt_stats = ttnn.experimental.all_gather_async(
        tt_stats,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
        num_links=1,
        cluster_axis=1,
        mesh_device=mesh_device,
        topology=ttnn.Topology.Linear,
        memory_config=ln_sharded_stats_memcfg,
        subdevice_id=tt_ccl.worker_sub_device_id,
        chunks_per_sync=10,
        num_workers_per_link=2,
        num_buffers_per_channel=2,
    )

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp,
        epsilon=epsilon,
        weight=gamma,
        program_config=ln_sharded_progcfg,
        stats=tt_stats,
    )
    if not tt_ccl.is_using_preallocated_persistent_buffers():
        tt_stats.deallocate(True)

    return tt_out

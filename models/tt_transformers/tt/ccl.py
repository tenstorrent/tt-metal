# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.utility_functions import is_blackhole


class TT_CCL:
    def __init__(
        self,
        mesh_device,
    ):
        self.mesh_device = mesh_device
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

        self.barrier_semaphore_idx = 0
        self.barrier_semaphore_handles = []

        self.ag_semaphores_idx = 0
        self.ag_semaphore_handles = [[], []]

        self.rs_semaphores_idx = 0
        self.rs_semaphore_handles = [[], []]

        for i in range(2):
            self.barrier_semaphore_handles.append(
                ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
            )
            for _ in range(2):
                self.ag_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
            for _ in range(3):
                self.rs_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

    def get_and_cycle_barrier_semaphore_handle(self):
        current_idx = self.barrier_semaphore_idx
        self.barrier_semaphore_idx = (self.barrier_semaphore_idx + 1) % 2
        return self.barrier_semaphore_handles[current_idx]

    def get_and_cycle_ag_semaphore_handles(self):
        current_idx = self.ag_semaphores_idx
        self.ag_semaphores_idx = (self.ag_semaphores_idx + 1) % 2
        return self.ag_semaphore_handles[current_idx]

    def get_and_cycle_rs_semaphore_handles(self):
        current_idx = self.rs_semaphores_idx
        self.rs_semaphores_idx = (self.rs_semaphores_idx + 1) % 2
        return self.rs_semaphore_handles[current_idx]


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

        # TODO: 26411
        # Remove this blackhole condition once fabric CCLs are working on blackhole
        if is_blackhole():
            reduced = ttnn.reduce_scatter(
                input_tensor,
                dim=dim,
                math_op=ttnn.ReduceType.Sum,
                num_links=num_reduce_scatter_links,
                topology=topology,
                memory_config=memory_config,
            )
        else:
            reduced = ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor,
                persistent_output_buffers=None,
                dim=dim,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=num_reduce_scatter_links,
                memory_config=memory_config,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
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
        # TODO: 26411
        # Remove this blackhole condition once fabric CCLs are working on blackhole
        if is_blackhole():
            gathered_tensor = ttnn.all_gather(
                input_tensor,
                dim,
                num_links=num_all_gather_links,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                topology=topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            )
        else:
            gathered_tensor = ttnn.experimental.all_gather_async(
                input_tensor,
                persistent_output_buffer=None,
                dim=dim,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=num_all_gather_links,
                cluster_axis=cluster_axis,
                topology=topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
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

        gathered_tensor.deallocate(True)
    else:
        input_mem_cfg = input_tensor.memory_config()
        # TODO: 26411
        # Remove this blackhole condition once fabric CCLs are working on blackhole
        if is_blackhole():
            reduced_tensor = ttnn.reduce_scatter(
                input_tensor,
                dim=dim,
                num_links=num_reduce_scatter_links,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                math_op=ttnn.ReduceType.Sum,
                topology=topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            )

            reduced_tensor = ttnn.all_gather(
                reduced_tensor,
                dim,
                num_links=num_all_gather_links,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                topology=topology,
                memory_config=input_mem_cfg,
            )
        else:
            reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor,
                persistent_output_buffers=None,
                dim=dim,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(),
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                num_links=num_reduce_scatter_links,
                cluster_axis=cluster_axis,
                memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=topology,
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
                topology=topology,
                memory_config=input_mem_cfg,
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
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
        # TODO: 26411
        # Remove this blackhole condition once fabric CCLs are working on blackhole
        if is_blackhole():
            gathered = ttnn.all_gather(
                input_tensor,
                dim,
                num_links=num_links,
                topology=topology,
                memory_config=memory_config,
            )
        else:
            gathered = ttnn.experimental.all_gather_async(
                input_tensor,
                persistent_output_buffer=None,
                dim=dim,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=num_links,
                topology=topology,
                memory_config=memory_config,
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
    else:
        # TODO: 26411
        # Remove this blackhole condition once fabric CCLs are working on blackhole
        if is_blackhole():
            gathered = ttnn.all_gather(
                input_tensor,
                dim,
                num_links=num_links,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                topology=topology,
                memory_config=memory_config,
            )
        else:
            gathered = ttnn.experimental.all_gather_async(
                input_tensor,
                persistent_output_buffer=None,
                dim=dim,
                multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=num_links,
                cluster_axis=cluster_axis,
                topology=topology,
                memory_config=memory_config,
                barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
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

    # TODO: 26411
    # Remove this blackhole condition once fabric CCLs are working on blackhole
    if is_blackhole():
        tt_stats = ttnn.all_gather(
            tt_stats,
            3,
            num_links=1,
            cluster_axis=1,
            mesh_device=mesh_device,
            memory_config=ln_sharded_stats_memcfg,
            topology=ttnn.Topology.Linear,
        )
    else:
        tt_stats = ttnn.experimental.all_gather_async(
            tt_stats,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=1,
            cluster_axis=1,
            topology=ttnn.Topology.Linear,
            memory_config=ln_sharded_stats_memcfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
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
    tt_stats.deallocate(True)

    return tt_out


# TODO: #26351
# Fabric AG currently doesn't support gathering on a padded dim 3
# This functionality is in development, and when ready should replace this reshape workaround
def ag_on_padded_dim_3(inp, mesh_device, tt_ccl, is_galaxy, cluster_axis, num_links, topology):
    ag_memory_config = inp.memory_config()
    output_memory_config = inp.memory_config()
    input_shape = inp.shape
    reshape_required = input_shape[3] % 32 != 0

    # TODO: 26411
    # Remove this blackhole condition once fabric CCLs are working on blackhole
    if is_blackhole():
        if is_galaxy:
            output_tensor = ttnn.all_gather(
                inp,
                dim=3,
                num_links=num_links,
                cluster_axis=cluster_axis,
                mesh_device=mesh_device,
                topology=topology,
            )
        else:
            output_tensor = ttnn.all_gather(
                inp,
                dim=3,
                num_links=num_links,
                topology=topology,
            )
    else:
        if reshape_required:
            gather_dim_input_tensor_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
            if gather_dim_input_tensor_size % 32 != 0:
                assert False, "AG does not support gathering on padded dim 3"

            ag_memory_config = ttnn.DRAM_MEMORY_CONFIG
            inp = ttnn.reshape(
                inp,
                (1, 1, 1, gather_dim_input_tensor_size),
                memory_config=ag_memory_config,
            )

        ag_output = ttnn.experimental.all_gather_async(
            inp,
            persistent_output_buffer=None,
            dim=3,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(),
            num_links=num_links,
            memory_config=ag_memory_config,
            cluster_axis=cluster_axis,
            topology=topology,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(),
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        ttnn.deallocate(inp)

        output_tensor = ag_output
        if reshape_required:
            split_size = gather_dim_input_tensor_size
            split_tensors = ttnn.split(ag_output, split_size=split_size, dim=3)
            split_tensors = [ttnn.reshape(tensor, input_shape) for tensor in split_tensors]

            output_tensor = ttnn.concat(split_tensors, dim=3)
            output_tensor = ttnn.to_memory_config(output_tensor, output_memory_config)

    return output_tensor

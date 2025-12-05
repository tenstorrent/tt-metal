# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


class TT_CCL:
    def __init__(
        self,
        mesh_device,
        mode="decode",
    ):
        self.mesh_device = mesh_device
        self.mode = mode
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

        self.barrier_semaphore_idx = [0, 0, 0]
        self.barrier_semaphore_handles = [[], [], []]

        self.ag_semaphores_idx = [0, 0, 0]
        self.ag_semaphore_handles = [[], [], []]

        self.rs_semaphores_idx = [0, 0, 0]
        self.rs_semaphore_handles = [[], [], []]

        # cluster-axis-0, cluster-axis-1, no-cluster-axis
        for i in range(3):
            # double buffered semaphores
            for _ in range(2):
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

                self.ag_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
                )

                self.rs_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
                )

        # Initialize persistent buffers for decode mode
        self.persistent_ag_buffers = {}
        self.persistent_rs_buffers = {}
        if mode == "decode":
            self._init_decode_persistent_buffers()

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis=None):
        semaphore_index = 2 if not cluster_axis else cluster_axis
        current_idx = self.barrier_semaphore_idx[semaphore_index]
        self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % 2
        return self.barrier_semaphore_handles[semaphore_index][current_idx]

    def get_and_cycle_ag_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if not cluster_axis else cluster_axis
        current_idx = self.ag_semaphores_idx[semaphore_index]
        self.ag_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.ag_semaphore_handles[semaphore_index][current_idx]

    def get_and_cycle_rs_semaphore_handles(self, cluster_axis=None):
        semaphore_index = 2 if not cluster_axis else cluster_axis
        current_idx = self.rs_semaphores_idx[semaphore_index]
        self.rs_semaphores_idx[semaphore_index] = (current_idx + 1) % 2
        return self.rs_semaphore_handles[semaphore_index][current_idx]

    def _init_decode_persistent_buffers(self):
        """Initialize persistent buffers for decode mode CCL operations

        Allocates buffers for all CCL operations used in decode:
        - QKV_OUT: attention QKV all_reduce
        - ATTN_OUT: attention output all_gather
        - MLP_W2_OUT: MLP w2 all_reduce
        - PRE_FF_NORM: pre-feedforward norm all_reduce
        - POST_FF_NORM: post-feedforward norm all_reduce
        """
        import torch

        # For multi-device configurations, allocate persistent buffers
        # Single device (N150) doesn't need persistent buffers
        if list(self.mesh_device.shape) == [1, 1]:
            return

        # Common shape for batch_size=32, dim=4096
        common_shape = (1, 1, 32, 4096)

        buffer_configs = {
            # Attention QKV all_reduce (cluster_axis=1)
            "QKV_OUT": (common_shape, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
            # Attention output all_gather (cluster_axis=1)
            "ATTN_OUT": (common_shape, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
            # MLP w2 all_reduce (cluster_axis=0)
            "MLP_W2_OUT": (common_shape, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
            # Pre-feedforward norm all_reduce (cluster_axis=0)
            "PRE_FF_NORM": (common_shape, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
            # Post-feedforward norm all_reduce (cluster_axis=0)
            "POST_FF_NORM": (common_shape, ttnn.bfloat16, ttnn.DRAM_MEMORY_CONFIG),
        }

        for key, (shape, dtype, mem_cfg) in buffer_configs.items():
            try:
                buffer = ttnn.from_torch(
                    torch.zeros(shape, dtype=torch.bfloat16),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=dtype,
                    memory_config=mem_cfg,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
                self.persistent_ag_buffers[key] = buffer
                print(f"[TT_CCL] Allocated persistent buffer for {key}")
            except Exception as e:
                print(f"Warning: Failed to allocate persistent buffer for {key}: {e}")

    def all_reduce(
        self,
        input_tensor,
        cluster_axis=0,
        dim=0,
        num_reduce_scatter_links=1,
        num_all_gather_links=2,
        topology=ttnn.Topology.Linear,
        memory_config=None,
        sharded=False,
        dtype=ttnn.bfloat16,
        use_composite=False,
        buffer_key=None,
    ):
        """Wrapper for tt_all_reduce that uses persistent buffers in decode mode"""
        return tt_all_reduce(
            input_tensor,
            self.mesh_device,
            self,
            cluster_axis=cluster_axis,
            dim=dim,
            num_reduce_scatter_links=num_reduce_scatter_links,
            num_all_gather_links=num_all_gather_links,
            topology=topology,
            memory_config=memory_config,
            sharded=sharded,
            dtype=dtype,
            use_composite=use_composite,
            buffer_key=buffer_key,
        )

    def all_gather(
        self,
        input_tensor,
        cluster_axis,
        dim,
        num_links=2,
        memory_config=None,
        sharded=False,
        topology=ttnn.Topology.Linear,
        dtype=ttnn.bfloat16,
        buffer_key=None,
    ):
        """Wrapper for tt_all_gather that uses persistent buffers in decode mode"""
        return tt_all_gather(
            input_tensor,
            self.mesh_device,
            self,
            cluster_axis=cluster_axis,
            dim=dim,
            num_links=num_links,
            memory_config=memory_config,
            sharded=sharded,
            topology=topology,
            dtype=dtype,
            buffer_key=buffer_key,
        )


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
    buffer_key=None,
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

        # Get persistent buffer if available and in decode mode
        persistent_buffer = None
        if tt_ccl.mode == "decode" and buffer_key is not None:
            persistent_buffer = tt_ccl.persistent_rs_buffers.get(buffer_key, None)
            print("persistent buffer being used")

        reduced = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=persistent_buffer,
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
        # Get persistent buffer if available and in decode mode
        persistent_buffer = None
        if tt_ccl.mode == "decode" and buffer_key is not None:
            persistent_buffer = tt_ccl.persistent_ag_buffers.get(buffer_key, None)

        gathered_tensor = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=persistent_buffer,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
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

        # Get persistent buffer for reduce_scatter if available
        rs_persistent_buffer = None
        if tt_ccl.mode == "decode" and buffer_key is not None:
            rs_persistent_buffer = tt_ccl.persistent_rs_buffers.get(buffer_key, None)

        reduced_tensor = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor,
            persistent_output_buffers=rs_persistent_buffer,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis),
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=num_reduce_scatter_links,
            cluster_axis=cluster_axis,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if not sharded else memory_config,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=topology,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )

        # Get persistent buffer for all_gather if available
        ag_persistent_buffer = None
        if tt_ccl.mode == "decode" and buffer_key is not None:
            ag_persistent_buffer = tt_ccl.persistent_ag_buffers.get(buffer_key, None)

        reduced_tensor = ttnn.experimental.all_gather_async(
            reduced_tensor,
            persistent_output_buffer=ag_persistent_buffer,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_all_gather_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=input_mem_cfg,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
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
    buffer_key=None,
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

    # Get persistent buffer if available and in decode mode
    persistent_buffer = None
    if tt_ccl.mode == "decode" and buffer_key is not None:
        persistent_buffer = tt_ccl.persistent_ag_buffers.get(buffer_key, None)

    if cluster_axis is None:
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=persistent_buffer,
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
        gathered = ttnn.experimental.all_gather_async(
            input_tensor,
            persistent_output_buffer=persistent_buffer,
            dim=dim,
            multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
            num_links=num_links,
            cluster_axis=cluster_axis,
            topology=topology,
            memory_config=memory_config,
            barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
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
    cluster_axis = 1
    tt_stats = ttnn.experimental.all_gather_async(
        tt_stats,
        persistent_output_buffer=None,
        dim=3,
        multi_device_global_semaphore=tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis),
        num_links=1,
        cluster_axis=cluster_axis,
        topology=ttnn.Topology.Linear,
        memory_config=ln_sharded_stats_memcfg,
        barrier_semaphore=tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis),
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

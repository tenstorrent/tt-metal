# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import os

is_RING_6U = os.environ.get("RING_6U", "0") == "1"


class TT_CCL:
    def __init__(
        self,
        mesh_device,
        model_args,
        worker_sub_device_id,
        mode="decode",
        allocate_prefill_buffers=True,
    ):
        self.mode = mode
        all_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])

        self.mesh_device = mesh_device
        self.sub_device_crs = all_crs if mode == "prefill" else model_args.sub_core_grids
        self.worker_sub_device_id = worker_sub_device_id
        self.model_config = model_args.model_config
        self.weight_cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        self.num_cbs = 2
        self.from_remote_semaphore_handles = []
        self.to_remote_semaphore_handles = []
        self.all_gather_concat_inter_tensor = self.get_all_gather_concat_inter_buffer()

        # Double buffered on each axis
        self.gather_semaphore_handles = [[], []]
        if mode == "prefill":
            self.from_semaphore_handles = [[], []]
            self.to_semaphore_handles = [[], []]
        for i in range(2):
            for _ in range(self.num_cbs):
                self.gather_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )
                if mode == "prefill":
                    self.from_semaphore_handles[i].append(
                        ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                    )
                    self.to_semaphore_handles[i].append(
                        ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                    )

        self.gather_idx = [0, 0]
        self.reduce_scatter_buffer_idx = [0, 0]
        self.persistent_buffers = {}
        self.all_gather_buffers = {}
        if mode == "decode":
            self.persistent_buffers = self.get_persistent_buffers()
            self.all_gather_buffers = self.get_all_gather_buffers()
            self.reduce_scatter_buffers = self.get_decode_reduce_scatter_buffers()
            self.rs_create_heads_buffers = self.get_decode_rs_create_heads_buffers()
        if mode == "prefill":
            self.support_seqlens = [8192, 4096, 1024, 2048, 128]
            if allocate_prefill_buffers:
                self.persistent_buffers = self.get_prefill_reduce_scatter_buffers()
                self.all_gather_buffers = self.get_prefill_all_gather_buffers()

            else:
                for seqlen in self.support_seqlens:
                    self.persistent_buffers[seqlen] = {}
                    self.all_gather_buffers[seqlen] = {}

    def reset_gather_and_buffer_idx(self):
        self.gather_idx = [0, 0]
        self.reduce_scatter_buffer_idx = [0, 0]

    def get_all_gather_concat_inter_buffer(self):
        intermediate_core_range_set = ttnn.CoreRangeSet(
            [
                ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 4)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 6), ttnn.CoreCoord(6, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 7), ttnn.CoreCoord(6, 7)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 9), ttnn.CoreCoord(6, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 0), ttnn.CoreCoord(6, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 1), ttnn.CoreCoord(6, 1)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 2), ttnn.CoreCoord(6, 2)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 4), ttnn.CoreCoord(6, 4)),
                ttnn.CoreRange(ttnn.CoreCoord(6, 5), ttnn.CoreCoord(6, 5)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 5), ttnn.CoreCoord(5, 5)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 6), ttnn.CoreCoord(5, 6)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 7), ttnn.CoreCoord(5, 7)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 9), ttnn.CoreCoord(5, 9)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(5, 0)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 1), ttnn.CoreCoord(5, 1)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 2), ttnn.CoreCoord(5, 2)),
                ttnn.CoreRange(ttnn.CoreCoord(5, 4), ttnn.CoreCoord(5, 4)),
                ttnn.CoreRange(ttnn.CoreCoord(1, 5), ttnn.CoreCoord(1, 5)),
            ]
        )
        intermediate_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                intermediate_core_range_set,
                [32, 128],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        temp_shape = [8, 128, 32, 128]
        intermediate_tensor = torch.zeros(temp_shape, dtype=torch.bfloat16)
        tt_intermediate_tensor = ttnn.from_torch(
            intermediate_tensor,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=intermediate_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=[0, 1], mesh_shape=[8, 4]),
        )
        tt_intermediate_tensors = [tt_intermediate_tensor]
        return tt_intermediate_tensors

    def get_all_gather_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes.

        Here are the current persistent buffers generated by this fuction:
        - SDPA: (1, 32, 32, 128)
        - LAYERNORM: (1, 1, 32, 128)
        - SAMPLING_VALUES: (1, 1, 32, 256)
        - SAMPLING_INDICES: (1, 1, 32, 256)
        - BINARY_MUL: (1, 1, 32, 3584)

        """

        persistent_buffers = {}

        if self.model_config is None:
            return persistent_buffers

        M = 32

        # SDPA
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 32, M, 128)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["GATHER_USERS_MEMCFG"](list(self.mesh_device.shape)[1]),
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["SDPA"] = tt_buffer

        # Layernorm
        grid_offset = ttnn.CoreCoord(1, 0)
        tt_stats_sharded_config = ttnn.create_sharded_memory_config(
            shape=(32, 128),
            core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(grid_offset, grid_offset)]),
            strategy=ttnn.ShardStrategy.WIDTH,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 1, M, 128)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=tt_stats_sharded_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["LAYERNORM"] = tt_buffer

        # Sampling values
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 32, 256)),  # TODO: fix for k > 32, see issue #22925
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["SAMPLING_VALUES"] = tt_buffer

        # Sampling indices
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 32, 256)),  # TODO: fix for k > 32!
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["SAMPLING_INDICES"] = tt_buffer

        # Binary Mult + Silu
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 1, 32, 3584)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["BINARY_MUL"] = tt_buffer

        return persistent_buffers

    def get_persistent_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes.

        Creates double buffered persistent CCL buffers for each cluster axis.

        """

        persistent_buffers = [None, None]

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
        tt_buffer = ttnn.from_torch(
            torch.zeros((*cluster_shape, M, N_per_shard * num_cores)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=buffer_mem_cfg,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )
        persistent_buffers[cluster_axis] = tt_buffer

        # Create persistent buffers for cluster axis 1
        cluster_axis = 1
        num_input_cores_create_qkv = 10
        N_per_shard = 1280 // num_input_cores_create_qkv * cluster_shape[cluster_axis]  # QKV
        buffer_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sub_device_crs,
                [M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_buffer = ttnn.from_torch(
            torch.zeros((*cluster_shape, M, N_per_shard * num_cores)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=buffer_mem_cfg,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )
        persistent_buffers[cluster_axis] = tt_buffer

        # Create persistent buffer for lm_head
        num_cores_after_lm_head = 32  # Use 32 cores instead of 16 to reduce L1 memory usage per core
        N_per_shard = (16 * 1024) // num_cores_after_lm_head * cluster_shape[cluster_axis]  # LM Head
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
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )

        return persistent_buffers

    def get_decode_reduce_scatter_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes.

        Creates double buffered persistent CCL buffers for each cluster axis.

        """

        persistent_buffers = [[], []]

        cluster_shape = (8, 4)

        # Create persistent buffers for cluster axis 1
        cluster_axis = 1
        buffer_mem_cfg = self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"]
        for _ in range(self.num_cbs):
            tt_buffer = ttnn.from_torch(
                torch.zeros((*cluster_shape, 32, 512 * buffer_mem_cfg.shard_spec.num_cores())),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=buffer_mem_cfg,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
            persistent_buffers[cluster_axis].append(tt_buffer)

        return persistent_buffers

    def get_decode_rs_create_heads_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes.

        Creates double buffered persistent CCL buffers for each cluster axis.

        """

        persistent_buffers = [None, None]

        cluster_shape = (8, 4)
        num_pages_per_packet = 4
        shard_height = 32

        # Create persistent buffers for cluster axis 1
        cluster_axis = 1
        buffer_mem_cfg = self.model_config["RS_CREATE_HEADS_INTERIM_MEMCFG"]
        torch_buffer = torch.zeros(
            (*cluster_shape, shard_height, cluster_shape[cluster_axis] * num_pages_per_packet * 32 * 5)
        )
        persistent_buffers[cluster_axis] = ttnn.from_torch(
            torch_buffer,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=buffer_mem_cfg,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
        )

        return persistent_buffers

    def get_prefill_reduce_scatter_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes.

        Here are the current persistent buffers generated by this fuction:
        - QKV: (1, 1, 128, 1280)
        - FF1/FF3: (1, 1, 128, 3584)
        - FF2/WO: (1, 1, 128, 2048)

        """
        persistent_buffers_all = {}
        for seqlen in self.support_seqlens:
            persistent_buffers = {}

            if self.model_config is None:
                return persistent_buffers

            buffers_dict = {
                "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                "WO": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
                "FF1": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                "FF3": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                "FF2": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
            }
            for key, shape in buffers_dict.items():
                tt_buffers = []
                for i in range(1):
                    tt_buffer = ttnn.as_tensor(
                        torch.zeros(shape[1]),
                        device=self.mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat8_b,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        cache_file_name=self.weight_cache_path / (f"pb_rs_00_{key}_{i}_{seqlen}"),
                    )
                    tt_buffers.append(tt_buffer)
                for i in range(2):
                    tt_buffer = ttnn.as_tensor(
                        torch.zeros(shape[0]),
                        device=self.mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat8_b,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        cache_file_name=self.weight_cache_path / (f"pb_rs_01_{key}_{i}_{seqlen}"),
                    )
                    tt_buffers.append(tt_buffer)
                for i in range(2):
                    tt_buffer = ttnn.as_tensor(
                        torch.zeros(shape[1]),
                        device=self.mesh_device,
                        layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat8_b,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                        cache_file_name=self.weight_cache_path / (f"pb_rs_02_{key}_{i}_{seqlen}"),
                    )
                    tt_buffers.append(tt_buffer)
                persistent_buffers[key] = tt_buffers
            persistent_buffers_all[seqlen] = persistent_buffers
        return persistent_buffers_all

    def get_prefill_all_gather_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes.

        Creates double buffered persistent CCL buffers for each cluster axis.

        """
        ag_persistent_buffers_all = {}
        for seqlen in self.support_seqlens:
            ag_persistent_buffers = {}

            buffers_dict = {
                "QKV": [(1, 1, seqlen, 1280)],
                "WO": [(1, 1, seqlen, 2048)],
                "FF1": [(1, 1, seqlen, 3584)],
                "FF3": [(1, 1, seqlen, 3584)],
                "FF2": [(1, 1, seqlen, 2048)],
                "LAYERNORM": [(1, 1, seqlen, 128)],
                # "SAMPLING": [(1, 1, 32, 128 * 1024)]
            }
            for key, shape in buffers_dict.items():
                tt_buffer = ttnn.as_tensor(
                    torch.zeros(shape[0]),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16 if key == "LAYERNORM" else ttnn.bfloat8_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    cache_file_name=self.weight_cache_path / ("pb_ag_" + key + str(seqlen)),
                )
                ag_persistent_buffers[key] = tt_buffer
            ag_persistent_buffers_all[seqlen] = ag_persistent_buffers
        return ag_persistent_buffers_all

    def line_all_reduce(
        self, input_tensor_mesh, cluster_axis, num_links, memory_config, dtype=None, lm_head=False, buffer_key=None
    ):
        if self.mode == "decode":
            if lm_head:
                persistent_buffer = self.tt_lm_head_buffer_l1
            else:
                persistent_buffer = self.persistent_buffers[cluster_axis]

            output_tensor_mesh = ttnn.experimental.all_reduce_async(
                input_tensor_mesh,
                persistent_buffer,
                cluster_axis=cluster_axis,
                mesh_device=self.mesh_device,
                multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][
                    self.gather_idx[cluster_axis]
                ],
                num_links=num_links,
                memory_config=memory_config,
                dtype=dtype,
                topology=ttnn.Topology.Ring if is_RING_6U else ttnn.Topology.Linear,
                subdevice_id=self.worker_sub_device_id,
            )

            if lm_head:
                persistent_buffer.deallocate(True)
        else:
            if lm_head:
                ttnn_tensor_gathered = self.line_all_gather(
                    input_tensor_mesh,
                    dim=0,
                    num_links=num_links,
                    cluster_axis=cluster_axis,
                    memory_config=memory_config,
                    buffer_key=buffer_key,
                )
                ttnn_tensor_out = ttnn.experimental.fast_reduce_nc(
                    ttnn_tensor_gathered,
                    dims=[0],
                    output=None,
                    compute_kernel_config=None,
                    memory_config=memory_config,
                )
                return ttnn_tensor_out
            # ttnn.synchronize_device(self.mesh_device)
            output_tensor_scattered = self.line_reduce_scatter(
                input_tensor_mesh,
                memory_config,
                dim=3,
                cluster_axis=cluster_axis,
                num_links=num_links,
                math_op=ttnn.ReduceType.Sum,
                buffer_key=buffer_key,
            )
            # ttnn.synchronize_device(self.mesh_device)
            # Gather the scattered tensor
            output_tensor_mesh = self.line_all_gather(
                output_tensor_scattered,
                dim=3,
                cluster_axis=cluster_axis,
                memory_config=memory_config,
                num_links=num_links,
                buffer_key=buffer_key,
            )
            # ttnn.synchronize_device(self.mesh_device)

        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return output_tensor_mesh

    def line_all_reduce_create_heads(
        self,
        input_tensor_mesh,
        cluster_axis,
        num_links,
        num_heads,
        memory_config,
        num_kv_heads,
        qkv_memory_config,
        batch_offset,
        slice_size,
        dtype=None,
    ):
        (
            xqkv_reduced,
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.all_reduce_create_qkv_heads(
            input_tensor_mesh,
            self.persistent_buffers[cluster_axis],
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            num_heads=num_heads,
            memory_config=memory_config,
            topology=ttnn.Topology.Ring if is_RING_6U else ttnn.Topology.Linear,
            num_links=num_links,
            subdevice_id=self.worker_sub_device_id,
            num_kv_heads=num_kv_heads,
            final_memory_config=qkv_memory_config,
            batch_offset=batch_offset,
            slice_size=slice_size,
            dtype=dtype,
        )
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return xqkv_reduced, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD

    def matmul_line_reduce_scatter(
        self,
        # Matmul
        matmul_input,
        matmul_weight,
        # Reduce Scatter
        input_tensor_mesh,
        # Matmul
        compute_kernel_config=None,
        dtype=None,
        program_config=None,
        memory_config=None,
        global_cb=None,
        sub_device_id=None,
        # Reduce Scatter
        dim=3,
        num_links=1,
        math_op=ttnn.ReduceType.Sum,
        buffer_key=None,
        RS_memory_config=None,
        cluster_axis=1,
    ):
        persistent_interim_buffer = self.reduce_scatter_buffers[cluster_axis][
            self.reduce_scatter_buffer_idx[cluster_axis]
        ]
        w3_out, ttnn_tensor_out = ttnn.experimental.llama_rs_matmul(
            matmul_input,
            matmul_weight,
            input_tensor_mesh,
            persistent_interim_buffer,
            dim,
            self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            cluster_axis,
            self.mesh_device,
            num_links,
            self.worker_sub_device_id,
            memory_config_rs=RS_memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            program_config=program_config,
            memory_config_mm=memory_config,
            global_cb=global_cb,
            topology=ttnn.Topology.Ring if is_RING_6U else ttnn.Topology.Linear,
        )
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        self.reduce_scatter_buffer_idx[cluster_axis] = (self.reduce_scatter_buffer_idx[cluster_axis] + 1) % self.num_cbs
        # ttnn.synchronize_device(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])
        return ttnn_tensor_out, w3_out

    def llama_rs_create_heads(
        self,
        input_tensor_mesh,
        num_links,
        cluster_axis,
        dim,
        qkv_memory_config,
    ):
        persistent_interim_buffer = self.rs_create_heads_buffers[cluster_axis]
        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.llama_rs_create_heads(
            input_tensor=input_tensor_mesh,
            intermediate_packet_buffer=persistent_interim_buffer,
            dim=dim,
            cross_device_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            subdevice_id=self.worker_sub_device_id,
            cluster_axis=1,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Ring if is_RING_6U else ttnn.Topology.Linear,
            num_links=num_links,
            num_heads=8,
            num_kv_heads=1,
            memory_config=qkv_memory_config,
            qkv_memory_config=qkv_memory_config,
        )
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD

    def line_reduce_scatter(
        self,
        input_tensor_mesh,
        memory_config,
        cluster_axis,
        dim=3,
        num_links=1,
        math_op=ttnn.ReduceType.Sum,
        buffer_key=None,
    ):
        if self.mode == "prefill":
            # reshape input to [1, 1, S, x]
            B = input_tensor_mesh.shape[1]
            input_tensor_mesh = ttnn.reshape(
                input_tensor_mesh, (1, 1, B * input_tensor_mesh.shape[-2], input_tensor_mesh.shape[-1])
            )
            seqlen = input_tensor_mesh.shape[-2]
            persistent_buffers = (
                None
                if seqlen not in self.persistent_buffers.keys()
                else self.persistent_buffers[seqlen].get(buffer_key, None)
            )
            ttnn_tensor_out = ttnn.experimental.reduce_scatter_async(
                input_tensor_mesh,
                dim,
                cluster_axis=cluster_axis,
                mesh_device=self.mesh_device,
                from_remote_multi_device_global_semaphore=self.from_semaphore_handles[cluster_axis][
                    self.gather_idx[cluster_axis]
                ],
                to_remote_multi_device_global_semaphore=self.to_semaphore_handles[cluster_axis][
                    self.gather_idx[cluster_axis]
                ],
                math_op=math_op,
                memory_config=memory_config,
                topology=ttnn.Topology.Linear,
                num_links=num_links,
                subdevice_id=self.worker_sub_device_id,
                persistent_output_tensors=persistent_buffers,
            )
            # reshape input back
            ttnn_tensor_out = ttnn.reshape(ttnn_tensor_out, (1, B, seqlen // B, ttnn_tensor_out.shape[-1]))
            self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs

        else:
            persistent_interim_buffer = self.reduce_scatter_buffers[cluster_axis][
                self.reduce_scatter_buffer_idx[cluster_axis]
            ]

            ttnn_tensor_out = ttnn.experimental.llama_reduce_scatter(
                input_tensor_mesh,
                persistent_interim_buffer,
                dim,
                self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
                self.worker_sub_device_id,
                cluster_axis=1,
                mesh_device=self.mesh_device,
                num_links=num_links,
                memory_config=memory_config,
                topology=ttnn.Topology.Ring if is_RING_6U else ttnn.Topology.Linear,
            )
            self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
            self.reduce_scatter_buffer_idx[cluster_axis] = (
                self.reduce_scatter_buffer_idx[cluster_axis] + 1
            ) % self.num_cbs
        # ttnn.synchronize_device(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])
        return ttnn_tensor_out

    def line_all_gather(self, input_tensor_mesh, dim, cluster_axis, memory_config, num_links=1, buffer_key=None):
        topology = ttnn.Topology.Linear
        if self.mode == "prefill":
            if buffer_key is None:
                persistent_buffer = None
            else:
                # reshape input to [1, 1, S, x]
                B = input_tensor_mesh.shape[1]
                input_tensor_mesh = ttnn.reshape(
                    input_tensor_mesh, (1, 1, B * input_tensor_mesh.shape[-2], input_tensor_mesh.shape[-1])
                )
                seqlen = input_tensor_mesh.shape[-2]
                persistent_buffer = (
                    None
                    if seqlen not in self.all_gather_buffers.keys()
                    else self.all_gather_buffers[seqlen].get(buffer_key, None)
                )
        else:
            topology = ttnn.Topology.Ring if is_RING_6U else ttnn.Topology.Linear
            assert buffer_key is not None, "buffer_key is None"
            persistent_buffer = self.all_gather_buffers.get(buffer_key, None)
        # ttnn.synchronize_device(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])
        ttnn_tensor_out = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            topology=topology,
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            persistent_output_tensor=persistent_buffer,
            num_links=num_links,
            memory_config=memory_config,
            subdevice_id=self.worker_sub_device_id,
        )
        if self.mode == "prefill" and buffer_key is not None:
            # reshape input back
            ttnn_tensor_out = ttnn.reshape(ttnn_tensor_out, (1, B, seqlen // B, ttnn_tensor_out.shape[-1]))

        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        # ttnn.synchronize_device(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])
        return ttnn_tensor_out

    def all_gather_concat(self, input_tensor_mesh, dim, cluster_axis, memory_config, num_links=1, num_heads=8):
        ttnn_tensor_out = ttnn.experimental.all_gather_concat(
            input_tensor_mesh,
            self.all_gather_concat_inter_tensor[0],
            dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Ring if is_RING_6U else ttnn.Topology.Linear,
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            num_links=num_links,
            num_heads=num_heads,
            memory_config=memory_config,
            subdevice_id=self.worker_sub_device_id,
        )
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return ttnn_tensor_out

    def line_all_reduce_host(self, input_tensor_mesh, cluster_axis, num_links, memory_config):
        dim = 3

        ##### Host side implementation #####
        rs_output_tensor_mesh = self.line_reduce_scatter_host(
            input_tensor_mesh,
            memory_config,
            dim,
            cluster_axis,
            num_links=num_links,
            math_op=ttnn.ReduceType.Sum,
        )

        output_tensor_mesh = self.line_all_gather_host(
            rs_output_tensor_mesh,
            dim,
            cluster_axis,
            memory_config,
            num_links=num_links,
        )

        return output_tensor_mesh

    def line_reduce_scatter_host(
        self, input_tensor_mesh, memory_config, dim, cluster_axis, num_links=1, math_op=ttnn.ReduceType.Sum
    ):
        ##### Host side implementation #####
        dims = [0, 1]
        dtype = input_tensor_mesh.get_dtype()
        torch_tensor_mesh = ttnn.to_torch(
            input_tensor_mesh, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=dims, mesh_shape=(8, 4))
        )

        torch_tensor_mesh = torch.sum(torch_tensor_mesh, dim=cluster_axis, keepdim=True)

        dims[cluster_axis] = dim
        ttnn_tensor_out = ttnn.from_torch(
            torch_tensor_mesh,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=(8, 4)),
            dtype=dtype,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        return ttnn_tensor_out

    def line_all_gather_host(self, input_tensor_mesh, dim, cluster_axis, memory_config, num_links=1):
        ##### Host side implementation #####
        dims = [0, 0] if dim != 0 else [1, 1]
        dims[cluster_axis] = dim
        dtype = input_tensor_mesh.get_dtype()
        torch_tensor_mesh = ttnn.to_torch(
            input_tensor_mesh, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=dims, mesh_shape=(8, 4))
        )

        dims[cluster_axis] = None
        ttnn_tensor_out = ttnn.from_torch(
            torch_tensor_mesh,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=dims, mesh_shape=(8, 4)),
            dtype=dtype,
            memory_config=memory_config,
            layout=ttnn.TILE_LAYOUT,
        )

        return ttnn_tensor_out

    def close(self):
        self.mesh_device.reset_sub_device_stall_group()


def tt_distributed_rmsnorm(
    inp,
    epsilon,
    gamma,
    mesh_device,
    compute_kernel_config,
    tt_ccl=None,
):
    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16)
    padded_shape = (1, 1, inp.shape[-2], 32)

    tt_stats_gathered = tt_ccl.line_all_gather(
        tt_stats, dim=3, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, buffer_key="LAYERNORM"
    )

    tt_stats.deallocate(True)

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp, tt_stats_gathered, epsilon=epsilon, weight=gamma, compute_kernel_config=compute_kernel_config
    )

    # tt_stats_gathered.deallocate(True)
    # inp.deallocate(True)

    return tt_out, None


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
    output_mem_config=None,
):
    # inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    cluster_axis = 1
    semaphore = tt_ccl.gather_semaphore_handles[cluster_axis][tt_ccl.gather_idx[cluster_axis]]
    persistent_buffer = tt_ccl.all_gather_buffers.get("LAYERNORM", None)
    tt_out = ttnn.fused_rms_1_1_32_8192(
        inp,
        ln_sharded_progcfg,
        cluster_axis,
        tt_ccl.mesh_device,
        semaphore,
        topology=ttnn.Topology.Ring if is_RING_6U else ttnn.Topology.Linear,
        residual_input_tensor=res,
        num_links=1,
        epsilon=epsilon,
        weight=gamma,
        stats=persistent_buffer,
        memory_config=output_mem_config,
    )
    tt_ccl.gather_idx[cluster_axis] = (tt_ccl.gather_idx[cluster_axis] + 1) % tt_ccl.num_cbs
    return tt_out, res

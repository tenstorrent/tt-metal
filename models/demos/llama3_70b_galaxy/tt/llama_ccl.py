# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import os

# Only for prefill, check tt-metal/models/demos/llama3_70b_galaxy/README.md
LINE_RS = os.environ.get("LINE_RS", "0") == "1"
LINE_AG = os.environ.get("LINE_AG", "0") == "1"

# If set, use line for those AG ops in prefill, otherwise use ring if available
USE_LINE_AG = {
    # "QKV",
    # "WO",
    # "FF1",
    # "FF3",
    # "FF2",
    # "LAYERNORM",
}


class TT_CCL:
    def __init__(
        self,
        mesh_device,
        model_args,
        worker_sub_device_id,
        mode="decode",
        allocate_prefill_buffers=True,
        is_qwen=False,
        is_olmo=False,
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
        self.max_top_k = model_args.max_top_k
        self.max_batch_size = model_args.max_batch_size
        self.cluster_shape = model_args.cluster_shape
        self.is_qwen = is_qwen
        self.is_olmo = is_olmo
        self.mode = mode
        # Must be after is_olmo/max_batch_size are set (buffer size depends on them)
        # Only create for decode mode - prefill doesn't use all_gather_concat
        if mode == "decode":
            self.all_gather_concat_inter_tensor = self.get_all_gather_concat_inter_buffer()
        else:
            self.all_gather_concat_inter_tensor = None

        self.ring_topology = self.model_config["CCL_TOPOLOGY"] == ttnn.Topology.Ring
        self.use_ring_prefill = self.ring_topology and mode == "prefill"
        self.use_ring_ag_prefill = (self.ring_topology and not LINE_AG) and mode == "prefill"
        self.use_ring_rs_prefill = (self.ring_topology and not LINE_RS) and mode == "prefill"

        # Double buffered on each axis
        self.gather_semaphore_handles = [[], []]
        self.barrier_semaphore_handles = [[], []]
        self.reduce_semaphore_handles = [[], []]  # Now needed for decode too (OLMo uses reduce_scatter_minimal_async)
        if mode == "prefill":
            self.from_semaphore_handles = [[], []]
            self.to_semaphore_handles = [[], []]

        for i in range(2):
            for _ in range(self.num_cbs):
                self.barrier_semaphore_handles[i].append(
                    ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                )

                if self.use_ring_ag_prefill:
                    self.gather_semaphore_handles[i].append(
                        [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(2)]
                    )
                # line prefill and both decode
                else:
                    self.gather_semaphore_handles[i].append(
                        ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0)
                    )

                # reduce_scatter_minimal_async needs 3 semaphores per double-buffer
                self.reduce_semaphore_handles[i].append(
                    [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
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
        self.barrier_semaphore_idx = [0, 0]
        self.persistent_buffers = {}
        self.all_gather_buffers = {}
        if mode == "decode":
            self.persistent_buffers = self.get_persistent_buffers()
            self.all_gather_buffers = self.get_all_gather_buffers()
            self.reduce_scatter_buffers = self.get_decode_reduce_scatter_buffers()
            self.rs_create_heads_buffers = self.get_decode_rs_create_heads_buffers()
            if is_olmo and self.model_config.get("USE_AGMM_FF2", False):
                self.agmm_ff2_intermediate_buffers = self.get_agmm_ff2_intermediate_buffers()
                self.agmm_ff2_buffer_idx = 0
            else:
                self.agmm_ff2_intermediate_buffers = None
                self.agmm_ff2_buffer_idx = 0
        if mode == "prefill":
            # For some prefill seqlens we always allocate CCL buffers. Otherwise they will require barrier syncing
            # 256 and 512 added to support short prompts (128-256, 257-512 tokens) so they don't pad
            # all the way to 1024, which fills the KV cache with 847 EOS tokens and causes incoherence.
            # OLMo: 8K removed - async CCL deadlocks even with pre-allocated buffers.
            # 8K will run in eager mode with sync CCL like 16K+.
            self.support_seqlens = [4096, 2048, 1024, 512, 256, 128]
            if allocate_prefill_buffers:
                self.persistent_buffers = (
                    self.get_ring_prefill_reduce_scatter_buffers()
                    if self.use_ring_rs_prefill
                    else self.get_prefill_reduce_scatter_buffers()
                )
                self.all_gather_buffers = self.get_prefill_all_gather_buffers()
            else:
                for seqlen in self.support_seqlens:
                    self.persistent_buffers[seqlen] = {}
                    self.all_gather_buffers[seqlen] = {}

    def reset_gather_and_buffer_idx(self):
        self.gather_idx = [0, 0]
        self.reduce_scatter_buffer_idx = [0, 0]
        self.barrier_semaphore_idx = [0, 0]

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis):
        semaphore_index = cluster_axis
        current_idx = self.barrier_semaphore_idx[semaphore_index]
        self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % self.num_cbs
        return self.barrier_semaphore_handles[semaphore_index][current_idx]

    def get_all_gather_concat_inter_buffer(self):
        # Buffer dimensions depend on batch size:
        # - Llama: batch=128 -> per device [1, 32, 32, 128] = 32 cores
        # - OLMo: batch=32 -> per device [1, 8, 32, 128] = 8 cores
        batch_size = self.max_batch_size
        batch_per_device = batch_size // 4  # 4 columns in 8x4 mesh

        # Calculate cores needed: (batch_per_device * 32 * 128) / (32 * 128) = batch_per_device
        num_cores_needed = batch_per_device

        if self.is_olmo or batch_size <= 32:
            # OLMo: 8 cores for batch=32 (per device: 8 * 32 * 128 = 32768 elements)
            intermediate_core_range_set = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 3)),  # 2x4 = 8 cores
                ]
            )
        else:
            # Llama: 32 cores for batch=128 (per device: 32 * 32 * 128 = 131072 elements)
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
        # Minimum batch=4 required: ShardTensor2dMesh splits dim1 across 4 column devices
        buffer_batch_size = max(batch_size, 4)
        temp_shape = [8, buffer_batch_size, 32, 128]
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
        - LOGPROBS_MAX_REDUCTION: (1, 8, 32, 1)
        - LOGPROBS_SUM_EXP_REDUCTION: (1, 8, 32, 1)
        - LOGPROBS_LOGITS: (1, 8, 1, 32)
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
            torch.zeros((1, 1, self.max_batch_size, self.max_top_k * self.cluster_shape[0])),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            # dtype=ttnn.bfloat8_b,  # TODO: use bfp8_b when issue #23644 is fixed
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["SAMPLING_VALUES"] = tt_buffer

        # Sampling indices
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.max_batch_size, self.max_top_k * self.cluster_shape[0])),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["SAMPLING_INDICES"] = tt_buffer
        if self.is_olmo:
            # OLMo: padded_vocab_size=100352, per_device=12544, total after row all_gather = 12544*8=100352
            sampling_shape = (1, 1, 32, 12544 * 8)
        elif self.is_qwen:
            sampling_shape = (1, 1, 32, 155648)
        else:
            sampling_shape = (1, 1, 32, 128 * 1024)
        tt_buffer = ttnn.from_torch(
            torch.zeros(sampling_shape),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["SAMPLING"] = tt_buffer

        # LogProbs
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 8, 32, 1)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["LOGPROBS_MAX_REDUCTION"] = tt_buffer
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 8, 32, 1)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["LOGPROBS_SUM_EXP_REDUCTION"] = tt_buffer
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 8, 1, 32)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["LOGPROBS_LOGITS"] = tt_buffer

        # Binary Mult + Silu
        # OLMo: 3456 (intermediate_dim_per_tp, 27 cores × 128), Qwen: 3200, Llama: 3584
        if self.is_olmo:
            binary_mul_width = 3456  # intermediate_dim_per_tp (unpadded)
            binary_mul_mem_config = ttnn.DRAM_MEMORY_CONFIG
        elif self.is_qwen:
            binary_mul_width = 3200
            binary_mul_mem_config = self.model_config["FF2_IN_RING_MEMCFG"]
        else:
            binary_mul_width = 3584  # Llama default
            binary_mul_mem_config = self.model_config["FF2_IN_RING_MEMCFG"]
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.max_batch_size, binary_mul_width)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=binary_mul_mem_config,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["BINARY_MUL"] = tt_buffer

        # OLMo decode: bfloat16 version of BINARY_MUL — same shape/memcfg but bfloat16.
        # Used by line_all_gather for ff1ff3 (SwiGLU output) to avoid the bfloat8_b
        # quantisation error that compounds across 64 layers. The optimal-CCL path
        # (use_optimal_ccl_for_llama=True) has hardcoded bfloat8_b assumptions in the
        # C++ kernel, so we keep use_optimal_ccl_for_llama=False; providing a persistent
        # output tensor here makes the call trace-safe (no per-iteration allocation).
        if self.is_olmo:
            tt_buffer_bf16 = ttnn.from_torch(
                torch.zeros((1, 1, self.max_batch_size, binary_mul_width)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=binary_mul_mem_config,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            persistent_buffers["BINARY_MUL_BF16"] = tt_buffer_bf16

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
        # OLMo/Qwen have dim_per_tp=1280, Llama has 2048
        if self.is_olmo or self.is_qwen:
            N_per_shard = 1280 // 10 * cluster_shape[cluster_axis]  # 128 * 8 = 1024
        else:
            N_per_shard = 2048 // 16 * cluster_shape[cluster_axis]  # 128 * 8 = 1024  # FF2/DO
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
        N_per_shard = (
            (16 * 1024) // num_cores_after_lm_head * cluster_shape[cluster_axis]
            if not self.is_qwen
            else (155648 // 8) // num_cores_after_lm_head * cluster_shape[cluster_axis]
        )  # LM Head
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
            tt_buffer = (
                # 512 = 4 devices * 4 pages per packet * 32 tile_width
                ttnn.from_torch(
                    torch.zeros((*cluster_shape, 32, 512 * buffer_mem_cfg.shard_spec.num_cores())),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat8_b,
                    memory_config=buffer_mem_cfg,
                    mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
                )
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

    def get_agmm_ff2_intermediate_buffers(self):
        """Pre-allocate double-buffered intermediate tensors for FF2 AllGather+Matmul (OLMo decode).

        The intermediate holds the gathered in0 (ff1ff3 after all-gather) used by the fused
        llama_all_gather_matmul_async kernel.  Shape: [32, 3456] (intermediate_dim_per_tp)
        split across 4 L1 cores at (3,0)-(3,3), each holding [32, 864].
        """
        cluster_shape = (8, 4)
        intermediate_mem_config = self.model_config["FF2_AGMM_INTERMEDIATE_MEMCFG"]
        buffers = []
        # intermediate_dim_per_tp = 3456; 4 col TP-devices → 864 per device → 3456 total gathered
        total_width = 3456  # intermediate_dim_per_tp
        for _ in range(self.num_cbs):
            tt_buffer = ttnn.from_torch(
                torch.zeros((*cluster_shape, 32, total_width)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=intermediate_mem_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
            buffers.append(tt_buffer)
        return buffers

    def olmo_ff2_all_gather_matmul(self, ff1ff3, w2, compute_kernel_config, sub_device_id):
        """Fused AllGather+Matmul for OLMo FF2 decode using llama_all_gather_matmul_async.

        Replaces the separate line_all_gather + ttnn.linear steps.
        ff1ff3: [32, 864] DRAM (silu*gate output after reduce_scatter along cluster_axis=1)
        w2:     DRAM-interleaved [1,1,3456,1280]; moved to L1 here (freed after AGMM call)
        Returns: [32, 1280] L1 WIDTH_SHARDED on 10 cores (FF2_OUT_RING_MEMCFG_OLMO)
        """
        cluster_axis = 1
        intermediate = self.agmm_ff2_intermediate_buffers[self.agmm_ff2_buffer_idx]
        semaphore = self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]]

        # Move ff1ff3 to L1 sharded (required by llama_all_gather_matmul_async)
        ff1ff3_l1 = ttnn.to_memory_config(ff1ff3, self.model_config["FF2_AGMM_INPUT_MEMCFG"])

        # Move w2 from DRAM-interleaved to L1 WIDTH_SHARDED for AGMM kernel.
        # Allocated and freed within this call — avoids per-layer L1 pressure during model init.
        w2_l1 = ttnn.to_memory_config(w2, self.model_config["W2_AGMM_MEMCFG"])

        mm_out = ttnn.experimental.llama_all_gather_matmul_async(
            ff1ff3_l1,
            w2_l1,
            intermediate,
            dim=3,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=semaphore,
            ag_memory_config=self.model_config["FF2_AGMM_INTERMEDIATE_MEMCFG"],
            mm_memory_config=self.model_config["FF2_OUT_RING_MEMCFG_OLMO"],
            topology=ttnn.Topology.Linear,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            subdevice_id=sub_device_id,
            program_config=self.model_config["FF2_AGMM_PROGCFG"],
            compute_kernel_config=compute_kernel_config,
            dtype=ttnn.bfloat8_b,
        )

        ttnn.deallocate(ff1ff3_l1)
        ttnn.deallocate(w2_l1)
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        self.agmm_ff2_buffer_idx = (self.agmm_ff2_buffer_idx + 1) % self.num_cbs
        return mm_out  # [32, 1280] in FF2_OUT_RING_MEMCFG_OLMO (10 cores × [32, 128])

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

            if self.is_olmo:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 896), (1, 1, seqlen, 896 // 4)],
                    # "WO": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "FF1": [(1, 1, seqlen, 3456), (1, 1, seqlen, 3456 // 4)],
                    "FF3": [(1, 1, seqlen, 3456), (1, 1, seqlen, 3456 // 4)],
                    "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                }
            elif self.is_qwen:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    # "WO": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "FF1": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF3": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                }
            else:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    # "WO": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
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

    def get_ring_prefill_reduce_scatter_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes with hardcoded padding.
        """
        persistent_buffers_all = {}
        for seqlen in self.support_seqlens:
            persistent_buffers = {}

            if self.model_config is None:
                return persistent_buffers

            # Batched entries to be removed once https://github.com/tenstorrent/tt-metal/issues/35087 and
            # https://github.com/tenstorrent/tt-metal/issues/35319 gets resolved
            if self.is_olmo:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 896), (1, 1, seqlen, 896 // 4)],
                    # "WO": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "FF1": [(1, 1, seqlen, 3456), (1, 1, seqlen, 3456 // 4)],
                    "FF3": [(1, 1, seqlen, 3456), (1, 1, seqlen, 3456 // 4)],
                    "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "QKV_batched": [(1, 32, seqlen // 32, 896), (1, 32, seqlen // 32, 896 // 4)],
                    # "WO_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 8)],
                    "FF1_batched": [(1, 32, seqlen // 32, 3456), (1, 32, seqlen // 32, 3456 // 4)],
                    "FF3_batched": [(1, 32, seqlen // 32, 3456), (1, 32, seqlen // 32, 3456 // 4)],
                    "FF2_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 8)],
                }
            elif self.is_qwen:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    # "WO": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "FF1": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF3": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "QKV_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 4)],
                    # "WO_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 8)],
                    "FF1_batched": [(1, 32, seqlen // 32, 3200), (1, 32, seqlen // 32, 3200 // 4)],
                    "FF3_batched": [(1, 32, seqlen // 32, 3200), (1, 32, seqlen // 32, 3200 // 4)],
                    "FF2_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 8)],
                }
            else:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    # "WO": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
                    "FF1": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                    "FF3": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                    "FF2": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
                    "QKV_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 4)],
                    # "WO_batched": [(1, 32, seqlen // 32, 2048), (1, 32, seqlen // 32, 2048 // 8)],
                    "FF1_batched": [(1, 32, seqlen // 32, 3584), (1, 32, seqlen // 32, 3584 // 4)],
                    "FF3_batched": [(1, 32, seqlen // 32, 3584), (1, 32, seqlen // 32, 3584 // 4)],
                    "FF2_batched": [(1, 32, seqlen // 32, 2048), (1, 32, seqlen // 32, 2048 // 8)],
                }
            for key, shape in buffers_dict.items():
                tt_intermediate_buffer = ttnn.as_tensor(
                    torch.zeros(shape[0]),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    cache_file_name=self.weight_cache_path / (f"pb_rs_01_{key}_0_{seqlen}"),
                )
                # output buffer is reused from line imlementation
                tt_output_buffer = ttnn.as_tensor(
                    torch.zeros(shape[1]),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat8_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    cache_file_name=self.weight_cache_path / (f"pb_rs_00_{key}_0_{seqlen}"),
                )
                persistent_buffers[key] = {"intermediate": tt_intermediate_buffer, "output": tt_output_buffer}
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

            if self.is_olmo:
                # OLMo dimensions:
                # - qkv_size_per_device = 128 * (2*8 + 40) / 8 = 896, padded to 1536
                # - n_local_heads * head_dim = 5 * 128 = 640
                # - dim_per_tp = 5120 / 4 = 1280
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 896)],  # qkv_size_per_device (unpadded for all_gather)
                    # QKV_BF16: bfloat16 output buffer for the xqkv all_reduce in forward_prefill.
                    # OLMo uses bfloat16 xqkv to avoid bfloat8_b quantization error before SDPA.
                    "QKV_BF16": [(1, 1, seqlen, 896)],
                    "SDPA": [(1, 1, seqlen // 2, 640)],  # n_local_heads * head_dim = 5 * 128
                    "SDPA_REVERSE": [(1, 1, seqlen // 2, 640)],
                    "WO_AG": [(8, 1, seqlen, 1280)],  # dim_per_tp = 5120/4
                    # WO_AG_BF16: bfloat16 output buffer for the WO all_reduce in forward_prefill.
                    # WO projection outputs bfloat16 for OLMo to preserve attention output precision.
                    "WO_AG_BF16": [(8, 1, seqlen, 1280)],
                    "FF1": [(1, 1, seqlen, 3456)],  # intermediate/8
                    "FF3": [(1, 1, seqlen, 3456)],
                    # FF3_BF16: bfloat16 output buffer for the SwiGLU all_gather in forward_prefill.
                    # Using bfloat8_b (FF3) would re-quantize the bfloat16 ff1ff3 result and defeat
                    # the precision gain that keeps 64L prefill PCC near 0.99.
                    "FF3_BF16": [(1, 1, seqlen, 3456)],
                    "FF2": [(1, 1, seqlen, 1280)],  # dim_per_tp
                    "LAYERNORM": [(1, 1, seqlen, 128)],  # head_dim
                }
            elif self.is_qwen:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280)],
                    "SDPA": [(1, 1, seqlen // 2, 1024)],
                    "SDPA_REVERSE": [(1, 1, seqlen // 2, 1024)],
                    "WO_AG": [(8, 1, seqlen, 1280)],
                    "FF1": [(1, 1, seqlen, 3200)],
                    "FF3": [(1, 1, seqlen, 3200)],
                    "FF2": [(1, 1, seqlen, 1280)],
                    "LAYERNORM": [(1, 1, seqlen, 128)],
                }
            else:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280)],
                    "SDPA": [(1, 1, seqlen // 2, 1024)],
                    "SDPA_REVERSE": [(1, 1, seqlen // 2, 1024)],
                    "WO_AG": [(8, 1, seqlen, 2048)],
                    "FF1": [(1, 1, seqlen, 3584)],
                    "FF3": [(1, 1, seqlen, 3584)],
                    "FF2": [(1, 1, seqlen, 2048)],
                    "LAYERNORM": [(1, 1, seqlen, 128)],
                }
            for key, shape in buffers_dict.items():
                # LAYERNORM, FF3_BF16 (OLMo SwiGLU gather), WO_AG_BF16 (OLMo WO projection),
                # and QKV_BF16 (OLMo xqkv matmul output) need bfloat16 precision.
                use_bf16 = key in ("LAYERNORM", "FF3_BF16", "WO_AG_BF16", "QKV_BF16")
                tt_buffer = ttnn.as_tensor(
                    torch.zeros(shape[0]),
                    device=self.mesh_device,
                    layout=ttnn.TILE_LAYOUT,
                    dtype=ttnn.bfloat16 if use_bf16 else ttnn.bfloat8_b,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                    cache_file_name=self.weight_cache_path / ("pb_ag_" + key + str(seqlen)),
                )
                ag_persistent_buffers[key] = tt_buffer
            ag_persistent_buffers_all[seqlen] = ag_persistent_buffers

        # Additional buffers for fixed lengths (1 Tile = 32)
        # Buffer sizes depend on per-device vocab size after sharding across 8 row devices
        if self.is_olmo:
            # OLMo: padded_vocab_size=100288, per_device=12536, tile-aligned=12544
            buffers_fixed_length = {
                "LM_HEAD": [(4, 1, 32, 12544)],
                "SAMPLING": [(1, 1, 32, 12544 * 8)],
            }
        elif self.is_qwen:
            buffers_fixed_length = {
                "LM_HEAD": [(4, 1, 32, 19456)],
                "SAMPLING": [(1, 1, 32, 19456 * 8)],
            }
        else:
            buffers_fixed_length = {
                "LM_HEAD": [(4, 1, 32, 16384)],
                "SAMPLING": [(1, 1, 32, 128 * 1024)],
            }
        for key, shape in buffers_fixed_length.items():
            tt_buffer = ttnn.as_tensor(
                torch.zeros(shape[0]),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=self.weight_cache_path / ("pb_ag_" + key + "_32"),
            )
            ag_persistent_buffers[key] = tt_buffer

        ag_persistent_buffers_all[32] = ag_persistent_buffers
        return ag_persistent_buffers_all

    def line_all_reduce(
        self,
        input_tensor_mesh,
        cluster_axis,
        num_links,
        memory_config,
        dtype=None,
        lm_head=False,
        buffer_key=None,
        use_noc1_only=False,
        use_optimal_ccl_for_llama=False,
        batch_size=1,
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
                topology=self.model_config["CCL_TOPOLOGY"],
                subdevice_id=self.worker_sub_device_id,
                use_noc1_only=use_noc1_only,
                use_optimal_ccl_for_llama=use_optimal_ccl_for_llama,
            )
            if lm_head:
                persistent_buffer.deallocate(True)

        else:
            if buffer_key in ("WO_AG", "WO_AG_BF16") or lm_head:
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
                batch_size=batch_size,
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
            # Deallocate scattered tensor after gather (prevents memory leak in prefill)
            ttnn.deallocate(output_tensor_scattered)
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
        use_noc1_only=False,
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
            topology=self.model_config["CCL_TOPOLOGY"],
            num_links=num_links,
            subdevice_id=self.worker_sub_device_id,
            num_kv_heads=num_kv_heads,
            final_memory_config=qkv_memory_config,
            batch_offset=batch_offset,
            slice_size=slice_size,
            dtype=dtype,
            use_noc1_only=use_noc1_only,
        )
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return xqkv_reduced, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD

    def double_matmul_line_reduce_scatter(
        self,
        # Matmul
        matmul_input,
        matmul_weightA,
        matmul_weightB,
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
        use_noc1_only=False,
    ):
        persistent_interim_buffer = self.reduce_scatter_buffers[cluster_axis][
            self.reduce_scatter_buffer_idx[cluster_axis]
        ]
        w1_out, w3_out, ttnn_tensor_out = ttnn.experimental.llama_rs_matmul(
            matmul_input,
            matmul_weightA,
            persistent_interim_buffer,
            dim,
            self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            cluster_axis,
            self.mesh_device,
            num_links,
            self.worker_sub_device_id,
            second_weight_tensor=matmul_weightB,
            memory_config_rs=RS_memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            program_config=program_config,
            memory_config_mm=memory_config,
            global_cb=global_cb,
            topology=self.model_config["CCL_TOPOLOGY"],
            use_noc1_only=use_noc1_only,
        )
        w1_out.deallocate(True)
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        self.reduce_scatter_buffer_idx[cluster_axis] = (self.reduce_scatter_buffer_idx[cluster_axis] + 1) % self.num_cbs
        # ttnn.synchronize_device(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])
        return ttnn_tensor_out, w3_out

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
        use_noc1_only=False,
    ):
        persistent_interim_buffer = self.reduce_scatter_buffers[cluster_axis][
            self.reduce_scatter_buffer_idx[cluster_axis]
        ]
        w3_out, ttnn_tensor_out = ttnn.experimental.llama_rs_matmul(
            matmul_input,
            matmul_weight,
            persistent_interim_buffer,
            dim,
            self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            cluster_axis,
            self.mesh_device,
            num_links,
            self.worker_sub_device_id,
            rs_tensor=input_tensor_mesh,
            memory_config_rs=RS_memory_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            program_config=program_config,
            memory_config_mm=memory_config,
            global_cb=global_cb,
            second_weight_tensor=None,
            topology=self.model_config["CCL_TOPOLOGY"],
            use_noc1_only=use_noc1_only,
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
        use_noc1_only=False,
        use_optimal_ccl_for_llama=False,
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
            topology=self.model_config["CCL_TOPOLOGY"],
            num_links=num_links,
            num_heads=8,
            num_kv_heads=1,
            memory_config=qkv_memory_config,
            qkv_memory_config=qkv_memory_config,
            use_noc1_only=use_noc1_only,
            use_optimal_ccl_for_llama=use_optimal_ccl_for_llama,
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
        use_noc1_only=False,
        batch_size=1,
    ):
        if self.mode == "prefill":
            if self.use_ring_rs_prefill:
                return self.ring_reduce_scatter(
                    input_tensor_mesh,
                    memory_config,
                    cluster_axis,
                    dim=dim,
                    num_links=num_links,
                    buffer_key=buffer_key,
                    batch_size=batch_size,
                )
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
            ttnn_tensor_out = ttnn.reduce_scatter(
                input_tensor_mesh,
                dim,
                cluster_axis=cluster_axis,
                memory_config=memory_config,
                topology=ttnn.Topology.Linear,
                num_links=num_links,
                subdevice_id=self.worker_sub_device_id,
            )
            # reshape input back
            ttnn_tensor_out = ttnn.reshape(ttnn_tensor_out, (1, B, seqlen // B, ttnn_tensor_out.shape[-1]))
            self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs

        else:
            # OLMo decode: Use synchronous reduce_scatter to avoid Inf corruption from
            # reduce_scatter_minimal_async (which produces garbage with DRAM bfloat8_b input).
            if self.is_olmo:
                ttnn_tensor_out = ttnn.reduce_scatter(
                    input_tensor_mesh,
                    dim,
                    cluster_axis=cluster_axis,
                    memory_config=memory_config,
                    topology=ttnn.Topology.Linear,
                    num_links=num_links,
                    subdevice_id=self.worker_sub_device_id,
                )
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
                    topology=self.model_config["CCL_TOPOLOGY"],
                    use_noc1_only=use_noc1_only,
                )
                self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
                self.reduce_scatter_buffer_idx[cluster_axis] = (
                    self.reduce_scatter_buffer_idx[cluster_axis] + 1
                ) % self.num_cbs

        return ttnn_tensor_out

    def ring_reduce_scatter(
        self,
        input_tensor_mesh,
        memory_config,
        cluster_axis,
        dim=3,
        num_links=1,
        buffer_key=None,
        batch_size=1,
    ):
        # reshape input to [1, 1, S, x]
        B = input_tensor_mesh.shape[1]
        seqlen = input_tensor_mesh.shape[-2]
        persistent_buffers_list = None
        if batch_size > 1:
            # Temporary workaround to fix pcc issue with reduce scatter
            # To be removed once https://github.com/tenstorrent/tt-metal/issues/35087 and
            # https://github.com/tenstorrent/tt-metal/issues/35319 gets resolved
            input_tensor_mesh = ttnn.reshape(input_tensor_mesh, (1, 32, B * seqlen // 32, input_tensor_mesh.shape[-1]))
            buffer_key += "_batched"
        else:
            input_tensor_mesh = ttnn.reshape(input_tensor_mesh, (1, 1, B * seqlen, input_tensor_mesh.shape[-1]))

        persistent_buffers = (
            self.persistent_buffers[B * seqlen].get(buffer_key, None) if B * seqlen in self.persistent_buffers else None
        )
        # reduce_scatter_minimal_async expects only output buffers, not intermediate
        persistent_buffers_list = [persistent_buffers["output"]] if persistent_buffers else None
        num_links = 4
        # Seeing better performance for longer sequence lengths with num_workers_per_link = 4
        if seqlen > 128:
            num_workers_per_link = 4
        else:
            num_workers_per_link = 1
        # OLMo prefill: sync reduce_scatter (no subdevice). Avoids barrier_semaphore deadlocks
        # inside captured traces and avoids DRAM OOM from async intermediate buffers.
        # Non-OLMo (Llama/Qwen) and OLMo >8k: async with DRAM intermediate.
        if self.is_olmo and self.mode == "prefill":
            ttnn_tensor_out = ttnn.reduce_scatter(
                input_tensor_mesh,
                dim,
                cluster_axis=cluster_axis,
                memory_config=memory_config,
                topology=ttnn.Topology.Ring,
                num_links=1,  # Force num_links=1 for sync CCL (multi-link can deadlock)
                subdevice_id=None,
            )
        else:
            barrier_semaphore = None
            if persistent_buffers_list is None:
                barrier_semaphore = self.get_and_cycle_barrier_semaphore_handle(cluster_axis)
            ttnn_tensor_out = ttnn.experimental.reduce_scatter_minimal_async(
                input_tensor=input_tensor_mesh,
                persistent_output_buffers=persistent_buffers_list,
                dim=dim,
                multi_device_global_semaphore=self.reduce_semaphore_handles[cluster_axis][
                    self.gather_idx[cluster_axis]
                ],
                barrier_semaphore=barrier_semaphore,
                num_links=num_links,
                memory_config=memory_config,
                intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                topology=ttnn.Topology.Ring,
                subdevice_id=self.worker_sub_device_id,
                cluster_axis=cluster_axis,
                num_workers_per_link=num_workers_per_link,
            )

        # reshape input back
        ttnn_tensor_out = ttnn.reshape(ttnn_tensor_out, (1, B, seqlen, ttnn_tensor_out.shape[-1]))
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return ttnn_tensor_out

    def line_all_gather(
        self,
        input_tensor_mesh,
        dim,
        cluster_axis,
        memory_config,
        num_links=1,
        buffer_key=None,
        use_optimal_ccl_for_llama=False,
    ):
        topology = ttnn.Topology.Linear

        if self.mode == "prefill":
            persistent_buffer = None
            if self.use_ring_ag_prefill and buffer_key is not None:
                if buffer_key in USE_LINE_AG:
                    seqlen = input_tensor_mesh.shape[1] * input_tensor_mesh.shape[-2]
                    persistent_buffer = (
                        self.all_gather_buffers[seqlen][buffer_key] if seqlen in self.all_gather_buffers else None
                    )
                else:
                    return self.ring_all_gather(
                        input_tensor_mesh,
                        dim,
                        cluster_axis,
                        memory_config,
                        num_links=num_links,
                        buffer_key=buffer_key,
                    )

            if buffer_key is not None:
                # reshape input to [1, 1, S, x]
                B = input_tensor_mesh.shape[1]
                input_tensor_mesh = ttnn.reshape(
                    input_tensor_mesh, (1, 1, B * input_tensor_mesh.shape[-2], input_tensor_mesh.shape[-1])
                )
                seqlen = input_tensor_mesh.shape[-2]
                if persistent_buffer is None and seqlen in self.all_gather_buffers:
                    persistent_buffer = (
                        self.all_gather_buffers[seqlen].get(buffer_key, None)
                        if seqlen in self.all_gather_buffers
                        else None
                    )

        else:
            topology = self.model_config["CCL_TOPOLOGY"]
            persistent_buffer = self.all_gather_buffers.get(buffer_key, None) if buffer_key is not None else None
        # ttnn.synchronize_device(self.mesh_device, sub_device_ids=[self.worker_sub_device_id])
        barrier_semaphore = None
        if persistent_buffer is None:
            barrier_semaphore = self.get_and_cycle_barrier_semaphore_handle(cluster_axis)
        semaphores = (
            self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]][0]
            if self.use_ring_ag_prefill
            else [
                self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
                self.gather_semaphore_handles[cluster_axis][(self.gather_idx[cluster_axis] + 1) % self.num_cbs],
            ]
        )
        ttnn_tensor_out = ttnn.experimental.all_gather_async(
            input_tensor_mesh,
            dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            topology=topology,
            multi_device_global_semaphore=semaphores,
            persistent_output_tensor=persistent_buffer,
            barrier_semaphore=barrier_semaphore,
            num_links=num_links,
            memory_config=memory_config,
            subdevice_id=self.worker_sub_device_id,
            use_optimal_ccl_for_llama=use_optimal_ccl_for_llama,
        )
        if self.mode == "prefill" and buffer_key is not None:
            # reshape input back
            if buffer_key != "LM_HEAD":
                ttnn_tensor_out = ttnn.reshape(ttnn_tensor_out, (1, B, seqlen // B, ttnn_tensor_out.shape[-1]))

        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return ttnn_tensor_out

    def ring_all_gather(
        self, input_tensor_mesh, dim, cluster_axis, memory_config, num_links=1, buffer_key=None, reverse_order=False
    ):
        B = input_tensor_mesh.shape[1]
        input_tensor_mesh = ttnn.reshape(
            input_tensor_mesh, (1, 1, B * input_tensor_mesh.shape[-2], input_tensor_mesh.shape[-1])
        )
        seqlen = input_tensor_mesh.shape[-2]
        if "SDPA" in buffer_key:
            # SDPA input is 8x (4= ring_size (number of devices in ring), 2 = number of chunks per device) shorter than the sequence length
            seqlen = seqlen * 8
        persistent_buffers = (
            self.all_gather_buffers[seqlen].get(buffer_key, None) if seqlen in self.all_gather_buffers else None
        )
        # persistent_buffers = None

        num_links = 4
        # OLMo prefill: sync all_gather (no subdevice). Avoids barrier_semaphore deadlocks
        # inside captured traces. Non-OLMo (Llama/Qwen): async with persistent buffers.
        if self.is_olmo and self.mode == "prefill":
            ttnn_tensor_out = ttnn.all_gather(
                input_tensor_mesh,
                dim,
                cluster_axis=cluster_axis,
                topology=ttnn.Topology.Ring,
                num_links=1,  # Force num_links=1 for sync CCL (multi-link can deadlock)
                memory_config=memory_config,
            )
        else:
            barrier_semaphore = None
            if persistent_buffers is None:
                barrier_semaphore = self.get_and_cycle_barrier_semaphore_handle(cluster_axis)
            if reverse_order:
                all_gather_function = ttnn.experimental.all_gather_async_reversed
            else:
                all_gather_function = ttnn.experimental.all_gather_async
            ttnn_tensor_out = all_gather_function(
                input_tensor=input_tensor_mesh,
                persistent_output_buffer=persistent_buffers,
                dim=dim,
                multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][
                    self.gather_idx[cluster_axis]
                ],
                num_links=num_links,
                barrier_semaphore=barrier_semaphore,
                memory_config=memory_config,
                topology=ttnn.Topology.Ring,
                subdevice_id=self.worker_sub_device_id,
                cluster_axis=cluster_axis,
            )
        if self.mode == "prefill" and buffer_key is not None and dim != 2:
            # This condition excludes SDPA tensors (which use dim=2) from reshaping
            # All other tensors (QKV, WO, FF1, FF3, FF2, LAYERNORM) use dims 0, 1, or 3
            # reshape input back
            if buffer_key not in ["LM_HEAD", "WO_AG", "WO_AG_BF16"]:
                ttnn_tensor_out = ttnn.reshape(ttnn_tensor_out, (1, B, seqlen // B, ttnn_tensor_out.shape[-1]))
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return ttnn_tensor_out

    def all_gather_concat(
        self, input_tensor_mesh, dim, cluster_axis, memory_config, num_links=1, num_heads=8, use_noc1_only=False
    ):
        ttnn_tensor_out = ttnn.experimental.all_gather_concat(
            input_tensor_mesh,
            self.all_gather_concat_inter_tensor[0],
            dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            topology=self.model_config["CCL_TOPOLOGY"],
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            num_links=num_links,
            num_heads=num_heads,
            memory_config=memory_config,
            subdevice_id=self.worker_sub_device_id,
            use_noc1_only=use_noc1_only,
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
        """
        Host-side reduce-scatter implementation.
        1. Get tensors from all devices
        2. Sum partial results element-wise across cluster_axis (devices have same positions)
        3. Scatter the summed result along dim - each device gets 1/N of that dimension
        4. Pad output to tile-aligned size (30 cores * 32 = 960 for OLMo)
        """
        import torch.nn.functional as F

        dtype = input_tensor_mesh.get_dtype()

        # Number of devices in each axis
        num_rows, num_cols = 8, 4
        num_devices_in_axis = num_cols if cluster_axis == 1 else num_rows

        # Use ConcatMesh2dToTensor to concatenate along both dimensions
        # dims=(0, 1) gives shape [8, 4, H, W] - row devices concat on dim 0, col devices on dim 1
        torch_tensor_concat = ttnn.to_torch(
            input_tensor_mesh,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 1), mesh_shape=(num_rows, num_cols)),
        )

        # Sum across cluster_axis in the concatenated tensor
        summed = torch.sum(torch_tensor_concat, dim=cluster_axis, keepdim=True)

        # Scatter: split along 'dim' for each device in cluster_axis
        if cluster_axis == 1:
            # Shape is [8, 1, H, W], squeeze to [8, H, W]
            summed = summed.squeeze(1)
            scatter_dim = dim if dim <= cluster_axis else dim - 1
            chunks = torch.chunk(summed, num_cols, dim=scatter_dim)

            # Pad each chunk to tile-aligned size for OLMo
            # OLMo needs 960 per device (30 cores * 32) instead of 864 (3456/4)
            if self.is_olmo:
                padded_chunks = []
                for c in chunks:
                    chunk_width = c.shape[-1]
                    target_width = 960  # 30 cores * 32 tiles
                    if chunk_width < target_width:
                        pad_width = target_width - chunk_width
                        c = F.pad(c, (0, pad_width), mode="constant", value=0)
                    padded_chunks.append(c.unsqueeze(1))
                chunks = padded_chunks
            else:
                chunks = [c.unsqueeze(1) for c in chunks]

            output = torch.cat(chunks, dim=1)
        else:  # cluster_axis == 0
            summed = summed.squeeze(0)
            scatter_dim = dim if dim <= cluster_axis else dim - 1
            chunks = torch.chunk(summed, num_rows, dim=scatter_dim)
            chunks = [c.unsqueeze(0) for c in chunks]
            output = torch.cat(chunks, dim=0)

        # Convert back to ttnn tensor on mesh
        ttnn_tensor_out = ttnn.from_torch(
            output,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=(num_rows, num_cols)),
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
    use_2d_grid = False

    # Run distributed rmsnorm part 1
    tt_stats = ttnn.rms_norm_pre_all_gather(
        inp, compute_kernel_config=compute_kernel_config, dtype=ttnn.bfloat16, use_2d_core_grid=use_2d_grid
    )

    tt_stats_gathered = tt_ccl.line_all_gather(
        tt_stats, dim=3, cluster_axis=1, num_links=1, memory_config=ttnn.DRAM_MEMORY_CONFIG, buffer_key="LAYERNORM"
    )
    tt_stats.deallocate(True)

    # Run distributed rmsnorm part 2
    tt_out = ttnn.rms_norm_post_all_gather(
        inp,
        tt_stats_gathered,
        epsilon=epsilon,
        weight=gamma,
        compute_kernel_config=compute_kernel_config,
        use_2d_core_grid=use_2d_grid,
    )

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
    use_noc1_only=False,
    ccl_topology=None,
):
    # Ensure input is in the expected sharded memory config
    # This is needed for OLMo decode where input may be in DECODE_RESIDUAL_MEMCFG
    if inp.memory_config() != ln_sharded_input_memcfg:
        inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

    # Run distributed rmsnorm part 1
    cluster_axis = 1
    semaphore = tt_ccl.gather_semaphore_handles[cluster_axis][tt_ccl.gather_idx[cluster_axis]]
    persistent_buffer = tt_ccl.all_gather_buffers.get("LAYERNORM", None)
    tt_out = ttnn.fused_rms_minimal(
        inp,
        ln_sharded_progcfg,
        cluster_axis,
        tt_ccl.mesh_device,
        semaphore,
        topology=ccl_topology,
        residual_input_tensor=res,
        num_links=1,
        epsilon=epsilon,
        weight=gamma,
        stats=persistent_buffer,
        dtype=ttnn.bfloat16,
        memory_config=output_mem_config,
        use_noc1_only=use_noc1_only,
    )
    tt_ccl.gather_idx[cluster_axis] = (tt_ccl.gather_idx[cluster_axis] + 1) % tt_ccl.num_cbs
    return tt_out, res

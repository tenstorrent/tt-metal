# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import os
from loguru import logger

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
    @staticmethod
    def _ensure_min_interim_shard_width(memory_config, min_width):
        if memory_config is None or not memory_config.is_sharded() or memory_config.shard_spec is None:
            return memory_config
        shard_spec = memory_config.shard_spec
        if shard_spec.shape[1] >= min_width:
            return memory_config
        return ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                shard_spec.grid,
                [shard_spec.shape[0], min_width],
                shard_spec.orientation,
            ),
        )

    def __init__(
        self,
        mesh_device,
        model_args,
        worker_sub_device_id,
        mode="decode",
        allocate_prefill_buffers=True,
        is_qwen=False,
    ):
        self.mode = mode
        self.is_qwen = is_qwen
        # Wormhole / TG always run with the prefetcher; the Blackhole bring-up runs without it.
        # The no-prefetcher (Blackhole) buffer-sizing and stable-CCL fallbacks below are gated on
        # this so the prefetcher (Wormhole) path stays byte-for-byte identical to main.
        self.use_prefetcher = getattr(model_args, "use_prefetcher", True)
        all_crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 9))])

        self.mesh_device = mesh_device
        # If no worker subdevice is provided (e.g. prefetcher disabled), keep CCL on the full core set.
        self.sub_device_crs = (
            all_crs if mode == "prefill" or worker_sub_device_id is None else model_args.sub_core_grids
        )
        # Some experimental CCL kernels require a non-optional subdevice_id argument.
        # When prefetcher is disabled, use the default full-device subdevice id.
        self.worker_sub_device_id = worker_sub_device_id if worker_sub_device_id is not None else ttnn.SubDeviceId(0)
        self.model_config = model_args.model_config
        if mode == "decode" and is_qwen and not self.use_prefetcher:
            # Runtime guard: keep Qwen decode interim shards well above kernel minimum.
            self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"] = self._ensure_min_interim_shard_width(
                self.model_config.get("REDUCE_SCATTER_INTERIM_MEMCFG"), 1280
            )
            self.model_config["RS_CREATE_HEADS_INTERIM_MEMCFG"] = self._ensure_min_interim_shard_width(
                self.model_config.get("RS_CREATE_HEADS_INTERIM_MEMCFG"), 1280
            )
            logger.info(
                f"TT_CCL decode interim shard widths (post-guard): "
                f"RS_CREATE_HEADS={self.model_config['RS_CREATE_HEADS_INTERIM_MEMCFG'].shard_spec.shape[1]}, "
                f"REDUCE_SCATTER={self.model_config['REDUCE_SCATTER_INTERIM_MEMCFG'].shard_spec.shape[1]}"
            )
        self.weight_cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        self.num_cbs = 2
        self.from_remote_semaphore_handles = []
        self.to_remote_semaphore_handles = []
        self.cluster_shape = model_args.cluster_shape
        self.all_gather_concat_inter_tensor = self.get_all_gather_concat_inter_buffer()

        self.ring_topology = self.model_config["CCL_TOPOLOGY"] == ttnn.Topology.Ring
        self.use_ring_prefill = self.ring_topology and mode == "prefill"
        self.use_ring_ag_prefill = (self.ring_topology and not LINE_AG) and mode == "prefill"
        self.use_ring_rs_prefill = (self.ring_topology and not LINE_RS) and mode == "prefill"
        self.max_top_k = model_args.max_top_k
        self.max_batch_size = model_args.max_batch_size

        # Double buffered on each axis
        self.gather_semaphore_handles = [[], []]
        self.barrier_semaphore_handles = [[], []]
        if mode == "prefill":
            self.from_semaphore_handles = [[], []]
            self.to_semaphore_handles = [[], []]
            self.reduce_semaphore_handles = [[], []]

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

                if mode == "prefill":
                    if self.use_ring_rs_prefill:
                        # current ring implementation of reduce scatter expects 3 semaphores
                        self.reduce_semaphore_handles[i].append(
                            [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
                        )

                    else:
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
        # WAR-hazard semaphore for safe reuse of the persistent SAMPLING_VALUES/INDICES all-gather
        # buffers under decode trace. After its final read, the downstream sampling op increments this
        # (once per user core) on the SAMPLING_VALUES gather's drain core; that gather waits on it
        # before overwriting the buffer, closing the cross-sub-device Write-After-Read race that made
        # the first decode token non-deterministic under trace. Created on sub_device_crs (a superset
        # of the CCL worker cores, so the drain core's local copy is allocated).
        #
        # war_sem_drain_core is the gather's drain core = choose_worker_cores(worker_sub_device_id)[0]
        # = first row-wise core of worker_cores_range_set (model_config.get_core_ranges), which is
        # (1, 0). If that worker-core layout ever changes, update this coordinate to match.
        self.war_semaphore = None
        self.war_sem_drain_core = None
        self.war_sem_wait_value = self.max_batch_size
        if mode == "decode":
            self.war_sem_drain_core = ttnn.CoreCoord(1, 0)
            # Init value == wait value so the very first decode step's gather wait passes immediately
            # (no prior sampling has signalled yet); each step then rebalances (32 signals -> reset 0).
            self.war_semaphore = ttnn.create_global_semaphore(
                self.mesh_device, self.sub_device_crs, self.war_sem_wait_value
            )

        if mode == "decode":
            self.persistent_buffers = self.get_persistent_buffers()
            self.all_gather_buffers = self.get_all_gather_buffers()
            self.reduce_scatter_buffers = self.get_decode_reduce_scatter_buffers()
            self.rs_create_heads_buffers = self.get_decode_rs_create_heads_buffers()
        if mode == "prefill":
            # For some prefill seqlens we always allocate CCL buffers. Otherwise they will require barrier syncing
            self.support_seqlens = [4096, 2048, 1024, 128]
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
        if ttnn.get_arch_name().lower() == "blackhole":
            # BH Galaxy has a smaller compute grid than TG; avoid hardcoded y=9 coordinates.
            bh_intermediate_cores = min(32, self.sub_device_crs.num_cores())
            intermediate_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                ttnn.CoreCoord(1, 0), bh_intermediate_cores, self.sub_device_crs, row_wise=True
            )
        else:
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
        temp_shape = [self.cluster_shape[0], 32 * self.cluster_shape[1], 32, 128]
        intermediate_tensor = torch.zeros(temp_shape, dtype=torch.bfloat16)
        tt_intermediate_tensor = ttnn.from_torch(
            intermediate_tensor,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
            memory_config=intermediate_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=[0, 1], mesh_shape=self.cluster_shape),
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
        tt_buffer = (
            ttnn.from_torch(
                torch.zeros((1, 1, 32, 128 * 1024)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if not self.is_qwen
            else ttnn.from_torch(
                torch.zeros((1, 1, 32, 155648)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
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
        tt_buffer = (
            ttnn.from_torch(
                torch.zeros((1, 1, self.max_batch_size, 3584)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
            if not self.is_qwen
            else ttnn.from_torch(
                torch.zeros((1, 1, self.max_batch_size, 3200)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        )
        persistent_buffers["BINARY_MUL"] = tt_buffer

        return persistent_buffers

    def get_persistent_buffers(self):
        """
        Currently, this is hardcoded with llama specific shapes.

        Creates double buffered persistent CCL buffers for each cluster axis.

        """

        persistent_buffers = [None, None]

        cluster_shape = tuple(self.cluster_shape)
        default_M = 32
        num_cores = self.sub_device_crs.num_cores()

        # Create persistent buffers for cluster axis 0
        cluster_axis = 0
        N_per_shard = (
            2048 // 16 * cluster_shape[cluster_axis] if not self.is_qwen else 1280 // 10 * cluster_shape[cluster_axis]
        )  # FF2/DO
        M_axis0 = default_M
        if self.is_qwen:
            # all_reduce_async validates: buffer_shard_volume >= output_shard_volume * ring_size.
            # For decode residual all-reduce on axis 0 this requires width >= output_width * mesh_rows.
            decode_residual_memcfg = self.model_config.get("DECODE_RESIDUAL_MEMCFG", None)
            if (
                decode_residual_memcfg is not None
                and decode_residual_memcfg.is_sharded()
                and decode_residual_memcfg.shard_spec is not None
            ):
                required_width = decode_residual_memcfg.shard_spec.shape[1] * cluster_shape[cluster_axis]
                N_per_shard = max(N_per_shard, required_width)
                M_axis0 = max(M_axis0, decode_residual_memcfg.shard_spec.shape[0])
        buffer_mem_cfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(
                self.sub_device_crs,
                [M_axis0, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_buffer = ttnn.from_torch(
            torch.zeros((*cluster_shape, M_axis0, N_per_shard * num_cores)),
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
                [default_M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )
        tt_buffer = ttnn.from_torch(
            torch.zeros((*cluster_shape, default_M, N_per_shard * num_cores)),
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
                [default_M, N_per_shard],
                ttnn.ShardOrientation.ROW_MAJOR,
            ),
        )

        self.tt_lm_head_buffer = ttnn.from_torch(
            torch.zeros((*cluster_shape, default_M, N_per_shard * num_cores)),
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
        if self.use_prefetcher:
            # Wormhole / prefetcher path: identical to main.
            buffer_mem_cfg = self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"]
            shard_width = 512  # 512 = 4 devices * 4 pages per packet * 32 tile_width
        else:
            buffer_mem_cfg = self._ensure_min_interim_shard_width(
                self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"], 1280 if self.is_qwen else 512
            )
            shard_width = buffer_mem_cfg.shard_spec.shape[1]
        for _ in range(self.num_cbs):
            tt_buffer = (
                # Derive packet width from memcfg (BH needs 640 here).
                ttnn.from_torch(
                    torch.zeros((*cluster_shape, 32, shard_width * buffer_mem_cfg.shard_spec.num_cores())),
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

        cluster_shape = tuple(self.cluster_shape)

        # Create persistent buffers for cluster axis 1
        cluster_axis = 1
        if self.use_prefetcher:
            # Wormhole / prefetcher path: identical to main.
            num_pages_per_packet = 4
            shard_height = 32
            buffer_mem_cfg = self.model_config["RS_CREATE_HEADS_INTERIM_MEMCFG"]
            torch_buffer = torch.zeros(
                (*cluster_shape, shard_height, cluster_shape[cluster_axis] * num_pages_per_packet * 32 * 5)
            )
        else:
            rs_interim_memcfg = self.model_config.get(
                "REDUCE_SCATTER_INTERIM_MEMCFG", self.model_config["RS_CREATE_HEADS_INTERIM_MEMCFG"]
            )
            buffer_mem_cfg = self._ensure_min_interim_shard_width(rs_interim_memcfg, 1280 if self.is_qwen else 512)
            shard_height = buffer_mem_cfg.shard_spec.shape[0]
            shard_width = buffer_mem_cfg.shard_spec.shape[1]
            num_shard_cores = buffer_mem_cfg.shard_spec.num_cores()
            torch_buffer = torch.zeros((*cluster_shape, shard_height, shard_width * num_shard_cores))
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

            buffers_dict = (
                {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    # "WO": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
                    "FF1": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                    "FF3": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                    "FF2": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128), (1, 1, seqlen, 128 // 4)],
                }
                if not self.is_qwen
                else {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    # "WO": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "FF1": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF3": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128), (1, 1, seqlen, 128 // 4)],
                }
            )
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
            buffers_dict = (
                {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    # "WO": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
                    "FF1": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                    "FF3": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                    "FF2": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128), (1, 1, seqlen, 128 // 4)],
                    "QKV_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 4)],
                    # "WO_batched": [(1, 32, seqlen // 32, 2048), (1, 32, seqlen // 32, 2048 // 8)],
                    "FF1_batched": [(1, 32, seqlen // 32, 3584), (1, 32, seqlen // 32, 3584 // 4)],
                    "FF3_batched": [(1, 32, seqlen // 32, 3584), (1, 32, seqlen // 32, 3584 // 4)],
                    "FF2_batched": [(1, 32, seqlen // 32, 2048), (1, 32, seqlen // 32, 2048 // 8)],
                }
                if not self.is_qwen
                else {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    # "WO": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "FF1": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF3": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128), (1, 1, seqlen, 128 // 4)],
                    "QKV_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 4)],
                    # "WO_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 8)],
                    "FF1_batched": [(1, 32, seqlen // 32, 3200), (1, 32, seqlen // 32, 3200 // 4)],
                    "FF3_batched": [(1, 32, seqlen // 32, 3200), (1, 32, seqlen // 32, 3200 // 4)],
                    "FF2_batched": [(1, 32, seqlen // 32, 1280), (1, 32, seqlen // 32, 1280 // 8)],
                }
            )
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

            buffers_dict = (
                {
                    "QKV": [(1, 1, seqlen, 1280)],
                    "SDPA": [(1, 1, seqlen // 2, 1024)],
                    "SDPA_REVERSE": [(1, 1, seqlen // 2, 1024)],
                    "WO_AG": [(8, 1, seqlen, 2048)],
                    "FF1": [(1, 1, seqlen, 3584)],
                    "FF3": [(1, 1, seqlen, 3584)],
                    "FF2": [(1, 1, seqlen, 2048)],
                    "LAYERNORM": [(1, 1, seqlen, 128)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128)],  # For prefix caching column replication
                }
                if not self.is_qwen
                else {
                    "QKV": [(1, 1, seqlen, 1280)],
                    "SDPA": [(1, 1, seqlen // 2, 1024)],
                    "SDPA_REVERSE": [(1, 1, seqlen // 2, 1024)],
                    "WO_AG": [(8, 1, seqlen, 1280)],
                    "FF1": [(1, 1, seqlen, 3200)],
                    "FF3": [(1, 1, seqlen, 3200)],
                    "FF2": [(1, 1, seqlen, 1280)],
                    "LAYERNORM": [(1, 1, seqlen, 128)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128)],  # For prefix caching column replication
                }
            )
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

        # Additional buffers for fixed lengths (1 Tile = 32)
        buffers_fixed_length = (
            {
                "LM_HEAD": [(4, 1, 32, 16384)],
                "SAMPLING": [(1, 1, 32, 128 * 1024)],
            }
            if not self.is_qwen
            else {
                "LM_HEAD": [(4, 1, 32, 19456)],
                "SAMPLING": [(1, 1, 32, 19456 * 8)],
            }
        )
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
            if not self.use_prefetcher and input_tensor_mesh.memory_config().buffer_type == ttnn.BufferType.DRAM:
                # Blackhole all_reduce_async cannot read a DRAM input (the kernel has no accessor for
                # the fabric NoC address). Bring the input into an L1 layout first: prefer the
                # all-reduce output layout when it is L1, otherwise the persistent interim buffer's.
                target_mem_cfg = (
                    memory_config
                    if (memory_config is not None and memory_config.buffer_type != ttnn.BufferType.DRAM)
                    else persistent_buffer.memory_config()
                )
                input_tensor_mesh = ttnn.to_memory_config(input_tensor_mesh, target_mem_cfg)
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
                # fp32 dest accumulation for the LM-head all_reduce only: its bf16 cross-device sum was
                # order-dependent (ETH ring arrival order) -> per-row logit non-determinism -> greedy flips.
                fp32_dest_acc=lm_head,
            )
            if lm_head:
                persistent_buffer.deallocate(True)

        else:
            if buffer_key == "WO_AG" or lm_head:
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
            if not self.use_prefetcher:
                # BH no-prefetch (Qwen and Llama): reduce_scatter_minimal_async (the ring prefill
                # path) and the Linear-topology reduce_scatter deadlock on the 2D-torus fabric for
                # the column reduction. Use the stable ttnn.reduce_scatter with the configured Ring
                # topology, mirroring the working decode no-prefetch path.
                B = input_tensor_mesh.shape[1]
                input_tensor_mesh = ttnn.reshape(
                    input_tensor_mesh, (1, 1, B * input_tensor_mesh.shape[-2], input_tensor_mesh.shape[-1])
                )
                seqlen = input_tensor_mesh.shape[-2]
                ttnn_tensor_out = ttnn.reduce_scatter(
                    input_tensor_mesh,
                    dim,
                    cluster_axis=cluster_axis,
                    memory_config=memory_config,
                    topology=self.model_config["CCL_TOPOLOGY"],
                    num_links=num_links,
                    subdevice_id=self.worker_sub_device_id,
                )
                ttnn_tensor_out = ttnn.reshape(ttnn_tensor_out, (1, B, seqlen // B, ttnn_tensor_out.shape[-1]))
                self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
                return ttnn_tensor_out
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
            if not self.use_prefetcher:
                # BH no-prefetch (Qwen and Llama): use the stable ttnn.reduce_scatter with the
                # configured Ring topology instead of llama_reduce_scatter, which deadlocks on the
                # 2D-torus fabric column reduction.
                ttnn_tensor_out = ttnn.reduce_scatter(
                    input_tensor_mesh,
                    dim,
                    cluster_axis=cluster_axis,
                    memory_config=memory_config,
                    topology=self.model_config["CCL_TOPOLOGY"],
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
        persistent_buffers_list = list(persistent_buffers.values()) if persistent_buffers else None
        # Wormhole / prefetcher path keeps main's fixed link count (4); Blackhole caps to the links
        # physically available on the mesh (Blackhole Galaxy exposes 2, not 4).
        num_links = 4 if self.use_prefetcher else min(4, self.model_config["GALAXY_NUM_LINKS"])
        # Seeing better performance for longer sequence lengths with num_workers_per_link = 4
        if seqlen > 128:
            num_workers_per_link = 4
        else:
            num_workers_per_link = 1
        ttnn_tensor_out = ttnn.experimental.reduce_scatter_minimal_async(
            input_tensor=input_tensor_mesh,
            persistent_output_buffers=persistent_buffers_list,
            dim=dim,
            multi_device_global_semaphore=self.reduce_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            barrier_semaphore=self.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            num_links=num_links,
            memory_config=memory_config,
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
        use_subdevice=True,
        use_experimental_all_gather=False,
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
            assert buffer_key is not None, "buffer_key is None"
            persistent_buffer = self.all_gather_buffers.get(buffer_key, None)
        if use_experimental_all_gather or self.use_prefetcher:
            # Wormhole / prefetcher path uses the experimental async all-gather (identical to main).
            # The Blackhole no-prefetcher bring-up uses the stable public all_gather below.
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
            # Only the SAMPLING_VALUES gather (first sampling gather of the decode step) waits on the WAR
            # semaphore; being first on the worker sub-device it program-orders the whole sampling section
            # after the previous step's ttnn.sampling, so the later SAMPLING_INDICES gather is safe too.
            war_semaphore = self.war_semaphore if buffer_key == "SAMPLING_VALUES" else None
            war_wait_value = self.war_sem_wait_value if buffer_key == "SAMPLING_VALUES" else None
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
                subdevice_id=self.worker_sub_device_id if use_subdevice else None,
                use_optimal_ccl_for_llama=use_optimal_ccl_for_llama,
                war_semaphore=war_semaphore,
                war_wait_value=war_wait_value,
            )
        else:
            # Default to the stable/public all_gather API.
            # Keep stable path simple: avoid preallocated output tensor because
            # its program cache/sub-device interaction can mismatch core groups.
            # Also prefer TILE input to avoid composite row-major all_gather (concat path),
            # which is where sub-device core-group mismatches are observed.
            all_gather_input = input_tensor_mesh
            if all_gather_input.get_layout() == ttnn.ROW_MAJOR_LAYOUT:
                all_gather_input = ttnn.to_layout(all_gather_input, ttnn.TILE_LAYOUT)

            stable_memory_config = memory_config
            if (
                stable_memory_config is not None
                and stable_memory_config.is_sharded()
                and stable_memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
                and stable_memory_config.shard_spec is not None
            ):
                # Stable device all_gather validates width-sharded output specs strictly:
                # shard height must equal full physical height (all dims except width).
                output_shape = list(all_gather_input.shape)
                output_shape[dim] *= self.cluster_shape[cluster_axis]
                physical_height = 1
                for shape_dim in output_shape[:-1]:
                    physical_height *= shape_dim
                shard_spec = stable_memory_config.shard_spec
                if shard_spec.shape[0] != physical_height:
                    stable_memory_config = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        stable_memory_config.buffer_type,
                        ttnn.ShardSpec(
                            shard_spec.grid,
                            [physical_height, shard_spec.shape[1]],
                            shard_spec.orientation,
                        ),
                    )

            ttnn_tensor_out = ttnn.all_gather(
                all_gather_input,
                dim,
                cluster_axis=cluster_axis,
                topology=topology,
                num_links=num_links,
                memory_config=stable_memory_config,
                subdevice_id=self.worker_sub_device_id if use_subdevice else None,
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

        # Wormhole / prefetcher path keeps main's fixed link count (4); Blackhole caps to the links
        # physically available on the mesh (Blackhole Galaxy exposes 2, not 4).
        num_links = 4 if self.use_prefetcher else min(4, self.model_config["GALAXY_NUM_LINKS"])
        if reverse_order:
            all_gather_function = ttnn.experimental.all_gather_async_reversed
        else:
            all_gather_function = ttnn.experimental.all_gather_async
        ttnn_tensor_out = all_gather_function(
            input_tensor=input_tensor_mesh,
            persistent_output_buffer=persistent_buffers,
            dim=dim,
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            num_links=num_links,
            barrier_semaphore=self.get_and_cycle_barrier_semaphore_handle(cluster_axis),
            memory_config=memory_config,
            topology=ttnn.Topology.Ring,
            subdevice_id=self.worker_sub_device_id,
            cluster_axis=cluster_axis,
        )

        if self.mode == "prefill" and buffer_key is not None and dim != 2:
            # This condition excludes SDPA tensors (which use dim=2) from reshaping
            # All other tensors (QKV, WO, FF1, FF3, FF2, LAYERNORM) use dims 0, 1, or 3
            # reshape input back
            if buffer_key not in ["LM_HEAD", "WO_AG"]:
                ttnn_tensor_out = ttnn.reshape(ttnn_tensor_out, (1, B, seqlen // B, ttnn_tensor_out.shape[-1]))
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return ttnn_tensor_out

    def line_all_gather_matmul(
        self,
        input_tensor_mesh,
        weight_tensor,
        dim,
        cluster_axis,
        memory_config,
        matmul_config,
        compute_kernel_config,
        dtype=ttnn.bfloat8_b,
    ):
        """
        Fused AllGather + MatMul for prefill using all_gather_minimal_matmul_async.

        """
        topology = self.model_config.get("CCL_TOPOLOGY", ttnn.Topology.Ring)
        force_transpose = True

        B = input_tensor_mesh.shape[1]
        input_tensor_mesh = ttnn.reshape(
            input_tensor_mesh, (1, 1, B * input_tensor_mesh.shape[-2], input_tensor_mesh.shape[-1])
        )

        grid_size = matmul_config.compute_with_storage_grid_size
        if hasattr(grid_size, "x"):
            core_grid = ttnn.CoreCoord(grid_size.x, grid_size.y)
        else:
            core_grid = ttnn.CoreCoord(grid_size[0], grid_size[1])

        div_axis = core_grid.x if force_transpose else core_grid.y
        # Wormhole / prefetcher path is unconstrained (matches main); Blackhole caps to GALAXY_NUM_LINKS.
        max_links = 4 if self.use_prefetcher else self.model_config["GALAXY_NUM_LINKS"]
        num_links = 1
        for nl in [4, 3, 2, 1]:
            if nl <= max_links and div_axis % nl == 0:
                num_links = nl
                break

        max_workers_total = core_grid.x if force_transpose else core_grid.y
        num_workers_per_link = max(1, min(8 // num_links, max_workers_total // num_links))

        sem_current = self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]]
        sem_next = self.gather_semaphore_handles[cluster_axis][(self.gather_idx[cluster_axis] + 1) % self.num_cbs]
        if isinstance(sem_current, list):
            semaphores = sem_current + sem_next
        else:
            semaphores = [sem_current, sem_next]

        output = ttnn.experimental.all_gather_minimal_matmul_async(
            input_tensor=input_tensor_mesh,
            weight_tensor=weight_tensor,
            config=matmul_config,
            compute_kernel_config=compute_kernel_config,
            multi_device_global_semaphore=semaphores,
            num_links=num_links,
            topology=topology,
            cluster_axis=cluster_axis,
            memory_config=memory_config,
            dtype=dtype,
            force_transpose=force_transpose,
            num_workers_per_link=num_workers_per_link,
            num_buffers_per_channel=8,
        )

        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return output[0]

    def all_gather_concat(
        self,
        input_tensor_mesh,
        dim,
        cluster_axis,
        memory_config,
        num_links=1,
        num_heads=8,
        use_noc1_only=False,
        batch_first=False,
    ):
        if self.use_prefetcher:
            # Wormhole / prefetcher path: identical to main (fused experimental all_gather_concat).
            ttnn_tensor_out = ttnn.experimental.all_gather_concat(
                input_tensor_mesh,
                self.all_gather_concat_inter_tensor[0],
                dim,
                cluster_axis=cluster_axis,
                mesh_device=self.mesh_device,
                topology=self.model_config["CCL_TOPOLOGY"],
                multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][
                    self.gather_idx[cluster_axis]
                ],
                num_links=num_links,
                num_heads=num_heads,
                memory_config=memory_config,
                subdevice_id=self.worker_sub_device_id,
                use_noc1_only=use_noc1_only,
            )
            self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
            return ttnn_tensor_out

        # Non-experimental fallback path: all-gather heads, then concatenate heads.
        # num_links is forced to 1 for decode stability.
        effective_num_links = 1
        gather_dim = dim
        gather_memory_config = memory_config
        if self.mode == "decode" and dim == 1:
            # Keep the existing gather axis: it produces the 32x32 tile shape
            # required by nlp_concat_heads_decode. We only swap the logical
            # users/head axes below before invoking the concat op.
            gather_memory_config = self.model_config["GATHER_USERS_MEMCFG"](self.cluster_shape[1])
        buffer_key = "SDPA" if self.mode == "decode" else None
        gathered = self.line_all_gather(
            input_tensor_mesh,
            dim=gather_dim,
            cluster_axis=cluster_axis,
            memory_config=gather_memory_config,
            num_links=effective_num_links,
            buffer_key=buffer_key,
            use_subdevice=True,
        )

        if self.mode == "decode":
            concat_heads_op = ttnn.experimental.nlp_concat_heads_decode
        else:
            concat_heads_op = getattr(ttnn, "nlp_concat_heads", None)
            if concat_heads_op is None:
                concat_heads_op = ttnn.experimental.nlp_concat_heads

        concat_input = gathered
        if self.mode == "decode" and dim == 1:
            concat_input = ttnn.transpose(gathered, 1, 2)
            concat_input = ttnn.to_layout(concat_input, ttnn.TILE_LAYOUT)
            concat_input = ttnn.to_memory_config(concat_input, gather_memory_config)
        else:
            try:
                gathered_memcfg = gathered.memory_config()
                if self.mode != "decode" and gathered_memcfg.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
                    # nlp_concat_heads does not accept WIDTH_SHARDED input.
                    concat_input = ttnn.to_memory_config(gathered, ttnn.DRAM_MEMORY_CONFIG)
            except Exception as e:
                # nlp_concat_heads does not accept WIDTH_SHARDED input.
                logger.warning(f"all_gather_concat: memory_config introspection failed, using gathered as-is: {e}")
                concat_input = gathered

        decode_concat_sub_core_grids = None
        if self.mode == "decode":
            try:
                decode_concat_sub_core_grids = concat_input.memory_config().shard_spec.grid
            except Exception as e:
                logger.warning(f"all_gather_concat: could not read concat_input shard grid, using None: {e}")
                decode_concat_sub_core_grids = None
        use_two_step_concat = False
        try:
            use_two_step_concat = (
                memory_config is not None
                and memory_config.is_sharded()
                and memory_config.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
            )
        except Exception as e:
            logger.warning(f"all_gather_concat: could not inspect output memory_config, disabling two-step concat: {e}")
            use_two_step_concat = False

        if use_two_step_concat:
            # Avoid passing WIDTH_SHARDED output memcfg directly into concat_heads,
            # then explicitly reshard output in a second step.
            if self.mode == "decode":
                concat_tmp = concat_heads_op(
                    concat_input,
                    num_heads=num_heads,
                    sub_core_grids=decode_concat_sub_core_grids,
                )
            else:
                concat_tmp = concat_heads_op(concat_input)
            target_memcfg = memory_config
            try:
                shard_spec = memory_config.shard_spec
                concat_shape = tuple(concat_tmp.shape)
                physical_height = 1
                for shape_dim in concat_shape[:-1]:
                    physical_height *= shape_dim
                output_width = concat_shape[-1]
                # Build a safe width-sharded spec directly from concat output shape:
                # - shard height must match physical height for WIDTH_SHARDED
                # - use full output width per shard to avoid over-sharding on width
                adjusted_shard_width = ((output_width + 31) // 32) * 32
                target_memcfg = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    memory_config.buffer_type,
                    ttnn.ShardSpec(
                        shard_spec.grid,
                        [physical_height, adjusted_shard_width],
                        shard_spec.orientation,
                    ),
                )
            except Exception as e:
                logger.warning(
                    f"all_gather_concat: could not build width-sharded target memcfg, "
                    f"falling back to requested memory_config: {e}"
                )
                target_memcfg = memory_config

            ttnn_tensor_out = ttnn.to_memory_config(concat_tmp, target_memcfg)
            concat_tmp.deallocate(True)
        else:
            if self.mode == "decode":
                ttnn_tensor_out = concat_heads_op(
                    concat_input,
                    num_heads=num_heads,
                    memory_config=memory_config,
                    sub_core_grids=decode_concat_sub_core_grids,
                )
            else:
                ttnn_tensor_out = concat_heads_op(concat_input, memory_config=memory_config)
        if concat_input is not gathered:
            concat_input.deallocate(True)
        gathered.deallocate(True)
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
    use_noc1_only=False,
    ccl_topology=None,
):
    # inp = ttnn.to_memory_config(inp, memory_config=ln_sharded_input_memcfg)

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
        memory_config=output_mem_config,
        use_noc1_only=use_noc1_only,
    )
    tt_ccl.gather_idx[cluster_axis] = (tt_ccl.gather_idx[cluster_axis] + 1) % tt_ccl.num_cbs
    return tt_out, res

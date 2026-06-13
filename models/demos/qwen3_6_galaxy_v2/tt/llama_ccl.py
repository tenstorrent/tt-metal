# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import os

import torch

import ttnn

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
        is_qwen36=None,
    ):
        self.mode = mode
        self.allocate_prefill_buffers = allocate_prefill_buffers
        grid_size = mesh_device.compute_with_storage_grid_size()
        all_crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_size.x - 1, grid_size.y - 1))]
        )

        self.mesh_device = mesh_device
        self.sub_device_crs = all_crs if mode == "prefill" else model_args.sub_core_grids
        self.worker_sub_device_id = worker_sub_device_id
        self.model_config = model_args.model_config
        self.weight_cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
        self.num_cbs = 2
        self.from_remote_semaphore_handles = []
        self.to_remote_semaphore_handles = []
        self.all_gather_concat_inter_tensor = self.get_all_gather_concat_inter_buffer()

        self.ring_topology = self.model_config["CCL_TOPOLOGY"] == ttnn.Topology.Ring
        self.use_ring_prefill = self.ring_topology and mode == "prefill"
        self.use_ring_ag_prefill = (self.ring_topology and not LINE_AG) and mode == "prefill"
        self.use_ring_rs_prefill = (self.ring_topology and not LINE_RS) and mode == "prefill"
        self.max_top_k = model_args.max_top_k
        self.max_batch_size = model_args.max_batch_size
        self.cluster_shape = model_args.cluster_shape
        self.is_qwen = is_qwen
        # qwen3.6 marker — read by olmo-style dual-dtype persistent-buffer branches
        # below.  Falls back to model_args.is_qwen36 when caller did not pass it,
        # so existing test call-sites that only pass is_qwen=True keep working.
        if is_qwen36 is None:
            self.is_qwen36 = getattr(model_args, "is_qwen36", False)
        else:
            self.is_qwen36 = is_qwen36

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
        # V2-14: qwen3.6 residual-stream persistent buffers for the 3rd-overload
        # ``ttnn.experimental.all_reduce_async`` (width-sharded full-dim=5120
        # output). Sized for ring=8 on axis=0 and ring=4 on axis=1.
        # Populated only when ``is_qwen36`` and mode == "decode".
        self.qwen36_residual_buffers = [None, None]
        self.qwen36_residual_input_memcfgs = [None, None]
        self.qwen36_residual_output_memcfgs = [None, None]
        # QWEN36_DECODE_RS_ASYNC (DEFAULT ON): route the decode-MLP interleaved-DRAM reduce_scatter
        # through ttnn.experimental.reduce_scatter_minimal_async (ring) instead of sync ttnn.reduce_scatter.
        # Unit-validated (test_mlp_rs_async_kernel): PCC 0.99999 bit-identical, 1.26x faster. 64L A/B
        # +1.9% coherent; combined-wins 64L +3.8% coherent. Set =0 to revert to sync. Needs 3 ring sems/cb.
        self._decode_rs_async = os.environ.get("QWEN36_DECODE_RS_ASYNC", "1") == "1"
        self.decode_rs_semaphores = [[], []]
        self.decode_rs_idx = [0, 0]
        if self._decode_rs_async and mode == "decode":
            for i in range(2):
                for _ in range(self.num_cbs):
                    self.decode_rs_semaphores[i].append(
                        [ttnn.create_global_semaphore(self.mesh_device, self.sub_device_crs, 0) for _ in range(3)]
                    )
        if mode == "decode":
            # V2-14: build qwen36 residual buffers FIRST so they get placed
            # near the top of the L1 stack.  Allocating them after KV cache /
            # other persistent buffers ran into a static-CB clash on core (0,0).
            if self.is_qwen36:
                self._build_qwen36_residual_buffers()
            self.persistent_buffers = self.get_persistent_buffers()
            self.all_gather_buffers = self.get_all_gather_buffers()
            self.reduce_scatter_buffers = self.get_decode_reduce_scatter_buffers()
            self.rs_create_heads_buffers = self.get_decode_rs_create_heads_buffers()
        if mode == "prefill":
            # For some prefill seqlens we always allocate CCL buffers. Otherwise they will require barrier syncing
            # qwen3.6: 1024 and 2048 are NOT valid prefill buckets (their matmul progcfg is broken:
            # num_blocks_x=9 > num_cores_x=5). get_padded_prefill_len pads >128 -> 4096, so only the
            # 128 and >=4096 buckets are ever used. Allocating buffers for the invalid 1024/2048 buckets
            # is wasteful and shifts the persistent-buffer DRAM layout — drop them.
            self.support_seqlens = [4096, 128] if self.is_qwen36 else [4096, 2048, 1024, 128]
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

    # ------------------------------------------------------------------
    # qwen3.6 dual-dtype persistent-buffer selectors
    # ------------------------------------------------------------------
    # Olmo galaxy hit a class of bugs at long ISL where persistent CCL
    # buffers had hardcoded dtypes but the producing matmul changed dtype
    # depending on ISL: ring SDPA forces bfloat8_b at ISL ≥ 4096 while
    # shorter ISL paths produce bfloat16. Mismatched dtype between a
    # matmul's output and the persistent buffer it writes into causes
    # silent garbage output (no crash, just NaN/Inf).
    #
    # The fix replicated here for qwen3.6: pre-allocate BOTH a bfloat8_b
    # and a bfloat16 variant of each affected persistent buffer (QKV,
    # WO_AG, FF1, FF3, plus the decode BINARY_MUL residual buffer). The
    # call-sites in `llama_attention.py` / `llama_mlp.py` (V2-4 / V2-5)
    # select the variant whose dtype matches the producing matmul's
    # output via `get_qkv_buffer_key()` / `get_wo_ag_buffer_key()` /
    # `get_ff_buffer_key()` below.
    #
    # ISL selection rule (olmo session-12 precedent, BRINGUP_LOG.md):
    #   ISL ≤ 2048           → bfloat16 ("..._BF16")  — bfloat16 xqkv path
    #   ISL ≥ 4096           → bfloat8_b (no suffix)  — ring SDPA bf8b path
    QWEN36_BF16_ISL_THRESHOLD = 4096  # >= → bfloat8_b key; < → bfloat16 key

    @staticmethod
    def _qwen36_pick_bf16(seq_len, bf16_key, bf8_key):
        return bf8_key if seq_len >= TT_CCL.QWEN36_BF16_ISL_THRESHOLD else bf16_key

    def get_qkv_buffer_key(self, seq_len):
        """Return the persistent QKV all-gather buffer key for qwen3.6 prefill.

        ISL < 4096 → bfloat16 buffer ("QKV_BF16"); ISL ≥ 4096 → bfloat8_b ("QKV").
        Non-qwen36 paths keep the canonical key for backwards compat.
        """
        if not self.is_qwen36:
            return "QKV"
        return self._qwen36_pick_bf16(seq_len, "QKV_BF16", "QKV")

    def get_wo_ag_buffer_key(self, seq_len):
        """Return the persistent WO_AG all-gather buffer key for qwen3.6 prefill."""
        if not self.is_qwen36:
            return "WO_AG"
        return self._qwen36_pick_bf16(seq_len, "WO_AG_BF16", "WO_AG")

    def get_ff_buffer_key(self, name, seq_len):
        """Return the persistent FF1/FF3 all-gather buffer key for qwen3.6 prefill.

        `name` is "FF1" or "FF3". FF2 has a single dtype (bfloat16 reduction
        output) so the caller does not go through this helper.
        """
        if not self.is_qwen36:
            return name
        return self._qwen36_pick_bf16(seq_len, f"{name}_BF16", name)

    def get_and_cycle_barrier_semaphore_handle(self, cluster_axis):
        semaphore_index = cluster_axis
        current_idx = self.barrier_semaphore_idx[semaphore_index]
        self.barrier_semaphore_idx[semaphore_index] = (current_idx + 1) % self.num_cbs
        return self.barrier_semaphore_handles[semaphore_index][current_idx]

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

        # Sampling values / indices.
        # TTSampling forces its batch to max(32, round_up_32(max_batch_size)) (tt_sampling.py),
        # and its device-offsets / k / p / temp tensors are that many rows. The decode topk
        # gather writes into these persistent buffers, so their row count MUST match — else the
        # gather collapses to max_batch_size rows (e.g. 1 at batch-1) while the offsets add
        # broadcasts indices back to 32, giving values=[1,1,1,256] vs indices=[1,1,32,256] and
        # ttnn.sampling's "values and indices must have the same shape" assert. The decode
        # logits are already 32-tile-padded ([1,1,32,V]), so 32 rows is the correct gather width.
        _samp_rows = max(32, ((self.max_batch_size + 31) // 32) * 32)

        # Sampling values
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 1, _samp_rows, self.max_top_k * self.cluster_shape[0])),
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
            torch.zeros((1, 1, _samp_rows, self.max_top_k * self.cluster_shape[0])),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.uint16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["SAMPLING_INDICES"] = tt_buffer
        # Decode return_logits / host-sampling all-gather output buffer (full gathered vocab).
        # qwen3.6: width = padded_vocab 248832 (= per-col 31104 * 8 rows) and dtype bf16 to match
        # the decode lm_head logits (typecast to bf16 in llama_model.py before this gather) — the
        # llama-sized bf8/131072 buffer both under-sized AND dtype-mismatched qwen3.6 and crashed
        # the decode return_logits path (all_gather_async output.dtype==input assert).
        if self.is_qwen36:
            tt_buffer = ttnn.from_torch(
                torch.zeros((1, 1, 32, 248832)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        elif self.is_qwen:
            tt_buffer = ttnn.from_torch(
                torch.zeros((1, 1, 32, 155648)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )
        else:
            tt_buffer = ttnn.from_torch(
                torch.zeros((1, 1, 32, 128 * 1024)),
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
        if self.is_qwen36:
            # qwen3.6 intermediate=17408, 8-row TP → per-row 2176.
            # (Was 1728 stale-from-qwen3-32B; mismatched the actual w1_out
            # shape on the V2-7b MLP path.)
            binary_mul_width = 2176
        elif self.is_qwen:
            binary_mul_width = 3200
        else:
            binary_mul_width = 3584
        tt_buffer = ttnn.from_torch(
            torch.zeros((1, 1, self.max_batch_size, binary_mul_width)),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        persistent_buffers["BINARY_MUL"] = tt_buffer

        # qwen3.6 decode dual-dtype variant — same shape/memcfg but bfloat16.
        # See olmo BRINGUP session-10/12: the optimal-CCL path has hardcoded
        # bfloat8_b assumptions in the C++ kernel, so the SwiGLU output
        # `ff1ff3` carried as bfloat16 (avoids compounding quantisation error
        # across 64 layers) needs a bfloat16 destination buffer.  Pre-allocating
        # here makes the call trace-safe (no per-iteration alloc) and avoids
        # the `ttnn.deallocate(w2_in)` use-after-free bug olmo hit.
        if self.is_qwen36:
            tt_buffer_bf16 = ttnn.from_torch(
                torch.zeros((1, 1, self.max_batch_size, binary_mul_width)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                memory_config=self.model_config["FF2_IN_RING_MEMCFG"],
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
        N_per_shard = (
            2048 // 16 * cluster_shape[cluster_axis] if not self.is_qwen else 1280 // 10 * cluster_shape[cluster_axis]
        )  # FF2/DO
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

    # ------------------------------------------------------------------
    # V2-14: qwen3.6 residual-stream persistent buffers
    # ------------------------------------------------------------------
    # Separate pair of width-sharded persistent buffers for the full-dim=5120
    # residual reduction at the DeltaNet ``_output_proj_and_reduce`` (axis=0,
    # 48 DeltaNet layers) and the full-attn ``_forward_decode_qwen36`` WO
    # (axis=1, 16 layers).  The buffer is spread across more cores than the
    # output (default 40 vs 10) to keep per-core L1 reservation under the
    # static-CB collision threshold.
    def _build_qwen36_residual_buffers(self):
        assert self.is_qwen36, "_build_qwen36_residual_buffers requires is_qwen36"
        # Off by default — opt in via env var so the L1 footprint is only
        # paid when the swap is enabled.  See V2-14 in PERF.md.  Auto-enable
        # when the swap env vars are set so callers don't need both.
        _auto_on = (
            os.environ.get("QWEN36_RESIDUAL_BUF_ON", "0") == "1"
            or os.environ.get("QWEN36_DELTA_LAR", "0") == "1"
            or os.environ.get("QWEN36_FULLATTN_LAR", "0") == "1"
            or os.environ.get("QWEN36_FULLATTN_WO_TUNED", "0") == "1"
            or os.environ.get("QWEN36_DELTA_OP_TUNED", "0") == "1"
            # V2-CCL-followup: needs the residual_output_memcfg even though
            # it does not use the persistent buffer.
            or os.environ.get("QWEN36_FULLATTN_WO_SHARDED", "0") == "1"
            # V2-CCL-P1: full-attention WO line_all_reduce path
            # (see llama_attention.py `_forward_decode_qwen36`).
            or os.environ.get("QWEN36_FULLATTN_WO_LAR", "0") == "1"
        )
        if not _auto_on:
            return

        M = 32
        H = 5120  # qwen3.6 residual-stream dim
        # Number of cores for the buffer; default 40 keeps per-core L1
        # reservation similar to the existing FF2 buffer.  Buffer's grid
        # must contain output's grid (output stays on the 10-core
        # DECODE_RESIDUAL grid for downstream compatibility).
        num_cores_buf = int(os.environ.get("QWEN36_RESIDUAL_CORES", "40"))
        rows_required = num_cores_buf // 5
        residual_core_range = ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(5, max(0, rows_required - 1)))
        residual_core_range_set = ttnn.CoreRangeSet([residual_core_range])

        # Output / input shard widths: per-core w = H / num_cores_buf.
        per_core_w = H // num_cores_buf

        cluster_shape = tuple(self.cluster_shape) if not isinstance(self.cluster_shape, tuple) else self.cluster_shape

        # Output memcfg (same for both axes): width-sharded on the buffer's
        # core grid.  Shard shape (32, per_core_w).
        residual_output_memcfg = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(residual_core_range_set, [M, per_core_w], ttnn.ShardOrientation.ROW_MAJOR),
        )

        _BUILD_AXES = os.environ.get("QWEN36_RESIDUAL_BUF_AXES", "0,1").split(",")
        _BUILD_AXES = tuple(int(a) for a in _BUILD_AXES if a.strip())
        # The all_reduce_async persistent output buffer MUST match the dtype the
        # DeltaNet / full-attention decode out-proj feeds it: both produce bf16 by
        # default and bf8 only under QWEN36_ATTN_OUT_BF8 (see _dn_out_dtype in
        # qwen36_delta_attention.py and _attn_out_dtype in llama_attention.py).
        # Hardcoding bf8 here while the producer is bf16 made all_reduce_async size
        # its CB for bf16 (65536 B) against a bf8-sized L1 bank (34816 B) ->
        # "Cannot set circular buffer size" crash on the FIRST GDN layer of the
        # decode-CCL (switch_mode) path at ISL-4096. The inline demo never hit this
        # because prefill-CCL routes line_all_reduce through reduce_scatter+all_gather,
        # not the persistent-buffer all_reduce_async.
        _resid_buf_dtype = ttnn.bfloat8_b if os.environ.get("QWEN36_ATTN_OUT_BF8", "0") == "1" else ttnn.bfloat16
        for cluster_axis in _BUILD_AXES:
            ring_size = cluster_shape[cluster_axis]
            buf_per_core_w = per_core_w * ring_size
            buf_mem_cfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(residual_core_range_set, [M, buf_per_core_w], ttnn.ShardOrientation.ROW_MAJOR),
            )
            tt_buffer = ttnn.from_torch(
                torch.zeros((*cluster_shape, M, buf_per_core_w * num_cores_buf)),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=_resid_buf_dtype,
                memory_config=buf_mem_cfg,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=cluster_shape),
            )
            self.qwen36_residual_buffers[cluster_axis] = tt_buffer
            self.qwen36_residual_output_memcfgs[cluster_axis] = residual_output_memcfg
            self.qwen36_residual_input_memcfgs[cluster_axis] = residual_output_memcfg

    @staticmethod
    def _free_buffer_tree(obj):
        """Recursively deallocate a nested dict/list/tensor buffer collection."""
        if obj is None:
            return
        if isinstance(obj, dict):
            for v in obj.values():
                TT_CCL._free_buffer_tree(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                TT_CCL._free_buffer_tree(v)
        elif hasattr(obj, "deallocate"):
            try:
                obj.deallocate(True)
            except Exception:
                pass

    def rebuild_prefill_persistent_buffers(self):
        """Re-allocate (fresh-zeroed) the prefill persistent CCL buffers.

        The prefill MLP reduce_scatter / all_gather persistent buffers are DRAM
        tensors built once at init. The decode-mode sub-device manager has an
        INDEPENDENT allocator (qwen3.6 builds a separate all-cores manager per
        mode), so while decode runs it can place its own buffers over the idle
        prefill buffers' backing memory and fill them with large values. On the
        next prefill the MLP CCL reads its persistent buffer's stale
        (decode-written) content -> inf at ISL >= ~4k (coherent first request,
        garbage on every request after the first decode — cross-request
        contamination). At ISL-128 the small prefill buffers don't overlap, so
        short prompts always worked. Rebuilding them fresh (zeroed) at each
        prefill entry restores clean buffers; prefill is eager (no trace), so
        re-placing the addresses is safe."""
        if self.mode != "prefill" or not self.allocate_prefill_buffers:
            return
        self._free_buffer_tree(self.persistent_buffers)
        self._free_buffer_tree(self.all_gather_buffers)
        self.persistent_buffers = (
            self.get_ring_prefill_reduce_scatter_buffers()
            if self.use_ring_rs_prefill
            else self.get_prefill_reduce_scatter_buffers()
        )
        self.all_gather_buffers = self.get_prefill_all_gather_buffers()

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

            # qwen3.6 per-device prefill reduce_scatter widths (FFN=17408, H=5120,
            # 8 rows × 4 cols). FF1/FF3 K-dim per row = 2176 (intermediate_dim_per_tp);
            # FF2 K-dim per col = 1280 (dim_per_tp). The non-qwen36 branches below use
            # 70B / qwen3-32B specific values; mixing them with qwen3.6 weights
            # caused a 1728 vs 2176 matmul mismatch in V2-7b.
            if self.is_qwen36:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    "FF1": [(1, 1, seqlen, 2176), (1, 1, seqlen, 2176 // 4)],
                    "FF3": [(1, 1, seqlen, 2176), (1, 1, seqlen, 2176 // 4)],
                    "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 256), (1, 1, seqlen, 256 // 4)],  # head_dim=256
                }
            elif self.is_qwen:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    "FF1": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF3": [(1, 1, seqlen, 3200), (1, 1, seqlen, 3200 // 4)],
                    "FF2": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 8)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128), (1, 1, seqlen, 128 // 4)],
                }
            else:
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 1280), (1, 1, seqlen, 1280 // 4)],
                    "FF1": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                    "FF3": [(1, 1, seqlen, 3584), (1, 1, seqlen, 3584 // 4)],
                    "FF2": [(1, 1, seqlen, 2048), (1, 1, seqlen, 2048 // 8)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128), (1, 1, seqlen, 128 // 4)],
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

            if self.is_qwen36:
                # qwen3.6 per-device prefill all_gather widths (intermediate=17408,
                # H=5120, head_dim=256, 8 rows × 4 cols):
                #   QKV+Gate per col = (n_q + n_q + n_kv + n_kv)/n_cols * head_dim
                #                    = (24 + 24 + 4 + 4)/4 * 256 = 14*256 = 3584
                #   WO_AG per col   = dim_per_tp = 5120/4 = 1280
                #   FF1/FF3 per row = intermediate_dim_per_tp = 17408/8 = 2176
                #     (line_all_gather inside the MLP gathers the col-scattered
                #      product back to the per-row width 2176; v2 corrected from
                #      the stale qwen3-32B value 13824/8=1728 that crashed the
                #      V2-7b TtTransformer.forward path with a 1728 vs 2176
                #      matmul mismatch on the W2 input gather.)
                #   FF2 per col     = dim_per_tp = 1280
                # SDPA per col      = n_local_heads * head_dim = 6 * 256 = 1536
                # ATTN_REPLICATE     = head_dim per slice (256 for qwen3.6 head_dim).
                buffers_dict = {
                    "QKV": [(1, 1, seqlen, 3584)],
                    "QKV_BF16": [(1, 1, seqlen, 3584)],
                    "SDPA": [(1, 1, seqlen // 2, 1536)],
                    "SDPA_REVERSE": [(1, 1, seqlen // 2, 1536)],
                    "WO_AG": [(8, 1, seqlen, 1280)],
                    "WO_AG_BF16": [(8, 1, seqlen, 1280)],
                    "FF1": [(1, 1, seqlen, 2176)],
                    "FF1_BF16": [(1, 1, seqlen, 2176)],
                    "FF3": [(1, 1, seqlen, 2176)],
                    "FF3_BF16": [(1, 1, seqlen, 2176)],
                    "FF2": [(1, 1, seqlen, 1280)],
                    "LAYERNORM": [(1, 1, seqlen, 128)],
                    "ATTN_REPLICATE": [(1, 1, seqlen, 256)],  # qwen3.6 head_dim=256
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
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128)],  # For prefix caching column replication
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
                    "ATTN_REPLICATE": [(1, 1, seqlen, 128)],  # For prefix caching column replication
                }
            for key, shape in buffers_dict.items():
                # qwen3.6: dual-dtype keys ending with `_BF16` are bfloat16 variants
                # used by the ISL ≤ 2048 / bfloat16 matmul path (olmo session-12).
                # LAYERNORM stats also stay bfloat16.
                use_bf16 = key == "LAYERNORM" or (self.is_qwen36 and key.endswith("_BF16"))
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
        if self.is_qwen36:
            # qwen3.6 padded_vocab_size = 248_832, per-device = 248_832 / 8 = 31_104.
            qwen36_lm_head_width = 31_104
            buffers_fixed_length = {
                "LM_HEAD": [(4, 1, 32, qwen36_lm_head_width)],
                "SAMPLING": [(1, 1, 32, qwen36_lm_head_width * 8)],
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
        # V2-decode: build a FRESH dict for the seqlen=32 entry. The previous
        # ``ag_persistent_buffers[key] = ...`` mutation accidentally leaked the
        # final seqlen=128 LAYERNORM / QKV / FFx buffers into the seqlen=32
        # entry (because ``ag_persistent_buffers`` was the loop's last dict).
        # That bites decode (T=1 tile-padded to 32) when ``line_all_gather``
        # looks up ``all_gather_buffers[32]["LAYERNORM"]`` and gets a [1,1,128,128]
        # buffer — output shape validation then fails ("dim 2 should be 32 but
        # has 128"). Keep the seqlen=32 entry strictly the fixed-length set.
        ag_persistent_buffers_32: dict = {}
        for key, shape in buffers_fixed_length.items():
            # qwen3.6: the SAMPLING all-gather output feeds host-argmax (prefill
            # return_logits) / on-device sampling (decode). The prefill lm_head
            # (lm_head.py) and the decode return_logits gather (llama_model.py ~1092)
            # now emit/typecast logits to bf16 to preserve precision, so this
            # persistent output buffer MUST be bf16 too — else all_gather_async
            # asserts output_tensor.dtype()==input.dtype (bf8 here crashed the
            # generator prefill return_logits path at ISL-4096). LM_HEAD stays bf8.
            _fixed_dtype = ttnn.bfloat16 if (self.is_qwen36 and key == "SAMPLING") else ttnn.bfloat8_b
            tt_buffer = ttnn.as_tensor(
                torch.zeros(shape[0]),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=_fixed_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=self.weight_cache_path / ("pb_ag_" + key + "_32"),
            )
            ag_persistent_buffers_32[key] = tt_buffer

        ag_persistent_buffers_all[32] = ag_persistent_buffers_32
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
        use_qwen36_residual_buffer=False,
    ):
        if self.mode == "decode":
            # BISECTION gate (QWEN36_DECODE_RESIDUAL_RSAG=1): route the decode-only SHARDED
            # residual all_reduce_async (DeltaNet/full-attn out-proj, use_qwen36_residual_buffer)
            # through the SAME reduce_scatter+all_gather path the COHERENT prefill-CCL uses, to
            # test whether the sharded all_reduce_async is the decode-CCL drift source. The
            # residual input/output are L1-width-sharded, so round-trip via DRAM for the
            # DRAM-capable RS+AG and restore the requested sharded memory_config on exit.
            _resid_rsag = (
                self.is_qwen36
                and use_qwen36_residual_buffer
                and not lm_head
                and os.environ.get("QWEN36_DECODE_RESIDUAL_RSAG", "0") == "1"
            )
            if _resid_rsag:
                _in_dram = (
                    ttnn.to_memory_config(input_tensor_mesh, ttnn.DRAM_MEMORY_CONFIG)
                    if input_tensor_mesh.memory_config().shard_spec is not None
                    else input_tensor_mesh
                )
                rs = self.line_reduce_scatter(
                    _in_dram,
                    ttnn.DRAM_MEMORY_CONFIG,
                    dim=3,
                    cluster_axis=cluster_axis,
                    num_links=num_links,
                    math_op=ttnn.ReduceType.Sum,
                    buffer_key=buffer_key,
                    batch_size=batch_size,
                )
                if _in_dram is not input_tensor_mesh:
                    ttnn.deallocate(_in_dram)
                ag = self.line_all_gather(
                    rs,
                    dim=3,
                    cluster_axis=cluster_axis,
                    num_links=num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    buffer_key=buffer_key if buffer_key is not None else "RESID_AR",
                )
                ttnn.deallocate(rs)
                out = ttnn.to_memory_config(ag, memory_config)
                ttnn.deallocate(ag)
                return out
            if (
                self.is_qwen36
                and not lm_head
                and not use_qwen36_residual_buffer
                and input_tensor_mesh.memory_config().shard_spec is None
            ):
                # qwen3.6 DRAM decode reduces (e.g. _mlp_decode_qwen36 w2/FF2) feed an
                # INTERLEAVED input. The persistent-buffer all_reduce_async path expects
                # L1-width-sharded I/O (and a dtype-matched persistent buffer), so an
                # interleaved input crashes (CB clash / bad optional access). Decompose
                # into reduce_scatter + all_gather (both DRAM-capable) — identical to what
                # the prefill-CCL path does for the same reduce, so the math matches the
                # proven inline demo. Sharded inputs keep the fast all_reduce_async below.
                rs = self.line_reduce_scatter(
                    input_tensor_mesh,
                    memory_config,
                    dim=3,
                    cluster_axis=cluster_axis,
                    num_links=num_links,
                    math_op=ttnn.ReduceType.Sum,
                    buffer_key=buffer_key,
                    batch_size=batch_size,
                )
                out = self.line_all_gather(
                    rs,
                    dim=3,
                    cluster_axis=cluster_axis,
                    num_links=num_links,
                    memory_config=memory_config,
                    # decode line_all_gather asserts a non-None buffer_key; it only uses it
                    # for an all_gather_buffers .get() (a miss => fresh output buffer), so
                    # any non-None key is safe. Use the caller's key when present.
                    buffer_key=buffer_key if buffer_key is not None else "INTERLEAVED_AR",
                )
                ttnn.deallocate(rs)
                return out
            if lm_head:
                persistent_buffer = self.tt_lm_head_buffer_l1
            elif use_qwen36_residual_buffer:
                # V2-14: qwen3.6 residual-stream full-dim=5120 reduction.
                assert self.is_qwen36, "use_qwen36_residual_buffer requires is_qwen36"
                persistent_buffer = self.qwen36_residual_buffers[cluster_axis]
                assert persistent_buffer is not None, (
                    f"qwen36 residual buffer for cluster_axis={cluster_axis} not built — "
                    f"check _build_qwen36_residual_buffers was called"
                )
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

    def all_gather_matmul(
        self,
        input_tensor_mesh,
        weight,
        dim=3,
        cluster_axis=1,
        num_links=1,
        ag_memory_config=None,
        mm_memory_config=None,
        program_config=None,
        compute_kernel_config=None,
        dtype=None,
        global_cb=None,
        buffer_key=None,
    ):
        """Fused all-gather + matmul wrapper around
        ``ttnn.experimental.llama_all_gather_matmul_async``.

        Gathers ``input_tensor_mesh`` along ``dim`` across ``cluster_axis`` and
        immediately matmuls the gathered activation against ``weight``, returning
        the matmul output. The (all-gather) interim buffer and the cross-device
        semaphore are sourced from tt_ccl's EXISTING pools (NEVER hand-built):

          * cross-device semaphore: ``self.gather_semaphore_handles[cluster_axis]
            [self.gather_idx[cluster_axis]]`` (the same pool every other wrapper
            uses), cycled via ``gather_idx`` after the call.
          * interim buffer: a persistent width-sharded L1 tensor of the GATHERED
            activation shape, cached in ``self.all_gather_buffers`` keyed by
            ``buffer_key`` so it is allocated once and reused (mirrors how the
            ff2_qwen reference test pre-allocates its own ``tt_intermediate``).
            Its shard spec is taken from ``ag_memory_config`` (the all-gather
            output memcfg) — the hang-prone part is this buffer matching the AG
            output layout exactly, which by construction it does.
          * sub-device / topology: ``self.worker_sub_device_id`` /
            ``self.model_config["CCL_TOPOLOGY"]``.

        Callers therefore never touch raw CCL primitives.
        """
        # --- source/allocate the persistent interim (all-gather) buffer from the pool ---
        interim_key = buffer_key if buffer_key is not None else "AG_MM_DEFAULT"
        intermediate_buffer = self.all_gather_buffers.get(interim_key, None)
        if intermediate_buffer is None:
            # Gathered activation shape, mirroring the ff2_qwen reference test's
            # ``intermediate_tensor``: per-device width = input width, then the
            # mesh's cluster_axis dim carries the ring so the gathered result is
            # ``per_device_width * ring_size`` wide.  Build a full
            # ``[*cluster_shape, M, per_device_width * ring_size]`` host tensor and
            # 2D-shard it (dims=(0,1)) so each device owns the AG-output layout
            # exactly described by ``ag_memory_config``.
            ring_size = self.cluster_shape[cluster_axis]
            per_device_width = input_tensor_mesh.shape[dim]
            M = input_tensor_mesh.shape[-2]
            intermediate_shape = [*self.cluster_shape, M, per_device_width * ring_size]
            intermediate_buffer = ttnn.from_torch(
                torch.zeros(intermediate_shape),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=input_tensor_mesh.dtype,
                memory_config=ag_memory_config,
                mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(0, 1), mesh_shape=self.cluster_shape),
            )
            self.all_gather_buffers[interim_key] = intermediate_buffer

        mm_out = ttnn.experimental.llama_all_gather_matmul_async(
            input_tensor_mesh,
            weight,
            intermediate_buffer,
            dim=dim,
            cluster_axis=cluster_axis,
            mesh_device=self.mesh_device,
            multi_device_global_semaphore=self.gather_semaphore_handles[cluster_axis][self.gather_idx[cluster_axis]],
            ag_memory_config=ag_memory_config,
            mm_memory_config=mm_memory_config,
            topology=self.model_config["CCL_TOPOLOGY"],
            num_links=num_links,
            subdevice_id=self.worker_sub_device_id,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            dtype=dtype,
            global_cb=global_cb,
        )
        self.gather_idx[cluster_axis] = (self.gather_idx[cluster_axis] + 1) % self.num_cbs
        return mm_out

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

        elif input_tensor_mesh.memory_config().shard_spec is None:
            # qwen3.6 DRAM decode MLP (_mlp_decode_qwen36) feeds an INTERLEAVED (DRAM)
            # input here. The sharded-only ``llama_reduce_scatter`` below dereferences
            # the input's shard_spec -> `bad optional access` on interleaved input.
            # Route interleaved input through the generic DRAM-capable
            # ``ttnn.reduce_scatter`` (olmo-style), mirroring the prefill branch above;
            # sharded inputs still take the fast llama_reduce_scatter path.
            B = input_tensor_mesh.shape[1]
            input_reshaped = ttnn.reshape(
                input_tensor_mesh, (1, 1, B * input_tensor_mesh.shape[-2], input_tensor_mesh.shape[-1])
            )
            seqlen = input_reshaped.shape[-2]
            if self._decode_rs_async and self.decode_rs_semaphores[cluster_axis]:
                # async ring reduce-scatter (unit-validated PCC 0.99999, 1.26x). Tolerates interleaved DRAM.
                _idx = self.decode_rs_idx[cluster_axis]
                ttnn_tensor_out = ttnn.experimental.reduce_scatter_minimal_async(
                    input_tensor=input_reshaped,
                    persistent_output_buffers=None,
                    dim=dim,
                    multi_device_global_semaphore=self.decode_rs_semaphores[cluster_axis][_idx],
                    barrier_semaphore=self.get_and_cycle_barrier_semaphore_handle(cluster_axis),
                    num_links=num_links,
                    memory_config=memory_config,
                    topology=ttnn.Topology.Ring,
                    subdevice_id=self.worker_sub_device_id,
                    cluster_axis=cluster_axis,
                    num_workers_per_link=1,
                )
                self.decode_rs_idx[cluster_axis] = (_idx + 1) % self.num_cbs
            else:
                ttnn_tensor_out = ttnn.reduce_scatter(
                    input_reshaped,
                    dim,
                    cluster_axis=cluster_axis,
                    memory_config=memory_config,
                    topology=ttnn.Topology.Linear,
                    num_links=num_links,
                    subdevice_id=self.worker_sub_device_id,
                )
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
        num_links = self.model_config["GALAXY_NUM_LINKS"]
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
            # qwen3.6 DRAM decode (e.g. _mlp_decode_qwen36's step-4 ff gather, and the
            # interleaved all_reduce decomposition above) issues all-gathers with no
            # persistent buffer — valid in prefill, which simply allocates a fresh
            # output. Tolerate buffer_key=None here too (None -> fresh output via the
            # barrier-semaphore path below) instead of asserting. llama70b decode
            # callers always pass a key, so their behaviour is unchanged.
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
            # reshape input back; skip for LM_HEAD and WO_AG (both dtype variants)
            # since their callers handle the shape (e.g. fast_reduce_nc along dim=0).
            if buffer_key not in ["LM_HEAD", "WO_AG", "WO_AG_BF16"]:
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

        num_links = self.model_config["GALAXY_NUM_LINKS"]
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

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule


class TtLlamaAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher_setup=None,
        tt_ccl=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.TG = self.num_devices == 32
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.num_reduce_scatter_links = configuration.num_reduce_scatter_links
        self.num_all_gather_links = configuration.num_all_gather_links

        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl

        # TODO: Fix this once all-gather supports < tile_size
        if self.TG:
            weight = torch.zeros(1, 32, 8, 32)
            for i in range(32):
                col = i % 4  # This determines which group of 8 to select
                weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

            # Select batch_offset with create_qkv_heads_decode instead of selection matmul
            batch_offset = [
                0,
                8,
                16,
                24,
            ]  # TODO: batch offset is 8 for batch=32, this should be adjusted for variable batch_size
            self.batch_offset_tt_tensor = ttnn.as_tensor(
                torch.tensor(batch_offset, dtype=torch.int32).reshape(4, 1),
                dtype=ttnn.int32,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device=mesh_device, dims=(None, 0), mesh_shape=list(mesh_device.shape)
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.slice_size = 8  # Slice size is 8 since we are consuming 8 users per chip

        self.dtype = dtype

        self.max_seq_len = configuration.max_seq_len
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats

        self.model_config = configuration.get_model_config()
        self.model_config["USE_PREFETCHER"] = configuration.use_prefetcher
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        assert configuration.qkv_size % self.num_devices_per_group == 0
        assert configuration.dim % self.num_devices_per_group == 0

        # wqkv: 4096 x 3072 (2 devices): width-sharded on 12 banks, 3072 over 12 banks.
        wqkv_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim, configuration.qkv_size // configuration.num_devices
        )

        qkv_list = []
        for i in range(self.num_devices_per_group):
            # Chunk weights
            wq_selected = torch.chunk(self.state_dict[wq_str], self.num_devices_per_group, dim=0)[i]
            wk_selected = torch.chunk(self.state_dict[wk_str], self.num_devices_per_group, dim=0)[i]
            wv_selected = torch.chunk(self.state_dict[wv_str], self.num_devices_per_group, dim=0)[i]

            # Transpose the selected chunks
            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        # Ring stuff
        # 9216, 12288

        # [1, 1, 8192, 10240] -> [2304, 1536]
        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["SHARDED_QKV_RING_MEMCFG"] if self.TG else wqkv_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2) if self.TG else (2, 3), mesh_shape=configuration.cluster_shape
            ),
            cache_file_name=cache_name("wqkv_sharded_2d_prefetcher"),  ## TODO: Fix caching
        )
        self.wqkv_interleaved = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2) if self.TG else (2, 3), mesh_shape=configuration.cluster_shape
            ),
            cache_file_name=cache_name("wqkv_sharded_2d_dram"),  ## TODO: Fix caching
        )

        # For ring topology we can use all gather matmul for wo
        self.use_fused_all_gather_matmul = self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]
        pt_wo = self.state_dict[wo_str].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        wo_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim // configuration.num_devices, configuration.dim
        )

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["SHARDED_WO_RING_MEMCFG"]
            if (self.use_fused_all_gather_matmul or self.TG)
            else wo_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3) if (self.use_fused_all_gather_matmul or self.TG) else (3, 2),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wo_width_sharded_2d_prefetcher")
            if (self.use_fused_all_gather_matmul or self.TG)
            else cache_name("wo"),
        )
        self.wo_interleaved = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3) if (self.use_fused_all_gather_matmul or self.TG) else (3, 2),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wo_width_sharded_2d_dram"),
        )
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        self.scale = self.head_dim**-0.5
        if tt_ccl.mode == "decode":
            self.prefetch(prefetcher_setup, tt_ccl)

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        if tt_ccl.mode == "decode":
            self.prefetcher_setup.insert_tensor(self.wqkv)
            self.prefetcher_setup.insert_tensor(self.wo)
        self.tt_ccl = tt_ccl

    def init_kv_cache(self, configuration, weight_cache_path):
        """
        Generates empty KV cache and pushed to device memory
        """

        if self.paged_attention_config:
            cache_k = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
        else:
            cache_k = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=self.dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                if weight_cache_path and not configuration.dummy_weights
                else None,
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """
        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        xqkv_fused_sharded = ttnn.matmul(
            x,
            self.wqkv,
            program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
            memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            dtype=ttnn.bfloat16,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id,
        )
        ttnn.deallocate(x)
        # xqkv_fused_sharded -> [1, 1, 32, 12288 // 8]

        ###
        # Reshape and rotary embeddings
        ###
        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = self.tt_ccl.llama_rs_create_heads(
            xqkv_fused_sharded,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            dim=3,
            qkv_memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )

        # print("done create qkv heads")
        ttnn.deallocate(xqkv_fused_sharded)
        # Q, K Rotary Embeddings
        q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)

        # print("done rotary embeddings")

        ###
        # KV update
        ###
        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]
        # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
        # v_heads [seqlen, n_kv_heads, bsz, head_dim]
        # keys, [max_batch_size, n_kv_heads // configuration.num_devices, max_seq_len, head_dim]
        ttnn.experimental.paged_fused_update_cache(
            keys, k_heads_1BKD, values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
        )

        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        # print("done update cache")
        # NOTE: Varying the batch size will result in slightly different outputs.
        # For example, a prompt w/ 1 user vs, the same prompt repeated N times for N users, will produce different outputs
        # This is because the SDPA op in decode mode has different number of reductions depending on batch size
        # Which leads to slightly different outputs from attention (due to accumulated errors)
        sdpa_out_mem_cfg = self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group)
        if page_table:
            attn_output_1G4D_sharded = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                page_table_tensor=page_table,
                scale=self.scale,
                program_config=self.model_config["PAGED_SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=sdpa_out_mem_cfg,
            )
        else:
            attn_output_1G4D_sharded = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=sdpa_out_mem_cfg,  # FIXME: why not L1 height sharded e.g. SCORES_BATCHED_MM_OUTPUT_MEMCFG?
            )

        ttnn.deallocate(q_heads_1BQD)

        attn_output_cat = self.tt_ccl.all_gather_concat(
            attn_output_1G4D_sharded,
            dim=1,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_1G4D_sharded)
        # print("done concat heads")

        # Original matmul on each device [1, 1, 32, 1024] @ [1, 1, 1024, 2048]
        dense_out_ttnn = ttnn.matmul(
            attn_output_cat,
            self.wo,
            program_config=self.model_config["WO_DECODE_RING_PROGCFG"],
            memory_config=self.model_config["SHARDED_WO_OUT_RING_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            dtype=ttnn.bfloat8_b,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id,
        )
        # [1, 1, 32, 2304]
        # print("done matmul")
        dense_out_reduced = self.tt_ccl.line_all_reduce(
            dense_out_ttnn,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )
        ttnn.deallocate(dense_out_ttnn)

        # print("done all reduce")

        return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ):
        if batch_size > 1:
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 2048, 2048, -1])

        xqkv = ttnn.linear(
            x_11SH,
            self.wqkv_interleaved,
            dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )

        ttnn.deallocate(x_11SH)

        xqkv_fused = self.tt_ccl.line_all_reduce(
            xqkv,
            cluster_axis=1,
            num_links=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="QKV",
        )
        ttnn.deallocate(xqkv)

        if seq_len > 2048:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        if batch_size > 1:
            xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])

        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            q_heads_1QSD_pre_rot_bf8 = q_heads_1QSD_pre_rot
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)
            ttnn.deallocate(q_heads_1QSD_pre_rot_bf8)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            k_heads_1KSD_pre_rot_bf8 = k_heads_1KSD_pre_rot
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)
            ttnn.deallocate(k_heads_1KSD_pre_rot_bf8)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        # Fill KV-Cache
        if kv_cache:
            keys_BKSD, values_BKSD = kv_cache[0], kv_cache[1]
        else:
            keys_BKSD, values_BKSD = self.layer_past[0], self.layer_past[1]

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(k_heads_1KSD)

        k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)

        ttnn.deallocate(v_heads_1VSD)

        v_fill = v_heads_1VSD_8b

        if batch_size > 1:
            k_fill = ttnn.reshape(k_fill, [1, 1, seq_len, -1])
            v_fill = ttnn.reshape(v_fill, [1, 1, seq_len, -1])

        if self.TG and not page_table:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)
        if page_table:
            if isinstance(user_id, int):
                user_id = ttnn.from_torch(
                    torch.tensor([user_id], dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill, page_table, batch_idx_tensor=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill, page_table, batch_idx_tensor=user_id)

        else:
            ttnn.fill_cache(
                keys_BKSD,
                k_fill,
                user_id % self.batch_size_per_device_group,
            )
            ttnn.fill_cache(
                values_BKSD,
                v_fill,
                user_id % self.batch_size_per_device_group,
            )

        # SDPA
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        # Run ring_distributed_sdpa for > 1k seqlen because we are seeing worse perf for <=1k seqlen as compared to regular SDPA
        # ring_distributed_sdpa needs seqlen//8 to be atleast one tile (32)
        ring_distributed_sdpa = seq_len > 1024 and batch_size == 1
        if ring_distributed_sdpa:
            # Ring attention splits seqlen into 8 chunks and computes chunk i and chunk ring_size - i - 1 per device
            # where i (device id on a mesh column) ranges from 0 to ring_size-1 (0 to 3), so ring_size - i - 1 ranges from 3 to 0
            # This ensures each device processes two complementary chunks of the attention matrix
            attn_output_1QSD = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                ring_size=4,  # Number of devices in the ring topology (4 devices per row in 8x4 mesh)
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            )
        else:
            attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                is_causal=True,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            )

        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        if ring_distributed_sdpa:
            # Split the attention output into two chunks along the sequence dimension for ring all-gather
            # 4 = ring_size (number of devices in ring), 2 = number of chunks per device
            attn_output_11SH_chunks = ttnn.split(attn_output_11SH, seq_len // 4 // 2, dim=2)
            attn_output_11SH.deallocate(True)

            # Perform ring all-gather on the first chunk (normal order)
            attn_output_11SH_chunk_0 = self.tt_ccl.ring_all_gather(
                attn_output_11SH_chunks[0],
                dim=2,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="SDPA",
            )
            attn_output_11SH_chunks[0].deallocate(True)

            # Perform ring all-gather on the second chunk (reverse order)
            attn_output_11SH_chunk_1 = self.tt_ccl.ring_all_gather(
                attn_output_11SH_chunks[1],
                dim=2,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                reverse_order=True,
                buffer_key="SDPA_REVERSE",
            )
            attn_output_11SH_chunks[1].deallocate(True)

            # Concatenate the gathered chunks along the sequence dimension to form the final output
            attn_output_11SH = ttnn.concat([attn_output_11SH_chunk_0, attn_output_11SH_chunk_1], dim=2)
        if batch_size > 1:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 1, seq_len, -1])

        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo_interleaved,
            compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # Reduce-scatter
        output_11SH = self.tt_ccl.line_all_reduce(
            output_11SH,
            cluster_axis=0,
            num_links=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="WO",
        )

        return output_11SH

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
        else:
            return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)

    def prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer, user_id):
        tensor_copy = ttnn.clone(key_or_value_layer)
        # key_or_value_layer.deallocate(True)
        # Get all tensors from multi-device tensor
        tensors = ttnn.get_device_tensors(tensor_copy)
        # Get only tensors from specific column chips
        # Get every 4th tensor starting from user_id // 8
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: 4]
        # Create multi-device tensor
        multi_device_tensor = ttnn.combine_device_tensors(single_column_tensors)

        return multi_device_tensor

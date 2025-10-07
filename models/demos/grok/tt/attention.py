# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

# Simplified Attention for TG=True, use_fused_all_gather_matmul=False, no QK norm, decode mode only


import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.model_config import TensorGroup


class Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        # Simplified: TG=True always (num_devices == 32)
        self.TG = True
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.ccl_dtype = configuration.ccl_dtype
        self.num_reduce_scatter_links = configuration.num_reduce_scatter_links
        self.num_all_gather_links = configuration.num_all_gather_links
        self.tile_size = configuration.tile_size
        self.num_device_groups = self.num_devices // self.n_kv_heads
        # TG=True: use n_kv_heads for devices per group
        self.num_devices_per_group = self.n_kv_heads
        # TG=True: batch size calculation
        self.batch_size_per_device_group = max(self.max_batch_size // self.num_device_groups, 1)

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        # TG=True: Always create slice and user selection matrices
        weight = torch.zeros(1, 32, 8, 32)
        for i in range(32):
            col = i % 4  # This determines which group of 8 to select
            weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

        self.slice_mat = ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
        )
        user_selection_matrix = torch.eye(8, 8)
        user_selection_matrix = torch.nn.functional.pad(user_selection_matrix, (0, 24), "constant", 0)  # (8, 32)
        user_selection_matrix = [user_selection_matrix] * 4
        user_selection_matrix = torch.block_diag(*user_selection_matrix)  # (32, 128)
        self.user_selection_matrix = ttnn.from_torch(
            user_selection_matrix,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.dtype = dtype

        self.max_seq_len = configuration.max_seq_len
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2

        self.transformation_mats = transformation_mats

        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        self.wqkv_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WQKV
        )
        self.wo_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WO
        )
        self.kv_cache_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
        )
        self.li_qkv_decode_compute_kernel_cfg = configuration.compute_kernel_config_hifi2
        self.sdpa_decode_compute_kernel_cfg = configuration.compute_kernel_config_hifi2
        self.li_o_decode_compute_kernel_cfg = configuration.compute_kernel_config_hifi2
        self.sdpa_prefill_compute_kernel_cfg = configuration.compute_kernel_config_hifi2
        self.li_qkv_prefill_compute_kernel_cfg = configuration.compute_kernel_config_hifi2
        self.li_o_prefill_compute_kernel_cfg = configuration.compute_kernel_config_hifi2

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wqkv_str = f"{layer_name}.wqkv"
        wo_str = f"{layer_name}.wo"

        wqkv_torch = state_dict[f"{wqkv_str}.weight"].unsqueeze(0).unsqueeze(0)
        self.wqkv = ttnn.as_tensor(
            wqkv_torch,
            dtype=self.wqkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # TG=True
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape  # TG=True
            ),
            # cache_file_name=cache_name("wqkv_sharded_2d"),
        )

        # Simplified: use_fused_all_gather_matmul=False, TG=True
        self.use_fused_all_gather_matmul = False
        wo_torch = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        self.wo = ttnn.as_tensor(
            wo_torch,
            dtype=self.wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,  # TG=True
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3),  # TG=True
                mesh_shape=configuration.cluster_shape,
            ),
            # cache_file_name=cache_name("wo_width_sharded_2d"),  # TG=True
        )
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        self.scale = self.head_dim**-0.5

    def init_kv_cache(self, configuration, weight_cache_path):
        """
        Generates empty KV cache and pushed to device memory
        """
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
        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=self.kv_cache_dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=(
                    f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                    if weight_cache_path and not configuration.dummy_weights
                    else None
                ),
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def forward_decode(
        self, x: ttnn.Tensor, current_pos, rot_mats=None, page_table=None, kv_cache=None, attn_mask=None
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """

        xqkv_fused_sharded = ttnn.linear(
            x,
            self.wqkv,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.model_config["XQKV_DECODE_PROGCFG"],
            compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
            dtype=self.ccl_dtype,  # TG=True
        )

        # ttnn.deallocate(x)
        xqkv_fused = tt_all_reduce(
            xqkv_fused_sharded,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            num_reduce_scatter_links=self.num_reduce_scatter_links,
            num_all_gather_links=self.num_all_gather_links,
            memory_config=self.model_config["QKV_OUT_GATHERED_MEMCFG"](list(self.mesh_device.shape)[1]),
            sharded=True,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
        )
        # xqkv_torch = ttnn.to_torch(xqkv_fused, mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(3, 1), mesh_shape=(8,4)))

        # TG=True: Always slice the fused_query_key_value tensor get batch=8
        xqkv_fused = ttnn.matmul(
            self.slice_mat,
            xqkv_fused,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["CREATE_HEAD_INPUT_MEMCFG"],
        )

        ttnn.deallocate(xqkv_fused_sharded)

        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3])
        )

        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=self.model_config["CREATE_QKV_DECODE_SHARD"],
        )

        ttnn.deallocate(xqkv_fused)

        # Q Rotary Embeddings
        q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre_rot_1BQD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )
        # K Rotary Embeddings
        k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)

        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]

        ttnn.experimental.paged_update_cache(keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(
            values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
        )

        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_heads_1BQD,
            keys,
            values,
            cur_pos_tensor=current_pos,
            page_table_tensor=page_table,
            attn_mask=attn_mask,
            is_causal=True if attn_mask is None else False,
            scale=self.scale,
            program_config=self.model_config["SDPA_DECODE_PROGCFG"],
            compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(q_heads_1BQD)

        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D,
            memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group),
        )
        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_11BH)
        ttnn.deallocate(attn_output_1G4D)

        # Simplified: use_fused_all_gather_matmul=False always
        attn_output = tt_all_gather(
            attn_output_cat,
            self.mesh_device,
            self.tt_ccl,
            dim=2,
            cluster_axis=1,
            num_links=2,
            memory_config=self.model_config["GATHER_USERS_MEMCFG"](list(self.mesh_device.shape)[1]),
            sharded=True,
            # dtype=self.ccl_dtype,  # Running bf16 until we have SDPA output bfp8 df; otherwise we have two sharded to interleaved/interleaved to sharded conversions
        )
        # TG=True: Always apply user selection matrix
        attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
        # user_selection_matrix = [1, 1, 32, 128]
        # user_selection_matrix @ activation -> [1, 1, 32, 128] * [1, 1, 128, 2048] -> [1, 1, 32, 2048]
        attn_output = ttnn.matmul(
            self.user_selection_matrix,
            attn_output,
            core_grid=ttnn.CoreGrid(y=4, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        # TG=True: Always use TG-specific matmul config
        dense_out_sharded = ttnn.matmul(
            attn_output,
            self.wo,
            core_grid=ttnn.CoreGrid(y=4, x=8),  # TG=True
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,  # TG=True
            compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
        )

        ttnn.deallocate(attn_output_cat)

        # All reduce
        dense_out_reduced = tt_all_reduce(
            dense_out_sharded,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=0,
            num_reduce_scatter_links=self.num_reduce_scatter_links,
            num_all_gather_links=self.num_all_gather_links,
            dim=3,  # TG=True and hidden_size=8192, so use dim=3
            topology=self.ccl_topology,
            memory_config=self.model_config["SELF_OUT_REDUCE_SCATTER_MEMCFG"],  # TG=True and hidden_size=8192
            sharded=True,
            dtype=self.ccl_dtype,
            use_composite=True,  # hidden_size=8192
        )

        # TG=True: No need to convert memory config
        return dense_out_reduced

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
        attn_mask=None,
    ):
        # Simplified: decode mode only
        return self.forward_decode(
            x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache, attn_mask=attn_mask
        )

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm


class TtLlamaCrossAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        dim,
        head_dim,
        n_heads,
        n_kv_heads,
        norm_eps,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps

        self.n_local_heads = self.n_heads // configuration.num_devices
        self.n_local_kv_heads = self.n_kv_heads // configuration.num_devices

        self.dtype = dtype

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4
        self.compute_kernel_config_sdpa = configuration.compute_kernel_config_sdpa

        self.configuration = configuration

        self.model_config = configuration.get_model_config()
        self.is_multichip = configuration.is_multichip

        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{state_dict_prefix}.{name}")

        wq_str = f"{state_dict_prefix}wq.weight"
        wk_str = f"{state_dict_prefix}wk.weight"
        wv_str = f"{state_dict_prefix}wv.weight"
        wo_str = f"{state_dict_prefix}wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % configuration.num_devices == 0
        assert self.n_kv_heads % configuration.num_devices == 0

        # TODO DRAM Shard the weights (see llama3 text)
        self.wq = ttnn.as_tensor(
            self.state_dict[wq_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wq_sharded"),
        )

        self.wk = ttnn.as_tensor(
            self.state_dict[wk_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wk_sharded"),
        )

        self.wv = ttnn.as_tensor(
            self.state_dict[wv_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wv_sharded"),
        )

        self.wo = ttnn.as_tensor(
            self.state_dict[wo_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wo_sharded"),
        )

        self.scale = self.head_dim**-0.5

        self.q_norm = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}",
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_key="q_norm",
            eps=self.norm_eps,
        )

        self.k_norm = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}",
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_key="k_norm",
            eps=self.norm_eps,
        )

    def compute_xattn_kv_cache(self, xattn_tokens, user_id, xattn_cache, cross_page_table=None):
        """
        Uses xattn_tokens to compute K, V. Should be run inside of forward_prefill.
        Updates xattn_cache with K, V (TODO: support page table for KV cache)
        Returns contiguous K, V of this user in DRAM
        """
        # Always runs with batch=1
        B, seqlen_y = xattn_tokens.shape[1], xattn_tokens.shape[2]
        assert B == 1, "Batch size must be 1"
        MAX_MM_SEQ_LEN = self.configuration.VISION_MAX_MM_SEQ
        if seqlen_y > MAX_MM_SEQ_LEN:
            xattn_tokens = ttnn.reshape(xattn_tokens, [1, B * seqlen_y // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        xk = ttnn.linear(
            xattn_tokens,
            self.wk,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_KV_PROGCFG"](seqlen_y, MAX_MM_SEQ_LEN),
        )
        xv = ttnn.linear(
            xattn_tokens,
            self.wv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_KV_PROGCFG"](seqlen_y, MAX_MM_SEQ_LEN),
        )
        if seqlen_y > MAX_MM_SEQ_LEN:
            xk = ttnn.reshape(xk, [1, B, seqlen_y, -1])
            xv = ttnn.reshape(xv, [1, B, seqlen_y, -1])

        if self.n_local_kv_heads == 1:
            # Only a simple reshape required, no need to split
            xk = ttnn.reshape(xk, [B, 1, seqlen_y, -1])
            xv = ttnn.reshape(xv, [B, 1, seqlen_y, -1])
        else:
            # 1, B, S, D -> B, NH, S, DH
            xk, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                xk,
                xk,
                num_heads=self.n_local_kv_heads,
                num_kv_heads=self.n_local_kv_heads // 2,
                transpose_k_heads=False,
            )
            xv, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                xv,
                xv,
                num_heads=self.n_local_kv_heads,
                num_kv_heads=self.n_local_kv_heads // 2,
                transpose_k_heads=False,
            )

        xk = self.k_norm(xk, mode="decode")

        k_cache, v_cache = xattn_cache

        if cross_page_table:
            ttnn.experimental.paged_fill_cache(
                k_cache, ttnn.experimental.typecast(xk, k_cache.dtype), cross_page_table, batch_idx=user_id
            )
            ttnn.experimental.paged_fill_cache(
                v_cache, ttnn.experimental.typecast(xv, v_cache.dtype), cross_page_table, batch_idx=user_id
            )
        else:
            # Work around fill_cache memory constraint by making these sharded
            k_fill = ttnn.interleaved_to_sharded(xk, self.model_config["XATTN_KV_PREFILL_MEM_CFG"](seqlen_y))
            v_fill = ttnn.interleaved_to_sharded(xv, self.model_config["XATTN_KV_PREFILL_MEM_CFG"](seqlen_y))

            ttnn.fill_cache(k_cache, k_fill, user_id)
            ttnn.fill_cache(v_cache, v_fill, user_id)

        return xk, xv

    def forward_decode(
        self, x_11SH, xattn_mask, full_text_row_masked_out_mask_1NSH, xattn_cache, cross_page_table=None
    ):
        batch = xattn_mask.shape[1]

        x_11SH = ttnn.sharded_to_interleaved(x_11SH, ttnn.L1_MEMORY_CONFIG)  # TODO support sharded input

        xq = ttnn.linear(
            x_11SH,
            self.wq,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_Q_PROGCFG"](batch),
        )

        xq, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xq, xq, num_heads=self.n_local_heads, num_kv_heads=self.n_local_heads // 2, transpose_k_heads=False
        )
        xq = ttnn.to_layout(xq, layout=ttnn.ROW_MAJOR_LAYOUT)
        xq = ttnn.slice(xq, (0, 0, 0, 0), (xq.shape[0], xq.shape[1], batch, xq.shape[3]))
        xq = ttnn.transpose(xq, 1, 2)
        xq = ttnn.to_layout(xq, layout=ttnn.TILE_LAYOUT)

        xq = self.q_norm(xq, mode="decode")

        xk, xv = xattn_cache
        cache_seq_len = xk.shape[-2]

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        # TODO: Can I get rid of the KV repeat_interleave?
        if cross_page_table:
            output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                xq,
                xk,
                xv,
                is_causal=False,
                attn_mask=xattn_mask,
                page_table_tensor=cross_page_table,
                scale=self.scale,
                program_config=program_config,
                compute_kernel_config=self.compute_kernel_config_sdpa,
            )
        else:
            output = ttnn.transformer.scaled_dot_product_attention_decode(
                xq,
                xk,
                xv,
                is_causal=False,
                attn_mask=xattn_mask,
                scale=self.scale,
                program_config=program_config,
                compute_kernel_config=self.compute_kernel_config_sdpa,
            )

        # WARNING: this broadcast is also broken, must broadcast on host
        output = ttnn.mul(output, full_text_row_masked_out_mask_1NSH)

        output = ttnn.to_layout(output, layout=ttnn.ROW_MAJOR_LAYOUT)
        output = ttnn.transpose(output, 1, 2)  # 1, B, NH, DH -> 1, NH, B, DH
        output = ttnn.slice(output, (0, 0, 0, 0), (1, self.n_local_heads, batch, self.head_dim))
        output = ttnn.to_layout(output, layout=ttnn.TILE_LAYOUT)
        output = ttnn.experimental.nlp_concat_heads(output)
        # NOTE: The rest of the model expects output to be padded to tile height
        output = ttnn.reshape(output, output.padded_shape, output.padded_shape)

        output = ttnn.matmul(
            output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["VISION_XATTN_DENSE_PROGCFG"](batch),
        )

        # All reduce
        if self.is_multichip:
            output = ttnn.reduce_scatter(
                output,
                dim=3,
                math_op=ttnn.ReduceType.Sum,
                num_links=1,
                topology=self.configuration.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        return ttnn.to_memory_config(output, self.model_config["DECODE_RESIDUAL_MEMCFG"])

    def forward_prefill(
        self,
        x_11SH,
        xattn_mask,
        full_text_row_masked_out_mask_1NSH,
        xattn_cache,
        user_id,
        vision_tokens,
        cross_page_table=None,
    ):
        seq_len = x_11SH.shape[-2]
        # B, S, D
        assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 32"

        # Compute cross attention cache. Return contiguous caches
        k_cache_user, v_cache_user = self.compute_xattn_kv_cache(
            vision_tokens, user_id, xattn_cache, cross_page_table=cross_page_table
        )
        cache_seq_len = k_cache_user.shape[-2]

        if seq_len > 1024:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 1024, 1024, -1])

        xq = ttnn.linear(
            x_11SH,
            self.wq,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_Q_PROGCFG"](seq_len),
        )
        if seq_len > 1024:
            xq = ttnn.reshape(xq, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

        xq, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xq, xq, num_heads=self.n_local_heads, num_kv_heads=self.n_local_heads // 2, transpose_k_heads=False
        )

        xq = self.q_norm(xq, mode="prefill")

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )

        output = ttnn.transformer.scaled_dot_product_attention(
            xq,
            k_cache_user,
            v_cache_user,
            is_causal=False,
            attn_mask=xattn_mask,
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=self.compute_kernel_config_sdpa,
        )

        # WARNING: this broadcast is also broken, must broadcast on host
        output = ttnn.mul(output, full_text_row_masked_out_mask_1NSH)

        output = ttnn.experimental.nlp_concat_heads(output)
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, seq_len // 1024, 1024, -1])

        output = ttnn.matmul(
            output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["VISION_XATTN_DENSE_PROGCFG"](seq_len),
        )
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        # Reduce-scatter
        if self.is_multichip:  # TODO use_fused_all_gather_matmul
            dense_out_reduced = ttnn.reduce_scatter(
                output,
                dim=3,
                math_op=ttnn.ReduceType.Sum,
                num_links=1,
                topology=self.configuration.ccl_topology(),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            return dense_out_reduced
        else:
            return output

    def forward(
        self,
        x_11SH,
        xattn_mask,
        full_text_row_masked_out_mask_1NSH,
        xattn_cache,
        mode,
        user_id=0,
        vision_tokens=None,
        cross_page_table=None,
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x_11SH,
                xattn_mask,
                full_text_row_masked_out_mask_1NSH,
                xattn_cache,
                user_id=user_id,
                vision_tokens=vision_tokens,
                cross_page_table=cross_page_table,
            )
        else:
            return self.forward_decode(
                x_11SH, xattn_mask, full_text_row_masked_out_mask_1NSH, xattn_cache, cross_page_table=cross_page_table
            )

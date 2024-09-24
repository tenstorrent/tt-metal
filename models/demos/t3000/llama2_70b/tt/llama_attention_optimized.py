# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import math
import torch
import ttnn
from ttnn import ShardTensorToMesh


class TtLlamaAttention_optimized:
    def __init__(
        self,
        mesh_device,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        transformation_mats,
        cache_path=None,
        batch_size=None,
        read_cache=False,
        paged_attention_config=None,
        vllm=False,
    ):
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
        self.model_config = model_config
        self.read_cache = read_cache
        self.paged_attention_config = paged_attention_config
        self.vllm = vllm

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_kv_heads = configuration.n_kv_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = model_config["MAX_BATCH_SIZE"]
        self.llama3 = configuration.vocab_size == 128256
        self.scale = 1 / math.sqrt(self.head_dim)

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0
        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices
        self.padded_local_heads = 32

        self.layer_num = layer_num
        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path
        self.transformation_mats = transformation_mats

        self.kv_dtype = ttnn.bfloat8_b

        self.load_weights()
        if not vllm:
            # vLLM provides its own kv cache
            self.init_kv_cache()

    def set_model_config(self, model_config):
        self.model_config = model_config

    def init_kv_cache(self):
        """
        Generates empty KV cache and pushed to device memory
        """

        if self.paged_attention_config:
            cache_k = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
        else:
            cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads,
                    self.model_config["MAX_CONTEXT_LEN"],
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads,
                    self.model_config["MAX_CONTEXT_LEN"],
                    self.head_dim,
                )
            )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.to_device(
                ttnn.as_tensor(
                    lp,
                    device=self.mesh_device,
                    mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=1),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=self.kv_dtype,
                    cache_file_name=self.cache_path / f"empty_attn_cache{cache_k.shape}",
                ),
                self.mesh_device,
            )
            for lp in layer_past
        ]

    def load_weights(self):
        assert not hasattr(self, "qkv_list"), "qkv_list is already an attribute of this object"
        assert not hasattr(self, "wo_list"), "wo_list is already an attribute of this object"
        # Load weights
        wqkv_cache_str = f"{self.layer_name}.attention.wqkv_fused.weight"
        wq_str = f"{self.layer_name}.attention.wq.weight"
        wk_str = f"{self.layer_name}.attention.wk.weight"
        wv_str = f"{self.layer_name}.attention.wv.weight"
        wo_str = f"{self.layer_name}.attention.wo.weight"

        qkv_cat = None
        pt_wo = None
        if not self.read_cache:
            qkv_list = []
            for i in range(self.num_devices):
                ### Fused QKV Weights
                # Chunk weights
                wq_chunks = torch.chunk(self.state_dict[wq_str], self.n_heads, dim=0)
                wk_chunks = torch.chunk(self.state_dict[wk_str], self.n_kv_heads, dim=0)
                wv_chunks = torch.chunk(self.state_dict[wv_str], self.n_kv_heads, dim=0)

                # Select chunks for the current device
                wq_selected = torch.cat(wq_chunks[i * self.n_local_heads : (i + 1) * self.n_local_heads], dim=0)
                wk_selected = torch.cat(wk_chunks[i * self.n_local_kv_heads : (i + 1) * self.n_local_kv_heads], dim=0)
                wv_selected = torch.cat(wv_chunks[i * self.n_local_kv_heads : (i + 1) * self.n_local_kv_heads], dim=0)

                # Transpose the selected chunks
                wq = torch.transpose(wq_selected, -2, -1)
                wk = torch.transpose(wk_selected, -2, -1)
                wv = torch.transpose(wv_selected, -2, -1)

                # Create interleaved qkv list
                n_repeat = self.n_heads // self.n_kv_heads
                qkv_interleaved = [
                    [
                        wq[..., i * n_repeat * self.head_dim : (i + 1) * n_repeat * self.head_dim],
                        wk[..., i * self.head_dim : (i + 1) * self.head_dim],
                        wv[..., i * self.head_dim : (i + 1) * self.head_dim],
                    ]
                    for i in range(self.n_local_kv_heads)
                ]
                qkv_interleaved = [item for sublist in qkv_interleaved for item in sublist]

                # Concatenate Q, K, V for the current device
                qkv = torch.cat(qkv_interleaved, dim=-1)
                qkv_list.append(qkv)

            qkv_cat = torch.cat(qkv_list, dim=-1)
            qkv_cat = qkv_cat.unsqueeze(0).unsqueeze(0)

            pt_wo = self.state_dict[wo_str].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        qkv_ttnn = ttnn.as_tensor(
            qkv_cat,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=3),
            cache_file_name=self.cache_path / wqkv_cache_str,
        )
        self.qkv = ttnn.to_device(qkv_ttnn, self.mesh_device)

        wo_ttnn = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=3),
            cache_file_name=self.cache_path / wo_str,
        )

        self.wo = ttnn.to_device(wo_ttnn, self.mesh_device)

    def __call__(
        self,
        xs,
        rot_mats,
        start_pos: int,
        user_id: int = 0,
        cache_idxs=None,
        page_table=None,
        kv_cache=None,
        mode="decode",
    ):
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if mode == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, cache_idxs, page_table=page_table, kv_cache=kv_cache)
        # Prefill should have input tensor of shape (1, batch=1, seqlen, hidden_size)
        elif mode == "prefill":
            return self.prefill_forward(xs, rot_mats, user_id, page_table=page_table, kv_cache=kv_cache)
        else:
            raise ValueError(f"Unknown llm_mode: {mode}")

    def decode_forward(self, xs, rot_mats, start_pos: int, cache_idxs, page_table=None, kv_cache=None):
        query_layer, key_layer, value_layer = self.attn_qkv(xs, rot_mats)
        attn_outputs = self.attn_mqa(
            query_layer, key_layer, value_layer, start_pos, cache_idxs, page_table=page_table, kv_cache=kv_cache
        )
        return self.attn_selfout(attn_outputs)

    def attn_qkv(
        self,
        xs,
        rot_mats,
    ):
        # Fused QKV
        fused_query_key_value = ttnn.matmul(
            xs,
            self.qkv,
            program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
            memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        xs.deallocate(True)

        d = fused_query_key_value.get_legacy_shape()[-1]
        fused_query_key_value = ttnn.reshape(
            fused_query_key_value,
            ttnn.Shape((1, 1, self.max_batch_size, d), (1, 1, self.model_config["PADDED_BATCH_SIZE"], d)),
        )

        # Split QKV
        (
            query_layer,  # [seqlen, n_local_heads, bsz, head_dim]
            key_layer,  # [seqlen, n_local_kv_heads, bsz, head_dim]
            value_layer,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

        fused_query_key_value.deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        query_layer = ttnn.matmul(
            query_layer,
            rot_mats,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
            # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_heads, bsz, head_dim]
        )

        key_layer = ttnn.matmul(
            key_layer,
            rot_mats,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )

        return query_layer, key_layer, value_layer

    def attn_mqa(self, query_layer, key_layer, value_layer, start_pos: int, cache_idxs, page_table=None, kv_cache=None):
        # K CACHE UPDATE
        if kv_cache:
            keys = kv_cache[self.layer_num][0]
            values = kv_cache[self.layer_num][1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]
        # ttnn.update_cache(keys, key_layer, start_pos, batch_offset=batch_offset)
        ttnn.experimental.paged_update_cache(keys, key_layer, update_idxs_tensor=cache_idxs, page_table=page_table)

        key_layer.deallocate(True)

        # V CACHE UPDATE
        # ttnn.update_cache(values, value_layer, start_pos, batch_offset=batch_offset)
        ttnn.experimental.paged_update_cache(values, value_layer, update_idxs_tensor=cache_idxs, page_table=page_table)

        value_layer.deallocate(True)

        if page_table:
            attn_output = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                query_layer,
                keys,
                values,
                cur_pos_tensor=cache_idxs,
                page_table_tensor=page_table,
                scale=self.scale,
                program_config=self.model_config["SDPA_DECODE_PROGRAM_CONFIG"],
                compute_kernel_config=self.model_config["SDPA_COMPUTE_KERNEL_CONFIG"],
                memory_config=self.model_config["SDPA_OUTPUT_MEMCFG"],
            )

        else:
            # Have to reshape back since sdpa expects batch in dim 1
            keys_reshaped = ttnn.reshape(keys, [self.n_local_kv_heads, self.max_batch_size, -1, self.head_dim])
            values_reshaped = ttnn.reshape(values, [self.n_local_kv_heads, self.max_batch_size, -1, self.head_dim])
            attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
                query_layer,
                keys_reshaped,
                values_reshaped,
                # [start_pos for _ in range(self.max_batch_size)],
                cur_pos_tensor=cache_idxs,
                scale=self.scale,
                program_config=self.model_config["SDPA_DECODE_PROGRAM_CONFIG"],
                compute_kernel_config=self.model_config["SDPA_COMPUTE_KERNEL_CONFIG"],
                memory_config=self.model_config["SDPA_OUTPUT_MEMCFG"],
            )
        return attn_output

    def attn_selfout(
        self,
        attn_output,
    ):
        # ATTENTION SELFOUT
        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.n_local_heads,
        )  # seqlen, 1, batch, hidden_size

        attn_output = ttnn.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        attn_output = ttnn.matmul(
            attn_output,
            self.wo,
            program_config=self.model_config["SELFOUT_MM_PROGCFG"],
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )  # seqlen, 1, batch, hidden_size

        return attn_output

    def prefill_forward(self, xs, rot_mats, user_id: int = 0, page_table=None, kv_cache=None):
        query_layer, key_layer, value_layer = self.prefill_attn_qkv(xs, rot_mats)
        attn_outputs = self.prefill_attn_mqa(
            query_layer, key_layer, value_layer, user_id, page_table=page_table, kv_cache=kv_cache
        )
        return self.prefill_attn_selfout(attn_outputs)

    def prefill_attn_qkv(
        self,
        xs,
        rot_mats,
    ):
        assert xs.shape[1] == 1, "batch must be 1"
        _, _, seq_len, _ = xs.shape
        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
        if seq_len >= max_mm_seq_len:
            if seq_len % max_mm_seq_len != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {max_mm_seq_len}")
            batch_dim = seq_len // max_mm_seq_len  # Find the division factor
            xs = ttnn.reshape(xs, (1, batch_dim, seq_len // batch_dim, -1))
            pc_qkv = self.model_config["PREFILL_FUSED_QKV_MM_PROGCFG"]
        elif seq_len == 128:
            pc_qkv = self.model_config["PREFILL_FUSED_QKV_MM_PROGCFG_128"]
        else:
            # Use default program configs
            pc_qkv = None

        fused_query_key_value = ttnn.linear(
            xs,
            self.qkv,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_qkv else None,
            dtype=ttnn.bfloat16,
            program_config=pc_qkv,
        )

        if seq_len >= max_mm_seq_len:
            fused_query_key_value = ttnn.reshape(fused_query_key_value, (1, 1, seq_len, -1))

        xs.deallocate(True)

        (
            query_layer,  # [bsz, n_local_heads, seq_len, head_dim]
            key_layer,  # [bsz, n_local_kv_heads, seq_len, head_dim]
            value_layer,  # [bsz, n_local_kv_heads, seq_len, head_dim]
        ) = ttnn.experimental.nlp_create_qkv_heads(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        fused_query_key_value.deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        # query_layer: ttnn.Shape([1, 8, seq_len, 128]) -> [bsz, n_local_heads, seq_len, head_dim]
        query_layer_ret = ttnn.experimental.rotary_embedding_llama(
            query_layer, rot_mats[0], rot_mats[1], self.transformation_mats
        )
        query_layer.deallocate(True)

        # K Rotary Embeddings
        # key_layer: ttnn.Shape([1, 1, seq_len, 128]) -> [bsz, n_local_kv_heads, seq_len, head_dim]
        key_layer_ret = ttnn.experimental.rotary_embedding_llama(
            key_layer, rot_mats[0], rot_mats[1], self.transformation_mats
        )
        key_layer.deallocate(True)

        return query_layer_ret, key_layer_ret, value_layer

    def prefill_attn_mqa(self, query_layer, key_layer, value_layer, user_id: int = 0, page_table=None, kv_cache=None):
        if kv_cache:
            keys = kv_cache[self.layer_num][0]
            values = kv_cache[self.layer_num][1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]

        if page_table:
            ttnn.experimental.paged_fill_cache(
                keys, ttnn.experimental.typecast(key_layer, self.kv_dtype), page_table, batch_idx=user_id
            )
            ttnn.experimental.paged_fill_cache(
                values, ttnn.experimental.typecast(value_layer, self.kv_dtype), page_table, batch_idx=user_id
            )
        else:
            ttnn.fill_cache(keys, ttnn.experimental.typecast(key_layer, self.kv_dtype), user_id)
            ttnn.fill_cache(values, ttnn.experimental.typecast(value_layer, self.kv_dtype), user_id)

        seq_len = query_layer.shape[2]
        q_chunk_size = 128 if seq_len % 128 == 0 else 32
        k_chunk_size = q_chunk_size

        pc_sdpa = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=[8, 7],
            q_chunk_size=q_chunk_size,
            k_chunk_size=k_chunk_size,
        )
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            is_causal=True,
            scale=self.scale,
            program_config=pc_sdpa,
        )

        # deallocate keys and values
        query_layer.deallocate(True)
        key_layer.deallocate(True)
        value_layer.deallocate(True)

        return attn_output

    def prefill_attn_selfout(self, attn_output):
        # ATTENTION SELFOUT
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # seqlen, 1, batch, hidden_size

        attn_output = ttnn.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        _, _, seq_len, _ = attn_output.shape
        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
        if seq_len >= max_mm_seq_len:
            if seq_len % max_mm_seq_len != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {max_mm_seq_len}")
            batch_dim = seq_len // max_mm_seq_len  # Find the division factor
            attn_output = ttnn.reshape(attn_output, (1, batch_dim, seq_len // batch_dim, -1))
            pc_dense_out = self.model_config["PREFILL_SELFOUT_MM_PROGCFG"]
        elif seq_len == 128:
            pc_dense_out = self.model_config["PREFILL_SELFOUT_MM_PROGCFG_128"]
        else:
            # Use default program configs
            pc_dense_out = None

        attn_output = ttnn.linear(
            attn_output,
            self.wo,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
            core_grid=ttnn.CoreGrid(y=8, x=8) if not pc_dense_out else None,
            dtype=ttnn.bfloat16,
            program_config=pc_dense_out,
        )

        attn_output = ttnn.reshape(attn_output, (1, 1, seq_len, -1))

        return attn_output

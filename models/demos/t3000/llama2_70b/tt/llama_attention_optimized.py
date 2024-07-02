# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import math
import torch
from ttnn import experimental as tt_lib
import ttnn
from ttnn import ShardTensorToMesh
from models.utility_functions import nearest_32
from models.demos.t3000.llama2_70b.tt.llama_common import (
    MAX_SEQ_LEN,
    MAX_SEQ_LEN_LLAMA3,
)


class TtLlamaAttention_optimized:
    def __init__(
        self,
        device_mesh,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        transformation_mats,
        cache_path=None,
        batch_size=None,
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        self.model_config = model_config
        self.read_cache = read_cache

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

        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path
        self.transformation_mats = transformation_mats

        self.load_weights()
        self.init_kv_cache()

    def set_model_config(self, model_config):
        self.model_config = model_config

    def init_kv_cache(self):
        """
        Generates empty KV cache and pushed to device memory
        """

        cache_k = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                # self.max_seq_len,
                self.model_config["MAX_CONTEXT_LEN"],
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                # self.max_seq_len,
                self.model_config["MAX_CONTEXT_LEN"],
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.to_device(
                ttnn.as_tensor(
                    lp,
                    device=self.device_mesh,
                    mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=self.model_config["DRAM_MEMCFG"],
                    dtype=ttnn.bfloat8_b,
                    cache_file_name=self.cache_path / f"empty_attn_cache{cache_k.shape}",
                ),
                self.device_mesh,
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
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=self.cache_path / wqkv_cache_str,
        )
        self.qkv = ttnn.to_device(qkv_ttnn, self.device_mesh)

        wo_ttnn = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=self.model_config["DRAM_MEMCFG"],
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=3),
            cache_file_name=self.cache_path / wo_str,
        )

        self.wo = ttnn.to_device(wo_ttnn, self.device_mesh)

    def __call__(
        self,
        xs,
        rot_mats,
        start_pos: int,
        attn_masks,
        user_id: int = 0,
    ):
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        # Prefill should have input tensor of shape (1, batch=1, seqlen, hidden_size)
        elif self.model_config["LLM_MODE"] == "prefill":
            return self.prefill_forward(xs, rot_mats, attn_masks, user_id)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def decode_forward(
        self,
        xs,
        rot_mats,
        start_pos: int,
        attn_masks,
    ):
        query_layer, key_layer, value_layer = self.attn_qkv(xs, rot_mats)
        attn_outputs = self.attn_mqa(query_layer, key_layer, value_layer, start_pos, attn_masks)
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

        # Split QKV
        (
            query_layer,  # [seqlen, n_local_heads, bsz, head_dim]
            key_layer,  # [seqlen, n_local_kv_heads, bsz, head_dim]
            value_layer,  # [seqlen, n_local_kv_heads, bsz, head_dim]
        ) = tt_lib.tensor.nlp_create_qkv_heads_decode(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )

        fused_query_key_value.deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        query_layer = ttnn.matmul(
            query_layer,
            rot_mats,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
            # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_heads, bsz, head_dim]
        )

        key_layer = ttnn.matmul(
            key_layer,
            rot_mats,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )

        return query_layer, key_layer, value_layer

    def attn_mqa(
        self,
        query_layer,
        key_layer,
        value_layer,
        start_pos: int,
        attn_masks,
        batch_offset: int = 0,
    ):
        padded_layer_past_len = nearest_32(start_pos + 1)

        # K Cache Update
        kv_cache_memcfg = self.model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"]
        if kv_cache_memcfg.is_sharded():
            kv_cache_shard_shape = kv_cache_memcfg.shard_spec.shape
            kv_cache_shard_shape[0] = self.layer_past[0].shape[0] * padded_layer_past_len
            kv_cache_memcfg.shard_spec.shape = kv_cache_shard_shape

        keys = self.layer_past[0]
        tt_lib.tensor.update_cache(keys, key_layer, start_pos, batch_offset=batch_offset)
        key_layer.deallocate(True)

        # key and value layers will have kv_seq_len padded to nearest 32
        keys = self.layer_past[0]
        key_layer = tt_lib.tensor.nlp_kv_cache_load_slice(keys, 0, padded_layer_past_len)

        # PRE-SOFTMAX MM
        key_layer_transposed = tt_lib.tensor.transpose(
            key_layer,
            -2,
            -1,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )

        key_layer.deallocate(True)

        attn_prog_config = self.model_config["ATTN_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)
        attn_output_memcfg = self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"]
        attn_output_memcfg.shard_spec.shape[1] = padded_layer_past_len

        attn_weights = ttnn.matmul(
            query_layer,
            key_layer_transposed,
            program_config=attn_prog_config,
            memory_config=attn_output_memcfg,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )

        query_layer.deallocate(True)
        key_layer_transposed.deallocate(True)

        # SOFTMAX
        softmax_progcfg = self.model_config["BATCHED_SOFTMAX_PROGCFG"]
        softmax_progcfg.block_w = padded_layer_past_len // 32

        attn_weights = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
            attn_weights,
            self.scale,
            attn_masks,
            program_config=self.model_config["BATCHED_SOFTMAX_PROGCFG"],
            is_causal_mask=True,
        )

        # V CACHE UPDATE
        values = self.layer_past[1]
        tt_lib.tensor.update_cache(values, value_layer, start_pos, batch_offset=batch_offset)
        value_layer.deallocate(True)

        values = self.layer_past[1]
        value_layer = tt_lib.tensor.nlp_kv_cache_load_slice(values, 0, padded_layer_past_len)

        # POST-SOFTMAX MM
        scores_prog_config = self.model_config["SCORES_BATCHED_MM_PROGCFG_LAMBDA"](padded_layer_past_len // 32)

        attn_output = ttnn.matmul(
            attn_weights,
            value_layer,
            program_config=scores_prog_config,
            memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        attn_weights.deallocate(True)
        value_layer.deallocate(True)

        return attn_output

    def attn_selfout(
        self,
        attn_output,
    ):
        # ATTENTION SELFOUT
        attn_output = tt_lib.tensor.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.n_local_heads,
        )  # seqlen, 1, batch, hidden_size

        attn_output = ttnn.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            # memory_config=self.model_config["L1_MEMCFG"],
            memory_config=self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"],
        )

        attn_output = ttnn.matmul(
            attn_output,
            self.wo,
            program_config=self.model_config["SELFOUT_MM_PROGCFG"],
            memory_config=self.model_config["WIDTH_SHARDED_MEMCFG"],
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )  # seqlen, 1, batch, hidden_size

        return attn_output

    def prefill_forward(
        self,
        xs,
        rot_mats,
        attn_masks,
        user_id: int = 0,
    ):
        query_layer, key_layer, value_layer = self.prefill_attn_qkv(xs, rot_mats)
        attn_outputs = self.prefill_attn_mqa(query_layer, key_layer, value_layer, attn_masks, user_id)
        return self.prefill_attn_selfout(attn_outputs)

    def prefill_attn_qkv(
        self,
        xs,
        rot_mats,
    ):
        assert xs.shape[1] == 1, "batch must be 1"
        assert xs.shape[2] % 128 == 0 and xs.shape[2] > 0, "Seqlen must be divisible by 128"
        _, _, seq_len, _ = xs.shape

        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
        batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor

        xs = ttnn.reshape(xs, (1, batch_dim, seq_len // batch_dim, -1))

        # Fused QKV
        fused_query_key_value = ttnn.matmul(
            xs,
            self.qkv,
            program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
            memory_config=self.model_config["DRAM_MEMCFG"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )
        fused_query_key_value = ttnn.reshape(fused_query_key_value, (1, 1, seq_len, -1))

        xs.deallocate(True)

        (
            query_layer,  # [bsz, n_local_heads, seq_len, head_dim]
            key_layer,  # [bsz, n_local_kv_heads, seq_len, head_dim]
            value_layer,  # [bsz, n_local_kv_heads, seq_len, head_dim]
        ) = tt_lib.tensor.nlp_create_qkv_heads(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            output_mem_config=self.model_config["DRAM_MEMCFG"],
        )

        fused_query_key_value.deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        # query_layer: ttnn.Shape([1, 8, seq_len, 128]) -> [bsz, n_local_heads, seq_len, head_dim]
        query_layer_ret = ttnn.experimental.tensor.rotary_embedding_llama(
            query_layer, rot_mats[0], rot_mats[1], self.transformation_mats
        )
        query_layer.deallocate(True)

        # K Rotary Embeddings
        # key_layer: ttnn.Shape([1, 1, seq_len, 128]) -> [bsz, n_local_kv_heads, seq_len, head_dim]
        key_layer_ret = ttnn.experimental.tensor.rotary_embedding_llama(
            key_layer, rot_mats[0], rot_mats[1], self.transformation_mats
        )
        key_layer.deallocate(True)

        return query_layer_ret, key_layer_ret, value_layer

    def prefill_attn_mqa(
        self,
        query_layer,
        key_layer,
        value_layer,
        attn_masks,
        user_id: int = 0,
    ):
        # FILL K CACHE
        keys = self.layer_past[0]
        # Fill cache expects batch in dim0
        keys_reshaped = ttnn.reshape(keys, [self.max_batch_size, self.n_local_kv_heads, -1, self.head_dim])
        tt_lib.tensor.fill_cache(keys_reshaped, tt_lib.tensor.typecast(key_layer, ttnn.bfloat8_b), user_id)

        # FILL V CACHE
        values = self.layer_past[1]
        # Fill cache expects batch in dim0
        values_reshaped = ttnn.reshape(values, [self.max_batch_size, self.n_local_kv_heads, -1, self.head_dim])
        tt_lib.tensor.fill_cache(values_reshaped, tt_lib.tensor.typecast(value_layer, ttnn.bfloat8_b), user_id)

        # SPDA
        attn_output = tt_lib.operations.primary.transformers.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_masks,
            is_causal=True,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"],
        )

        # deallocate keys and values
        query_layer.deallocate(True)
        key_layer.deallocate(True)
        value_layer.deallocate(True)

        return attn_output

    def prefill_attn_selfout(self, attn_output):
        # ATTENTION SELFOUT
        attn_output = tt_lib.tensor.nlp_concat_heads(
            attn_output,
            output_mem_config=self.model_config["L1_MEMCFG"],
        )  # seqlen, 1, batch, hidden_size

        attn_output = ttnn.all_gather(
            attn_output,
            dim=3,
            num_links=self.model_config["ALL_GATHER_NUM_LINKS"],
            memory_config=self.model_config["DRAM_MEMCFG"],
        )

        _, _, seq_len, _ = attn_output.shape

        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"]
        batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor
        attn_output = ttnn.reshape(attn_output, (1, batch_dim, seq_len // batch_dim, -1))

        attn_output = ttnn.matmul(
            attn_output,
            self.wo,
            program_config=self.model_config["SELFOUT_MM_PROGCFG"],
            memory_config=self.model_config["DRAM_MEMCFG"],
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"],
        )  # seqlen, 1, batch, hidden_size

        attn_output = ttnn.reshape(attn_output, (1, 1, seq_len, -1))

        return attn_output

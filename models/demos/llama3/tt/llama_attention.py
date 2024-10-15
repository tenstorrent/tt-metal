# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule


class TtLlamaAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        configuration,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = configuration.paged_attention_config

        self.n_local_heads = self.n_heads // configuration.num_devices
        self.n_local_kv_heads = self.n_kv_heads // configuration.num_devices

        self.dtype = dtype

        self.kv_seq_len = configuration.kv_seq_len
        self.sliding_window = configuration.sliding_window
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.model_config = configuration.get_model_config()

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
        assert self.n_heads % configuration.num_devices == 0
        assert self.n_kv_heads % configuration.num_devices == 0
        assert configuration.qkv_size % configuration.num_devices == 0
        assert configuration.dim % configuration.num_devices == 0

        # wqkv: 4096 x 3072 (2 devices): width-sharded on 12 banks, 3072 over 12 banks.
        wqkv_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim, configuration.qkv_size // configuration.num_devices
        )
        self.wqkv = ttnn.as_tensor(
            torch.concat(
                [
                    torch.concat(
                        [
                            torch.transpose(
                                torch.chunk(self.state_dict[wq_str], configuration.num_devices)[i],
                                -2,
                                -1,
                            ),
                            torch.transpose(
                                torch.chunk(self.state_dict[wk_str], configuration.num_devices)[i],
                                -2,
                                -1,
                            ),
                            torch.transpose(
                                torch.chunk(self.state_dict[wv_str], configuration.num_devices)[i],
                                -2,
                                -1,
                            ),
                        ],
                        dim=-1,
                    )
                    for i in range(configuration.num_devices)
                ],
                dim=-1,
            ),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=wqkv_mem_config,
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name("wqkv_sharded"),
        )

        # wo: 2048 (2devices) x 4096: width-sharded on 12 banks, 4224 over 12 banks.
        wo_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim // configuration.num_devices, configuration.dim
        )
        self.wo = ttnn.as_tensor(
            torch.transpose(
                self.state_dict[wo_str],
                -2,
                -1,
            ),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            memory_config=wo_mem_config,
            dtype=self.dtype,
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name("wo_sharded"),
        )

        if self.paged_attention_config:
            cache_k = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_kv_heads // configuration.num_devices,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_kv_heads // configuration.num_devices,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
        else:
            cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads,
                    self.sliding_window,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads,
                    self.sliding_window,
                    self.head_dim,
                )
            )

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                dtype=self.dtype,
                cache_file_name=f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                if weight_cache_path and not configuration.dummy_weights
                else None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for k_or_v in [cache_k, cache_v]
        ]

        self.scale = self.head_dim**-0.5

    def forward_decode(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mat=None,
        page_table=None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """
        assert self.max_batch_size * self.n_kv_heads < 64
        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        x_sharded = ttnn.interleaved_to_sharded(x, self.model_config["SHARDED_ATTN_INPUT_MEMCFG"])
        xqkv_fused_sharded = ttnn.linear(
            x_sharded,
            self.wqkv,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.model_config["XQKV_DECODE_PROGCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
        )
        ttnn.deallocate(x_sharded)

        xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(xqkv_fused_sharded)

        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, ttnn.Shape((1, 1, self.max_batch_size, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))
        )

        ###
        # Reshape and rotary embeddings
        ###
        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )

        ttnn.deallocate(xqkv_fused)

        q_heads_1BQD = ttnn.linear(
            q_heads_pre_rot_1BQD,
            rot_mat,
            program_config=self.model_config["ROT_MAT_BMM_PROGCFG"](
                q_heads_pre_rot_1BQD.shape[-2], q_heads_pre_rot_1BQD.shape[-1], rot_mat.shape[-1]
            ),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
        )

        k_heads_1BKD = ttnn.linear(
            k_heads_pre_rot_1BKD,
            rot_mat,
            program_config=self.model_config["ROT_MAT_BMM_PROGCFG"](
                k_heads_pre_rot_1BKD.shape[-2], k_heads_pre_rot_1BKD.shape[-1], rot_mat.shape[-1]
            ),
            memory_config=k_heads_pre_rot_1BKD.memory_config(),
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)

        ###
        # KV update
        ###
        keys = self.layer_past[0]
        values = self.layer_past[1]

        # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
        # v_heads [seqlen, n_kv_heads, bsz, head_dim]
        # keys, [max_batch_size, n_kv_heads // configuration.num_devices, sliding_window, head_dim]
        ttnn.experimental.paged_update_cache(keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table)
        ttnn.experimental.paged_update_cache(
            values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
        )
        self.layer_past[0] = keys
        self.layer_past[1] = values

        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        if page_table:
            attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                page_table_tensor=page_table,
                transpose_q=False,
                scale=self.scale,
                program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode_gqa(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                transpose_q=False,
                scale=self.scale,
                program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        ttnn.deallocate(q_heads_1BQD)

        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D, memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"]
        )
        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_11BH)
        ttnn.deallocate(attn_output_1G4D)

        dense_out_sharded = ttnn.linear(
            attn_output_cat,
            self.wo,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.model_config["ATTN_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
        )  # seqlen, 1, batch, hidden_size

        ttnn.deallocate(attn_output_cat)

        # All reduce
        if self.num_devices > 1:
            dense_out_gathered = ttnn.all_gather(
                dense_out_sharded,
                dim=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
                memory_config=self.model_config["SHARDED_ATTN_DECODE_WO_OUT_GATHERED_MEMCFG"],
            )
            # This sharded to interleaved call is needed as fast_reduce only supports interleaved inputs
            dense_out_gathered = ttnn.sharded_to_interleaved(dense_out_gathered, ttnn.L1_MEMORY_CONFIG)
            dense_out_reduced = ttnn.experimental.fast_reduce_nc(
                dense_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            ttnn.deallocate(dense_out_sharded)
            ttnn.deallocate(dense_out_gathered)
            return dense_out_reduced
        else:
            dense_out = ttnn.sharded_to_interleaved(dense_out_sharded, ttnn.L1_MEMORY_CONFIG)
            ttnn.deallocate(dense_out_sharded)
            return dense_out

    def forward_prefill(self, x_11SH, rot_mats, transformation_mats, user_id: int = 0, page_table=None):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 2048, 2048, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )

        if seq_len > 2048:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

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

        ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        # Fill KV-Cache
        keys_BKSD, values_BKSD = self.layer_past[0], self.layer_past[1]

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(k_heads_1KSD)
        # sharding k_fill to deal with update_cache memory limitation
        if seq_len > 256:
            k_fill = ttnn.interleaved_to_sharded(k_heads_1KSD_8b, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        else:
            k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)

        ttnn.deallocate(v_heads_1VSD)
        # sharding v_fill to deal with update_cache memory limitation
        if seq_len > 256:
            v_fill = ttnn.interleaved_to_sharded(v_heads_1VSD_8b, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        else:
            v_fill = v_heads_1VSD_8b

        if page_table:
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill, page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill, page_table, batch_idx=user_id)
        else:
            ttnn.fill_cache(
                keys_BKSD,
                k_fill,
                user_id,
            )
            ttnn.fill_cache(
                values_BKSD,
                v_fill,
                user_id,
            )

        if seq_len > 256:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        self.layer_past = [keys_BKSD, values_BKSD]

        # SDPA

        # reshaping to put group in batch dim to do sdpa on 8 MQAs in parallel
        k_heads_K1SD_8b = ttnn.reshape(k_heads_1KSD_8b, [self.n_local_kv_heads, 1, -1, self.head_dim])
        v_heads_V1SD_8b = ttnn.reshape(v_heads_1VSD_8b, [self.n_local_kv_heads, 1, -1, self.head_dim])

        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        q_heads_84SD_8b = ttnn.reshape(
            q_heads_1QSD_8b, [self.n_local_kv_heads, self.n_local_heads // self.n_local_kv_heads, -1, self.head_dim]
        )

        attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_84SD_8b,
            k_heads_K1SD_8b,
            v_heads_V1SD_8b,
            is_causal=True,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
        )

        # deallocate keys and values
        ttnn.deallocate(q_heads_84SD_8b)
        ttnn.deallocate(k_heads_K1SD_8b)
        ttnn.deallocate(v_heads_V1SD_8b)

        attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 2048, 2048, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )
        if seq_len > 2048:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # All reduce
        if self.num_devices > 1:
            dense_out_gathered = ttnn.all_gather(output_11SH, dim=1, num_links=1, topology=ttnn.Topology.Linear)
            dense_out_reduced = ttnn.experimental.fast_reduce_nc(
                dense_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            ttnn.deallocate(output_11SH)
            ttnn.deallocate(dense_out_gathered)
            return dense_out_reduced
        else:
            return output_11SH

    def forward(
        self, x, current_pos, rot_mats=None, transformation_mats=None, user_id=0, mode="decode", page_table=None
    ):
        if mode == "prefill":
            return self.forward_prefill(x, rot_mats, transformation_mats, user_id, page_table)
        else:
            return self.forward_decode(x, current_pos, rot_mats, page_table)

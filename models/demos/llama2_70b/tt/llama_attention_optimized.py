# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from loguru import logger

import math
import tt_lib
from models.utility_functions import torch2tt_tensor, tt2torch_tensor, nearest_32
from models.demos.llama2_70b.tt.llama_common import (
    tt_all_gather,
    tt_all_gather_torch,
    precompute_freqs as tt_precompute_freqs,
    freqs_to_rotation_matrix,
    gather_rotary_emb as tt_gather_rotary_emb,
    tt_all_gather,
)


class TtLlamaAttention_optimized(torch.nn.Module):
    def __init__(self, devices, state_dict, base_url, layer_num, model_config, configuration):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
        self.model_config = model_config

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.scale = 1 / math.sqrt(self.head_dim)

        assert self.num_devices == 4 or self.num_devices == 8
        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        layer_name = f"{base_url}.{layer_num}"

        wq_str = f"{layer_name}.attention.wq.weight"
        wk_str = f"{layer_name}.attention.wk.weight"
        wv_str = f"{layer_name}.attention.wv.weight"
        wo_str = f"{layer_name}.attention.wo.weight"

        self.qkv_list = []
        self.wo_list = []
        self.layer_past_list = []

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

            # Append the processed tensor to the list, assuming torch2tt_tensor is a defined method
            self.qkv_list.append(
                torch2tt_tensor(
                    qkv,
                    self.devices[i],
                    tt_memory_config=self.model_config["FUSED_QKV_MM_WEIGHTS_MEMCFG"],
                    tt_dtype=self.model_config["FUSED_QKV_MM_WEIGHTS_DTYPE"],
                ),
            )

            ### WO Weights and KV Cache
            wo = torch2tt_tensor(
                torch.transpose(
                    torch.chunk(self.state_dict[wo_str], self.num_devices)[i],
                    -2,
                    -1,
                ),
                self.devices[i],
            )
            cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    self.max_seq_len,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    self.max_seq_len,
                    self.head_dim,
                )
            )
            layer_past = [cache_k, cache_v]
            layer_past = [torch2tt_tensor(lp, self.devices[i]) for lp in layer_past]

            # add to the list
            self.wo_list.append(wo)
            self.layer_past_list.append(layer_past)

    def get_rotation_mat(self, dhead, end, start_pos, seqlen, batch):
        cos, sin = tt_precompute_freqs(dhead, end)
        rot_mat = freqs_to_rotation_matrix(cos, sin)
        position_ids = torch.ones(seqlen, batch, dtype=torch.long) * start_pos
        rot_emb = tt_gather_rotary_emb(rot_mat, position_ids)
        return rot_emb

    def prepare_inputs(self, x, start_pos):
        """
        Prepare inputs for decode mode. Assume that current token is at
        start_pos, and KV cache has valid data up to start_pos.
        x: (batch, seq, hidden_dim)
        start_pos: int
        """
        assert x.size(2) == self.hidden_size
        assert len(x.size()) == 3

        batch = x.size(0)
        seq_len = x.size(1)
        assert seq_len == 1, "Only supporting decode mode"
        x = x.transpose(0, 1).unsqueeze(1)  # [seq_len, 1, batch, hidden_dim]
        rot_mat = self.get_rotation_mat(
            dhead=self.head_dim, end=self.max_seq_len * 2, start_pos=start_pos, seqlen=seq_len, batch=batch
        )
        rot_mat = rot_mat[:, :1]

        padded_layer_past_len = nearest_32(start_pos + 1)
        attn_mask = torch.zeros(seq_len, 1, batch, padded_layer_past_len)
        # attn_mask[:, :, :, : start_pos + 1] = -1e9
        attn_mask[:, :, :, start_pos + 1 :] = torch.finfo(attn_mask.dtype).min
        attn_mask = attn_mask.expand(-1, self.n_local_heads, -1, -1)

        # expected shapes:
        # x: (seq_len, 1, batch, hidden_dim)
        # start_pos: int
        # rot_mat: [1, bsz, head_dim, head_dim]
        # attn_mask: [seq_len, n_heads, batch, padded_layer_past_len]
        assert x.size() == (seq_len, 1, batch, self.hidden_size)
        assert rot_mat.size() == (1, 1, self.head_dim, self.head_dim)
        assert attn_mask.size() == (seq_len, self.n_local_heads, batch, padded_layer_past_len)
        xs, rot_mats, attn_masks = [], [], []
        for i in range(self.num_devices):
            device = self.devices[i]
            xs.append(torch2tt_tensor(x.clone(), device))
            rot_mats.append(torch2tt_tensor(rot_mat.clone(), device))

            # Put attn_mask on the device with the sharded config
            attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
            if attention_mask_memconfig.is_sharded():
                attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
                attn_mask_shard_shape[-1] = padded_layer_past_len
                attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
            attn_masks.append(
                torch2tt_tensor(
                    attn_mask.clone(),
                    device,
                    tt_memory_config=attention_mask_memconfig,
                    tt_dtype=self.model_config["ATTN_MASK_DTYPE"],
                )
            )
        return (
            xs,
            start_pos,
            rot_mats,
            attn_masks,
        )

    def forward(
        self,
        xs: tt_lib.tensor.Tensor,
        rot_mats: tt_lib.tensor.Tensor,
        start_pos: int,
        attn_masks: tt_lib.tensor.Tensor,
    ) -> tt_lib.tensor.Tensor:
        padded_layer_past_len = nearest_32(start_pos + 1)

        # TODO: Move sharding inputs to model
        for i, device in enumerate(self.devices):
            xs[i] = xs[i].to(device, self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"])
        # Reshard
        # if self.model_config["LN_ATTN_OUTPUT_MEMCFG"] != self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"]:
        #     for i in range(len(xs)):
        #         xs[i] = tt_lib.tensor.sharded_to_interleaved(
        #             xs[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
        #         )
        #     for i in range(len(xs)):
        #         xs[i] = tt_lib.tensor.interleaved_to_sharded(
        #             xs[i], sharded_mem_config=self.model_config["FUSED_QKV_MM_INPUT_MEMCFG"]
        #         )

        # Fused QKV
        fused_query_key_value = []
        for i in range(len(xs)):
            fused_query_key_value.append(
                tt_lib.operations.primary.matmul_1d(
                    xs[i],
                    self.qkv_list[i],
                    program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
                    output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["FUSED_QKV_MM_OUTPUT_DTYPE"],
                    # compute_kernel_config=self.model_config["COMPUTE_KERNEL_CONFIG"], #TODO: Need to pad & change PROCFG
                )
            )
            xs[i].deallocate(True)
        # TMs
        if self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"] != self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]:
            for i in range(len(fused_query_key_value)):
                fused_query_key_value[i] = tt_lib.tensor.sharded_to_interleaved(
                    fused_query_key_value[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
                )
            for i in range(len(fused_query_key_value)):
                fused_query_key_value[i] = tt_lib.tensor.interleaved_to_sharded(
                    fused_query_key_value[i], sharded_mem_config=self.model_config["CREATE_QKV_HEADS_INPUT_MEMCFG"]
                )
        query_layer = []
        key_layer = []
        value_layer = []
        for i in range(len(fused_query_key_value)):
            rot_mat = rot_mats[i]
            (
                q_heads,  # [seqlen, n_local_heads, bsz, head_dim]
                k_heads,  # [seqlen, n_local_kv_heads, bsz, head_dim]
                v_heads,  # [seqlen, n_local_kv_heads, bsz, head_dim]
            ) = tt_lib.tensor.nlp_create_qkv_heads(
                fused_query_key_value[i],
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                output_mem_config=self.model_config["CREATE_QKV_HEADS_OUTPUT_MEMCFG"],
            )

            # rot_mat = tt_lib.tensor.interleaved_to_sharded(
            #     rot_mat, sharded_mem_config=self.model_config["ROT_MAT_K_OUTPUT_MEMCFG"]
            # )
            # k_heads = tt_lib.matmul(
            #     k_heads,
            #     rot_mat,
            #     # program_config=self.model_config["ROT_MAT_K_MM_PROGCFG"],
            #     output_mem_config=self.model_config["ROT_MAT_K_MM_OUTPUT_MEMCFG"],
            #     # [seqlen, n_kv_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_kv_heads, bsz, head_dim]
            # )
            # k_heads = tt_lib.tensor.sharded_to_interleaved(
            #     k_heads, output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            # )

            # TODO: matmul_1d is bugged for single core and causes hang.
            k_heads = tt_lib.tensor.sharded_to_interleaved(
                k_heads, output_mem_config=self.model_config["L1_K_HEADS_INTERLEAVED_MEMCFG"]
            )
            k_heads = tt_lib.operations.primary.matmul(
                k_heads,
                rot_mat,
                # [seqlen, n_kv_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_kv_heads, bsz, head_dim]
            )

            # ROTARY EMBEDDINGS
            q_heads = tt_lib.operations.primary.matmul_1d(
                q_heads,
                rot_mat,
                program_config=self.model_config["ROT_MAT_Q_MM_PROGCFG"],
                output_mem_config=self.model_config["ROT_MAT_Q_MM_OUTPUT_MEMCFG"],
                # [seqlen, n_heads, bsz, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, n_kv_heads, bsz, head_dim]
            )

            query_layer.append(q_heads)
            key_layer.append(k_heads)
            value_layer.append(v_heads)
            fused_query_key_value[i].deallocate(True)

        # K Cache Update
        kv_cache_memcfg = self.model_config["KV_CACHE_SLICE_OUTPUT_MEMCFG"]
        if kv_cache_memcfg.is_sharded():
            kv_cache_shard_shape = kv_cache_memcfg.shard_spec.shape
            kv_cache_shard_shape[0] = self.layer_past_list[0][0].shape()[1] * padded_layer_past_len
            kv_cache_memcfg.shard_spec.shape = kv_cache_shard_shape
        for i in range(len(key_layer)):
            k_heads = key_layer[i]
            keys = self.layer_past_list[i][0]
            tt_lib.tensor.update_cache(keys, k_heads, start_pos)
            key_layer[i].deallocate(True)

        # key and value layers will have kv_seq_len padded to nearest 32
        for i in range(len(key_layer)):
            keys = self.layer_past_list[i][0]
            key_layer[i] = tt_lib.tensor.unpad(
                keys,
                [0, 0, 0, 0],
                [
                    self.max_batch_size - 1,
                    self.n_local_kv_heads - 1,
                    padded_layer_past_len - 1,
                    self.head_dim - 1,
                ],
                output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            )
        for i in range(len(key_layer)):
            key_layer[i] = tt_lib.tensor.interleaved_to_sharded(key_layer[i], sharded_mem_config=kv_cache_memcfg)
        key_layer_transposed = []
        for i in range(len(key_layer)):
            key_layer_transposed.append(
                tt_lib.tensor.transpose(
                    key_layer[i],
                    -2,
                    -1,
                    output_mem_config=self.model_config["K_TRANSPOSED_OUTPUT_MEMCFG"],
                )
            )
            key_layer[i].deallocate(True)

        attn_weights = []
        for i in range(len(query_layer)):
            # Put query head back to L1 height sharded, currently causes PCC issues with group attention matmul
            # query_layer[i] = tt_lib.tensor.interleaved_to_sharded(
            #     query_layer[i], sharded_mem_config=self.model_config["Q_ROTARY_EMB_OUTPUT_MEMCFG"]
            # )
            attn_weights.append(
                tt_lib.operations.primary.transformers.group_attn_matmul(
                    query_layer[i],
                    key_layer_transposed[i],
                    compute_with_storage_grid_size=self.devices[i].compute_with_storage_grid_size(),
                    output_mem_config=self.model_config["PRE_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["PRE_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
            )
            query_layer[i].deallocate(True)
            key_layer_transposed[i].deallocate(True)
        ##Softmax
        softmax_progcfg = self.model_config["SOFTMAX_PROGCFG"]
        softmax_progcfg.block_w = padded_layer_past_len // 32
        # # TODO:Also move to prepare inputs
        # attention_mask_memconfig = self.model_config["ATTN_MASK_MEMCFG"]
        # if attention_mask_memconfig.is_sharded():
        #     attn_mask_shard_shape = attention_mask_memconfig.shard_spec.shape
        #     attn_mask_shard_shape[-1] = padded_layer_past_len
        #     attention_mask_memconfig.shard_spec.shape = attn_mask_shard_shape
        for i in range(len(attn_weights)):
            attn_mask = attn_masks[i]
            # TODO: Put mask on L1 in prepare inputs
            # attn_mask = tt_lib.tensor.interleaved_to_sharded(attn_mask, sharded_mem_config=attention_mask_memconfig)
            attn_weights[i] = tt_lib.operations.primary.transformers.scale_mask_softmax_in_place(
                attn_weights[i],
                self.scale,
                attn_mask,
                program_config=self.model_config["SOFTMAX_PROGCFG"],
                is_causal_mask=True,
            )
            attn_mask.deallocate(True)
        # V CACHE UPDATE
        for i in range(len(value_layer)):
            v_heads = value_layer[i]
            values = self.layer_past_list[i][1]
            tt_lib.tensor.update_cache(values, v_heads, start_pos)
            value_layer[i].deallocate(True)
        for i in range(len(value_layer)):
            values = self.layer_past_list[i][1]
            value_layer[i] = tt_lib.tensor.unpad(
                values,
                [0, 0, 0, 0],
                [
                    self.max_batch_size - 1,
                    self.n_local_kv_heads - 1,
                    padded_layer_past_len - 1,
                    self.head_dim - 1,
                ],
                output_mem_config=self.model_config["DEFAULT_MEMCFG"],
            )
        for i in range(len(value_layer)):
            value_layer[i] = tt_lib.tensor.interleaved_to_sharded(value_layer[i], sharded_mem_config=kv_cache_memcfg)

        # POST-SOFTMAX MM
        attn_output = []
        for i in range(len(attn_weights)):
            attn_output.append(
                tt_lib.operations.primary.transformers.group_attn_matmul(
                    attn_weights[i],
                    value_layer[i],
                    compute_with_storage_grid_size=self.devices[i].compute_with_storage_grid_size(),
                    output_mem_config=self.model_config["POST_SOFTMAX_MM_OUTPUT_MEMCFG"],
                    output_dtype=self.model_config["POST_SOFTMAX_MM_OUTPUT_DTYPE"],  # Must be BFLOAT16
                )
            )
            attn_weights[i].deallocate(True)
            value_layer[i].deallocate(True)

        for i in range(len(attn_output)):
            attn_output[i] = tt_lib.tensor.nlp_concat_heads(
                attn_output[i],
                output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
            )  # seqlen, 1, batch, hidden_size

            attn_output[i] = tt_lib.tensor.sharded_to_interleaved(
                attn_output[i], output_mem_config=self.model_config["DEFAULT_MEMCFG"]
            )
        # All gather input to dense
        # dense_output_replicated = tt_all_gather_torch(attn_output, dim=-1)
        breakpoint()
        dense_output_replicated = tt_lib.tensor.all_gather(
            attn_output,
            dim=3,
            num_links=1,
            output_mem_config=self.model_config["DEFAULT_MEMCFG"],
        )
        breakpoint()

        for i in range(len(dense_output_replicated)):
            dense_output_replicated[i] = tt_lib.tensor.interleaved_to_sharded(
                dense_output_replicated[i], sharded_mem_config=self.model_config["ATTN_ALL_GATHER_OUTPUT_MEMCFG"]
            )
        dense_outs = []
        breakpoint()
        for i in range(len(dense_output_replicated)):
            dense_out = tt_lib.operations.primary.matmul_1d(
                dense_output_replicated[i],
                self.wo_list[i],
                program_config=self.model_config["SELFOUT_MM_PROGCFG"],
                output_mem_config=self.model_config["SELFOUT_MM_OUTPUT_MEMCFG"],
                output_dtype=self.model_config["SELFOUT_MM_OUTPUT_DTYPE"],
            )  # seqlen, 1, batch, hidden_size
            dense_output_replicated[i].deallocate(True)
            dense_outs.append(dense_out)
        breakpoint()
        # FOR BRINGUP! Outputs are sharded. Interleave them
        dense_outs = [
            tt_lib.tensor.sharded_to_interleaved(t, output_mem_config=self.model_config["DEFAULT_MEMCFG"])
            for t in dense_outs
        ]

        return dense_outs

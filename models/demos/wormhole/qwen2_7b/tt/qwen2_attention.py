# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch
from torch import nn

import ttnn
from models.utility_functions import (
    nearest_32,
)


class TtQwen2Attention(nn.Module):
    def __init__(
        self,
        devices,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        configuration,
        rot_mat,
        start_pos,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.start_pos = start_pos

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        self.kv_seq_len = configuration.kv_seq_len
        self.sliding_window = configuration.sliding_window
        self.grid_size = configuration.max_grid_size

        self.model_config = configuration.get_model_config()
        self.compute_kernel_config = configuration.get_compute_kernel_config()

        self.rot_mat = rot_mat  # Rotational matrix in the form of a list of 8K tensors [1,1,head_dim,head_dim] for positional embedding on device

        layer_name = f"model.layers.{layer_num}.self_attn"
        cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.q_proj.weight"
        wk_str = f"{layer_name}.k_proj.weight"
        wv_str = f"{layer_name}.v_proj.weight"
        wo_str = f"{layer_name}.o_proj.weight"

        bias_q_str = f"{layer_name}.q_proj.bias"
        bias_k_str = f"{layer_name}.k_proj.bias"
        bias_v_str = f"{layer_name}.v_proj.bias"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.wqkv_list = []
        self.bias_qkv_list = []
        self.wo_list = []
        self.layer_past_list = []

        for i in range(self.num_devices):
            wqkv = ttnn.as_tensor(
                torch.concat(
                    [
                        torch.transpose(
                            torch.chunk(self.state_dict[wq_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                        torch.transpose(
                            torch.chunk(self.state_dict[wk_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                        torch.transpose(
                            torch.chunk(self.state_dict[wv_str], self.num_devices)[i],
                            -2,
                            -1,
                        ),
                    ],
                    dim=-1,
                ),
                device=self.devices[i],
                dtype=self.dtype,
                memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                cache_file_name=cache_name("wqkv"),
            )

            bias_qkv = ttnn.as_tensor(
                torch.concat(
                    [
                        torch.chunk(self.state_dict[bias_q_str], self.num_devices)[i],
                        torch.chunk(self.state_dict[bias_k_str], self.num_devices)[i],
                        torch.chunk(self.state_dict[bias_v_str], self.num_devices)[i],
                    ],
                    dim=-1,
                ).unsqueeze(0),
                device=self.devices[i],
                dtype=self.dtype,
                memory_config=self.model_config["ATTN_BIAS_WEIGHTS_MEMCFG"],
                layout=self.model_config["ATTN_B_LAYOUT_TILE"],
                cache_file_name=cache_name("bias_qkv"),
            )

            wo = ttnn.as_tensor(
                torch.transpose(
                    torch.chunk(self.state_dict[wo_str], self.num_devices, dim=-1)[i],
                    -2,
                    -1,
                ),
                device=self.devices[i],
                memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
                dtype=self.dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                cache_file_name=cache_name("wo"),
            )

            cache_k = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    self.sliding_window,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.max_batch_size,
                    self.n_kv_heads // self.num_devices,
                    self.sliding_window,
                    self.head_dim,
                )
            )
            layer_past = [cache_k, cache_v]
            layer_past = [
                ttnn.from_torch(
                    lp, device=self.devices[i], layout=self.model_config["ATTN_W_LAYOUT_TILE"], dtype=self.dtype
                )
                for lp in layer_past
            ]

            # add to the list
            self.wqkv_list.append(wqkv)
            self.bias_qkv_list.append(bias_qkv)
            self.wo_list.append(wo)
            self.layer_past_list.append(layer_past)

        # Pre-scaled head dimension (for softmax) to avoid fallbacking to host
        self.head_dims = [
            ttnn.from_torch(
                torch.ones(1, self.n_heads, 32, self.head_dim)
                * (self.head_dim**-0.5),  # [seqlen, n_heads, bsz, head_dim] [1,32,32,128]
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(self.num_devices)
        ]

        self.q_heads_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )
        self.k_heads_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )
        expand_D_8D_torch = torch.eye(128, 128).repeat(1, 1, 1, 8)
        self.expand_D_8D = [
            ttnn.from_torch(
                expand_D_8D_torch,
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(len(devices))
        ]

        reduce_8D_D_torch = torch.eye(128, 128).repeat(1, 1, 8, 1)
        self.reduce_8D_D = [
            ttnn.from_torch(
                reduce_8D_D_torch,
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(len(devices))
        ]

        mask_Q_8D_torch = torch.zeros(1, self.max_batch_size, 32, 8 * 128)
        for j in range(8):
            mask_Q_8D_torch[:, :, j * 4 : (j + 1) * 4, j * 128 : (j + 1) * 128] = 1
        self.mask_Q_8D = [
            ttnn.from_torch(
                mask_Q_8D_torch,
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat8_b,
            )
            for i in range(len(devices))
        ]
        self.expand_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=2,
            out_subblock_w=2,
            per_core_M=4,
            per_core_N=4,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.reduce_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(self.grid_size.x, self.grid_size.y),
            in0_block_w=4,
            out_subblock_h=4,
            out_subblock_w=1,
            per_core_M=4,
            per_core_N=1,
            transpose_mcast=False,
            fused_activation=None,
        )

        self.attn_program_config = ttnn.MatmulMultiCoreReuseProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=32,
        )
        self.compute_kernel_config_attn = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        self.attention_grid = ttnn.CoreCoord(8, 4)
        self.scale = self.head_dim**-0.5

    def forward_decode(
        self,
        xs: List[ttnn.Tensor],
        current_pos: int,
        attn_masks: Optional[List[ttnn.Tensor]] = None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        attn_mask: (seq_len, n_heads, batch, cache_len + seqlen
        """
        # Indices for slicing tensors
        padded_layer_past_len = min(nearest_32(current_pos + 1), self.sliding_window)
        layer_slice = min((current_pos + 1), self.sliding_window)

        dense_outputs = []
        for i in range(self.num_devices):
            x = xs[i]
            if attn_masks is not None:
                attn_mask = attn_masks[i]
            else:
                attn_mask = None
            device = self.devices[i]
            wqkv = self.wqkv_list[i]
            bias_qkv = self.bias_qkv_list[i]
            wo = self.wo_list[i]
            layer_past = self.layer_past_list[i]
            head_dim = self.head_dims[i]
            expand_D_8D = self.expand_D_8D[i]
            reduce_8D_D = self.reduce_8D_D[i]
            mask_Q_8D = self.mask_Q_8D[i]

            ###
            # QKV matmuls
            ###
            xqkv_fused = ttnn.linear(
                x,
                wqkv,
                bias=bias_qkv,
                memory_config=self.model_config["XQKV_MM_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                dtype=self.dtype,
                core_grid=self.grid_size,
            )

            # Reshape such that true unpadded batch is tracked in shape
            if self.max_batch_size < 32:
                fqkv_shape = xqkv_fused.shape
                xqkv_fused = ttnn.reshape(
                    xqkv_fused, ttnn.Shape((1, 1, self.max_batch_size, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))
                )

            # ttnn.deallocate(x)

            ###
            # Reshape and rotary embeddings
            ###
            (
                q_heads_pre_rot,  # [seqlen, n_heads, bsz, head_dim]
                k_heads_pre_rot,  # [seqlen, n_kv_heads, bsz, head_dim]
                v_heads,  # [seqlen, n_kv_heads, bsz, head_dim]
            ) = ttnn.experimental.nlp_create_qkv_heads(
                xqkv_fused,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                transpose_k_heads=False,
                memory_config=self.model_config["QKV_HEADS_OUTPUT_MEMCFG"],
            )

            ttnn.deallocate(xqkv_fused)

            # Update rotary matrix on device
            rotary_mat = self.rot_mat[current_pos]

            q_heads = ttnn.linear(
                q_heads_pre_rot,
                rotary_mat,
                program_config=self.q_heads_program_config,
                memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                dtype=self.dtype,
            )

            k_heads = ttnn.linear(
                k_heads_pre_rot,
                rotary_mat,
                program_config=self.k_heads_program_config,
                memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                dtype=self.dtype,
            )

            ttnn.deallocate(q_heads_pre_rot)
            ttnn.deallocate(k_heads_pre_rot)

            ###
            # KV update
            ###
            keys = layer_past[0]
            values = layer_past[1]

            # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
            # v_heads [seqlen, n_kv_heads, bsz, head_dim]
            # keys, [max_batch_size, n_kv_heads // self.num_devices, sliding_window, head_dim]
            ttnn.kv_cache.update_cache_for_token_(keys, k_heads, current_pos)
            ttnn.kv_cache.update_cache_for_token_(values, v_heads, current_pos)
            self.layer_past_list[i] = [keys, values]

            ttnn.deallocate(k_heads)
            ttnn.deallocate(v_heads)

            ###
            # Attention
            ###
            # splitting attention implementation into 2 parts because for token id>575 we run out of memory for group_attn_matmul op
            if current_pos + 1 < 575:
                keys_sliced = keys[:, :, :padded_layer_past_len, :]
                keys_sliced_T = ttnn.permute(
                    keys_sliced, (0, 1, 3, 2)
                )  #  [batch, num_kv_heads, dhead, cache_len + seqlen]
                ttnn.deallocate(keys_sliced)
                q_heads = q_heads * head_dim  # Scale q_heads instead of QK before softmax

                # Reshape such that true unpadded batch is tracked in shape
                if self.max_batch_size < 32:
                    keys_sliced_T_shape = keys_sliced_T.shape
                    keys_sliced_T = ttnn.reshape(
                        keys_sliced_T, ttnn.Shape([32, self.n_kv_heads, self.head_dim, keys_sliced_T_shape[3]])
                    )

                attn = ttnn.experimental.group_attn_matmul(
                    q_heads,
                    keys_sliced_T,
                    compute_with_storage_grid_size=self.attention_grid,
                    memory_config=self.model_config["QK_MM_OUTPUT_MEMCFG"],
                    dtype=ttnn.bfloat16,  # Force bfloat16 for higher accuracy
                )  # seqlen, n_heads, batch, cache_len + seqlen

                ttnn.deallocate(keys_sliced_T)
                ttnn.deallocate(q_heads)

                attn_sliced = attn[:, :, :, :layer_slice]
                attn_sliced = ttnn.softmax(
                    attn_sliced,
                    dim=-1,
                    numeric_stable=True,
                )

                # Reshape such that true unpadded batch is tracked in shape
                if self.max_batch_size < 32:
                    values_sliced_shape = values.shape
                    values = ttnn.reshape(
                        values, ttnn.Shape([32, self.n_kv_heads, values_sliced_shape[2], self.head_dim])
                    )
                values_sliced = values[:, :, :layer_slice, :]
                attn_output = ttnn.experimental.group_attn_matmul(
                    attn_sliced,
                    values_sliced,
                    compute_with_storage_grid_size=self.attention_grid,
                    memory_config=self.model_config["QKV_MM_OUTPUT_MEMCFG"],
                    dtype=ttnn.bfloat8_b,  # Force bfloat16 for higher accuracy
                )  # seqlen, n_heads, batch, dhead

                ttnn.deallocate(attn_sliced)
                ttnn.deallocate(values_sliced)
                attn_output_cat = ttnn.transformer.concatenate_heads(
                    attn_output, memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"]
                )
                # seqlen, 1, batch, hidden_size

                ttnn.deallocate(attn_output)

            else:
                # reshape keys
                keys_BKPD = keys[:, :, :padded_layer_past_len, :]
                keys_1B_P_8D = ttnn.unsqueeze_to_4D(ttnn.transformer.concatenate_heads(keys_BKPD))
                keys_1B_P_8D = ttnn.clone(
                    keys_1B_P_8D, dtype=ttnn.bfloat16, memory_config=self.model_config["KV_UNPAD_OUTPUT_MEMCFG"]
                )
                keys_1B_8D_P_preshard = ttnn.permute(keys_1B_P_8D, (0, 1, 3, 2))

                keys_BKPD.deallocate()
                keys_1B_P_8D.deallocate()

                # reshape values
                values_BKPD = values[:, :, :padded_layer_past_len, :]
                values_B1_P_8D = ttnn.transformer.concatenate_heads(values_BKPD)
                values_1B_P_8D_preshard = ttnn.unsqueeze_to_4D(values_B1_P_8D)  # [:, :, :layer_slice, :]
                values_BKPD.deallocate()

                # reshape queries
                q_heads_1QBD = q_heads * head_dim  # Scale q_heads instead of QK before softmax
                q_heads_1QBD = ttnn.clone(
                    q_heads_1QBD, dtype=ttnn.bfloat16, memory_config=self.model_config["KV_UNPAD_OUTPUT_MEMCFG"]
                )
                q_heads_1BQD = ttnn.permute(q_heads_1QBD, (0, 2, 1, 3))
                if self.max_batch_size < 32:
                    q_heads_1BQD = q_heads_1BQD[:, : self.max_batch_size, :, :]
                q_heads_1QBD.deallocate()
                q_heads_1B_Q_8D_preshard = (
                    ttnn.matmul(
                        q_heads_1BQD,
                        expand_D_8D,
                        program_config=self.expand_program_config,
                        compute_kernel_config=self.compute_kernel_config,
                        memory_config=self.model_config["KV_UNPAD_OUTPUT_MEMCFG"],
                    )
                    * mask_Q_8D
                )
                q_heads_1BQD.deallocate()

                # scores matmul
                attn_1BQP = ttnn.matmul(
                    q_heads_1B_Q_8D_preshard,
                    keys_1B_8D_P_preshard,
                    core_grid=ttnn.CoreGrid(y=4, x=8),
                    compute_kernel_config=self.compute_kernel_config_attn,
                    dtype=ttnn.bfloat16,
                    memory_config=self.model_config["KV_UNPAD_OUTPUT_MEMCFG"],
                )
                keys_1B_8D_P_preshard.deallocate()
                q_heads_1B_Q_8D_preshard.deallocate()

                # scores softmax
                attn_1BQP_presoftmax = attn_1BQP[:, :, :, :layer_slice]
                attn_1BQP = ttnn.softmax(attn_1BQP_presoftmax, dim=-1, numeric_stable=True)
                attn_1BQP = ttnn.pad(attn_1BQP, ((0, 0), (0, 0), (0, 0), (0, 0)), value=0.0)

                # attention matmul
                attn_output_1B_Q_8D = ttnn.matmul(
                    attn_1BQP,
                    values_1B_P_8D_preshard,
                    program_config=self.attn_program_config,
                    memory_config=self.model_config["QKV_MM_OUTPUT_MEMCFG"],
                    dtype=ttnn.bfloat16,
                    compute_kernel_config=self.compute_kernel_config_attn,
                )

                attn_1BQP.deallocate()

                # reduce and reshape
                attn_output_1BQD = ttnn.matmul(
                    attn_output_1B_Q_8D * mask_Q_8D,
                    reduce_8D_D,
                    compute_kernel_config=self.compute_kernel_config,
                    program_config=self.reduce_program_config,
                    memory_config=self.model_config["QKV_MM_OUTPUT_MEMCFG"],
                )
                if self.max_batch_size < 32:
                    attn_output_1BQD_shape = attn_output_1BQD.shape
                    attn_output_1BQD = ttnn.reshape(
                        attn_output_1BQD, ttnn.Shape([1, 32, attn_output_1BQD_shape[2], 128])
                    )
                attn_output_1QBD = ttnn.permute(attn_output_1BQD, (0, 2, 1, 3))

                attn_output_1BQD.deallocate()
                attn_output_1B_Q_8D.deallocate()

                attn_output_cat = ttnn.transformer.concatenate_heads(
                    attn_output_1QBD, memory_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"]
                )
                attn_output_1QBD.deallocate()

            dense_out = ttnn.linear(
                attn_output_cat,
                wo,
                memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config,
                core_grid=self.grid_size,
            )  # seqlen, 1, batch, hidden_size

            ttnn.deallocate(attn_output_cat)
            dense_outputs.append(dense_out)

        # return the sum of the outputs
        if len(dense_outputs) > 1:
            return None  # tt_all_reduce(dense_outputs)
        else:
            return dense_outputs

    def forward_prefill(self, xs_11SH, attn_masks, rot_mats, transformation_mats, user_id: int = 0):
        seq_len = xs_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        wqkv = self.wqkv_list[0]
        bias_qkv = self.bias_qkv_list[0]
        wo = self.wo_list[0]
        self.layer_past = self.layer_past_list[0]
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            xs_11SH = ttnn.reshape(xs_11SH, [1, 2, seq_len // 2, -1])
        xqkv_fused = ttnn.linear(
            xs_11SH,
            wqkv,
            bias=bias_qkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )
        if seq_len > 2048:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        xs_11SH.deallocate(True)

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

        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        q_heads_1QSD_pre_rot.deallocate(True)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        k_heads_1KSD_pre_rot.deallocate(True)

        # Fill KV-Cache
        keys_BKSD = self.layer_past[0]
        values_BKSD = self.layer_past[1]

        k_fill = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
        # sharding k_fill to deal with update_cache memory limitation
        if seq_len > 128:
            k_fill = ttnn.interleaved_to_sharded(k_fill, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        v_fill = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)
        # sharding v_fill to deal with update_cache memory limitation
        if seq_len > 128:
            v_fill = ttnn.interleaved_to_sharded(v_fill, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
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

        self.layer_past = [keys_BKSD, values_BKSD]

        # SDPA

        # reshaping to put group in batch dim to do sdpa on 8 MQAs in parallel
        k_heads_K1SD = ttnn.reshape(k_heads_1KSD, [self.n_local_kv_heads, 1, -1, self.head_dim])
        v_heads_V1SD = ttnn.reshape(v_heads_1VSD, [self.n_local_kv_heads, 1, -1, self.head_dim])
        q_heads_84SD = ttnn.reshape(
            q_heads_1QSD, [self.n_local_kv_heads, self.n_local_heads // self.n_local_kv_heads, -1, self.head_dim]
        )
        attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_84SD,
            k_heads_K1SD,
            v_heads_V1SD,
            is_causal=True,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
        )

        attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])

        # deallocate keys and values
        q_heads_84SD.deallocate(True)
        k_heads_K1SD.deallocate(True)
        v_heads_V1SD.deallocate(True)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_output_1QSD.deallocate(True)

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 2, seq_len // 2, -1])
        output_11SH = ttnn.linear(
            attn_output_11SH,
            wo,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )
        if seq_len > 2048:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        attn_output_11SH.deallocate(True)
        return [output_11SH]

    def forward(
        self, xs, current_pos, attn_masks=None, rot_mats=None, transformation_mats=None, user_id=0, mode="decode"
    ):
        if mode == "prefill":
            return self.forward_prefill(xs[0], attn_masks[0], rot_mats, transformation_mats, user_id)
        else:
            return self.forward_decode(xs, current_pos, attn_masks)

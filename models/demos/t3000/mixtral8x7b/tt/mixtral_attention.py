# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import nearest_32
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor
from models.common.lightweightmodule import LightweightModule


class TtMixtralAttention(LightweightModule):
    def __init__(self, mesh_device, state_dict, args, layer_num, dtype):
        super().__init__()
        self.num_devices = 8
        self.tile_size = 32
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.model_args = args

        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        self.model_config = self.model_args.get_model_config()

        layer_name = f"layers.{layer_num}.attention"

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: self.model_args.weight_cache_path(dtype) / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        self.wqkv = ttnn.as_tensor(
            torch.concat(
                [
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
                    )
                    for i in range(self.num_devices)
                ],
                dim=-1,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wqkv_multidevice_4d"),
        )

        self.wo = ttnn.as_tensor(
            torch.transpose(
                self.state_dict[wo_str],
                -2,
                -1,
            )
            .unsqueeze(0)
            .unsqueeze(0),
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=-2),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wo_multidevice4d"),
        )

        cache_k = torch.zeros(
            (
                self.max_batch_size,
                self.n_kv_heads,
                self.max_seq_len,
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.max_batch_size,
                self.n_kv_heads,
                self.max_seq_len,
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.as_tensor(
                lp,
                device=self.mesh_device,
                mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=1),
                dtype=ttnn.bfloat8_b,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                memory_config=self.model_config["ATTN_CACHE_WEIGHTS_MEMCFG"],
                cache_file_name=cache_name(f"empty_attn_cache_T_{cache_k.shape}"),
            )
            for lp in layer_past
        ]

        self.scale = self.head_dim**-0.5

        reduce_mask_torch = torch.zeros(1, 1, self.tile_size, self.tile_size * 8)
        for i in range(self.tile_size):
            reduce_mask_torch[:, :, i, range(i, self.tile_size * 8, self.tile_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )

        self.compute_kernel = self.model_args.get_compute_kernel_config()
        self.compute_kernel_attn = self.model_args.get_compute_kernel_attn_config()

        self.core_grid = self.model_args.max_grid_size
        self.core_grid_attention = self.model_args.core_grid_attention

        # Will be filled during the initial warmup run
        self.q_mem_config = None
        self.k_mem_config = None

    def forward_decode(
        self,
        xs,
        start_pos_ids,
        rot_mat,
    ):
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos_ids: start position ids
        rot_mats: rotation matrix for each device

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        D : head_dim (128)
        P : padded_layer_past_len
        """

        x_11BH = xs
        wo = self.wo
        layer_past = self.layer_past
        ###
        # QKV matmuls
        ###

        xqkv_fused = ttnn.matmul(
            x_11BH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            program_config=self.model_config["QKV_MM_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
        )

        # Reshape such that true unpadded batch is tracked in shape
        if self.max_batch_size < 32:
            fqkv_shape = xqkv_fused.shape
            xqkv_fused = ttnn.reshape(
                xqkv_fused, ttnn.Shape((1, 1, self.max_batch_size, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3]))
            )

        # split qkv into heads
        (
            q_heads_1B4D,
            k_heads_1B1D,
            v_heads_1B1D,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###
        if self.q_mem_config is None:
            self.q_mem_config = q_heads_1B4D.memory_config()
        if self.k_mem_config is None:
            self.k_mem_config = k_heads_1B1D.memory_config()

        q_heads_1B4D = ttnn.matmul(
            q_heads_1B4D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=self.q_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
            # [seqlen, bsz, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, bsz, padd_heads, head_dim]
        )

        k_heads_1B1D = ttnn.matmul(
            k_heads_1B1D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=self.k_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )

        ###
        # KV update
        ###
        keys_1BPD = layer_past[0]
        values_1BPD = layer_past[1]
        ttnn.experimental.paged_update_cache(keys_1BPD, k_heads_1B1D, update_idxs=start_pos_ids)
        ttnn.experimental.paged_update_cache(values_1BPD, v_heads_1B1D, update_idxs=start_pos_ids)
        self.layer_past = [keys_1BPD, values_1BPD]
        k_heads_1B1D.deallocate(True)
        v_heads_1B1D.deallocate(True)

        attn_output_1B4D = ttnn.transformer.scaled_dot_product_attention_decode(
            q_heads_1B4D,
            keys_1BPD,
            values_1BPD,
            start_pos_ids,
            scale=self.scale,
            program_config=self.model_config["SDPA_DECODE_PROGCFG"],
            compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
            memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
        )

        attn_output_11BH = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_1B4D,
            num_heads=4,
        )
        attn_output_1B4D.deallocate(True)

        ###
        # Output matmul
        ###
        dense_out_11BH = ttnn.matmul(
            attn_output_11BH,
            wo,
            memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
            # compute_with_storage_grid_size=(8, 8),
            program_config=self.model_config["LM_HEAD_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat8_b,
        )
        attn_output_11BH.deallocate(True)
        # All gather
        dense_outputs_11BH = ttnn.all_gather(dense_out_11BH, dim=2, num_links=1)

        # return the sum of the outputs
        dense_outputs_11BH = ttnn.matmul(self.reduce_mask, dense_outputs_11BH)
        return dense_outputs_11BH

    def forward_prefill(self, xs_11SH, attn_masks, rot_mats, transformation_mats, user_id: int = 0):
        """
        x: (1, 1, seq_len, hidden_dim)
        attn_masks: (1, 1, seq_len, seq_len)
        rot_mats: rotation matrices for each device
        transformation_mats: transformation matrix (rotary embedding) for each device
        user_id: user id for the kv cache

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        S : seq_len
        H : dim (4096)
        D : head_dim (128)
        """

        seq_len = xs_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        ###
        # QKV matmuls
        ###
        xqkv_program_config = None
        xqkv_mem_config = ttnn.L1_MEMORY_CONFIG
        if seq_len > 8192:  # Too large to fit in L1. Reshape and parallelize in multiple cores
            xs_11SH = ttnn.reshape(xs_11SH, (1, seq_len // 2048, 2048, -1))
            xqkv_program_config = self.model_config["WQKV_PREFILL_PROGCFG"]
            xqkv_mem_config = ttnn.DRAM_MEMORY_CONFIG

        xqkv_fused = ttnn.linear(
            xs_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=xqkv_mem_config,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not xqkv_program_config else None,
            compute_kernel_config=self.compute_kernel,
            program_config=xqkv_program_config,
        )

        xs_11SH.deallocate(True)

        if seq_len > 8192:  # Reshape back again to intended shape
            xqkv_fused = ttnn.reshape(xqkv_fused, (1, 1, seq_len, -1))
        # split qkv into heads
        (
            q_heads_14SD_pre_rot,
            k_heads_11SD_pre_rot,
            v_heads_11SD,
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

        q_heads_14SD = ttnn.experimental.rotary_embedding_llama(
            q_heads_14SD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        q_heads_14SD_pre_rot.deallocate(True)

        k_heads_11SD = ttnn.experimental.rotary_embedding_llama(
            k_heads_11SD_pre_rot, rot_mats[0], rot_mats[1], transformation_mats
        )
        k_heads_11SD_pre_rot.deallocate(True)

        # Fill KV-Cache
        keys_11SD = self.layer_past[0]
        values_11SD = self.layer_past[1]

        if seq_len > 128:
            k_heads_11SD = ttnn.typecast(k_heads_11SD, dtype=ttnn.bfloat8_b)
            v_heads_11SD = ttnn.typecast(v_heads_11SD, dtype=ttnn.bfloat8_b)
            q_heads_14SD = ttnn.typecast(q_heads_14SD, dtype=ttnn.bfloat8_b)
        ttnn.kv_cache.fill_cache_for_user_(keys_11SD, ttnn.typecast(k_heads_11SD, dtype=ttnn.bfloat8_b), user_id)
        ttnn.kv_cache.fill_cache_for_user_(values_11SD, ttnn.typecast(v_heads_11SD, dtype=ttnn.bfloat8_b), user_id)

        self.layer_past = [keys_11SD, values_11SD]

        # SDPA
        attn_output_14SD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_14SD,
            k_heads_11SD,
            v_heads_11SD,
            is_causal=True,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
        )

        # deallocate keys and values
        q_heads_14SD.deallocate(True)
        k_heads_11SD.deallocate(True)
        v_heads_11SD.deallocate(True)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_14SD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_output_14SD.deallocate(True)

        if seq_len >= 1024:  # Specific program config to make better usage of cores and to avoid di/dt ND issues
            wo_program_config = self.model_config["WO_PREFILL_PROGCFG"]
        else:
            wo_program_config = None  # For 128 seqlen case just use default program config

        if seq_len > 2048:  # To big to compute. Reshape and manually parallelize across multiple cores
            attn_output_11SH = ttnn.reshape(attn_output_11SH, (1, seq_len // 2048, 2048, -1))

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            core_grid=ttnn.CoreGrid(y=8, x=8) if not wo_program_config else None,
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=wo_program_config,
        )
        attn_output_11SH.deallocate(True)

        if seq_len > 2048:  # Reshape back to intended shape
            output_11SH = ttnn.reshape(output_11SH, (1, 1, seq_len, -1))
        output_11BH_gathered = ttnn.all_gather(output_11SH, dim=1, num_links=1)
        output_11SH.deallocate(True)
        output_11BH_reduced = ttnn.experimental.fast_reduce_nc(
            output_11BH_gathered, dims=[1], output=None, compute_kernel_config=None
        )
        output_11BH_gathered.deallocate(True)
        return output_11BH_reduced

    def forward(self, xs, start_pos_ids, attn_masks, rot_mats, transformation_mats=None, user_id=0, mode="decode"):
        if mode == "prefill":
            return self.forward_prefill(xs, attn_masks, rot_mats, transformation_mats, user_id)
        else:
            assert attn_masks is None, "attn_masks should be None for decode mode"
            return self.forward_decode(xs, start_pos_ids, rot_mats)

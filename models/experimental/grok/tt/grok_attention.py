# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import nearest_32
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor
from models.experimental.grok.tt.grok_common import LightweightModule


class TtGrokAttention(LightweightModule):
    def __init__(self, device_mesh, state_dict, args, layer_num, dtype):
        super().__init__()
        self.num_devices = 8
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.model_args = args

        self.hidden_size = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.head_dim = args.head_dim
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.num_key_value_heads
        self.max_attn_value = args.max_attn_value
        self.attn_output_multiplier = args.attn_output_multiplier

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        self.model_config = self.model_args.get_model_config()

        layer_name = f"model.layers.{layer_num}.attn"

        if args.dummy_weights:
            cache_name = lambda _: None
        else:
            cache_name = lambda name: self.model_args.weight_cache_path(dtype) / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.q_proj.weight"
        wk_str = f"{layer_name}.k_proj.weight"
        wv_str = f"{layer_name}.v_proj.weight"
        wo_str = f"{layer_name}.o_proj.weight"

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
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=-1),
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
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wo_multidevice4d_H"),
        )

        cache_k = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.max_seq_len,
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.max_seq_len,
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.as_tensor(
                lp,
                device=self.device_mesh,
                mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
                dtype=ttnn.bfloat8_b,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                memory_config=self.model_config["ATTN_CACHE_WEIGHTS_MEMCFG"],
                cache_file_name=cache_name(f"empty_attn_cache_{cache_k.shape}"),
            )
            for lp in layer_past
        ]

        self.compute_kernel = self.model_args.get_compute_kernel_config()
        self.compute_kernel_attn = self.model_args.get_compute_kernel_attn_config()

        self.core_grid = self.model_args.max_grid_size
        self.core_grid_attention = self.model_args.core_grid_attention

        # Will be filled during the initial warmup run
        self.q_mem_config = None
        self.k_mem_config = None

    def forward(
        self,
        xs,
        current_pos,
        attn_masks,
        rot_mats,
    ):
        """
        x: (seq_len, 1, batch, hidden_dim)
        current_pos: the length of the KV cache. Same as current token's index.
        attn_masks: (seq_len, batch, n_heads, cache_len+seq_len)
        rot_mats: list of rotation matrices for each device

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (6144)
        D : head_dim (128)
        P : padded_layer_past_len
        """
        padded_layer_past_len = min(nearest_32(current_pos + 1), self.max_seq_len)

        x_11BH = xs
        wo = self.wo
        layer_past = self.layer_past
        rot_mat = rot_mats[current_pos]
        attn_mask_1B4P = attn_masks
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
        # new_key_states = ttnn.to_torch(k_heads_1B1D, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))

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
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"]
            # [seqlen, bsz, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, bsz, padd_heads, head_dim]
        )
        k_heads_1B1D = ttnn.matmul(
            k_heads_1B1D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=self.k_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )
        # rotmat_key_states = ttnn.to_torch(k_heads_1B1D, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))
        # rotmat = ttnn.to_torch(rot_mat, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))[0]

        ###
        # KV update
        ###
        keys_1BPD = layer_past[0]
        values_1BPD = layer_past[1]
        ttnn.kv_cache.update_cache_for_token_(keys_1BPD, k_heads_1B1D, current_pos)
        ttnn.kv_cache.update_cache_for_token_(values_1BPD, v_heads_1B1D, current_pos)
        self.layer_past = [keys_1BPD, values_1BPD]
        k_heads_1B1D.deallocate(True)
        v_heads_1B1D.deallocate(True)

        keys_1BPD = ttnn.experimental.nlp_kv_cache_load_slice(
            keys_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )

        # query_states = ttnn.to_torch(q_heads_1B4D, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=-2))
        # key_states = ttnn.to_torch(keys_1BPD, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))

        ###
        # Attention
        ###
        # transpose keys
        keys_1BDP = ttnn.transpose(
            keys_1BPD,
            -2,
            -1,
            memory_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        keys_1BPD.deallocate(True)

        # scores matmul
        attn_1B4P_memconfig = self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"](padded_layer_past_len)
        attn_1B4P = ttnn.matmul(
            q_heads_1B4D,
            keys_1BDP,
            dtype=ttnn.bfloat16,
            program_config=self.model_config["SCORES_BATCHED_MM_PROGCFG"](padded_layer_past_len // 32),
            memory_config=attn_1B4P_memconfig,
            compute_kernel_config=self.compute_kernel_attn,
        )
        q_heads_1B4D.deallocate(True)
        keys_1BDP.deallocate(True)

        # Softmax and scaling
        # FIXME: Maintain sharded memory layout when #9773 is fixed
        attn_1B4P = ttnn.sharded_to_interleaved(attn_1B4P, memory_config=ttnn.L1_MEMORY_CONFIG)
        attn_1B4P = attn_1B4P * self.attn_output_multiplier
        attn_1B4P = self.max_attn_value * ttnn.tanh(attn_1B4P * (1.0 / self.max_attn_value))
        attn_1B4P = ttnn.interleaved_to_sharded(attn_1B4P, attn_1B4P_memconfig)

        attn_1B4P = ttnn.scale_mask_softmax_in_place(
            attn_1B4P,
            1.0,
            attn_mask_1B4P,
            program_config=self.model_config["ATTN_BATCHED_SOFTMAX_PROGCFG"](padded_layer_past_len),
            is_causal_mask=True,
        )
        # post_softmax = ttnn.to_torch(attn_1B4P, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=-2))[0]

        # values matmul
        values_1BPD = ttnn.experimental.nlp_kv_cache_load_slice(
            values_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )

        # value_states = ttnn.to_torch(values_1BPD, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))
        # x = ttnn.to_torch(x_11BH, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))[0]
        attn_output_1B4D = ttnn.matmul(
            attn_1B4P,
            values_1BPD,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
            program_config=self.model_config["VALUES_BATCHED_MM_PROGCFG"](padded_layer_past_len // 32),
            compute_kernel_config=self.compute_kernel_attn,
        )
        attn_1B4P.deallocate(True)
        values_1BPD.deallocate(True)

        # value_output = ttnn.to_torch(attn_output_1B4D, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))[0]

        attn_output_11BH = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_1B4D,
            num_heads=6,
        )
        attn_output_1B4D.deallocate(True)

        attn_output_11BH = ttnn.sharded_to_interleaved(attn_output_11BH, memory_config=ttnn.L1_MEMORY_CONFIG)

        ###
        # Output matmul
        ###
        # All gather
        dense_outputs_11BH_gathered = ttnn.all_gather(attn_output_11BH, dim=3, num_links=1)

        # return the sum of the outputs
        dense_outputs_11BH = ttnn.matmul(
            dense_outputs_11BH_gathered,
            wo,
            memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
            # compute_with_storage_grid_size=(8, 8),
            program_config=self.model_config["LM_HEAD_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat8_b,
        )

        # attn_output = ttnn.to_torch(dense_outputs_11BH, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))[0]
        # attn_mask = ttnn.to_torch(attn_mask_1B4P, mesh_composer=ConcatMeshToTensor(self.device_mesh, dim=0))[0]

        # torch.save({'x': x,
        #             'query_states': query_states,
        #             'rotmat': rotmat,
        #             'rotmat_key_states': rotmat_key_states,
        #             'new_key_states': new_key_states,
        #             'key_states': key_states,
        #             'value_states': value_states,
        #             'attn_mask': attn_mask,
        #             'pre_scale': pre_scale,
        #             'attn_output_multiplier': self.attn_output_multiplier,
        #             'pre_tanh': pre_tanh,
        #             'pre_softmax': pre_softmax,
        #             'post_softmax': post_softmax,
        #             'value_output': value_output,
        #             'attn_output': attn_output,
        # }, 'our.pt')

        dense_outputs_11BH_gathered.deallocate(True)
        return dense_outputs_11BH

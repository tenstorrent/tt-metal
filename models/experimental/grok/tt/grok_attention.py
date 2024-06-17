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

        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.sliding_window = args.sliding_window

        self.n_local_heads = self.n_heads // self.num_devices
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices

        self.dtype = dtype

        self.model_config = self.model_args.get_model_config()

        layer_name = f"layers.{layer_num}.attn"

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
                self.sliding_window,
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.n_kv_heads,
                self.max_batch_size,
                self.sliding_window,
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

        self.scale = self.head_dim**-0.5

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
        start_pos,
        attn_masks,
        rot_mats,
    ):
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        attn_masks: (seq_len, batch, n_heads, cache_len+seq_len)
        rot_mats: list of rotation matrices for each device

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (6144)
        D : head_dim (128)
        P : padded_layer_past_len
        """
        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)

        x_11BH = xs
        wo = self.wo
        layer_past = self.layer_past
        rot_mat = rot_mats[start_pos]
        attn_mask_1B4P = attn_masks
        ###
        # QKV matmuls
        ###

        xqkv_fused = ttnn.experimental.operations.primary.matmul(
            x_11BH,
            self.wqkv,
            output_dtype=ttnn.bfloat16,
            output_mem_config=self.model_config["FUSED_QKV_MM_OUTPUT_MEMCFG"],
            program_config=self.model_config["QKV_MM_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
        )

        # split qkv into heads
        (
            q_heads_1B4D,
            k_heads_1B1D,
            v_heads_1B1D,
        ) = ttnn.experimental.tensor.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        xqkv_fused.deallocate(True)

        ###
        # Rotary embeddings
        ###
        if self.q_mem_config is None:
            self.q_mem_config = q_heads_1B4D.memory_config()
        if self.k_mem_config is None:
            self.k_mem_config = k_heads_1B1D.memory_config()

        q_heads_1B4D = ttnn.experimental.operations.primary.matmul(
            q_heads_1B4D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            output_mem_config=self.q_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"]
            # [seqlen, bsz, padd_heads, head_dim]  # [1, 1, head_dim, head_dim]  => [seqlen, bsz, padd_heads, head_dim]
        )
        k_heads_1B1D = ttnn.experimental.operations.primary.matmul(
            k_heads_1B1D,
            rot_mat,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            output_mem_config=self.k_mem_config,
            compute_kernel_config=self.model_config["ROT_MAT_COMPUTE_KERNEL_CONFIG"],
        )

        ###
        # KV update
        ###
        keys_1BPD = layer_past[0]
        values_1BPD = layer_past[1]
        ttnn.kv_cache.update_cache_for_token_(keys_1BPD, k_heads_1B1D, start_pos)
        ttnn.kv_cache.update_cache_for_token_(values_1BPD, v_heads_1B1D, start_pos)
        self.layer_past = [keys_1BPD, values_1BPD]
        k_heads_1B1D.deallocate(True)
        v_heads_1B1D.deallocate(True)

        keys_1BPD = ttnn.experimental.tensor.nlp_kv_cache_load_slice(
            keys_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )

        ###
        # Attention
        ###
        # transpose keys
        keys_1BDP = ttnn.experimental.tensor.transpose(
            keys_1BPD,
            -2,
            -1,
            output_mem_config=self.model_config["HEIGHT_SHARDED_MEMCFG"],
        )
        keys_1BPD.deallocate(True)

        # scores matmul

        attn_1B4P = ttnn.experimental.operations.primary.matmul(
            q_heads_1B4D,
            keys_1BDP,
            output_dtype=ttnn.bfloat16,
            program_config=self.model_config["SCORES_BATCHED_MM_PROGCFG"](padded_layer_past_len // 32),
            output_mem_config=self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"](padded_layer_past_len),
            compute_kernel_config=self.compute_kernel_attn,
        )
        q_heads_1B4D.deallocate(True)
        keys_1BDP.deallocate(True)

        # Softmax and scaling

        # TODO: attn_weights = self.max_attn_val * F.tanh(attn_weights / self.max_attn_val)

        attn_1B4P = ttnn.experimental.operations.primary.transformers.scale_mask_softmax_in_place(
            attn_1B4P,
            self.scale,
            attn_mask_1B4P,
            program_config=self.model_config["ATTN_BATCHED_SOFTMAX_PROGCFG"](padded_layer_past_len),
            is_causal_mask=True,
        )

        # values matmul
        values_1BPD = ttnn.experimental.tensor.nlp_kv_cache_load_slice(
            values_1BPD, seq_len_start=0, seq_len_end=padded_layer_past_len
        )

        attn_output_1B4D = ttnn.experimental.operations.primary.matmul(
            attn_1B4P,
            values_1BPD,
            output_dtype=ttnn.bfloat16,
            output_mem_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"],
            program_config=self.model_config["VALUES_BATCHED_MM_PROGCFG"](padded_layer_past_len // 32),
            compute_kernel_config=self.compute_kernel_attn,
        )
        attn_1B4P.deallocate(True)
        values_1BPD.deallocate(True)

        attn_output_11BH = ttnn.experimental.tensor.nlp_concat_heads_decode(
            attn_output_1B4D,
            num_heads=4,
        )
        attn_output_1B4D.deallocate(True)

        attn_output_11BH = ttnn.experimental.tensor.sharded_to_interleaved(
            attn_output_11BH, output_mem_config=ttnn.L1_MEMORY_CONFIG
        )

        ###
        # Output matmul
        ###
        # All gather
        dense_outputs_11BH_gathered = ttnn.all_gather(attn_output_11BH, dim=3, num_links=1)

        # return the sum of the outputs
        dense_outputs_11BH = ttnn.experimental.operations.primary.matmul(
            dense_outputs_11BH_gathered,
            wo,
            output_mem_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
            # compute_with_storage_grid_size=(8, 8),
            program_config=self.model_config["LM_HEAD_OUTPUT_PROGCFG"],
            compute_kernel_config=self.compute_kernel,
            output_dtype=ttnn.bfloat8_b,
        )

        dense_outputs_11BH_gathered.deallocate(True)
        return dense_outputs_11BH

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import nearest_32
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh, ConcatMeshToTensor


class TtMixtralAttention(torch.nn.Module):
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

        layer_name = f"layers.{layer_num}.attention"
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
            ),
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=1),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wqkv_multidevice"),
        )
        self.wqkv = ttnn.to_device(self.wqkv, self.device_mesh)
        self.wo = ttnn.as_tensor(
            torch.transpose(
                self.state_dict[wo_str],
                -2,
                -1,
            ),
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
            dtype=self.dtype,
            memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
            layout=self.model_config["ATTN_W_LAYOUT_TILE"],
            cache_file_name=cache_name(f"wo_multidevice"),
        )
        self.wo = ttnn.to_device(self.wo, self.device_mesh)

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
            ttnn.from_torch(
                lp,
                device=self.device_mesh,
                mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=0),
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                dtype=ttnn.bfloat16,
            )
            for lp in layer_past
        ]

        self.layer_past = [ttnn.to_device(lp, self.device_mesh) for lp in self.layer_past]

        # Scale tensor for q_heads to avoid falling back to host.
        self.head_dims = ttnn.from_torch(
            torch.ones(1, 4, self.max_batch_size, self.head_dim) * (self.head_dim**-0.5),
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        self.head_dims = ttnn.to_device(self.head_dims, self.device_mesh)
        reduce_mask_torch = torch.zeros(1, 1, self.max_batch_size, self.max_batch_size * 8)
        for i in range(self.max_batch_size):
            reduce_mask_torch[:, :, i, range(i, self.max_batch_size * 8, self.max_batch_size)] = 1
        self.reduce_mask = ttnn.from_torch(
            reduce_mask_torch,
            device=self.device_mesh,
            mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )

        self.reduce_mask = ttnn.to_device(self.reduce_mask, self.device_mesh)
        self.compute_kernel = self.model_args.get_compute_kernel_config()
        self.compute_kernel_attn = self.model_args.get_compute_kernel_attn_config()

        self.core_grid = self.model_args.max_grid_size
        self.core_grid_attention = self.model_args.core_grid_attention

    def forward(
        self,
        xs,
        start_pos,
        current_pos,
        rot_mats,
    ):
        """
        x: (seq_len, 1, batch, hidden_dim)
        start_pos: the length of the KV cache. Same as current token's index.
        current_pos: start_pos % self.sliding_window
        rot_mats: list of rotation matrices for each device

        Tensors are postfixed with 4 characters that represent their 4-D shape:
        B : batch_size (32)
        H : dim (4096)
        D : head_dim (128)
        P : padded_layer_past_len
        """
        padded_layer_past_len = min(nearest_32(start_pos + 1), self.sliding_window)
        layer_slice = min((start_pos + 1), self.sliding_window)

        x_11BH = xs
        wo = self.wo
        layer_past = self.layer_past
        rot_mat = rot_mats[start_pos]
        head_dim_14BD = self.head_dims
        ###
        # QKV matmuls
        ###
        xqkv_fused = ttnn.linear(
            x_11BH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["XQKV_MM_OUTPUT_MEMCFG"],
            core_grid=self.core_grid,
            compute_kernel_config=self.compute_kernel,
        )

        # split qkv into heads
        (
            q_heads_14BD,
            k_heads_11BD,
            v_heads_11BD,
        ) = ttnn.experimental.tensor.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            output_mem_config=self.model_config["QKV_HEADS_OUTPUT_MEMCFG"],
        )

        ###
        # Rotary embeddings
        ###
        q_heads_14BD = ttnn.linear(
            q_heads_14BD,
            rot_mat,
            core_grid=self.core_grid,
            use_1d_systolic_array=True,
            memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
        )
        k_heads_11BD = ttnn.linear(
            k_heads_11BD,
            rot_mat,
            core_grid=self.core_grid,
            use_1d_systolic_array=True,
            memory_config=self.model_config["QV_ROT_EMB_OUTPUT_MEMCFG"],
        )

        q_heads_14BD = q_heads_14BD * head_dim_14BD  # Scale q_heads
        q_heads_14BD = ttnn.pad(
            q_heads_14BD, ((0, 0), (0, self.max_batch_size - self.n_local_heads), (0, 0), (0, 0)), value=0
        )
        q_heads_1B4D = ttnn.permute(q_heads_14BD, (0, 2, 1, 3))
        k_heads_11BD = ttnn.pad(k_heads_11BD, ((0, 0), (0, self.max_batch_size - 1), (0, 0), (0, 0)), value=0)
        k_heads_1B1D = ttnn.permute(k_heads_11BD, (0, 2, 1, 3))
        v_heads_11BD = ttnn.pad(v_heads_11BD, ((0, 0), (0, self.max_batch_size - 1), (0, 0), (0, 0)), value=0)
        v_heads_1B1D = ttnn.permute(v_heads_11BD, (0, 2, 1, 3))
        ###
        # KV update
        ###
        keys_1BPD = layer_past[0]
        values_1BPD = layer_past[1]
        ttnn.kv_cache.update_cache_for_token_(keys_1BPD, k_heads_1B1D, current_pos)
        ttnn.kv_cache.update_cache_for_token_(values_1BPD, v_heads_1B1D, current_pos)
        self.layer_past = [keys_1BPD, values_1BPD]
        keys_1BPD = keys_1BPD[:, :, :padded_layer_past_len, :]
        values_1BPD = values_1BPD[:, :, :layer_slice, :]

        ###
        # Attention
        ###
        # transpose keys
        keys_1BDP = ttnn.permute(keys_1BPD, (0, 1, 3, 2))

        # pad, transpose, scale, and shard q head
        # q_heads_1B4D = ttnn.to_memory_config(q_heads_1B4D, self.model_config["Q_TRANSPOSE_MEMCFG"])
        # #shard keys
        # k_cache_memcfg = self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"](padded_layer_past_len)
        # keys_1BDP = ttnn.to_memory_config(keys_1BDP, k_cache_memcfg)
        # #shard values
        # v_cache_memcfg = self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"](padded_layer_past_len)
        # values_1BPD = ttnn.to_memory_config(values_1BPD, v_cache_memcfg)
        # #create out cfg
        # attn_output_memcfg = self.model_config["ATTN_BATCHED_MM_OUTPUT_MEMCFG"](padded_layer_past_len)

        # scores matmul
        attn_1B4P = ttnn.matmul(
            q_heads_1B4D,
            keys_1BDP,
            dtype=ttnn.bfloat16,
            # memory_config=attn_output_memcfg,
            core_grid=self.core_grid_attention,
            compute_kernel_config=self.compute_kernel_attn,
        )
        # scores slice and softmax
        attn_1B4P = attn_1B4P[:, :, :, :layer_slice]
        attn_1B4P = ttnn.softmax(attn_1B4P, dim=-1)  # , memory_config=attn_output_memcfg_post_sm)
        # values matmul
        attn_output_1B4D = ttnn.matmul(
            attn_1B4P,
            values_1BPD,
            dtype=ttnn.bfloat16,
            memory_config=self.model_config["QKV_MM_OUTPUT_MEMCFG"],
            core_grid=self.core_grid_attention,
            compute_kernel_config=self.compute_kernel_attn,
        )

        # transpose attn
        attn_output_14BD = ttnn.permute(attn_output_1B4D, (0, 2, 1, 3))
        # unpad attn
        attn_output_14BD = attn_output_14BD[:, : self.n_local_heads, :, :]

        attn_output_11BH = ttnn.experimental.tensor.nlp_concat_heads(
            attn_output_14BD,
            output_mem_config=self.model_config["CONCAT_HEADS_OUTPUT_MEMCFG"],
        )

        ###
        # Output matmul
        ###
        dense_out_11BH = ttnn.linear(
            attn_output_11BH,
            wo,
            memory_config=self.model_config["LM_HEAD_OUTPUT_MEMCFG"],
            core_grid=self.core_grid,
            compute_kernel_config=self.compute_kernel,
            dtype=ttnn.bfloat8_b,
        )
        # All gather
        dense_outputs_11BH = ttnn.all_gather(dense_out_11BH, dim=2, num_links=1)

        # return the sum of the outputs
        dense_outputs_11BH = ttnn.matmul(self.reduce_mask, dense_outputs_11BH)

        return dense_outputs_11BH

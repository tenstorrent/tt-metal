# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.utility_functions import nearest_32


class TtMixtralAttention(torch.nn.Module):
    def __init__(self, devices, state_dict, args, layer_num, dtype):
        super().__init__()

        self.state_dict = state_dict
        self.devices = devices
        self.num_devices = len(devices)
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

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices == 0
        assert self.n_kv_heads % self.num_devices == 0

        self.wqkv_list = []
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
                cache_file_name=cache_name(f"wqkv_{i}_"),
            )
            wo = ttnn.as_tensor(
                torch.transpose(
                    torch.chunk(self.state_dict[wo_str], self.num_devices, dim=-1)[i],
                    -2,
                    -1,
                ),
                device=self.devices[i],
                dtype=self.dtype,
                memory_config=self.model_config["ATTN_WEIGHTS_MEMCFG"],
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                cache_file_name=cache_name(f"wo_{i}_"),
            )

            cache_k = torch.zeros(
                (
                    self.n_kv_heads // self.num_devices,
                    self.max_batch_size,
                    self.sliding_window,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.n_kv_heads // self.num_devices,
                    self.max_batch_size,
                    self.sliding_window,
                    self.head_dim,
                )
            )
            layer_past = [cache_k, cache_v]
            layer_past = [
                ttnn.from_torch(
                    lp, device=self.devices[i], layout=self.model_config["ATTN_W_LAYOUT_TILE"], dtype=ttnn.bfloat16
                )
                for lp in layer_past
            ]

            # add to the list
            self.wqkv_list.append(wqkv)
            self.wo_list.append(wo)
            self.layer_past_list.append(layer_past)

        # Scale tensor for q_heads to avoid falling back to host.
        self.head_dims = [
            ttnn.from_torch(
                torch.ones(1, self.n_local_heads, self.max_batch_size, self.head_dim) * (self.head_dim**-0.5),
                device=self.devices[i],
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            for i in range(self.num_devices)
        ]
        reduce_mask_torch = torch.zeros(1, 1, self.max_batch_size, self.max_batch_size * len(self.devices))
        for i in range(self.max_batch_size):
            reduce_mask_torch[:, :, i, range(i, self.max_batch_size * len(self.devices), self.max_batch_size)] = 1
        self.reduce_mask = [
            ttnn.from_torch(reduce_mask_torch, device=device, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
            for device in self.devices
        ]

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
        dense_outputs_11BH = []

        for i in range(self.num_devices):
            x_11BH = xs[i]
            wqkv = self.wqkv_list[i]
            wo = self.wo_list[i]
            layer_past = self.layer_past_list[i]
            rot_mat = rot_mats[i][start_pos]
            head_dim_14BD = self.head_dims[i]
            ###
            # QKV matmuls
            ###
            xqkv_fused = ttnn.linear(
                x_11BH,
                wqkv,
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
            self.layer_past_list[i] = [keys_1BPD, values_1BPD]
            keys_1BPD = keys_1BPD[:, :, :padded_layer_past_len, :]
            values_1BPD = values_1BPD[:, :, :layer_slice, :]

            ###
            # Attention
            ###
            # transpose keys
            keys_1BDP = ttnn.permute(keys_1BPD, (0, 1, 3, 2))

            # pad, transpose, scale, and shard q head
            # q_heads_1B4D = ttnn.to_memory_config(q_heads_1B4D, self.model_config["Q_TRANSPOSE_MEMCFG"])

            # shard keys
            # k_cache_memcfg = self.model_config["K_CACHE_SLICE_OUTPUT_MEMCFG"](padded_layer_past_len)
            # keys_1BDP = ttnn.to_memory_config(keys_1BDP, k_cache_memcfg)
            # shard values
            # v_cache_memcfg = self.model_config["V_CACHE_SLICE_OUTPUT_MEMCFG"](padded_layer_past_len)
            # values_1BPD = ttnn.to_memory_config(values_1BPD, v_cache_memcfg)
            # create out cfg
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
            dense_outputs_11BH.append(dense_out_11BH)

        # All gather
        dense_outputs_11BH = ttnn.experimental.tensor.all_gather(dense_outputs_11BH, dim=2, num_links=1)

        # return the sum of the outputs
        for i in range(len(dense_outputs_11BH)):
            dense_outputs_11BH[i] = ttnn.matmul(self.reduce_mask[i], dense_outputs_11BH[i])

        return dense_outputs_11BH

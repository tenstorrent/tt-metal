# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32


class TtLlamaImageAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices

        self.hidden_size = configuration.vision_dim
        self.n_heads = configuration.vision_attn_n_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.n_kv_heads = self.n_heads

        self.n_local_heads = self.n_heads // configuration.num_devices
        self.n_local_kv_heads = self.n_kv_heads // configuration.num_devices

        self.dtype = dtype

        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4
        self.compute_kernel_config_sdpa = configuration.compute_kernel_config_sdpa
        self.sdpa_cfg = configuration.sdpa_cfg if hasattr(configuration, "sdpa_cfg") else None
        self.configuration = configuration

        self.model_config = configuration.get_model_config()

        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{state_dict_prefix}{name}")

        wq_str = f"{state_dict_prefix}wq.weight"
        wk_str = f"{state_dict_prefix}wk.weight"
        wv_str = f"{state_dict_prefix}wv.weight"
        wo_str = f"{state_dict_prefix}wo.weight"

        bq_str = f"{state_dict_prefix}wq.bias"
        bk_str = f"{state_dict_prefix}wk.bias"
        bv_str = f"{state_dict_prefix}wv.bias"
        bo_str = f"{state_dict_prefix}wo.bias"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % configuration.num_devices == 0
        assert self.n_kv_heads % configuration.num_devices == 0

        # Pad head_dim to multiple of 32
        def pad_head_dim(weight, heads_out=True):
            # Pad head dim to multiple of 32
            # heads_out means that the output dim of this weight contains heads.
            dim = weight.shape[1]
            assert weight.shape[0] == dim
            padded_head_dim = nearest_32(self.head_dim)
            padding_size = padded_head_dim - self.head_dim
            if padding_size > 0:
                if heads_out:
                    weight = weight.transpose(-1, -2)
                weight = weight.reshape(dim, self.n_heads, self.head_dim)
                padding = torch.zeros(dim, self.n_heads, padding_size, dtype=weight.dtype)
                weight = torch.cat([weight, padding], dim=-1)
                weight = weight.reshape(dim, self.n_heads * padded_head_dim)
                if heads_out:
                    weight = weight.transpose(-1, -2)
            return weight

        def pad_bias(bias):
            bias = bias.view(1, -1)
            padded_head_dim = nearest_32(self.head_dim)
            padding_size = padded_head_dim - self.head_dim
            if padding_size > 0:
                bias = bias.reshape(1, self.n_heads, self.head_dim)
                padding = torch.zeros(1, self.n_heads, padding_size, dtype=bias.dtype)
                bias = torch.cat([bias, padding], dim=-1)
                bias = bias.reshape(self.n_heads * padded_head_dim)
            return bias

        wq_padded = pad_head_dim(state_dict[wq_str])
        wk_padded = pad_head_dim(state_dict[wk_str])
        wv_padded = pad_head_dim(state_dict[wv_str])
        wo_padded = pad_head_dim(state_dict[wo_str], heads_out=False)

        bq_padded = pad_bias(state_dict[bq_str]) if state_dict.get(bq_str) is not None else None
        bk_padded = pad_bias(state_dict[bk_str]) if state_dict.get(bk_str) is not None else None
        bv_padded = pad_bias(state_dict[bv_str]) if state_dict.get(bv_str) is not None else None
        bo_padded = state_dict[bo_str] if state_dict.get(bo_str) is not None else None

        wq_chunked, wk_chunked, wv_chunked = (
            torch.chunk(w, configuration.num_devices) for w in [wq_padded, wk_padded, wv_padded]
        )

        if bq_padded is not None:
            bq_chunked, bk_chunked, bv_chunked = (
                torch.chunk(b, configuration.num_devices) for b in [bq_padded, bk_padded, bv_padded]
            )

        self.wqkv = ttnn.as_tensor(
            torch.concat(
                [
                    torch.concat(
                        [
                            torch.transpose(
                                wq_chunked[i],
                                -2,
                                -1,
                            ),
                            torch.transpose(
                                wk_chunked[i],
                                -2,
                                -1,
                            ),
                            torch.transpose(
                                wv_chunked[i],
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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wqkv_sharded"),
        )
        if bq_padded is not None:
            self.wqkv_bias = ttnn.as_tensor(
                torch.concat(
                    [
                        torch.concat(
                            [bq_chunked[i], bk_chunked[i], bv_chunked[i]],
                            dim=-1,
                        )
                        for i in range(configuration.num_devices)
                    ],
                    dim=-1,
                ),
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wqkv_bias_sharded"),
            )
            self.wqkv_bias = ttnn.reshape(
                self.wqkv_bias,
                (1, 1, 1, self.wqkv_bias.shape[-1]),
                (1, 1, self.wqkv_bias.shape[-2], self.wqkv_bias.shape[-1]),
            )
        else:
            self.wqkv_bias = None

        self.wo = ttnn.as_tensor(
            torch.transpose(
                wo_padded,
                -2,
                -1,
            ),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wo_sharded"),
        )

        if bo_padded is not None:
            self.bo = ttnn.as_tensor(
                bo_padded,
                device=self.mesh_device,
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("bo_sharded"),
            )
        else:
            self.bo = None

        self.scale = self.head_dim**-0.5

    def forward(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]

        MAX_MM_SEQ_LEN = self.configuration.VISION_MAX_MM_SEQ
        num_chunks = seq_len // MAX_MM_SEQ_LEN

        if seq_len > MAX_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            bias=self.wqkv_bias,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["IMAGE_ATTN_QKV_PROGCFG"](seq_len, MAX_MM_SEQ_LEN),
        )
        if seq_len > MAX_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        # split qkv into heads
        (
            q_heads_1QSD,
            k_heads_1KSD,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        ttnn.deallocate(xqkv_fused)
        # TODO: get this from model_config
        sdpa_cfg = self.sdpa_cfg or ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=32 * num_chunks,
            k_chunk_size=32 * num_chunks,
            exp_approx_mode=False,
        )

        attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_1QSD,
            k_heads_1KSD,
            v_heads_1VSD,
            is_causal=False,
            scale=self.scale,
            attn_mask=mask,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config_sdpa,
        )
        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD)
        ttnn.deallocate(k_heads_1KSD)
        ttnn.deallocate(v_heads_1VSD)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        # reshaping long sequence to matmul fit on device
        if seq_len > MAX_MM_SEQ_LEN:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            bias=self.bo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["IMAGE_ATTN_OUT_PROGCFG"](seq_len, MAX_MM_SEQ_LEN),
        )
        if seq_len > MAX_MM_SEQ_LEN:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # All reduce
        if self.num_devices > 1:  # replace with reduce_scatter and all_gather
            dense_out_gathered = ttnn.experimental.all_gather_async(
                output_11SH,
                persistent_output_buffer=None,
                dim=1,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=ttnn.Topology.Linear,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )
            dense_out_reduced = ttnn.experimental.fast_reduce_nc(
                dense_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            return dense_out_reduced
        else:
            return output_11SH

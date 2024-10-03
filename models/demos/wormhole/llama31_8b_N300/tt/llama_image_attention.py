# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule


class TtLlamaImageAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
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

        self.model_config = configuration.get_model_config()

        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{state_dict_prefix}.{name}")

        wq_str = f"{state_dict_prefix}wq.weight"
        wk_str = f"{state_dict_prefix}wk.weight"
        wv_str = f"{state_dict_prefix}wv.weight"
        wo_str = f"{state_dict_prefix}wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % configuration.num_devices == 0
        assert self.n_kv_heads % configuration.num_devices == 0

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wqkv_sharded"),
        )

        self.wo = ttnn.as_tensor(
            torch.transpose(
                self.state_dict[wo_str],
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
        self.attention_grid = ttnn.CoreCoord(8, 4)
        self.scale = self.head_dim**-0.5

    def forward(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 1024, 1024, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.model_config["IMAGE_ATTN_QKV_PROGCFG"](seq_len),
        )
        if seq_len > 1024:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

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

        attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_1QSD,
            k_heads_1KSD,
            v_heads_1VSD,
            is_causal=True,
            scale=self.scale,
            program_config=self.model_config["SDPA_PROGCFG"](seq_len),
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
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["IMAGE_ATTN_OUT_PROGCFG"](seq_len),
        )
        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # All reduce
        if self.num_devices > 1:
            dense_out_gathered = ttnn.all_gather(output_11SH, dim=1, num_links=1, topology=ttnn.Topology.Linear)
            dense_out_reduced = ttnn.experimental.fast_reduce_nc(
                dense_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            return dense_out_reduced
        else:
            return output_11SH

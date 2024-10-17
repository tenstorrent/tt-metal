# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule
import os
from loguru import logger


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
        self.compute_kernel_config_sdpa = configuration.compute_kernel_config_sdpa

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

        wq_padded = pad_head_dim(self.state_dict[wq_str])
        wk_padded = pad_head_dim(self.state_dict[wk_str])
        wv_padded = pad_head_dim(self.state_dict[wv_str])
        wo_padded = pad_head_dim(self.state_dict[wo_str], heads_out=False)
        wq_chunked, wk_chunked, wv_chunked = (
            torch.chunk(w, configuration.num_devices) for w in [wq_padded, wk_padded, wv_padded]
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

        self.scale = self.head_dim**-0.5

    def forward(self, x, mask):
        return self.forward_tt(x, mask)
        if os.environ.get("ATTN") == "tt":
            return self.forward_tt(x, mask)
        else:
            return self.forward_pt(x, mask)

    def forward_pt(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        x_torch = ttnn.to_torch(x_11SH, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        x_torch = x_torch[0].unsqueeze(0)
        wqkv = ttnn.to_torch(self.wqkv, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        wqkv = wqkv.view(2, wqkv.shape[0] // 2, wqkv.shape[1])

        wqkv = wqkv.reshape(2, self.hidden_size, 3 * self.n_heads // 2, -1)
        wqkv = wqkv[..., : self.head_dim]
        # wqkv = wqkv.reshape(2, self.hidden_size, -1)
        wqkv = wqkv.reshape(2, self.hidden_size, 3, self.n_heads // 2, -1).permute(1, 2, 0, 3, 4)
        wqkv = wqkv.reshape(self.hidden_size, 3, self.n_heads, -1).reshape(self.hidden_size, -1)

        # xqkv_fused_torch = torch.matmul(x_torch, wqkv).bfloat16().float()
        xqkv_fused_torch = torch.nn.functional.linear(x_torch, wqkv.T).bfloat16().float()
        # xqkv_fused_torch = torch.nn.functional.linear(x_torch, wqkv.tranpose).bfloat16().float()
        logger.info(xqkv_fused_torch.shape)
        # n, s, d = xqkv_fused_torch.shape[-3:]
        s, d = xqkv_fused_torch.shape[-2:]
        xqkv = xqkv_fused_torch.reshape(s, 3, d // 3)
        q = xqkv[..., 0, :]
        k = xqkv[..., 1, :]
        v = xqkv[..., 2, :]
        # xq = q.reshape(n, s, self.n_heads//2, -1).transpose(1, 2)
        # xk = k.reshape(n, s, self.n_heads//2, -1).transpose(1, 2)
        # xv = v.reshape(n, s, self.n_heads//2, -1).transpose(1, 2)
        xq = q.reshape(s, self.n_heads, -1).transpose(0, 1)
        xk = k.reshape(s, self.n_heads, -1).transpose(0, 1)
        xv = v.reshape(s, self.n_heads, -1).transpose(0, 1)

        mask_torch = ttnn.to_torch(mask, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        mask_torch = mask_torch[0]
        attn_output = (
            torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask_torch, scale=self.scale)
            .bfloat16()
            .float()
        )  # [...,:self.head_dim]

        # attn_output = attn_output.transpose(1, 2).reshape(n, s, -1).transpose(0, 1).reshape(s, -1)
        attn_output = attn_output.transpose(0, 1).reshape(s, -1)

        wo = ttnn.to_torch(self.wo, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        wo = wo.view(2, wo.shape[0] // 2, wo.shape[1])  # .reshape(-1, wo.shape[1])

        wo = (
            wo.transpose(1, 2)
            .reshape(2, self.hidden_size, self.n_heads // 2, -1)[..., : self.head_dim]
            .reshape(2, self.hidden_size, -1)
            .transpose(1, 2)
            .reshape(-1, self.hidden_size)
        )

        out = torch.nn.functional.linear(attn_output, wo.T).bfloat16().float()
        # out = torch.sum(out, dim=0).unsqueeze(0).unsqueeze(0).bfloat16().float()
        out = out.view(1, 1, 4224, -1)

        # breakpoint()

        out_tt = ttnn.from_torch(
            out,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )
        return out_tt

    def forward_tt(self, x_11SH, mask=None):
        seq_len = x_11SH.shape[-2]
        # assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device

        # x_torch = ttnn.to_torch(x_11SH, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        # x_torch = x_torch.view(2, seq_len, -1)
        # wqkv = ttnn.to_torch(self.wqkv, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        # wqkv = wqkv.view(2, wqkv.shape[0]//2, wqkv.shape[1])

        # xqkv_fused_torch = torch.bmm(x_torch, wqkv).unsqueeze(1)
        # xqkv_fused = ttnn.from_torch(
        #     xqkv_fused_torch,
        #     device=self.mesh_device,
        #     mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        #     dtype=ttnn.bfloat16,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     layout=ttnn.TILE_LAYOUT,
        # )

        # Depends on whether we are padding or not
        MAX_MM_SEQ_LEN = 1056
        # MAX_MM_SEQ_LEN = 1024

        # DEBUG: Don't batch it up
        # MAX_MM_SEQ_LEN = 10000

        if seq_len > MAX_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["IMAGE_ATTN_QKV_PROGCFG"](seq_len, MAX_MM_SEQ_LEN),
        )
        if seq_len > MAX_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        # ttnn.deallocate(x_11SH)

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
        # sdpa_cfg = self.model_config["SDPA_PROGCFG"](seq_len)
        sdpa_cfg = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            q_chunk_size=128,
            k_chunk_size=128,
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

        # q_heads_torch = ttnn.to_torch(q_heads_1QSD, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        # k_heads_torch = ttnn.to_torch(k_heads_1KSD, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        # v_heads_torch = ttnn.to_torch(v_heads_1VSD, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        # mask_torch = ttnn.to_torch(mask, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()

        # attn_output_torch = torch.nn.functional.scaled_dot_product_attention(
        #     q_heads_torch,
        #     k_heads_torch,
        #     v_heads_torch,
        #     attn_mask=mask_torch,
        #     scale=self.scale,
        # )

        # attn_output_1QSD = ttnn.from_torch(
        #     attn_output_torch,
        #     device=self.mesh_device,
        #     mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=0),
        #     dtype=ttnn.bfloat16,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     layout=ttnn.TILE_LAYOUT,
        # )

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        # attn_output_torch = ttnn.to_torch(attn_output_11SH, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        # # breakpoint()
        # wo = ttnn.to_torch(self.wo, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh_device, dim=0)).float()
        # wo = wo.view(2, 1, wo.shape[0]//2, wo.shape[1])
        # output = torch.matmul(attn_output_torch, wo)
        # output = torch.sum(output, dim=0).unsqueeze(0).unsqueeze(0)

        # output_11SH = ttnn.from_torch(
        #     output,
        #     device=self.mesh_device,
        #     mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        #     dtype=ttnn.bfloat16,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     layout=ttnn.TILE_LAYOUT,
        # )
        # return output_11SH
        # breakpoint()

        # reshaping long sequence to matmul fit on device
        if seq_len > MAX_MM_SEQ_LEN:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // MAX_MM_SEQ_LEN, MAX_MM_SEQ_LEN, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["IMAGE_ATTN_OUT_PROGCFG"](seq_len, MAX_MM_SEQ_LEN),
        )
        if seq_len > MAX_MM_SEQ_LEN:
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

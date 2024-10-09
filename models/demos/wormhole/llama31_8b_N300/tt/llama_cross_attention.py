# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional
import torch

import ttnn
from models.utility_functions import (
    nearest_32,
)
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm


class TtLlamaCrossAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        state_dict_prefix,
        weight_cache_path,
        dtype,
        configuration,
        dim,
        head_dim,
        n_heads,
        n_kv_heads,
        norm_eps,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices

        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.norm_eps = norm_eps

        self.n_local_heads = self.n_heads // configuration.num_devices
        self.n_local_kv_heads = self.n_kv_heads // configuration.num_devices

        self.dtype = dtype

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

        self.wq = ttnn.as_tensor(
            self.state_dict[wq_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wq_sharded"),
        )

        self.wk = ttnn.as_tensor(
            self.state_dict[wk_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wk_sharded"),
        )

        self.wv = ttnn.as_tensor(
            self.state_dict[wv_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wv_sharded"),
        )

        self.wo = ttnn.as_tensor(
            self.state_dict[wo_str].transpose(-2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=cache_name("wo_sharded"),
        )

        self.scale = self.head_dim**-0.5

        self.q_norm = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}",
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_dtype=dtype,
            weight_key="q_norm",
            eps=self.norm_eps,
        )

        self.k_norm = RMSNorm(
            device=mesh_device,
            dim=self.head_dim,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}",
            weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
            weight_dtype=dtype,
            weight_key="k_norm",
            eps=self.norm_eps,
        )

    def compute_xattn_kv_cache(self, xattn_tokens):
        bsz, seqlen_y = xattn_tokens.shape[1], xattn_tokens.shape[2]
        if seqlen_y > 1024:
            xattn_tokens = ttnn.reshape(xattn_tokens, [1, bsz * seqlen_y // 1024, 1024, -1])

        # breakpoint()
        xk = ttnn.linear(
            xattn_tokens,
            self.wk,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_KV_PROGCFG"](seqlen_y),
        )
        # return xk

        xv = ttnn.linear(
            xattn_tokens,
            self.wv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_KV_PROGCFG"](seqlen_y),
        )
        # return xv
        if seqlen_y > 1024:
            xk = ttnn.reshape(xk, [1, bsz, seqlen_y, -1])
            xv = ttnn.reshape(xv, [1, bsz, seqlen_y, -1])

        # breakpoint()
        xk, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xk, xk, num_heads=self.n_local_kv_heads, num_kv_heads=self.n_local_kv_heads // 2, transpose_k_heads=False
        )
        xv, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xv, xv, num_heads=self.n_local_kv_heads, num_kv_heads=self.n_local_kv_heads // 2, transpose_k_heads=False
        )

        xk = self.k_norm(xk)
        return [xk, xv]

        # breakpoint()
        # return xk

        ### EVERYTHING BELOW IS BROKEN OMG
        # BEWARNED! TMs are dangerous!
        # WORKAROUND
        # breakpoint()
        xk = ttnn.to_layout(xk, layout=ttnn.ROW_MAJOR_LAYOUT)
        xv = ttnn.to_layout(xv, layout=ttnn.ROW_MAJOR_LAYOUT)

        xk = xk.reshape(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        xv = xv.reshape(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        # breakpoint()
        # xk = ttnn.to_memory_config(xk, ttnn.L1_MEMORY_CONFIG)
        # xk = ttnn.to_memory_config(xk, ttnn.DRAM_MEMORY_CONFIG)
        return xk

        xk = ttnn.transpose(xk, 1, 2, memory_config=ttnn.L1_MEMORY_CONFIG)
        xv = ttnn.transpose(xv, 1, 2)

        xk = ttnn.to_layout(xk, layout=ttnn.TILE_LAYOUT)
        xv = ttnn.to_layout(xv, layout=ttnn.TILE_LAYOUT)

        # PREFERRED METHOD
        # xk = xk.reshape(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        # xv = xv.reshape(bsz, seqlen_y, self.n_local_kv_heads, self.head_dim)
        # xk, xv = [ttnn.transpose(tensor, 1, 2) for tensor in (xk, xv)] # HANG!
        return [xk, xv]

    def forward(self, x_11SH, xattn_mask, full_text_row_masked_out_mask, xattn_cache):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        if seq_len > 1024:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 1024, 1024, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
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
        sdpa_cfg = self.model_config["SDPA_PROGCFG"](seq_len)
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
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
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

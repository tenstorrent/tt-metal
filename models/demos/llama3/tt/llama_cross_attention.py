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

        xk = ttnn.linear(
            xattn_tokens,
            self.wk,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_KV_PROGCFG"](seqlen_y),
        )

        xv = ttnn.linear(
            xattn_tokens,
            self.wv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_KV_PROGCFG"](seqlen_y),
        )
        if seqlen_y > 1024:
            xk = ttnn.reshape(xk, [1, bsz, seqlen_y, -1])
            xv = ttnn.reshape(xv, [1, bsz, seqlen_y, -1])

        xk, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xk, xk, num_heads=self.n_local_kv_heads, num_kv_heads=self.n_local_kv_heads // 2, transpose_k_heads=False
        )
        xv, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xv, xv, num_heads=self.n_local_kv_heads, num_kv_heads=self.n_local_kv_heads // 2, transpose_k_heads=False
        )

        xk = self.k_norm(xk)
        return [xk, xv]

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

    def forward(self, x_11SH, xattn_mask, full_text_row_masked_out_mask_1NSH, xattn_cache, mode):
        seq_len = x_11SH.shape[-2]
        # assert seq_len % 32 == 0 and seq_len > 0, "Seqlen must be divisible by 128"

        if seq_len > 1024:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 1024, 1024, -1])

        xq = ttnn.linear(
            x_11SH,
            self.wq,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_Q_PROGCFG"](seq_len),
        )
        if seq_len > 1024:
            xq = ttnn.reshape(xq, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

        xq, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            xq, xq, num_heads=self.n_local_heads, num_kv_heads=self.n_local_heads // 2, transpose_k_heads=False
        )

        xq = self.q_norm(xq)

        xk, xv = xattn_cache
        cache_seq_len = xk.shape[-2]

        # NOTE: Using naive SDPA for now since FlashDecode does not allow non-causal mask
        # xq = ttnn.reshape(xq, [self.n_local_heads // self.n_local_kv_heads, self.n_local_kv_heads, seq_len, self.head_dim])
        # NOTE: repeat doesn't work, need to use repeat_interleave
        # xk = ttnn.repeat(xk, ttnn.Shape((self.n_local_heads // self.n_local_kv_heads, 1, 1, 1)))
        xk = ttnn.repeat_interleave(xk, self.n_local_heads // self.n_local_kv_heads, dim=1)
        # xv = ttnn.repeat(xv, ttnn.Shape((self.n_local_heads // self.n_local_kv_heads, 1, 1, 1)))
        xv = ttnn.repeat_interleave(xv, self.n_local_heads // self.n_local_kv_heads, dim=1)

        scores = ttnn.matmul(
            xq,
            ttnn.transpose(xk, -1, -2),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_SCORE_PROGCFG"](seq_len, cache_seq_len),
        )

        scores = ttnn.multiply(scores, self.scale)
        # WARNING: This add is buggy if xattn_mask has to be broadcasted to n_local_heads. Workaround is to broadcast on host side
        scores = ttnn.add(scores, xattn_mask)
        scores = ttnn.softmax(scores, dim=-1, numeric_stable=True)

        # TODO: scale_mask_softmax doesn't work for this xattn_mask shape
        # scores = ttnn.scale_mask_softmax(scores, self.scale, xattn_mask, numeric_stable=True)
        output = ttnn.matmul(
            scores,
            xv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            program_config=self.model_config["VISION_XATTN_OUTPUT_PROGCFG"](seq_len, cache_seq_len),
        )

        # WARNING: this broadcast is also broken, must broadcast on host
        output = ttnn.mul(output, full_text_row_masked_out_mask_1NSH)

        output = ttnn.experimental.nlp_concat_heads(output)
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, seq_len // 1024, 1024, -1])

        output = ttnn.matmul(
            output,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi4,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["VISION_XATTN_DENSE_PROGCFG"](seq_len),
        )
        if seq_len > 1024:
            output = ttnn.reshape(output, [1, 1, seq_len, -1])

        # All reduce
        if self.num_devices > 1:
            dense_out_gathered = ttnn.all_gather(output, dim=1, num_links=1, topology=ttnn.Topology.Linear)
            dense_out_reduced = ttnn.experimental.fast_reduce_nc(
                dense_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            return dense_out_reduced
        else:
            return output

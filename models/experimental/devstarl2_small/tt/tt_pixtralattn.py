# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0
"""
Modified vision attention for Mistral-Small / Pixtral-class checkpoints.
Uses ``apply_rotary_pos_emb_vision_tt`` for Devstral-compatible RoPE.
"""

import os

import torch
import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import is_blackhole, nearest_32
from models.experimental.devstarl2_small.tt.tt_pixtral_seq_chunk import pixtral_vision_seq_chunk_len


def _pixtral_sdpa_qk_chunk_sizes() -> tuple[int, int]:
    """
    Tile chunk sizes for :func:`scaled_dot_product_attention` program config.

    Must stay **small** for L1: scaling by ``num_matmul_chunks`` (``seq_len / max_mm_seq_len``) is wrong here
    and drives ``q_chunk_size`` into the thousands when vision sequences are long.
    """
    q = int(os.environ.get("PIXTRAL_SDPA_Q_CHUNK", "32"))
    k = int(os.environ.get("PIXTRAL_SDPA_K_CHUNK", "32"))
    q = max(32, min(q, 128))
    k = max(32, min(k, 128))
    return q, k


def rotate_half(x):
    last_dim = x.shape[-1]
    half = last_dim // 2

    x1 = ttnn.slice(x, (0, 0, 0, 0), (x.shape[0], x.shape[1], x.shape[2], half))
    x2 = ttnn.slice(x, (0, 0, 0, half), (x.shape[0], x.shape[1], x.shape[2], last_dim))

    neg_x2 = ttnn.mul(x2, -1, use_legacy=False)
    return ttnn.concat([neg_x2, x1], dim=-1)


def apply_rotary_pos_emb_vision_tt(q, k, cos, sin):
    cos = ttnn.unsqueeze(cos, 0)
    sin = ttnn.unsqueeze(sin, 0)

    q_embed = ttnn.add(ttnn.mul(q, cos, use_legacy=None), ttnn.mul(rotate_half(q), sin, use_legacy=None))
    k_embed = ttnn.add(ttnn.mul(k, cos), ttnn.mul(rotate_half(k), sin))
    return q_embed, k_embed


class TtMistralImageAttention(LightweightModule):
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

        self.state_dict = state_dict
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
        self.compute_kernel_config_sdpa = configuration.compute_kernel_config_sdpa
        self.configuration = configuration

        self.model_config = configuration.get_model_config()

        wq_str = f"{state_dict_prefix}wq.weight"
        wk_str = f"{state_dict_prefix}wk.weight"
        wv_str = f"{state_dict_prefix}wv.weight"
        wo_str = f"{state_dict_prefix}wo.weight"

        assert self.n_heads % configuration.num_devices == 0
        assert self.n_kv_heads % configuration.num_devices == 0

        def pad_head_dim(weight, heads_out=True):
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
                            torch.transpose(wq_chunked[i], -2, -1),
                            torch.transpose(wk_chunked[i], -2, -1),
                            torch.transpose(wv_chunked[i], -2, -1),
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
        )

        self.wo = ttnn.as_tensor(
            torch.transpose(wo_padded, -2, -1),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
        )

        self.scale = self.head_dim**-0.5

    def _linear_qkv_seq_chunked(self, x_11SH, seq_len: int, max_mm_seq_len: int) -> ttnn.Tensor:
        """Fused QKV ``ttnn.linear`` over the sequence axis; chunk so matmul ``m`` fits L1 CB budget."""
        hidden_w = x_11SH.shape[-1]
        if seq_len <= max_mm_seq_len:
            return ttnn.linear(
                x_11SH,
                self.wqkv,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=self.model_config["IMAGE_ATTN_QKV_PROGCFG"](seq_len, seq_len),
            )
        if seq_len % max_mm_seq_len == 0:
            x_batched = ttnn.reshape(x_11SH, [1, seq_len // max_mm_seq_len, max_mm_seq_len, -1])
            return ttnn.linear(
                x_batched,
                self.wqkv,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=self.model_config["IMAGE_ATTN_QKV_PROGCFG"](seq_len, max_mm_seq_len),
            )

        parts: list[ttnn.Tensor] = []
        start = 0
        while start < seq_len:
            end = min(start + max_mm_seq_len, seq_len)
            clen = end - start
            x_chunk = ttnn.slice(x_11SH, (0, 0, start, 0), (1, 1, end, hidden_w))
            parts.append(
                ttnn.linear(
                    x_chunk,
                    self.wqkv,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    program_config=self.model_config["IMAGE_ATTN_QKV_PROGCFG"](clen, clen),
                )
            )
            ttnn.deallocate(x_chunk)
            start = end
        if len(parts) == 1:
            return parts[0]
        out = ttnn.concat(parts, dim=2)
        for p in parts:
            ttnn.deallocate(p)
        return out

    def _linear_wo_seq_chunked(self, attn_output_11SH, seq_len: int, max_mm_seq_len: int) -> ttnn.Tensor:
        """Output ``wo`` linear with the same chunking."""
        hidden_w = attn_output_11SH.shape[-1]
        if seq_len <= max_mm_seq_len:
            return ttnn.linear(
                attn_output_11SH,
                self.wo,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.model_config["IMAGE_ATTN_OUT_PROGCFG"](seq_len, seq_len),
            )
        if seq_len % max_mm_seq_len == 0:
            x_batched = ttnn.reshape(attn_output_11SH, [1, seq_len // max_mm_seq_len, max_mm_seq_len, -1])
            return ttnn.linear(
                x_batched,
                self.wo,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.model_config["IMAGE_ATTN_OUT_PROGCFG"](seq_len, max_mm_seq_len),
            )

        parts: list[ttnn.Tensor] = []
        start = 0
        while start < seq_len:
            end = min(start + max_mm_seq_len, seq_len)
            clen = end - start
            x_chunk = ttnn.slice(attn_output_11SH, (0, 0, start, 0), (1, 1, end, hidden_w))
            parts.append(
                ttnn.linear(
                    x_chunk,
                    self.wo,
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    program_config=self.model_config["IMAGE_ATTN_OUT_PROGCFG"](clen, clen),
                )
            )
            ttnn.deallocate(x_chunk)
            start = end
        if len(parts) == 1:
            return parts[0]
        out = ttnn.concat(parts, dim=2)
        for p in parts:
            ttnn.deallocate(p)
        return out

    def forward(self, x_11SH, position_embeddings=None):
        # DRAM residency before fused matmuls; L1 circular buffers inside the kernel still scale with M.
        x_11SH = ttnn.to_memory_config(x_11SH, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        seq_len = int(x_11SH.shape[-2])
        max_mm_seq_len = pixtral_vision_seq_chunk_len(self.configuration)

        xqkv_fused = self._linear_qkv_seq_chunked(x_11SH, seq_len, max_mm_seq_len)
        if seq_len > max_mm_seq_len and seq_len % max_mm_seq_len == 0:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

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

        if position_embeddings is not None:
            cos, sin = position_embeddings
            q_heads_1QSD, k_heads_1KSD = apply_rotary_pos_emb_vision_tt(q_heads_1QSD, k_heads_1KSD, cos, sin)
        ttnn.deallocate(xqkv_fused)

        if seq_len > max_mm_seq_len:
            q_tile, k_tile = _pixtral_sdpa_qk_chunk_sizes()
            sdpa_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=q_tile,
                k_chunk_size=k_tile,
                exp_approx_mode=False,
            )
        else:
            sdpa_cfg = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=(8, 8),
                q_chunk_size=128,
                k_chunk_size=128,
                exp_approx_mode=False,
            )
        attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_1QSD,
            k_heads_1KSD,
            v_heads_1VSD,
            is_causal=False,
            scale=self.scale,
            program_config=sdpa_cfg,
            compute_kernel_config=self.compute_kernel_config_sdpa,
        )
        ttnn.deallocate(q_heads_1QSD)
        ttnn.deallocate(k_heads_1KSD)
        ttnn.deallocate(v_heads_1VSD)

        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        attn_output_11SH = ttnn.to_memory_config(attn_output_11SH, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(attn_output_1QSD)

        output_11SH = self._linear_wo_seq_chunked(attn_output_11SH, seq_len, max_mm_seq_len)
        if seq_len > max_mm_seq_len and seq_len % max_mm_seq_len == 0:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        if self.num_devices > 1:
            if is_blackhole():
                dense_out_gathered = ttnn.all_gather(output_11SH, dim=1, num_links=1, topology=ttnn.Topology.Linear)
            else:
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
            output_11SH.deallocate(True)
            dense_out_reduced = ttnn.experimental.fast_reduce_nc(
                dense_out_gathered, dims=[1], output=None, compute_kernel_config=None
            )
            dense_out_reduced = dense_out_reduced[:, :, : dense_out_gathered.shape[-2], :]
            return dense_out_reduced
        return output_11SH


__all__ = ["TtMistralImageAttention", "apply_rotary_pos_emb_vision_tt", "rotate_half"]

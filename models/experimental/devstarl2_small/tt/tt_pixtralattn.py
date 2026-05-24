# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
# Pixtral vision attention with Devstral-compatible vision RoPE.

import os

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.experimental.devstarl2_small.devstral_utils.vision_ccl import vision_sum_all_reduce
from models.experimental.devstarl2_small.devstral_utils.pixtral_seq_chunk import (
    pad_seq_to_chunk_multiple,
    pixtral_vision_seq_chunk_len,
    trim_seq_dim2,
    vision_rope_memcfg,
)


def _pixtral_sdpa_grid_size(configuration) -> tuple[int, int]:
    grid = configuration.max_grid_size
    if hasattr(grid, "x") and hasattr(grid, "y"):
        return (int(grid.x), int(grid.y))
    if isinstance(grid, (tuple, list)) and len(grid) >= 2:
        return (int(grid[0]), int(grid[1]))
    return (8, 8)


def _pixtral_sdpa_program_config(
    seq_len: int, max_mm_seq_len: int, grid_size: tuple[int, int]
) -> ttnn.SDPAProgramConfig:
    """SDPA tiles scale with matmul seq chunks (same policy as ``llama_image_attention``)."""
    force_q = os.environ.get("PIXTRAL_SDPA_Q_CHUNK")
    if force_q is not None and str(force_q).strip() != "":
        q_chunk = max(32, min(256, nearest_32(int(force_q))))
        force_k = os.environ.get("PIXTRAL_SDPA_K_CHUNK")
        k_chunk = max(32, min(256, nearest_32(int(force_k or force_q))))
        return ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=q_chunk,
            k_chunk_size=k_chunk,
            exp_approx_mode=False,
        )

    if seq_len < 2048:
        chunk = 128
    else:
        num_chunks = max(1, (seq_len + max_mm_seq_len - 1) // max_mm_seq_len)
        chunk = min(256, max(64, nearest_32(32 * num_chunks)))
    return ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=grid_size,
        q_chunk_size=chunk,
        k_chunk_size=chunk,
        exp_approx_mode=False,
    )


def apply_rotary_pos_emb_vision_tt(q, k, cos, sin):
    seq_len = int(q.shape[2])
    head_dim = int(q.shape[-1])
    rope_mem_cfg = vision_rope_memcfg(seq_len, head_dim)
    cos = ttnn.unsqueeze(cos, 0)
    sin = ttnn.unsqueeze(sin, 0)

    def _rope_mem(t: ttnn.Tensor) -> ttnn.Tensor:
        if t.memory_config().buffer_type != rope_mem_cfg.buffer_type:
            return ttnn.to_memory_config(t, rope_mem_cfg)
        return t

    q = _rope_mem(q)
    k = _rope_mem(k)
    cos = _rope_mem(cos)
    sin = _rope_mem(sin)

    q_embed = ttnn.experimental.rotary_embedding(q, cos, sin, memory_config=rope_mem_cfg)
    k_embed = ttnn.experimental.rotary_embedding(k, cos, sin, memory_config=rope_mem_cfg)

    if q_embed.shape[2] != seq_len:
        q_embed = q_embed[:, :, :seq_len, :]
    if k_embed.shape[2] != seq_len:
        k_embed = k_embed[:, :, :seq_len, :]
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

        if self.n_heads % configuration.num_devices != 0:
            raise ValueError(f"n_heads {self.n_heads} must divide num_devices {configuration.num_devices}")
        if self.n_kv_heads % configuration.num_devices != 0:
            raise ValueError(f"n_kv_heads {self.n_kv_heads} must divide num_devices {configuration.num_devices}")

        def pad_head_dim(weight, heads_out=True):
            dim = weight.shape[1]
            assert weight.shape[0] == dim
            padded_head_dim = nearest_32(self.head_dim)
            padding_size = padded_head_dim - self.head_dim
            if padding_size > 0:
                if heads_out:
                    weight = weight.transpose(-1, -2)
                weight = weight.reshape(dim, self.n_heads, self.head_dim)
                padded = weight.new_zeros((dim, self.n_heads, padded_head_dim))
                padded[:, :, : self.head_dim] = weight
                weight = padded
                weight = weight.reshape(dim, self.n_heads * padded_head_dim)
                if heads_out:
                    weight = weight.transpose(-1, -2)
            return weight

        wq_padded = pad_head_dim(state_dict[wq_str])
        wk_padded = pad_head_dim(state_dict[wk_str])
        wv_padded = pad_head_dim(state_dict[wv_str])
        wo_padded = pad_head_dim(state_dict[wo_str], heads_out=False)

        def pack_qkv_for_sharding(wq, wk, wv):
            local_width = wq.shape[0] // configuration.num_devices
            packed = wq.new_empty((configuration.num_devices, self.hidden_size, local_width * 3))
            for index, weight in enumerate((wq, wk, wv)):
                start = index * local_width
                packed[:, :, start : start + local_width] = weight.reshape(
                    configuration.num_devices, local_width, self.hidden_size
                ).transpose(-1, -2)
            return packed.transpose(0, 1).reshape(self.hidden_size, -1)

        wqkv_cache = None if weight_cache_path is None else weight_cache_path / f"{state_dict_prefix}wqkv.weight"
        wo_cache = None if weight_cache_path is None else weight_cache_path / f"{state_dict_prefix}wo.weight"

        self.wqkv = ttnn.as_tensor(
            pack_qkv_for_sharding(wq_padded, wk_padded, wv_padded),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=wqkv_cache,
        )

        self.wo = ttnn.as_tensor(
            wo_padded.transpose(-1, -2),
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-2),
            dtype=self.dtype,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.TILE_LAYOUT,
            cache_file_name=wo_cache,
        )

        self.scale = self.head_dim**-0.5

    def _linear_qkv_seq_chunked(self, x_11SH, seq_len: int, max_mm_seq_len: int) -> ttnn.Tensor:
        """Fused QKV ``ttnn.linear`` over the sequence axis; chunk so matmul ``m`` fits L1 CB budget."""
        x_11SH, seq_len, original_seq_len = pad_seq_to_chunk_multiple(x_11SH, seq_len, max_mm_seq_len)
        if seq_len <= max_mm_seq_len:
            out = ttnn.linear(
                x_11SH,
                self.wqkv,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=self.model_config["IMAGE_ATTN_QKV_PROGCFG"](seq_len, seq_len),
            )
            return trim_seq_dim2(out, original_seq_len)

        x_batched = ttnn.reshape(x_11SH, [1, seq_len // max_mm_seq_len, max_mm_seq_len, -1])
        out = ttnn.linear(
            x_batched,
            self.wqkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.model_config["IMAGE_ATTN_QKV_PROGCFG"](seq_len, max_mm_seq_len),
        )
        out = ttnn.reshape(out, [1, 1, seq_len, -1])
        return trim_seq_dim2(out, original_seq_len)

    def _linear_wo_seq_chunked(
        self,
        attn_output_11SH,
        seq_len: int,
        max_mm_seq_len: int,
        output_memory_config=None,
    ) -> ttnn.Tensor:
        """Output ``wo`` linear with the same chunking."""
        attn_output_11SH, seq_len, original_seq_len = pad_seq_to_chunk_multiple(
            attn_output_11SH, seq_len, max_mm_seq_len
        )
        wo_mem_cfg = output_memory_config if output_memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
        if seq_len <= max_mm_seq_len:
            out = ttnn.linear(
                attn_output_11SH,
                self.wo,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                dtype=ttnn.bfloat16,
                memory_config=wo_mem_cfg,
                program_config=self.model_config["IMAGE_ATTN_OUT_PROGCFG"](seq_len, seq_len),
            )
            return trim_seq_dim2(out, original_seq_len)

        x_batched = ttnn.reshape(attn_output_11SH, [1, seq_len // max_mm_seq_len, max_mm_seq_len, -1])
        out = ttnn.linear(
            x_batched,
            self.wo,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            dtype=ttnn.bfloat16,
            memory_config=wo_mem_cfg,
            program_config=self.model_config["IMAGE_ATTN_OUT_PROGCFG"](seq_len, max_mm_seq_len),
        )
        out = ttnn.reshape(out, [1, 1, seq_len, -1])
        return trim_seq_dim2(out, original_seq_len)

    def forward(self, x_11SH, position_embeddings=None):
        if x_11SH.memory_config().buffer_type != ttnn.BufferType.DRAM:
            x_11SH = ttnn.to_memory_config(x_11SH, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        seq_len = int(x_11SH.shape[-2])
        max_mm_seq_len = pixtral_vision_seq_chunk_len(self.configuration)
        wo_out_mem_cfg = ttnn.DRAM_MEMORY_CONFIG if self.num_devices > 1 else None

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

        sdpa_cfg = _pixtral_sdpa_program_config(seq_len, max_mm_seq_len, _pixtral_sdpa_grid_size(self.configuration))
        attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
            q_heads_1QSD,
            k_heads_1KSD,
            v_heads_1VSD,
            is_causal=False,
            scale=self.scale,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
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
        ttnn.deallocate(attn_output_1QSD)

        output_11SH = self._linear_wo_seq_chunked(
            attn_output_11SH,
            seq_len,
            max_mm_seq_len,
            output_memory_config=wo_out_mem_cfg,
        )
        if seq_len > max_mm_seq_len and seq_len % max_mm_seq_len == 0:
            if not (len(output_11SH.shape) == 4 and int(output_11SH.shape[0]) == 1 and int(output_11SH.shape[1]) == 1):
                output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        if self.num_devices > 1:
            if not (len(output_11SH.shape) == 4 and int(output_11SH.shape[0]) == 1 and int(output_11SH.shape[1]) == 1):
                output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
            return vision_sum_all_reduce(
                output_11SH,
                self.mesh_device,
                self.tt_ccl,
                seq_len,
                self.hidden_size,
                self.configuration,
            )
        return output_11SH


__all__ = ["TtMistralImageAttention", "apply_rotary_pos_emb_vision_tt"]

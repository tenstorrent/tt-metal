# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Source lineage: HuggingFace PatchTST and PatchTST paper implementation details
# - https://huggingface.co/docs/transformers/en/model_doc/patchtst
# - https://github.com/huggingface/transformers/tree/main/src/transformers/models/patchtst
# - https://arxiv.org/abs/2211.14730

from __future__ import annotations

from dataclasses import dataclass

import torch

import ttnn
from models.demos.wormhole.patchtst.tt.common import (
    MEMORY_CONFIG_BY_TIER,
    PatchTSTRuntimePolicy,
    TTLinear,
    build_linear,
    build_linear_from_state,
    ensure_interleaved,
)


@dataclass
class PatchTSTAttention:
    qkv: TTLinear
    out_proj: TTLinear
    num_heads: int
    padded_head_dim: int

    def __call__(
        self, hidden_state: ttnn.Tensor, runtime: PatchTSTRuntimePolicy, device, dtype: ttnn.DataType = ttnn.bfloat16
    ) -> ttnn.Tensor:
        mem_cfg = MEMORY_CONFIG_BY_TIER[runtime.activation_memory_tier]
        hidden_size = int(hidden_state.shape[-1])
        real_head_dim = hidden_size // self.num_heads
        if real_head_dim * self.num_heads != hidden_size:
            raise ValueError(
                f"Hidden size must be divisible by num_heads: hidden={hidden_size}, heads={self.num_heads}"
            )
        qkv = ensure_interleaved(
            ttnn.linear(hidden_state, self.qkv.weight, bias=self.qkv.bias, memory_config=mem_cfg, dtype=dtype),
            mem_cfg,
        )
        query, key, value = ttnn.transformer.split_query_key_value_and_split_heads(
            qkv, memory_config=mem_cfg, num_heads=self.num_heads, transpose_key=False
        )
        ttnn.deallocate(qkv)

        if runtime.use_sdpa:
            compute_grid = device.compute_with_storage_grid_size()
            context = ttnn.transformer.scaled_dot_product_attention(
                query,
                key,
                value,
                scale=1.0 / (real_head_dim**0.5),
                attn_mask=None,
                is_causal=False,
                program_config=ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(compute_grid.x, compute_grid.y),
                    q_chunk_size=int(runtime.sdpa_q_chunk_size),
                    k_chunk_size=int(runtime.sdpa_k_chunk_size),
                    exp_approx_mode=True,
                ),
                memory_config=mem_cfg,
            )
        else:
            key_t = ttnn.permute(key, (0, 1, 3, 2))
            attn_scores = ttnn.matmul(query, key_t, memory_config=mem_cfg, dtype=dtype)
            ttnn.deallocate(key_t)
            attn_probs = ttnn.transformer.attention_softmax(attn_scores, head_size=self.padded_head_dim)
            ttnn.deallocate(attn_scores)
            context = ttnn.matmul(attn_probs, value, memory_config=mem_cfg, dtype=dtype)
            ttnn.deallocate(attn_probs)

        ttnn.deallocate(query)
        ttnn.deallocate(key)
        ttnn.deallocate(value)
        context = ensure_interleaved(ttnn.transformer.concatenate_heads(context, memory_config=mem_cfg), mem_cfg)
        output_tt = ttnn.linear(
            context, self.out_proj.weight, bias=self.out_proj.bias, memory_config=mem_cfg, dtype=dtype
        )
        ttnn.deallocate(context)
        return ensure_interleaved(output_tt, mem_cfg)

    def release(self) -> None:
        self.qkv.release()
        self.out_proj.release()


def _build_linear_padded_input_per_head(
    state: dict[str, torch.Tensor],
    prefix: str,
    num_heads: int,
    padded_head_dim: int,
    *,
    device,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> TTLinear:
    weight = state[f"{prefix}.weight"]
    hidden_size = int(weight.shape[1])
    real_head_dim = hidden_size // num_heads
    padded_input_dim = num_heads * padded_head_dim
    padded_weight = torch.zeros((weight.shape[0], padded_input_dim), dtype=weight.dtype)
    for head_idx in range(num_heads):
        src_start = head_idx * real_head_dim
        dst_start = head_idx * padded_head_dim
        padded_weight[:, dst_start : dst_start + real_head_dim] = weight[:, src_start : src_start + real_head_dim]
    bias_key = f"{prefix}.bias"
    return build_linear(
        padded_weight,
        state[bias_key] if bias_key in state else None,
        device=device,
        dtype=dtype,
        memory_config=memory_config,
    )


def _build_fused_qkv(
    state: dict[str, torch.Tensor],
    layer_prefix: str,
    num_heads: int,
    padded_head_dim: int,
    *,
    device,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> TTLinear:
    q_w = state[f"{layer_prefix}.self_attn.q_proj.weight"]
    k_w = state[f"{layer_prefix}.self_attn.k_proj.weight"]
    v_w = state[f"{layer_prefix}.self_attn.v_proj.weight"]
    q_b = state[f"{layer_prefix}.self_attn.q_proj.bias"]
    k_b = state[f"{layer_prefix}.self_attn.k_proj.bias"]
    v_b = state[f"{layer_prefix}.self_attn.v_proj.bias"]
    hidden_size = int(q_w.shape[0])
    real_head_dim = hidden_size // num_heads
    padded_hidden = num_heads * padded_head_dim
    q_w_padded = torch.zeros((padded_hidden, q_w.shape[1]), dtype=q_w.dtype)
    k_w_padded = torch.zeros((padded_hidden, k_w.shape[1]), dtype=k_w.dtype)
    v_w_padded = torch.zeros((padded_hidden, v_w.shape[1]), dtype=v_w.dtype)
    q_b_padded = torch.zeros((padded_hidden,), dtype=q_b.dtype)
    k_b_padded = torch.zeros((padded_hidden,), dtype=k_b.dtype)
    v_b_padded = torch.zeros((padded_hidden,), dtype=v_b.dtype)
    for head_idx in range(num_heads):
        src_start = head_idx * real_head_dim
        dst_start = head_idx * padded_head_dim
        q_w_padded[dst_start : dst_start + real_head_dim] = q_w[src_start : src_start + real_head_dim]
        k_w_padded[dst_start : dst_start + real_head_dim] = k_w[src_start : src_start + real_head_dim]
        v_w_padded[dst_start : dst_start + real_head_dim] = v_w[src_start : src_start + real_head_dim]
        q_b_padded[dst_start : dst_start + real_head_dim] = q_b[src_start : src_start + real_head_dim]
        k_b_padded[dst_start : dst_start + real_head_dim] = k_b[src_start : src_start + real_head_dim]
        v_b_padded[dst_start : dst_start + real_head_dim] = v_b[src_start : src_start + real_head_dim]
    return build_linear(
        torch.cat([q_w_padded, k_w_padded, v_w_padded], dim=0),
        torch.cat([q_b_padded, k_b_padded, v_b_padded], dim=0),
        device=device,
        dtype=dtype,
        memory_config=memory_config,
    )


def build_attention(
    state: dict[str, torch.Tensor],
    layer_prefix: str,
    num_heads: int,
    padded_head_dim: int,
    *,
    device,
    dtype: ttnn.DataType,
    memory_config: ttnn.MemoryConfig,
) -> PatchTSTAttention:
    real_head_dim = int(state[f"{layer_prefix}.self_attn.q_proj.weight"].shape[0]) // num_heads
    return PatchTSTAttention(
        qkv=_build_fused_qkv(
            state, layer_prefix, num_heads, padded_head_dim, device=device, dtype=dtype, memory_config=memory_config
        ),
        out_proj=(
            build_linear_from_state(
                state, f"{layer_prefix}.self_attn.out_proj", device=device, dtype=dtype, memory_config=memory_config
            )
            if padded_head_dim == real_head_dim
            else _build_linear_padded_input_per_head(
                state,
                f"{layer_prefix}.self_attn.out_proj",
                num_heads,
                padded_head_dim,
                device=device,
                dtype=dtype,
                memory_config=memory_config,
            )
        ),
        num_heads=num_heads,
        padded_head_dim=padded_head_dim,
    )

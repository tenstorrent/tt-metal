# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

import ttnn


@dataclass(frozen=True)
class AttentionWeights:
    q_proj: ttnn.Tensor
    k_proj: ttnn.Tensor
    v_proj: ttnn.Tensor
    o_proj: ttnn.Tensor
    q_norm: ttnn.Tensor  # +1 pre-offset (zero-centered RMSNorm)
    k_norm: ttnn.Tensor
    # De-interleaved / packed projections for the prefill fast path (gated_attention_forward_ttnn
    # with T > 1): avoid the qg reshape+chunk and separate k/v transposes by feeding
    # ttnn.experimental.nlp_create_qkv_heads a plain q block and a packed [k|v] block directly.
    # Decode still uses q_proj/k_proj/v_proj above.
    q_deint: ttnn.Tensor  # [dim, H*Dh] -- q-only columns of q_proj, head-major
    gate_deint: ttnn.Tensor  # [dim, H*Dh] -- gate-only columns of q_proj, head-major
    kv_packed: ttnn.Tensor  # [dim, 2*Hkv*Dh] -- concat(k_proj, v_proj) columns
    # F9: single fused [Q-block | K-block | V-block] weight (each block head-major), so the
    # prefill fast path can use the single-input form of ttnn.experimental.nlp_create_qkv_heads
    # (2 matmuls: qkv_fused + gate) instead of the 2-input form (3 matmuls: q + kv_packed + gate).
    # Built from the same de-interleaved q rows / k_w / v_w that back q_deint/kv_packed above.
    qkv_fused: ttnn.Tensor  # [dim, H*Dh + 2*Hkv*Dh] -- concat(q_deint, k_proj, v_proj) columns


def load_attention_weights(mesh_device, state_dict, tensor_cache_path=None) -> AttentionWeights:
    def load_2d(name):
        return ttnn.as_tensor(
            state_dict[f"{name}.weight"],
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight") if tensor_cache_path else None,
            preprocess=lambda t: t.T.contiguous(),  # [in, out] for ttnn.linear; cache-miss only
        )

    def load_norm(name):
        t = state_dict[f"{name}.weight"] + 1.0
        return ttnn.as_tensor(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{name}.weight_offset") if tensor_cache_path else None,
        )

    def load_packed(tensor, cache_name):
        # Same shape/dtype/layout convention as load_2d, but from an already-assembled torch
        # tensor (row slice/concat below) rather than a direct state_dict lookup. Cached under a
        # name distinct from q_proj/k_proj/v_proj so stale caches from before this de-interleave
        # are never reused.
        return ttnn.as_tensor(
            tensor,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            cache_file_name=(tensor_cache_path / f"self_attn.{cache_name}.weight") if tensor_cache_path else None,
            preprocess=lambda t: t.T.contiguous(),  # [in, out] for ttnn.linear; cache-miss only
        )

    # q_proj.weight is HF [H*2*Dh, dim]: per head h, rows [h*2*Dh : h*2*Dh+Dh] are the query
    # columns and rows [h*2*Dh+Dh : (h+1)*2*Dh] are the gate columns (see
    # models/experimental/gated_attention_gated_deltanet/torch_functional/gated_attention.py
    # ~line 145: qg.view(B,T,H,2*Dh).chunk(2,dim=-1) — first half of each head's 2*Dh block is
    # query, second half is gate). head_dim comes from q_norm/k_norm ([head_dim] per the
    # reference docstring), so it doesn't need to be threaded through this function's signature.
    q_w = state_dict["q_proj.weight"]
    k_w = state_dict["k_proj.weight"]
    v_w = state_dict["v_proj.weight"]
    head_dim = state_dict["q_norm.weight"].shape[0]
    num_heads = q_w.shape[0] // (2 * head_dim)

    q_only_rows = torch.cat([q_w[h * 2 * head_dim : h * 2 * head_dim + head_dim, :] for h in range(num_heads)], dim=0)
    gate_only_rows = torch.cat(
        [q_w[h * 2 * head_dim + head_dim : (h + 1) * 2 * head_dim, :] for h in range(num_heads)], dim=0
    )
    kv_rows = torch.cat([k_w, v_w], dim=0)
    qkv_rows = torch.cat([q_only_rows, k_w, v_w], dim=0)

    return AttentionWeights(
        q_proj=load_2d("q_proj"),
        k_proj=load_2d("k_proj"),
        v_proj=load_2d("v_proj"),
        o_proj=load_2d("o_proj"),
        q_norm=load_norm("q_norm"),
        k_norm=load_norm("k_norm"),
        q_deint=load_packed(q_only_rows, "q_proj_deint"),
        gate_deint=load_packed(gate_only_rows, "gate_proj_deint"),
        kv_packed=load_packed(kv_rows, "kv_proj_packed"),
        qkv_fused=load_packed(qkv_rows, "qkv_fused_deint"),
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Host-side HF weight layout helpers for Llama-3.2-1B-Instruct (TTTv2 port).

Q/K linear weights use HuggingFace head layout; TTNN RoPE expects Meta layout.
This duplicates the small permute helpers from ``load_checkpoints.reverse_permute``
without importing ``models/tt_transformers``.

Llama 3.2 1B has no QKV bias and no Q/K norm.
"""

from __future__ import annotations

from typing import Any

import torch


def reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """HF Q/K weight rows → Meta layout (RoPE-compatible)."""
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute_hf_rope_to_meta_tables(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert HF cos/sin tables to Meta-style tables for ``RotarySetup1D``."""
    if len(cos.shape) == 3:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return cos, sin


def build_rope_cos_sin_torch(
    rotary_emb: Any, table_len: int, head_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build ``[1, 1, table_len, head_dim]`` cos/sin tensors (Meta layout) from HF rotary module."""
    x = torch.zeros(1, 1, table_len, head_dim, dtype=dtype)
    position_ids = torch.arange(table_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        cos_hf, sin_hf = rotary_emb(x, position_ids)
    cos_m, sin_m = permute_hf_rope_to_meta_tables(cos_hf.float(), sin_hf.float())
    return cos_m.to(dtype), sin_m.to(dtype)


def attention_wqkv_wo_from_hf_layer(
    self_attn: Any,
    num_devices: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack QKV and WO from an HF Llama attention block for ``Attention1D`` sharding."""
    wq_raw = self_attn.q_proj.weight
    wk_raw = self_attn.k_proj.weight
    wv_raw = self_attn.v_proj.weight
    wo_raw = self_attn.o_proj.weight

    dim = wq_raw.shape[1]
    cfg = self_attn.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // n_heads
    n_heads_times_head_dim = n_heads * head_dim
    n_kv_heads_times_head_dim = n_kv_heads * head_dim

    wq_meta = reverse_permute(wq_raw, n_heads, n_heads_times_head_dim, dim)
    wk_meta = reverse_permute(wk_raw, n_kv_heads, n_kv_heads_times_head_dim, dim)
    wv_meta = wv_raw
    wo_meta = wo_raw

    wq = wq_meta.T
    wk = wk_meta.T
    wv = wv_meta.T
    wo = wo_meta.T

    qkv_list = []
    for i in range(num_devices):
        wq_chunk = torch.chunk(wq, num_devices, dim=1)[i]
        wk_chunk = torch.chunk(wk, num_devices, dim=1)[i]
        wv_chunk = torch.chunk(wv, num_devices, dim=1)[i]
        qkv = torch.cat([wq_chunk, wk_chunk, wv_chunk], dim=-1)
        qkv_list.append(qkv)
    wqkv = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0).clone()
    wo = wo.unsqueeze(0).unsqueeze(0).clone()
    return wqkv, wo


def mlp_weights_from_hf_layer(mlp: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (w1, w2, w3) TTNN layouts: gate^T, down^T, up^T for ``MLP1D``."""
    w1 = mlp.gate_proj.weight.T.contiguous().clone()
    w3 = mlp.up_proj.weight.T.contiguous().clone()
    w2 = mlp.down_proj.weight.T.contiguous().clone()
    return w1, w2, w3


def rms_weight_torch(layernorm: Any) -> torch.Tensor:
    return layernorm.weight.detach().float().clone()


def embed_tokens_torch(embed: Any) -> torch.Tensor:
    w = embed.weight.detach().float().clone()
    return w.unsqueeze(0).unsqueeze(0)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side Hugging Face weight conversion for Llama-3.3-70B on WH Galaxy.

Two conversions are specific to the 2D mesh and therefore live here rather than
in the shared Galaxy layer:

* Q/K projections are permuted from HuggingFace head layout to the Meta layout
  TTNN's rotary embedding expects; and
* the fused QKV projection is packed so that each of the eight mesh rows owns a
  contiguous ``[Q_row, K_row, V_row]`` block, which is what the row-sharded
  fused create-QKV-heads collective consumes.

Llama 3.3 70B has no QKV bias and no Q/K normalization.
"""

from __future__ import annotations

from typing import Any

import torch

GALAXY_ROWS = 8


def reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """HF Q/K weight rows to Meta layout (RoPE compatible)."""

    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute_hf_rope_to_meta_tables(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert HF cos/sin tables to the Meta-style tables RotarySetup2D takes."""

    if len(cos.shape) == 3:
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = cos[:, : cos.shape[1] // 2]
    cos = torch.stack((cos, cos), dim=-1).flatten(-2)
    sin = sin[:, : sin.shape[1] // 2]
    sin = torch.stack((sin, sin), dim=-1).flatten(-2)
    return cos.unsqueeze(0).unsqueeze(0), sin.unsqueeze(0).unsqueeze(0)


def build_rope_cos_sin_torch(
    rotary_emb: Any, table_len: int, head_dim: int, dtype: torch.dtype = torch.bfloat16
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build `[1, 1, table_len, head_dim]` cos/sin tables from the HF rotary module.

    The HF module already applies Llama 3 RoPE scaling, so the scaled tables are
    data for ``RotarySetup2D`` and never a model-family branch inside it.
    """

    x = torch.zeros(1, 1, table_len, head_dim, dtype=dtype)
    position_ids = torch.arange(table_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        cos_hf, sin_hf = rotary_emb(x, position_ids)
    cos, sin = permute_hf_rope_to_meta_tables(cos_hf.float(), sin_hf.float())
    return cos.to(dtype).contiguous(), sin.to(dtype).contiguous()


def fuse_qkv_by_mesh_row(
    wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, *, rows: int = GALAXY_ROWS
) -> torch.Tensor:
    """Pack transposed Q/K/V projections into one row-major fused QKV weight.

    Each input has shape ``[dim, heads * head_dim]``. The result has shape
    ``[dim, qkv_size]`` with mesh row ``r`` owning ``[Q_r, K_r, V_r]``.
    """

    for name, tensor in (("wq", wq), ("wk", wk), ("wv", wv)):
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be a 2D [dim, out] projection, got {tuple(tensor.shape)}")
        if tensor.shape[-1] % rows:
            raise ValueError(f"{name} output width {tensor.shape[-1]} must shard over {rows} mesh rows")
    q_chunks = torch.chunk(wq, rows, dim=-1)
    k_chunks = torch.chunk(wk, rows, dim=-1)
    v_chunks = torch.chunk(wv, rows, dim=-1)
    return torch.cat(
        [torch.cat((q_chunks[row], k_chunks[row], v_chunks[row]), dim=-1) for row in range(rows)],
        dim=-1,
    ).contiguous()


def attention_weights_from_hf_layer(self_attn: Any, *, rows: int = GALAXY_ROWS) -> tuple[torch.Tensor, torch.Tensor]:
    """Return `(wqkv [dim, qkv_size], wo [n_heads * head_dim, dim])`."""

    config = self_attn.config
    n_heads = int(config.num_attention_heads)
    n_kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", 0) or config.hidden_size // n_heads)
    dim = int(self_attn.q_proj.weight.shape[1])

    wq = reverse_permute(self_attn.q_proj.weight.detach(), n_heads, n_heads * head_dim, dim).T
    wk = reverse_permute(self_attn.k_proj.weight.detach(), n_kv_heads, n_kv_heads * head_dim, dim).T
    wv = self_attn.v_proj.weight.detach().T
    wo = self_attn.o_proj.weight.detach().T
    return fuse_qkv_by_mesh_row(wq, wk, wv, rows=rows), wo.contiguous().clone()


def mlp_weights_from_hf_layer(mlp: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(w1 [dim, hidden], w2 [hidden, dim], w3 [dim, hidden])`."""

    w1 = mlp.gate_proj.weight.detach().T.contiguous().clone()
    w3 = mlp.up_proj.weight.detach().T.contiguous().clone()
    w2 = mlp.down_proj.weight.detach().T.contiguous().clone()
    return w1, w2, w3


def rms_weight_torch(layernorm: Any) -> torch.Tensor:
    """Return the `[dim]` RMSNorm weight."""

    return layernorm.weight.detach().float().contiguous().clone()


def embedding_table_torch(embed: Any) -> torch.Tensor:
    """Return the `[vocab_size, dim]` embedding table Embedding2D takes."""

    return embed.weight.detach().to(torch.bfloat16).contiguous().clone()


def lm_head_weight_torch(
    lm_head: Any, *, dim: int, vocab_size: int, padded_vocab_size: int
) -> torch.Tensor:
    """Return the `[dim, padded_vocab_size]` LM-head weight LMHead2D takes.

    Rows of the source (`[vocab, dim]`) become columns so that mesh rows shard
    the vocabulary in natural token order. Padding columns are zero; the module
    masks them to ``-inf`` so they can never be sampled.
    """

    weight = lm_head.weight.detach()
    if tuple(weight.shape) != (vocab_size, dim):
        raise ValueError(f"LM-head weight must have shape {(vocab_size, dim)}, got {tuple(weight.shape)}")
    if padded_vocab_size < vocab_size:
        raise ValueError("padded vocabulary cannot be smaller than the logical vocabulary")
    transposed = weight.to(torch.bfloat16).T.contiguous()
    if padded_vocab_size == vocab_size:
        return transposed.clone()
    padding = torch.zeros(dim, padded_vocab_size - vocab_size, dtype=transposed.dtype)
    return torch.cat([transposed, padding], dim=-1).contiguous()

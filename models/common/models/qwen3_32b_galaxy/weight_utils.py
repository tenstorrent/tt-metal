# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Host-side Hugging Face weight conversion for Qwen3-32B on WH Galaxy.

Qwen3 specifics this conversion owns:

* per-head Q/K RMSNorm weights (width ``head_dim``), permuted with the same
  interleave the Q/K projections use so head-local normalization matches;
* ``head_dim`` decoupled from ``hidden_size`` (128, not 5120 / 64), so the
  output projection reduces ``n_heads * head_dim`` back to ``dim``; and
* bias-free QKV projections, with the packing path kept for checkpoints that
  do carry a bias.

The fused QKV packing is row-major over the eight mesh rows, matching the
row-sharded fused create-QKV-heads collective.
"""

from __future__ import annotations

from typing import Any

import torch

GALAXY_ROWS = 8


def reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """HF Q/K weight rows to Meta layout (RoPE compatible)."""

    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def reverse_permute_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Undo the HF split of a per-head vector into interleaved complex pairs."""

    shape = tensor.shape
    width = shape[-1]
    if width % 2:
        raise ValueError(f"per-head vector width must be even, got {width}")
    reals = tensor[..., : width // 2]
    imags = tensor[..., width // 2 :]
    return torch.stack((reals, imags), dim=-1).flatten(start_dim=len(shape) - 1)


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
    """Build `[1, 1, table_len, head_dim]` cos/sin tables from the HF rotary module."""

    x = torch.zeros(1, 1, table_len, head_dim, dtype=dtype)
    position_ids = torch.arange(table_len, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        cos_hf, sin_hf = rotary_emb(x, position_ids)
    cos, sin = permute_hf_rope_to_meta_tables(cos_hf.float(), sin_hf.float())
    return cos.to(dtype).contiguous(), sin.to(dtype).contiguous()


def fuse_qkv_by_mesh_row(
    wq: torch.Tensor, wk: torch.Tensor, wv: torch.Tensor, *, rows: int = GALAXY_ROWS
) -> torch.Tensor:
    """Pack transposed Q/K/V projections into one row-major fused QKV weight."""

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


def fuse_qkv_bias_by_mesh_row(
    bq: torch.Tensor, bk: torch.Tensor, bv: torch.Tensor, *, rows: int = GALAXY_ROWS
) -> torch.Tensor:
    """Pack Q/K/V biases in the same row-major order as the fused weight."""

    chunks = [
        torch.cat(
            (
                torch.chunk(bq, rows, dim=0)[row],
                torch.chunk(bk, rows, dim=0)[row],
                torch.chunk(bv, rows, dim=0)[row],
            ),
            dim=-1,
        )
        for row in range(rows)
    ]
    return torch.cat(chunks, dim=-1).contiguous()


def attention_weights_from_hf_layer(
    self_attn: Any, *, rows: int = GALAXY_ROWS
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Return `(wqkv, wo, q_norm, k_norm, wqkv_bias)` in TT layout."""

    config = self_attn.config
    n_heads = int(config.num_attention_heads)
    n_kv_heads = int(config.num_key_value_heads)
    head_dim = int(
        getattr(self_attn, "head_dim", 0)
        or getattr(config, "head_dim", 0)
        or config.hidden_size // n_heads
    )
    dim = int(self_attn.q_proj.weight.shape[1])

    wq = reverse_permute(self_attn.q_proj.weight.detach(), n_heads, n_heads * head_dim, dim).T
    wk = reverse_permute(self_attn.k_proj.weight.detach(), n_kv_heads, n_kv_heads * head_dim, dim).T
    wv = self_attn.v_proj.weight.detach().T
    wo = self_attn.o_proj.weight.detach().T
    wqkv = fuse_qkv_by_mesh_row(wq, wk, wv, rows=rows)

    q_norm = k_norm = None
    if getattr(self_attn, "q_norm", None) is not None:
        q_norm = reverse_permute_1d(self_attn.q_norm.weight.detach().float()).contiguous().clone()
    if getattr(self_attn, "k_norm", None) is not None:
        k_norm = reverse_permute_1d(self_attn.k_norm.weight.detach().float()).contiguous().clone()

    wqkv_bias = None
    if getattr(self_attn.q_proj, "bias", None) is not None:
        bq = reverse_permute_1d(self_attn.q_proj.bias.detach().view(n_heads, head_dim)).reshape(-1)
        bk = reverse_permute_1d(self_attn.k_proj.bias.detach().view(n_kv_heads, head_dim)).reshape(-1)
        bv = self_attn.v_proj.bias.detach()
        wqkv_bias = fuse_qkv_bias_by_mesh_row(bq, bk, bv, rows=rows)

    return wqkv, wo.contiguous().clone(), q_norm, k_norm, wqkv_bias


def mlp_weights_from_hf_layer(mlp: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return `(w1 [dim, hidden], w2 [hidden, dim], w3 [dim, hidden])`."""

    w1 = mlp.gate_proj.weight.detach().T.contiguous().clone()
    w3 = mlp.up_proj.weight.detach().T.contiguous().clone()
    w2 = mlp.down_proj.weight.detach().T.contiguous().clone()
    return w1, w2, w3


def rms_weight_torch(layernorm: Any) -> torch.Tensor:
    return layernorm.weight.detach().float().contiguous().clone()


def embedding_table_torch(embed: Any) -> torch.Tensor:
    """Return the `[vocab_size, dim]` embedding table Embedding2D takes."""

    return embed.weight.detach().to(torch.bfloat16).contiguous().clone()


def lm_head_weight_torch(lm_head: Any, *, dim: int, vocab_size: int, padded_vocab_size: int) -> torch.Tensor:
    """Return the `[dim, padded_vocab_size]` LM-head weight LMHead2D takes.

    Qwen3-32B's 151936-token vocabulary pads to 152064 so each of the eight
    mesh rows owns a tile-aligned shard; the padding columns are zero and the
    module masks them to ``-inf``.
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

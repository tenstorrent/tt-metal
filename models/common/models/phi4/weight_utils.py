# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side HF weight layout helpers for Phi-4 (TTTv2 port).

Phi-4 (``transformers.models.phi3``) ships **fused** projections:
  - ``self_attn.qkv_proj``  : ``[n_heads·head_dim + 2·n_kv·head_dim, dim]``
  - ``mlp.gate_up_proj``    : ``[2·intermediate_size, dim]``  (HF order: gate first, up second)

This module splits those into the per-projection tensors the TTTv2 ``Attention1D`` /
``MLP1D`` modules expect, then applies the Llama/Qwen HF→Meta ``reverse_permute`` to Q/K
(Phi-3/Phi-4 use GPT-NeoX ``rotate_half`` RoPE, identical to the Llama path). The permute
helpers are duplicated from ``load_checkpoints.reverse_permute`` to avoid importing
``models/tt_transformers``.

Phi-4 has **no** QKV bias (``attention_bias=false``) and **no** q/k RMSNorm.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight


def reverse_permute(tensor: torch.Tensor, n_heads: int, dim1: int, dim2: int) -> torch.Tensor:
    """HF Q/K weight rows → Meta layout (RoPE-compatible)."""
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def permute_hf_rope_to_meta_tables(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert HF cos/sin tables to Meta-style tables for ``RotarySetup1D`` (see attention tests)."""
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


def split_fused_qkv(self_attn: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Slice Phi-3/Phi-4 fused ``qkv_proj.weight`` into (q, k, v) raw HF weight rows.

    ``qkv_proj`` is ``nn.Linear(dim, n_heads·head_dim + 2·n_kv·head_dim)`` so its weight has
    shape ``[op_size, dim]`` laid out as ``[Q rows | K rows | V rows]`` (HF ``Phi3Attention.forward``
    splits along the projection dim in exactly this order).
    """
    cfg = self_attn.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(self_attn, "head_dim", cfg.hidden_size // n_heads)
    q_size = n_heads * head_dim
    kv_size = n_kv_heads * head_dim

    qkv = self_attn.qkv_proj.weight
    assert qkv.shape[0] == q_size + 2 * kv_size, (
        f"Unexpected qkv_proj rows {qkv.shape[0]} != {q_size + 2 * kv_size} "
        f"(n_heads={n_heads}, n_kv={n_kv_heads}, head_dim={head_dim})"
    )
    wq_raw = qkv[:q_size]
    wk_raw = qkv[q_size : q_size + kv_size]
    wv_raw = qkv[q_size + kv_size : q_size + 2 * kv_size]
    return wq_raw, wk_raw, wv_raw, n_heads, n_kv_heads, head_dim


def attention_wqkv_wo_from_hf_layer(
    self_attn: Any,
    num_devices: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack Phi-4 attention weights for ``Attention1D`` (per-device interleaved QKV).

    Returns ``(wqkv, wo)`` where ``wqkv`` is ``[1, 1, dim, sum_per_dev·num_devices]`` with each
    device's slice laid out as ``[Q_chunk | K_chunk | V_chunk]``, matching
    ``get_attention_weights_from_ref_model`` in ``test_attention_1d.py``. Phi-4 has no QKV bias
    and no q/k RMSNorm, so neither is returned.
    """
    wq_raw, wk_raw, wv_raw, n_heads, n_kv_heads, head_dim = split_fused_qkv(self_attn)
    dim = wq_raw.shape[1]
    n_heads_times_head_dim = n_heads * head_dim
    n_kv_heads_times_head_dim = n_kv_heads * head_dim

    wq_meta = reverse_permute(wq_raw, n_heads, n_heads_times_head_dim, dim)
    wk_meta = reverse_permute(wk_raw, n_kv_heads, n_kv_heads_times_head_dim, dim)
    wv_meta = wv_raw
    wo_meta = self_attn.o_proj.weight

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
    """Split Phi-4 fused ``gate_up_proj`` and return (w1, w2, w3) = (gate^T, down^T, up^T).

    HF ``Phi3MLP.forward``: ``gate, up = gate_up_proj(x).chunk(2, -1)``; ``down_proj(up * silu(gate))``.
    So the ``gate_up_proj`` weight rows are ``[gate | up]`` — gate is the first ``intermediate_size``
    rows, up the second.
    """
    gate_up = mlp.gate_up_proj.weight
    inter = gate_up.shape[0] // 2
    assert gate_up.shape[0] == 2 * inter, f"gate_up_proj rows {gate_up.shape[0]} not even"
    gate = gate_up[:inter]
    up = gate_up[inter:]
    w1 = gate.T.contiguous().clone()
    w3 = up.T.contiguous().clone()
    w2 = mlp.down_proj.weight.T.contiguous().clone()
    return w1, w2, w3


def rms_weight_torch(layernorm: Any) -> torch.Tensor:
    return layernorm.weight.detach().float().clone()


def embed_tokens_torch(embed: Any) -> torch.Tensor:
    w = embed.weight.detach().float().clone()
    return w.unsqueeze(0).unsqueeze(0)


def build_lm_head_lazy_weights(
    mesh_device: ttnn.MeshDevice,
    lm_head_weight: torch.Tensor,
    *,
    dim: int,
    vocab_size: int,
    max_columns_per_device: int = 8192,
    dtype: ttnn.DataType = ttnn.bfloat8_b,
    cache_dir: Path | None = None,
) -> tuple[list[LazyWeight], list[int], list[ttnn.MemoryConfig]]:
    """
    Column-split LM head for ``LMHead1D`` (adapted from ``LMHead1D.from_model_args`` math, no v1 args).
    ``lm_head_weight`` shape: ``[vocab_size, dim]`` (HF).

    Returns ``(lazy_weights, split_sizes, weights_memcfgs)`` for ``LMHead1DConfig`` DRAM-sharded matmuls.
    """
    num_devices = mesh_device.get_num_devices()
    torch_w = lm_head_weight.T.contiguous().to(torch.bfloat16)
    padded_vocab_size = math.ceil(vocab_size / 32) * 32
    if vocab_size < padded_vocab_size:
        pad = padded_vocab_size - vocab_size
        torch_w = torch.cat([torch_w, torch.zeros(torch_w.shape[0], pad, dtype=torch_w.dtype)], dim=-1)

    size_per_device = padded_vocab_size // num_devices
    num_splits = math.ceil(size_per_device / max_columns_per_device)
    split_sizes = [min(size_per_device, max_columns_per_device)] * (num_splits - 1)
    split_sizes.append(size_per_device - sum(split_sizes))

    dram_size = mesh_device.dram_grid_size()
    dram_grid = ttnn.CoreRangeSet(
        {ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram_size.x - 1, dram_size.y - 1))}
    )
    tile = ttnn.TILE_SIZE

    def dram_sharded_memcfg(k: int, n: int) -> ttnn.MemoryConfig:
        padded_n = math.ceil(n / (tile * dram_size.x)) * (tile * dram_size.x)
        shard_spec = ttnn.ShardSpec(dram_grid, (k, padded_n // dram_size.x), ttnn.ShardOrientation.ROW_MAJOR)
        return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, shard_spec)

    output_weights: list[LazyWeight] = []
    weights_memcfgs: list[ttnn.MemoryConfig] = []
    for i, split_size in enumerate(split_sizes):
        device_splits = []
        for device_idx in range(num_devices):
            start = device_idx * size_per_device + sum(split_sizes[:i])
            end = start + split_size
            device_splits.append(torch_w[:, start:end])
        combined = torch.cat(device_splits, dim=-1)
        mem_cfg = dram_sharded_memcfg(dim, math.ceil(combined.shape[-1] / num_devices))
        weights_memcfgs.append(mem_cfg)
        name = f"lm_head_split_{i}_{combined.shape[-1]}"
        output_weights.append(
            LazyWeight(
                source=combined,
                dtype=dtype,
                device=mesh_device,
                mesh_mapper_config=ttnn.MeshMapperConfig(
                    placements=[ttnn.PlacementShard(-1)],
                    mesh_shape_override=ttnn.MeshShape([mesh_device.get_num_devices()]),
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem_cfg,
                cache_dir_weight_name=(cache_dir, name) if cache_dir else None,
            )
        )
    return output_weights, split_sizes, weights_memcfgs

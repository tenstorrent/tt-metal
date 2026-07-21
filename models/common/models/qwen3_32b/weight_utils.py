# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Host-side HF weight layout helpers for Qwen3-32B (TTTv2 port).

Q/K linear weights use HuggingFace head layout; TTNN RoPE expects Meta layout.
This duplicates the small permute helpers from ``load_checkpoints.reverse_permute``
without importing ``models/tt_transformers``. Logic is shape-agnostic and mirrors
``get_attention_weights_from_ref_model`` in ``test_attention_1d.py``.

Qwen3 specifics vs the Qwen2.5 family this was forked from:
  - **Per-head Q/K RMSNorm** (``q_norm`` / ``k_norm``, dim = ``head_dim``) is present and
    extracted here (``reverse_permute_1d`` applied to match the Q/K weight head-dim permute).
  - **No QKV bias** (``attention_bias=False``), so ``wqkv_bias`` is ``None``.
  - **Decoupled ``head_dim``** (128, not ``hidden_size / n_heads`` = 80).
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


def reverse_permute_1d(tensor: torch.Tensor) -> torch.Tensor:
    """Undo HF split of last dim into interleaved complex pairs for norm/bias vectors."""
    shape = tensor.shape
    dim = shape[-1]
    assert dim % 2 == 0, "Last dimension must be even"
    reals = tensor[..., : dim // 2]
    imags = tensor[..., dim // 2 :]
    interleaved = torch.stack((reals, imags), dim=-1).flatten(start_dim=len(shape) - 1)
    return interleaved


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


def attention_wqkv_wo_from_hf_layer(
    self_attn: Any,
    num_devices: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Mirror ``get_attention_weights_from_ref_model`` in ``test_attention_1d.py`` (HF module API)."""
    wq_raw = self_attn.q_proj.weight
    wk_raw = self_attn.k_proj.weight
    wv_raw = self_attn.v_proj.weight
    wo_raw = self_attn.o_proj.weight

    dim = wq_raw.shape[1]
    cfg = self_attn.config
    n_heads = cfg.num_attention_heads
    n_kv_heads = cfg.num_key_value_heads
    # Qwen3 decouples head_dim from hidden_size (head_dim=128 ≠ hidden/n_heads=80). Prefer the
    # HF module / config field over the coupled formula; matches test_attention_1d's reference.
    head_dim = getattr(self_attn, "head_dim", None) or getattr(cfg, "head_dim", None) or (cfg.hidden_size // n_heads)
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

    q_norm = None
    k_norm = None
    if getattr(self_attn, "q_norm", None) is not None:
        q_norm = reverse_permute_1d(self_attn.q_norm.weight).clone()
    if getattr(self_attn, "k_norm", None) is not None:
        k_norm = reverse_permute_1d(self_attn.k_norm.weight).clone()

    wqkv_bias = None
    if getattr(self_attn.q_proj, "bias", None) is not None:
        bq_raw = self_attn.q_proj.bias
        bk_raw = self_attn.k_proj.bias
        bv_raw = self_attn.v_proj.bias
        bq_meta = reverse_permute_1d(bq_raw.view(n_heads, head_dim)).view(-1)
        bk_meta = reverse_permute_1d(bk_raw.view(n_kv_heads, head_dim)).view(-1)
        bv_meta = bv_raw
        qkv_bias_list = []
        for i in range(num_devices):
            bq_chunk = torch.chunk(bq_meta, num_devices, dim=0)[i]
            bk_chunk = torch.chunk(bk_meta, num_devices, dim=0)[i]
            bv_chunk = torch.chunk(bv_meta, num_devices, dim=0)[i]
            qkv_bias_list.append(torch.cat([bq_chunk, bk_chunk, bv_chunk], dim=-1))
        wqkv_bias = torch.cat(qkv_bias_list, dim=-1).clone()

    return wqkv, wo, q_norm, k_norm, wqkv_bias


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


def lm_head_padded_vocab_size(vocab_size: int, num_devices: int) -> int:
    """Vocab width padded so each device's LM-head column shard is tile-aligned.

    Single source of truth for the padded vocab: ``build_lm_head_lazy_weights`` pads the
    LM head to this width, and the on-device sampler (``Sampling1D``) must use the SAME
    width so its per-device ``index_offsets`` (``device_id * vocab // num_devices``) line up
    with the real shard boundaries. Pass ``vocab_size=lm_head_padded_vocab_size(...)`` and
    ``valid_vocab_size=<real vocab>`` to ``Sampling1D``; the real vocab drives the padding mask.

    Qwen3-32B: 151936 → 152064 (size_per_device 19008, a multiple of TILE_SIZE=32).
    """
    align = 32 * num_devices
    return math.ceil(vocab_size / align) * align


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
    # Pad the vocab so the PER-DEVICE column count is tile-aligned, not just the total.
    # Qwen3-32B: vocab=151936 → 151936/8 = 18992 per device, which is NOT a multiple of
    # TILE_SIZE (=32). The width-sharded LM matmul tile-pads each device's logits to 19008,
    # so an all-gather on dim 3 reconstructs the vocab with a 16-column gap per device and
    # every token id on device d>0 is shifted by d*16 — producing wrong argmax / low PCC vs HF
    # (observed: HF " Rome"=21718 vs TT 21734). Padding to a multiple of (TILE_SIZE*num_devices)
    # makes size_per_device tile-aligned (19008) so the gather is gap-free; the trailing pad
    # columns are zeros at the very end of the vocab. Coder-32B's 152064 was already aligned and
    # never hit this. See the qwen3_32b debugging notes.
    padded_vocab_size = lm_head_padded_vocab_size(vocab_size, num_devices)
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

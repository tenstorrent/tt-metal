# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

import ttnn


def _prefill_embed_l1_max_bytes() -> int:
    """Max bf16 activation bytes for L1 embedding output (S=128, H=2048 → 512 KiB)."""
    raw = os.environ.get("GLM4_MOE_LITE_PREFILL_EMBED_L1_MAX_BYTES", "").strip()
    return int(raw) if raw else 512 * 1024


def convert_embedding_weight_to_tt(
    *,
    device,
    embed_weight: torch.Tensor,
    cache_file_name: Optional[Path] = None,
    dtype: ttnn.DataType = ttnn.bfloat16,
) -> ttnn.Tensor:
    """
    Convert HF `model.embed_tokens.weight` into a TT tensor usable with `ttnn.embedding`.

    Note:
    - We keep weights in row-major layout and let `ttnn.embedding` produce tiled activations.
    - We replicate across the mesh for the initial bring-up. Later phases can shard.
    """
    # Match tt-transformers convention: [1, 1, vocab, dim]
    torch_weight = embed_weight.unsqueeze(0).unsqueeze(0)
    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    return ttnn.as_tensor(
        torch_weight,
        device=device,
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None,
        cache_file_name=None if cache_file_name is None else Path(cache_file_name),
    )


def prefill_embed_memory_config(*, seq_tokens: int, hidden_dim: int) -> ttnn.MemoryConfig:
    """Pick embedding output memory: L1 when small enough to skip DRAM→L1 copy at decoder entry."""
    if os.environ.get("GLM4_MOE_LITE_PREFILL_EMBED_L1", "1").strip() == "0":
        return ttnn.DRAM_MEMORY_CONFIG
    act_bytes = int(seq_tokens) * int(hidden_dim) * 2  # bf16
    if act_bytes <= _prefill_embed_l1_max_bytes():
        return ttnn.L1_MEMORY_CONFIG
    return ttnn.DRAM_MEMORY_CONFIG


def run_tt_embedding(
    *,
    device,
    token_ids: torch.Tensor,
    tt_weight: ttnn.Tensor,
    output_layout: ttnn.Layout = ttnn.TILE_LAYOUT,
    memory_config: ttnn.MemoryConfig | None = None,
) -> ttnn.Tensor:
    """
    Run embedding lookup on TT.

    token_ids:
      torch int32/64 tensor on CPU, shape [B, S] (or [S]).
    """
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(0)
    assert token_ids.ndim == 2, f"expected [B,S] token ids, got shape={tuple(token_ids.shape)}"

    # embedding expects non-negative indices.
    if token_ids.dtype != torch.int32:
        token_ids = token_ids.to(dtype=torch.int32)
    if (token_ids < 0).any():
        raise ValueError("token_ids must be non-negative for embedding lookup")

    # ttnn.embedding expects indices on-device as a TT tensor.
    tt_ids = ttnn.from_torch(
        token_ids,
        device=device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None,
    )
    out_mc = memory_config if memory_config is not None else ttnn.DRAM_MEMORY_CONFIG
    return ttnn.embedding(tt_ids, tt_weight, layout=output_layout, memory_config=out_mc)

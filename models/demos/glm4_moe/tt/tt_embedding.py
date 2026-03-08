# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

import ttnn


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


def run_tt_embedding(
    *,
    device,
    token_ids: torch.Tensor,
    tt_weight: ttnn.Tensor,
    output_layout: ttnn.Layout = ttnn.TILE_LAYOUT,
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

    import os, sys
    if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
        print(f"  [DEBUG EMBED] before ttnn.embedding, ids={list(tt_ids.shape)}, weight={list(tt_weight.shape)}", flush=True, file=sys.stderr)
        ttnn.synchronize_device(device)
        print(f"  [DEBUG EMBED] sync before embedding OK", flush=True, file=sys.stderr)

    result = ttnn.embedding(tt_ids, tt_weight, layout=output_layout, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    if os.environ.get("GLM4_MOE_DEBUG_SYNC", "0") != "0":
        print(f"  [DEBUG EMBED] after ttnn.embedding, result={list(result.shape)}", flush=True, file=sys.stderr)
        ttnn.synchronize_device(device)
        print(f"  [DEBUG EMBED] sync after embedding OK", flush=True, file=sys.stderr)

    return result

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch

import ttnn

from models.experimental.glm4_moe_lite.tt.linear_helpers import decode_width_sharded_norm_input_config


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


def _decode_embed_width_sharded_enabled() -> bool:
    """Embed directly into WIDTH_SHARDED L1 matching decode input RMSNorm (requires sharded norm)."""
    if os.environ.get("GLM4_MOE_LITE_SHARDED_DECODE_NORM", "").strip() != "1":
        return False
    raw = os.environ.get("GLM4_MOE_LITE_DECODE_EMBED_WIDTH_SHARDED", "0").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _decode_embed_width_sharded_viable(batch: int) -> bool:
    """True when embed output height matches the norm WIDTH_SHARDED shard height.

    TTNN requires physical_height == shard_height for WIDTH_SHARDED tensors.
    Norm uses shard height ``mt * TILE_SIZE`` where ``mt = ceil(batch / TILE_SIZE)``.
    Embed output ``[batch, 1, 1, H]`` has physical height ``batch``, so we need
    ``batch == mt * TILE_SIZE`` (e.g. 32, 64).  Smaller batches (e.g. B=1) fall
    back to interleaved embed + InterleavedToSharded before norm.
    """
    b = int(batch)
    if b <= 0:
        return False
    mt = max(1, (b + ttnn.TILE_SIZE - 1) // ttnn.TILE_SIZE)
    return b == mt * ttnn.TILE_SIZE


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


def run_tt_decode_embedding(
    *,
    device,
    token_ids: torch.Tensor,
    tt_weight: ttnn.Tensor,
    tokens_tt: ttnn.Tensor | None = None,
    skip_defensive_clone: bool = False,
    width_sharded_norm: bool | None = None,
) -> ttnn.Tensor:
    """Decode embedding: [B,1] token ids -> [1,1,B,hidden] TILE activations.

    When ``tokens_tt`` is provided, copies host token ids into the persistent
    device buffer instead of allocating a new ``from_torch`` tensor each step.
    When ``skip_defensive_clone`` is set and the batch dimension is already
    tile-tight, skips the post-embed ``clone`` (caller must not deallocate the
    embedding output buffer separately).

    When ``width_sharded_norm`` is enabled (``GLM4_MOE_LITE_DECODE_EMBED_WIDTH_SHARDED=1``
    and sharded decode norm), embeds directly into the WIDTH_SHARDED L1 layout
    expected by input RMSNorm, skipping InterleavedToSharded before norm.
    """
    if token_ids.ndim == 1:
        token_ids = token_ids.unsqueeze(1)
    if token_ids.ndim != 2 or int(token_ids.shape[1]) != 1:
        raise ValueError(f"expected token_ids [B,1], got shape={tuple(token_ids.shape)}")

    batch = int(token_ids.shape[0])
    if token_ids.dtype != torch.int32:
        token_ids = token_ids.to(dtype=torch.int32)
    if (token_ids < 0).any():
        raise ValueError("token_ids must be non-negative for embedding lookup")

    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None

    if tokens_tt is not None:
        host_ids = ttnn.from_torch(
            token_ids,
            device=None,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )
        ttnn.copy_host_to_device_tensor(host_ids, tokens_tt)
        tt_ids = tokens_tt
    else:
        tt_ids = ttnn.from_torch(
            token_ids,
            device=device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    hidden = int(tt_weight.shape[-1])
    use_width_sharded = (
        _decode_embed_width_sharded_enabled() if width_sharded_norm is None else bool(width_sharded_norm)
    )
    if use_width_sharded and not _decode_embed_width_sharded_viable(batch):
        use_width_sharded = False
    embed_mc = ttnn.DRAM_MEMORY_CONFIG
    if use_width_sharded:
        try:
            embed_mc = decode_width_sharded_norm_input_config(device, batch, hidden)
        except Exception:
            use_width_sharded = False

    x = ttnn.embedding(
        tt_ids,
        tt_weight,
        layout=ttnn.TILE_LAYOUT,
        memory_config=embed_mc,
    )
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    # Match trace / model_tt layout: [1,1,B,H]
    if len(x.shape) == 4 and int(x.shape[0]) == batch and int(x.shape[2]) == 1:
        x = ttnn.reshape(x, (1, 1, batch, hidden))
    elif len(x.shape) == 4 and int(x.shape[1]) == batch:
        x = ttnn.reshape(x, (1, batch, 1, hidden))
        x = ttnn.permute(x, (0, 2, 1, 3))
    elif len(x.shape) == 3 and int(x.shape[0]) == batch:
        x = ttnn.reshape(x, (1, 1, batch, hidden))

    if use_width_sharded:
        return x

    if skip_defensive_clone and int(x.shape[2]) == batch:
        return x

    x_view = ttnn.slice(x, [0, 0, 0, 0], [1, 1, batch, hidden])
    if skip_defensive_clone:
        return x_view
    x_tight = ttnn.clone(x_view, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x, force=False)
    return x_tight

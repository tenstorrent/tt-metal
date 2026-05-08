# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Token embedding lookup matching Hugging Face ``nn.Embedding`` / ``F.embedding``."""

from __future__ import annotations

import torch
import torch.nn.functional as F

import ttnn


def embedding_lookup_reference_torch(input_ids_bs: torch.Tensor, weight_vh: torch.Tensor) -> torch.Tensor:
    """CPU reference: ``F.embedding`` with HF weight layout ``[vocab_size, hidden_size]``."""
    idx = input_ids_bs.to(torch.long)
    return F.embedding(idx, weight_vh.to(torch.bfloat16))


def embedding_lookup_bf16(mesh_device, input_ids_bs: torch.Tensor, weight_vh: torch.Tensor) -> torch.Tensor:
    """
    Embedding gather on device; returns host bf16 ``[B, S, hidden_size]``.

    Args:
        input_ids_bs: ``[B, S]`` token indices (``int64`` / ``int32`` / ``uint32``; must be in-range).
        weight_vh: HF ``embed_tokens.weight``, shape ``[vocab_size, hidden_size]``.
    """
    if input_ids_bs.ndim != 2:
        raise ValueError(f"expected input_ids [B,S], got {tuple(input_ids_bs.shape)}")
    if weight_vh.ndim != 2:
        raise ValueError(f"expected weight [V,H], got {tuple(weight_vh.shape)}")

    b, s = int(input_ids_bs.shape[0]), int(input_ids_bs.shape[1])
    v = int(weight_vh.shape[0])
    w_bf16 = weight_vh.to(torch.bfloat16).contiguous()

    idx = input_ids_bs.to(torch.int64)
    if (idx < 0).any() or (idx >= v).any():
        raise ValueError("input_ids must be in [0, vocab_size) for embedding lookup")

    # ttnn.embedding indices are typically uint32 on device (see ttnn unit tests).
    ids = idx.to(torch.uint32).contiguous()

    tt_ids = ttnn.from_torch(
        ids,
        device=mesh_device,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_w = ttnn.from_torch(
        w_bf16,
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_y = ttnn.embedding(tt_ids, tt_w, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)
    ttnn.deallocate(tt_ids)
    ttnn.deallocate(tt_w)

    y = ttnn.to_torch(tt_y, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    ttnn.deallocate(tt_y)

    while y.ndim > 3:
        y = y.squeeze(0)
    if y.ndim == 3 and y.shape[0] != b:
        y = y[:b]
    h = int(w_bf16.shape[1])
    assert y.shape == (b, s, h), f"unexpected embedding output shape {tuple(y.shape)}"
    return y.to(torch.bfloat16)

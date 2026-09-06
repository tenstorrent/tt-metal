# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Global-attention KV cache layout shared by prefill and migration.

Gemma 4 global attention projects one tied KV vector.  After RMSNorm (without
gamma), the 512-wide vector is V.  K differs only by the per-channel gamma and
partial RoPE, whose active lanes are 128-wide.  Store exactly those independent
values in one 640-wide row::

    [K rotary (128) | V non-rotary (384) | V rotary (128)]

The first 512 channels form the effective K used by prefill SDPA, while the
last 512 channels form V.  These views overlap by 384 channels and therefore do
not require a second cache allocation.

This module intentionally keeps the layout arithmetic host-only.  Device
writers/readers consume the resulting fixed indices without inventing their
own permutations.
"""

from __future__ import annotations

import torch

import ttnn

GLOBAL_HEAD_DIM = 512
GLOBAL_ROTARY_DIM = 128
GLOBAL_PACKED_DIM = GLOBAL_HEAD_DIM + GLOBAL_ROTARY_DIM
SLIDING_HEAD_DIM = 256


def sliding_kv_indices(head_dim: int = SLIDING_HEAD_DIM) -> torch.Tensor:
    """Return NeoX channels in adjacent-pair order for decode-compatible K."""
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even number, got {head_dim}")
    half = head_dim // 2
    return torch.stack((torch.arange(half), torch.arange(half, head_dim)), dim=1).reshape(-1)


def global_kv_indices(
    head_dim: int = GLOBAL_HEAD_DIM, rotary_dim: int = GLOBAL_ROTARY_DIM
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(rotary, nonrotary, value)`` source-column indices.

    Gemma's partial NeoX RoPE operates on the first ``rotary_dim / 2`` lanes
    from each half of the head.  The packed K prefix interleaves each rotary
    pair for the decode-side representation.  V keeps non-rotary lanes first
    so it overlaps the non-rotary part of the effective K view.
    """
    if head_dim <= 0 or head_dim % 2:
        raise ValueError(f"head_dim must be a positive even number, got {head_dim}")
    if rotary_dim <= 0 or rotary_dim >= head_dim or rotary_dim % 2:
        raise ValueError(f"rotary_dim must be positive, even, and smaller than {head_dim}, got {rotary_dim}")

    half = head_dim // 2
    rotary_half = rotary_dim // 2
    rotary_first = torch.arange(rotary_half, dtype=torch.long)
    rotary_second = torch.arange(half, half + rotary_half, dtype=torch.long)
    rotary = torch.stack((rotary_first, rotary_second), dim=1).reshape(-1)
    nonrotary = torch.cat(
        (
            torch.arange(rotary_half, half, dtype=torch.long),
            torch.arange(half + rotary_half, head_dim, dtype=torch.long),
        )
    )
    # Keep each section in HF channel order.  This makes the V view a simple
    # [nonrotary | rotary] permutation while K's active pairs stay adjacent.
    value = torch.cat((nonrotary, rotary_first, rotary_second))
    return rotary, nonrotary, value


def pack_global_kv_reference(k_rotary: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    """Pack canonical post-RoPE K and no-gamma RMSNorm V into one cache row."""
    if k_rotary.shape != value.shape or k_rotary.shape[-1] != GLOBAL_HEAD_DIM:
        raise ValueError(
            f"expected equal K/V shapes ending in {GLOBAL_HEAD_DIM}, got {tuple(k_rotary.shape)} and "
            f"{tuple(value.shape)}"
        )
    rotary, _, value_order = global_kv_indices()
    return torch.cat((k_rotary.index_select(-1, rotary), value.index_select(-1, value_order)), dim=-1)


def pack_global_query_reference(query_rotary: torch.Tensor, k_gamma: torch.Tensor) -> torch.Tensor:
    """Transform Q for a dot product with the packed cache's K view.

    K's non-rotary gamma is moved to Q.  This preserves the attention score
    while allowing those K channels to alias the no-gamma V data in cache.
    """
    if query_rotary.shape[-1] != GLOBAL_HEAD_DIM or k_gamma.numel() != GLOBAL_HEAD_DIM:
        raise ValueError(
            f"expected Q width and gamma length {GLOBAL_HEAD_DIM}, got {query_rotary.shape[-1]} and {k_gamma.numel()}"
        )
    rotary, nonrotary, _ = global_kv_indices()
    gamma = k_gamma.reshape(-1).to(device=query_rotary.device, dtype=query_rotary.dtype)
    return torch.cat(
        (
            query_rotary.index_select(-1, rotary),
            query_rotary.index_select(-1, nonrotary) * gamma.index_select(0, nonrotary),
        ),
        dim=-1,
    )


def unpack_global_value_reference(packed_value: torch.Tensor) -> torch.Tensor:
    """Restore an SDPA result from packed V order to canonical HF order."""
    if packed_value.shape[-1] != GLOBAL_HEAD_DIM:
        raise ValueError(f"expected packed V width {GLOBAL_HEAD_DIM}, got {packed_value.shape[-1]}")
    _, _, value_order = global_kv_indices()
    inverse = torch.empty_like(value_order)
    inverse[value_order] = torch.arange(GLOBAL_HEAD_DIM, dtype=torch.long)
    return packed_value.index_select(-1, inverse.to(packed_value.device))


_DEVICE_INDEX_CACHE: dict[tuple, ttnn.Tensor] = {}


def _device_column_index(tensor: ttnn.Tensor, columns: torch.Tensor) -> ttnn.Tensor:
    """Materialize and reuse a gather index matching the tensor except in width."""
    shape = tuple(int(x) for x in tensor.shape)
    cols = tuple(int(x) for x in columns.tolist())
    key = (id(tensor.device()), shape[:-1], cols)
    cached = _DEVICE_INDEX_CACHE.get(key)
    if cached is not None:
        return cached

    index = torch.tensor(cols, dtype=torch.uint32).reshape(1, 1, 1, -1)
    index = index.expand(*shape[:-1], len(cols)).contiguous()
    cached = ttnn.from_torch(
        index,
        device=tensor.device(),
        dtype=ttnn.uint32,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(tensor.device()),
    )
    _DEVICE_INDEX_CACHE[key] = cached
    return cached


def _gather_columns(tensor: ttnn.Tensor, columns: torch.Tensor, memory_config) -> ttnn.Tensor:
    return ttnn.gather(
        tensor,
        dim=-1,
        index=_device_column_index(tensor, columns),
        memory_config=memory_config,
    )


def pack_global_query_device(
    query: ttnn.Tensor,
    packed_q_scale_weight: ttnn.Tensor | None,
    *,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Map canonical post-RoPE Q to the effective K prefix of the 640-wide cache."""
    rotary, nonrotary, _ = global_kv_indices()
    query_order = torch.cat((rotary, nonrotary))
    ordered = _gather_columns(query, query_order, memory_config)
    if packed_q_scale_weight is None:
        return ordered
    scale = ttnn.reshape(packed_q_scale_weight, (1, 1, 1, GLOBAL_HEAD_DIM))
    result = ttnn.multiply(ordered, scale, memory_config=memory_config)
    ordered.deallocate(True)
    return result


def pack_sliding_rope_device(
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    *,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> tuple[ttnn.Tensor, ttnn.Tensor]:
    """Prepare adjacent-pair RoPE lanes once for all sliding layers."""
    order = sliding_kv_indices(int(cos_cache.shape[-1]))
    return (
        _gather_columns(cos_cache, order, memory_config),
        _gather_columns(sin_cache, order, memory_config),
    )


def pack_global_rope_device(
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    *,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
    """Prepare packed-Q and active-K RoPE lanes once for all global layers."""
    rotary, _, _ = global_kv_indices()
    rotary_neox = torch.sort(rotary).values
    return (
        _gather_columns(cos_cache, rotary, memory_config),
        _gather_columns(sin_cache, rotary, memory_config),
        _gather_columns(cos_cache, rotary_neox, memory_config),
        _gather_columns(sin_cache, rotary_neox, memory_config),
    )


def pack_global_kv_device(
    value: ttnn.Tensor,
    k_norm_rotary_weight: ttnn.Tensor,
    cos_cache: ttnn.Tensor,
    sin_cache: ttnn.Tensor,
    *,
    canonical_k: ttnn.Tensor | None = None,
    packed_rope_mats: tuple[ttnn.Tensor, ...] | None = None,
    value_is_packed: bool = False,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Create Krot128 plus Vordered512 from the no-gamma normalized tied KV.

    When no legacy canonical K cache is needed, only the active 128 K channels
    receive gamma and RoPE. canonical_k is accepted during the transition for
    paths that still maintain the separate paged cache.
    """
    rotary, _, value_order = global_kv_indices()
    if canonical_k is not None:
        k_rotary = _gather_columns(canonical_k, rotary, memory_config)
    else:
        rotary_neox = torch.sort(rotary).values
        interleave = torch.stack(
            (
                torch.arange(GLOBAL_ROTARY_DIM // 2, dtype=torch.long),
                torch.arange(GLOBAL_ROTARY_DIM // 2, GLOBAL_ROTARY_DIM, dtype=torch.long),
            ),
            dim=1,
        ).reshape(-1)
        if value_is_packed:
            active_value = ttnn.slice(
                value,
                (0, 0, 0, GLOBAL_HEAD_DIM - GLOBAL_ROTARY_DIM),
                tuple(value.shape)[:-1] + (GLOBAL_HEAD_DIM,),
                memory_config=memory_config,
            )
        else:
            active_value = _gather_columns(value, rotary_neox, memory_config)
        gamma = ttnn.reshape(k_norm_rotary_weight, (1, 1, 1, GLOBAL_ROTARY_DIM))
        scaled = ttnn.multiply(active_value, gamma, memory_config=memory_config)
        if packed_rope_mats is None:
            active_cos = _gather_columns(cos_cache, rotary_neox, memory_config)
            active_sin = _gather_columns(sin_cache, rotary_neox, memory_config)
            owns_active_rope = True
        else:
            active_cos, active_sin = packed_rope_mats[2:4]
            owns_active_rope = False
        roped = ttnn.experimental.rotary_embedding(
            scaled,
            active_cos,
            active_sin,
            None,
            memory_config=memory_config,
        )
        k_rotary = _gather_columns(roped, interleave, memory_config)
        for tensor in (active_value, scaled, roped):
            tensor.deallocate(True)
        if owns_active_rope:
            active_cos.deallocate(True)
            active_sin.deallocate(True)

    value_ordered = value if value_is_packed else _gather_columns(value, value_order, memory_config)
    packed = ttnn.concat((k_rotary, value_ordered), dim=-1, memory_config=memory_config)
    k_rotary.deallocate(True)
    if value_ordered is not value:
        value_ordered.deallocate(True)
    return packed


def unpack_global_value_device(
    packed_value: ttnn.Tensor,
    *,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
) -> ttnn.Tensor:
    """Restore SDPA output from the packed cache V order to HF head order."""
    _, _, value_order = global_kv_indices()
    inverse = torch.empty_like(value_order)
    inverse[value_order] = torch.arange(GLOBAL_HEAD_DIM, dtype=torch.long)
    return _gather_columns(packed_value, inverse, memory_config)

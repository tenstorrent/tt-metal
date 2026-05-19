# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared TTNN helpers for Seamless M4T v2 (grid, tile padding, attention masks)."""

from __future__ import annotations

from typing import Any, Optional, Tuple

import ttnn

from models.common.utility_functions import nearest_32

# ``torch.finfo(torch.bfloat16).min`` — the additive-mask "minus infinity" HF uses. Bf16-representable.
NEG_INF = -3.3895313892515355e38

# Tile alignment: TT SDPA must score against tile-aligned key sequences; pad to ``ceil(seq/32)*32``.
TILE = 32


def core_grid(device: ttnn.Device) -> ttnn.CoreGrid:
    grid = device.compute_with_storage_grid_size()
    return ttnn.CoreGrid(y=grid.y, x=grid.x)


def ensure_tile_bf16_sdpa_mask(x: ttnn.Tensor) -> ttnn.Tensor:
    """SDPA requires a TILE bf16 mask; ``expand``/``add`` paths often yield ROW_MAJOR."""
    if x.get_layout() == ttnn.TILE_LAYOUT and x.dtype == ttnn.bfloat16:
        return x
    out = ttnn.to_layout(x, ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x)
    return out


def tile_align(seq: int) -> int:
    return ((seq + TILE - 1) // TILE) * TILE


def tt_position_ids(input_ids: ttnn.Tensor, pad_id: int) -> ttnn.Tensor:
    """HF ``create_position_ids_from_input_ids`` on device — ``cumsum`` of non-pad mask + offset."""
    ids_tile = (
        ttnn.to_layout(input_ids, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if input_ids.get_layout() != ttnn.TILE_LAYOUT
        else input_ids
    )
    mask = ttnn.ne(ids_tile, pad_id)
    if ids_tile is not input_ids:
        ttnn.deallocate(ids_tile)
    mask_i32 = ttnn.typecast(mask, ttnn.int32)
    ttnn.deallocate(mask)
    cumsum = ttnn.cumsum(mask_i32, dim=1, dtype=ttnn.int32)
    pos = ttnn.multiply(cumsum, mask_i32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(cumsum)
    ttnn.deallocate(mask_i32)
    pos = ttnn.add(pos, pad_id, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    pos = ttnn.typecast(pos, ttnn.uint32)
    if pos.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
        pos = ttnn.to_layout(pos, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return pos


def tt_seq_position_ids(bsz: int, seq: int, pad_id: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``create_position_ids_from_inputs_embeds`` on device — ``[pad+1, pad+2, …, pad+seq]``."""
    pos_1d = ttnn.arange(
        pad_id + 1,
        seq + pad_id + 1,
        dtype=ttnn.uint32,
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    pos_2d = ttnn.reshape(pos_1d, [1, seq])
    if bsz <= 1:
        return pos_2d
    pos_out = ttnn.expand(pos_2d, [bsz, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pos_2d)
    return pos_out


def key_padding_additive(mask_2d: ttnn.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """``[B, S]`` 0/1 → ``[B, S]`` bf16 with ``0`` at real and ``NEG_INF`` at padded positions."""
    pad_bool = ttnn.eq(mask_2d, 0)
    pad_bf = ttnn.typecast(pad_bool, ttnn.bfloat16)
    ttnn.deallocate(pad_bool)
    additive = ttnn.multiply(pad_bf, NEG_INF, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pad_bf)
    return additive


def build_causal_mask_4d(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_causal_attention_mask`` (causal half only) on device."""
    full_neg = ttnn.full(
        [seq, seq],
        NEG_INF,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    causal_2d = ttnn.triu(full_neg, diagonal=1)
    ttnn.deallocate(full_neg)
    causal_4d = ttnn.reshape(causal_2d, [1, 1, seq, seq])
    if batch <= 1:
        return ensure_tile_bf16_sdpa_mask(causal_4d)
    expanded = ttnn.expand(causal_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ensure_tile_bf16_sdpa_mask(expanded)
    if causal_4d is not out:
        ttnn.deallocate(causal_4d)
    return out


def build_causal_with_padding_4d(
    attention_mask_2d: Optional[ttnn.Tensor], batch: int, seq: int, device: ttnn.Device
) -> ttnn.Tensor:
    """HF ``_prepare_4d_causal_attention_mask`` (causal + key padding) on device → ``[B, 1, S, S]`` bf16."""
    causal_4d = build_causal_mask_4d(batch, seq, device)
    if attention_mask_2d is None:
        return ensure_tile_bf16_sdpa_mask(causal_4d)
    pad_add_2d = key_padding_additive(attention_mask_2d, device=device)
    pad_add_4d = ttnn.reshape(pad_add_2d, [batch, 1, 1, seq])
    pad_add_expanded = ttnn.expand(pad_add_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    combined = ttnn.add(causal_4d, pad_add_expanded, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ensure_tile_bf16_sdpa_mask(combined)
    ttnn.deallocate(pad_add_2d)
    for t in (pad_add_4d, pad_add_expanded, causal_4d):
        if t is not out:
            ttnn.deallocate(t)
    return out


def build_cross_attn_mask_4d(encoder_pad_mask_2d: ttnn.Tensor, *, tgt_seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_attention_mask`` for cross-attn → ``[B, 1, tgt_seq, src_seq]`` bf16."""
    batch = int(encoder_pad_mask_2d.shape[0])
    src_seq = int(encoder_pad_mask_2d.shape[1])
    add_2d = key_padding_additive(encoder_pad_mask_2d, device=device)
    add_4d = ttnn.reshape(add_2d, [batch, 1, 1, src_seq])
    if tgt_seq == 1:
        # ``ttnn.expand`` from ``[B, 1, 1, S]`` to ``[B, 1, 1, S]`` is a no-op view that shares
        # storage with ``add_2d``. Deallocating ``add_2d`` afterwards would free the returned
        # mask and trip SDPA's internal ``multiply(mask, scale)`` on the first decode step.
        # ``ensure_tile_bf16_sdpa_mask`` either returns ``add_4d`` as-is (already TILE bf16) or
        # allocates a tile-converted copy and deallocates ``add_4d`` itself — both ownership-safe.
        return ensure_tile_bf16_sdpa_mask(add_4d)
    expanded = ttnn.expand(add_4d, [batch, 1, tgt_seq, src_seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ensure_tile_bf16_sdpa_mask(expanded)
    ttnn.deallocate(add_2d)
    if add_4d is not out:
        ttnn.deallocate(add_4d)
    return out


def build_encoder_self_mask_4d(attention_mask_2d: ttnn.Tensor, *, device: ttnn.Device) -> ttnn.Tensor:
    """HF ``_prepare_4d_attention_mask`` for encoder self-attn → ``[B, 1, S, S]`` bf16."""
    batch = int(attention_mask_2d.shape[0])
    seq = int(attention_mask_2d.shape[1])
    add_2d = key_padding_additive(attention_mask_2d, device=device)
    add_4d = ttnn.reshape(add_2d, [batch, 1, 1, seq])
    if seq == 1:
        # Same use-after-free guard as ``build_cross_attn_mask_4d``: when ``seq == 1`` the
        # ``ttnn.expand`` becomes a no-op view that shares storage with ``add_2d``, so a
        # later ``deallocate(add_2d)`` would free the returned mask. Defer ownership to
        # ``ensure_tile_bf16_sdpa_mask`` which either returns ``add_4d`` as-is or copies
        # and deallocates the input — both ownership-safe.
        return ensure_tile_bf16_sdpa_mask(add_4d)
    expanded = ttnn.expand(add_4d, [batch, 1, seq, seq], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ensure_tile_bf16_sdpa_mask(expanded)
    ttnn.deallocate(add_2d)
    if add_4d is not out:
        ttnn.deallocate(add_4d)
    return out


def encoder_self_additive_mask_all_zeros_4d(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """Additive encoder self-attention mask when every position is valid (all keys visible)."""
    zeros = ttnn.zeros(
        [batch, 1, seq, seq],
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return ensure_tile_bf16_sdpa_mask(zeros)


def pad_input_ids_to(input_ids: ttnn.Tensor, padded_seq: int, pad_id: int, device: ttnn.Device) -> ttnn.Tensor:
    """Right-pad ``[B, S]`` uint32 to ``[B, padded_seq]`` with ``pad_id`` (on device, ``ttnn.concat``)."""
    bsz = int(input_ids.shape[0])
    seq = int(input_ids.shape[1])
    if padded_seq == seq:
        return input_ids
    pad_tail = ttnn.full(
        [bsz, padded_seq - seq],
        float(pad_id),
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded = ttnn.concat([input_ids, pad_tail], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(pad_tail)
    return padded


def pad_mask_to(mask: ttnn.Tensor, padded_seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """Right-pad ``[B, S]`` uint32 attention mask to ``[B, padded_seq]`` with 0 (on device)."""
    bsz = int(mask.shape[0])
    seq = int(mask.shape[1])
    if padded_seq == seq:
        return mask
    zeros = ttnn.full(
        [bsz, padded_seq - seq],
        0.0,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    padded = ttnn.concat([mask, zeros], dim=1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(zeros)
    return padded


def ones_mask(batch: int, seq: int, device: ttnn.Device) -> ttnn.Tensor:
    """``[B, S]`` uint32 all-ones (real-position mask) — used when caller omits ``attention_mask``."""
    return ttnn.full(
        [batch, seq],
        1.0,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def build_ln_sharded_config(
    device: ttnn.Device,
    m_tiles: int,
    n_tiles: int,
    cache: dict[Tuple[int, int], Tuple[ttnn.MemoryConfig, ttnn.LayerNormShardedMultiCoreProgramConfig]],
) -> Tuple[ttnn.MemoryConfig, ttnn.LayerNormShardedMultiCoreProgramConfig]:
    """Width-/block-sharded LN program config + memory config for ``[M_tiles, N_tiles]`` tile shape."""
    key = (m_tiles, n_tiles)
    cached = cache.get(key)
    if cached is not None:
        return cached

    device_grid = device.compute_with_storage_grid_size()
    grid_x = device_grid.x
    while grid_x > 1 and n_tiles % grid_x != 0:
        grid_x -= 1
    block_w = n_tiles // grid_x

    grid_y = min(device_grid.y, m_tiles)
    while grid_y > 1 and m_tiles % grid_y != 0:
        grid_y -= 1
    block_h = m_tiles // grid_y

    subblock_w = min(block_w, 4)
    while block_w % subblock_w != 0:
        subblock_w -= 1

    program_config = ttnn.LayerNormShardedMultiCoreProgramConfig(
        compute_with_storage_grid_size=(grid_x, grid_y),
        subblock_w=subblock_w,
        block_h=block_h,
        block_w=block_w,
        inplace=False,
    )
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        [block_h * TILE, block_w * TILE],
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    memory_config = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.WIDTH_SHARDED if grid_y == 1 else ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        shard_spec,
    )
    cached = (memory_config, program_config)
    cache[key] = cached
    return cached


def sdpa_program_config(
    device: ttnn.Device,
    seq_q: int,
    seq_k: int,
    cache: dict[Any, ttnn.SDPAProgramConfig],
    *,
    large_chunks: bool = True,
) -> ttnn.SDPAProgramConfig:
    """Chunk sizes for ``ttnn.transformer.scaled_dot_product_attention`` (cached per caller dict)."""
    key: Any = (seq_q, seq_k) if large_chunks else (seq_q, seq_k, large_chunks)
    cached = cache.get(key)
    if cached is not None:
        return cached

    if large_chunks:
        q_chunk = max(64, min(256, nearest_32(seq_q)))
        k_chunk = max(64, min(256, nearest_32(seq_k)))
    else:
        q_chunk = max(32, min(256, nearest_32(seq_q)))
        k_chunk = max(32, min(256, nearest_32(seq_k)))
    out = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    cache[key] = out
    return out

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""KV chunk address table builder for **paged** prefill KV (``kv_cache_prefill_only_paged.py``).

Targets caches allocated by ``tt/attention/kv_cache_prefill_only_paged.py``:

  * shape per device:
      ``[num_blocks_local, num_kv_heads // TP, block_size, head_dim]``
  * ``ShardTensor2dMesh(dims=(sp_axis, None))`` when ``sp_shard_blocks=True`` (default)
  * **Production cache:** INTERLEAVED DRAM via ``make_paged_kv_memory_config`` in
    ``kv_cache_prefill_only_paged.init_kv_cache`` (``paged_fill_cache`` / chunked SDPA)
  * outer SP ownership — global token ``G`` maps to row ``G // (max_seq_len // SP)``;
    local token index within that row selects block ``local // block_size`` in the
    slot's block range
  * slots — ``num_blocks_local == num_slots × blocks_per_sp_shard``; slot index matches
    ``paged_fill_cache(..., batch_idx=slot)``

Addressing uses **INTERLEAVED DRAM page indices** (``interleaved_chunk_first_page_index``,
``noc_addr_for_interleaved_page``). Migration readback gathers consecutive width tiles across
banks via ``KvChunkAddressTable.read_device_chunk_interleaved``.
"""

from __future__ import annotations

import socket
from typing import Sequence

from loguru import logger

import ttnn
from models.demos.gpt_oss_d_p.tt.attention.kv_cache_prefill_only_paged import blocks_per_sp_shard

from .kv_cache_table import (
    BFP8_TILE_BYTES,
    BH_NUM_DRAM_BANKS,
    CHUNK_N_TOKENS,
    KvChunkAddressTableMulti,
    KvChunkTableGroup,
    _device_tensor_index,
    _mesh_coordinate,
    compute_kv_chunk_size_bytes,
    global_head_to_tp_shard,
    global_position_to_sp_local,
    make_kv_chunk_table_configs,
    noc_addr_for_bank_chunk,
)


def width_tiles_for_head_dim(head_dim: int) -> int:
    tile = ttnn.TILE_SIZE
    return (head_dim + tile - 1) // tile


def interleaved_chunk_first_page_index(
    *,
    block_idx: int,
    head_idx: int,
    local_position: int,
    block_size: int,
    head_dim: int,
    num_heads: int,
    num_blocks: int,
) -> int:
    """First interleaved TILE page index for a paged KV chunk in ``[blocks, heads, block_size, head_dim]``.

    Page order matches TTNN TILE layout: width tiles vary fastest, then seq tiles within a block,
    then heads, then blocks.
    """
    tile = ttnn.TILE_SIZE
    wt = width_tiles_for_head_dim(head_dim)
    st = (block_size + tile - 1) // tile
    s_tile = local_position // tile
    return ((block_idx * num_heads + head_idx) * st + s_tile) * wt


def noc_addr_for_interleaved_page(
    dram_base: int,
    page_index: int,
    *,
    num_banks: int = BH_NUM_DRAM_BANKS,
    aligned_page_size: int = BFP8_TILE_BYTES,
) -> int:
    """Pack NOC addr for the first byte of an interleaved DRAM TILE page."""
    bank_id = page_index % num_banks
    bank_offset = (page_index // num_banks) * aligned_page_size
    return noc_addr_for_bank_chunk(dram_base, bank_id, bank_offset)


def canonical_paged_cache_shape(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    """Normalize per-device paged KV shape to ``[num_blocks, num_heads, block_size, head_dim]``.

    ``init_kv_cache(..., sp_shard_blocks=True)`` uploads host
    ``[SP, num_blocks, num_heads, block_size, head_dim]``; ``get_device_tensors`` on each
    device often reports rank 5 with a leading singleton SP dim:
    ``[1, num_blocks, num_heads, block_size, head_dim]``.
    """
    shape = tuple(shape)
    if len(shape) == 4:
        return shape  # type: ignore[return-value]
    if len(shape) == 5 and shape[0] == 1:
        return shape[1], shape[2], shape[3], shape[4]
    raise ValueError(f"expected paged KV cache rank 4 or rank 5 with leading SP singleton, got shape {shape}")


def read_paged_device_chunk(
    table: ttnn.experimental.disaggregation.KvChunkAddressTable,
    *,
    layer: int,
    position: int,
    slot: int,
) -> bytes:
    """Read one migration chunk from an INTERLEAVED paged KV cache (multi-bank gather)."""
    return table.read_device_chunk_interleaved(layer, position, slot)


def _validate_paged_cache_tensor(
    cache: ttnn.Tensor,
    *,
    block_size: int,
    num_slots: int,
    max_seq_len: int,
    sp_len: int,
    head_dim: int,
) -> tuple[int, int]:
    """Return ``(num_blocks_local, num_local_kv_heads)`` from a per-device cache tensor."""
    num_blocks_local, num_local_kv_heads, cache_block_size, cache_head_dim = canonical_paged_cache_shape(
        tuple(cache.shape)
    )
    if cache_block_size != block_size:
        raise ValueError(f"cache block_size {cache_block_size} != expected {block_size}")
    if cache_head_dim != head_dim:
        raise ValueError(f"cache head_dim {cache_head_dim} != expected {head_dim}")
    expected_blocks = num_slots * blocks_per_sp_shard(max_seq_len, sp_len, block_size)
    if num_blocks_local != expected_blocks:
        raise ValueError(
            f"cache num_blocks {num_blocks_local} != expected {expected_blocks} "
            f"(num_slots={num_slots}, max_seq_len={max_seq_len}, sp={sp_len}, block_size={block_size})"
        )
    return num_blocks_local, num_local_kv_heads


def _populate_group_table_paged(
    *,
    lookup_table: ttnn.experimental.disaggregation.KvChunkAddressTable,
    mesh_device,
    sp_axis: int,
    tp_axis: int,
    sp_len: int,
    tp_len: int,
    num_transformer_layers: int,
    max_seq_len: int,
    chunk_n_tokens: int,
    chunk_size_bytes: int,
    tp_col: int,
    head_idx_local: int,
    num_blocks_local: int,
    num_local_heads: int,
    block_size: int,
    head_dim: int,
    num_slots: int,
    layer_caches: Sequence[ttnn.Tensor],
    group_index: int,
    host_name: str,
) -> None:
    singleton_group_by_coord: dict[tuple[int, int], ttnn.experimental.disaggregation.DeviceGroupIndex] = {}
    blocks_per_slot = num_blocks_local // num_slots

    def singleton_group(sp_row: int, col: int):
        key = (sp_row, col)
        if key not in singleton_group_by_coord:
            coord = _mesh_coordinate(sp_row, col, sp_axis, tp_axis)
            fabric_node_id = mesh_device.get_fabric_node_id(coord)
            singleton_group_by_coord[key] = lookup_table.add_device_group([fabric_node_id])
            lookup_table.set_fabric_node_host(fabric_node_id, host_name=host_name)
        return singleton_group_by_coord[key]

    for t_layer in range(num_transformer_layers):
        cache = layer_caches[t_layer]
        device_tensors = ttnn.get_device_tensors(cache)
        assert len(device_tensors) == sp_len * tp_len, (
            f"group {group_index} layer {t_layer}: expected {sp_len * tp_len} device tensors, "
            f"got {len(device_tensors)}"
        )

        for slot in range(num_slots):
            for position in range(0, max_seq_len, chunk_n_tokens):
                sp_row, local_position = global_position_to_sp_local(position, max_seq_len, sp_len)
                group_idx = singleton_group(sp_row, tp_col)

                block_in_slot = local_position // block_size
                block_idx = slot * blocks_per_slot + block_in_slot
                page_index = interleaved_chunk_first_page_index(
                    block_idx=block_idx,
                    head_idx=head_idx_local,
                    local_position=local_position % block_size,
                    block_size=block_size,
                    head_dim=head_dim,
                    num_heads=num_local_heads,
                    num_blocks=num_blocks_local,
                )

                dt = device_tensors[_device_tensor_index(sp_row, tp_col, tp_len)]
                dram_base = dt.buffer_address()
                aligned_page_size = dt.buffer_aligned_page_size()

                location = ttnn.experimental.disaggregation.KvCacheLocation()
                location.noc_addr = noc_addr_for_interleaved_page(
                    dram_base, page_index, aligned_page_size=aligned_page_size
                )
                location.size_bytes = chunk_size_bytes
                location.device_group_index = group_idx
                lookup_table.set(t_layer, position, slot, location)


def create_kv_chunk_address_table_gpt_oss_prefill_paged(
    *,
    mesh_device,
    mesh_shape,
    sp_axis: int,
    tp_axis: int,
    kv_caches: Sequence[Sequence[ttnn.Tensor]],
    num_transformer_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    block_size: int,
    num_slots: int = 1,
    chunk_n_tokens: int = CHUNK_N_TOKENS,
) -> KvChunkAddressTableMulti:
    """Populate multi-config K/V chunk address tables for paged prefill KV caches.

    Same group layout as ``create_kv_chunk_address_table_gpt_oss_prefill`` (one table
    per K/V × global head). Requires ``kv_caches`` from
    ``kv_cache_prefill_only_paged.init_kv_cache`` with ``sp_shard_blocks=True``.

    Args:
        kv_caches: ``kv_caches[layer] == [k_cache, v_cache]`` mesh tensors.
        block_size: Paged block size (must match cache dim 2, typically 64).
        max_seq_len: Global sequence length.
        num_slots: ``max_local_batch_size`` / migration slot count.
    """
    if num_slots < 1:
        raise ValueError(f"num_slots must be >= 1, got {num_slots}")
    if block_size % chunk_n_tokens != 0:
        raise ValueError(f"block_size ({block_size}) must be a multiple of chunk_n_tokens ({chunk_n_tokens})")

    tp_len = mesh_shape[tp_axis]
    sp_len = mesh_shape[sp_axis]
    num_kv_heads_local = num_kv_heads // tp_len
    if num_kv_heads % tp_len != 0:
        raise ValueError(f"num_kv_heads ({num_kv_heads}) must be divisible by TP ({tp_len})")
    if max_seq_len % sp_len != 0:
        raise ValueError(f"max_seq_len ({max_seq_len}) must be divisible by SP ({sp_len})")

    assert (
        len(kv_caches) == num_transformer_layers
    ), f"kv_caches has {len(kv_caches)} layers but num_transformer_layers={num_transformer_layers}"

    # Validate cache layout from layer 0 K tensor on device 0.
    sample_tensors = ttnn.get_device_tensors(kv_caches[0][0])
    num_blocks_local, num_local_kv_heads = _validate_paged_cache_tensor(
        sample_tensors[0],
        block_size=block_size,
        num_slots=num_slots,
        max_seq_len=max_seq_len,
        sp_len=sp_len,
        head_dim=head_dim,
    )

    config_groups = make_kv_chunk_table_configs(
        num_transformer_layers=num_transformer_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        num_slots=num_slots,
        chunk_n_tokens=chunk_n_tokens,
    )
    bundle = KvChunkAddressTableMulti([cfg for _, _, cfg in config_groups])
    host_name = socket.gethostname()
    chunk_size_bytes = compute_kv_chunk_size_bytes(head_dim, chunk_n_tokens)

    for group_index, (kv_kind, global_head, group_cfg) in enumerate(config_groups):
        tp_col, head_idx_local = global_head_to_tp_shard(global_head, num_kv_heads_local)
        assert tp_col < tp_len, f"global_head {global_head} maps to tp_col {tp_col} >= tp_len {tp_len}"
        assert head_idx_local < num_local_kv_heads, (
            f"global_head {global_head} head_idx_local {head_idx_local} >= " f"num_local_kv_heads {num_local_kv_heads}"
        )

        group_table = ttnn.experimental.disaggregation.KvChunkAddressTable(group_cfg)
        layer_caches = [kv_caches[t][int(kv_kind)] for t in range(num_transformer_layers)]

        _populate_group_table_paged(
            lookup_table=group_table,
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            sp_len=sp_len,
            tp_len=tp_len,
            num_transformer_layers=num_transformer_layers,
            max_seq_len=max_seq_len,
            chunk_n_tokens=chunk_n_tokens,
            chunk_size_bytes=chunk_size_bytes,
            tp_col=tp_col,
            head_idx_local=head_idx_local,
            num_blocks_local=num_blocks_local,
            num_local_heads=num_local_kv_heads,
            block_size=block_size,
            head_dim=head_dim,
            num_slots=num_slots,
            layer_caches=layer_caches,
            group_index=group_index,
            host_name=host_name,
        )

        bundle.add_group(
            KvChunkTableGroup(
                group_index=group_index,
                kv_kind=kv_kind,
                global_head=global_head,
                tp_col=tp_col,
                config=group_cfg,
                table=group_table,
            )
        )

    blocks_per_row = blocks_per_sp_shard(max_seq_len, sp_len, block_size)
    logger.info(
        f"[gpt-oss-d-p-kv-table-paged] multi-config: transformer_layers={num_transformer_layers} "
        f"num_groups={len(bundle.groups)} (2×{num_kv_heads} K/V×head) "
        f"sp={sp_len} tp={tp_len} slots={num_slots} seq={max_seq_len} "
        f"block_size={block_size} blocks_per_sp_row={blocks_per_row} "
        f"chunk_bytes={chunk_size_bytes} "
        f"total_entries={sum(g.table.total_entries() for g in bundle.groups)}"
    )
    return bundle

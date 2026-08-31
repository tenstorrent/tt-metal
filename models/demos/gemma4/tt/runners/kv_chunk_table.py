# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Source table for Gemma 4 global and sliding migration caches."""

from __future__ import annotations

import socket

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.gemma4.tt.attention.global_migration import ROW_DIM as GLOBAL_ROW_DIM
from models.demos.gemma4.tt.attention.global_migration import TOKENS_PER_CHUNK
from models.demos.gemma4.tt.attention.sliding_migration import HEAD_DIM as SLIDING_ROW_DIM

BFP8_TILE_BYTES = 1088
GLOBAL_CHUNK_SIZE_BYTES = (GLOBAL_ROW_DIM // ttnn.TILE_SIZE) * BFP8_TILE_BYTES
SLIDING_CHUNK_SIZE_BYTES = (SLIDING_ROW_DIM // ttnn.TILE_SIZE) * BFP8_TILE_BYTES

GLOBAL_LABELS = tuple(f"kv_h{head}" for head in range(4))
SLIDING_K_LABELS = tuple(f"k_h{head}" for head in range(16))
SLIDING_V_LABELS = tuple(f"v_h{head}" for head in range(16))
CONFIG_LABELS = GLOBAL_LABELS + SLIDING_K_LABELS + SLIDING_V_LABELS
# KvChunkAddressTable's protobuf path uses a lexicographically ordered map.
CONFIG_NAMES = tuple(f"{idx:02d}_{label}" for idx, label in enumerate(CONFIG_LABELS))


def iter_source_chunk_locations(
    *,
    seq_len: int,
    chunk_size: int,
    sp: int,
    num_users: int,
    semantic_layers: tuple[int, ...],
    num_banks: int,
    chunk_size_bytes: int,
    heads_per_device: int = 1,
    local_head: int = 0,
):
    """Yield the exact ROUND_ROBIN_1D NdShard address walk for one local head."""
    if not 0 <= local_head < heads_per_device:
        raise ValueError(f"local_head {local_head} outside [0, {heads_per_device})")
    local_tokens = chunk_size // sp
    blocks_local = (seq_len // sp) // TOKENS_PER_CHUNK
    blocks_per_prefill_chunk = local_tokens // TOKENS_PER_CHUNK
    num_prefill_chunks = seq_len // chunk_size

    for cp_row in range(sp):
        for slot in range(num_users):
            for compact_layer, semantic_layer in enumerate(semantic_layers):
                for prefill_chunk in range(num_prefill_chunks):
                    for block_in_chunk in range(blocks_per_prefill_chunk):
                        shard = (
                            ((slot * len(semantic_layers) + compact_layer) * heads_per_device + local_head)
                            * blocks_local
                            + prefill_chunk * blocks_per_prefill_chunk
                            + block_in_chunk
                        )
                        position = (
                            prefill_chunk * chunk_size + cp_row * local_tokens + block_in_chunk * TOKENS_PER_CHUNK
                        )
                        yield (
                            cp_row,
                            slot,
                            semantic_layer,
                            position,
                            shard % num_banks,
                            (shard // num_banks) * chunk_size_bytes,
                        )


def _table_config(*, num_layers: int, max_seq_len: int, num_users: int, chunk_size_bytes: int):
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = num_layers
    config.max_sequence_length = max_seq_len
    config.num_slots = num_users
    config.chunk_n_tokens = TOKENS_PER_CHUNK
    config.chunk_size_bytes = chunk_size_bytes
    return config


def build_kv_chunk_address_table(
    *,
    mesh_device,
    global_cache,
    sliding_cache,
    seq_len: int,
    sliding_seq_len: int,
    mesh_shape: tuple,
    sp_axis: int,
    num_users: int,
    chunk_size: int,
    global_layers: tuple[int, ...],
):
    """Build the 36-config mixed global/sliding source table."""
    tp_axis = 1 - sp_axis
    sp = int(mesh_shape[sp_axis])
    tp = int(mesh_shape[tp_axis])
    if sp_axis != 0 or tp != 4:
        raise ValueError(f"Gemma 4 source table requires CPx/TP4, got mesh={mesh_shape}, sp_axis={sp_axis}")
    for name, tensor in (
        ("global", global_cache.kv),
        ("sliding K", sliding_cache.k),
        ("sliding V", sliding_cache.v),
    ):
        if tensor.dtype != ttnn.bfloat8_b:
            raise ValueError(f"Gemma 4 {name} source cache requires BFP8, got {tensor.dtype}")
    for label, extent in (("global", seq_len), ("sliding", sliding_seq_len)):
        if extent % chunk_size or chunk_size % (sp * TOKENS_PER_CHUNK):
            raise ValueError(
                f"{label} seq_len={extent} must divide by chunk_size={chunk_size}, and chunk_size must "
                f"contain CP-aligned {TOKENS_PER_CHUNK}-token blocks"
            )

    expected_global = tuple(range(5, 60, 6))
    if tuple(global_layers) != expected_global:
        raise ValueError(f"unexpected global layer map: {global_layers}")
    sliding_layers = tuple(layer for layer in range(60) if layer not in global_layers)
    if int(global_cache.kv.shape[0]) != num_users * len(global_layers):
        raise ValueError("global migration cache has the wrong user/layer row count")
    if int(sliding_cache.k.shape[0]) != num_users * len(sliding_layers):
        raise ValueError("sliding migration cache has the wrong user/layer row count")

    configs = {}
    for idx, name in enumerate(CONFIG_NAMES):
        is_global = idx < len(GLOBAL_LABELS)
        configs[name] = _table_config(
            num_layers=60,
            max_seq_len=seq_len if is_global else sliding_seq_len,
            num_users=num_users,
            chunk_size_bytes=GLOBAL_CHUNK_SIZE_BYTES if is_global else SLIDING_CHUNK_SIZE_BYTES,
        )
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs)
    actual_names = tuple(table.config_name(i) for i in range(table.num_configs()))
    if actual_names != CONFIG_NAMES:
        raise RuntimeError(f"protobuf config ordering changed: expected {CONFIG_NAMES}, got {actual_names}")

    num_banks = get_num_dram_banks(mesh_device)
    host_name = socket.gethostname()
    mapped_hosts = set()

    def _populate(
        *,
        config_id: int,
        tp_column: int,
        local_head: int,
        heads_per_device: int,
        tensor,
        semantic_layers: tuple[int, ...],
        extent: int,
        chunk_bytes: int,
    ):
        groups = {}
        for cp_row in range(sp):
            fnid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(cp_row, tp_column))
            groups[cp_row] = table.add_device_group([fnid])
            host_key = (int(fnid.mesh_id), int(fnid.chip_id))
            if host_key not in mapped_hosts:
                table.set_fabric_node_host(fnid, host_name=host_name)
                mapped_hosts.add(host_key)

        base_addr = int(tensor.buffer_address())
        for cp_row, slot, semantic_layer, position, bank_id, bank_offset in iter_source_chunk_locations(
            seq_len=extent,
            chunk_size=chunk_size,
            sp=sp,
            num_users=num_users,
            semantic_layers=semantic_layers,
            num_banks=num_banks,
            chunk_size_bytes=chunk_bytes,
            heads_per_device=heads_per_device,
            local_head=local_head,
        ):
            location = ttnn.experimental.disaggregation.KvCacheLocation()
            location.noc_addr = (bank_id << 32) | (base_addr + bank_offset)
            location.size_bytes = chunk_bytes
            location.device_group_index = groups[cp_row]
            table.set(semantic_layer, position, slot, location, config_id)

    for head in range(4):
        _populate(
            config_id=head,
            tp_column=head,
            local_head=0,
            heads_per_device=1,
            tensor=global_cache.kv,
            semantic_layers=global_layers,
            extent=seq_len,
            chunk_bytes=GLOBAL_CHUNK_SIZE_BYTES,
        )
    for head in range(16):
        common = dict(
            tp_column=head // 4,
            local_head=head % 4,
            heads_per_device=4,
            semantic_layers=sliding_layers,
            extent=sliding_seq_len,
            chunk_bytes=SLIDING_CHUNK_SIZE_BYTES,
        )
        _populate(config_id=4 + head, tensor=sliding_cache.k, **common)
        _populate(config_id=20 + head, tensor=sliding_cache.v, **common)

    logger.info(
        f"[gemma4-kv-table] configs={len(CONFIG_NAMES)}, global_layers={global_layers}, "
        f"sliding_layers={len(sliding_layers)}, entries={table.total_entries()}"
    )
    return table


def build_and_serialize_kv_chunk_table(*, path: str, **kwargs) -> str:
    table = build_kv_chunk_address_table(**kwargs)
    ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
    logger.info(f"[migration] Gemma 4 mixed KV table serialized to {path}")
    return path

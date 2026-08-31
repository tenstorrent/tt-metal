# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Source address table for Gemma 4's compact global migration cache."""

from __future__ import annotations

import socket

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.gemma4.tt.attention.global_migration import ROW_DIM, TOKENS_PER_CHUNK

BFP8_TILE_BYTES = 1088
CHUNK_SIZE_BYTES = (ROW_DIM // ttnn.TILE_SIZE) * BFP8_TILE_BYTES
CONFIG_NAMES = tuple(f"kv_h{head}" for head in range(4))


def iter_source_chunk_locations(
    *,
    seq_len: int,
    chunk_size: int,
    sp: int,
    num_users: int,
    global_layers: tuple[int, ...],
    num_banks: int,
):
    """Yield the ND-shard bank walk in physical compact-cache order."""
    local_tokens = chunk_size // sp
    num_prefill_chunks = seq_len // chunk_size
    for cp_row in range(sp):
        bank_id = 0
        bank_offset = 0
        for slot in range(num_users):
            for semantic_layer in global_layers:
                for prefill_chunk in range(num_prefill_chunks):
                    token_start = prefill_chunk * chunk_size + cp_row * local_tokens
                    token_end = token_start + local_tokens
                    for position in range(token_start, token_end, TOKENS_PER_CHUNK):
                        yield cp_row, slot, semantic_layer, position, bank_id, bank_offset
                        bank_id = (bank_id + 1) % num_banks
                        if bank_id == 0:
                            bank_offset += CHUNK_SIZE_BYTES


def _table_config(*, num_layers: int, max_seq_len: int, num_users: int):
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = num_layers
    config.max_sequence_length = max_seq_len
    config.num_slots = num_users
    config.chunk_n_tokens = TOKENS_PER_CHUNK
    config.chunk_size_bytes = CHUNK_SIZE_BYTES
    return config


def build_kv_chunk_address_table(
    *,
    mesh_device,
    cache,
    seq_len: int,
    mesh_shape: tuple,
    sp_axis: int,
    num_users: int,
    chunk_size: int,
    global_layers: tuple[int, ...],
):
    """Build four per-head configs while retaining semantic 60-layer rows."""
    tp_axis = 1 - sp_axis
    sp = int(mesh_shape[sp_axis])
    tp = int(mesh_shape[tp_axis])
    if mesh_shape != (8, 4) or sp_axis != 0 or tp != 4:
        raise ValueError(f"Gemma 4 source table requires CP8/TP4, got mesh={mesh_shape}, sp_axis={sp_axis}")
    if cache.kv.dtype != ttnn.bfloat8_b:
        raise ValueError(f"Gemma 4 source table requires BFP8, got {cache.kv.dtype}")
    if seq_len % chunk_size or chunk_size % (sp * TOKENS_PER_CHUNK):
        raise ValueError(
            f"seq_len={seq_len} must divide by chunk_size={chunk_size}, and chunk_size must divide "
            f"into CP-aligned {TOKENS_PER_CHUNK}-token blocks"
        )
    if tuple(global_layers) != tuple(range(5, 60, 6)):
        raise ValueError(f"unexpected global layer map: {global_layers}")
    if int(cache.kv.shape[0]) != num_users * len(global_layers):
        raise ValueError(
            f"migration cache rows {cache.kv.shape[0]} != users({num_users}) * globals({len(global_layers)})"
        )

    num_semantic_layers = max(global_layers) + 1
    configs = {
        name: _table_config(num_layers=num_semantic_layers, max_seq_len=seq_len, num_users=num_users)
        for name in CONFIG_NAMES
    }
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs)
    if tuple(table.config_name(i) for i in range(table.num_configs())) != CONFIG_NAMES:
        raise RuntimeError("protobuf config ordering changed; expected kv_h0..kv_h3")

    base_addr = int(cache.kv.buffer_address())
    num_banks = get_num_dram_banks(mesh_device)
    host_name = socket.gethostname()
    mapped_hosts = set()

    for config_id, head in enumerate(range(4)):
        groups = {}
        for cp_row in range(sp):
            fnid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(cp_row, head))
            groups[cp_row] = table.add_device_group([fnid])
            host_key = (int(fnid.mesh_id), int(fnid.chip_id))
            if host_key not in mapped_hosts:
                table.set_fabric_node_host(fnid, host_name=host_name)
                mapped_hosts.add(host_key)

        for cp_row, slot, semantic_layer, position, bank_id, bank_offset in iter_source_chunk_locations(
            seq_len=seq_len,
            chunk_size=chunk_size,
            sp=sp,
            num_users=num_users,
            global_layers=global_layers,
            num_banks=num_banks,
        ):
            location = ttnn.experimental.disaggregation.KvCacheLocation()
            location.noc_addr = (bank_id << 32) | (base_addr + bank_offset)
            location.size_bytes = CHUNK_SIZE_BYTES
            location.device_group_index = groups[cp_row]
            table.set(semantic_layer, position, slot, location, config_id)

    logger.info(
        f"[gemma4-kv-table] configs={CONFIG_NAMES}, globals={global_layers}, "
        f"entries={table.total_entries()}, chunk_bytes={CHUNK_SIZE_BYTES}"
    )
    return table


def build_and_serialize_kv_chunk_table(*, path: str, **kwargs) -> str:
    table = build_kv_chunk_address_table(**kwargs)
    ttnn.experimental.disaggregation.export_to_protobuf_file(table, path)
    logger.info(f"[migration] Gemma 4 global KV table serialized to {path}")
    return path

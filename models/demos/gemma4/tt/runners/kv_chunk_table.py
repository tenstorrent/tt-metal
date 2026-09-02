# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Raw-transfer address table for Gemma 4 durable ring caches."""

from __future__ import annotations

import socket
import zlib

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks, serialize_prebuilt_kv_chunk_table
from models.demos.gemma4.tt.attention.global_kv_cache import GLOBAL_PACKED_DIM, SLIDING_HEAD_DIM
from models.demos.gemma4.tt.attention.ring_prefill import TILE_HEIGHT, PackedRingKVCache
from models.demos.gemma4.tt.runners.kv_caches import Gemma4KvCaches

_BFP8_TILE_BYTES = 1088
GLOBAL_CHUNK_BYTES = GLOBAL_PACKED_DIM // ttnn.TILE_SIZE * _BFP8_TILE_BYTES
SLIDING_CHUNK_BYTES = SLIDING_HEAD_DIM // ttnn.TILE_SIZE * _BFP8_TILE_BYTES
GLOBAL_CONFIGS = tuple(f"{idx:02d}_global_h{idx}" for idx in range(4))
SLIDING_K_CONFIGS = tuple(f"{idx + 4:02d}_sliding_k_h{idx}" for idx in range(16))
SLIDING_V_CONFIGS = tuple(f"{idx + 20:02d}_sliding_v_h{idx}" for idx in range(16))
CONFIG_NAMES = GLOBAL_CONFIGS + SLIDING_K_CONFIGS + SLIDING_V_CONFIGS


def worker_host_name(hostname: str | None = None) -> str:
    """Stable host key expected by the migration worker."""
    hostname = socket.gethostname() if hostname is None else hostname
    return f"host-{zlib.crc32(hostname.encode()) & 0x7FFFFFFF:08x}"


def iter_cache_chunk_locations(
    *,
    seq_len: int,
    chunk_size: int,
    sp: int,
    num_users: int,
    heads_per_device: int,
    local_head: int,
    num_banks: int,
    chunk_size_bytes: int,
):
    """Yield the ROUND_ROBIN_1D address walk for one head in one layer buffer."""
    if not 0 <= local_head < heads_per_device:
        raise ValueError(f"local_head {local_head} outside [0, {heads_per_device})")
    if seq_len % chunk_size or chunk_size % (sp * TILE_HEIGHT):
        raise ValueError("sequence and prefill chunks must align to CP-local 32-token rows")
    local_seq = seq_len // sp
    local_chunk = chunk_size // sp
    blocks_local = local_seq // TILE_HEIGHT
    blocks_per_chunk = local_chunk // TILE_HEIGHT
    for cp_row in range(sp):
        for slot in range(num_users):
            for prefill_chunk in range(seq_len // chunk_size):
                for block_in_chunk in range(blocks_per_chunk):
                    shard = (
                        (slot * heads_per_device + local_head) * blocks_local
                        + prefill_chunk * blocks_per_chunk
                        + block_in_chunk
                    )
                    position = prefill_chunk * chunk_size + cp_row * local_chunk + block_in_chunk * TILE_HEIGHT
                    yield cp_row, slot, position, shard % num_banks, shard // num_banks * chunk_size_bytes


def _config(*, num_layers, max_seq_len, num_users, chunk_size_bytes):
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = num_layers
    config.max_sequence_length = max_seq_len
    config.num_slots = num_users
    config.chunk_n_tokens = TILE_HEIGHT
    config.chunk_size_bytes = chunk_size_bytes
    return config


def build_kv_chunk_address_table(*, mesh_device, kv_caches: Gemma4KvCaches, chunk_size: int, sp_axis: int = 0):
    """Describe global packed rows and sliding K/V rows directly from compute caches."""
    if not isinstance(kv_caches, Gemma4KvCaches):
        raise TypeError(f"expected Gemma4KvCaches, got {type(kv_caches).__name__}")
    tp_axis = 1 - sp_axis
    sp = int(mesh_device.shape[sp_axis])
    tp = int(mesh_device.shape[tp_axis])
    if (sp, tp) != (8, 4) or kv_caches.sp != sp or kv_caches.tp != tp:
        raise ValueError(f"Gemma 4 migration currently requires CP8/TP4, got mesh={tuple(mesh_device.shape)}")
    num_layers = len(kv_caches)
    configs = {
        name: _config(
            num_layers=num_layers,
            max_seq_len=kv_caches.max_seq_len,
            num_users=kv_caches.num_users,
            chunk_size_bytes=GLOBAL_CHUNK_BYTES if idx < 4 else SLIDING_CHUNK_BYTES,
        )
        for idx, name in enumerate(CONFIG_NAMES)
    }
    table = ttnn.experimental.disaggregation.KvChunkAddressTable(configs)
    actual_names = tuple(table.config_name(i) for i in range(table.num_configs()))
    if actual_names != CONFIG_NAMES:
        raise RuntimeError(f"protobuf config ordering changed: expected {CONFIG_NAMES}, got {actual_names}")

    num_banks = get_num_dram_banks(mesh_device)
    mapped_hosts = set()
    group_cache = {}

    def device_group(cp_row, tp_column):
        key = (cp_row, tp_column)
        if key not in group_cache:
            fnid = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(cp_row, tp_column))
            group_cache[key] = table.add_device_group([fnid])
            host_key = (int(fnid.mesh_id), int(fnid.chip_id))
            if host_key not in mapped_hosts:
                table.set_fabric_node_host(fnid, host_name=worker_host_name())
                mapped_hosts.add(host_key)
        return group_cache[key]

    def populate(*, config_id, semantic_layer, tensor, tp_column, heads_per_device, local_head, chunk_bytes):
        base_addr = int(tensor.buffer_address())
        if tensor.dtype != ttnn.bfloat8_b:
            raise ValueError(f"migration cache must be BFP8_B, got {tensor.dtype}")
        for cp_row, slot, position, bank_id, bank_offset in iter_cache_chunk_locations(
            seq_len=kv_caches.max_seq_len,
            chunk_size=chunk_size,
            sp=sp,
            num_users=kv_caches.num_users,
            heads_per_device=heads_per_device,
            local_head=local_head,
            num_banks=num_banks,
            chunk_size_bytes=chunk_bytes,
        ):
            location = ttnn.experimental.disaggregation.KvCacheLocation()
            location.noc_addr = bank_id << 32 | base_addr + bank_offset
            location.size_bytes = chunk_bytes
            location.device_group_index = device_group(cp_row, tp_column)
            table.set(semantic_layer, position, slot, location, config_id)

    for layer_idx in kv_caches.global_layers:
        cache = kv_caches[layer_idx]
        if not isinstance(cache, PackedRingKVCache):
            raise TypeError(f"global layer {layer_idx} does not own PackedRingKVCache")
        for head in range(4):
            populate(
                config_id=head,
                semantic_layer=layer_idx,
                tensor=cache.kv,
                tp_column=head,
                heads_per_device=1,
                local_head=0,
                chunk_bytes=GLOBAL_CHUNK_BYTES,
            )
    for layer_idx in kv_caches.sliding_layers:
        cache_k, cache_v = kv_caches[layer_idx]
        for head in range(16):
            common = dict(
                semantic_layer=layer_idx,
                tp_column=head // 4,
                heads_per_device=4,
                local_head=head % 4,
                chunk_bytes=SLIDING_CHUNK_BYTES,
            )
            populate(config_id=4 + head, tensor=cache_k, **common)
            populate(config_id=20 + head, tensor=cache_v, **common)
    return table


def build_and_serialize_kv_chunk_table(*, path: str, **kwargs) -> str:
    table = build_kv_chunk_address_table(**kwargs)
    return serialize_prebuilt_kv_chunk_table(table=table, path=path)

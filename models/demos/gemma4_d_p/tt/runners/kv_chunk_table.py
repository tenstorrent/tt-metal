# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Raw-transfer address table for Gemma 4 durable ring caches."""

from __future__ import annotations

import hashlib
import socket
import zlib
from pathlib import Path
from typing import NamedTuple

from loguru import logger

import ttnn
from models.demos.common.prefill.runners.migration import get_num_dram_banks
from models.demos.gemma4_d_p.config import MeshConfig
from models.demos.gemma4_d_p.tt.attention.global_kv_cache import GLOBAL_PACKED_DIM, SLIDING_HEAD_DIM
from models.demos.gemma4_d_p.tt.attention.ring_prefill import TILE_HEIGHT, GlobalRingKVCache
from models.demos.gemma4_d_p.tt.runners.kv_caches import Gemma4KvCaches

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


def _cache_chunk_runs(
    *,
    seq_len: int,
    chunk_size: int,
    cp: int,
    num_users: int,
    heads_per_device: int,
    local_head: int,
):
    """Yield (cp_row, slot, first_position, first_shard, count) runs of consecutive 32-token blocks.

    chunk_size is one prefill width for every slot, or a {slot: width} mapping when slots are prefilled at
    different widths. The ring cache is block-cyclic with period = width, so each slot's walk uses its own.
    """
    if not 0 <= local_head < heads_per_device:
        raise ValueError(f"local_head {local_head} outside [0, {heads_per_device})")
    widths = chunk_size if isinstance(chunk_size, dict) else {slot: chunk_size for slot in range(num_users)}
    if sorted(widths) != list(range(num_users)):
        raise ValueError(f"need a prefill width for every slot in [0, {num_users}), got {sorted(widths)}")
    for width in set(widths.values()):
        if seq_len % width or width % (cp * TILE_HEIGHT):
            raise ValueError("sequence and prefill chunks must align to CP-local 32-token rows")
    blocks_local = seq_len // cp // TILE_HEIGHT
    for cp_row in range(cp):
        for slot in range(num_users):
            chunk_size = widths[slot]
            local_chunk = chunk_size // cp
            blocks_per_chunk = local_chunk // TILE_HEIGHT
            for prefill_chunk in range(seq_len // chunk_size):
                shard = (slot * heads_per_device + local_head) * blocks_local + prefill_chunk * blocks_per_chunk
                position = prefill_chunk * chunk_size + cp_row * local_chunk
                yield cp_row, slot, position, shard, blocks_per_chunk


def iter_cache_chunk_locations(*, num_banks: int, chunk_size_bytes: int, **walk):
    """Yield the ROUND_ROBIN_1D address walk for one head in one layer buffer."""
    for cp_row, slot, position, first_shard, count in _cache_chunk_runs(**walk):
        for shard in range(first_shard, first_shard + count):
            yield cp_row, slot, position, shard % num_banks, shard // num_banks * chunk_size_bytes
            position += TILE_HEIGHT


def _config(*, num_layers, max_seq_len, num_users, chunk_size_bytes):
    config = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    config.num_layers = num_layers
    config.max_sequence_length = max_seq_len
    config.num_slots = num_users
    config.chunk_n_tokens = TILE_HEIGHT
    config.chunk_size_bytes = chunk_size_bytes
    return config


class _CacheStream(NamedTuple):
    config_id: int
    layer: int
    tensor: object
    tp_column: int
    heads_per_device: int
    local_head: int
    chunk_bytes: int


def _cache_streams(kv_caches):
    """List every (config, layer) row the table describes, in table population order."""
    streams = []
    for layer_idx in kv_caches.global_layers:
        cache = kv_caches[layer_idx]
        if not isinstance(cache, GlobalRingKVCache):
            raise TypeError(f"global layer {layer_idx} does not own GlobalRingKVCache")
        streams.extend(_CacheStream(head, layer_idx, cache.kv, head, 1, 0, GLOBAL_CHUNK_BYTES) for head in range(4))
    for layer_idx in kv_caches.sliding_layers:
        cache = kv_caches[layer_idx]
        for head in range(16):
            for config_id, tensor in ((4 + head, cache.k), (20 + head, cache.v)):
                streams.append(_CacheStream(config_id, layer_idx, tensor, head // 4, 4, head % 4, SLIDING_CHUNK_BYTES))
    for stream in streams:
        if stream.tensor.dtype != ttnn.bfloat8_b:
            raise ValueError(f"migration cache must be BFP8_B, got {stream.tensor.dtype}")
    return streams


def _mesh_config(mesh_device, kv_caches):
    if not isinstance(kv_caches, Gemma4KvCaches):
        raise TypeError(f"expected Gemma4KvCaches, got {type(kv_caches).__name__}")
    mesh_config = MeshConfig(mesh_device)
    cp = mesh_config.cp_degree
    tp = mesh_config.tp_degree
    if (cp, tp) != (8, 4) or kv_caches.cp != cp or kv_caches.tp != tp:
        raise ValueError(f"Gemma 4 migration currently requires CP8/TP4, got mesh={tuple(mesh_device.shape)}")
    return mesh_config


def _fabric_node_id(mesh_device, mesh_config, cp_row, tp_column):
    coordinates = [0, 0]
    coordinates[mesh_config.cp_axis] = cp_row
    coordinates[mesh_config.tp_axis] = tp_column
    return mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coordinates))


def build_kv_chunk_address_table(*, mesh_device, kv_caches: Gemma4KvCaches, chunk_size: int):
    """Describe global packed rows and sliding K/V rows directly from compute caches."""
    mesh_config = _mesh_config(mesh_device, kv_caches)
    cp = mesh_config.cp_degree
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
            fnid = _fabric_node_id(mesh_device, mesh_config, cp_row, tp_column)
            group_cache[key] = table.add_device_group([fnid])
            host_key = (int(fnid.mesh_id), int(fnid.chip_id))
            if host_key not in mapped_hosts:
                table.set_fabric_node_host(fnid, host_name=worker_host_name())
                mapped_hosts.add(host_key)
        return group_cache[key]

    # ~80M entries at 256K x 6 slots: one reused location (set() copies it) and the walk's
    # per-block arithmetic inlined keep this loop near the binding's per-call floor.
    table_set = table.set
    for stream in _cache_streams(kv_caches):
        base_addr = int(stream.tensor.buffer_address())
        chunk_bytes = stream.chunk_bytes
        location = ttnn.experimental.disaggregation.KvCacheLocation()
        location.size_bytes = chunk_bytes
        for cp_row, slot, position, first_shard, count in _cache_chunk_runs(
            seq_len=kv_caches.max_seq_len,
            chunk_size=chunk_size,
            cp=cp,
            num_users=kv_caches.num_users,
            heads_per_device=stream.heads_per_device,
            local_head=stream.local_head,
        ):
            location.device_group_index = device_group(cp_row, stream.tp_column)
            for shard in range(first_shard, first_shard + count):
                location.noc_addr = (shard % num_banks) << 32 | base_addr + shard // num_banks * chunk_bytes
                table_set(stream.layer, position, slot, location, stream.config_id)
                position += TILE_HEIGHT
    return table


# The table layout is owned by this module, so its source seeds the on-disk table cache.
_TABLE_CACHE_SEED = "gemma4_d_p:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def kv_chunk_table_cache_key(*, mesh_device, kv_caches: Gemma4KvCaches, chunk_size: int):
    """Every input build_kv_chunk_address_table reads, as JSON-serializable values."""
    mesh_config = _mesh_config(mesh_device, kv_caches)
    nodes = []
    for cp_row in range(mesh_config.cp_degree):
        for tp_column in range(mesh_config.tp_degree):
            fnid = _fabric_node_id(mesh_device, mesh_config, cp_row, tp_column)
            nodes.append((cp_row, tp_column, int(fnid.mesh_id), int(fnid.chip_id)))
    return dict(
        configs=CONFIG_NAMES,
        chunk_bytes=(GLOBAL_CHUNK_BYTES, SLIDING_CHUNK_BYTES),
        tile_height=TILE_HEIGHT,
        num_layers=len(kv_caches),
        max_seq_len=kv_caches.max_seq_len,
        num_users=kv_caches.num_users,
        chunk_size=chunk_size,
        num_banks=get_num_dram_banks(mesh_device),
        host=worker_host_name(),
        nodes=nodes,
        streams=[
            (s.config_id, s.layer, int(s.tensor.buffer_address()), s.tp_column, s.heads_per_device, s.local_head)
            for s in _cache_streams(kv_caches)
        ],
    )


def build_and_serialize_kv_chunk_table(*, path: str, mesh_device, kv_caches: Gemma4KvCaches, chunk_size: int) -> str:
    inputs = dict(mesh_device=mesh_device, kv_caches=kv_caches, chunk_size=chunk_size)
    hit = ttnn.experimental.disaggregation.get_or_build_kv_chunk_table(
        _TABLE_CACHE_SEED, kv_chunk_table_cache_key(**inputs), lambda: build_kv_chunk_address_table(**inputs), path
    )
    logger.info(f"[migration] KV chunk address table {'reused from cache' if hit else 'built'} -> {path}")
    return path

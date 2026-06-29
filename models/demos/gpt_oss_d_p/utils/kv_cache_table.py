# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""KV chunk address table builder for GPT-OSS disaggregated prefill (`gpt_oss_d_p`).

Targets caches allocated by ``tt/attention/kv_cache_prefill_only.py``:

  * shape per device: ``[batch, num_kv_heads // TP, max_seq_len, head_dim]``
  * ``ReplicateTensorToMesh`` upload (separate DRAM buffer per mesh coordinate)
  * non-paged ``ttnn.fill_cache`` writes
  * ``NdShardSpec`` DRAM layout (32-token shards round-robin across 8 banks), same
    pattern as DeepSeek ``init_kvpe_cache`` — see ``make_gpt_oss_prefill_kv_memory_config``
  * outer SP in ``GptOssPrefillPipeline`` — each SP row fills a token shard at
    **local** cache indices ``0 .. max_seq_len // SP - 1`` for its **global** slice
  * TP — each column holds a different global KV head (tensor head dim is local)
  * single user / ``num_slots == 1`` only (``prefill_runner`` batch size 1)

Multi-config layout (one ``KvChunkAddressTableConfig`` per K/V × global head group):

  * ``num_layers == num_transformer_layers`` (36), not ``layers × heads``
  * table ``layer`` index == transformer layer index
  * migration iterates config groups; each group covers one of K/V and one global head

``KvChunkAddressTableMulti`` mocks the future C++ constructor
``KvChunkAddressTable(std::span<KvChunkAddressTableConfig>)`` — it stores all configs
and one populated ``KvChunkAddressTable`` per group. ``unified_table`` aliases
``groups[0].table`` (config index 0 = K, global head 0) until TT-Metal ships a
true multi-config table object.

Example::

    from models.demos.gpt_oss_d_p.utils.kv_cache_table import (
        make_kv_chunk_table_configs,
        create_kv_chunk_address_table_gpt_oss_prefill,
    )

    bundle = create_kv_chunk_address_table_gpt_oss_prefill(
        mesh_device=mesh_device,
        mesh_shape=mesh_device.shape,
        sp_axis=0,
        tp_axis=1,
        kv_caches=tt_kv_cache,
        num_transformer_layers=36,
        num_kv_heads=8,
        head_dim=64,
        max_seq_len=131072,
    )
    # bundle.configs — 16 configs (8 K + 8 V)
    # bundle.groups[i].table — populated table for that group
"""

from __future__ import annotations

import socket
from dataclasses import dataclass
from enum import IntEnum
from typing import Sequence

from loguru import logger

import ttnn

CHUNK_N_TOKENS = 32
BFP8_TILE_BYTES = 1088
BH_NUM_DRAM_BANKS = 8


class KvKind(IntEnum):
    K = 0
    V = 1


def make_gpt_oss_prefill_kv_memory_config(head_dim: int) -> ttnn.MemoryConfig:
    """DRAM memory config for prefill K/V cache — matches DeepSeek ``init_kvpe_cache`` striping."""
    core_ranges = [
        ttnn.CoreRange(ttnn.CoreCoord(bank_id, 0), ttnn.CoreCoord(bank_id, 0)) for bank_id in range(BH_NUM_DRAM_BANKS)
    ]
    grid = ttnn.CoreRangeSet(core_ranges)
    kv_nd_shard_spec = ttnn.NdShardSpec(
        shard_shape=[1, 1, CHUNK_N_TOKENS, head_dim],
        grid=grid,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        shard_distribution_strategy=ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
    )
    return ttnn.MemoryConfig(
        buffer_type=ttnn.BufferType.DRAM,
        nd_shard_spec=kv_nd_shard_spec,
    )


def noc_addr_for_bank_chunk(dram_base: int, bank_id: int, bank_offset: int) -> int:
    """Pack ``(bank_id, dram_base + bank_offset)`` for ``KvCacheLocation.noc_addr``."""
    return (bank_id << 32) | ((dram_base + bank_offset) & 0xFFFFFFFF)


def advance_bank_walk(bank_id: int, bank_offset: int, chunk_size_bytes: int) -> tuple[int, int]:
    """Advance round-robin bank cursor after one 32-token chunk (DeepSeek table walk)."""
    bank_id = (bank_id + 1) % BH_NUM_DRAM_BANKS
    if bank_id == 0:
        bank_offset += chunk_size_bytes
    return bank_id, bank_offset


def compute_kv_chunk_size_bytes(head_dim: int, chunk_n_tokens: int = CHUNK_N_TOKENS) -> int:
    """Byte size of one chunk ``[1, 1, chunk_n_tokens, head_dim]`` in bfp8 TILE layout."""
    tile = ttnn.TILE_SIZE
    assert chunk_n_tokens % tile == 0, f"chunk_n_tokens ({chunk_n_tokens}) must be tile-aligned"
    width_tiles = (head_dim + tile - 1) // tile
    seq_tiles = chunk_n_tokens // tile
    return width_tiles * seq_tiles * BFP8_TILE_BYTES


def global_position_to_sp_local(position: int, max_seq_len: int, sp_factor: int) -> tuple[int, int]:
    """Map global token index → ``(sp_row, local_position)`` for outer SP prefill."""
    assert max_seq_len % sp_factor == 0, f"max_seq_len={max_seq_len} must be divisible by SP={sp_factor}"
    isl_per_row = max_seq_len // sp_factor
    return position // isl_per_row, position % isl_per_row


def global_head_to_tp_shard(global_head: int, num_kv_heads_local: int) -> tuple[int, int]:
    """Map global KV head → ``(tp_col, head_idx_local)``."""
    return global_head // num_kv_heads_local, global_head % num_kv_heads_local


def make_kv_chunk_table_config_for_group(
    *,
    num_transformer_layers: int,
    head_dim: int,
    max_seq_len: int,
    chunk_n_tokens: int = CHUNK_N_TOKENS,
) -> ttnn.experimental.disaggregation.KvChunkAddressTableConfig:
    """One config for a single (K|V, global_head) group — ``num_layers`` = transformer depth only."""
    cfg = ttnn.experimental.disaggregation.KvChunkAddressTableConfig()
    cfg.num_layers = num_transformer_layers
    cfg.max_sequence_length = max_seq_len
    cfg.num_slots = 1
    cfg.chunk_n_tokens = chunk_n_tokens
    cfg.chunk_size_bytes = compute_kv_chunk_size_bytes(head_dim, chunk_n_tokens)
    return cfg


def make_kv_chunk_table_configs(
    *,
    num_transformer_layers: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int,
    chunk_n_tokens: int = CHUNK_N_TOKENS,
) -> list[tuple[KvKind, int, ttnn.experimental.disaggregation.KvChunkAddressTableConfig]]:
    """Build one config per (K|V, global_head). Returns ``(kv_kind, global_head, config)`` tuples."""
    return [
        (
            KvKind(kv_kind),
            global_head,
            make_kv_chunk_table_config_for_group(
                num_transformer_layers=num_transformer_layers,
                head_dim=head_dim,
                max_seq_len=max_seq_len,
                chunk_n_tokens=chunk_n_tokens,
            ),
        )
        for kv_kind in (KvKind.K, KvKind.V)
        for global_head in range(num_kv_heads)
    ]


@dataclass
class KvChunkTableGroup:
    """One config group: a single K or V global head."""

    group_index: int
    kv_kind: KvKind
    global_head: int
    tp_col: int
    config: ttnn.experimental.disaggregation.KvChunkAddressTableConfig
    table: ttnn.experimental.disaggregation.KvChunkAddressTable


class KvChunkAddressTableMulti:
    """Mock multi-config table until ``KvChunkAddressTable(std::span<configs>)`` exists in TT-Metal."""

    def __init__(self, configs: Sequence[ttnn.experimental.disaggregation.KvChunkAddressTableConfig]):
        if not configs:
            raise ValueError("KvChunkAddressTableMulti requires at least one config")
        self.configs = list(configs)
        # Future: KvChunkAddressTable(std::span(self.configs))
        self._unified_table: ttnn.experimental.disaggregation.KvChunkAddressTable | None = None
        self.groups: list[KvChunkTableGroup] = []

    @property
    def unified_table(self) -> ttnn.experimental.disaggregation.KvChunkAddressTable:
        """Alias of ``groups[0].table`` (configs[0] = K, global head 0) for single-table call sites."""
        if self._unified_table is None:
            raise RuntimeError("unified_table is unavailable until group 0 is added")
        return self._unified_table

    def add_group(self, group: KvChunkTableGroup) -> None:
        self.groups.append(group)
        if group.group_index == 0:
            self._unified_table = group.table


def _mesh_coordinate(sp_idx: int, tp_idx: int, sp_axis: int, tp_axis: int) -> ttnn.MeshCoordinate:
    coords = [0, 0]
    coords[sp_axis] = sp_idx
    coords[tp_axis] = tp_idx
    return ttnn.MeshCoordinate(coords[0], coords[1])


def _device_tensor_index(sp_row: int, tp_col: int, tp_len: int) -> int:
    return sp_row * tp_len + tp_col


def _populate_group_table(
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
    layer_caches: Sequence[ttnn.Tensor],
    group_index: int,
    host_name: str,
) -> None:
    slot = 0
    singleton_group_by_coord: dict[tuple[int, int], ttnn.experimental.disaggregation.DeviceGroupIndex] = {}
    bank_state: dict[tuple[int, int, int], list[int]] = {}

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

        for position in range(0, max_seq_len, chunk_n_tokens):
            sp_row, _local_position = global_position_to_sp_local(position, max_seq_len, sp_len)
            group_idx = singleton_group(sp_row, tp_col)

            dev_key = (t_layer, sp_row, tp_col)
            if dev_key not in bank_state:
                bank_state[dev_key] = [0, 0]
            bank_id, bank_offset = bank_state[dev_key]

            dt = device_tensors[_device_tensor_index(sp_row, tp_col, tp_len)]
            dram_base = dt.buffer_address()

            location = ttnn.experimental.disaggregation.KvCacheLocation()
            location.noc_addr = noc_addr_for_bank_chunk(dram_base, bank_id, bank_offset)
            location.size_bytes = chunk_size_bytes
            location.device_group_index = group_idx
            lookup_table.set(t_layer, position, slot, location)

            bank_state[dev_key] = list(advance_bank_walk(bank_id, bank_offset, chunk_size_bytes))


def create_kv_chunk_address_table_gpt_oss_prefill(
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
    chunk_n_tokens: int = CHUNK_N_TOKENS,
) -> KvChunkAddressTableMulti:
    """Populate multi-config K/V chunk address tables for prefill KV caches.

    Creates ``2 × num_kv_heads`` config groups (K and V for each global head). Each
    group's ``config.num_layers == num_transformer_layers``; table ``layer`` equals
    the transformer layer index.

    Args:
        kv_caches: ``kv_caches[layer] == [k_cache, v_cache]`` device tensors.
        num_transformer_layers: HF ``num_hidden_layers``.
        num_kv_heads: Global GQA KV head count.
        head_dim: Per-head dimension (64 for GPT-OSS 120B).
        max_seq_len: Cache sequence length (matches ``kv_cache_prefill_only``).
    """
    tp_len = mesh_shape[tp_axis]
    sp_len = mesh_shape[sp_axis]
    num_kv_heads_local = num_kv_heads // tp_len
    if num_kv_heads % tp_len != 0:
        raise ValueError(f"num_kv_heads ({num_kv_heads}) must be divisible by TP ({tp_len})")
    assert (
        len(kv_caches) == num_transformer_layers
    ), f"kv_caches has {len(kv_caches)} layers but num_transformer_layers={num_transformer_layers}"

    config_groups = make_kv_chunk_table_configs(
        num_transformer_layers=num_transformer_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_seq_len=max_seq_len,
        chunk_n_tokens=chunk_n_tokens,
    )
    all_configs = [cfg for _, _, cfg in config_groups]
    bundle = KvChunkAddressTableMulti(all_configs)
    host_name = socket.gethostname()

    for group_index, (kv_kind, global_head, group_cfg) in enumerate(config_groups):
        tp_col, _head_idx_local = global_head_to_tp_shard(global_head, num_kv_heads_local)
        assert tp_col < tp_len, f"global_head {global_head} maps to tp_col {tp_col} >= tp_len {tp_len}"
        assert group_cfg.num_layers == num_transformer_layers
        assert group_cfg.num_slots == 1
        assert group_cfg.max_sequence_length == max_seq_len
        assert group_cfg.chunk_size_bytes == compute_kv_chunk_size_bytes(head_dim, chunk_n_tokens)

        group_table = ttnn.experimental.disaggregation.KvChunkAddressTable(group_cfg)
        # Per transformer layer: K or V device tensor for this group (kv_kind selects index 0/1).
        # layer_caches[t] is the multi-device cache for layer t; only global_head/tp_col is filled below.
        layer_caches = [kv_caches[t][int(kv_kind)] for t in range(num_transformer_layers)]

        _populate_group_table(
            lookup_table=group_table,
            mesh_device=mesh_device,
            sp_axis=sp_axis,
            tp_axis=tp_axis,
            sp_len=sp_len,
            tp_len=tp_len,
            num_transformer_layers=num_transformer_layers,
            max_seq_len=max_seq_len,
            chunk_n_tokens=chunk_n_tokens,
            chunk_size_bytes=group_cfg.chunk_size_bytes,
            tp_col=tp_col,
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

    logger.info(
        f"[gpt-oss-d-p-kv-table] multi-config: transformer_layers={num_transformer_layers} "
        f"num_groups={len(bundle.groups)} (2×{num_kv_heads} K/V×head) "
        f"layers_per_config={num_transformer_layers} sp={sp_len} tp={tp_len} "
        f"seq={max_seq_len} chunk_bytes={bundle.configs[0].chunk_size_bytes} "
        f"total_entries={sum(g.table.total_entries() for g in bundle.groups)}"
    )
    return bundle

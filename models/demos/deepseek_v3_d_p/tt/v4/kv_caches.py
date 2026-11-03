# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""The migratable part of a V4 rank's attention state, as one tensor per kind, and its KV chunk table.

Every HCA layer's compressed cache, every CSA layer's joint table and every CSA layer's index key cache
live in one tensor per kind, user-major like the MLA KVPE cache: user ``u``'s copy of the ``i``-th layer of
that kind on this rank is row ``u * layers + i``. One tensor per kind is what lets a migration table name it
by a single base address. The carries (sliding windows, HCA tail, compressor overlap) stay per slot in each
attention's state; nothing migrates them.

The model allocates these, not the engine: the runner builds the runtime before it asks the adapter for
caches, so engine-owned tensors could only reach the layers after they had built their own. The runtime's
migration hooks read them off the model instead.

Table configs, in order, all on the model's layer axis:

  ``"00"`` HCA compressed rows, a position per entry, 32-row pages, every chip a replica;
  ``"01"`` CSA compressed rows, a position per entry, one-row pages (the joint table is DRAM-interleaved);
  ``"02"`` CSA index keys, a position per entry, block-cyclic over SP with a period of one chunk's entries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import ttnn
from models.demos.common.prefill.runners.migration import KvCacheStage, serialize_prebuilt_kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.mla.compressed_sparse_attention import csa_capacity, csa_index_seq_len
from models.demos.deepseek_v3_d_p.tt.mla.heavily_compressed_attention import hca_cache_rows
from models.demos.deepseek_v3_d_p.tt.mla.v4_attention_base import alloc_compressed_cache
from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import _BFP8_TILE_BYTES, _TILE_DIM, config_name
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK,
    init_kvpe_cache,
    populate_kv_chunk_address_table_block_cyclic,
    populate_kv_chunk_address_table_replicated,
)

HCA = "heavily_compressed_attention"
CSA = "compressed_sparse_attention"
CACHE_KINDS = ("hca_compressed", "csa_compressed", "csa_index")  # table config order
_BF16_BYTES = 2


@dataclass(frozen=True)
class LayerCache:
    """One layer's place in its kind's tensors: ``table`` is the HCA compressed cache or the CSA joint
    table, ``index`` the CSA index key cache, and the layer writes row ``u * layer_num + layer_pos``.
    The default is a standalone layer's private layout."""

    table: Optional[ttnn.Tensor] = None
    index: Optional[ttnn.Tensor] = None
    layer_num: int = 1
    layer_pos: int = 0


@dataclass(frozen=True)
class _Rows:
    """Per-user rows of each tensor, as each attention's ``build_tables`` sizes its own."""

    hca: int  # HCA compressed cache, all published
    csa_entries: int  # CSA compressed rows, published
    csa: int  # CSA joint table: the compressed rows, then the window, then one chunk
    index: int  # CSA index keys, globally (each chip holds index / sp)
    period: int  # one chunk's CSA entries, the index cache's block-cyclic period

    @classmethod
    def of(cls, config, max_seq_len, chunk, sp):
        rate = config.compress_rates[CSA]
        entries = csa_capacity(max_seq_len, chunk, rate)
        return cls(
            hca=hca_cache_rows(max_seq_len, chunk, config.compress_rates[HCA]),
            csa_entries=entries,
            csa=entries + config.sliding_window + chunk,
            index=csa_index_seq_len(max_seq_len, chunk, rate, sp) // rate,
            period=chunk // rate,
        )


def _layers_of(config, kind, layer_ids):
    return tuple(i for i in layer_ids if config.layer_types[i] == kind)


@dataclass(frozen=True)
class V4KvCaches:
    first_layer: int
    hca_layers: tuple[int, ...]  # global layer ids, in row order
    csa_layers: tuple[int, ...]
    hca: Optional[ttnn.Tensor]  # [slots * len(hca_layers), 1, rows, head_dim], ND-sharded, replicated
    csa: Optional[ttnn.Tensor]  # [slots * len(csa_layers), 1, rows, head_dim], ROW_MAJOR, replicated
    index: Optional[ttnn.Tensor]  # [slots * len(csa_layers), 1, rows / sp, index_head_dim], block-cyclic

    def for_layer(self, layer_idx: int) -> LayerCache:
        if layer_idx in self.hca_layers:
            return LayerCache(self.hca, None, len(self.hca_layers), self.hca_layers.index(layer_idx))
        if layer_idx in self.csa_layers:
            return LayerCache(self.csa, self.index, len(self.csa_layers), self.csa_layers.index(layer_idx))
        return LayerCache()

    def migration_stages(self, config) -> list[KvCacheStage]:
        """One stage per table config, always three, numbered in each kind's compacted space: the
        per-stage all-gather is collective, so a rank without a kind still reports it, at its offset."""
        stages = []
        for kind, layers, tensor in (
            (HCA, self.hca_layers, self.hca),
            (CSA, self.csa_layers, self.csa),
            (CSA, self.csa_layers, self.index),
        ):
            offset = len(_layers_of(config, kind, range(self.first_layer)))
            stages.append(KvCacheStage(int(tensor.buffer_address()) if layers else 0, offset, len(layers)))
        return stages


def allocate_v4_kv_caches(mesh_device, config, layer_ids, *, slot_num, max_seq_len, chunk, sp_axis) -> V4KvCaches:
    hca_layers, csa_layers = _layers_of(config, HCA, layer_ids), _layers_of(config, CSA, layer_ids)
    rows = _Rows.of(config, max_seq_len, chunk, mesh_device.shape[sp_axis])
    shared = dict(head_dim=config.head_dim, sp_axis=sp_axis, slot_num=slot_num)
    hca = csa = index = None
    if hca_layers:
        hca = alloc_compressed_cache(mesh_device, rows=rows.hca, layer_num=len(hca_layers), **shared)
    if csa_layers:
        csa = alloc_compressed_cache(
            mesh_device,
            rows=rows.csa,
            layer_num=len(csa_layers),
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **shared,
        )
        # TtCSA's private fallback (_build_indexer) allocates the same.
        index = init_kvpe_cache(
            kvpe_cache_head_dim=config.index_head_dim,
            mesh_device=mesh_device,
            seq_len=rows.index,
            mesh_shape=list(mesh_device.shape),
            sp_axis=sp_axis,
            num_kvpe_cache_layers=len(csa_layers),
            num_users=slot_num,
        )
    return V4KvCaches(min(layer_ids), hca_layers, csa_layers, hca, csa, index)


def build_v4_kv_chunk_table(config, *, stage_layouts, mesh_shape, sp_axis, num_users, max_seq_len, chunk, path):
    """The three configs from the gathered ``stage_layouts`` (``V4KvCaches.migration_stages`` order),
    serialized to ``path``. Needs no tensor: everything comes from the layouts and the geometry."""
    disagg = ttnn.experimental.disaggregation
    rows = _Rows.of(config, max_seq_len, chunk, mesh_shape[sp_axis])
    every = range(config.num_hidden_layers)
    hca_layers, csa_layers = _layers_of(config, HCA, every), _layers_of(config, CSA, every)

    def table_config(rows_a_page, published, page_bytes):
        cfg = disagg.KvChunkAddressTableConfig()
        cfg.num_layers = config.num_hidden_layers
        cfg.max_sequence_length = published
        cfg.num_slots = num_users
        cfg.chunk_n_tokens = rows_a_page
        cfg.chunk_size_bytes = page_bytes
        return cfg

    page = NUM_CONTIGUOUS_TOKENS_IN_DRAM_BANK
    hca, csa, index = configs = (
        table_config(page, rows.hca, page * config.head_dim * _BF16_BYTES),
        table_config(1, rows.csa_entries, config.head_dim * _BF16_BYTES),
        table_config(page, rows.index, (config.index_head_dim // _TILE_DIM) * _BFP8_TILE_BYTES),
    )
    names = [config_name(i, len(configs)) for i in range(len(configs))]
    table = disagg.KvChunkAddressTable(dict(zip(names, configs)))
    hca_id, csa_id, index_id = (table.config_id_of(name) for name in names)

    populate_kv_chunk_address_table_replicated(
        table,
        hca,
        stage_layout=stage_layouts[0],
        rows_per_layer=rows.hca,
        num_users=num_users,
        config_id=hca_id,
        layer_rows=hca_layers,
    )
    populate_kv_chunk_address_table_replicated(
        table,
        csa,
        stage_layout=stage_layouts[1],
        rows_per_layer=rows.csa,
        num_users=num_users,
        config_id=csa_id,
        layer_rows=csa_layers,
    )
    populate_kv_chunk_address_table_block_cyclic(
        lookup_table=table,
        config=index,
        mesh_device=None,
        mesh_shape=mesh_shape,
        seq_len=rows.index,
        sp_axis=sp_axis,
        tt_kvpe_cache=None,
        chunk_size_bytes=index.chunk_size_bytes,
        num_users=num_users,
        config_id=index_id,
        stage_layout=stage_layouts[2],
        layer_rows=csa_layers,
        period=rows.period,
    )
    return serialize_prebuilt_kv_chunk_table(table=table, path=path)

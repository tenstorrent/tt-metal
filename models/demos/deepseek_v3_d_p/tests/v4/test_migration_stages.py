# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only: V4 exposes HCA compressed rows, CSA compressed rows and CSA index keys as migration stages
0, 1, 2, and the table built from their gathered layouts addresses every chunk where the caches put it.

Every rank must return exactly three stages (the per-stage all-gather is collective), numbered in each
kind's compacted space, and each config is published on the model's layer axis.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v4 import DeepSeekV4FlashAdapter
from models.demos.deepseek_v3_d_p.tt.v4.kv_caches import CSA, HCA, V4KvCaches, _Rows, build_v4_kv_chunk_table

FLASH, PRO = v4_hf_config(DeepSeekV4FlashConfig), v4_hf_config(DeepSeekV4ProConfig)
# (first, count) per rank: one rank holding the model, and Pro split as test_transformer_depth runs it.
SPLITS = {"flash": (FLASH, [(0, 43)]), "pro": (PRO, [(0, 61)]), "pro-3rank": (PRO, [(0, 10), (10, 26), (36, 25)])}


def _caches(config, first, count, base=0x1000):
    ids = range(first, first + count)
    hca = tuple(i for i in ids if config.layer_types[i] == HCA)
    csa = tuple(i for i in ids if config.layer_types[i] == CSA)
    tensor = lambda offset: SimpleNamespace(buffer_address=lambda: base + offset)
    return V4KvCaches(first, hca, csa, tensor(0) if hca else None, tensor(0x100) if csa else None, tensor(0x200))


@pytest.mark.parametrize("name", SPLITS)
def test_every_rank_emits_three_stages_that_tile_each_kind(name):
    config, splits = SPLITS[name]
    stages = [_caches(config, first, count).migration_stages(config) for first, count in splits]
    assert all(len(rank) == 3 for rank in stages)
    for config_idx, kind in enumerate((HCA, CSA, CSA)):
        total = sum(t == kind for t in config.layer_types[: config.num_hidden_layers])
        covered = [(rank[config_idx].first_layer, rank[config_idx].count) for rank in stages]
        assert covered[0][0] == 0 and all(f + c == nf for (f, c), (nf, _) in zip(covered, covered[1:]))
        assert covered[-1][0] + covered[-1][1] == total


def test_rank_without_a_kind_reports_an_empty_stage():
    # Flash layers 0 and 1 are sliding attention: no HCA or CSA layer, and nothing allocated.
    stages = _caches(FLASH, 0, 2).migration_stages(FLASH)
    assert [(s.base_addr, s.first_layer, s.count) for s in stages] == [(0, 0, 0)] * 3


def test_adapter_names_the_three_kinds():
    adapter = DeepSeekV4FlashAdapter()
    assert [adapter.cache_kind(i) for i in range(4)] == ["hca_compressed", "csa_compressed", "csa_index", "other"]


def _layout(stage, fnids, num_banks=8):
    return [
        dict(
            rank=0,
            base_addr=stage.base_addr,
            num_banks=num_banks,
            first_layer=stage.first_layer,
            count=stage.count,
            host_tag=0,
            fnids=fnids,
        )
    ]


def test_table_addresses_each_kind_where_its_cache_puts_it(tmp_path):
    """Flash's first five layers (SWA, SWA, CSA, HCA, CSA) at two users on a 2x2 mesh, round-tripped."""
    config, mesh_shape, users, max_seq_len, chunk, banks = FLASH, (2, 2), 2, 10240, 5120, 8
    fnids = [[ttnn.FabricNodeId(ttnn.MeshId(0), r * 2 + c) for c in range(2)] for r in range(2)]
    caches = _caches(config, 0, 5)
    layouts = [_layout(stage, fnids, banks) for stage in caches.migration_stages(config)]
    path = build_v4_kv_chunk_table(
        config,
        stage_layouts=layouts,
        mesh_shape=mesh_shape,
        sp_axis=0,
        num_users=users,
        max_seq_len=max_seq_len,
        chunk=chunk,
        path=str(tmp_path / "v4.pb"),
    )
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(path)
    assert [table.config_id_of(n) for n in ("00", "01", "02")] == [0, 1, 2]
    rows = _Rows.of(config, max_seq_len, chunk, mesh_shape[0])

    def where(layer, position, slot, config_id):
        loc = table.lookup(layer, position, slot, config_id)
        chips = sorted(int(f.chip_id) for f in table.get_device_group(loc.device_group_index).fabric_node_ids)
        return loc.noc_addr >> 32, loc.noc_addr & 0xFFFFFFFF, chips

    def page(base, page_index, page_bytes):
        return page_index % banks, base + (page_index // banks) * page_bytes

    # HCA: layer 3 is the only one, so user 1 is row 1; 32-row pages of bf16 head_dim, every chip a replica.
    hca_bytes = 32 * config.head_dim * 2
    assert where(3, 64, 1, 0) == (*page(0x1000, 1 * (rows.hca // 32) + 2, hca_bytes), [0, 1, 2, 3])
    # CSA: layers 2 and 4; user 1's layer 4 is row 3, one row a page, the table's full stride per row.
    csa_bytes = config.head_dim * 2
    assert where(4, 7, 1, 1) == (*page(0x1100, 3 * rows.csa + 7, csa_bytes), [0, 1, 2, 3])
    # Index: block-cyclic over the two SP rows with a period of one chunk's entries; TP columns replicate.
    # Entry period + 640 is the second chunk's first entry on SP row 1, so it is that row's 21st page.
    index_bytes = (config.index_head_dim // 32) * 1088
    bank, offset, chips = where(2, rows.period + rows.period // 2, 0, 2)
    assert chips == [2, 3] and (bank, offset) == page(0x1200, rows.period // 2 // 32, index_bytes)

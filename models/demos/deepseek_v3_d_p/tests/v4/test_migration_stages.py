# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-only: V4 exposes HCA compressed rows, CSA compressed rows and CSA index keys as migration stages
0, 1, 2, and the table built from their gathered layouts addresses every chunk where the caches put it.

Every rank must return exactly three stages (the per-stage all-gather is collective), numbered in each
kind's compacted space, and each config is published on the model's layer axis.
"""

import pytest

from models.demos.common.prefill.runners.runner_utils import compute_layer_split
from models.demos.deepseek_v3_d_p.reference.deepseek_v4.hf_config import v4_hf_config
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import DeepSeekV4FlashConfig
from models.demos.deepseek_v3_d_p.reference.deepseek_v4_pro_config import DeepSeekV4ProConfig
from models.demos.deepseek_v3_d_p.tests.v4.helpers import (
    KIND_OFFSETS,
    build_fake_table,
    fake_fnids,
    fake_v4_caches,
    stage_layout,
)
from models.demos.deepseek_v3_d_p.tt.runners.adapters.deepseek_v4 import DeepSeekV4FlashAdapter, DeepSeekV4ProAdapter
from models.demos.deepseek_v3_d_p.tt.v4.kv_caches import CSA, HCA, _Rows

FLASH, PRO = v4_hf_config(DeepSeekV4FlashConfig), v4_hf_config(DeepSeekV4ProConfig)
# (first, count) per rank: one rank holding the model, and Pro split as test_transformer_depth runs it.
SPLITS = {"flash": (FLASH, [(0, 43)]), "pro": (PRO, [(0, 61)]), "pro-3rank": (PRO, [(0, 10), (10, 26), (36, 25)])}


@pytest.mark.parametrize("name", SPLITS)
def test_every_rank_emits_three_stages_that_tile_each_kind(name):
    config, splits = SPLITS[name]
    stages = [fake_v4_caches(config, first, count).migration_stages(config) for first, count in splits]
    assert all(len(rank) == 3 for rank in stages)
    for config_idx, kind in enumerate((HCA, CSA, CSA)):
        total = sum(t == kind for t in config.layer_types[: config.num_hidden_layers])
        covered = [(rank[config_idx].first_layer, rank[config_idx].count) for rank in stages]
        assert covered[0][0] == 0 and all(f + c == nf for (f, c), (nf, _) in zip(covered, covered[1:]))
        assert covered[-1][0] + covered[-1][1] == total


def test_rank_without_a_kind_reports_an_empty_stage():
    # Flash layers 0 and 1 are sliding attention: no HCA or CSA layer, and nothing allocated.
    stages = fake_v4_caches(FLASH, 0, 2).migration_stages(FLASH)
    assert [(s.base_addr, s.first_layer, s.count) for s in stages] == [(0, 0, 0)] * 3


def test_adapter_names_the_three_kinds():
    adapter = DeepSeekV4FlashAdapter()
    assert [adapter.cache_kind(i) for i in range(4)] == ["hca_compressed", "csa_compressed", "csa_index", "other"]


def test_table_addresses_each_kind_where_its_cache_puts_it(tmp_path):
    """Flash's first five layers (SWA, SWA, CSA, HCA, CSA) at two users on a 2x2 mesh, round-tripped."""
    config, mesh_shape, users, max_seq_len, chunk, banks = FLASH, (2, 2), 2, 10240, 5120, 8
    caches = fake_v4_caches(config, 0, 5)
    table = build_fake_table(
        tmp_path,
        config,
        [[stage_layout(s, fake_fnids(0), banks)] for s in caches.migration_stages(config)],
        mesh_shape,
        users,
    )
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
    assert where(3, 64, 1, 0) == (*page(0x1000 + KIND_OFFSETS[0], 1 * (rows.hca // 32) + 2, hca_bytes), [0, 1, 2, 3])
    # CSA: layers 2 and 4; user 1's layer 4 is row 3, one row a page, the table's full stride per row.
    csa_bytes = config.head_dim * 2
    assert where(4, 7, 1, 1) == (*page(0x1000 + KIND_OFFSETS[1], 3 * rows.csa + 7, csa_bytes), [0, 1, 2, 3])
    # Index: block-cyclic over the two SP rows with a period of one chunk's entries; TP columns replicate.
    # Entry period + 640 is the second chunk's first entry on SP row 1, so it is that row's 21st page.
    index_bytes = (config.index_head_dim // 32) * 1088
    bank, offset, chips = where(2, rows.period + rows.period // 2, 0, 2)
    assert chips == [2, 3] and (bank, offset) == page(0x1000 + KIND_OFFSETS[2], rows.period // 2 // 32, index_bytes)


def test_four_stage_table_points_each_layer_at_its_own_ranks_cache(tmp_path):
    """Pro split as the runner splits it on sc4: every rank's first layer of each kind resolves to that rank's
    chips, at its own base, at that layer's row in the rank's compacted space."""
    config, mesh_shape, users, banks = PRO, (2, 2), 2, 8
    split = compute_layer_split(config.num_hidden_layers, 4, DeepSeekV4ProAdapter().layer_split_boundaries(61))
    ranks = [fake_v4_caches(config, first, count, base=0x10000 * (r + 1)) for r, (first, count) in enumerate(split)]
    stages = [caches.migration_stages(config) for caches in ranks]
    layouts = [
        [stage_layout(rank_stages[i], fake_fnids(r), banks, r) for r, rank_stages in enumerate(stages)]
        for i in range(3)
    ]
    table = build_fake_table(tmp_path, config, layouts, mesh_shape, users)
    rows = _Rows.of(config, 10240, 5120, mesh_shape[0])
    pages = (rows.hca // 32, rows.csa, rows.index // mesh_shape[0] // 32)  # per (user, layer) row, per chip
    for r, caches in enumerate(ranks):
        for config_id, (layers, offset) in enumerate(
            zip((caches.hca_layers, caches.csa_layers, caches.csa_layers), KIND_OFFSETS)
        ):
            layer = layers[0]
            loc = table.lookup(layer, 0, 1, config_id)
            members = table.get_device_group(loc.device_group_index).fabric_node_ids
            assert {int(f.mesh_id) for f in members} == {r}, f"rank {r} config {config_id} resolves off-rank"
            page = (1 * len(layers)) * pages[config_id]  # user 1, the rank's first layer of the kind
            want = ((page % banks) << 32) | (
                0x10000 * (r + 1) + offset + (page // banks) * table.config(config_id).chunk_size_bytes
            )
            assert loc.noc_addr == want, f"rank {r} config {config_id} layer {layer}"

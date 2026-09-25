# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the KDA address-table walk against a recording fake table and a fake gathered layout.

Every (row, segment, slot) the walk emits must resolve to bank ``s % num_banks`` and offset
``base + (s // num_banks) * segment_bytes`` of the owning column's slab, with one replica group per TP
column spanning every SP row, and rows published on the model's layer axis.
"""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    KDA_VERSIONS,
    kda_chunk_n_tokens,
    kda_max_sequence_length,
    kda_position,
    kda_window,
    populate_kv_chunk_address_table_kda,
)

SP, TP = 8, 4


class _RecordingTable:
    def __init__(self):
        self.groups = []
        self.hosts = {}
        self.entries = {}

    def add_device_group(self, members):
        key = tuple(sorted(members))
        if key not in self.groups:
            self.groups.append(key)
        return self.groups.index(key)

    def set_fabric_node_host(self, fid, host_name):
        self.hosts[fid] = host_name

    def set(self, layer, position, slot, location, config_id):
        key = (config_id, layer, position, slot)
        assert key not in self.entries, f"duplicate entry {key}"
        self.entries[key] = (location.noc_addr, location.size_bytes, location.device_group_index)


def _fake_layout(count_per_rank, num_banks_per_rank):
    """Two ranks, distinct bases and bank counts, KDA slot space 0..sum(counts)."""
    stages, first = [], 0
    for rank, (count, banks) in enumerate(zip(count_per_rank, num_banks_per_rank)):
        stages.append(
            {
                "rank": rank,
                "first_layer": first,
                "count": count,
                "base_addr": 0x1000_0000 * (rank + 1),
                "num_banks": banks,
                "host_tag": 0xABC0 + rank,
                "fnids": [[(rank, row, col) for col in range(TP)] for row in range(SP)],
            }
        )
        first += count
    return stages


@pytest.fixture(autouse=True)
def _fake_location(monkeypatch):
    import ttnn

    monkeypatch.setattr(
        ttnn.experimental.disaggregation,
        "KvCacheLocation",
        lambda: SimpleNamespace(noc_addr=None, size_bytes=None, device_group_index=None),
        raising=False,
    )


def _config(geometry, kind, num_layers, num_users):
    bytes_ = geometry.recurrent_segment_bytes if kind == "kda_recurrent" else geometry.convolution_segment_bytes
    return SimpleNamespace(
        num_layers=num_layers,
        max_sequence_length=kda_max_sequence_length(geometry),
        num_slots=num_users,
        chunk_n_tokens=kda_chunk_n_tokens(geometry, kind),
        chunk_size_bytes=bytes_,
    )


def test_contract_axis_constants():
    """The numbers k3_disagg_contract.md fixes at 96 heads: one 36864-position window, strides 96 / 64."""
    geometry = KdaContractGeometry.from_kda_config(kimi_k3_kda_config(), mesh_shape=(SP, TP), sp_axis=0, tp_axis=1)
    assert kda_window(geometry) == 36_864
    assert kda_chunk_n_tokens(geometry, "kda_recurrent") == 96
    assert kda_chunk_n_tokens(geometry, "kda_convolution") == 64
    assert kda_max_sequence_length(geometry) == 8 * 36_864
    # Segment i of version v; the last segment of each state ends exactly at its window's end.
    assert kda_position(geometry, "kda_recurrent", 383, 7) + 96 == 8 * 36_864
    assert kda_position(geometry, "kda_convolution", 575, 0) + 64 == 36_864


@pytest.mark.parametrize("kind", ["kda_recurrent", "kda_convolution"])
@pytest.mark.parametrize("num_users", [1, 2], ids=["1user", "2users"])
def test_walk_addresses_every_segment_of_every_stage(kind, num_users):
    geometry = KdaContractGeometry.from_kda_config(kimi_k3_kda_config(), mesh_shape=(SP, TP), sp_axis=0, tp_axis=1)
    counts, banks = (18, 15), (8, 7)
    layout = _fake_layout(counts, banks)
    layer_rows = KimiK3Config.kda_layer_ids()[: sum(counts)]
    table = _RecordingTable()
    populate_kv_chunk_address_table_kda(
        table,
        _config(geometry, kind, KimiK3Config.NUM_LAYERS, num_users),
        (SP, TP),
        0,
        1,
        geometry,
        kind,
        num_users=num_users,
        config_id=7,
        stage_layout=layout,
        layer_rows=layer_rows,
    )

    segments = (
        geometry.recurrent_segments_per_layer if kind == "kda_recurrent" else geometry.convolution_segments_per_layer
    )
    seg_bytes = geometry.recurrent_segment_bytes if kind == "kda_recurrent" else geometry.convolution_segment_bytes
    shards_per_layer = (
        geometry.recurrent_shards_per_layer if kind == "kda_recurrent" else geometry.convolution_shards_per_layer
    )

    assert len(table.entries) == segments * sum(counts) * num_users * KDA_VERSIONS
    assert all(key[0] == 7 for key in table.entries)
    # One replica group per (rank, TP column), each spanning all SP rows.
    assert len(table.groups) == len(layout) * TP
    for group in table.groups:
        ranks = {m[0] for m in group}
        cols = {m[2] for m in group}
        assert len(ranks) == 1 and len(cols) == 1 and {m[1] for m in group} == set(range(SP))
    assert set(table.hosts.values()) == {"host-0000abc0", "host-0000abc1"}

    for stage in layout:
        for local_layer in range(stage["count"]):
            row = layer_rows[stage["first_layer"] + local_layer]
            assert row in KimiK3Config.kda_layer_ids()
            for slot in range(num_users):
                batch = slot * stage["count"] + local_layer
                for segment in range(segments):
                    # Every version window aliases the same location.
                    aliases = {
                        table.entries[(7, row, kda_position(geometry, kind, segment, v), slot)]
                        for v in range(KDA_VERSIONS)
                    }
                    assert len(aliases) == 1, f"segment {segment}: version windows disagree"
                    noc_addr, size, group_idx = aliases.pop()
                    assert size == seg_bytes
                    if kind == "kda_recurrent":
                        tp_col, h_local, band = geometry.decompose_recurrent(segment)
                        shard = batch * shards_per_layer + geometry.recurrent_local_shard(0, h_local, band)
                    else:
                        branch, tp_col, h_local, half = geometry.decompose_convolution(segment)
                        shard = batch * shards_per_layer + geometry.convolution_local_column(branch, h_local, half)
                    bank, addr = noc_addr >> 32, noc_addr & 0xFFFFFFFF
                    assert bank == shard % stage["num_banks"]
                    assert addr == stage["base_addr"] + (shard // stage["num_banks"]) * seg_bytes
                    assert {m[2] for m in table.groups[group_idx]} == {tp_col}
                    assert {m[0] for m in table.groups[group_idx]} == {stage["rank"]}


def test_walk_skips_null_stages_and_rejects_wrong_config(expect_error):
    geometry = KdaContractGeometry.from_kda_config(kimi_k3_kda_config(), mesh_shape=(SP, TP), sp_axis=0, tp_axis=1)
    layout = _fake_layout((0, 3), (8, 8))
    layer_rows = KimiK3Config.kda_layer_ids()[:3]
    table = _RecordingTable()
    populate_kv_chunk_address_table_kda(
        table,
        _config(geometry, "kda_recurrent", 93, 1),
        (SP, TP),
        0,
        1,
        geometry,
        "kda_recurrent",
        stage_layout=layout,
        layer_rows=layer_rows,
    )
    assert len(table.groups) == TP  # the count == 0 stage registers nothing
    assert len(table.entries) == 3 * geometry.recurrent_segments_per_layer * KDA_VERSIONS

    bad = _config(geometry, "kda_recurrent", 93, 1)
    bad.chunk_n_tokens = 32
    with expect_error(AssertionError, "contract stride"):
        populate_kv_chunk_address_table_kda(
            _RecordingTable(),
            bad,
            (SP, TP),
            0,
            1,
            geometry,
            "kda_recurrent",
            stage_layout=layout,
            layer_rows=layer_rows,
        )
    with expect_error(ValueError, "gathered stage layout"):
        populate_kv_chunk_address_table_kda(
            _RecordingTable(), _config(geometry, "kda_recurrent", 93, 1), (SP, TP), 0, 1, geometry, "kda_recurrent"
        )

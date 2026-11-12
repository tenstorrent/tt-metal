# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: the KDA address-table walk against a recording fake table and a fake gathered layout.

Every (row, segment, slot) the walk emits must resolve to bank ``s % num_banks`` and offset
``base + (s // num_banks) * segment_bytes`` of the owning store's slab, and rows are published on the
model's layer axis. Replicated storage publishes one replica group per TP column spanning every SP row;
SP-sharded storage one singleton group per (SP row, TP column), with keys, bytes and positions unchanged.
"""

from types import SimpleNamespace

import pytest

from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config, kimi_k3_kda_config
from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import (
    kda_chunk_n_tokens,
    kda_position,
    kda_segment_bytes,
    kda_segments_per_layer,
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


def _fake_layout(count_per_rank, num_banks_per_rank, sp=SP, tp=TP):
    """One stage per rank, distinct bases and bank counts, KDA slot space 0..sum(counts)."""
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
                "fnids": [[(rank, row, col) for col in range(tp)] for row in range(sp)],
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
    import ttnn

    return SimpleNamespace(
        num_layers=num_layers,
        max_sequence_length=kda_window(geometry),
        num_slots=num_users,
        chunk_n_tokens=kda_chunk_n_tokens(geometry, kind),
        chunk_size_bytes=kda_segment_bytes(geometry, kind),
        tensor_type=ttnn.experimental.disaggregation.KvChunkTensorType.Snapshot,
    )


def test_contract_axis_constants():
    """The numbers decode shares at 96 heads (tt-blaze cache_migration.py): one window of 36864, strides 96 / 64."""
    geometry = KdaContractGeometry.from_kda_config(kimi_k3_kda_config(), mesh_shape=(SP, TP), sp_axis=0, tp_axis=1)
    assert kda_window(geometry) == 36_864
    assert kda_chunk_n_tokens(geometry, "kda_recurrent") == 96
    assert kda_chunk_n_tokens(geometry, "kda_convolution") == 64
    # The last segment of each state ends exactly at the window's end.
    assert kda_position(geometry, "kda_recurrent", 383) + 96 == 36_864
    assert kda_position(geometry, "kda_convolution", 575) + 64 == 36_864


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

    segments = kda_segments_per_layer(geometry, kind)
    seg_bytes = kda_segment_bytes(geometry, kind)
    shards_per_layer = (
        geometry.recurrent_shards_per_layer if kind == "kda_recurrent" else geometry.convolution_shards_per_layer
    )

    assert len(table.entries) == segments * sum(counts) * num_users
    assert all(key[0] == 7 for key in table.entries)
    # One singleton group per (rank, SP row, TP column): each chip stores its own heads.
    assert sorted(table.groups) == sorted(
        ((r, row, col),) for r in range(len(layout)) for row in range(SP) for col in range(TP)
    )
    assert set(table.hosts.values()) == {"host-0000abc0", "host-0000abc1"}

    for stage in layout:
        for local_layer in range(stage["count"]):
            row = layer_rows[stage["first_layer"] + local_layer]
            assert row in KimiK3Config.kda_layer_ids()
            for slot in range(num_users):
                batch = slot * stage["count"] + local_layer
                for segment in range(segments):
                    noc_addr, size, group_idx = table.entries[(7, row, kda_position(geometry, kind, segment), slot)]
                    assert size == seg_bytes
                    if kind == "kda_recurrent":
                        tp_col, h_local, band = geometry.decompose_recurrent(segment)
                        sp_row, j = geometry.stored_head(h_local)
                        shard = batch * shards_per_layer + geometry.recurrent_local_shard(0, j, band)
                    else:
                        branch, tp_col, h_local, half = geometry.decompose_convolution(segment)
                        sp_row, j = geometry.stored_head(h_local)
                        shard = batch * shards_per_layer + geometry.convolution_stored_page(branch, j, half)
                    bank, addr = noc_addr >> 32, noc_addr & 0xFFFFFFFF
                    assert bank == shard % stage["num_banks"]
                    assert addr == stage["base_addr"] + (shard // stage["num_banks"]) * seg_bytes
                    assert table.groups[group_idx] == ((stage["rank"], sp_row, tp_col),)


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
    assert len(table.groups) == SP * TP  # one chip each; the count == 0 stage registers nothing
    assert len(table.entries) == 3 * geometry.recurrent_segments_per_layer

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
    eight_windows = _config(geometry, "kda_recurrent", 93, 1)
    eight_windows.max_sequence_length = 8 * kda_window(geometry)  # the former 8-version extent
    with expect_error(AssertionError, "not one window of"):
        populate_kv_chunk_address_table_kda(
            _RecordingTable(),
            eight_windows,
            (SP, TP),
            0,
            1,
            geometry,
            "kda_recurrent",
            stage_layout=layout,
            layer_rows=layer_rows,
        )
    import ttnn

    tagged_kv = _config(geometry, "kda_recurrent", 93, 1)
    tagged_kv.tensor_type = ttnn.experimental.disaggregation.KvChunkTensorType.Kv
    with expect_error(AssertionError, "not Snapshot"):
        populate_kv_chunk_address_table_kda(
            _RecordingTable(),
            tagged_kv,
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


# --- SP-sharded storage ---

# (SP, TP, global heads): Galaxy K3; LoudBox SP8xTP1 and QuietBox SP4xTP1 perf layouts (24 heads per chip);
# QuietBox SP2xTP2 report layout.
SHARDED_MESHES = [(8, 4, 96), (8, 1, 24), (4, 1, 24), (2, 2, 48)]
SHARDED_IDS = ["galaxy-SP8xTP4", "loudbox-SP8xTP1", "quietbox-SP4xTP1", "quietbox-SP2xTP2"]


def _geometry(sp, tp, heads):
    return KdaContractGeometry(
        num_heads=heads, head_dim=128, conv_history=3, sequence_parallel_size=sp, tensor_parallel_size=tp
    )


def _walk(geometry, kind, layout, num_users, layer_rows=None):
    sp, tp = geometry.sequence_parallel_size, geometry.tensor_parallel_size
    table = _RecordingTable()
    populate_kv_chunk_address_table_kda(
        table,
        _config(geometry, kind, 2 * sum(s["count"] for s in layout), num_users),
        (sp, tp),
        0,
        1,
        geometry,
        kind,
        num_users=num_users,
        config_id=3,
        stage_layout=layout,
        layer_rows=layer_rows,
    )
    return table


@pytest.mark.parametrize("kind", ["kda_recurrent", "kda_convolution"])
@pytest.mark.parametrize("sp,tp,heads", SHARDED_MESHES, ids=SHARDED_IDS)
def test_sharded_walk_publishes_each_segment_on_its_one_store(sp, tp, heads, kind):
    geometry = _geometry(sp, tp, heads)
    counts, banks, num_users = (3, 2), (8, 7), 2
    layout = _fake_layout(counts, banks, sp, tp)
    table = _walk(geometry, kind, layout, num_users)

    segments = kda_segments_per_layer(geometry, kind)
    seg_bytes = kda_segment_bytes(geometry, kind)
    recurrent = kind == "kda_recurrent"
    shards_per_layer = geometry.recurrent_shards_per_layer if recurrent else geometry.convolution_shards_per_layer
    assert len(table.entries) == segments * sum(counts) * num_users
    # One singleton group per (rank, SP row, TP column): every chip is addressed on its own.
    assert sorted(table.groups) == sorted(((r, row, col),) for r in range(2) for row in range(sp) for col in range(tp))

    for stage in layout:
        for local_layer in range(stage["count"]):
            for slot in range(num_users):
                batch = slot * stage["count"] + local_layer
                shards_by_chip = {}
                for segment in range(segments):
                    noc_addr, size, group_idx = table.entries[
                        (3, local_layer + stage["first_layer"], kda_position(geometry, kind, segment), slot)
                    ]
                    assert size == seg_bytes
                    if recurrent:
                        tp_col, h_local, band = geometry.decompose_recurrent(segment)
                        sp_row, j = geometry.stored_head(h_local)
                        shard = geometry.recurrent_local_shard(batch, j, band)
                    else:
                        branch, tp_col, h_local, half = geometry.decompose_convolution(segment)
                        sp_row, j = geometry.stored_head(h_local)
                        shard = geometry.convolution_local_shard(batch, branch, j, half)
                    (chip,) = table.groups[group_idx]
                    assert chip == (stage["rank"], sp_row, tp_col)
                    assert noc_addr >> 32 == shard % stage["num_banks"]
                    assert noc_addr & 0xFFFFFFFF == stage["base_addr"] + (shard // stage["num_banks"]) * seg_bytes
                    shards_by_chip.setdefault(chip, []).append(shard)
                # Each chip's store of this batch is exactly its contiguous shard run, each shard once.
                assert len(shards_by_chip) == sp * tp
                for shards in shards_by_chip.values():
                    assert sorted(shards) == list(range(batch * shards_per_layer, (batch + 1) * shards_per_layer))


# Hand-checked (bank, offset) on one stage at base 0x1000_0000 with 8 banks, 3 layers, 2 users. A store's shards of
# batch b are [b * S, (b + 1) * S), S = H_store * 4 (recurrent) or 6 * H_store (convolution); shard s sits in
# bank s % 8 at base + (s // 8) * segment bytes. Owners as in test_state_adapter_geometry.py's HAND_CHECKED.
#   (mesh, kind, segment, slot, local layer) -> (sp_row, tp_col, bank, offset)
HAND_CHECKED = [
    # S 12; batch 1*3+2 = 5; shard 5*12 + 6 = 66 -> bank 2, 8 * 16384.
    ("galaxy-SP8xTP4", (8, 4, 96), "kda_recurrent", 126, 1, 2, (2, 1, 2, 131_072)),
    # S 18; batch 0; shard 17 -> bank 1, 2 * 384.
    ("galaxy-SP8xTP4", (8, 4, 96), "kda_convolution", 575, 0, 0, (7, 3, 1, 768)),
    # batch 1; shard 18 + 10 = 28 -> bank 4, 3 * 384.
    ("galaxy-SP8xTP4", (8, 4, 96), "kda_convolution", 244, 0, 1, (0, 1, 4, 1_152)),
    # S 12; batch 2; shard 24 + 7 = 31 -> bank 7, 3 * 16384.
    ("loudbox-SP8xTP1", (8, 1, 24), "kda_recurrent", 55, 0, 2, (4, 0, 7, 49_152)),
    # batch 0; shard 4 -> bank 4, 0.
    ("loudbox-SP8xTP1", (8, 1, 24), "kda_convolution", 46, 0, 0, (7, 0, 4, 0)),
    # S 24; batch 1; shard 24 + 21 = 45 -> bank 5, 5 * 16384.
    ("quietbox-SP4xTP1", (4, 1, 24), "kda_recurrent", 69, 0, 1, (2, 0, 5, 81_920)),
    # S 36; batch 3; shard 108 + 15 = 123 -> bank 3, 15 * 384.
    ("quietbox-SP4xTP1", (4, 1, 24), "kda_convolution", 63, 1, 0, (1, 0, 3, 5_760)),
    # S 48; batch 0; shard 4 -> bank 4, 0.
    ("quietbox-SP2xTP2", (2, 2, 48), "kda_recurrent", 148, 0, 0, (1, 1, 4, 0)),
    # S 72; batch 2; shard 144 + 71 = 215 -> bank 7, 26 * 384.
    ("quietbox-SP2xTP2", (2, 2, 48), "kda_convolution", 215, 0, 2, (0, 0, 7, 9_984)),
]


@pytest.mark.parametrize(
    "mesh,kind,segment,slot,local_layer,expected", [case[1:] for case in HAND_CHECKED], ids=[c[0] for c in HAND_CHECKED]
)
def test_sharded_entries_are_hand_checked(mesh, kind, segment, slot, local_layer, expected):
    sp, tp, heads = mesh
    geometry = _geometry(sp, tp, heads)
    layout = [
        {
            "rank": 0,
            "first_layer": 0,
            "count": 3,
            "base_addr": 0x1000_0000,
            "num_banks": 8,
            "host_tag": 1,
            "fnids": [[(0, row, col) for col in range(tp)] for row in range(sp)],
        }
    ]
    table = _walk(geometry, kind, layout, num_users=2)
    noc_addr, _size, group_idx = table.entries[(3, local_layer, kda_position(geometry, kind, segment), slot)]
    ((_rank, sp_row, tp_col),) = table.groups[group_idx]
    assert (sp_row, tp_col, noc_addr >> 32, (noc_addr & 0xFFFFFFFF) - 0x1000_0000) == expected

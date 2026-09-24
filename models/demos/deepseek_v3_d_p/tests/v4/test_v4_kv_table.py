# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Device-free checks of the V4 linear chunk-table walk (tt/v4/kv_table.py) against a host model of the ND-shard
placement, and of the populate loop against a fake table."""

from models.demos.deepseek_v3_d_p.reference.deepseek_v4_flash_config import deepseek_v4_flash_hf_config
from models.demos.deepseek_v3_d_p.tt.v4 import kv_contract as kc
from models.demos.deepseek_v3_d_p.tt.v4 import kv_table as kt
from models.demos.deepseek_v3_d_p.tt.v4.layer_kinds import CSA, HCA, V4FlashKvGeometry, layers_of_kind


def _host_placement(num_slots, num_layers, rows, num_banks, base, chunk_bytes):
    """Independent model: the tensor [slots*layers, 1, rows, W] is cut into 32-row shards in row-major order and
    dealt to banks round-robin; shard k of bank b sits at base + k * chunk_bytes."""
    out = {}
    per_bank_count = [0] * num_banks
    flat = 0
    for slot in range(num_slots):
        for layer in range(num_layers):
            for r0 in range(0, rows, 32):
                b = flat % num_banks
                out[(slot, layer, r0)] = (b, base + per_bank_count[b] * chunk_bytes)
                per_bank_count[b] += 1
                flat += 1
    return out


def test_walk_linear_matches_the_round_robin_shard_model():
    spec = kc.spec("csa_unified")
    rows, banks, base = 128 + 1280, 8, 0x1000
    got = {
        (a.slot, a.layer, a.position): (a.bank, a.offset)
        for a in kt.walk_linear(
            num_slots=2,
            num_layers=3,
            rows=rows,
            num_banks=banks,
            base_addr=base,
            chunk_size_bytes=spec.chunk_size_bytes,
        )
    }
    assert got == _host_placement(2, 3, rows, banks, base, spec.chunk_size_bytes)
    a0 = next(kt.walk_linear(num_slots=1, num_layers=1, rows=64, num_banks=7, base_addr=0x40, chunk_size_bytes=100))
    assert a0.noc_addr == (0 << 32) | 0x40
    # a harvested part with 7 banks wraps at 7
    addrs = list(kt.walk_linear(num_slots=1, num_layers=1, rows=32 * 9, num_banks=7, base_addr=0, chunk_size_bytes=10))
    assert [a.bank for a in addrs] == [0, 1, 2, 3, 4, 5, 6, 0, 1] and addrs[7].offset == 10
    # headroom rows are addressed (they shift the next layer's chunks) but not emitted
    sub = list(
        kt.walk_linear(num_slots=1, num_layers=2, rows=96, num_banks=8, base_addr=0, chunk_size_bytes=10, extent=64)
    )
    assert [(a.layer, a.position, a.bank) for a in sub] == [(0, 0, 0), (0, 32, 1), (1, 0, 3), (1, 32, 4)]


def test_first_layer_offsets_the_kind_rank_for_pp_stages():
    cfg = deepseek_v4_flash_hf_config()
    all_hca = layers_of_kind(cfg, HCA)
    g1 = V4FlashKvGeometry.from_config(cfg, max_seq_len=10240, sp_factor=8, first_layer_idx=11, num_layers=11)
    first, count = kt.kind_rank_range(all_hca, g1.hca_layers)
    assert (first, count) == (4, 6) and all_hca[first : first + count] == list(g1.hca_layers)
    g0 = V4FlashKvGeometry.from_config(cfg, max_seq_len=10240, sp_factor=8, first_layer_idx=0, num_layers=11)
    assert kt.kind_rank_range(layers_of_kind(cfg, CSA), g0.csa_layers) == (0, 5)
    assert kt.kind_rank_range(all_hca, ()) == (0, 0)
    layers = sorted(
        {
            a.layer
            for a in kt.walk_linear(
                num_slots=1, num_layers=count, rows=32, num_banks=8, base_addr=0, chunk_size_bytes=8, first_layer=first
            )
        }
    )
    assert layers == list(range(4, 10))


class _FakeTable:
    def __init__(self):
        self.groups, self.hosts, self.sets = [], {}, []

    def add_device_group(self, fnids):
        self.groups.append(list(fnids))
        return len(self.groups) - 1

    def set_fabric_node_host(self, fid, host_name):
        self.hosts[fid] = host_name

    def set(self, layer, position, slot, loc, config_id):
        self.sets.append((config_id, layer, position, slot, loc.noc_addr, loc.size_bytes, loc.device_group_index))


def test_populate_group_merges_two_stages_into_one_config(monkeypatch):
    import types

    fake_ttnn = types.SimpleNamespace(
        experimental=types.SimpleNamespace(
            disaggregation=types.SimpleNamespace(KvCacheLocation=lambda: types.SimpleNamespace())
        )
    )
    monkeypatch.setitem(__import__("sys").modules, "ttnn", fake_ttnn)
    spec = kc.spec("hca_unified")
    stages = [
        {
            "first_layer": 0,
            "count": 2,
            "base_addr": 0x100,
            "num_banks": 8,
            "host_tag": 1,
            "fnids": [["a0", "a1"], ["a2", "a3"]],
        },
        {
            "first_layer": 2,
            "count": 0,
            "base_addr": 0,
            "num_banks": 8,
            "host_tag": 2,
            "fnids": [["b0"]],
        },  # a rank with none
        {"first_layer": 2, "count": 1, "base_addr": 0x900, "num_banks": 8, "host_tag": 3, "fnids": [["c0", "c1"]]},
    ]
    t = _FakeTable()
    kt.populate_group(t, 5, spec=spec, rows=192, num_slots=1, stages=stages)
    assert t.groups == [["a0", "a1", "a2", "a3"], ["c0", "c1"]] and t.hosts["c1"] == "host-00000003"
    layers = sorted({s[1] for s in t.sets})
    assert layers == [0, 1, 2] and all(s[0] == 5 for s in t.sets)
    assert len(t.sets) == 3 * (192 // 32)
    stage2 = [s for s in t.sets if s[1] == 2]
    assert stage2[0][4] == (0 << 32) | 0x900 and stage2[0][6] == 1 and stage2[0][5] == spec.chunk_size_bytes
    assert [s[2] for s in stage2] == [0, 32, 64, 96, 128, 160]

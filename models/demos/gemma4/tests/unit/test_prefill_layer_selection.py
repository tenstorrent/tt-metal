# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks: allocations and decoder layers are replaced before use."""

from types import SimpleNamespace

import pytest

from models.demos.gemma4.tt.runners import kv_caches, kv_chunk_table
from models.demos.gemma4.tt.runners.layer_selection import prefill_layer_ids


@pytest.mark.parametrize("start,count", [(0, 60), (57, 3)])
def test_selected_cache_families_and_semantic_table_rows(monkeypatch, start, count):
    layer_types = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)]
    seen = []

    def attention_config(config, layer):
        seen.append(layer)
        return SimpleNamespace(num_key_value_heads=16, head_dim=256)

    class Tensor:
        dtype = kv_caches.ttnn.bfloat8_b

        def buffer_address(self):
            return 0x100000

    monkeypatch.setattr(kv_caches, "Gemma4AttentionConfig", attention_config)
    monkeypatch.setattr(kv_caches, "init_ring_kv_cache", lambda *a, **kw: (Tensor(), Tensor()))
    monkeypatch.setattr(
        kv_caches, "init_packed_ring_kv_cache", lambda *a, **kw: kv_chunk_table.PackedRingKVCache(Tensor())
    )
    caches = kv_caches.allocate_ring_kv_caches(
        object(),
        SimpleNamespace(num_hidden_layers=60, layer_types=layer_types),
        SimpleNamespace(prefill=SimpleNamespace(sp=8), tp=4),
        num_users=2,
        max_seq_len=8192,
        num_layers=count,
        first_layer_idx=start,
    )
    assert seen == list(range(start, start + count))
    assert caches.layer_types[-3:] == ("sliding_attention", "sliding_attention", "full_attention")

    class Table:
        def __init__(self, configs):
            self.configs = configs
            self.rows = set()

        def config_name(self, i):
            return list(self.configs)[i]

        def num_configs(self):
            return len(self.configs)

        def add_device_group(self, nodes):
            return 0

        def set_fabric_node_host(self, *args, **kwargs):
            pass

        def set(self, layer, position, slot, location, config_id):
            self.rows.add((layer, config_id))

    monkeypatch.setattr(kv_chunk_table.ttnn.experimental.disaggregation, "KvChunkAddressTable", Table)
    monkeypatch.setattr(kv_chunk_table.ttnn.experimental.disaggregation, "KvCacheLocation", SimpleNamespace)
    monkeypatch.setattr(kv_chunk_table, "get_num_dram_banks", lambda mesh: 8)
    mesh = SimpleNamespace(
        shape=(8, 4),
        get_fabric_node_id=lambda coord: SimpleNamespace(mesh_id=0, chip_id=int(coord[0]) * 4 + int(coord[1])),
    )
    table = kv_chunk_table.build_kv_chunk_address_table(mesh_device=mesh, kv_caches=caches, chunk_size=8192)
    assert all(cfg.num_layers == 60 for cfg in table.configs.values())
    assert table.rows == {(layer, cfg) for layer in seen for cfg in (range(4) if layer % 6 == 5 else range(4, 36))}


def test_model_constructs_tail_weights_with_corresponding_external_caches(monkeypatch):
    from models.demos.gemma4.tt import model
    from models.demos.gemma4.tt.model_config import Gemma4ModelArgs

    seen = []
    caches = [object(), object(), object()]
    ropes_seen = []

    class Layer:
        def __init__(self, cache):
            self.self_attn = SimpleNamespace(kv_cache=None, ring_kv_cache=cache)

        def __call__(self, hidden_states, **kwargs):
            ropes_seen.append(kwargs["rope_mats"])
            return hidden_states

    def layer(**kwargs):
        seen.append((kwargs["layer_idx"], kwargs["ring_kv_cache"]))
        return Layer(kwargs["ring_kv_cache"])

    monkeypatch.setattr(model, "Gemma4DecoderLayer", layer)
    config = Gemma4ModelArgs()
    config.num_hidden_layers = 60
    config.num_kv_shared_layers = 0
    config.hidden_size_per_layer_input = 0
    config.layer_types = ["full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)]
    built = model.Gemma4Model(
        mesh_device=object(),
        hf_config=config,
        state_dict={},
        ccl_manager=None,
        num_layers=3,
        first_layer_idx=57,
        create_kv_cache=False,
        ring_kv_caches=caches,
        prefill_weights_only=True,
    )
    assert seen == list(zip((57, 58, 59), caches))
    assert built.layer_ids == (57, 58, 59)
    assert built.last_kv_layer_by_type == {"sliding_attention": 1, "full_attention": 2}
    ropes = {"sliding_attention": object(), "full_attention": object()}
    hidden = SimpleNamespace(shape=(1, 1, 32, config.hidden_size))
    acknowledgements = []
    monkeypatch.setattr(model.ttnn, "synchronize_device", lambda mesh: None)
    assert built(hidden, rope_mats=ropes, is_decode=False, on_layer_complete=acknowledgements.append) is hidden
    assert ropes_seen == [ropes["sliding_attention"], ropes["sliding_attention"], ropes["full_attention"]]
    assert acknowledgements == [0, 1, 2]


@pytest.mark.parametrize("count,start", [(60, 57), (0, 0), (3, -1), (4, 57)])
def test_invalid_prefill_ranges(count, start):
    with pytest.raises(ValueError):  # allow-pytest.raises: no device fixtures in this host-only test
        prefill_layer_ids(count, start, 60)

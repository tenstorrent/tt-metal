# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host checks for DFlash checkpoint, model taps, acknowledgements, and migration."""

import json
from dataclasses import asdict, replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from safetensors.torch import save_file

import ttnn
from models.demos.common.prefill.adapter import PrefillModelAdapter
from models.demos.gemma4_d_p.tests.unit.test_prefill_runner import service_params
from models.demos.gemma4_d_p.tt.dflash import DFlashKVCache, DFlashPrefill, allocate_dflash_kv_cache
from models.demos.gemma4_d_p.tt.dflash_config import DFlashConfig, dflash_tensor_cache_path, load_dflash_weights
from models.demos.gemma4_d_p.tt.model import Gemma4Model
from models.demos.gemma4_d_p.tt.runners.adapters import gemma4
from models.demos.gemma4_d_p.tt.runners.kv_caches import Gemma4KvCaches
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import build_kv_chunk_address_table
from models.demos.gemma4_d_p.tt.runners.runtime import Gemma4PrefillRuntime


def config():
    return DFlashConfig(128, 5, 8, 128, (1, 3), 1e-6, 1e6, 16384)


def config_json():
    data = asdict(config())
    data["model_type"] = "qwen3"
    data["dflash_config"] = {"target_layer_ids": data.pop("target_layer_ids")}
    return data


def test_selective_checkpoint_loading_and_cache_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    (tmp_path / "config.json").write_text(json.dumps(config_json()))
    weights = {key: torch.ones(shape, dtype=torch.bfloat16) for key, shape in config().weight_shapes().items()}
    save_file({**weights, "layers.0.self_attn.q_proj.weight": torch.zeros(7)}, str(tmp_path / "model.safetensors"))
    parsed, loaded = load_dflash_weights(tmp_path)
    assert parsed == config()
    assert loaded.keys() == weights.keys()
    first = dflash_tensor_cache_path(parsed, loaded, (8, 4))
    assert dflash_tensor_cache_path(parsed, loaded, (8, 4)) == first
    assert dflash_tensor_cache_path(parsed, loaded, (4, 8)) != first
    loaded["fc.weight"][0, 0] = 2
    assert dflash_tensor_cache_path(parsed, loaded, (8, 4)) != first


@pytest.mark.parametrize("bad_weight", ["missing", "shape"])
def test_invalid_checkpoint_weights(tmp_path, bad_weight, expect_error):
    (tmp_path / "config.json").write_text(json.dumps(config_json()))
    weights = {key: torch.ones(shape) for key, shape in config().weight_shapes().items()}
    if bad_weight == "missing":
        del weights["fc.weight"]
    else:
        weights["fc.weight"] = torch.ones(3, 4)
    save_file(weights, str(tmp_path / "model.safetensors"))
    with expect_error(ValueError, "fc.weight"):
        load_dflash_weights(tmp_path)


@pytest.mark.parametrize(
    "override",
    [
        {"attention_bias": True},
        {"rope_scaling": {"factor": 2}},
        {"model_type": "gemma4"},
        {"dflash_config": {"target_layer_ids": [1, 1]}},
        {"num_hidden_layers": 0},
    ],
)
def test_reject_unsupported_config(override, expect_error):
    with expect_error(ValueError, "DFlash"):
        DFlashConfig.from_dict({**config_json(), **override})


@pytest.mark.parametrize("kwargs", [{"hidden_size": 64}, {"num_layers": 3}])
def test_validate_target(kwargs, expect_error):
    with expect_error(ValueError, "DFlash"):
        config().validate(SimpleNamespace(tp_degree=4, cp_degree=8), 8192, **kwargs)


def test_validate_mesh_capacity_and_slots(expect_error):
    mesh_config = SimpleNamespace(tp_degree=4, cp_degree=8)
    for capacity in (0, 16385, 32768):
        with expect_error(ValueError, "capacity"):
            config().validate(mesh_config, capacity)
    with expect_error(ValueError, "heads"):
        replace(config(), num_key_value_heads=7).validate(mesh_config, 8192)
    with expect_error(ValueError, "slot"):
        allocate_dflash_kv_cache(mesh_config, config(), num_users=0, max_seq_len=8192)


@pytest.mark.parametrize("bad", ["dtype", "shape", "capacity"])
def test_reject_mismatched_external_cache(bad, expect_error):
    mesh_config = SimpleNamespace(tp_degree=4, cp_degree=8)
    tensor = SimpleNamespace(shape=(10, 2, 1024, 128), dtype=ttnn.bfloat8_b)
    cache = DFlashKVCache(tensor, tensor, config(), 2, 8192)
    if bad == "dtype":
        tensor.dtype = ttnn.bfloat16
    elif bad == "shape":
        tensor.shape = (5, 2, 1024, 128)
    else:
        cache = replace(cache, max_seq_len=4096)
    with expect_error(ValueError, "DFlash cache"):
        DFlashPrefill(mesh_config, config(), {}, None, cache, max_seq_len=8192, chunk_size=8192)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("ack_mode", ["callback", "segmented_trace", "socket"])
def test_model_taps_reset_each_chunk_and_ack_after_writes(monkeypatch, enabled, ack_mode):
    events = []
    model = object.__new__(Gemma4Model)
    model.mesh_device = object()
    model.mesh_config = SimpleNamespace(cp_degree=8)
    model.hf_config = SimpleNamespace(layer_types=["sliding_attention"] * 4)
    model._packed_global_rope_trans_mat = None
    model._rope_prefill_positions = None
    model._prefill_metadata_external = False
    model.prefill_metadata = SimpleNamespace(update=Mock())
    model._get_rope_mats = lambda *args, **kwargs: (None, None)
    model._prefill_trace_controller = (
        SimpleNamespace(layer_ack=lambda idx: events.append(("ack", idx))) if ack_mode == "segmented_trace" else None
    )

    def decoder(idx):
        def forward(hidden, **kwargs):
            events.append(("target", idx))
            return SimpleNamespace(shape=hidden.shape, value=hidden.value * 2 + idx)

        return forward

    model.layers = [decoder(idx) for idx in range(4)]
    taps = []

    def tap(hidden, idx, accumulator):
        taps.append((idx, hidden.value))
        if idx == 1:
            assert accumulator is None
        return (accumulator or 0) + hidden.value

    def write_kv(accumulator, metadata, *, positions, on_layer_complete):
        assert accumulator == sum(value for _, value in taps[-2:])
        assert metadata is model.prefill_metadata and positions is None
        for idx in range(5):
            events.append(("draft", idx))
            on_layer_complete(idx)

    model.dflash = (
        SimpleNamespace(fc={1: None, 3: None}, tap=tap, write_kv=write_kv, stage_positions=Mock()) if enabled else None
    )
    monkeypatch.setattr(ttnn, "synchronize_device", lambda device: None)

    def socket_ack(service, *, metadata):
        writes = sum(kind in ("target", "draft") for kind, _ in events)
        events.append(("ack", writes - 1))

    monkeypatch.setattr(ttnn.experimental.deepseek_prefill, "outbound_socket_service_sync", socket_ack)
    for turn in range(2):
        events.clear()
        result = model(
            SimpleNamespace(shape=(1, 1, 1024, 128), value=turn + 1),
            user_id=turn,
            chunk_start_idx=turn * 8192,
            on_layer_complete=lambda idx: events.append(("ack", idx)),
            d2h_service=object() if ack_mode == "socket" else None,
            metadata_msg=object(),
        )
        expected = []
        for idx in range(4):
            expected += [("target", idx), ("ack", idx)]
        if enabled:
            for idx in range(5):
                expected += [("draft", idx), ("ack", idx + 4)]
            assert taps[-2:] == [(1, 4 * (turn + 1) + 1), (3, result.value)]
            model.dflash.stage_positions.assert_called_with(turn * 8192)
        assert events == expected


def test_service_capabilities_and_ack_counts():
    assert not PrefillModelAdapter.supports_dflash_trace
    adapter = gemma4.Gemma4PrefillAdapter()
    assert adapter.supports_dflash and adapter.supports_dflash_trace
    params = replace(service_params(), dflash_enabled=True)
    gemma4.validate_params(params)
    runtime = object.__new__(Gemma4PrefillRuntime)
    runtime.config = params
    runtime.model = SimpleNamespace(dflash=SimpleNamespace(config=config()))
    runtime.d2h_service = object()
    assert runtime.num_ack_layers == runtime.warmup_ack_count() == 65
    assert runtime.layer_ack_layers(60, 60) == (65, 65)
    runtime.model.dflash = None
    assert runtime.layer_ack_layers(60, 60) == (60, 60)


def test_draft_migration_rows_addresses_and_stages(monkeypatch):
    from models.demos.common.prefill.runners import migration_driver, prefill_producer
    from models.demos.gemma4_d_p.tt.attention.ring_prefill import GlobalRingKVCache, SlidingRingKVCache

    def tensor(address):
        return SimpleNamespace(dtype=ttnn.bfloat8_b, buffer_address=lambda: address)

    draft = DFlashKVCache(tensor(0x300000), tensor(0x400000), config(), 2, 1024)
    caches = Gemma4KvCaches(
        layers=[
            GlobalRingKVCache(tensor(0x100000))
            if i % 6 == 5
            else SlidingRingKVCache(tensor(0x100000), tensor(0x200000))
            for i in range(60)
        ],
        layer_types=tuple("full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)),
        num_users=2,
        max_seq_len=1024,
        cp=8,
        tp=4,
        dflash=draft,
    )
    mesh_device = SimpleNamespace(
        shape=(8, 4),
        dram_grid_size=lambda: SimpleNamespace(x=8),
        get_fabric_node_id=lambda coords: ttnn.FabricNodeId(ttnn.MeshId(0), coords[0] * 4 + coords[1]),
    )
    table = build_kv_chunk_address_table(mesh_device=mesh_device, kv_caches=caches, chunk_size=256)
    assert table.num_configs() == 52
    assert table.config_name(35) == "35_sliding_v_h15"
    for kind, base in [("k", 0x300000), ("v", 0x400000)]:
        for layer in range(5):
            for head in range(8):
                cfg_id = table.config_id_of(f"dflash_{kind}_h{head:02d}")
                assert table.config(cfg_id).num_layers == 65
                for slot in range(2):
                    # Third chunk, CP rank 3; each rank holds one 32-token row per chunk.
                    entry = table.lookup(60 + layer, 2 * 256 + 3 * 32, slot, cfg_id)
                    shard = ((slot * 5 + layer) * 2 + head % 2) * 4 + 2
                    assert entry.noc_addr == ((shard % 8) << 32 | base + (shard // 8) * 4352)
                    assert entry.size_bytes == 4352
                    nodes = table.get_device_group(entry.device_group_index).fabric_node_ids
                    assert int(nodes[0].chip_id) == 3 * 4 + head // 2

    monkeypatch.setattr(gemma4, "load_dflash_config", lambda path: config())
    monkeypatch.setattr(prefill_producer, "ADAPTER", gemma4.Gemma4PrefillAdapter())
    monkeypatch.setattr(prefill_producer, "NUM_LAYERS", 60)
    plan = migration_driver._cache_plan(table, None)
    assert len(plan) == 52
    for entry in plan[36:]:
        assert entry["rows"] == {i: i for i in range(60, 65)}
        assert entry["head_dim"] == 128
    runtime = object.__new__(Gemma4PrefillRuntime)
    runtime.config = SimpleNamespace(num_layers=60)
    monkeypatch.setattr(runtime, "_check_cache", lambda supplied: None)
    stages = runtime.kv_migration_stages(caches, 0, 60)
    assert [(s.base_addr, s.first_layer, s.count) for s in stages[-2:]] == [
        (0x300000, 60, 5),
        (0x400000, 60, 5),
    ]

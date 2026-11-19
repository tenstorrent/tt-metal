# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only: Kimi-K3 exposes kvpe, KDA recurrent and KDA convolution as migration stages 0, 1, 2;
Kimi-K3.1 adds its DSA index cache as stage 1, moving the KDA state to 2 and 3.

Every rank must return the same number of stages (the per-stage allgather is collective), numbered in
compacted slot space, and the adapter hooks must map the configs back to model layers.
"""

from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.adapter import get_adapter
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.layer_schedule import KimiK3LayerSchedule
from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
from models.demos.deepseek_v3_d_p.tt.runners.adapters.kimi_k3 import KimiK3Adapter
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import merged_num_layers

SPLITS = [(0, 24), (24, 24), (48, 24), (72, 21)]


def _tensor(address, shape):
    return SimpleNamespace(buffer_address=lambda: address, shape=shape)


def _runtime(first, count, num_users=1, dflash=False, model="kimi_k3"):
    runtime = TtKimiK3Runtime.__new__(TtKimiK3Runtime)
    runtime.config = SimpleNamespace(
        first_layer_idx=first, num_layers=count, num_users=num_users, dflash_enabled=dflash
    )
    runtime.hf_config = get_adapter(model).load_hf_config()
    return runtime


def _caches(first, count, num_users=1, *, kvpe=True, kda=True, index=False):
    schedule = KimiK3LayerSchedule.build(KimiK3Config, first, count)
    kvpe_cache = None
    if kvpe and schedule.num_mla_layers:
        kvpe_cache = SimpleNamespace(storage=_tensor(0x1000, (num_users * schedule.num_mla_layers, 1, 7040, 576)))
    kda_states = None
    ids = schedule.rank_kda_layer_ids()
    if kda and ids:
        kda_states = SimpleNamespace(
            layer_ids=ids,
            num_slots=num_users,
            recurrent=_tensor(0x2000, (num_users * len(ids), 24, 128, 128)),
            convolution=_tensor(0x3000, (num_users * len(ids), 144, 192)),
        )
    index_cache = None
    if index and kvpe_cache is not None:
        index_cache = _tensor(0x1800, (num_users * schedule.num_mla_layers, 1, 7040, 128))
    return SimpleNamespace(kvpe=kvpe_cache, index=index_cache, kda_states=kda_states)


@pytest.mark.parametrize("first,count", SPLITS, ids=[f"L{f}-{f + c}" for f, c in SPLITS])
def test_four_galaxy_split_emits_three_stages_in_compacted_slot_space(first, count):
    stages = _runtime(first, count).kv_migration_stages(_caches(first, count))
    assert len(stages) == 3
    kvpe, recurrent, convolution = stages
    mla_before = sum(1 for layer in KimiK3Config.mla_layer_ids() if layer < first)
    kda_before = sum(1 for layer in KimiK3Config.kda_layer_ids() if layer < first)
    assert (kvpe.base_addr, kvpe.first_layer, kvpe.count) == (0x1000, mla_before, 6)
    assert (recurrent.first_layer, recurrent.count) == (kda_before, count - 6)
    assert (convolution.first_layer, convolution.count) == (kda_before, count - 6)
    assert recurrent.base_addr == 0x2000 and convolution.base_addr == 0x3000


def test_compacted_kda_slots_tile_the_model():
    firsts = [_runtime(f, c).kv_migration_stages(_caches(f, c))[1] for f, c in SPLITS]
    assert [(s.first_layer, s.count) for s in firsts] == [(0, 18), (18, 18), (36, 18), (54, 15)]
    assert sum(s.count for s in firsts) == len(KimiK3Config.kda_layer_ids()) == 69


def test_single_galaxy_24_layer_rank():
    stages = _runtime(0, 24).kv_migration_stages(_caches(0, 24))
    assert [(s.first_layer, s.count) for s in stages] == [(0, 6), (0, 18), (0, 18)]


def test_layer_zero_only_rank_has_no_kvpe_stage_but_keeps_three():
    stages = _runtime(0, 1).kv_migration_stages(_caches(0, 1))
    assert (stages[0].base_addr, stages[0].first_layer, stages[0].count) == (0, 0, 0)
    assert [(s.first_layer, s.count) for s in stages[1:]] == [(0, 1), (0, 1)]


def test_kda_only_later_rank_keeps_the_kvpe_layout_contiguous():
    first, second = (_runtime(f, c).kv_migration_stages(_caches(f, c))[0] for f, c in [(0, 12), (12, 3)])
    assert (second.base_addr, second.first_layer, second.count) == (0, 3, 0)
    layout = [{"first_layer": s.first_layer, "count": s.count} for s in (first, second)]
    assert merged_num_layers(layout) == 3


def test_missing_slabs_or_mismatched_slabs_are_refused(expect_error):
    with expect_error(RuntimeError, "no KDA state slabs"):
        _runtime(0, 24).kv_migration_stages(_caches(0, 24, kda=False))
    caches = _caches(24, 24)
    caches.kda_states.layer_ids = caches.kda_states.layer_ids[:-1]
    with expect_error(RuntimeError, "cover layers"):
        _runtime(24, 24).kv_migration_stages(caches)


def test_dspark_drafter_rows_match_its_layer_completions():
    """The drafter's K and V follow the three stages, on the rows its layer completions are numbered by."""
    from models.demos.common.prefill.runners.layer_completion_layout import layer_completion_layout

    tail = _runtime(72, 21, dflash=True)
    tail._dflash_k_cache, tail._dflash_v_cache = _tensor(0x4000, ()), _tensor(0x5000, ())
    tail.drafter = SimpleNamespace(config=SimpleNamespace(num_hidden_layers=5))
    *_, k, v = tail.kv_migration_stages(_caches(72, 21))
    assert [(s.base_addr, s.first_layer, s.count) for s in (k, v)] == [(0x4000, 93, 5), (0x5000, 93, 5)]
    layout = layer_completion_layout(SPLITS, 3, extra_layers=5)
    records = range(layout.rank_first_record, layout.rank_first_record + layout.rank_record_count)
    assert list(records[-5:]) == list(range(k.first_layer, k.first_layer + k.count))

    upstream = _runtime(0, 24, dflash=True)
    upstream._dflash_k_cache = None
    stages = upstream.kv_migration_stages(_caches(0, 24))
    assert [(s.base_addr, s.count) for s in stages[3:]] == [(0, 0), (0, 0)]


def test_table_layer_rows_follow_the_gathered_stages():
    runtime = _runtime(0, 24)
    layout = [{"first_layer": f, "count": c} for f, c in [(0, 18), (18, 18), (36, 18), (54, 15)]]
    assert runtime.kda_table_layer_rows(layout) == KimiK3Config.kda_layer_ids()
    assert runtime.kda_table_layer_rows(layout[:1]) == KimiK3Config.kda_layer_ids()[:18]
    assert runtime.kv_table_layer_rows([[{"first_layer": 0, "count": 6}]]) == KimiK3Config.mla_layer_ids()[:6]


def test_adapter_hooks_name_the_three_configs():
    adapter = KimiK3Adapter()
    assert [adapter.cache_kind(i) for i in range(4)] == ["kvpe", "kda_recurrent", "kda_convolution", "other"]
    assert adapter.cache_layer_rows(2, 24) == {layer: layer for layer in KimiK3Config.kda_layer_ids() if layer < 24}
    assert len(adapter.cache_layer_rows(2, 24)) == 18
    assert adapter.cache_head_dim(0) == 576 and adapter.cache_head_dim(1) is None
    assert set(KimiK3Config.kda_layer_ids()) | set(KimiK3Config.mla_layer_ids()) == set(range(93))


def test_layer_position_range_follows_the_contract():
    """MLA layers over the request; KDA layers over their one Snapshot window, whatever the prompt length."""
    adapter = KimiK3Adapter()
    assert adapter.layer_position_range(3, 100) == (0, 100)
    assert adapter.layer_position_range(0, 100) == (0, 36_864)
    assert adapter.layer_position_range(0, 56_320) == (0, 36_864)
    assert adapter.layer_position_range(0, 1) == (0, 36_864)
    assert adapter.layer_position_range(92, 56_320) == (0, 56_320)


def test_driver_groups_consecutive_equal_range_layers_into_runs():
    from models.demos.common.prefill.runners.migration_driver import _layer_runs

    real_len = 56_320
    runs = _layer_runs(range(KimiK3Config.NUM_LAYERS), real_len, KimiK3Adapter().layer_position_range)
    window = (0, 36_864)
    assert runs[:4] == [(0, 3, *window), (3, 4, 0, real_len), (4, 7, *window), (7, 8, 0, real_len)]
    assert runs[-1] == (91, 93, 0, real_len)  # layers 91 and 92 are both MLA: one call
    assert len(runs) == 46
    assert all(runs[idx][1] == runs[idx + 1][0] for idx in range(len(runs) - 1))
    mla = set(KimiK3Config.mla_layer_ids())
    for start, end, pos_start, pos_end in runs:
        kinds = {layer in mla for layer in range(start, end)}
        assert len(kinds) == 1
        assert (pos_start, pos_end) == ((0, real_len) if kinds.pop() else window)
    # At real_len == W the KDA window equals the MLA token range, so every layer shares one call.
    assert _layer_runs(range(KimiK3Config.NUM_LAYERS), 36_864, KimiK3Adapter().layer_position_range) == [
        (0, KimiK3Config.NUM_LAYERS, 0, 36_864)
    ]
    # A subset keeps its own layers only and never merges across a gap.
    assert _layer_runs([22, 23, 27], real_len, KimiK3Adapter().layer_position_range) == [
        (22, 23, *window),
        (23, 24, 0, real_len),
        (27, 28, 0, real_len),
    ]


# ---- Kimi-K3.1: the DSA index cache is config 1 -------------------------------------------------------


@pytest.mark.parametrize("first,count", SPLITS, ids=[f"L{f}-{f + c}" for f, c in SPLITS])
def test_kimi_k3_1_index_stage_shares_the_kvpe_slabs(first, count):
    k3 = _runtime(first, count).kv_migration_stages(_caches(first, count))
    k31 = _runtime(first, count, model="kimi_k3_1").kv_migration_stages(_caches(first, count, index=True))
    assert len(k31) == 4
    kvpe, index, recurrent, convolution = k31
    assert (index.base_addr, index.first_layer, index.count) == (0x1800, kvpe.first_layer, kvpe.count)
    # kvpe and the KDA state are exactly K3's.
    assert [kvpe, recurrent, convolution] == k3


def test_kimi_k3_1_rank_without_mla_layers_still_emits_four_stages():
    stages = _runtime(0, 1, model="kimi_k3_1").kv_migration_stages(_caches(0, 1, index=True))
    assert [(s.base_addr, s.first_layer, s.count) for s in stages[:2]] == [(0, 0, 0), (0, 0, 0)]
    assert [(s.first_layer, s.count) for s in stages[2:]] == [(0, 1), (0, 1)]


def test_index_cache_must_match_the_config(expect_error):
    with expect_error(RuntimeError, "allocate_kv_cache disagrees"):
        _runtime(0, 24).kv_migration_stages(_caches(0, 24, index=True))
    with expect_error(RuntimeError, "allocate_kv_cache disagrees"):
        _runtime(0, 24, model="kimi_k3_1").kv_migration_stages(_caches(0, 24))


# ---- adapter answers vs the table the runtime publishes ------------------------------------------------
_MESH = (8, 4)
# Physical kvpe rows: K3's bf8 TILE (512 + 64 wide), K3.1's scaled FP8 (656 B row on a 704 B page).
_KVPE_STORAGE = {
    "kimi_k3": dict(shape=(1, 1, 32, 576), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT),
    "kimi_k3_1": dict(
        shape=(1, 1, 32, 656), dtype=ttnn.fp8_e4m3, layout=ttnn.ROW_MAJOR_LAYOUT, buffer_aligned_page_size=lambda: 704
    ),
}
_TOKEN_CHUNK_BYTES = {("kimi_k3", "kvpe"): 18 * 1088, ("kimi_k3_1", "kvpe"): 32 * 704, ("kimi_k3_1", "index"): 4 * 1088}


def _gathered(stage):
    """One rank's stage as `allgather_kv_stage_layout` reports it."""
    rows, cols = _MESH
    return [
        {
            "rank": 0,
            "first_layer": stage.first_layer,
            "count": stage.count,
            "base_addr": stage.base_addr,
            "num_banks": 8,
            "host_tag": 0xABC0000,
            "fnids": [[ttnn.FabricNodeId(ttnn.MeshId(0), r * cols + c) for c in range(cols)] for r in range(rows)],
        }
    ]


def _published_table(model, tmp_path, monkeypatch) -> str:
    """The table `build_kv_chunk_table` publishes for a single-galaxy 24-layer rank, host-only (the real
    populate walks over gathered stage layouts)."""
    from models.demos.deepseek_v3_d_p.tt.runners import kv_chunk_table
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import PREFILL_CHUNK_TOKENS

    monkeypatch.setattr(kv_chunk_table, "is_tp_deduped_kv_cache", lambda *_: False)
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: 0)
    runtime = _runtime(0, 24, model=model)
    runtime.mesh_device = None
    runtime.config.__dict__.update(
        max_seq_len=PREFILL_CHUNK_TOKENS, mesh_shape=_MESH, sp_axis=0, tp_axis=1, chunk_size=PREFILL_CHUNK_TOKENS
    )
    caches = _caches(0, 24, index=model == "kimi_k3_1")
    layouts = [_gathered(stage) for stage in runtime.kv_migration_stages(caches)]
    # Then the physical row of each token cache (the chunk sizing reads only the last dim and the encoding).
    caches.kvpe.storage.__dict__.update(_KVPE_STORAGE[model])
    if caches.index is not None:
        caches.index.__dict__.update(shape=(1, 1, 32, 128), dtype=ttnn.bfloat8_b)
    caches.kda_states = None  # the KDA configs are described by the gathered layouts alone
    return runtime.build_kv_chunk_table(caches, str(tmp_path / f"{model}.pb"), stage_layouts=layouts)


@pytest.mark.parametrize("model", ["kimi_k3", "kimi_k3_1"])
def test_adapter_answers_match_the_published_table(model, tmp_path, monkeypatch):
    """Every config of the published table holds exactly the kind, layer rows, row width and position window
    the adapter reports for it."""
    adapter = get_adapter(model)
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(_published_table(model, tmp_path, monkeypatch))

    tensor_type = ttnn.experimental.disaggregation.KvChunkTensorType
    kinds = [adapter.cache_kind(i) for i in range(table.num_configs())]
    assert kinds == ["kvpe", *(["index"] if model == "kimi_k3_1" else []), "kda_recurrent", "kda_convolution"]
    assert adapter.cache_kind(table.num_configs()) == "other"
    for config_id, kind in enumerate(kinds):
        cfg = table.config(config_id)
        populated = {layer for layer in range(cfg.num_layers) if table.lookup(layer, 0, 0, config_id).size_bytes}
        rows = adapter.cache_layer_rows(config_id, 24)
        assert all(layer == row for layer, row in rows.items()) and populated == set(rows), kind
        # Past this rank too: over the whole model, the identity on the layers of the config's kind.
        ids = KimiK3Config.mla_layer_ids() if kind in ("kvpe", "index") else KimiK3Config.kda_layer_ids()
        assert adapter.cache_layer_rows(config_id, KimiK3Config.NUM_LAYERS) == {layer: layer for layer in ids}, kind
        head_dim = adapter.cache_head_dim(config_id)
        if kind in ("kvpe", "index"):
            assert cfg.tensor_type == tensor_type.Kv
            assert cfg.chunk_size_bytes == _TOKEN_CHUNK_BYTES[(model, kind)]
            assert head_dim == {"kvpe": 576, "index": 128}[kind]
            assert all(adapter.layer_position_range(layer, 100) == (0, 100) for layer in rows)
        else:
            assert cfg.tensor_type == tensor_type.Snapshot and head_dim is None
            assert all(adapter.layer_position_range(layer, 100) == (0, cfg.max_sequence_length) for layer in rows)


@pytest.mark.parametrize("model", ["kimi_k3", "kimi_k3_1"])
def test_kda_state_readback_walks_the_model_kda_configs(model, tmp_path, monkeypatch):
    """The read-back tool picks the KDA configs by the model's adapter: 1/2 on K3, 2/3 on K3.1 (never the
    index cache). With no local device every published layer is reported remote, so nothing is read."""
    import json

    from models.demos.deepseek_v3_d_p.tt.runners.kda_state_readback import main

    table = _published_table(model, tmp_path, monkeypatch)
    (tmp_path / "device_map.json").write_text("{}")
    out = tmp_path / "readback.json"
    argv = ["--table", table, "--device-map", str(tmp_path / "device_map.json"), "--real-len", "1024"]
    argv += ["--model", model, "--golden", str(tmp_path), "--num-layers", "24", "--json", str(out)]
    assert main(argv) == 2  # no layer reachable from this host
    walked = {(r["config"], r["kind"]) for r in json.loads(out.read_text())["layers"]}
    first_kda = 2 if model == "kimi_k3_1" else 1
    assert walked == {(first_kda, "kda_recurrent"), (first_kda + 1, "kda_convolution")}
    assert {r["status"] for r in json.loads(out.read_text())["layers"]} == {"remote"}


@pytest.mark.parametrize("model", ["kimi_k3", "kimi_k3_1"])
def test_driver_cache_plan_follows_cache_kind(model, tmp_path, monkeypatch):
    """The migration driver plans each published config by the adapter's `cache_kind`: rows on the model layer
    axis, a decode width for the token caches only (kvpe, and K3.1's index cache at 01), none for the KDA
    Snapshot configs (01/02 on K3, 02/03 on K3.1), which its dump skips and its byte verify walks whole."""
    from models.demos.common.prefill.runners import prefill_producer
    from models.demos.common.prefill.runners.migration_driver import _cache_plan

    adapter = get_adapter(model)
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(_published_table(model, tmp_path, monkeypatch))
    monkeypatch.setattr(prefill_producer, "ADAPTER", adapter)
    monkeypatch.setattr(prefill_producer, "NUM_LAYERS", 24)

    mla = [layer for layer in KimiK3Config.mla_layer_ids() if layer < 24]
    kda = [layer for layer in KimiK3Config.kda_layer_ids() if layer < 24]
    index = [("index", mla, 128)] if model == "kimi_k3_1" else []
    expected = [("kvpe", mla, 576), *index, ("kda_recurrent", kda, None), ("kda_convolution", kda, None)]
    plan = _cache_plan(table, None)
    assert [entry["config_id"] for entry in plan] == list(range(len(expected)))
    assert [(entry["kind"], sorted(entry["rows"]), entry["head_dim"]) for entry in plan] == expected


def test_producer_scores_the_k31_index_cache_against_the_trace(tmp_path, monkeypatch):
    """The runner gate's index check on K3.1: it reads the trace's single-file
    `indexer/indexer_k_layer_N` golden, walks config 1 at the DSA layers only (K3.1 has no `indexer_types`, and
    its other rows hold KDA state), and scores the device keys in the trace's natural basis (K3.1's index cache holds
    unrotated keys)."""
    import json

    import torch
    from safetensors.torch import save_file

    from models.demos.common.prefill.runners import prefill_producer

    real_len = 64
    adapter = get_adapter("kimi_k3_1")
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(
        _published_table("kimi_k3_1", tmp_path, monkeypatch)
    )
    monkeypatch.setattr(prefill_producer, "ADAPTER", adapter)
    monkeypatch.setattr(prefill_producer, "NUM_LAYERS", 24)
    monkeypatch.setattr(prefill_producer, "MTP_LEVELS", 0)
    for name in ("PREFILL_PCC_TAIL_WINDOW", "PREFILL_PCC_WINDOW_END", "PREFILL_PCC_GOLDEN_OFFSET"):
        monkeypatch.delenv(name, raising=False)

    trace = tmp_path / "trace"
    (trace / "kv_cache").mkdir(parents=True)
    (trace / "indexer").mkdir()
    (trace / "metadata.json").write_text(json.dumps({"n_prompt_tokens": real_len}))
    generator = torch.Generator().manual_seed(531)
    device = {}
    mla = [layer for layer in KimiK3Config.mla_layer_ids() if layer < 24]
    for layer in mla:
        kvpe, key = torch.randn(real_len, 576, generator=generator), torch.randn(real_len, 128, generator=generator)
        save_file(
            {f"kv_post_transform_layer_{layer}": kvpe.bfloat16()}, trace / "kv_cache" / f"layer_{layer}.safetensors"
        )
        save_file(
            {f"indexer_k_layer_{layer}": key.bfloat16()}, trace / "indexer" / f"indexer_k_layer_{layer}.safetensors"
        )
        device[0, layer] = kvpe.bfloat16().float()
        device[1, layer] = key.bfloat16().float()  # as TtIndexer.write_k stores it: unrotated

    reads = []

    def read_kv_slice(table, device_map, config_id, layer, slot_id, read_len, head_dim, decode, *, start=0):
        reads.append((config_id, layer))
        return device[config_id, layer][start:read_len]

    monkeypatch.setattr(prefill_producer, "_read_kv_slice", read_kv_slice)
    device_map = {(0, chip): chip for chip in range(_MESH[0] * _MESH[1])}
    mins = prefill_producer._read_slot_kv_and_check_pcc(table, device_map, 0, real_len, trace)

    assert mins == pytest.approx({"kvpe": 1.0, "index": 1.0}, abs=1e-4)
    assert sorted(layer for config, layer in reads if config == 1) == mla

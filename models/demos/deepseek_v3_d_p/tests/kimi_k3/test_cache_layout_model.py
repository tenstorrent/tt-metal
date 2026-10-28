# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks of the Kimi-K3 KVPE cache's declared layout (``CacheLayoutModel``) against the walked
merged table.

``TtKimiK3Runtime.build_kv_chunk_table`` publishes the walked merged table (kvpe "00", KDA "01" / "02").
The model declares "00" alone -- a compacted cache holding only the MLA layers, TP-deduped or not, on one
mesh or across pipeline stages -- and is written next to the table only when its derived "00" equals the
walked one. The KDA configs are never declared, so the published table is always the walk.
"""

import os
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.cache_layout import LAYOUT_MODEL_SUFFIX
from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import KimiK3Config
from models.demos.deepseek_v3_d_p.tt.kimi_k3.runtime import TtKimiK3Runtime
from models.demos.deepseek_v3_d_p.tt.runners import kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import KIMI_K3_LAYOUT_MODEL_ENV
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import BH_NUM_DRAM_BANKS, PREFILL_CHUNK_TOKENS

disaggregation = ttnn.experimental.disaggregation

SP, TP = 8, 4  # the KDA contract geometry's mesh
SEQ_LEN = 2 * PREFILL_CHUNK_TOKENS
NUM_USERS = 2
KVPE_DIM = 576  # kv_lora_rank 512 + qk_rope_head_dim 64
# Model-layer spans of the pipeline stages: layers 3 and 7 are MLA in the first, 11 in the second.
ONE_MESH = ((0, 12),)
TWO_STAGES = ((0, 8), (8, 4))


def _fnids(mesh_id, chip_id=lambda r, c: r * TP + c):
    return [[ttnn.FabricNodeId(ttnn.MeshId(mesh_id), chip_id(r, c)) for c in range(TP)] for r in range(SP)]


def _gathered(counts, base, chip_id):
    """One cache's gathered layout: stage i on mesh i, its slabs numbered in compacted slot space."""
    stages, first = [], 0
    for i, count in enumerate(counts):
        stages.append(
            {
                "rank": i,
                "first_layer": first,
                "count": count,
                "base_addr": base * (i + 1),
                "num_banks": BH_NUM_DRAM_BANKS,
                "host_tag": 0xABC0000 + i,
                "fnids": _fnids(i, chip_id),
            }
        )
        first += count
    return stages


def _stage_layouts(spans, chip_id=lambda r, c: r * TP + c):
    """The gathered kvpe, KDA recurrent and KDA convolution layouts of ``spans``."""
    mla = KimiK3Config.mla_layer_ids()
    mla_counts = [sum(1 for layer in mla if first <= layer < first + count) for first, count in spans]
    kda_counts = [count - mla_count for (_, count), mla_count in zip(spans, mla_counts)]
    return [
        _gathered(mla_counts, 0x10000, chip_id),
        _gathered(kda_counts, 0x1000000, chip_id),
        _gathered(kda_counts, 0x2000000, chip_id),
    ]


def _topology(tp_deduped):
    shape = ttnn.MeshShape(SP, TP)
    coords = [ttnn.MeshCoordinate(r, c) for r in range(SP) for c in range(TP)]
    if tp_deduped:
        return ttnn.TensorTopology(shape, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
    return ttnn.TensorTopology(ttnn.MeshShape([SP * TP]), [ttnn.PlacementReplicate()], coords)


def _build(path, spans, *, tp_deduped=True, chip_id=lambda r, c: r * TP + c):
    """Publishes rank 0's merged table over ``spans`` to ``path`` and returns it."""
    stage_layouts = _stage_layouts(spans, chip_id)
    kvpe_stage = stage_layouts[0][0]
    stripes = SP * TP if tp_deduped else SP
    storage = SimpleNamespace(
        shape=(NUM_USERS * kvpe_stage["count"], 1, SEQ_LEN // stripes, KVPE_DIM),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        tensor_topology=lambda: _topology(tp_deduped),
        buffer_address=lambda: kvpe_stage["base_addr"],
    )
    runtime = TtKimiK3Runtime.__new__(TtKimiK3Runtime)
    runtime.mesh_device = None
    runtime.config = SimpleNamespace(
        mesh_shape=(SP, TP),
        sp_axis=0,
        tp_axis=1,
        max_seq_len=SEQ_LEN,
        num_layers=spans[0][1],
        first_layer_idx=spans[0][0],
        num_users=NUM_USERS,
        chunk_size=PREFILL_CHUNK_TOKENS,
        dflash_enabled=False,
    )
    kv_caches = SimpleNamespace(kvpe=SimpleNamespace(storage=storage), kda_states=None, index=None)
    runtime.build_kv_chunk_table(kv_caches, path, stage_layouts=stage_layouts)
    return disaggregation.import_from_protobuf_file(path)


@pytest.fixture(autouse=True)
def _rank_zero(monkeypatch):
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: 0)
    monkeypatch.delenv(KIMI_K3_LAYOUT_MODEL_ENV, raising=False)


def _derived(model_path, table):
    """The table the model at ``model_path`` derives, its chips' hosts taken from ``table``."""
    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    derived = disaggregation.to_chunk_map(model)
    for mesh_id in {int(cache.mesh_id) for cache in model.caches}:
        for row in _fnids(mesh_id):
            for fid in row:
                derived.set_fabric_node_host(fid, host_name=table.get_host(fid))
    return derived


SPANS = pytest.mark.parametrize("spans", [ONE_MESH, TWO_STAGES], ids=["one_mesh", "two_stages"])
TP_DEDUPED = pytest.mark.parametrize("tp_deduped", [True, False], ids=["tp_deduped", "tp_replicated"])


@SPANS
@TP_DEDUPED
def test_the_model_derives_the_walked_kvpe_config(tmp_path, spans, tp_deduped):
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, spans, tp_deduped=tp_deduped)
    assert [walked.config_name(i) for i in range(walked.num_configs())] == ["00", "01", "02"]
    model_path = path + LAYOUT_MODEL_SUFFIX
    assert os.path.exists(model_path)
    derived = _derived(model_path, walked)
    assert disaggregation.first_table_difference(walked, derived, only_actual_configs=True) is None
    assert [derived.config_name(i) for i in range(derived.num_configs())] == ["00"]

    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    stage_mla_layers = [[3, 7], [11]] if spans == TWO_STAGES else [[3, 7, 11]]
    assert [list(cache.layers) for cache in model.caches] == stage_mla_layers
    assert all(cache.holds_only_published_layers and cache.name == "00" for cache in model.caches)
    assert [int(cache.mesh_id) for cache in model.caches] == list(range(len(spans)))
    assert model.geometry.num_layers == walked.config(0).num_layers == 12

    held = {3, 7, 11}
    for layer in range(12):
        assert (derived.lookup(layer, 0, 0, 0).size_bytes > 0) == (layer in held), f"layer {layer}"


@SPANS
def test_the_published_table_is_the_walk_kda_configs_included(tmp_path, monkeypatch, spans):
    walks = []
    walk = kv_chunk_table._build_merged_kv_chunk_table

    def recorded_walk(**kwargs):
        walks.append(walk(**kwargs))
        return walks[-1]

    monkeypatch.setattr(kv_chunk_table, "_build_merged_kv_chunk_table", recorded_walk)
    published = _build(os.path.join(str(tmp_path), "table.pb"), spans)
    assert len(walks) == 1
    assert disaggregation.first_table_difference(walks[0], published) is None
    # Layer 0 is KDA: its recurrent ("01") and convolution ("02") segment 0 are published.
    for kda_config in (1, 2):
        assert published.lookup(0, 0, 0, kda_config).size_bytes > 0


def test_a_model_that_derives_another_kvpe_config_is_not_written(tmp_path, monkeypatch):
    model_for = kv_chunk_table.merged_cache_layout_model
    # Declares a prefill chunk twice the writer's, so every chip owns the wrong positions.
    monkeypatch.setattr(
        kv_chunk_table,
        "merged_cache_layout_model",
        lambda **kwargs: model_for(**{**kwargs, "chunk_size": 2 * kwargs["chunk_size"]}),
    )
    path = os.path.join(str(tmp_path), "table.pb")
    published = _build(path, TWO_STAGES)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)
    assert published.num_configs() == 3


def test_a_layout_the_model_cannot_declare_keeps_the_walk(tmp_path):
    path = os.path.join(str(tmp_path), "table.pb")
    published = _build(path, TWO_STAGES, chip_id=lambda r, c: c * SP + r)
    assert published.num_configs() == 3 and published.total_entries() > 0
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_is_refused(tmp_path, monkeypatch, expect_error):
    """The model declares only "00", so the KDA configs have nothing to derive them from."""
    monkeypatch.setenv(KIMI_K3_LAYOUT_MODEL_ENV, "1")
    with expect_error(RuntimeError, "declares only some"):
        _build(os.path.join(str(tmp_path), "table.pb"), TWO_STAGES)

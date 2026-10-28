# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks of GLM-5.3's declared cache layout (``CacheLayoutModel``) against the walked merged
table.

``build_and_serialize_kv_chunk_table`` (merged path, ``index_kv_cache`` set) publishes the walked table of
config "00" (the scaled-FP8 row-major KVPE cache, every layer, with decode's extra MTP levels aliased to
level 1's rows) and "01" (the bfp8 lightning-indexer key cache, only the full-indexer layers). It derives
the same table from the caches' layout model and writes the model next to the table only when the two
agree, on one mesh and across pipeline stages, TP-deduped or not. With ``GLM53_KV_TABLE_FROM_LAYOUT_MODEL=1``
it publishes the derived table instead, which must equal the walked one.
"""

import os
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.cache_layout import LAYOUT_MODEL_SUFFIX
from models.demos.deepseek_v3_d_p.tt.runners import kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import (
    GLM53_LAYOUT_MODEL_ENV,
    build_and_serialize_kv_chunk_table,
)
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import BH_NUM_DRAM_BANKS, PREFILL_CHUNK_TOKENS

disaggregation = ttnn.experimental.disaggregation

SP = 4
TP = 2
SEQ_LEN = 2 * PREFILL_CHUNK_TOKENS
NUM_USERS = 2
KVPE_ROW_BYTES = 656  # GLM-5.3's packed scaled-FP8 latent + scales, padded to 704 in DRAM
INDEX_DIM = 128
DRAM_ALIGNMENT = 64
# Physical KVPE rows per stage: 6 decoder layers and the one MTP level prefill computes (row 6).
ONE_MESH = (7,)
TWO_STAGES = (4, 3)
# The full-indexer layers (the rest reuse an earlier layer's indexer), the MTP row's included.
INDEX_LAYER_IDS = [0, 2, 3, 6]
# Decode reads MTP levels 2 and 3 from level 1's rows.
MTP_ALIAS_ROWS = {7: 6, 8: 6}


def _fnids(mesh_id, chip_id=lambda r, c: r * TP + c):
    return [[ttnn.FabricNodeId(ttnn.MeshId(mesh_id), chip_id(r, c)) for c in range(TP)] for r in range(SP)]


def _gathered(counts, base, chip_id):
    """One cache's gathered layout: stage i on mesh i, its slabs numbered in the cache's own layer space."""
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


def _stage_layouts(kvpe_counts, chip_id=lambda r, c: r * TP + c):
    """The gathered KVPE and index layouts: each stage's index cache holds its full-indexer layers."""
    index_counts, first = [], 0
    for count in kvpe_counts:
        index_counts.append(sum(1 for layer in INDEX_LAYER_IDS if first <= layer < first + count))
        first += count
    return [_gathered(kvpe_counts, 0x10000, chip_id), _gathered(index_counts, 0x1000000, chip_id)]


def _topology(tp_deduped):
    shape = ttnn.MeshShape(SP, TP)
    coords = [ttnn.MeshCoordinate(r, c) for r in range(SP) for c in range(TP)]
    if tp_deduped:
        return ttnn.TensorTopology(shape, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
    return ttnn.TensorTopology(ttnn.MeshShape([SP * TP]), [ttnn.PlacementReplicate()], coords)


def _tensor(stage, width, dtype, layout, tp_deduped):
    stripes = SP * TP if tp_deduped else SP
    return SimpleNamespace(
        shape=(NUM_USERS * stage["count"], 1, SEQ_LEN // stripes, width),
        dtype=dtype,
        layout=layout,
        tensor_topology=lambda: _topology(tp_deduped),
        buffer_address=lambda: stage["base_addr"],
        buffer_aligned_page_size=lambda: -(-width // DRAM_ALIGNMENT) * DRAM_ALIGNMENT,
    )


def _build(path, kvpe_counts, *, tp_deduped=True, alias_rows=MTP_ALIAS_ROWS, chip_id=lambda r, c: r * TP + c):
    """Publishes rank 0's merged table over the stages ``kvpe_counts`` to ``path`` and returns it."""
    kvpe_layout, index_layout = _stage_layouts(kvpe_counts, chip_id)
    kvpe = _tensor(kvpe_layout[0], KVPE_ROW_BYTES, ttnn.fp8_e4m3, ttnn.ROW_MAJOR_LAYOUT, tp_deduped)
    index = _tensor(index_layout[0], INDEX_DIM, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, tp_deduped)
    build_and_serialize_kv_chunk_table(
        mesh_device=None,
        kvpe_cache=SimpleNamespace(storage=kvpe),
        seq_len=SEQ_LEN,
        num_layers=kvpe_counts[0],
        mesh_shape=(SP, TP),
        sp_axis=0,
        tp_axis=1,
        num_users=NUM_USERS,
        chunk_size_global=PREFILL_CHUNK_TOKENS,
        path=path,
        index_kv_cache=index,
        num_my_layers=kvpe_counts[0],
        stage_layouts=[kvpe_layout, index_layout],
        index_layer_ids=INDEX_LAYER_IDS,
        kvpe_alias_rows=alias_rows,
    )
    return disaggregation.import_from_protobuf_file(path)


@pytest.fixture(autouse=True)
def _rank_zero(monkeypatch):
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: 0)
    monkeypatch.delenv(GLM53_LAYOUT_MODEL_ENV, raising=False)


def _derives(model_path, table):
    """None when the model at ``model_path``, with the stages' hosts, derives ``table``."""
    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    derived = disaggregation.to_chunk_map(model)
    for mesh_id in {int(cache.mesh_id) for cache in model.caches}:
        for row in _fnids(mesh_id):
            for fid in row:
                derived.set_fabric_node_host(fid, host_name=table.get_host(fid))
    return disaggregation.first_table_difference(table, derived)


STAGES = pytest.mark.parametrize("kvpe_counts", [ONE_MESH, TWO_STAGES], ids=["one_mesh", "two_stages"])
TP_DEDUPED = pytest.mark.parametrize("tp_deduped", [True, False], ids=["tp_deduped", "tp_replicated"])
ALIASED = pytest.mark.parametrize("alias_rows", [MTP_ALIAS_ROWS, None], ids=["mtp_aliases", "no_aliases"])


@STAGES
@TP_DEDUPED
@ALIASED
def test_the_model_derives_the_walked_table(tmp_path, kvpe_counts, tp_deduped, alias_rows):
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, kvpe_counts, tp_deduped=tp_deduped, alias_rows=alias_rows)
    model_path = path + LAYOUT_MODEL_SUFFIX
    assert os.path.exists(model_path)
    assert _derives(model_path, walked) is None
    assert [walked.config_name(i) for i in range(walked.num_configs())] == ["00", "01"]
    assert walked.config(0).chunk_size_bytes == 32 * 704

    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    index = [cache for cache in model.caches if cache.name == "01"]
    assert all(cache.holds_only_published_layers for cache in index)
    assert sorted(layer for cache in index for layer in cache.layers) == INDEX_LAYER_IDS
    aliases = {alias: source for cache in model.caches for alias, source in cache.layer_aliases.items()}
    assert aliases == (alias_rows or {})


def test_an_alias_row_publishes_level_1s_chunks_and_the_index_only_its_layers(tmp_path):
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, TWO_STAGES)
    derived = disaggregation.to_chunk_map(
        disaggregation.import_cache_layout_model_from_protobuf_file(path + LAYOUT_MODEL_SUFFIX)
    )
    for alias, source in MTP_ALIAS_ROWS.items():
        for table in (walked, derived):
            assert table.lookup(alias, 0, 0, 0).noc_addr == table.lookup(source, 0, 0, 0).noc_addr
    for layer in range(max(MTP_ALIAS_ROWS) + 1):
        assert (derived.lookup(layer, 0, 0, 1).size_bytes > 0) == (layer in INDEX_LAYER_IDS), f"layer {layer}"


def test_a_model_that_derives_another_table_is_not_written(tmp_path, monkeypatch):
    model_for = kv_chunk_table.merged_cache_layout_model
    # Declares a prefill chunk twice the writer's, so every chip owns the wrong positions.
    monkeypatch.setattr(
        kv_chunk_table,
        "merged_cache_layout_model",
        lambda **kwargs: model_for(**{**kwargs, "chunk_size": 2 * kwargs["chunk_size"]}),
    )
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, TWO_STAGES)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)
    assert walked.num_configs() == 2


def test_a_mesh_numbered_column_major_keeps_the_walk(tmp_path):
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, TWO_STAGES, chip_id=lambda r, c: c * SP + r)
    assert walked.total_entries() > 0
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)


@STAGES
@TP_DEDUPED
def test_publishing_the_derived_table_gives_the_walked_one(tmp_path, monkeypatch, kvpe_counts, tp_deduped):
    walked = _build(os.path.join(str(tmp_path), "walked.pb"), kvpe_counts, tp_deduped=tp_deduped)
    monkeypatch.setenv(GLM53_LAYOUT_MODEL_ENV, "1")
    path = os.path.join(str(tmp_path), "derived.pb")
    derived = _build(path, kvpe_counts, tp_deduped=tp_deduped)
    assert disaggregation.first_table_difference(walked, derived) is None
    assert os.path.exists(path + LAYOUT_MODEL_SUFFIX)

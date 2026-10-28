# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks of the DeepSeek-V3 KVPE cache's declared layout (``CacheLayoutModel``) against the
classic table.

``build_and_serialize_kv_chunk_table`` (single-config path) publishes the walked table, derives the same
table from the cache's layout model with ``to_chunk_map``, and writes the model next to the table only when
the two agree, on one mesh and across pipeline stages on their own meshes. With
``DEEPSEEK_KV_TABLE_FROM_LAYOUT_MODEL=1`` it publishes the derived table instead, which must equal the
walked one.
"""

import os
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.cache_layout import LAYOUT_MODEL_SUFFIX
from models.demos.deepseek_v3_d_p.tt.runners import kv_chunk_table
from models.demos.deepseek_v3_d_p.tt.runners.kv_chunk_table import LAYOUT_MODEL_ENV, build_and_serialize_kv_chunk_table
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import BH_NUM_DRAM_BANKS, PREFILL_CHUNK_TOKENS

disaggregation = ttnn.experimental.disaggregation

SP = 4
TP = 2
SEQ_LEN = 2 * PREFILL_CHUNK_TOKENS
NUM_USERS = 2
KVPE_DIM = 576  # kv_lora_rank 512 + qk_rope_head_dim 64
ONE_MESH = (3,)  # layers per stage
TWO_STAGES = (2, 3)


def _fnids(mesh_id, chip_id=lambda r, c: r * TP + c):
    return [[ttnn.FabricNodeId(ttnn.MeshId(mesh_id), chip_id(r, c)) for c in range(TP)] for r in range(SP)]


def _stage_layout(stage_counts, *, chip_id=lambda r, c: r * TP + c, num_banks=BH_NUM_DRAM_BANKS):
    """The gathered KVPE layout: stage i on mesh i at its own base address and host."""
    stages, first = [], 0
    for i, count in enumerate(stage_counts):
        stages.append(
            {
                "rank": i,
                "first_layer": first,
                "count": count,
                "base_addr": 0x10000 * (i + 1),
                "num_banks": num_banks,
                "host_tag": 0xABC0000 + i,
                "fnids": _fnids(i, chip_id),
            }
        )
        first += count
    return stages


def _topology(tp_deduped):
    """The allocated topology: 1-D Replicate, or the TP-deduped cache's sequence sharded over both axes."""
    shape = ttnn.MeshShape(SP, TP)
    coords = [ttnn.MeshCoordinate(r, c) for r in range(SP) for c in range(TP)]
    if tp_deduped:
        return ttnn.TensorTopology(shape, [ttnn.PlacementShard(2), ttnn.PlacementShard(2)], coords)
    return ttnn.TensorTopology(ttnn.MeshShape([SP * TP]), [ttnn.PlacementReplicate()], coords)


def _stub_cache(stage_layers, *, tp_deduped=False, fp8=False):
    stripes = SP * TP if tp_deduped else SP
    storage = SimpleNamespace(
        shape=(NUM_USERS * stage_layers, 1, SEQ_LEN // stripes, KVPE_DIM),
        dtype=ttnn.fp8_e4m3 if fp8 else ttnn.bfloat8_b,
        layout=ttnn.ROW_MAJOR_LAYOUT if fp8 else ttnn.TILE_LAYOUT,
        tensor_topology=lambda: _topology(tp_deduped),
        buffer_aligned_page_size=lambda: KVPE_DIM,  # one fp8 token row
    )
    return SimpleNamespace(storage=storage)


def _build(path, stage_layout, *, tp_deduped=False, fp8=False):
    first_stage = stage_layout[0]["count"]
    build_and_serialize_kv_chunk_table(
        mesh_device=None,
        kvpe_cache=_stub_cache(first_stage, tp_deduped=tp_deduped, fp8=fp8),
        seq_len=SEQ_LEN,
        num_layers=first_stage,
        mesh_shape=(SP, TP),
        sp_axis=0,
        tp_axis=1,
        num_users=NUM_USERS,
        chunk_size_global=PREFILL_CHUNK_TOKENS,
        path=path,
        first_layer_idx=0,
        num_my_layers=first_stage,
        stage_layouts=[stage_layout],
    )
    return disaggregation.import_from_protobuf_file(path)


def _derives(model_path, table):
    """None when the model at ``model_path``, with the stages' hosts, derives ``table``."""
    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    derived = disaggregation.to_chunk_map(model)
    for mesh_id in range(len({int(cache.mesh_id) for cache in model.caches})):
        for row in _fnids(mesh_id):
            for fid in row:
                derived.set_fabric_node_host(fid, host_name=table.get_host(fid))
    return disaggregation.first_table_difference(table, derived)


@pytest.mark.parametrize("stage_counts", [ONE_MESH, TWO_STAGES], ids=["one_mesh", "two_stages"])
def test_the_model_derives_the_walked_table(tmp_path, monkeypatch, stage_counts):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, _stage_layout(stage_counts))
    model_path = path + LAYOUT_MODEL_SUFFIX
    assert os.path.exists(model_path)
    assert _derives(model_path, walked) is None
    assert walked.num_configs() == 1 and walked.config_name(0) == "0"

    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    assert [cache.name for cache in model.caches] == ["0"] * len(stage_counts)
    assert [(cache.first_layer, cache.num_layers, int(cache.mesh_id)) for cache in model.caches] == [
        (sum(stage_counts[:i]), count, i) for i, count in enumerate(stage_counts)
    ]
    assert model.geometry.num_layers == sum(stage_counts)


def test_a_model_that_derives_another_table_is_not_written(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    model_for = kv_chunk_table.deepseek_cache_layout_model
    # Declares a prefill chunk twice the writer's, so every SP row owns the wrong positions.
    monkeypatch.setattr(
        kv_chunk_table,
        "deepseek_cache_layout_model",
        lambda **kwargs: model_for(**{**kwargs, "chunk_size": 2 * kwargs["chunk_size"]}),
    )
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, _stage_layout(TWO_STAGES))
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)
    assert walked.num_configs() == 1


@pytest.mark.parametrize(
    "stage_layout, tp_deduped, fp8",
    [
        (_stage_layout(TWO_STAGES, chip_id=lambda r, c: c * SP + r), False, False),
        (_stage_layout(TWO_STAGES, num_banks=BH_NUM_DRAM_BANKS - 1), False, False),
        (_stage_layout(TWO_STAGES), True, False),
        (_stage_layout(TWO_STAGES), False, True),
    ],
    ids=["column_major_mesh", "harvested_bank", "tp_deduped", "fp8_row_major"],
)
def test_a_layout_the_model_cannot_declare_keeps_the_walk(tmp_path, monkeypatch, stage_layout, tp_deduped, fp8):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, stage_layout, tp_deduped=tp_deduped, fp8=fp8)
    assert walked.total_entries() > 0
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)


@pytest.mark.parametrize("stage_counts", [ONE_MESH, TWO_STAGES], ids=["one_mesh", "two_stages"])
def test_publishing_the_derived_table_gives_the_walked_one(tmp_path, monkeypatch, stage_counts):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    walked = _build(os.path.join(str(tmp_path), "walked.pb"), _stage_layout(stage_counts))
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    path = os.path.join(str(tmp_path), "derived.pb")
    derived = _build(path, _stage_layout(stage_counts))
    assert disaggregation.first_table_difference(walked, derived) is None
    assert os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_refuses_a_layout_the_model_cannot_declare(tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    with expect_error(RuntimeError, "not derivable"):
        _build(os.path.join(str(tmp_path), "table.pb"), _stage_layout(TWO_STAGES), tp_deduped=True)

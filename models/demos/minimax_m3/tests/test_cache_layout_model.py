# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks of the M3 caches' declared layout (``CacheLayoutModel``) against the classic table.

``build_and_serialize_kv_chunk_table`` publishes the walked table, derives the same table from the caches'
layout model with ``to_chunk_map``, and writes the model next to the table only when the two agree, on one
mesh and across pipeline stages on their own meshes. With ``M3_KV_TABLE_FROM_LAYOUT_MODEL=1`` it publishes
the derived table instead, which must equal the walked one.
"""

import os
import socket
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.cache_layout import LAYOUT_MODEL_SUFFIX
from models.demos.minimax_m3.tt.attention.kv_cache import BH_NUM_DRAM_BANKS
from models.demos.minimax_m3.tt.runners import kv_chunk_table
from models.demos.minimax_m3.tt.runners.kv_chunk_table import LAYOUT_MODEL_ENV, build_and_serialize_kv_chunk_table

disaggregation = ttnn.experimental.disaggregation

SP = 2
COLS = 4  # == num_kv_heads
CHUNK_SIZE = 256  # chunk_local = 128: four DRAM blocks per SP row, one per column when index_k is TP-deduped
SEQ_LEN = 512
NUM_USERS = 2
HEAD_DIM = 128
ONE_MESH = (4,)  # layers per stage
TWO_STAGES = (2, 3)
INDEX_K_LAYERS = [1, 3]


def _fnids(mesh_id, chip_id=lambda r, c: r * COLS + c):
    return [[ttnn.FabricNodeId(ttnn.MeshId(mesh_id), chip_id(r, c)) for c in range(COLS)] for r in range(SP)]


def _stage_layouts(stage_counts, chip_id=lambda r, c: r * COLS + c):
    """One layout per cache (k, v, index_k), each spanning the stages; every (stage, cache) has its own base
    address, and stage i is on mesh i."""
    layouts = []
    for cache_idx in range(3):
        stages, first = [], 0
        for i, count in enumerate(stage_counts):
            stages.append(
                {
                    "first_layer": first,
                    "count": count,
                    "base_addr": 0x10000 * (i + 1) + cache_idx * 0x4000,
                    "num_banks": BH_NUM_DRAM_BANKS,
                    "host_name": f"stage-{i}" if len(stage_counts) > 1 else socket.gethostname(),
                    "fnids": _fnids(i, chip_id),
                }
            )
            first += count
        layouts.append(stages)
    return layouts


def _stub_cache(stage_layers, index_k_tp_axis=None):
    def t(dtype):
        return SimpleNamespace(shape=(NUM_USERS * stage_layers, 1, SEQ_LEN, HEAD_DIM), dtype=dtype)

    return SimpleNamespace(
        k=t(ttnn.bfloat8_b), v=t(ttnn.bfloat8_b), index_k=t(ttnn.bfloat8_b), index_k_tp_axis=index_k_tp_axis
    )


def _build(path, stage_layouts, *, index_k_tp_axis=None):
    first_stage = stage_layouts[0][0]["count"]
    build_and_serialize_kv_chunk_table(
        mesh_device=None,
        kv_cache=_stub_cache(first_stage, index_k_tp_axis),
        seq_len=SEQ_LEN,
        num_layers=first_stage,
        mesh_shape=(SP, COLS),
        sp_axis=0,
        num_users=NUM_USERS,
        chunk_size=CHUNK_SIZE,
        num_kv_heads=COLS,
        head_dim=HEAD_DIM,
        path=path,
        stage_layouts=stage_layouts,
        index_k_layers=INDEX_K_LAYERS,
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
    walked = _build(path, _stage_layouts(stage_counts))
    model_path = path + LAYOUT_MODEL_SUFFIX
    assert os.path.exists(model_path)
    assert _derives(model_path, walked) is None

    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    names = [f"{i:02d}" for i in range(2 * COLS + 1)]
    assert [cache.name for cache in model.caches] == [name for name in names for _ in stage_counts]
    assert [
        (cache.first_layer, cache.num_layers, int(cache.mesh_id)) for cache in model.caches[: len(stage_counts)]
    ] == [(sum(stage_counts[:i]), count, i) for i, count in enumerate(stage_counts)]
    assert all(cache.layers == INDEX_K_LAYERS for cache in model.caches if cache.name == names[-1])
    assert model.geometry.num_layers == sum(stage_counts)


def test_a_model_that_derives_another_table_is_not_written(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    model_for = kv_chunk_table.m3_cache_layout_model
    # Declares a prefill chunk twice the writer's, so every SP row owns the wrong positions.
    monkeypatch.setattr(
        kv_chunk_table,
        "m3_cache_layout_model",
        lambda **kwargs: model_for(**{**kwargs, "chunk_size": 2 * kwargs["chunk_size"]}),
    )
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, _stage_layouts(TWO_STAGES))
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)
    assert walked.num_configs() == 2 * COLS + 1


@pytest.mark.parametrize(
    "stage_layouts, index_k_tp_axis",
    [
        (_stage_layouts(TWO_STAGES, chip_id=lambda r, c: c * SP + r), None),
        (_stage_layouts(TWO_STAGES), 1),
    ],
    ids=["column_major_mesh", "tp_deduped_index_k"],
)
def test_a_layout_the_model_cannot_declare_keeps_the_walk(tmp_path, monkeypatch, stage_layouts, index_k_tp_axis):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    path = os.path.join(str(tmp_path), "table.pb")
    _build(path, stage_layouts, index_k_tp_axis=index_k_tp_axis)
    assert os.path.exists(path)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_gives_the_walked_one(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    walked = _build(os.path.join(str(tmp_path), "walked.pb"), _stage_layouts(TWO_STAGES))
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    path = os.path.join(str(tmp_path), "derived.pb")
    derived = _build(path, _stage_layouts(TWO_STAGES))
    assert disaggregation.first_table_difference(walked, derived) is None
    assert os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_refuses_a_layout_the_model_cannot_declare(tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    with expect_error(RuntimeError, "not derivable"):
        _build(os.path.join(str(tmp_path), "table.pb"), _stage_layouts(TWO_STAGES), index_k_tp_axis=1)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Device-free checks of the GPT-OSS caches' declared layout (``CacheLayoutModel``) against the classic table.

``build_and_serialize_kv_chunk_table`` publishes the walked table, derives the same table from the caches'
layout model with ``to_chunk_map``, and writes the model next to the table only when the two agree. With
``GPT_OSS_KV_TABLE_FROM_LAYOUT_MODEL=1`` it publishes the derived table instead, which must equal the walked
one.
"""

import os
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.cache_layout import LAYOUT_MODEL_SUFFIX
from models.demos.gpt_oss_d_p.tt.runners import kv_chunk_table
from models.demos.gpt_oss_d_p.tt.runners.kv_chunk_table import LAYOUT_MODEL_ENV, build_and_serialize_kv_chunk_table

disaggregation = ttnn.experimental.disaggregation

SP = 2
COLS = 4  # == num_kv_heads
CHUNK_SIZE = 256  # chunk_local = 128: four DRAM blocks per SP row
SEQ_LEN = 512
NUM_USERS = 2
NUM_LAYERS = 3
HEAD_DIM = 64
BH_NUM_DRAM_BANKS = 8
K_BASE, V_BASE = 0x10000, 0x90000


def _row_major(r, c, cols):
    return r * cols + c


def _column_major(r, c, cols):
    return c * SP + r


def _stub_mesh_device(sp, cols, chip_id=_row_major):
    def get_fabric_node_id(coord):
        return ttnn.FabricNodeId(ttnn.MeshId(0), chip_id(coord[0], coord[1], cols))

    return SimpleNamespace(
        get_fabric_node_id=get_fabric_node_id,
        dram_grid_size=lambda: SimpleNamespace(x=BH_NUM_DRAM_BANKS, y=1),
        arch=lambda: ttnn.device.Arch.BLACKHOLE,
        shape=(sp, cols),
    )


def _stub_cache(num_layers, dtype, sp, bounded_sliding=False):
    def t(base):
        return SimpleNamespace(
            shape=(NUM_USERS * num_layers, 1, SEQ_LEN // sp, HEAD_DIM), dtype=dtype, buffer_address=lambda: base
        )

    return SimpleNamespace(k=t(K_BASE), v=t(V_BASE), bounded_sliding=bounded_sliding)


def _build(path, *, sp=SP, num_layers=NUM_LAYERS, dtype=ttnn.bfloat8_b, chip_id=_row_major, bounded_sliding=False):
    build_and_serialize_kv_chunk_table(
        mesh_device=_stub_mesh_device(sp, COLS, chip_id),
        kv_cache=_stub_cache(num_layers, dtype, sp, bounded_sliding),
        seq_len=SEQ_LEN,
        num_layers=num_layers,
        mesh_shape=(sp, COLS),
        sp_axis=0,
        num_users=NUM_USERS,
        chunk_size=CHUNK_SIZE,
        num_kv_heads=COLS,
        head_dim=HEAD_DIM,
        path=path,
    )
    return disaggregation.import_from_protobuf_file(path)


def _derives(model_path, table, sp):
    """None when the model at ``model_path``, with the mesh's hosts, derives ``table``."""
    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    derived = disaggregation.to_chunk_map(model)
    for r in range(sp):
        for c in range(COLS):
            fid = ttnn.FabricNodeId(ttnn.MeshId(0), r * COLS + c)
            derived.set_fabric_node_host(fid, host_name=table.get_host(fid))
    return disaggregation.first_table_difference(table, derived)


@pytest.mark.parametrize(
    "sp, num_layers, dtype",
    [
        (SP, NUM_LAYERS, ttnn.bfloat8_b),
        (SP, 1, ttnn.bfloat8_b),
        (SP, NUM_LAYERS, ttnn.bfloat16),
        (1, NUM_LAYERS, ttnn.bfloat8_b),
    ],
    ids=["bf8", "one_layer", "bf16", "sp1"],
)
def test_the_model_derives_the_walked_table(tmp_path, monkeypatch, sp, num_layers, dtype):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, sp=sp, num_layers=num_layers, dtype=dtype)
    model_path = path + LAYOUT_MODEL_SUFFIX
    assert os.path.exists(model_path)
    assert _derives(model_path, walked, sp) is None

    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    assert [cache.name for cache in model.caches] == [f"{i:02d}" for i in range(2 * COLS)]
    assert all((cache.first_layer, cache.num_layers) == (0, num_layers) for cache in model.caches)
    assert model.geometry.num_layers == num_layers
    assert walked.num_configs() == 2 * COLS


def test_a_model_that_derives_another_table_is_not_written(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    model_for = kv_chunk_table.gpt_oss_cache_layout_model
    # Declares a prefill chunk twice the writer's, so every SP row owns the wrong positions.
    monkeypatch.setattr(
        kv_chunk_table,
        "gpt_oss_cache_layout_model",
        lambda **kwargs: model_for(**{**kwargs, "chunk_size": 2 * kwargs["chunk_size"]}),
    )
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)
    assert walked.num_configs() == 2 * COLS


def test_a_mesh_not_numbered_row_major_keeps_the_walk(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    path = os.path.join(str(tmp_path), "table.pb")
    _build(path, chip_id=_column_major)
    assert os.path.exists(path)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_gives_the_walked_one(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    walked = _build(os.path.join(str(tmp_path), "walked.pb"))
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    path = os.path.join(str(tmp_path), "derived.pb")
    derived = _build(path)
    assert disaggregation.first_table_difference(walked, derived) is None
    assert os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_refuses_a_bounded_sliding_cache(tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    with expect_error(RuntimeError, "not derivable"):
        _build(os.path.join(str(tmp_path), "table.pb"), bounded_sliding=True)

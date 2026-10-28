# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the Llama prefill caches' declared layout (``CacheLayoutModel``) against the classic
table.

``build_and_serialize_kv_chunk_table`` publishes the walked table, derives the same table from the caches'
layout model with ``to_chunk_map``, and writes the model next to the table only when the two agree. With
``LLAMA_KV_TABLE_FROM_LAYOUT_MODEL=1`` it publishes the derived table instead, which must equal the walked one.
"""

import os
import socket
from types import SimpleNamespace

import ttnn
from models.demos.common.prefill.runners.cache_layout import LAYOUT_MODEL_SUFFIX
from models.demos.llama_3p1_8b_d_p.tt.kv_cache import _cache_memory_config
from models.demos.llama_3p1_8b_d_p.tt.runners import kv_chunk_table
from models.demos.llama_3p1_8b_d_p.tt.runners.kv_chunk_table import LAYOUT_MODEL_ENV, build_and_serialize_kv_chunk_table

disaggregation = ttnn.experimental.disaggregation

SP, TP = 4, 8
NUM_BANKS = 8  # Blackhole's DRAM banks
CHUNK_SIZE = 1024
MAX_SEQ_LEN = 2048
NUM_USERS = 2
NUM_LAYERS = 2  # fewer than the model's 32 to keep the walk small
HEAD_DIM = 128
CONFIG_NAMES = [f"{kv}_h{head}" for kv in ("k", "v") for head in range(TP)]


def _mesh(chip_id=lambda r, c: r * TP + c):
    return SimpleNamespace(
        shape=(SP, TP),
        arch=lambda: ttnn.device.Arch.BLACKHOLE,
        dram_grid_size=lambda: SimpleNamespace(x=NUM_BANKS, y=1),
        get_fabric_node_id=lambda coord: ttnn.FabricNodeId(ttnn.MeshId(0), chip_id(coord[0], coord[1])),
    )


def _cache(mesh):
    def tensor(base_addr):
        return SimpleNamespace(
            shape=(NUM_USERS * NUM_LAYERS, 1, MAX_SEQ_LEN // SP, HEAD_DIM),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=lambda: mesh,
            memory_config=lambda: _cache_memory_config(mesh),
            buffer_address=lambda: base_addr,
        )

    return SimpleNamespace(
        k=tensor(0x100000),
        v=tensor(0x900000),
        num_users=NUM_USERS,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN,
        sp=SP,
    )


def _build(path, mesh=None):
    mesh = mesh or _mesh()
    build_and_serialize_kv_chunk_table(mesh_device=mesh, kv_cache=_cache(mesh), chunk_size=CHUNK_SIZE, path=path)
    return disaggregation.import_from_protobuf_file(path)


def _derives(model_path, table):
    """None when the model at ``model_path``, with the mesh's hosts, derives ``table``."""
    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    derived = disaggregation.to_chunk_map(model)
    for chip in range(SP * TP):
        derived.set_fabric_node_host(ttnn.FabricNodeId(ttnn.MeshId(0), chip), host_name=socket.gethostname())
    return disaggregation.first_table_difference(table, derived)


def test_the_model_derives_the_walked_table(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path)
    assert walked.num_configs() == len(CONFIG_NAMES)
    assert walked.total_entries() == len(CONFIG_NAMES) * NUM_USERS * NUM_LAYERS * (MAX_SEQ_LEN // 32)
    model_path = path + LAYOUT_MODEL_SUFFIX
    assert os.path.exists(model_path)
    assert _derives(model_path, walked) is None

    model = disaggregation.import_cache_layout_model_from_protobuf_file(model_path)
    assert [cache.name for cache in model.caches] == CONFIG_NAMES
    assert all(
        (cache.first_layer, cache.num_layers, int(cache.mesh_id)) == (0, NUM_LAYERS, 0) for cache in model.caches
    )
    assert model.geometry.num_layers == NUM_LAYERS


def test_a_model_that_derives_another_table_is_not_written(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    model_for = kv_chunk_table.llama_cache_layout_model
    # Declares a prefill chunk twice the writer's, so every SP row owns the wrong positions.
    monkeypatch.setattr(
        kv_chunk_table,
        "llama_cache_layout_model",
        lambda **kwargs: model_for(**{**kwargs, "chunk_size": 2 * kwargs["chunk_size"]}),
    )
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)
    assert walked.num_configs() == len(CONFIG_NAMES)


def test_a_bank_count_other_than_the_archs_is_caught_and_not_written(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    mesh = _mesh()
    # The walk rotates over the mesh's DRAM grid; the model pages over the architecture's banks.
    mesh.dram_grid_size = lambda: SimpleNamespace(x=NUM_BANKS - 1, y=1)
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, mesh)
    assert walked.num_configs() == len(CONFIG_NAMES)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_a_mesh_not_numbered_row_major_keeps_the_walk(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path, _mesh(chip_id=lambda r, c: c * SP + r))
    assert walked.num_configs() == len(CONFIG_NAMES)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_gives_the_walked_one(tmp_path, monkeypatch):
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)
    walked = _build(os.path.join(str(tmp_path), "walked.pb"))
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    path = os.path.join(str(tmp_path), "derived.pb")
    derived = _build(path)
    assert disaggregation.first_table_difference(walked, derived) is None
    assert os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_refuses_a_layout_the_model_cannot_declare(tmp_path, monkeypatch, expect_error):
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    with expect_error(RuntimeError, "not derivable"):
        _build(os.path.join(str(tmp_path), "table.pb"), _mesh(chip_id=lambda r, c: c * SP + r))

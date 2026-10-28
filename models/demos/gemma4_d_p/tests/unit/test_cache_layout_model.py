# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Device-free checks of the Gemma 4 caches' declared layout (``CacheLayoutModel``) against the classic table.

``build_and_serialize_kv_chunk_table`` publishes the walked table, derives the same table from the caches'
layout model with ``to_chunk_map`` (global heads per TP column, sliding heads selected one per config), and
writes the model next to the table only when the two agree. With ``GEMMA4_KV_TABLE_FROM_LAYOUT_MODEL=1`` it
publishes the derived table instead, which must equal the walked one.
"""

import os
from types import SimpleNamespace

import pytest

import ttnn
from models.demos.common.prefill.runners.cache_layout import LAYOUT_MODEL_SUFFIX
from models.demos.gemma4_d_p.tt.attention.ring_prefill import GlobalRingKVCache, SlidingRingKVCache
from models.demos.gemma4_d_p.tt.runners import kv_chunk_table
from models.demos.gemma4_d_p.tt.runners.kv_caches import Gemma4KvCaches
from models.demos.gemma4_d_p.tt.runners.kv_chunk_table import (
    CONFIG_NAMES,
    LAYOUT_MODEL_ENV,
    build_and_serialize_kv_chunk_table,
)

disaggregation = ttnn.experimental.disaggregation

CHUNK_SIZE = 256


def _caches():
    address = 0

    def tensor():
        nonlocal address
        address += 0x100000
        base = address
        return SimpleNamespace(dtype=ttnn.bfloat8_b, buffer_address=lambda: base)

    return Gemma4KvCaches(
        layers=[SlidingRingKVCache(tensor(), tensor()) for _ in range(5)] + [GlobalRingKVCache(tensor())],
        layer_types=("sliding_attention",) * 5 + ("full_attention",),
        num_users=6,
        max_seq_len=512,
        cp=8,
        tp=4,
    )


def _mesh(chip_id=lambda r, c: r * 4 + c):
    return SimpleNamespace(
        shape=(8, 4),
        dram_grid_size=lambda: SimpleNamespace(x=8),
        get_fabric_node_id=lambda coordinates: ttnn.FabricNodeId(
            ttnn.MeshId(0), chip_id(coordinates[0], coordinates[1])
        ),
        arch=lambda: ttnn.device.Arch.BLACKHOLE,
    )


@pytest.fixture(autouse=True)
def _no_table_cache(monkeypatch):
    # An empty seed skips the on-disk table cache, so every build walks.
    monkeypatch.setattr(kv_chunk_table, "_TABLE_CACHE_SEED", "")
    monkeypatch.delenv(LAYOUT_MODEL_ENV, raising=False)


def _build(path, mesh=None):
    build_and_serialize_kv_chunk_table(
        path=path, mesh_device=mesh or _mesh(), kv_caches=_caches(), chunk_size=CHUNK_SIZE
    )
    return disaggregation.import_from_protobuf_file(path)


def test_the_model_derives_the_walked_table(tmp_path):
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path)
    model = disaggregation.import_cache_layout_model_from_protobuf_file(path + LAYOUT_MODEL_SUFFIX)
    derived = disaggregation.to_chunk_map(model)
    for node in range(32):
        fid = ttnn.FabricNodeId(ttnn.MeshId(0), node)
        derived.set_fabric_node_host(fid, host_name=walked.get_host(fid))
    assert disaggregation.first_table_difference(walked, derived) is None

    # 4 global heads on the one global layer, 16 sliding K and 16 sliding V heads on each of 5 sliding layers.
    assert len(model.caches) == 4 + 2 * 16 * 5
    assert {cache.name for cache in model.caches} == set(CONFIG_NAMES)
    sliding = [cache for cache in model.caches if "sliding" in cache.name]
    assert sorted({cache.head for cache in sliding}) == list(range(16))
    assert all(cache.head is None and cache.first_layer == 5 for cache in model.caches if "global" in cache.name)


def test_a_model_that_derives_another_table_is_not_written(tmp_path, monkeypatch):
    model_for = kv_chunk_table.gemma4_cache_layout_model
    # Declares a prefill chunk twice the writer's, so every CP row owns the wrong positions.
    monkeypatch.setattr(
        kv_chunk_table,
        "gemma4_cache_layout_model",
        lambda **kwargs: model_for(**{**kwargs, "chunk_size": 2 * kwargs["chunk_size"]}),
    )
    path = os.path.join(str(tmp_path), "table.pb")
    walked = _build(path)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)
    assert walked.num_configs() == len(CONFIG_NAMES)


def test_a_mesh_numbered_column_major_keeps_the_walk(tmp_path):
    path = os.path.join(str(tmp_path), "table.pb")
    _build(path, _mesh(chip_id=lambda r, c: c * 8 + r))
    assert os.path.exists(path)
    assert not os.path.exists(path + LAYOUT_MODEL_SUFFIX)


def test_publishing_the_derived_table_gives_the_walked_one(tmp_path, monkeypatch):
    walked = _build(os.path.join(str(tmp_path), "walked.pb"))
    monkeypatch.setenv(LAYOUT_MODEL_ENV, "1")
    path = os.path.join(str(tmp_path), "derived.pb")
    derived = _build(path)
    assert disaggregation.first_table_difference(walked, derived) is None
    assert os.path.exists(path + LAYOUT_MODEL_SUFFIX)

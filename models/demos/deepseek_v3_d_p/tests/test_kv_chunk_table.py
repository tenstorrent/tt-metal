# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

import re
from types import SimpleNamespace

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners import kv_chunk_table


def test_glm_merged_table_config_names_survive_protobuf_round_trip(tmp_path, monkeypatch):
    cache = SimpleNamespace(
        shape=(1, 1, 32, 32),
        dtype=ttnn.bfloat8_b,
        buffer_address=lambda: 0x10000,
    )
    index_cache = SimpleNamespace(
        shape=(1, 1, 32, 32),
        dtype=ttnn.bfloat8_b,
        buffer_address=lambda: 0x20000,
    )
    stage_layouts = [[{"rank": 0, "first_layer": 0, "count": 1, "base_addr": base}] for base in (0x10000, 0x20000)]
    monkeypatch.setattr(kv_chunk_table, "populate_kv_chunk_address_table_block_cyclic", lambda **kwargs: None)
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: 0)

    path = kv_chunk_table._build_and_serialize_merged_kv_chunk_table(
        mesh_device=None,
        caches=[("kvpe", cache), ("index", index_cache)],
        seq_len=32,
        num_layers=1,
        mesh_shape=(1, 1),
        sp_axis=0,
        num_users=1,
        path=str(tmp_path / "glm_merge_table.pb"),
        stage_layouts=stage_layouts,
    )
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(path)

    assert [table.config_name(i) for i in range(table.num_configs())] == ["00", "01"]


def test_kimi_k3_kda_config_names_match_decode(tmp_path, monkeypatch):
    """KV Manager pairs configs by name; decode publishes K3 as "00"/"01"/"02"."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
    from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry

    cache = SimpleNamespace(shape=(1, 1, 32, 32), dtype=ttnn.bfloat8_b, buffer_address=lambda: 0x10000)
    geometry = KdaContractGeometry.from_kda_config(kimi_k3_kda_config(), mesh_shape=(8, 4), sp_axis=0, tp_axis=1)
    kda_layout = [{"rank": 0, "first_layer": 0, "count": 1, "base_addr": 0x20000}]
    spec = kv_chunk_table.KdaTableSpec(
        geometry=geometry,
        recurrent=None,
        convolution=None,
        recurrent_layout=kda_layout,
        convolution_layout=kda_layout,
        layer_rows=[0],
    )
    monkeypatch.setattr(kv_chunk_table, "populate_kv_chunk_address_table_block_cyclic", lambda **kwargs: None)
    monkeypatch.setattr(kv_chunk_table, "populate_kv_chunk_address_table_kda", lambda **kwargs: None)
    # The cache double has no topology; this test is about config names, not the KV layout.
    monkeypatch.setattr(kv_chunk_table, "is_tp_deduped_kv_cache", lambda *_: False)
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: 0)

    path = kv_chunk_table._build_and_serialize_merged_kv_chunk_table(
        mesh_device=None,
        caches=[("kvpe", cache), ("kda_recurrent", spec), ("kda_convolution", spec)],
        seq_len=32,
        num_layers=2,
        mesh_shape=(8, 4),
        sp_axis=0,
        num_users=1,
        path=str(tmp_path / "k3_merge_table.pb"),
        stage_layouts=[[{"rank": 0, "first_layer": 0, "count": 1, "base_addr": 0x10000}]],
        kvpe_layer_rows=[1],
    )
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(path)

    assert [table.config_name(i) for i in range(table.num_configs())] == ["00", "01", "02"]
    assert table.config(0).num_layers == 2
    # The KDA configs reach the KV Manager as one Snapshot window each; kvpe stays a token cache.
    tensor_type = ttnn.experimental.disaggregation.KvChunkTensorType
    assert [table.config(i).tensor_type for i in range(3)] == [
        tensor_type.Kv,
        tensor_type.Snapshot,
        tensor_type.Snapshot,
    ]
    assert [table.config(i).max_sequence_length for i in (1, 2)] == [36_864, 36_864]
    assert [table.config(i).chunk_n_tokens for i in (1, 2)] == [96, 64]


def test_kimi_k3_kda_slab_must_match_the_geometry(tmp_path, monkeypatch, expect_error):
    """The walk strides each slab by the geometry's stored heads, so a slab sized otherwise would publish wrong
    addresses; the table build refuses it."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
    from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry

    cache = SimpleNamespace(shape=(1, 1, 32, 32), dtype=ttnn.bfloat8_b, buffer_address=lambda: 0x10000)
    kda_layout = [{"rank": 0, "first_layer": 0, "count": 1, "base_addr": 0x20000}]
    monkeypatch.setattr(kv_chunk_table, "populate_kv_chunk_address_table_block_cyclic", lambda **kwargs: None)
    monkeypatch.setattr(kv_chunk_table, "populate_kv_chunk_address_table_kda", lambda **kwargs: None)
    monkeypatch.setattr(kv_chunk_table, "is_tp_deduped_kv_cache", lambda *_: False)
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: 0)

    def build(stored_heads):
        geometry = KdaContractGeometry.from_kda_config(kimi_k3_kda_config(), mesh_shape=(8, 4), sp_axis=0, tp_axis=1)
        # Real ttnn.Shape, as a device slab reports it: it indexes but does not slice.
        slab = SimpleNamespace(shape=ttnn.Shape([1, stored_heads, 128, 128]), buffer_aligned_page_size=lambda: 4096)
        convolution = SimpleNamespace(
            shape=ttnn.Shape([1, 6 * stored_heads, 192]), buffer_aligned_page_size=lambda: 384
        )
        spec = kv_chunk_table.KdaTableSpec(
            geometry=geometry,
            recurrent=slab,
            convolution=convolution,
            recurrent_layout=kda_layout,
            convolution_layout=kda_layout,
            layer_rows=[0],
        )
        return kv_chunk_table._build_and_serialize_merged_kv_chunk_table(
            mesh_device=None,
            caches=[("kvpe", cache), ("kda_recurrent", spec), ("kda_convolution", spec)],
            seq_len=32,
            num_layers=2,
            mesh_shape=(8, 4),
            sp_axis=0,
            num_users=1,
            path=str(tmp_path / f"k3_{stored_heads}.pb"),
            stage_layouts=[[{"rank": 0, "first_layer": 0, "count": 1, "base_addr": 0x10000}]],
            kvpe_layer_rows=[1],
        )

    # The KDA configs decode agrees on (tt-blaze cache_migration.py), unchanged by storing each head on one chip.
    tensor_type = ttnn.experimental.disaggregation.KvChunkTensorType
    table = ttnn.experimental.disaggregation.import_from_protobuf_file(build(stored_heads=3))
    assert [table.config_name(i) for i in range(table.num_configs())] == ["00", "01", "02"]
    published = [
        (
            table.config(i).tensor_type,
            table.config(i).max_sequence_length,
            table.config(i).chunk_n_tokens,
            table.config(i).chunk_size_bytes,
        )
        for i in (1, 2)
    ]
    assert published == [(tensor_type.Snapshot, 36_864, 96, 16_384), (tensor_type.Snapshot, 36_864, 64, 384)]
    # A slab holding a TP column's full 24 heads per chip (the pre-sharding size) is refused.
    with expect_error(
        AssertionError, re.escape("slab holds (24, 128, 128) per layer but the geometry stores (3, 128, 128)")
    ):
        build(stored_heads=24)

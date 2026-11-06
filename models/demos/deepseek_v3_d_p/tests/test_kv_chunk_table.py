# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

from types import SimpleNamespace

import ttnn
from models.demos.deepseek_v3_d_p.tt.runners import kv_chunk_table
from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import assert_full_mesh_stripe_order


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


def _with_distribution_rank(rank):
    return SimpleNamespace(
        tensor_topology=lambda: SimpleNamespace(distribution_shape=lambda: [4] * rank),
    )


def test_full_mesh_gather_of_a_striped_cache_is_refused_with_sp_on_axis_1(expect_error):
    # Striped over every mesh coordinate (rank-2 topology): the row-major label is the stripe order only at
    # sp_axis=0, so a full-mesh gather at sp_axis=1 must raise instead of returning rows out of order.
    striped = _with_distribution_rank(2)
    assert_full_mesh_stripe_order(striped, 0, "test")
    with expect_error(RuntimeError, "needs sp_axis=0, got sp_axis=1"):
        assert_full_mesh_stripe_order(striped, 1, "test")
    # Sharded on one axis (rank-1 topology): no full-mesh ordering to get wrong.
    assert_full_mesh_stripe_order(_with_distribution_rank(1), 1, "test")

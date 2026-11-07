# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

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


# ---- Kimi-K3.1: DSA index cache next to the KDA slabs ------------------------------------------------
# Host-only: the real populate walks run off gathered stage layouts, so no device is needed.
_K31_MESH = (8, 4)
# 7 model layers: MLA (kvpe + DSA index) at 2 and 6, KDA elsewhere; split over two pipeline stages.
_K31_MLA_ROWS = [2, 6]
_K31_KDA_ROWS = [0, 1, 3, 4, 5]


def _k31_stage(rank, first_layer, count, base_addr):
    rows, cols = _K31_MESH
    return {
        "rank": rank,
        "first_layer": first_layer,
        "count": count,
        "base_addr": base_addr,
        "num_banks": 8,
        "host_tag": 0xABC0000 + rank,
        "fnids": [[ttnn.FabricNodeId(ttnn.MeshId(rank), r * cols + c) for c in range(cols)] for r in range(rows)],
    }


def _k31_table(tmp_path, monkeypatch, name, *, index, kda):
    """Build and round-trip a merged table over kvpe (+ the DSA index cache) (+ the KDA slabs)."""
    from models.demos.deepseek_v3_d_p.reference.kimi_k3_config import kimi_k3_kda_config
    from models.demos.deepseek_v3_d_p.tt.kda.state_adapter import KdaContractGeometry
    from models.demos.deepseek_v3_d_p.utils.kv_cache_utils import PREFILL_CHUNK_TOKENS

    monkeypatch.setattr(kv_chunk_table, "is_tp_deduped_kv_cache", lambda *_: False)
    monkeypatch.setattr(ttnn, "distributed_context_get_rank", lambda: 0)
    # Scaled-fp8 kvpe in decode's GLM format: one 704-byte row page per token, 22,528 B per 32 tokens.
    kvpe = SimpleNamespace(
        shape=(1, 1, 32, 704),
        dtype=ttnn.fp8_e4m3,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        buffer_aligned_page_size=lambda: 704,
        buffer_address=lambda: 0x1000_0000,
    )
    stage_layouts = [[_k31_stage(0, 0, 1, 0x1000_0000), _k31_stage(1, 1, 1, 0x1800_0000)]]
    index_kwargs = {}
    if index:
        index_kwargs = dict(
            index_kv_cache=SimpleNamespace(
                shape=(1, 1, 32, 128), dtype=ttnn.bfloat8_b, buffer_address=lambda: 0x2000_0000
            ),
            index_layer_ids=_K31_MLA_ROWS,  # every MLA layer owns its indexer
        )
        stage_layouts.append([_k31_stage(0, 0, 1, 0x2000_0000), _k31_stage(1, 1, 1, 0x2800_0000)])
    kda_spec = None
    if kda:
        kda_spec = kv_chunk_table.KdaTableSpec(
            geometry=KdaContractGeometry.from_kda_config(
                kimi_k3_kda_config(), mesh_shape=_K31_MESH, sp_axis=0, tp_axis=1
            ),
            recurrent=None,
            convolution=None,
            recurrent_layout=[_k31_stage(0, 0, 3, 0x3000_0000), _k31_stage(1, 3, 2, 0x3800_0000)],
            convolution_layout=[_k31_stage(0, 0, 3, 0x4000_0000), _k31_stage(1, 3, 2, 0x4800_0000)],
            layer_rows=_K31_KDA_ROWS,
        )
    path = kv_chunk_table.build_and_serialize_kv_chunk_table(
        mesh_device=None,
        kvpe_cache=SimpleNamespace(storage=kvpe),
        seq_len=PREFILL_CHUNK_TOKENS,
        num_layers=7,
        mesh_shape=_K31_MESH,
        sp_axis=0,
        num_users=2,
        chunk_size_global=PREFILL_CHUNK_TOKENS,
        path=str(tmp_path / f"{name}.pb"),
        stage_layouts=stage_layouts,
        layer_rows=_K31_MLA_ROWS,
        kda=kda_spec,
        **index_kwargs,
    )
    return ttnn.experimental.disaggregation.import_from_protobuf_file(path)


def _config_contents(table, config_id):
    """One config as plain data: its geometry and every (layer, slot) row resolved to addresses + replicas."""
    cfg = table.config(config_id)
    rows = {}
    for layer in range(cfg.num_layers):
        for slot in range(cfg.num_slots):
            rows[(layer, slot)] = [
                (
                    loc.noc_addr,
                    loc.size_bytes,
                    tuple(
                        (repr(fid), table.get_host(fid))
                        for fid in table.get_device_group(loc.device_group_index).fabric_node_ids
                    )
                    if loc.size_bytes
                    else (),
                )
                for loc in table.lookup_range(layer, 0, cfg.max_sequence_length, slot, config_id)
            ]
    geometry = (
        cfg.num_layers,
        cfg.max_sequence_length,
        cfg.num_slots,
        cfg.chunk_n_tokens,
        cfg.chunk_size_bytes,
        cfg.tensor_type,
    )
    return geometry, rows


def test_kimi_k31_index_and_kda_share_one_table(tmp_path, monkeypatch):
    """K3.1 publishes 00 kvpe / 01 DSA index / 02 KDA recurrent / 03 KDA convolution, and each config is
    exactly what the kvpe+index (GLM-style DSA) table and the kvpe+KDA (K3) table publish on their own."""
    k31 = _k31_table(tmp_path, monkeypatch, "k31", index=True, kda=True)
    dsa = _k31_table(tmp_path, monkeypatch, "dsa", index=True, kda=False)
    k3 = _k31_table(tmp_path, monkeypatch, "k3", index=False, kda=True)

    assert [k31.config_name(i) for i in range(k31.num_configs())] == ["00", "01", "02", "03"]
    tensor_type = ttnn.experimental.disaggregation.KvChunkTensorType
    assert [k31.config(i).tensor_type for i in range(4)] == [
        tensor_type.Kv,
        tensor_type.Kv,
        tensor_type.Snapshot,
        tensor_type.Snapshot,
    ]
    # Every config spans the model's layer axis, and the index rows sit at the MLA layers.
    assert [k31.config(i).num_layers for i in range(4)] == [7, 7, 7, 7]
    assert [k31.config(i).chunk_size_bytes for i in (0, 1)] == [22_528, 4 * 1088]
    index_geometry, index_rows = _config_contents(k31, 1)
    populated = sorted({layer for (layer, _), locs in index_rows.items() if any(loc[1] for loc in locs)})
    assert populated == _K31_MLA_ROWS

    for k31_id, (reference, ref_id) in enumerate([(dsa, 0), (dsa, 1), (k3, 1), (k3, 2)]):
        assert _config_contents(k31, k31_id) == _config_contents(reference, ref_id), f"config {k31_id:02d}"

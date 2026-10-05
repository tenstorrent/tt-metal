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

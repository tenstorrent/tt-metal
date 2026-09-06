# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.demos.gemma4.tt.runners import kv_caches


def test_external_cache_allocation_preserves_semantic_layer_order(monkeypatch):
    calls = []

    monkeypatch.setattr(
        kv_caches,
        "Gemma4AttentionConfig",
        lambda _hf, _idx: SimpleNamespace(num_key_value_heads=16, head_dim=256),
    )
    monkeypatch.setattr(
        kv_caches,
        "init_packed_ring_kv_cache",
        lambda *args, **kwargs: calls.append(("global", args, kwargs)) or "global-cache",
    )
    monkeypatch.setattr(
        kv_caches,
        "init_ring_kv_cache",
        lambda *args, **kwargs: calls.append(("sliding", args, kwargs)) or "sliding-cache",
    )
    hf = SimpleNamespace(
        num_hidden_layers=3,
        layer_types=["sliding_attention", "full_attention", "sliding_attention"],
    )
    mesh_config = SimpleNamespace(prefill=SimpleNamespace(sp=8), tp=4)

    result = kv_caches.allocate_ring_kv_caches(object(), hf, mesh_config, num_users=8, max_seq_len=262144)

    assert result.layers == ["sliding-cache", "global-cache", "sliding-cache"]
    assert result.global_layers == (1,)
    assert result.sliding_layers == (0, 2)
    assert [call[0] for call in calls] == ["sliding", "global", "sliding"]
    assert calls[0][1][2:] == (4, 256, 262144)
    assert calls[1][1][2:] == (1, 262144)
    assert all(call[2]["num_users"] == 8 for call in calls)


def test_chunk_locations_match_user_head_major_nd_shards():
    from models.demos.gemma4.tt.runners.kv_chunk_table import iter_cache_chunk_locations

    locations = list(
        iter_cache_chunk_locations(
            seq_len=1024,
            chunk_size=256,
            sp=2,
            num_users=2,
            heads_per_device=4,
            local_head=3,
            num_banks=8,
            chunk_size_bytes=8704,
        )
    )
    assert len(locations) == 64
    # CP rows name different global positions but point at the same local physical shard.
    assert locations[0][2:] == (0, 0, 52224)
    assert locations[32][2:] == (128, 0, 52224)
    # Slot 1 starts after all four heads of slot 0: 4 * 16 local blocks = 64 shards.
    assert locations[16][1:] == (1, 0, 0, 121856)


def test_migration_config_ids_are_decode_stream_order():
    from models.demos.gemma4.tt.runners.kv_chunk_table import CONFIG_NAMES

    assert len(CONFIG_NAMES) == 36
    assert CONFIG_NAMES[0] == "00_global_h0"
    assert CONFIG_NAMES[3] == "03_global_h3"
    assert CONFIG_NAMES[4] == "04_sliding_k_h0"
    assert CONFIG_NAMES[19] == "19_sliding_k_h15"
    assert CONFIG_NAMES[20] == "20_sliding_v_h0"
    assert CONFIG_NAMES[-1] == "35_sliding_v_h15"

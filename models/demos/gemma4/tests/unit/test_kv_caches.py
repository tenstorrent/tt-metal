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

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.experimental.diffusion_gemma.tt import hybrid_kv


class _Cache:
    def __init__(self, shape):
        self.shape = shape


def test_model_kwargs_enable_bounded_paged_cache():
    kwargs = hybrid_kv.model_owned_hybrid_kv_model_kwargs(max_seq_len=262144, max_batch_size=1)

    assert kwargs["create_kv_cache"] is True
    assert kwargs["bounded_sliding_kv_cache"] is True
    assert kwargs["paged_attention_config"].block_size == 64
    assert kwargs["paged_attention_config"].max_num_blocks == 4096


def test_attach_builds_sliding_tables_and_zero_copy_full_views(monkeypatch):
    calls = {}

    def fake_tables(num_layers, sliding_mask, **kwargs):
        calls["tables"] = (num_layers, list(sliding_mask), kwargs)
        return ["sliding-page-table", "full-page-table"]

    def fake_reshape(cache, shape):
        calls.setdefault("reshape", []).append((cache, shape))
        return SimpleNamespace(shape=shape)

    monkeypatch.setattr(hybrid_kv, "build_hybrid_page_tables", fake_tables)
    monkeypatch.setattr(hybrid_kv.ttnn, "reshape", fake_reshape)

    sliding = [_Cache([16, 2, 64, 256]), _Cache([16, 2, 64, 256])]
    full = [_Cache([64, 1, 64, 512]), _Cache([64, 1, 64, 512])]
    model = SimpleNamespace(
        hf_config=SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=1024,
        ),
        layers=[object(), object()],
        tt_kv_cache=[sliding, full],
    )

    page_tables = hybrid_kv.attach_model_owned_hybrid_kv(model, max_seq_len=4096)

    assert page_tables == ["sliding-page-table", "full-page-table"]
    assert calls["tables"] == (
        2,
        [True, False],
        {
            "num_users": 1,
            "block_size": 64,
            "max_seq_len": 4096,
            "sliding_window": 1024,
        },
    )
    assert [shape for _cache, shape in calls["reshape"]] == [
        (1, 1, 4096, 512),
        (1, 1, 4096, 512),
    ]
    assert model._dg_hybrid_sliding_layers == frozenset({0})
    assert model._dg_hybrid_logical_spans == (1024, 4096)
    assert model._dg_hybrid_page_tables_per_layer == page_tables

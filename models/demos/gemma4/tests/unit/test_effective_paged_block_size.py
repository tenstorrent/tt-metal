# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""``_effective_paged_block_size`` must return the smallest per-layer effective block size, so
page tables trimmed with it cover every kv-cache group (31B at TP=4: sliding 64, full 128)."""
from types import SimpleNamespace

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin as Generator


def _cache(shape):
    return [SimpleNamespace(shape=shape, padded_shape=shape), SimpleNamespace(shape=shape, padded_shape=shape)]


def _layer(head_dim, kv_heads, tp, kv_replicated=False):
    return SimpleNamespace(
        self_attn=SimpleNamespace(
            config=SimpleNamespace(head_dim=head_dim, num_key_value_heads=kv_heads),
            mesh_config=SimpleNamespace(tp=tp),
            weights=SimpleNamespace(kv_replicated=kv_replicated),
        )
    )


def test_tp4_31b_returns_sliding_block_not_full_view():
    # 31B on P150x4: sliding 16 heads -> 4/device at head_dim 256; full 4 heads -> 1/device at 512.
    shared = (10272, 4, 64, 256)
    layers = [_layer(256, 16, 4), _layer(256, 16, 4), _layer(512, 4, 4)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    kv = [_cache(shared) for _ in layers]
    assert Generator._effective_paged_block_size(fake, kv) == 64


def test_tp8_equal_views_unchanged():
    # Galaxy column: sliding 2/device at 256, full replicated 1/device at 512 -> both 64.
    shared = (23808, 2, 64, 256)
    layers = [_layer(256, 16, 8), _layer(512, 4, 8, kv_replicated=True)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    kv = [_cache(shared) for _ in layers]
    assert Generator._effective_paged_block_size(fake, kv) == 64


def test_non_hybrid_matching_caches_keep_declared_block():
    layers = [_layer(256, 16, 4), _layer(512, 4, 4)]
    fake = SimpleNamespace(model=[SimpleNamespace(layers=layers)])
    kv = [_cache((100, 4, 32, 256)), _cache((100, 1, 32, 512))]
    assert Generator._effective_paged_block_size(fake, kv) == 32

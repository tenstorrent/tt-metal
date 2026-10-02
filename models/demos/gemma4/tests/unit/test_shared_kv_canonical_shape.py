# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared-KV canonical allocation shapes (hybrid HMA tensor sharing).

The allocator must give a buffer shared by sliding (kv=2 x head_dim=256) and
full-attention (kv=1 x head_dim=512) layers the FULL-ATTENTION view: paged
ops reconcile any view via effective_block_size, but the chunked-prefill SDPA
validates k_shape[-1] == head_dim with no override — the old first-layer-wins
allocation TT_FATAL'd at the first chunked long prefill (hybrid ISL >= 8192
blocker)."""

import torch

from models.tt_transformers.tt.generator_vllm import _canonical_shared_kv_shapes


def test_widest_head_dim_wins_shared_buffer():
    # Gemma4-31B TP=8 per-device views: sliding first in layer order.
    specs = [
        ((1024, 2, 64, 256), torch.bfloat16, 0),  # sliding (layer 0)
        ((1024, 2, 64, 256), torch.bfloat16, 1),  # sliding (layer 1)
        ((1024, 1, 64, 512), torch.bfloat16, 0),  # full (layer 5) shares t0
    ]
    out = _canonical_shared_kv_shapes(specs)
    assert out[0] == (1024, 1, 64, 512)
    assert out[1] == (1024, 2, 64, 256)


def test_shrunk_sliding_blocks_do_not_undersize_shared_buffer():
    specs = [
        ((16, 2, 64, 256), torch.bfloat16, 0),  # bounded-shrunk sliding
        ((1024, 1, 64, 512), torch.bfloat16, 0),  # full needs the whole pool
    ]
    out = _canonical_shared_kv_shapes(specs)
    assert out[0] == (1024, 1, 64, 512)


def test_unshared_layers_keep_own_shapes():
    specs = [((512, 8, 64, 128), torch.bfloat16, i) for i in range(3)]
    out = _canonical_shared_kv_shapes(specs)
    assert all(out[i] == (512, 8, 64, 128) for i in range(3))


def test_inconsistent_per_block_bytes_raise(expect_error):
    specs = [
        ((1024, 2, 64, 256), torch.bfloat16, 0),
        ((1024, 2, 64, 512), torch.bfloat16, 0),  # 2x the per-block elements
    ]
    with expect_error(ValueError, "per-block element counts"):
        _canonical_shared_kv_shapes(specs)

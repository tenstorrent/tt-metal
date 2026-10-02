# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Shared-KV allocation shapes (hybrid HMA tensor sharing).

A buffer shared by sliding (kv=2 x head_dim=256) and full-attention
(kv=1 x head_dim=512) layers takes the FIRST layer's view — the sliding view
in layer order — which is the layout the paged kernels reinterpret correctly
for the fewer-heads layer. The buffer is sized by the larger num_blocks of
its views, and views with different per-block element counts raise."""

import pytest
import torch

# tt_transformers' generator_vllm imports vllm at module level; the unit tier has no vllm.
pytest.importorskip("vllm")

from models.tt_transformers.tt.generator_vllm import _canonical_shared_kv_shapes  # noqa: E402


def test_first_layer_view_wins_shared_buffer():
    specs = [
        ((1024, 2, 64, 256), torch.bfloat16, 0),  # sliding (layer 0)
        ((1024, 2, 64, 256), torch.bfloat16, 1),  # sliding (layer 1)
        ((1024, 1, 64, 512), torch.bfloat16, 0),  # full (layer 5) shares t0
    ]
    out = _canonical_shared_kv_shapes(specs)
    assert out[0] == (1024, 2, 64, 256)
    assert out[1] == (1024, 2, 64, 256)


def test_shrunk_sliding_blocks_do_not_undersize_shared_buffer():
    specs = [
        ((16, 2, 64, 256), torch.bfloat16, 0),  # bounded-shrunk sliding
        ((1024, 1, 64, 512), torch.bfloat16, 0),  # full needs the whole pool
    ]
    out = _canonical_shared_kv_shapes(specs)
    assert out[0] == (1024, 2, 64, 256)


def test_unshared_layers_keep_own_shapes():
    specs = [((512, 8, 64, 128), torch.bfloat16, i) for i in range(3)]
    out = _canonical_shared_kv_shapes(specs)
    assert all(out[i] == (512, 8, 64, 128) for i in range(3))


def test_inconsistent_per_block_bytes_raise(expect_error):
    specs = [
        ((1024, 2, 64, 256), torch.bfloat16, 0),
        ((1024, 1, 128, 512), torch.bfloat16, 0),  # 2x the per-block elements
    ]
    with expect_error(ValueError, "per-block element counts"):
        _canonical_shared_kv_shapes(specs)

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin


def _generator_with_bounded_window(window=1024):
    generator = object.__new__(ChunkedPrefillPageTableGuardMixin)
    config = SimpleNamespace(cache_position_modulo=window)
    layer = SimpleNamespace(self_attn=SimpleNamespace(config=config))
    generator.model = [SimpleNamespace(bounded_sliding_kv_cache=True, layers=[layer])]
    return generator


def test_activate_sequential_per_layer_row_refreshes_persistent_device_tables():
    """Sequential users must H2D-refresh B=1 persistent page tables.

    Host `_active` is sliced per user, but device buffers are keyed by batch=1
    and reused without content update unless ``update_persistent…`` runs.
    """
    generator = object.__new__(ChunkedPrefillPageTableGuardMixin)
    full = torch.tensor([[10, 11, 12], [20, 21, 22], [30, 31, 32]], dtype=torch.int32)
    sliding = torch.tensor([[110, 111], [120, 121], [130, 131]], dtype=torch.int32)
    updates = []

    def _update(sliced):
        updates.append([t.detach().clone() for t in sliced])

    model = SimpleNamespace(
        _active_page_tables_per_layer=[full, sliding],
        update_persistent_per_layer_page_tables=_update,
    )
    generator.model = [model]

    generator._activate_sequential_per_layer_row(full[1:2])

    assert model._active_page_tables_per_layer[0].shape == (1, 3)
    assert torch.equal(model._active_page_tables_per_layer[0], full[1:2])
    assert torch.equal(model._active_page_tables_per_layer[1], sliding[1:2])
    assert len(updates) == 1
    assert torch.equal(updates[0][0], full[1:2])
    assert torch.equal(updates[0][1], sliding[1:2])


def test_bounded_last_chunk_expansion_preserves_ring_origin():
    """100,793-token regression: expanded local rows must match absolute ring slots."""
    generator = _generator_with_bounded_window()

    start, last_idx = generator._adjust_last_prefill_chunk(
        last_chunk_start=100352,
        last_token_idx_in_chunk=440,
        last_token_idx_in_seq=100792,
        chunk_size=2048,
        block_size=64,
        model_id=0,
    )

    assert start == 99328
    assert start % 1024 == 0
    assert last_idx + 1 == 1465
    assert 1024 <= last_idx + 1 <= 2048


def test_bounded_last_chunk_no_expand_when_remnant_covers_window():
    generator = _generator_with_bounded_window()
    start, last_idx = generator._adjust_last_prefill_chunk(
        last_chunk_start=98304,
        last_token_idx_in_chunk=2047,
        last_token_idx_in_seq=100351,
        chunk_size=2048,
        block_size=64,
        model_id=0,
    )
    assert start == 98304
    assert last_idx == 2047


def test_unbounded_last_chunk_is_noop():
    generator = object.__new__(ChunkedPrefillPageTableGuardMixin)
    generator.model = [SimpleNamespace(bounded_sliding_kv_cache=False, layers=[])]
    start, last_idx = generator._adjust_last_prefill_chunk(
        last_chunk_start=100352,
        last_token_idx_in_chunk=440,
        last_token_idx_in_seq=100792,
        chunk_size=2048,
        block_size=64,
        model_id=0,
    )
    assert start == 100352
    assert last_idx == 440

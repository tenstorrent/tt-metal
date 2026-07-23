# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from models.demos.gemma4.tt.generator import ChunkedPrefillPageTableGuardMixin


def _generator_with_bounded_window(window=1024):
    generator = object.__new__(ChunkedPrefillPageTableGuardMixin)
    config = SimpleNamespace(cache_position_modulo=window)
    layer = SimpleNamespace(self_attn=SimpleNamespace(config=config))
    generator.model = [SimpleNamespace(bounded_sliding_kv_cache=True, layers=[layer])]
    return generator


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

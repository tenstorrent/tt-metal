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


def _two_chunk_prefill_path(monkeypatch, traced_chunks):
    """Which path ``prefill_forward_single_user_text`` takes for an unbounded two-chunk prompt."""
    monkeypatch.setenv("GEMMA4_CHUNKED_PREFILL_TRACE", "1")
    cls = type("Generator", (ChunkedPrefillPageTableGuardMixin,), {"_TRACED_PREFILL_CHUNKS": traced_chunks})
    generator = object.__new__(cls)
    generator.model = [
        SimpleNamespace(
            layers=[],
            bounded_sliding_kv_cache=False,
            process_logits_after_prefill_trace=lambda tt_out, last_token_idx: "traced",
        )
    ]
    generator.model_args = [SimpleNamespace(max_prefill_chunk_size=4096)]
    generator._activate_sequential_per_layer_row = lambda page_table: None
    generator._effective_paged_block_size = lambda kv_cache: 64
    generator._refresh_prefill_valid_seq_len = lambda **kwargs: None
    generator._chunk_prefill_page_table = lambda page_table, **kwargs: (page_table, 64)
    generator._easy_trace_prefill = lambda tokens, **kwargs: None
    generator._prefill_forward_single_user_text_eager = lambda tokens, **kwargs: "eager"
    return generator.prefill_forward_single_user_text(
        torch.ones((1, 8192), dtype=torch.int32),
        page_table=torch.arange(1, 129, dtype=torch.int32).unsqueeze(0),
        kv_cache=[object()],
        last_token_idx=8191,
    )


def test_two_chunk_prefill_replays_traced_chunks_by_default(monkeypatch):
    assert _two_chunk_prefill_path(monkeypatch, traced_chunks=True) == "traced"


def test_a_class_that_refuses_traced_chunks_prefills_eagerly(monkeypatch):
    """A traced chunk replay does not run python-side forward hooks, such as the dFlash tap capture."""
    assert _two_chunk_prefill_path(monkeypatch, traced_chunks=False) == "eager"


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

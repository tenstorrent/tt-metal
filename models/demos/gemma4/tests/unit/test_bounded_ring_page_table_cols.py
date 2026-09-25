# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Bounded sliding page tables follow the ring, not the bare window.

``paged_fill_cache`` requires ``cache_position_modulo <= block_size * columns``.
The modulo is ``bounded_ring_modulo(sliding_window)``: the window plus the
speculative headroom ``GEMMA4_SPEC_RING_HEADROOM_BLOCKS`` adds. The physical
pool is already sized from the ring; these tests pin the two page-table sizes
to it as well, so the headroom configuration can start (tt-metal#57573).
"""

from types import SimpleNamespace

import pytest
import torch

# The gemma4 vLLM generator imports vllm at module scope (through
# tt_transformers.generator_vllm), so COLLECTING this file fails outright on a
# runner without vLLM installed -- which is the tt-metal unit-test job. Skip
# before the import rather than inside the tests: the failure is at import.
pytest.importorskip("vllm")
from models.demos.gemma4.tt.attention import SPEC_RING_HEADROOM_ENV
from models.demos.gemma4.tt.generator_vllm import Gemma4ForCausalLM

WINDOW = 1024
BLOCK = 64


def _generator(max_batch=4):
    gen = Gemma4ForCausalLM.__new__(Gemma4ForCausalLM)
    gen._bounded_sliding_kv_cache = True
    gen.model_args = [SimpleNamespace(max_batch_size=max_batch)]
    gen.model = [SimpleNamespace(layers=[])]
    gen._text_config = lambda: SimpleNamespace(
        sliding_window=WINDOW, layer_types=["sliding_attention", "full_attention"]
    )
    gen._sliding_layer_indices = lambda: [0]
    gen._bounded_ring_slots = lambda pt, max_slots, authoritative: list(range(int(pt.shape[0])))
    return gen


class _Cache:
    """Stands in for a layer's K/V pair and for either tensor of it.

    The two sizing helpers index the cache differently (``kv_cache[0][0].shape``
    against ``kv_cache[0][layer][0].shape``), so one object answers both.
    """

    shape = (1, 1, BLOCK, 8)

    def __getitem__(self, index):
        return self


def _kv_cache():
    return [[_Cache(), _Cache()]]


def _tables(rows=2, cols=8):
    pt = torch.arange(1, rows * cols + 1, dtype=torch.int32).reshape(rows, cols)
    return [pt, pt]


def test_exact_window_ring_keeps_window_sized_tables(monkeypatch):
    monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
    gen = _generator()
    out = gen._pad_sliding_page_tables_for_bounded(_tables(), _kv_cache())
    assert tuple(out[0].shape) == (2, WINDOW // BLOCK)
    assert out[0][1].tolist() == list(range(16, 32))
    assert out[1] is _tables()[1] or torch.equal(out[1], _tables()[1])  # full-attention layer untouched
    assert gen._bounded_sliding_min_page_table_cols(_kv_cache()) == WINDOW // BLOCK


def test_headroom_widens_the_tables_to_the_ring(monkeypatch):
    """With ``GEMMA4_SPEC_RING_HEADROOM_BLOCKS`` the modulo is 2048; the page
    table must give every user ring/block_size blocks, contiguous per slot, so
    the pool (ring/block_size * max_batch) and the modulo agree."""
    monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(WINDOW // BLOCK))
    gen = _generator()
    ring = 2 * WINDOW
    cols = ring // BLOCK
    out = gen._pad_sliding_page_tables_for_bounded(_tables(), _kv_cache())
    assert tuple(out[0].shape) == (2, cols)
    assert out[0][0].tolist() == list(range(0, cols))
    assert out[0][1].tolist() == list(range(cols, 2 * cols))
    assert gen._bounded_sliding_min_page_table_cols(_kv_cache()) == cols
    assert gen._bounded_sliding_physical_blocks(BLOCK) == cols * 4


def test_pool_and_columns_agree_with_and_without_headroom(monkeypatch):
    for headroom in (None, WINDOW // BLOCK):
        if headroom is None:
            monkeypatch.delenv(SPEC_RING_HEADROOM_ENV, raising=False)
        else:
            monkeypatch.setenv(SPEC_RING_HEADROOM_ENV, str(headroom))
        gen = _generator(max_batch=32)
        cols = gen._bounded_sliding_min_page_table_cols(_kv_cache())
        assert gen._bounded_sliding_physical_blocks(BLOCK) == cols * 32
        assert gen._pad_sliding_page_tables_for_bounded(_tables(), _kv_cache())[0].shape[1] == cols

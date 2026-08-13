# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for multi-chunk page-table pad helpers (no torch.cat growth)."""

import torch

from models.demos.gemma4.tt.generator import (
    ChunkedPrefillPageTableGuardMixin,
    _align_page_table_blocks,
    ensure_page_table_width,
)


def test_align_page_table_blocks():
    assert _align_page_table_blocks(0) == 0
    assert _align_page_table_blocks(1) == 8
    assert _align_page_table_blocks(8) == 8
    assert _align_page_table_blocks(9) == 16


def test_ensure_page_table_width_noop_and_grow():
    t = torch.arange(10, dtype=torch.int32).view(1, 10)
    same = ensure_page_table_width(t, 10)
    assert same.shape == (1, 16)  # 10 → 16 aligned
    assert torch.equal(same[0, :10], t[0])
    assert torch.all(same[0, 10:] == -1)

    wide = ensure_page_table_width(same, 8)
    assert wide.shape == (1, 8)
    assert torch.equal(wide[0], same[0, :8])

    already = ensure_page_table_width(same, 16)
    assert already is same


def test_fill_chunk_page_table_reuses_scratch():
    class _G(ChunkedPrefillPageTableGuardMixin):
        pass

    g = _G()
    source = torch.arange(32, dtype=torch.int32).view(1, 32)
    a = g._fill_chunk_page_table(source, chunk_start_block=0, chunk_end_block=10, chunk_blocks=10)
    assert a.shape == (1, 16)
    assert torch.equal(a[0, :10], source[0, :10])
    assert torch.all(a[0, 10:] == -1)
    scratch_id = id(g._chunk_pt_scratch)
    b = g._fill_chunk_page_table(source, chunk_start_block=8, chunk_end_block=16, chunk_blocks=8)
    assert id(g._chunk_pt_scratch) == scratch_id
    assert torch.equal(b[0, :8], source[0, 8:16])
    assert torch.all(b[0, 8:] == -1)

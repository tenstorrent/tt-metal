# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of tt/kv_paged.py: geometry derived from config.json, page allocator (admit / grow / spec-decode rollback / release) and the
logical-entry -> physical-row translation that turns indexer output into sparse_sdpa row ids."""

import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tt.kv_paged import PageAllocator, PageLayout, V41Geometry


def test_geometry_from_config():
    g = V41Geometry.from_config()
    assert (
        g.kv_source(1) is None
        and g.kv_source(3) == 2
        and g.kv_source(19) == 14
        and g.kv_source(20) == 20
        and g.kv_source(39) == 20
    )
    assert g.index_source(23) == 20 and g.index_source(24) == 24 and g.index_source(39) == 36
    groups = g.selection_groups()
    assert len(groups) == 8 and sum(len(v) for v in groups.values()) == 38  # layers 2..39 read compressed latents
    assert groups[(20, 28)] == [28, 29, 30, 31]


def test_allocator_admit_grow_rollback_release():
    a = PageAllocator(num_pages=8, page_tokens=128)
    assert a.admit("u0", 300) == [0, 1, 2]  # 300 tokens -> 3 pages
    a.admit("u1", 128, reserve_tokens=5)  # a spec-decode step may write 5 more tokens: look-ahead page
    assert len(a.pages["u1"]) == 2 and a.free_pages() == 3
    a.grow("u0", 400)
    assert len(a.pages["u0"]) == 4
    with pytest.raises(MemoryError):  # allow-pytest.raises: CPU allocator unit test, no device fixture
        a.admit("u2", 128 * 4)
    assert "u2" not in a.pages and a.free_pages() == 2  # failed admission leaves nothing behind
    a.rollback(
        "u0", 250, keep_reserve_tokens=5
    )  # rejected drafts shrink the length; pages beyond the look-ahead are returned
    assert a.length["u0"] == 250 and len(a.pages["u0"]) == 2
    a.release("u1")
    assert a.free_pages() == 8 - 2
    t = a.page_table(["u0", "u1"], max_pages=4)
    assert (
        t.dtype == torch.int32
        and t.shape == (2, 4)
        and (t[1] == -1).all()
        and (t[0, :2] >= 0).all()
        and (t[0, 2:] == -1).all()
    )


def test_phys_rows_translation():
    lay = PageLayout(page_tokens=128)
    assert lay.rows_per_page == 64 * 3 + 128
    pt = torch.tensor([5, 2, 9, 0])  # logical page -> physical page
    # source 20 (ratio 1): entry j is token j -> page j // 128
    j = torch.tensor([0, 127, 128, 300])
    rows = lay.phys_rows(pt, 3, j)
    base = lay.row_offset(3)
    assert rows.tolist() == [5 * 320 + base + 0, 5 * 320 + base + 127, 2 * 320 + base + 0, 9 * 320 + base + 300 - 256]
    # source 8 (ratio 2): entry j covers tokens 2j, 2j+1 -> page (2j) // 128 = j // 64
    j = torch.tensor([0, 63, 64, 130])
    rows = lay.phys_rows(pt, 1, j)
    base = lay.row_offset(1)
    assert rows.tolist() == [5 * 320 + base, 5 * 320 + base + 63, 2 * 320 + base, 9 * 320 + base + 2]
    # distinct logical entries of one user never collide
    allj = torch.arange(0, 256)
    r = lay.phys_rows(pt, 3, allj)
    assert len(set(r.tolist())) == 256

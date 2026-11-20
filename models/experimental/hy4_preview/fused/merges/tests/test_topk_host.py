# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The top-k merge's arithmetic on the host (no device): the two-round protocol of kernels/topk_merge.hpp, written
in torch (topk_merge.simulate), against the sorted golden, on lists with many exact ties, short contexts and chips
with nothing visible; and the kernel's count_vector statement for statement against the vectorised counts."""

import pytest
import torch

from models.experimental.hy4_preview.fused.merges import topk_merge as tm
from models.experimental.hy4_preview.fused.merges.tests import cases


@pytest.mark.parametrize("chips", [2, 4, 8, 32])
@pytest.mark.parametrize("bits", [(8, 8), (6, 10), (11, 5), (6, 5, 5), (4, 4, 4, 4), (16,)])
def test_protocol_matches_golden(chips, bits):
    for seed in range(3):
        words = cases.make(chips, len(cases.KINDS), seed)
        gold, sim = tm.golden(words), tm.simulate(words, bits=bits)
        assert torch.equal(sim["keep"], gold["keep"]), (chips, bits, seed)
        keys = tm.keys_of(words)
        for u, kind in enumerate(cases.KINDS):
            seen = int((keys[:, u] >= tm.KEY_REAL_MIN).sum())
            assert int(gold["keep"][:, u].sum()) == min(tm.LIST, seen), (kind, seen)
            assert bool(sim["flags"][u] & tm.FLAG_ALL) == (seen < tm.LIST), kind
            assert int(sim["key"][u]) == int(gold["key"][u]), kind
            # the scores selected are the scores of torch.topk over all visible entries, as a multiset
            picked = torch.cat([keys[c, u, : int(gold["keep"][c, u])] for c in range(chips)])
            visible = keys[:, u][keys[:, u] >= tm.KEY_REAL_MIN]
            top = torch.topk(visible, min(tm.LIST, seen)).values
            assert torch.equal(torch.sort(picked, descending=True).values, top), kind


def test_smaller_k():
    words = cases.make(8, len(cases.KINDS), 5)
    for k in (1, 7, 512, 2047):
        assert torch.equal(tm.simulate(words, topk=k)["keep"], tm.golden(words, topk=k)["keep"]), k


@pytest.mark.parametrize("kind", cases.KINDS)
def test_count_vector(kind):
    """The kernel's walk over a list (galloping search, bucket jumps) gives the counts of both rounds."""
    words = cases.make(4, 1, 11, kinds=[kind])
    keys = tm.keys_of(words)[:, 0]
    for c in range(4):
        k = keys[c].tolist()
        seen = int((keys[c] >= tm.KEY_REAL_MIN).sum())
        v0 = tm.count_vector_reference(k, 0, 8, 256, 0, seen)
        assert v0 == tm.count_ge(keys[c], torch.arange(257) << 8).tolist()
        for bucket in sorted({x >> 8 for x in k if x >= tm.KEY_REAL_MIN} | {0, 15, 255}):
            v1 = tm.count_vector_reference(k, bucket << 8, 0, 256, v0[bucket + 1], v0[bucket])
            assert v1 == tm.count_ge(keys[c], (bucket << 8) + torch.arange(257)).tolist(), (kind, c, bucket)


def test_survivor_layout():
    """survivor_tiles puts the entry of rank q where the scan's root core does (indexer_scan raw_words/rank_word)."""
    from models.experimental.hy4_preview.fused.indexer_scan import indexer_scan as scan

    words = cases.make(1, 2, 3)[0]
    tiles = tm.survivor_tiles(words).to(torch.int64) & 0xFFFFFFFF
    raw = scan.raw_words(tiles).reshape(2, 2 * 32 * 32)
    assert torch.equal(raw[:, scan.rank_word(torch.arange(tm.LIST))], words)
    stamps = words & 0xFFFF
    rows = tm.stamp_row(stamps)
    assert int(rows.max()) < 32768
    back = scan.Scan.row_stamp(tm.SimpleNamespace(cfg=scan.ScanConfig()), rows)
    assert torch.equal(back, stamps)


def test_hub_search():
    """The hub's search from a hint (topk_hub.cpp last_reaching): every hint gives the bisection's answer, an
    exact hint in at most two totals."""
    g = torch.Generator().manual_seed(4)
    for n in (16, 32, 64, 256):
        for _ in range(40):
            steps = torch.sort(torch.randint(0, 5000, (n,), generator=g), descending=True).values.tolist()
            steps[0] = max(steps[0], 2048)
            total = lambda i: steps[i] if i < n else 0
            want = 2048
            answer = max(i for i in range(n) if steps[i] >= want)
            worst = 0
            for hint in range(n + 3):
                got, calls = tm.last_reaching_reference(total, n, want, hint)
                assert got == answer, (n, hint, got, answer)
                worst = max(worst, calls)
            assert tm.last_reaching_reference(total, n, want, answer)[1] <= 2
            assert worst <= 2 * n.bit_length() + 2

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host arithmetic of the log-sum-exp merge (no device): the page and tile layouts round-trip, the merge golden is
sparse_mla's contract (layout.merge) and equals the reference softmax over the union of the chips' rows, and a
chain of pairwise merges in the device's order (column, row, sink) gives the same result."""

import torch

from models.experimental.hy4_preview.fused.merges import lse_merge as lm
from models.experimental.hy4_preview.fused.sparse_mla import layout as lo


def test_layouts_round_trip():
    g = torch.Generator().manual_seed(0)
    x = torch.randn(3, 8, 512, generator=g)
    assert torch.equal(lm.unpack_tiny(lm.pack_tiny(x)), x)
    assert torch.equal(lm.pack_tiny(x[0]), lo.pack_tiny(x[0]))
    raw = torch.randn(2, 3 * 1024, generator=g)
    assert torch.equal(lm.tiles_to_raw(lm.raw_to_tiles(raw)), raw)
    o, m, s = torch.randn(8, 64, 512, generator=g), torch.randn(8, 64, generator=g), torch.rand(8, 64, generator=g)
    pages = lm.partial_pages(o, m, s)
    assert pages.shape == (64, lm.PAGE_VALUES)
    u, grp = 5, 3
    page = pages[u * 8 + grp]
    assert torch.equal(lo.unpack_tiny(page[: 16 * 256], 8), o[u, 8 * grp : 8 * grp + 8])
    ms = lo.unpack_tiny(page[16 * 256 : 17 * 256], 8)
    assert torch.equal(ms[:, 0], m[u, 8 * grp : 8 * grp + 8]) and torch.equal(ms[:, 1], s[u, 8 * grp : 8 * grp + 8])


def test_golden_is_the_reference_softmax():
    """Partials of 6 chips made from rows (layout.partial), one chip without rows: the merge golden equals
    layout.merge and the reference's softmax over all rows and the sink."""
    cfg = lo.SparseMlaConfig(rows_per_slot=4096)
    g = torch.Generator().manual_seed(1)
    q = torch.randn(64, cfg.width, generator=g)
    sink = torch.randn(64, generator=g)
    rows = [torch.randn(n, cfg.width, generator=g) for n in (40, 0, 7, 130, 1, 64)]
    parts = [lo.partial(q, r, cfg) for r in rows]
    o, m, s = (torch.stack([p[i] for p in parts]) for i in range(3))
    ours = lm.merge_golden(o, m, s, sink, cfg.scale)
    assert torch.allclose(ours, lo.merge(parts, sink, cfg), rtol=1e-12, atol=1e-12)
    assert torch.allclose(ours, lo.reference_ctx(q, torch.cat(rows), sink, cfg), rtol=1e-9, atol=1e-9)


def test_pairwise_chain_equals_the_merge():
    """The device merges pairwise (sdpa_tail): column peers in slot order, then row peers, then the sink with the
    division. In float64 that chain is the merge."""
    g = torch.Generator().manual_seed(2)
    scale = 1 / 16
    o = torch.randn(32, 8, 512, generator=g).double()
    m = (torch.randn(32, 8, generator=g) * 30).double()
    s = (1 + 50 * torch.rand(32, 8, generator=g)).double()
    m[5], s[5], o[5] = lm.EMPTY_MAX, 0, 0
    sink = torch.randn(8, generator=g).double()

    def pair(a, b):
        top = torch.maximum(a[1], b[1])
        pa, pb = torch.exp(scale * (a[1] - top)), torch.exp(scale * (b[1] - top))
        return a[0] * pa[:, None] + b[0] * pb[:, None], top, a[2] * pa + b[2] * pb

    columns = []
    for c in range(4):
        acc = (o[c], m[c], s[c])
        for r in range(1, 8):
            acc = pair(acc, (o[4 * r + c], m[4 * r + c], s[4 * r + c]))
        columns.append(acc)
    acc = columns[0]
    for c in range(1, 4):
        acc = pair(acc, columns[c])
    acc = pair(acc, (torch.zeros(8, 512, dtype=torch.float64), sink / scale, torch.ones(8, dtype=torch.float64)))
    assert torch.allclose(acc[0] / acc[2][:, None], lm.merge_golden(o, m, s, sink, scale), rtol=1e-9, atol=1e-12)

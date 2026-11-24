# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""DSparkDrafter with its local layers' context as rings (tt/mtp.RING) against the full-length caches: the same
rows in every draft's window after commit_chunk, commit and commit_rows, on the first lap and after the ring has
wrapped, and the same draft tokens. Needs the weight cache + ``<cache>/mtp``.

    pytest -q models/demos/inkling2_mini/tests/test_mtp_ring.py
"""

import pytest
import torch

import ttnn
from models.demos.inkling2_mini.tt.model import HIDDEN
from models.demos.inkling2_mini.tt.mtp import BLOCK, LOCAL, N_LAYERS, RING, WINDOW, DSparkDrafter
from models.demos.inkling2_mini.tt.spec_generator import TAP_LP, TappedModel

CTX = 4096


@pytest.fixture(scope="module")
def drafters():
    dev = ttnn.open_device(device_id=0)
    model = TappedModel(dev, max_ctx=CTX, log=lambda s: None)
    ring, full = (DSparkDrafter(model, k=2, ring=r) for r in (True, False))
    assert ring.ring == [RING, 0, RING] and not any(full.ring) and ring.rows[0] < full.rows[0]
    yield ring, full
    ttnn.close_device(dev)


def test_ring_equals_full(drafters):
    ring, full = drafters
    dev, ntap = ring.device, len(TAP_LP) + 1
    put = lambda t, **kw: ttnn.from_torch(t, device=dev, **kw)  # noqa: E731

    def window(d, p):
        """Rows p-1024 .. p-1 of each local layer's K and V as the draft at p reads them (it writes p .. p+7)."""
        q = p + BLOCK - 1
        out = []
        for j, c in enumerate(d.k_cache + d.v_cache):
            i = j % N_LAYERS
            if i not in LOCAL:
                continue
            c = ttnn.to_torch(c).float()[0]
            top = (q if q < RING else q % RING + RING) if d.ring[i] else q
            out.append(c[:, max(0, top - WINDOW - BLOCK + 1) : top - BLOCK + 1].clone())
        return out

    def check(p, what):
        a, b = window(ring, p), window(full, p)
        for j, (x, y) in enumerate(zip(a, b)):
            assert x.shape == y.shape and torch.equal(x, y), (what, p, j)
        toks = []
        for d in (ring, full):
            d.set_draft_pos(p)
            ttnn.copy_host_to_device_tensor(
                ttnn.from_torch(torch.tensor([[1000 + p]], dtype=torch.int32), dtype=ttnn.uint32), d.anchor
            )
            toks.append(ttnn.to_torch(d.draft(d.anchor)).reshape(-1).tolist())
        assert toks[0] == toks[1], (what, p, toks)

    torch.manual_seed(7)
    # the chunked prefill's feed, then drafts on the first lap and after a wrap
    for s, t in ((0, 1024), (1024, 1024), (2048, 1024), (3072, 352)):
        taps = torch.randn(ntap, t, HIDDEN)
        for d in (ring, full):
            d.commit_chunk([put(x.view(1, 1, t, HIDDEN), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT) for x in taps],
                           s, None)  # fmt: skip
        check(s + t, "chunk")  # (a draft writes rows p .. p+7: only at the end of what is committed)
    p = 3424
    assert p < 3 * RING < p + 36
    for _ in range(36):  # single positions (decode steps), across a ring boundary
        f = torch.zeros(1, 1, 32, HIDDEN)
        f[0, 0, :ntap] = torch.randn(ntap, HIDDEN)
        for d in (ring, full):
            d.set_commit_pos(p)
            d.commit(
                put(f, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                put(torch.tensor([p], dtype=torch.int32), dtype=ttnn.int32),
            )
        p += 1
        if p in (3 * RING - 3, 3 * RING, 3 * RING + 4):
            check(p, "commit")
    for R in (2, 3):  # verify passes
        p0 = p
        rows = torch.randn(1, 1, ntap * R, HIDDEN)
        for d in (ring, full):
            d.commit_rows(
                put(rows, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
                put(torch.tensor([p0], dtype=torch.int32), dtype=ttnn.int32),
                put(torch.arange(p0, p0 + R, dtype=torch.int32).view(1, R), dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT),  # fmt: skip
            )
        p = p0 + R
        check(p, "rows")

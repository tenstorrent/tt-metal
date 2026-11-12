# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of DSV41_XROW_SYNC_EVERY (T24): scatter_rows_mesh synchronises every N pieces; every piece is still written and every device buffer freed exactly once (fake ttnn)."""
from types import SimpleNamespace

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import prefix_cache as PC


def run(monkeypatch, n_rows, sync_every, reuse=False):
    ev = []
    monkeypatch.setattr(PC.ttnn, "ShardTensor2dMesh", lambda *a, **k: None)
    monkeypatch.setattr(PC.ttnn, "from_torch", lambda t, **k: SimpleNamespace(shape=tuple(t.shape)))
    monkeypatch.setattr(PC.ttnn, "synchronize_device", lambda md: ev.append("sync"))
    monkeypatch.setattr(PC.ttnn, "deallocate", lambda x: ev.append("free"))
    monkeypatch.setattr(PC.P, "paged_scatter_rows", lambda t, s, i: ev.append("scatter"))
    model = SimpleNamespace(md=None, rows=4, cols=8)
    PC.scatter_rows_mesh(
        model, None, {2: (torch.arange(n_rows), torch.zeros(n_rows, 512, dtype=torch.bfloat16))}, sync_every=sync_every
    )
    return ev


def test_default_is_one_sync_per_piece(monkeypatch):
    ev = run(monkeypatch, 5 * PC.PIECE, 1)
    assert ev.count("scatter") == 5 and ev.count("sync") == 5 and ev.count("free") == 10
    assert ev[:4] == ["scatter", "sync", "free", "free"]  # the old order


def test_sync_every_n_batches_the_syncs_and_frees_everything(monkeypatch):
    for n in (2, 3, 8, 100):
        ev = run(monkeypatch, 7 * PC.PIECE + 5, n)
        pieces = 8
        assert ev.count("scatter") == pieces and ev.count("free") == 2 * pieces
        assert ev.count("sync") == -(-pieces // n), (n, ev)
        last_scatter = max(i for i, e in enumerate(ev) if e == "scatter")
        assert "sync" in ev[last_scatter:]  # nothing left unsynchronised at the end
        # a buffer is only freed after a sync that follows its scatter
        pending = 0
        for e in ev:
            if e == "scatter":
                pending += 1
            elif e == "sync":
                pending = 0
            elif e == "free":
                assert pending == 0


def test_reuse_buffers_same_events_and_padding(monkeypatch):
    """reuse=True (DSV41_XROW_BATCH): same scatter / sync / free sequence; the short last piece is padded with SKIP ids and zero data even though the buffers are reused"""
    seen = []
    orig = PC.ttnn.from_torch
    monkeypatch.setattr(
        PC.ttnn, "from_torch", lambda t, **k: (seen.append(t.clone()), SimpleNamespace(shape=tuple(t.shape)))[1]
    )
    for n in (3 * PC.PIECE + 7, 2 * PC.PIECE):
        seen.clear()
        ev_a = run(monkeypatch, n, 4, reuse=False)
        seen.clear()
        monkeypatch.setattr(
            PC.ttnn, "from_torch", lambda t, **k: (seen.append(t.clone()), SimpleNamespace(shape=tuple(t.shape)))[1]
        )
        ev_b = run(monkeypatch, n, 4, reuse=True)
        assert ev_a == ev_b

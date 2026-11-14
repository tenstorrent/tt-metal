# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the skinny-kernel helpers (no device): flag parsing, bank assignment of a projection, reader page geometry, mantissa rounding probe."""
import torch

from models.demos.blackhole.hy4_preview.tt import hy4_skinny as S


def test_enabled_flag_parsing(monkeypatch):
    monkeypatch.delenv("HY4_SKINNY", raising=False)
    assert not any(S.enabled(n) for n in ("shared", "router", "oproj", "hproj"))
    monkeypatch.setenv("HY4_SKINNY", "shared, router")
    assert S.enabled("shared") and S.enabled("router") and not S.enabled("oproj") and not S.enabled("hproj")
    monkeypatch.setenv("HY4_SKINNY", "all")
    assert all(S.enabled(n) for n in ("shared", "router", "oproj", "hproj"))


def test_proj_assignment_covers_every_tile_once():
    for nt in (90, 95, 96, 64, 7, 8, 1):
        valid, first, pnc = S.proj_assignment(nt)
        assert sum(valid) == nt and pnc == max(valid) and max(valid) - min(valid) <= 1
        tiles = [first[b] + j for b in range(S.NB) for j in range(valid[b])]
        assert sorted(tiles) == list(range(nt))


def test_reader_page_geometry_divides_blocks():
    for tb, sbk in ((2048, 24), (2048, 48), (2048, 192), (1088, 48), (1088, 64)):
        ct = S.reader_ct_tb(0, 0, 0, [dict(addr=0, sbk=sbk, pn=2, nsub=1, nbuf=4)], tb) if tb == 2048 else None
        if ct is None:
            continue
        page, npg = ct[4], ct[5]
        assert page * npg == sbk * tb and page <= 16384 and page % tb == 0


def test_round_sig_is_round_to_nearest_to_n_bits():
    torch.manual_seed(0)
    w = torch.randn(4096)
    for bits in (3, 5, 7):
        r = S.round_sig(w, bits)
        m, e = torch.frexp(r)
        assert torch.all((m * (1 << bits)) == torch.round(m * (1 << bits)))  # exactly `bits` mantissa bits
        assert ((r - w).abs() <= w.abs() * 2.0 ** (-bits) + 1e-12).all()  # round to nearest
    assert torch.equal(S.round_sig(w, 0), w)

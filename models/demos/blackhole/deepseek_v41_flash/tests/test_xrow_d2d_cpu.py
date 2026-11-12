# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU test of the host-side planning of DSV41_XROW_D2D (tt/xrow_d2d.py): key slot pairs and the concatenated latent rows."""
from types import SimpleNamespace as NS

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import paged_ops as PO
from models.demos.blackhole.deepseek_v41_flash.tt import xrow_d2d as XD


def test_key_pairs_skip_blocks_without_key_slot():
    src = [(0, [], 10 + i, -1, -1, 0) for i in range(5)]
    cps = [(i, NS(key_slot=-1 if i == 2 else 100 + i)) for i in range(5)]
    assert XD.key_pairs(src, cps) == [(10, 100), (11, 101), (13, 103), (14, 104)]


def test_key_pairs_missing_donor_slot_raises():
    try:
        XD.key_pairs([(0, [], -1, -1, -1, 0)], [(0, NS(key_slot=3))])
    except RuntimeError:
        return
    raise AssertionError("no error")


def test_latent_rows_same_length_and_source_order():
    pt = torch.arange(4)
    ps_fn = lambda pt_, k, r: r + 1000 * (k + 1)  # donor row ids encode the source index
    pd_fn = lambda s_, r: r + 100000 * PO.SOURCES.index(s_)
    hit = 2048
    ps, pd = XD.latent_rows(pt, hit, set(PO.SOURCES), ps_fn, pd_fn)
    n = sum(hit // PO.SRC_RATIO[s] for s in PO.SOURCES)
    assert ps.numel() == pd.numel() == n == 2048 // 2 * 3 + 2048
    assert int(ps[0]) == 1000 and int(pd[0]) == 0 and int(pd[-1]) == 100000 * 3 + 2047
    ps2, pd2 = XD.latent_rows(pt, hit, {2, 20}, ps_fn, pd_fn)  # fewer layers: only their sources
    assert ps2.numel() == 1024 + 2048


def test_flags_default_off(monkeypatch):
    monkeypatch.delenv("DSV41_XROW_D2D", raising=False)
    assert not XD.enabled() and not XD.keys_enabled()
    monkeypatch.setenv("DSV41_XROW_D2D", "1")
    assert XD.enabled() and XD.keys_enabled()
    monkeypatch.setenv("DSV41_XROW_D2D_KEYS", "0")
    assert XD.enabled() and not XD.keys_enabled()

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only checks of the flat routed expert's Python side (no device): the weight bank layout places every weight
tile exactly once where the kernels read it, and FlatLayoutOptions keys / compares as the plan cache expects."""

import importlib.util
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[6]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


weights_py = _load(
    "flat_routed_expert_weights",
    REPO / "ttnn/cpp/ttnn/operations/experimental/moe_block/flat_routed_expert/flat_routed_expert_weights.py",
)


def _tagged(rows, cols, base):
    """[rows, cols] whose 32 x 32 tile (i, j) is filled with base + i * (cols / 32) + j (a unique tag per tile)."""
    tr, tc = rows // 32, cols // 32
    tags = base + torch.arange(tr * tc, dtype=torch.float64).reshape(tr, tc)
    return tags.repeat_interleave(32, 0).repeat_interleave(32, 1)


def _layout(E, H, I, banks, nd, n_rd, np_, nu):
    """A plan's layout facts for a small synthetic split (the fields flat_weight_regions reads)."""
    it, ht = I // 32, H // 32
    ngrp = 2 * it // nu if nu else it // np_
    pcds = [ht // nd + (d < ht % nd) for d in range(nd)]
    col0s = [sum(pcds[:d]) for d in range(nd)]
    sub = max(-(-E * (it // weights_py.kd_of(p, it)) // banks) * weights_py.kd_of(p, it) * p for p in pcds)
    return dict(
        n_rd=n_rd,
        n_rd_sg=n_rd,
        rg=ngrp // n_rd,
        np=np_,
        nu=nu,
        nk_gu=ht // weights_py.KBLK,
        nd=nd,
        pcds=pcds,
        col0s=col0s,
        banks=banks,
        wd_rot_tiles=sub,
    )


@pytest.mark.parametrize("units", [False, True], ids=["pairs", "units"])
def test_flat_weight_regions_places_every_tile(units):
    E, H, I, banks = 2, 256, 1024 if not units else 1536, 8
    lay = _layout(E, H, I, banks, nd=4, n_rd=16, np_=0 if units else 1, nu=3 if units else 0)
    W = [
        (_tagged(H, I, 1e6 * (3 * e + 1)), _tagged(H, I, 1e6 * (3 * e + 2)), _tagged(I, H, 1e6 * (3 * e + 3)))
        for e in range(E)
    ]
    regions, dregs = weights_py.flat_weight_regions(W, lay)
    assert len(regions) == lay["n_rd"] and len({r.shape for r in regions}) == 1
    assert len(dregs) == lay["nd"] * banks and len({r.shape for r in dregs}) == 1
    if not units:  # whole tiles: every gate / up tile once, every tile uniform
        got = torch.cat(regions)
        assert bool((got == got[:, :1, :1]).all())
        want = torch.cat([torch.cat([w[0], w[1]], 1)[::32, ::32].flatten() for w in W])
        assert sorted(got[:, 0, 0].tolist()) == sorted(want.tolist())
    else:  # 16-column units: every gate / up column half once (faces 16 wide)
        got = torch.cat(regions)
        halves = torch.cat([got[:, 0, :16][:, :1], got[:, 0, 16:][:, :1]]).flatten()
        want = torch.cat([torch.cat([w[0][::32, ::16], w[1][::32, ::16]], 1).flatten() for w in W])
        assert sorted(halves.tolist()) == sorted(want.tolist())
    # down: every Wd tile once over the banks (the rest is zero padding), block k of core d in bank (d + k) % banks
    dn = torch.cat(dregs)[:, 0, 0]
    nz = dn[dn != 0]
    want = torch.cat([w[2][::32, ::32].flatten() for w in W])
    assert sorted(nz.tolist()) == sorted(want.tolist())


def test_flat_layout_options():
    ttnn = pytest.importorskip("ttnn")
    mb = ttnn._ttnn.operations.experimental.moe_block
    a = mb.FlatLayoutOptions(rows_hint=40)
    assert (a.x_helpers, a.pin_ratio, a.rows_hint, a.x_ready) == (-1, 8, 40, -1)
    assert a == weights_py.flat_layout(dict(rows_hint=40))
    assert a != weights_py.flat_layout(a, x_ready=2)
    # (x_helpers, pin_ratio, rows_hint, x_ready, gu_source tensix, gu_read_depth, gu_read_pieces, no gu_place words)
    assert list(a.key()) == [2**32 - 1, 8, 40, 2**32 - 1, 0, 2**32 - 1, 2**32 - 1, 0]

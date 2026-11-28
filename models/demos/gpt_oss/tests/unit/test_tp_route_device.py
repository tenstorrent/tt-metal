# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""experts/tp_route.py (the fused routing of the tensor-parallel routed prefill) against a torch reference: counts,
regions, the dispatch row of every (token, slot) pair and the full row -> token table (zeros on every row no pair
writes), for random top-k picks (k distinct experts per token) and a skewed one, at several chunk lengths.

    pytest models/demos/gpt_oss/tests/unit/test_tp_route_device.py
"""

import os

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.experts.tp_route import TpRoute

E, K, TILE = 32, 4, 32

pytestmark = pytest.mark.skipif(os.environ.get("CI") == "true", reason="Blackhole only; run locally.")


def reference(ids, spare):
    """ids [C, K] -> counts [E], region [E], pos [C, K], tok [spare + 32]."""
    C = ids.shape[0]
    counts = torch.bincount(ids.reshape(-1), minlength=E)
    padded = (counts + TILE - 1) // TILE * TILE
    region = torch.cumsum(padded, 0) - padded
    seen = torch.zeros(E, dtype=torch.int64)
    pos = torch.zeros(C, K, dtype=torch.int64)
    tok = torch.zeros(spare + TILE, dtype=torch.int64)
    for t in range(C):
        for j in range(K):
            e = int(ids[t, j])
            r = int(region[e] + seen[e])
            seen[e] += 1
            pos[t, j] = r
            tok[r] = t
    return counts, region, pos, tok


def picks(C, gen, skew):
    if skew:  # most tokens on a few experts
        w = torch.full((E,), 0.02)
        w[:3] = 10.0
    else:
        w = torch.ones(E)
    return torch.stack([torch.multinomial(w, K, replacement=False, generator=gen) for _ in range(C)])


@pytest.mark.parametrize("C", [64, 128, 1024, 4096, 16384])
@pytest.mark.parametrize("skew", [False, True])
def test_tp_route(device, C, skew):
    gen = torch.Generator().manual_seed(C + skew)
    ids = picks(C, gen, skew)
    spare = K * C + TILE * (E - 1)
    tt_ids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint16, layout=ttnn.TILE_LAYOUT, device=device)
    route = TpRoute(device, E, K)
    for _ in range(2):  # (the second call runs the cached program)
        counts, region, pos, tok = route(tt_ids, spare)
        got = [ttnn.to_torch(t).to(torch.int64).reshape(-1) for t in (counts, region, pos, tok)]
        for t in (counts, region, pos, tok):
            ttnn.deallocate(t)
        want = reference(ids, spare)
        for name, g, w in zip(("counts", "region", "pos", "tok"), got, want):
            assert torch.equal(g, w.reshape(-1)), f"{name} differs at {(g != w.reshape(-1)).nonzero()[:5].tolist()}"

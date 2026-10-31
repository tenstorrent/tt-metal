# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Inputs and the expected result of the sparse 2D dispatch, in torch only (no ttnn): shared by the device test (this
tree) and dump_ref_sparse_dispatch_2d.py (the reference tree's environment).

Per chip (r, c), idx = r * C + c, with S own rows of width W:
  xbuf_in  [1, 1, N * S, W] bf16: random everywhere (stale contents), the own rows at slots idx * S + t
  route    [1, 1, 1, RW * S] uint32 bits (as int32): a route word per own row - the chips holding one of the row's top_k
           experts (uniform routing: top_k distinct experts uniformly at random), the own chip excluded; the order by
           dispatch_prep's rule (the dimension-order tree with fewer links, ties by bit 31 of t * 2654435761) and the
           link by bit 29 of that hash, as the MoE block's head writes them
  xbuf     the dispatch's result: slot home * S + t holds home's row t on every chip its route word names (and on home),
           every other slot keeps xbuf_in's contents
"""

from __future__ import annotations

import functools

import torch

HASH = 2654435761


def _i32(v: int) -> int:
    return v - (1 << 32) if v >= (1 << 31) else v


def _links(ab: int) -> int:
    return (ab & 15) + (ab >> 4)


def route_words(R, C, S, num_experts, top_k, num_links, wrap_x, wrap_y, span_entry, gen):
    """{(r, c): route [1, 1, 1, RW * S] int32}; span_entry(n, wrap, pos, mask, tie) as mesh_routing's."""
    N = R * C
    assert num_experts % N == 0
    epc = num_experts // N
    RW = 1 if N <= 16 else 2
    span = functools.lru_cache(maxsize=None)(span_entry)

    def xspan(pos, m, t):
        return span(C, wrap_x, pos, m & ~(1 << pos) & ((1 << C) - 1), (t * HASH & 0xFFFFFFFF) >> 30 & 1)

    def yspan(pos, m, t):
        return span(R, wrap_y, pos, m & ~(1 << pos) & ((1 << R) - 1), (t * HASH & 0xFFFFFFFF) >> 30 & 1)

    out = {}
    for r in range(R):
        for c in range(C):
            me = r * C + c
            experts = torch.rand(S, num_experts, generator=gen).argsort(dim=1)[:, :top_k]
            words = []
            for t in range(S):
                m = 0
                for e in experts[t].tolist():
                    m |= 1 << (e // epc)
                m &= ~(1 << me)
                if m == 0:
                    words += [0] * RW
                    continue
                cm = rm = 0
                for rr in range(R):
                    row = (m >> (rr * C)) & ((1 << C) - 1)
                    cm |= row
                    rm |= (row != 0) << rr
                hx = _links(xspan(c, cm, t))
                hy = _links(yspan(r, rm, t))
                for cc in range(C):
                    if ((cm | (1 << c)) >> cc) & 1:
                        col = sum(((m >> (rr * C + cc)) & 1) << rr for rr in range(R))
                        hx += _links(yspan(r, col, t))
                for rr in range(R):
                    if ((rm | (1 << r)) >> rr) & 1:
                        hy += _links(xspan(c, (m >> (rr * C)) & ((1 << C) - 1), t))
                hsh = t * HASH & 0xFFFFFFFF
                order = 0 if hx < hy else (1 if hy < hx else (hsh >> 31) & 1)
                link = (hsh >> 29) & 1 if num_links > 1 else 0
                words += [m | order << 16 | link << 17] if RW == 1 else [m, order | link << 1]
            out[(r, c)] = torch.tensor([_i32(w) for w in words], dtype=torch.int32).reshape(1, 1, 1, RW * S)
    return out


def xbuf_inputs(R, C, S, W, gen):
    """{(r, c): xbuf_in [1, 1, N * S, W] bf16}: random stale contents, the own rows (random) in place."""
    N = R * C
    return {(r, c): torch.randn(1, 1, N * S, W, generator=gen).to(torch.bfloat16) for r in range(R) for c in range(C)}


def expected_xbuf(R, C, S, xbuf_in, route):
    """{(r, c): the dispatch's xbuf} from the inputs."""
    N = R * C
    RW = 1 if N <= 16 else 2
    out = {ch: x.clone() for ch, x in xbuf_in.items()}
    for (hr, hc), words in route.items():
        home = hr * C + hc
        w = words.flatten().tolist()
        for t in range(S):
            m = (w[t] & 0xFFFF) if RW == 1 else (w[2 * t] & 0xFFFFFFFF)
            slot = home * S + t
            for dst in range(N):
                if (m >> dst) & 1 and dst != home:
                    out[(dst // C, dst % C)][0, 0, slot] = xbuf_in[(hr, hc)][0, 0, slot]
    return out


def all_destinations(route, N):
    """route with every chip a destination of every row (the masks only): expected_xbuf's home row in every slot."""
    RW = 1 if N <= 16 else 2
    full = (1 << N) - 1
    out = {}
    for ch, words in route.items():
        w = words.clone().flatten()
        if RW == 1:
            w = (w & ~0xFFFF) | full
        else:
            w[0::2] = _i32(full)
        out[ch] = w.reshape(words.shape)
    return out


def assert_dispatched(out, xbuf_in, route, R, C, S, what="dispatch"):
    """out {(r, c): xbuf} against the model: every destination's row exactly; a chip a store-and-forward route passes
    through may keep the home's row in that slot; every other row as it was."""
    expected = expected_xbuf(R, C, S, xbuf_in, route)
    homes = expected_xbuf(R, C, S, xbuf_in, all_destinations(route, R * C))
    for ch in out:
        o, e, h = (t[ch].view(torch.int16) for t in (out, expected, homes))
        ok = (o == e).all(-1) | (o == h).all(-1)
        assert torch.all(ok), f"{what}: chip {ch}: {int((~ok).sum())} rows != model"


def fabric_payload(W):
    """The MoE block's fabric payload for rows of W bf16 (closeout_models.fabric_payload)."""
    return max(7232, (2 * W + 32 + 63) // 64 * 64)

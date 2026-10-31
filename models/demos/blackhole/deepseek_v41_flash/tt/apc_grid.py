# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""APC M0' (APC_PAGED_DESIGN.md section 8): which chunk boundaries of a prompt get a snapshot, so that prefix hits land on the 512 grid instead of power-of-two points.

Pure policy, no device and no torch: ``GridPolicy.plan(tokens, total)`` -> ``{B: kind}`` for ONE prompt (B = a multiple of the chunk, the position of the carried state to save).
It is the fallback tier of the device-resident block cache too (same function, only the store behind it differs).

Points (the usual power-of-two points are chosen by ``PrefixCache.wants`` and stay as they are):
  * ``tip``  the last full chunk boundary of the prompt, ``total // C * C`` (the best hit for a longer prompt that extends this one: multi-turn), and, when it differs, the last boundary a
             REPEAT of the prompt can use, ``(total - margin) // C * C`` (the lookup rule s <= e - margin);
  * ``div``  the deepest chunk boundary (<= ``total - margin``) whose block an EARLIER prompt already contained (a hash set of seen chunk chains): the divergence point of a shared system
             prompt / a branching conversation. The first prompt of a family only gets its tip; the second one snapshots the divergence, from the third on everything shared is a hit.
Cost bound: at most 3 extra snapshots per prompt (tip, tip2, div), none when the store already holds that boundary (``PrefixCache`` checks), and none below ``start`` (restored part).
"""

import hashlib
from collections import OrderedDict

import numpy as np


def _as_np(tokens):
    if hasattr(tokens, "numpy"):
        tokens = tokens.reshape(-1).long().numpy() if hasattr(tokens, "long") else tokens.numpy()
    return np.ascontiguousarray(np.asarray(tokens, dtype=np.int64).reshape(-1))


def chunk_chain(tokens, top, chunk):
    """{s: 16-byte key of tokens[:s]} for every multiple s of ``chunk`` up to ``top`` (one incremental sha256 pass)."""
    t = _as_np(tokens)
    top = min(int(top), t.size) // chunk * chunk
    h, out = hashlib.sha256(), {}
    for s in range(chunk, top + 1, chunk):
        h.update(t[s - chunk : s].tobytes())
        out[s] = h.copy().digest()[:16]
    return out


class GridPolicy:
    def __init__(self, chunk=512, margin=128, seen_cap=1 << 20, tips=2, div=True):
        self.chunk, self.margin, self.tips, self.div = int(chunk), int(margin), int(tips), bool(div)
        self.seen = OrderedDict()  # chunk-chain key -> None (LRU, bounded: 16 B per block of 512 tokens)
        self.seen_cap = int(seen_cap)
        self.div_extra, self.div_step = 0, 2 * self.chunk  # (see ``plan``; set from DSV41_APC_GRID_LADDER)

    def plan(self, tokens, total=None, zone=0):
        """Choose the boundaries to snapshot for a prompt of ``total`` tokens (default: len(tokens)) and register its chunk chain as seen. -> {B: 'tip' | 'tip2' | 'zone' | 'div'}; ``zone``: first replay chunk start of the prompt (0 = every layer on every chunk)."""
        C, m = self.chunk, self.margin
        n = _as_np(tokens).size
        total = n if total is None else min(int(total), n)
        ch = chunk_chain(tokens, total, C)
        out = {}
        top = total // C * C
        if top > 0 and self.tips >= 1:
            out[top] = "tip"
        t2 = max(total - m, 0) // C * C
        if zone > 0:
            # bounded replay (DSV41_PF_REPLAY): a snapshot taken at B > zone is 'partial' (PrefixSnapshot.full False) and only restores into a prompt whose own zone start is >= B. A repeat / a same-length
            # family therefore never uses a boundary past its zone: the usable points are the zone start itself (kind 'zone') and, for a longer prompt, the tip; a divergence point is capped at the zone start.
            if zone not in out:
                out[zone] = "zone"
        elif t2 > 0 and t2 != top and self.tips >= 2:
            out[t2] = "tip2"
        if self.div:
            usable = max(total - m, 0) // C * C
            if zone > 0:
                usable = min(usable, int(zone))
            sdiv = max((s for s, k in ch.items() if s <= usable and k in self.seen), default=0)
            if sdiv > 0 and sdiv not in out:
                out[sdiv] = "div"
            for j in range(
                1, self.div_extra + 1
            ):  # a ladder below it: later prompts of the family have zone starts a little lower (their tails are shorter), a hit needs s <= that zone
                s = sdiv - j * self.div_step
                if sdiv > 0 and s > 0 and s not in out:
                    out[s] = "div"
        for k in ch.values():
            if k in self.seen:
                self.seen.move_to_end(k)
            else:
                self.seen[k] = None
        while len(self.seen) > self.seen_cap:
            self.seen.popitem(last=False)
        return out

    def clear(self):
        self.seen.clear()


def pow2_wants(B, chunk=512):
    k = B // chunk
    return k > 0 and (k & (k - 1)) == 0

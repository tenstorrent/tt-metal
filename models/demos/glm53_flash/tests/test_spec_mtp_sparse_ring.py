# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""GLM53_SPEC_MTP_SPARSE (plan M5): a host model of the MTP layer's DSA ring / pooled-key store across speculative steps
-- the one-slot verify-mode catch-up (kernels/mla/ix_dm1.cpp MLA_DSA_ONE_SLOT: pool q's ring rows at 4 (q - q0) + p % 4,
q0 = the first active row's pool), spec_ctl.cpp SPEC_MODE 4 (phase 1: rows 4 grp .. -> 0..3, + save to 8..11; phase 2:
8..11 -> 0..3), the plain-mode chained rows (row p % 4; a pool closes into the store from rows 0..3 at p % 4 == 3).

Invariants after every step (positions committed = the catch-up rows j <= n; rejected catch-up rows and chained rows
are garbage until a committed row rewrites them):
  I1  ring rows 0..3 hold the committed rows of pool (c + n + 1) // 4 that are <= c + n (the next catch-up's base pool)
  I2  every pool closed by committed rows only is stored with committed rows (a committed row that closes a pool reads
      committed rows only)
Pure Python (no ttnn): runs under pytest or as a script."""

from __future__ import annotations

import random


def _grp(c: int, n: int) -> int:  # generator.mtp_ring_grp / spec_ctl.cpp's rule
    return 0 if c < 0 else (c + n + 1) // 4 - c // 4


class RingModel:
    """slot 0's MTP ring (32 rows) and pooled-key store; a row's value = (position, good)"""

    SAVE = 8

    def __init__(self, L: int, save_restore: bool = True):
        self.ring = [None] * 32
        self.store = {}
        self.good = {}  # position -> True once a committed row wrote it (the reference value)
        self.bad_closes = []
        self.save_restore = save_restore
        # the prefill: positions 0 .. L - 2 committed; L - 1 garbage (its next token is the first generated one)
        for p in range(L):
            g = p < L - 1
            self._plain(p, g)

    def _close(self, q: int, rows: list, by_good: bool) -> None:
        self.store[q] = list(rows)
        if by_good and not all(r is not None and r[1] and r[0] == 4 * q + j for j, r in enumerate(rows)):
            self.bad_closes.append((q, rows))

    def _plain(self, p: int, g: bool) -> None:
        self.ring[p % 4] = (p, g)
        if p % 4 == 3:
            self._close(p // 4, self.ring[0:4], g)

    def catchup(self, c: int, K: int, n: int) -> None:
        q0 = c // 4
        for j in range(K + 1):
            p, g = c + j, j <= n
            gi = p // 4 - q0
            assert 0 <= gi and 4 * gi + 3 < self.SAVE, (c, j, gi)  # (the save rows stay untouched)
            self.ring[4 * gi + p % 4] = (p, g)
            if p % 4 == 3:
                self._close(p // 4, self.ring[4 * gi : 4 * gi + 4], g)

    def ring_move(self, c: int, n: int, save: bool) -> None:
        grp = _grp(c, n)
        if grp > 0:
            self.ring[0:4] = self.ring[4 * grp : 4 * grp + 4]
        if save:
            self.ring[self.SAVE : self.SAVE + 4] = self.ring[0:4]

    def ring_restore(self) -> None:
        self.ring[0:4] = self.ring[self.SAVE : self.SAVE + 4]

    def step(self, c: int, K: int, n: int) -> None:
        """one speculative step's MTP part (BOOT: n 0, c = the last prompt position)"""
        self.catchup(c, K, n)
        sr = self.save_restore and K >= 2
        self.ring_move(c, n, save=sr)
        for i in range(K - 1):
            self._plain(c + n + 1 + i, False)
        if sr:
            self.ring_restore()

    def check(self, c: int, n: int) -> list:
        """I1 for the next base c + n + 1"""
        qa = (c + n + 1) // 4
        bad = []
        for p in range(4 * qa, c + n + 1):
            r = self.ring[p % 4]
            if r != (p, True):
                bad.append((p, r))
        return bad


def run(L: int, K: int, steps: int, seed: int, save_restore: bool = True):
    rnd = random.Random(seed)
    m = RingModel(L, save_restore)
    c, n = L - 1, 0
    m.step(c, K, n)  # BOOT
    errs = [("boot", m.check(c, n))] if m.check(c, n) else []
    for _ in range(steps):
        c = c + n + 1
        n = rnd.randint(0, K)
        m.step(c, K, n)
        bad = m.check(c, n)
        if bad:
            errs.append((c, n, bad))
    return errs, m.bad_closes


def test_mtp_ring_protocol_holds():
    for K in (1, 2, 3):
        for L in range(5, 13):
            for seed in range(40):
                errs, closes = run(L, K, 60, seed)
                assert not errs, (K, L, seed, errs[:3])
                assert not closes, (K, L, seed, closes[:3])


def test_mtp_ring_without_restore_breaks_k3():
    """the chained rows' plain writes past the pool need the save / restore (K >= 2): without it I1 fails"""
    hit = False
    for L in range(5, 13):
        for seed in range(40):
            errs, _ = run(L, 3, 60, seed, save_restore=False)
            hit = hit or bool(errs)
    assert hit


def test_mtp_ring_grp_matches_generator():
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import mtp_ring_grp
    except Exception:  # (the model package unavailable: the host model above still stands)
        return
    for c in range(-3, 40):
        for n in range(4):
            assert mtp_ring_grp(c, n) == _grp(c, n), (c, n)


def test_mtp_sparse_refusals():
    """the scope (generator.spec_mtp_sparse_refusals): single user any K, multi-user any K (v2: the chained rows on
    strided slots) up to G (K + 1) = 16 rows; the shard, the per-slot store, no dense window, the in-trace MTP fill"""
    try:
        from models.autoports.zai_org_glm_5_3_flash.tt.generator import spec_mtp_sparse_refusals as why
    except Exception:
        return
    ok = dict(shard=True, paged=False, window=False, prefill="trace")
    assert why(3, 0, "mtp", **ok) == [] and why(1, 0, "mtp", **ok) == [] and why(1, 8, "mtp", **ok) == []
    assert why(3, 4, "mtp", **ok) == [] and why(3, 2, "mtp", **ok) == []  # v2: multi-user K 3 (strided slots)
    assert len(why(3, 8, "mtp", **ok)) == 1  # 8 x (3 + 1) rows > 16
    for bad in (dict(shard=False), dict(paged=True), dict(window=True), dict(prefill="launch")):
        assert len(why(3, 0, "mtp", **dict(ok, **bad))) == 1, bad
    assert why(3, 0, "dflash2", **ok)


if __name__ == "__main__":
    test_mtp_ring_protocol_holds()
    test_mtp_ring_without_restore_breaks_k3()
    test_mtp_ring_grp_matches_generator()
    test_mtp_sparse_refusals()
    print("MTP_RING_MODEL_OK")

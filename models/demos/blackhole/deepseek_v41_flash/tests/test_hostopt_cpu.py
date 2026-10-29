# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU check: ``hostopt.post_round`` makes the same state updates as the per-user loop of ``AdaptiveSpec.run`` (the code it replaces, copied below)."""

import copy
import random

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import hostopt

BLOCK = 5


class _Run:
    def __init__(self, B):
        self.gaps = [[] for _ in range(B)]
        self.m_hist = [[] for _ in range(B)]


def _reference(run, a, mm, d, X5, base, done, gen, emitted, ms, eos, max_new, top, conf, newconf, k, cal, draft, stale):
    B = base.shape[0]
    n = k + 1
    gp = torch.full((B, n), float("nan"))
    for b in range(B):
        if done[b]:
            continue
        mb = int(mm[b])
        if k > 0 and not stale:
            cal.append((b, k, [float(x) for x in conf[b]], mb))
        toks = [int(x) for x in a[b, : mb + 1]]
        if eos is not None and eos in toks:
            toks = toks[: toks.index(eos) + 1]
            done[b] = True
        gen[b] += toks
        run.gaps[b] += [float(x) for x in gp[b, : len(toks)]]
        emitted.append(len(toks))
        ms.append(mb)
        run.m_hist[b].append(mb)
        base[b] += mb + 1
        X5[b, 0] = a[b, mb]
        if draft:
            X5[b, 1:] = d[b]
        if newconf is not None:
            conf[b] = newconf[b]
        if len(gen[b]) >= max_new or int(base[b]) > top:
            done[b] = True


def test_post_round_matches_loop():
    rnd = random.Random(0)
    for trial in range(300):
        B, k = rnd.choice([4, 8]), rnd.choice([0, 3, 5])
        n = k + 1
        eos = rnd.choice([None, 7])
        draft, stale = rnd.random() < 0.8, rnd.random() < 0.2
        g = torch.Generator().manual_seed(trial)
        a = torch.randint(0, 12, (B, n), generator=g)
        mm = torch.randint(0, n, (B,), generator=g)
        d = torch.randint(0, 100, (B, BLOCK), generator=g)
        X5 = torch.randint(0, 100, (B, 1 + BLOCK), generator=g)
        base = torch.randint(10, 50, (B,), generator=g)
        done = torch.rand(B, generator=g) < 0.25
        conf = torch.rand(B, BLOCK, generator=g)
        newconf = torch.rand(B, BLOCK, generator=g) if rnd.random() < 0.7 else None
        gen = [[1, 2] for _ in range(B)]
        max_new, top = rnd.choice([4, 6, 100]), rnd.choice([40, 60, 1000])
        s1 = dict(X5=X5.clone(), base=base.clone(), done=done.clone(), conf=conf.clone(), gen=copy.deepcopy(gen), emitted=[], ms=[], cal=[], run=_Run(B))
        s2 = dict(X5=X5.clone(), base=base.clone(), done=done.clone(), conf=conf.clone(), gen=copy.deepcopy(gen), emitted=[], ms=[], cal=[], run=_Run(B))
        _reference(s1["run"], a, mm, d, s1["X5"], s1["base"], s1["done"], s1["gen"], s1["emitted"], s1["ms"], eos, max_new, top, s1["conf"], newconf, k, s1["cal"], draft, stale)
        hostopt.post_round(
            s2["run"], a, mm, d, s2["X5"], BLOCK, s2["base"], s2["done"], s2["gen"], s2["emitted"], s2["ms"], eos, max_new, top,
            conf=s2["conf"], newconf=newconf, k=k, cal=s2["cal"] if (k > 0 and not stale) else None, draft=draft,
        )
        for key in ("X5", "base", "done", "conf"):
            assert torch.equal(s1[key], s2[key]), (trial, key)
        for key in ("gen", "emitted", "ms", "cal"):
            assert s1[key] == s2[key], (trial, key)
        assert s1["run"].m_hist == s2["run"].m_hist
        assert str(s1["run"].gaps) == str(s2["run"].gaps)

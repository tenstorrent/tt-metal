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
        s1 = dict(
            X5=X5.clone(),
            base=base.clone(),
            done=done.clone(),
            conf=conf.clone(),
            gen=copy.deepcopy(gen),
            emitted=[],
            ms=[],
            cal=[],
            run=_Run(B),
        )
        s2 = dict(
            X5=X5.clone(),
            base=base.clone(),
            done=done.clone(),
            conf=conf.clone(),
            gen=copy.deepcopy(gen),
            emitted=[],
            ms=[],
            cal=[],
            run=_Run(B),
        )
        _reference(
            s1["run"],
            a,
            mm,
            d,
            s1["X5"],
            s1["base"],
            s1["done"],
            s1["gen"],
            s1["emitted"],
            s1["ms"],
            eos,
            max_new,
            top,
            s1["conf"],
            newconf,
            k,
            s1["cal"],
            draft,
            stale,
        )
        hostopt.post_round(
            s2["run"],
            a,
            mm,
            d,
            s2["X5"],
            BLOCK,
            s2["base"],
            s2["done"],
            s2["gen"],
            s2["emitted"],
            s2["ms"],
            eos,
            max_new,
            top,
            conf=s2["conf"],
            newconf=newconf,
            k=k,
            cal=s2["cal"] if (k > 0 and not stale) else None,
            draft=draft,
        )
        for key in ("X5", "base", "done", "conf"):
            assert torch.equal(s1[key], s2[key]), (trial, key)
        for key in ("gen", "emitted", "ms", "cal"):
            assert s1[key] == s2[key], (trial, key)
        assert s1["run"].m_hist == s2["run"].m_hist
        assert str(s1["run"].gaps) == str(s2["run"].gaps)


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# Engram hash: RaggedNgramHash (shipped) vs hostopt.FastNgramHash / TorchFastNgramHash (DSV41_PF_HASH_FAST)
# ---------------------------------------------------------------------------------------------------------------------------------------------------
class _Layout:
    def __init__(self, ng):
        self.max_ngram_size = ng


class _HState:
    """Synthetic NgramHashState (same attributes the two hashes read): int64 wrap-around multipliers, DEAD entries in the token map."""

    def __init__(self, rows, cap, ng=4, layers=3, heads=2, vocab=300, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.layout = _Layout(ng)
        self.DEAD, self.pad_id = -1, 2
        self.token_map = torch.randint(0, 50, (vocab,), generator=g)
        self.token_map[::37] = self.DEAD
        self.cache = torch.full((rows, cap), self.pad_id, dtype=torch.long)
        self.multipliers = torch.randint(1, 2**62, (layers, ng), generator=g)
        self.primes = torch.randint(10007, 99991, (layers, ng - 1, heads), generator=g)
        self.offsets = torch.randint(0, 5000, (layers, (ng - 1) * heads), generator=g)


def _pair(rows, cap, **kw):
    from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash

    a, b = _HState(rows, cap, **kw), _HState(rows, cap, **kw)
    return RaggedNgramHash(a), hostopt.TorchFastNgramHash(b), a, b


def test_fast_hash_matches_ragged_windows():
    rnd = random.Random(1)
    for trial in range(60):
        R = rnd.choice([1, 4, 16, 16])
        n_users = rnd.choice([1, 4, 16])
        n_users = min(n_users, R)
        ng = rnd.choice([2, 3, 4])
        old, new, sa, sb = _pair(R, 400, ng=ng, seed=trial)
        g = torch.Generator().manual_seed(trial)
        rowsel = torch.tensor(rnd.sample(range(R), n_users))  # global user of each input row
        pos = [0] * R
        for w in range(rnd.choice([1, 3, 6])):
            L = rnd.choice([1, 5, 32, 64])  # windows of varying length
            ids = torch.randint(0, 300, (n_users, L), generator=g)
            if rnd.random() < 0.5:
                ids[:, 1:] = ids[:, :1]  # repeated tokens
            start = torch.tensor([pos[int(r)] for r in rowsel])  # per-user starts
            kw = {}
            if rnd.random() < 0.5:
                kw["rows"] = rowsel
            else:
                rowsel, start = torch.arange(n_users), torch.tensor([pos[i] for i in range(n_users)])
            if rnd.random() < 0.4:
                kw["token_mask"] = torch.rand(n_users, L, generator=g) > 0.3  # image-span windows
            use_rows = rowsel if "rows" in kw else torch.arange(n_users)
            x, y = old(ids, start, **kw), new(ids, start, **kw)
            assert y.dtype == x.dtype and torch.equal(x, y), (trial, w)
            assert torch.equal(sa.cache, sb.cache), (trial, w, "history")
            for r, s in zip(use_rows.tolist(), start.tolist()):
                pos[r] = s + L


def test_fast_hash_keeps_other_users_history():
    old, new, sa, sb = _pair(16, 200)
    for h, st in ((old, sa), (new, sb)):
        st.cache[:] = torch.randint(0, 50, st.cache.shape, generator=torch.Generator().manual_seed(5))
    before = sb.cache.clone()
    ids = torch.randint(0, 300, (2, 8))
    old(ids, torch.tensor([10, 20]), rows=torch.tensor([3, 9]))
    new(ids, torch.tensor([10, 20]), rows=torch.tensor([3, 9]))
    assert torch.equal(sa.cache, sb.cache)
    untouched = [r for r in range(16) if r not in (3, 9)]
    assert torch.equal(sb.cache[untouched], before[untouched])


def test_fast_hash_verify_mode(monkeypatch):
    monkeypatch.setenv("DSV41_PF_HASH_VERIFY", "5")
    old, new, sa, sb = _pair(4, 200)
    ids = torch.randint(0, 300, (3, 20))
    for w in range(7):
        st = torch.full((3,), 20 * w)
        a, b = old(ids, st, rows=torch.tensor([0, 2, 3])), new(ids, st, rows=torch.tensor([0, 2, 3]))
        assert torch.equal(a, b)
    assert new._vstat == [5, 0] and torch.equal(sa.cache[:, :140], sb.cache[:, :140])

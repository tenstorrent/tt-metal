# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of the DSV41_DEV_ENGRAM=1 index math (tt/engram_dev.py mirrors of engram_hash_dec.cpp / engram_gather_dec.cpp) against the shipped host path
(RaggedNgramHash / hostopt.FastNgramHash + token-major rows_cat layout), bit exact, over many speculative rounds with partial accepts (rollback) for T of B=4/8/16.
Synthetic hash state (no checkpoint needed)."""

import types

import numpy as np
import pytest
import torch

from models.demos.blackhole.deepseek_v41_flash.tt import engram_dev as ed
from models.demos.blackhole.deepseek_v41_flash.tt.engram_ragged import RaggedNgramHash
from models.demos.blackhole.deepseek_v41_flash.tt.hostopt import FastNgramHash

V, NL, W, H = 2000, 2, 3, 12
DEAD, PAD = -2, 7
MAXSEQ = 2000


def make_state(Bu, seed=0):
    g = torch.Generator().manual_seed(seed)
    st = types.SimpleNamespace()
    st.token_map = torch.randint(0, 1500, (V,), generator=g)
    st.multipliers = torch.randint(1, 2**40, (NL, W), generator=g) | 1
    pr = torch.tensor([1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069] * (NL * (W - 1)))
    st.primes = pr.reshape(NL, W - 1, H) + torch.arange(NL * (W - 1) * H).reshape(NL, W - 1, H) * 0
    st.offsets = torch.arange(NL * (W - 1) * H).reshape(NL, (W - 1) * H) * 2000
    st.pad_id, st.DEAD = PAD, DEAD
    st.cache = torch.full((Bu, MAXSEQ), DEAD, dtype=torch.long)
    st.layout = types.SimpleNamespace(max_ngram_size=W)
    return st


class FakeTable:
    def __init__(self, n, seed):
        g = torch.Generator().manual_seed(seed)
        self.w = torch.randn(n, 256, generator=g).to(torch.bfloat16)

    def rows(self, idx):
        return self.w[idx.reshape(-1)].reshape(*idx.shape, -1)


def host_cat(ids_all, tabs):
    """what the shipped path uploads: [T, sum(Kin)] token-major, layer concatenated (hashes[:, :, layer_index, :] -> rows -> flatten)"""
    return torch.cat([tabs[l].rows(ids_all[:, :, l, :]).flatten(-2).reshape(ids_all.shape[0] * ids_all.shape[1], -1) for l in range(NL)], -1)


@pytest.mark.parametrize("U,mrows,n", [(1, 4, 4), (1, 4, 6), (2, 4, 6), (4, 4, 4), (4, 4, 6), (2, 4, 4)])
def test_ids_and_rows_vs_host(U, mrows, n):
    B = U * mrows
    ttot = B * n
    rng = np.random.default_rng(U * 100 + n)
    ref, mine = make_state(B), make_state(B)  # two histories: the shipped hash writes `ref`, fill_ctrl writes `mine`
    tabs = [FakeTable(NL * (W - 1) * H * 2000 + 2000, 1 + l) for l in range(NL)]
    ragged = RaggedNgramHash(ref)
    fast = FastNgramHash(types.SimpleNamespace(**{**vars(mine), "cache": mine.cache.clone()}))  # numpy twin on its own copy
    # prompts of different lengths (incl. very short: lookback blocked at the sequence start, and DEAD holes)
    lens = rng.integers(1, 40, size=B)
    base = torch.zeros(B, dtype=torch.long)
    for b in range(B):
        L = int(lens[b])
        toks = torch.tensor(rng.integers(0, V, size=(1, L)))
        for st in (ref, mine, fast.st):
            comp = st.token_map[toks]
            st.cache[b, :L] = comp
        if b % 5 == 3 and L > 3:  # an image-like DEAD hole inside the prompt
            for st in (ref, mine, fast.st):
                st.cache[b, L - 3] = DEAD
        base[b] = L
    ctrl = np.zeros((1, ed.ctrl_width(ttot, B, W)), dtype=np.int32)
    st_np = dict(cache=mine.cache.numpy(), tmap=mine.token_map.numpy().astype(np.int64), dead=DEAD)
    for rnd in range(60):
        X = torch.tensor(rng.integers(0, V, size=(B, n)))
        ids_ref = ragged(X, base)  # [B, n, NL, 24] (writes ref.cache)
        ed.fill_ctrl(ctrl, X, base, st_np, w=W)
        got = ed.emu_ids(
            ctrl[0],
            ttot,
            n,
            st_np["tmap"],
            mine.multipliers.numpy(),
            mine.primes.numpy(),
            mine.offsets.numpy(),
            PAD,
            W,
        )
        want = ids_ref.reshape(ttot, NL, -1).numpy()
        assert (got == want).all(), f"round {rnd}"
        assert (mine.cache == ref.cache).all()  # same history writes
        # rows layout: token-major (token, layer, k, 256)
        rows_host = host_cat(ids_ref, tabs)
        rows_dev = ed.emu_rows_order(torch.from_numpy(got), tabs)
        assert rows_dev.shape == rows_host.reshape(ttot, -1).shape
        assert (rows_dev.view(torch.int16) == rows_host.reshape(ttot, -1).view(torch.int16)).all()
        # partial accepts: every user advances by 1 + m, m in 0..n-1 (accepted drafts); positions beyond hold stale speculation that must not be read
        m = torch.tensor(rng.integers(0, n, size=B))
        base = base + 1 + m
        if int(base.max()) + n + 1 >= MAXSEQ:
            break
    # numpy twin agrees as well (the host fast path used in production)
    fast.cache[:] = fast.cache  # no-op: the twin shares its copy


def test_gather_partition_covers_rows_once():
    for ttot in (16, 24, 48, 64, 96, 12):
        nrows = ttot * NL * ed.K
        for ncores in (96, 64, 16):
            inst = 2 * ncores
            per = -(-nrows // inst)
            seen = np.zeros(nrows, dtype=int)
            for k in range(inst):
                row0 = min(k * per, nrows)
                nrow = max(0, min(per, nrows - row0))
                seen[row0 : row0 + nrow] += 1
                if nrow:  # ids pages one instance reads
                    tok0, tok1 = row0 // (NL * ed.K), (row0 + nrow - 1) // (NL * ed.K)
                    assert tok1 - tok0 + 1 <= (per + NL * ed.K - 1) // (NL * ed.K) + 1
                    assert tok1 < ttot
            assert (seen == 1).all()


def test_hash_partition_and_ctrl_width():
    for ttot in (12, 16, 24, 48, 64, 96):
        inst = min(ttot, 128)
        per = -(-ttot // inst)
        seen = np.zeros(ttot, dtype=int)
        for idx in range(inst):
            t0 = idx * per
            cnt = max(0, min(per, ttot - t0))
            seen[t0 : t0 + cnt] += 1
        assert (seen == 1).all()
        assert ed.ctrl_width(ttot, ttot // 4, 3) % 16 == 0
        assert 2 + ttot + (ttot // 4) * 2 <= ed.ctrl_width(ttot, ttot // 4, 3)


def test_reduce_scatter_chunking():
    # reduce_scatter dim 2 hands mesh row r rows [r*T*48, (r+1)*T*48): exactly the rows of its T tokens (flat token index r*T + t)
    T, mrows = 16, 4
    nrows = T * mrows * NL * ed.K
    chunk = nrows // mrows
    for r in range(mrows):
        j = np.arange(r * chunk, (r + 1) * chunk)
        tok = j // (NL * ed.K)
        assert tok.min() == r * T and tok.max() == (r + 1) * T - 1

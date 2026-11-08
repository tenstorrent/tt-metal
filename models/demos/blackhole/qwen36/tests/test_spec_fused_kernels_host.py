# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of QWEN36_SPEC_CHAIN_KERNEL / QWEN36_SPEC_ACCEPT_KERNEL (tt/spec_fused_kernels.py), no device:
  * line-for-line Python mirrors of kernels/spec_chain_fused.cpp and kernels/spec_accept_fused.cpp (int32 arithmetic, the
    uint32 saturation rule) equal the exact host references chain_reference / fused_greedy_reference on thousands of
    random batches (the device test then pins the C++ to the op graphs);
  * the accept kernel's column-chunk plan covers every byte of a reseed page-table row exactly once, for every width and
    core count, inside the L1 budget;
  * the flags default off.
Run: pytest -o addopts="" --import-mode=importlib -q models/demos/blackhole/qwen36/tests/test_spec_fused_kernels_host.py"""

import pytest
import torch

from models.demos.blackhole.qwen36.tests.test_spec_fused_kernels import ACCEPT_SHAPES, CHAIN_SHAPES, accept_case, chain_case
from models.demos.blackhole.qwen36.tt import tp_common
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_words
from models.demos.blackhole.qwen36.tt.spec_fused_kernels import ACCEPT_CHUNK_MAX, _al64


def _u32(v):
    return 0 if v < 0 else v


def _rope_row(x, delta, rows_m1):
    v = x + delta
    v = 0 if v < 0 else v
    return v if v < rows_m1 else rows_m1


def _s32(w):
    w = int(w) & 0xFFFFFFFF
    return w - (1 << 32) if w >= 1 << 31 else w


def emulate_accept(ids, tok, aux, pt_rows, B, K, W, nb):
    """spec_accept_fused.cpp on host words: (pack [2K+5, W], rs_pt [R, nb], result [2, W]) as int64 bit patterns."""
    T, R = K + 1, B * (K + 1)
    a = [[_s32(x) for x in row] for row in torch.as_tensor(aux).tolist()]
    idw = [int(x) & 0xFFFFFFFF for x in torch.as_tensor(ids).reshape(-1).tolist()]
    tkw = [int(x) & 0xFFFFFFFF for x in torch.as_tensor(tok).reshape(-1).tolist()]
    scratch, rows_m1 = a[5][0], a[5][2] - 1
    m_of, real_of = [], []
    for u in range(B):
        nv, m = a[0][u], 0
        while m < K and m < nv and idw[u * T + m] == tkw[u * T + m + 1]:
            m += 1
        m_of.append(m)
        kv = m + a[2][u] * a[3][u]
        real_of.append(0 if kv <= 0 else (T if kv >= T else kv))
    pk = [[0] * W for _ in range(2 * K + 5)]
    rr = [[0] * W for _ in range(2)]
    G2, G3, G4, G5 = K + 1, K + 3, K + 4, K + 5
    for u in range(B):
        m, ndec, p0, delta = m_of[u], 1 - a[2][u], a[4][u], a[6][u]
        p = (p0 + m) * ndec
        pk[0][u] = _u32(idw[u * T + m] * ndec)
        for k in range(K):
            pk[1 + k][u] = _u32(p + k)
            pk[G5 + k][u] = _u32(_rope_row(p + k, delta, rows_m1))
        pk[G3][u] = u * T + m
        for j in range(T):
            r = u * T + j
            cm = idw[r] if j <= m else -1
            real = 1 if j < real_of[u] else 0
            rpos = (p0 + j) * real
            pk[G2][r] = _u32(cm * real)
            pk[G2 + 1][r] = _u32(rpos)
            pk[G4][r] = _u32(_rope_row(rpos, delta, rows_m1))
            rr[0][r] = cm
        rr[1][u] = m + 1
    pt = torch.as_tensor(pt_rows).to(torch.int64)
    rs = torch.stack([pt[r] if r % T < real_of[r // T] else torch.full((nb,), scratch, dtype=torch.int64) for r in range(R)])
    w = lambda x: torch.as_tensor(x, dtype=torch.int64) & 0xFFFFFFFF
    return w(pk), w(rs), w(rr)


def emulate_chain(res, prev_aux, drafts, ovr, B, K, Nv, W_v, W_a):
    """spec_chain_fused.cpp on host words: (tok [R], vaux [3, W_v], next aux [7, W_a]) as int64 bit patterns."""
    T, R = K + 1, B * (K + 1)
    rs = [[_s32(x) for x in row] for row in torch.as_tensor(res).tolist()]
    pa = [[_s32(x) for x in row] for row in torch.as_tensor(prev_aux).tolist()]
    o = [[_s32(x) for x in row] for row in torch.as_tensor(ovr).tolist()]
    dr = [[int(x) for x in row] for row in torch.as_tensor(drafts).tolist()]  # [B][K]
    rows_m1, parity = o[5][2] - 1, o[9][0]
    tok, pos, ridx, ctrl = [0] * R, [0] * W_v, [0] * W_v, [0] * W_v
    naux = [[0] * W_a for _ in range(7)]
    ctrl[0] = _u32(parity)
    for u in range(B):
        nv, act, mask, cnt = o[0][u], o[1][u], o[6][u], rs[1][u]
        nm, m = 1 - mask, cnt - 1
        p0n = ((pa[4][u] + cnt) * nm + o[4][u] * mask) * act
        mi = (m * nm + o[8][u] * mask) * act
        tok0 = rs[0][u * T + m] if 0 <= m < T else 0
        for j in range(T):
            tdev = dr[u][j - 1] if j > 0 and j - 1 < nv else tok0
            r = u * T + j
            tok[r] = _u32((tdev * nm + o[7][r] * mask) * act)
            pos[r] = _u32(p0n + j)
            ridx[r] = _u32(_rope_row(p0n + j, o[10][u], rows_m1))
        ctrl[1 + u] = _u32(mi)
        for h in range(Nv):
            ctrl[1 + B + u * Nv + h] = _u32((mi * B + u) * Nv + h)
        naux[4][u] = p0n
    for i in range(W_a):
        for row in range(4):
            naux[row][i] = o[row][i]
        if i >= B:
            naux[4][i] = 0
        naux[5][i] = o[5][i]
        naux[6][i] = o[10][i]
    w = lambda x: torch.as_tensor(x, dtype=torch.int64) & 0xFFFFFFFF
    return w(tok), w([pos, ridx, ctrl]), w(naux)


@pytest.mark.parametrize("mixed", [False, True], ids=["common_delta", "row_deltas"])
def test_accept_mirror_matches_reference(mixed):
    n = 0
    for B, K, nb in ACCEPT_SHAPES:
        g = torch.Generator().manual_seed(31 * B + 7 * K + nb + int(mixed))
        for i in range(60):
            c = accept_case(g, B, K, nb, edge=i % 3 == 2, mixed=mixed)
            if c is None:
                continue
            R, ref = B * (K + 1), c["ref"]
            pk, rs, rr = emulate_accept(c["ids"][:R], c["tok"], c["aux"], c["pt_rows"], B, K, 32, nb)
            want_rr = torch.zeros(2, 32, dtype=torch.int64)
            want_rr[0, :R] = ref.committed.reshape(-1).to(torch.int64)
            want_rr[1, :B] = ref.counts.to(torch.int64)
            assert torch.equal(pk, ref.pack.to(torch.int64) & 0xFFFFFFFF), f"pack B={B} K={K} nb={nb} case {i}"
            assert torch.equal(rs, ref.rs_pt.to(torch.int64) & 0xFFFFFFFF), f"rs_pt B={B} K={K} nb={nb} case {i}"
            assert torch.equal(rr, want_rr & 0xFFFFFFFF), f"result B={B} K={K} nb={nb} case {i}"
            n += 1
    assert n > 1000, n


@pytest.mark.parametrize("mixed", [False, True], ids=["common_delta", "row_deltas"])
def test_chain_mirror_matches_reference(mixed):
    n = 0
    for B, K, Nv in CHAIN_SHAPES:
        g = torch.Generator().manual_seed(77 * B + 5 * K + Nv + int(mixed))
        R = B * (K + 1)
        W_v = -(-max(R, spec_ctrl_words(B, Nv)) // 32) * 32
        W_a = -(-max(B, 5) // 32) * 32
        for i in range(100):
            c = chain_case(g, B, K, Nv, i % 2, mixed=mixed)
            tok, vaux, naux = emulate_chain(c["res"], c["prev_aux"], c["drafts"], c["ovr"], B, K, Nv, W_v, W_a)
            rt, rv, ra = c["ref"]
            assert torch.equal(tok, rt.to(torch.int64) & 0xFFFFFFFF), f"tok B={B} K={K} Nv={Nv} case {i}"
            assert torch.equal(vaux, rv.to(torch.int64) & 0xFFFFFFFF), f"vaux B={B} K={K} Nv={Nv} case {i}"
            assert torch.equal(naux, ra.to(torch.int64) & 0xFFFFFFFF), f"next aux B={B} K={K} Nv={Nv} case {i}"
            n += 1
    assert n >= 1000, n


def _accept_plan(nb, n_cores, grid=(11, 10)):
    """AcceptKernel.__init__'s core / chunk choice without a device (same arithmetic)."""
    n = max(1, min(int(n_cores), grid[0] * grid[1]))
    row_bytes = 4 * nb
    n = max(1, min(n, -(-row_bytes // 64)))
    chunk = min(ACCEPT_CHUNK_MAX, _al64(-(-row_bytes // n)))
    n = min(n, -(-row_bytes // chunk))
    return n, chunk


@pytest.mark.parametrize("cores", [1, 7, 16, 32, 110])
def test_accept_chunk_plan_covers_rows(cores):
    for nb in list(range(1, 300)) + [537, 1024, 4096, 4097, 8193, 16385]:
        n, chunk = _accept_plan(nb, cores)
        row_bytes = 4 * nb
        n_chunks = -(-row_bytes // chunk)
        assert chunk % 64 == 0 and 64 <= chunk <= ACCEPT_CHUNK_MAX, (nb, cores, chunk)
        assert 1 <= n <= cores and n <= n_chunks, (nb, cores, n, n_chunks)
        cover = [0] * row_bytes
        for c in range(n):  # the kernel loop: chunks c, c + n, ... < n_chunks
            for ch in range(c, n_chunks, n):
                lo = ch * chunk
                for b in range(lo, min(lo + chunk, row_bytes)):
                    cover[b] += 1
        assert all(x == 1 for x in cover), (nb, cores)
        l1 = 128 + 128 + (7 + 11 + 2) * 128 + chunk + 32 * chunk + 64  # B=8 K=3 W=32 CB bytes
        assert l1 <= 40 * 1024, (nb, cores, l1)


def test_flags_default_off(monkeypatch):
    for k in ("QWEN36_SPEC_CHAIN_KERNEL", "QWEN36_SPEC_ACCEPT_KERNEL", "QWEN36_SPEC_ACCEPT_KERNEL_CORES"):
        monkeypatch.delenv(k, raising=False)
    assert not tp_common.spec_chain_kernel_enabled() and not tp_common.spec_accept_kernel_enabled()
    assert tp_common.spec_accept_kernel_cores() == 32
    monkeypatch.setenv("QWEN36_SPEC_CHAIN_KERNEL", "1")
    monkeypatch.setenv("QWEN36_SPEC_ACCEPT_KERNEL", "1")
    monkeypatch.setenv("QWEN36_SPEC_ACCEPT_KERNEL_CORES", "0")
    assert tp_common.spec_chain_kernel_enabled() and tp_common.spec_accept_kernel_enabled()
    assert tp_common.spec_accept_kernel_cores() == 1

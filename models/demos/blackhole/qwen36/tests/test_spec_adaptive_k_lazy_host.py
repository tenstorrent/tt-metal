# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for QWEN36_SPEC_ADAPTIVE_K on the K4 lazy GDN verify (QWEN36_SPEC_STEP_LAZY=1).

K4 keeps the state before the previous verify in ring slot 0 and that verify's stash (K^T, U, decays at rows
r0 = (u*T) % 32); the next verify replays the accepted prefix. Adaptive K alternates T between the base (B=8, T=4) and
wide (b=4, T=6) families, so after a family switch:
  * the stash rows sit at the PREVIOUS verify's r0 (u*T_prev % 32), which the ctrl page now carries (word 0 bits 8..15);
  * a carried mi may exceed T - 1 (a wide prefix replayed by a base verify), so FRESH moves to T_alloc (mi_max + 1);
  * a parked row is replayed on host at the stash's own T.
These tests check the kernel mirror (lazy_ref.ctrl_replay) and the host plumbing against the serial gated delta rule in
fp64, plus the flag-off identity of every word the op reads. No device, no op.
"""

import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt.gdn import lazy_ref as lr

Dk = Dv = 128
TILE = 32


def _tokens(gen, T, dt=torch.float64):
    q = torch.randn(T, Dk, generator=gen, dtype=dt)
    k = torch.randn(T, Dk, generator=gen, dtype=dt) + 1.5 * torch.randn(1, Dk, generator=gen, dtype=dt)
    v = torch.nn.functional.silu(0.8 * torch.randn(T, Dv, generator=gen, dtype=dt))
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * Dk**-0.5
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(torch.randn(T, generator=gen, dtype=dt))
    dec = torch.exp(-1.2 * torch.nn.functional.softplus(0.3 * torch.randn(T, generator=gen, dtype=dt) + 0.1))
    return qn, kn, v, beta, dec


def _stash(kn, U, dec, r0, T):
    kt = torch.zeros(Dk, TILE, dtype=torch.float64)
    kt[:, r0 : r0 + T] = kn.T
    ut = torch.zeros(TILE, Dv, dtype=torch.float64)
    ut[r0 : r0 + T] = U
    dect = torch.zeros(TILE, dtype=torch.float64)
    dect[r0 : r0 + T] = dec
    return kt, ut, dect


# --------------------------------------------------------------------------- #
# kernel mirror: flag off == prod
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("T", [3, 4, 6])
def test_ctrl_replay_flag_off_is_prod(T):
    """Without mi_max and with a parity-only word 0, the mirror is prod's writer: r0 = (u*T) % 32, r = mi + 1, FRESH at mi >= T."""
    for u in range(32 // T):
        for mi in range(T + 2):
            for par in (0, 1):
                r0p, r, p = lr.ctrl_replay(lr.ctrl_word0(par), mi, u, T)
                assert r0p == (u * T) % TILE and p == par
                assert r == (0 if mi >= T else mi + 1)
            assert lr.ctrl_replay(0, mi, u, T, hold=True)[1] == 0


def test_ctrl_word0_parity_only_by_default():
    assert lr.ctrl_word0(0) == 0 and lr.ctrl_word0(1) == 1
    assert lr.ctrl_word0(1, 6) == 1 | (6 << 8)
    assert lr.ctrl_word0(0, 6) & 1 == 0  # the reader / writer read the parity as word0 & 1


def test_fresh_moves_to_t_alloc_under_mi_max():
    """A base verify (T=4) after a wide one (T_alloc=6, mi_max=5): mi 4 / 5 are real 5- / 6-token prefixes, FRESH is 6."""
    T, mm = 4, 5
    assert lr.fresh_at(T) == 4 and lr.fresh_at(T, mm) == 6
    w0 = lr.ctrl_word0(0, 6)
    assert lr.ctrl_replay(w0, 4, 1, T, mm)[1] == 5
    assert lr.ctrl_replay(w0, 5, 1, T, mm)[1] == 6
    assert lr.ctrl_replay(w0, 6, 1, T, mm)[1] == 0
    assert lr.ctrl_replay(w0, 5, 1, T, mm)[0] == 6  # u=1's wide stash rows start at 1*6


def test_replay_never_reads_past_the_tile():
    """ctrl contents are data: a bogus T_prev / mi never makes r0p + r exceed 32."""
    for tp in range(0, 40):
        for mi in range(0, 40):
            for u in range(8):
                r0p, r, _ = lr.ctrl_replay(lr.ctrl_word0(0, tp), mi, u, 4, 31)
                assert r0p + r <= TILE


# --------------------------------------------------------------------------- #
# the contract across family switches tracks the serial recurrence
# --------------------------------------------------------------------------- #
def _family_schedule(steps, gen):
    """T per step: base (4) and wide (6) runs with switches both ways."""
    out, T = [], 4
    for _ in range(steps):
        if float(torch.rand(1, generator=gen)) < 0.4:
            T = 6 if T == 4 else 4
        out.append(T)
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_lazy_family_switch_tracks_serial(seed):
    """4 users (the wide family's width), 14 steps alternating T in {4, 6}: replaying the previous verify's stash at its
    own rows (r0p from the ctrl word) reproduces the serial committed state; the base verify's mi range is T_alloc."""
    B, T_alloc, steps = 4, 6, 14
    gen = torch.Generator().manual_seed(seed)
    sched = _family_schedule(steps, gen)
    S_true = [0.05 * torch.randn(Dk, Dv, generator=gen, dtype=torch.float64) for _ in range(B)]
    D = [s.clone() for s in S_true]
    stash = [None] * B
    mi_ctrl = [T_alloc] * B  # FRESH under adaptive K = T_alloc (spec_engine: _T_state)
    prev_T = None
    for step, T in enumerate(sched):
        mi_max = T_alloc - 1 if T < T_alloc else 0  # gdn/tp.py: mx only when Ta > T
        w0 = lr.ctrl_word0(step & 1, prev_T if (prev_T and prev_T != T) else 0)
        for u in range(B):
            r0p, r, _ = lr.ctrl_replay(w0, mi_ctrl[u], u, T, mi_max)
            if r:
                kt, ut, dect, r0_written = stash[u]
                assert r0p == r0_written, (step, u)  # the op looks where the previous verify wrote
                D[u] = lr.replay(D[u], kt, ut, dect, r0p, r)
            assert (D[u] - S_true[u]).abs().max() < 1e-12, (step, u, T, prev_T)
            qn, kn, v, beta, dec = _tokens(gen, T)
            O, U = lr.chunk_wy(D[u], qn, kn, v, beta, dec)
            O_s, states = lr.chunk_serial(S_true[u], qn, kn, v, beta, dec)
            assert (O - O_s).abs().max() < 1e-12
            r0 = (u * T) % TILE
            stash[u] = (*_stash(kn, U, dec, r0, T), r0)
            mi = int(torch.randint(0, T, (1,), generator=gen))
            S_true[u] = states[mi]
            mi_ctrl[u] = mi
        prev_T = T


def test_without_prev_t_a_switch_replays_the_wrong_rows():
    """The failure the ctrl word fixes: after a wide verify (T=6), a base verify (T=4) of user 2 reading its stash at its own
    r0 = 8 instead of the wide r0 = 12 gets a different (wrong) state."""
    gen = torch.Generator().manual_seed(7)
    u, Tw, Tb = 2, 6, 4
    S = 0.05 * torch.randn(Dk, Dv, generator=gen, dtype=torch.float64)
    qn, kn, v, beta, dec = _tokens(gen, Tw)
    _, U = lr.chunk_wy(S, qn, kn, v, beta, dec)
    _, states = lr.chunk_serial(S, qn, kn, v, beta, dec)
    kt, ut, dect = _stash(kn, U, dec, (u * Tw) % TILE, Tw)
    mi = 4
    good = lr.replay(S, kt, ut, dect, lr.ctrl_replay(lr.ctrl_word0(0, Tw), mi, u, Tb, Tw - 1)[0], mi + 1)
    bad = lr.replay(S, kt, ut, dect, (u * Tb) % TILE, mi + 1)
    assert (good - states[mi]).abs().max() < 1e-12
    assert (bad - states[mi]).abs().max() > 1e-6


# --------------------------------------------------------------------------- #
# host plumbing
# --------------------------------------------------------------------------- #
def _ctrl_model(T, last_T, B=8, Nv=6):
    return types.SimpleNamespace(_vfy_B=B, _vfy_Nv=Nv, _vfy_T=T, _lz_last_T=last_T)


@pytest.mark.parametrize(
    "lazy,ak,T,last,want_hi",
    [
        ("0", "0", 4, 6, 0),  # flag off: parity only
        ("1", "0", 4, 6, 0),  # lazy without adaptive: never set
        ("0", "1", 4, 6, 0),  # adaptive without lazy: the serial kernel needs no stash rows
        ("1", "1", 4, 4, 0),  # same T: parity only (the chain device's word)
        ("1", "1", 4, 6, 6),  # base after wide
        ("1", "1", 6, 4, 4),  # wide after base
        ("1", "1", 4, None, 0),  # no verify yet
    ],
)
def test_vfy_ctrl_page_prev_t(monkeypatch, lazy, ak, T, last, want_hi):
    from models.demos.blackhole.qwen36.tt.model import Qwen36Model

    monkeypatch.setenv("QWEN36_SPEC_STEP_LAZY", lazy)
    monkeypatch.setenv("QWEN36_SPEC_ADAPTIVE_K", ak)
    m = _ctrl_model(T, last)
    page = Qwen36Model._vfy_ctrl_page(m, [1, 2, 0, 3, 1, 0, 2, 1], 1).reshape(-1)
    assert int(page[0]) & 1 == 1
    assert (int(page[0]) >> 8) & 0xFF == want_hi
    # every other word is the flag-off page
    monkeypatch.setenv("QWEN36_SPEC_ADAPTIVE_K", "0")
    ref = Qwen36Model._vfy_ctrl_page(m, [1, 2, 0, 3, 1, 0, 2, 1], 1).reshape(-1)
    assert torch.equal(page[1:], ref[1:]) and int(ref[0]) == 1


def test_parked_rows_host_lazy_uses_the_stash_t():
    """A row parked right after a wide verify (T=6) is replayed at the wide rows even though the ring has T_alloc slots."""
    from models.demos.blackhole.qwen36.tt.spec_engine import parked_rows_host_lazy

    T_alloc, Tw, Nv, row, s, par = 6, 6, 1, 3, 4, 0
    gen = torch.Generator().manual_seed(11)
    ring = torch.randn(T_alloc, Nv, Dk, Dv, generator=gen)
    e_prev = torch.randn(1, 3 + T_alloc, 64, generator=gen)
    slot, _ = parked_rows_host_lazy(ring, e_prev, par, s, T_alloc, row, stash_T=Tw)
    kt, ut, dect = lr.stash_from_block(ring[1 + par, 0], Dk // TILE, Dv // TILE)
    assert torch.allclose(slot[0, 0], lr.replay(ring[0, 0], kt, ut, dect, (row * Tw) % TILE, s + 1))
    # a base-written stash (T=4) under the same ring
    slot4, _ = parked_rows_host_lazy(ring, e_prev, par, 2, T_alloc, row, stash_T=4)
    assert torch.allclose(slot4[0, 0], lr.replay(ring[0, 0], kt, ut, dect, (row * 4) % TILE, 3))
    # stash_T None = the ring's T (prod: T == the verify's T)
    slot_d, _ = parked_rows_host_lazy(ring, e_prev, par, s, T_alloc, row)
    assert torch.allclose(slot_d[0, 0], lr.replay(ring[0, 0], kt, ut, dect, (row * T_alloc) % TILE, s + 1))


def test_ak_mask_to_k():
    from models.demos.blackhole.qwen36.tt.spec_adaptive import ak_mask_to_k

    m = torch.arange(2 * 6 * 3, dtype=torch.int32).reshape(2, 6, 3)
    assert torch.equal(ak_mask_to_k(m, 3), m[:, :4])
    assert ak_mask_to_k(m, 5) is m or torch.equal(ak_mask_to_k(m, 5), m)

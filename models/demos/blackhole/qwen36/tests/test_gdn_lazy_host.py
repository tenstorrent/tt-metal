# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for K4 (QWEN36_SPEC_STEP_LAZY: gdn_spec_step kernel_opts bit 8, chunked WY token phase + lazy rollback).

* the math the kernel implements (tt/gdn/lazy_ref.py), in fp64: the WY chunk equals the serial gated delta rule
  (outputs and deltas), the replay of an accepted prefix equals the serial state after it, and a multi-step lazy
  sequence (FRESH rows, random accepted prefixes, parity double-buffered stash) tracks the serial committed state;
* the stash page layout (K^T pages, U pages, decay page inside one [Dk, Dv] ring block) round-trips, and the host park
  replay (parked_slot0_lazy / spec_engine.parked_rows_host_lazy) reconstructs the committed state from a ring copy;
* the flag plumbing: QWEN36_SPEC_STEP_LAZY off by default and NOT reachable through QWEN36_SPEC_STEP_OPT; kernel_opts
  bit 8 passed to the T > 1 verify only (never to the T = 1 seed) and only with fp32_state; T < 3 refused; the ctrl page
  of a FRESH row (mi = T) is the sentinel the op reads (block words carry nothing else); the model's mi range check
  admits mi = T only with the flag.

No device, no op. Run: pytest models/demos/blackhole/qwen36/tests/test_gdn_lazy_host.py -q
"""

import types

import pytest
import torch

from models.demos.blackhole.qwen36.tt.gdn import lazy_ref as lr

Dk = Dv = 128
TILE = 32


def _tokens(gen, T, dt=torch.float64, slow=False):
    q = torch.randn(T, Dk, generator=gen, dtype=dt)
    k = torch.randn(T, Dk, generator=gen, dtype=dt) + 1.5 * torch.randn(1, Dk, generator=gen, dtype=dt)
    v = torch.nn.functional.silu(0.8 * torch.randn(T, Dv, generator=gen, dtype=dt))
    qn = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * Dk**-0.5
    kn = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    beta = torch.sigmoid(torch.randn(T, generator=gen, dtype=dt))
    A = -0.05 if slow else -1.2
    dec = torch.exp(A * torch.nn.functional.softplus(0.3 * torch.randn(T, generator=gen, dtype=dt) + 0.1))
    return qn, kn, v, beta, dec


@pytest.mark.parametrize("T", [3, 4, 8])
@pytest.mark.parametrize("slow", [False, True])
def test_wy_chunk_equals_serial(T, slow):
    gen = torch.Generator().manual_seed(10 + T)
    S = 0.05 * torch.randn(Dk, Dv, generator=gen, dtype=torch.float64)
    qn, kn, v, beta, dec = _tokens(gen, T, slow=slow)
    O, U = lr.chunk_wy(S, qn, kn, v, beta, dec)
    O_s, states = lr.chunk_serial(S, qn, kn, v, beta, dec)
    assert (O - O_s).abs().max() < 1e-12
    # U[t] is the serial delta of token t: S_t - dec_t * S_{t-1} = k_t^T U_t
    prev = S
    for t in range(T):
        upd = states[t] - dec[t] * prev
        assert (upd - torch.outer(kn[t], U[t])).abs().max() < 1e-12
        prev = states[t]


@pytest.mark.parametrize("T", [3, 4, 8])
def test_replay_equals_serial_prefix(T):
    gen = torch.Generator().manual_seed(20 + T)
    D = 0.05 * torch.randn(Dk, Dv, generator=gen, dtype=torch.float64)
    qn, kn, v, beta, dec = _tokens(gen, T)
    _, U = lr.chunk_wy(D, qn, kn, v, beta, dec)
    _, states = lr.chunk_serial(D, qn, kn, v, beta, dec)
    for u in (0, 3, 7):  # the user's row offset r0 = (u * T) % 32 in the verify tile
        r0 = (u * T) % TILE
        if r0 + T > TILE:
            continue
        kt = torch.zeros(Dk, TILE, dtype=torch.float64)
        kt[:, r0 : r0 + T] = kn.T
        ut = torch.zeros(TILE, Dv, dtype=torch.float64)
        ut[r0 : r0 + T] = U
        dect = torch.zeros(TILE, dtype=torch.float64)
        dect[r0 : r0 + T] = dec
        assert torch.equal(lr.replay(D, kt, ut, dect, r0, 0), D)
        for r in range(1, T + 1):
            S = lr.replay(D, kt, ut, dect, r0, r)
            assert (S - states[r - 1]).abs().max() < 1e-12, (u, r)


def test_lazy_sequence_tracks_serial():
    """Engine-shaped sequence for 8 users x 12 steps, T = 4: the op's contract (D in slot 0, the stash of the previous
    verify, FRESH rows) reproduces the serial committed state and outputs (fp64)."""
    B, T, steps = 8, 4, 12
    gen = torch.Generator().manual_seed(3)
    S_true = [0.05 * torch.randn(Dk, Dv, generator=gen, dtype=torch.float64) for _ in range(B)]
    D = [s.clone() for s in S_true]
    stash = [None] * B
    mi_ctrl = [lr.fresh_mi(T)] * B
    for step in range(steps):
        for u in range(B):
            if step == 5 and u in (2, 6):  # a re-seeded row (fast join / unpark): slot 0 rewritten, FRESH
                S_true[u] = 0.05 * torch.randn(Dk, Dv, generator=gen, dtype=torch.float64)
                D[u] = S_true[u].clone()
                mi_ctrl[u] = lr.fresh_mi(T)
            qn, kn, v, beta, dec = _tokens(gen, T)
            r = 0 if mi_ctrl[u] >= T else mi_ctrl[u] + 1
            if r:
                kt, ut, dect, r0 = stash[u]
                D[u] = lr.replay(D[u], kt, ut, dect, r0, r)
            assert (D[u] - S_true[u]).abs().max() < 1e-12, (step, u)
            O, U = lr.chunk_wy(D[u], qn, kn, v, beta, dec)
            O_s, states = lr.chunk_serial(S_true[u], qn, kn, v, beta, dec)
            assert (O - O_s).abs().max() < 1e-12, (step, u)
            r0 = (u * T) % TILE
            kt = torch.zeros(Dk, TILE, dtype=torch.float64)
            kt[:, r0 : r0 + T] = kn.T
            ut = torch.zeros(TILE, Dv, dtype=torch.float64)
            ut[r0 : r0 + T] = U
            dect = torch.zeros(TILE, dtype=torch.float64)
            dect[r0 : r0 + T] = dec
            stash[u] = (kt, ut, dect, r0)
            mi = int(torch.randint(0, T, (1,), generator=gen))
            S_true[u] = states[mi]
            mi_ctrl[u] = mi


def test_stash_block_round_trip():
    Kt = Vt = 4
    gen = torch.Generator().manual_seed(4)
    kt = torch.randn(Kt * TILE, TILE, generator=gen)
    u = torch.randn(TILE, Vt * TILE, generator=gen)
    dec = torch.rand(TILE, generator=gen)
    blk = lr.stash_to_block(kt, u, dec, Kt, Vt)
    kt2, u2, dec2 = lr.stash_from_block(blk, Kt, Vt)
    assert torch.equal(kt, kt2) and torch.equal(u, u2) and torch.equal(dec, dec2)
    # page j of the block = tile (j // Vt, j % Vt): K^T tile 1 is block columns 32..63 of the first tile row
    assert torch.equal(blk[0:TILE, TILE : 2 * TILE], kt[TILE : 2 * TILE])
    assert torch.equal(blk[TILE : 2 * TILE, 0:TILE], u[:, 0:TILE])  # page Kt = 4 -> tile (1, 0)
    assert torch.equal(blk[2 * TILE : 3 * TILE, 0], dec)  # page Kt + Vt = 8 -> tile (2, 0), column 0


@pytest.mark.parametrize("par", [0, 1])
@pytest.mark.parametrize("s", [0, 2, 3])
def test_parked_slot0_lazy(par, s):
    """A parked row's host ring copy [D*T, Nv, Dk, Dv] (2 devices, 2 heads) -> slot 0 = D replayed s + 1 tokens."""
    T, Nv, Kt, Vt, row = 4, 2, 4, 4, 5
    r0 = (row * T) % TILE
    gen = torch.Generator().manual_seed(11 + s + 7 * par)
    ring = torch.randn(2 * T, Nv, Dk, Dv, generator=gen)  # garbage everywhere first
    want = torch.zeros(2, Nv, Dk, Dv)
    for d in range(2):
        for h in range(Nv):
            D = 0.05 * torch.randn(Dk, Dv, generator=gen)
            qn, kn, v, beta, dec = _tokens(gen, T, dt=torch.float32)
            _, U = lr.chunk_wy(D, qn, kn, v, beta, dec)
            kt = torch.zeros(Dk, TILE)
            kt[:, r0 : r0 + T] = kn.T
            ut = torch.zeros(TILE, Dv)
            ut[r0 : r0 + T] = U
            dect = torch.zeros(TILE)
            dect[r0 : r0 + T] = dec
            ring[d * T + 0, h] = D
            ring[d * T + 1 + par, h] = lr.stash_to_block(kt, ut, dect, Kt, Vt)
            want[d, h] = lr.chunk_serial(D.double(), qn.double(), kn.double(), v.double(), beta.double(), dec.double())[
                1
            ][s].float()
    got = lr.parked_slot0_lazy(ring, s, T, par, r0, Kt, Vt)
    assert (got - want).abs().max() < 2e-5


def test_parked_rows_host_lazy_matches_contract():
    """spec_engine.parked_rows_host_lazy: the E_prev shift of parked_rows_host + the host replay of slot 0."""
    from models.demos.blackhole.qwen36.tt.spec_engine import parked_rows_host, parked_rows_host_lazy

    T, Nv, row, s, par = 4, 1, 3, 1, 1
    gen = torch.Generator().manual_seed(9)
    ring = torch.randn(T, Nv, Dk, Dv, generator=gen)
    e_prev = torch.randn(1, 3 + T, 64, generator=gen)
    slot, e = parked_rows_host_lazy(ring, e_prev, par, s, T, row)
    _, e_ref = parked_rows_host(ring, e_prev, s, T)
    assert torch.equal(e, e_ref)
    kt, ut, dect = lr.stash_from_block(ring[1 + par, 0], Dk // TILE, Dv // TILE)
    assert torch.allclose(slot[0, 0], lr.replay(ring[0, 0], kt, ut, dect, (row * T) % TILE, s + 1))


# --------------------------------------------------------------------------- #
# flag plumbing (no op: the kda module is a recorder)
# --------------------------------------------------------------------------- #
def _spec_step_layer(fp32_state=True):
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

    g = types.SimpleNamespace(
        tw={"conv_taps_rows": "taps", "dt_bias_fp32": "dtb", "neg_exp_A_fp32": "nea", "norm_w_1d": "w"},
        Nv=6,
        Nk=2,
        Dk=128,
        Dv=128,
        qkvz_dim_tp=2048,
        K=4,
        scale=128**-0.5,
        _decode_step_fp32_state=fp32_state,
        _spec_max_users=0,
    )
    g._spec_hnew_depth = TPGatedDeltaNet._spec_hnew_depth
    return g


def _record_op(monkeypatch):
    import ttnn

    calls = []
    kda = types.SimpleNamespace(gdn_spec_step=lambda *a, **kw: calls.append((a, kw)) or "gated")
    monkeypatch.setattr(ttnn, "experimental", types.SimpleNamespace(kda=kda), raising=False)
    return calls


def test_lazy_flag_default_off_and_not_an_opt_bit(monkeypatch):
    from models.demos.blackhole.qwen36.tt.gdn.tp import (
        SPEC_STEP_OPT_KNOWN,
        SPEC_STEP_OPT_LAZY,
        gdn_spec_step_kernel_opts,
        gdn_spec_step_lazy_enabled,
    )

    monkeypatch.delenv("QWEN36_SPEC_STEP_LAZY", raising=False)
    assert not gdn_spec_step_lazy_enabled()
    monkeypatch.setenv("QWEN36_SPEC_STEP_LAZY", "1")
    assert gdn_spec_step_lazy_enabled()
    assert SPEC_STEP_OPT_LAZY == 256 and not (SPEC_STEP_OPT_KNOWN & SPEC_STEP_OPT_LAZY)
    monkeypatch.setenv("QWEN36_SPEC_STEP_OPT", "256")
    with pytest.raises(ValueError, match="QWEN36_SPEC_STEP_OPT"):
        gdn_spec_step_kernel_opts()


@pytest.mark.parametrize(
    "lazy,opt,T,want",
    [
        ("0", "250", 4, 250),  # flag off: kernel_opts exactly as before
        (None, None, 4, None),
        ("1", "250", 4, 250 | 256),
        ("1", None, 4, 256),
        ("1", "250", 1, 250),  # the T = 1 seed never gets the lazy bit
        ("1", None, 8, 256),
    ],
)
def test_lazy_bit_only_on_verify(monkeypatch, lazy, opt, T, want):
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

    for k, v in (("QWEN36_SPEC_STEP_LAZY", lazy), ("QWEN36_SPEC_STEP_OPT", opt)):
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    monkeypatch.delenv("QWEN36_SPEC_STEP_VALUE_BLOCKS", raising=False)
    calls = _record_op(monkeypatch)
    TPGatedDeltaNet._gdn_spec_step(_spec_step_layer(), "qkv", "wa", "wb", "ring", "ctrl", T, 8, max_users=8)
    ((_, kw),) = calls
    assert kw.get("kernel_opts") == want


def test_lazy_refuses_without_fp32_state_or_short_T(monkeypatch):
    from models.demos.blackhole.qwen36.tt.gdn.tp import TPGatedDeltaNet

    monkeypatch.setenv("QWEN36_SPEC_STEP_LAZY", "1")
    monkeypatch.delenv("QWEN36_SPEC_STEP_OPT", raising=False)
    _record_op(monkeypatch)
    with pytest.raises(AssertionError, match="fp32"):
        TPGatedDeltaNet._gdn_spec_step(_spec_step_layer(fp32_state=False), "q", "a", "b", "r", "c", 4, 8)
    with pytest.raises(AssertionError, match="T >= 3"):
        TPGatedDeltaNet._gdn_spec_step(_spec_step_layer(), "q", "a", "b", "r", "c", 2, 8)


def test_fresh_ctrl_page_words():
    """A FRESH row's ctrl words: mi = T (lazy_ref.fresh_mi) and the engine formula's block words (the op ignores their
    value except HOLD); every other row is unchanged."""
    from models.demos.blackhole.qwen36.tt.gdn.tp import spec_ctrl_page

    B, Nv, T = 8, 6, 4
    mi = [1, 3, lr.fresh_mi(T), 0, 2, 3, lr.fresh_mi(T), 1]
    page = spec_ctrl_page(mi, B, Nv, parity=1).reshape(-1)
    base = spec_ctrl_page([1, 3, 0, 0, 2, 3, 0, 1], B, Nv, parity=1).reshape(-1)
    assert int(page[0]) == 1 and [int(x) for x in page[1 : 1 + B]] == mi
    for u in range(B):
        blk = page[1 + B + u * Nv : 1 + B + (u + 1) * Nv]
        if mi[u] >= T:
            assert [int(x) for x in blk] == [(T * B + u) * Nv + h for h in range(Nv)]
            assert all(int(x) != -1 for x in blk)  # never the HOLD sentinel
        else:
            assert torch.equal(blk, base[1 + B + u * Nv : 1 + B + (u + 1) * Nv])


def test_lazy_refuses_device_slot_remap():
    """The stash keeps token t of row u at tile row / column (u*T) % 32, so a device slot remap (QWEN36_SPEC_SLOTMAP=0,
    remap_spec_slots) would hand a moved row another row's stash offsets: K4 refuses it. The host slot map (default) and
    the flag-off device remap are unchanged."""
    from models.demos.blackhole.qwen36.tt.spec_engine import MTPSpecEngine

    moved = []
    eng = MTPSpecEngine.__new__(MTPSpecEngine)
    eng.B, eng.host_slotmap, eng._lazy = 4, False, True
    eng._gdn = [types.SimpleNamespace(remap_spec_slots=moved.append)]
    with pytest.raises(ValueError, match="QWEN36_SPEC_SLOTMAP"):
        eng.remap_slots([1, 0, 2, 3])
    assert moved == []  # refused before any device work
    eng.remap_slots([0, 1, 2, 3])  # identity: still a no-op
    assert MTPSpecEngine._lazy is False  # engines built without __init__ (flag off) keep the r5 path

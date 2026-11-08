# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""K4 (QWEN36_SPEC_STEP_LAZY): pure-torch reference of gdn_spec_step's lazy-rollback ring contract (kernel_opts bit 8).

Per (user u, value head h) the op keeps, in the fp32 ring [T*M*Nv, Dk, Dv] (block (t*M + u)*Nv + h):
  slot 0       D: the state BEFORE the previous verify's T tokens (the seeds keep writing it, unchanged)
  slot 1 + p   the previous verify's STASH, p = the window parity that verify wrote (= the parity the next verify
               reads):
               tile page j < Kt   = K^T tile j (rows 32j.. of the normalized keys transposed; token t in column r0 + t)
               tile page Kt + v   = U tile v (the WY deltas; token t in row r0 + t)
               tile page Kt + Vt  = the per-token decay exp(g_t) at element (r0 + t, 0) (column 0 holds 0 elsewhere; the
                                    other columns of that page are unspecified)
with r0 = (u*T) % 32 (the user's rows in the verify tile). The NEXT verify replays the accepted prefix onto D,
  S' = (prod_{j<r} dec_j) D + sum_{i<r} (prod_{i<j<r} dec_j) k_i^T U_i,   r = mi + 1,
writes S' back to slot 0 and runs its own tokens from S'. A ctrl mi >= T means FRESH (slot 0 was just seeded or
restored: r = 0). The stash pages are a tile layout inside an ordinary [Dk, Dv] block, so a host copy of the ring
(TPGatedDeltaNet.save_spec_user) decodes with ``stash_from_block``.

Also the chunked (WY) form of one verify chunk, the serial recurrence it replaces, and the replay -- the math the
kernel implements (tests/test_gdn_lazy_host.py checks them against each other in fp64).
"""

import torch

TILE = 32


def fresh_mi(T):
    """The ctrl-page mi value that marks a FRESH row (nothing to replay): any value >= T; the engine uses T."""
    return int(T)


def _tile(block, page, Vt):
    """[32, 32] view of tile page ``page`` of a [Dk, Dv] block (tile (i, j) = page i*Vt + j)."""
    i, j = divmod(int(page), int(Vt))
    return block[i * TILE : (i + 1) * TILE, j * TILE : (j + 1) * TILE]


def stash_from_block(block, Kt, Vt):
    """Decode a stash block [Dk, Dv] (fp32, as ttnn.to_torch returns the ring) into (kt [Kt*32, 32] = K^T rows x token
    columns, u [32, Vt*32] = token rows x value columns, dec [32] = the decay per token row)."""
    kt = torch.cat([_tile(block, j, Vt) for j in range(Kt)], dim=0)
    u = torch.cat([_tile(block, Kt + v, Vt) for v in range(Vt)], dim=1)
    dec = _tile(block, Kt + Vt, Vt)[:, 0].clone()
    return kt, u, dec


def stash_to_block(kt, u, dec, Kt, Vt, dtype=torch.float32):
    """Inverse of stash_from_block (unused pages zero): the block a verify writes into its stash slot."""
    block = torch.zeros(Kt * TILE, Vt * TILE, dtype=dtype)
    for j in range(Kt):
        _tile(block, j, Vt).copy_(kt[j * TILE : (j + 1) * TILE])
    for v in range(Vt):
        _tile(block, Kt + v, Vt).copy_(u[:, v * TILE : (v + 1) * TILE])
    _tile(block, Kt + Vt, Vt)[:, 0] = dec
    return block


def replay_weights(dec, r0, r):
    """(c, w): c = prod_{j<r} dec[r0 + j]; w[i] = prod_{i<j<r} dec[r0 + j] for i < r, in dec's dtype. The kernel's
    writer computes the same products in the same order (descending i) in fp32 (an fp32 ``dec`` reproduces its
    rounding)."""
    w = [None] * r
    acc = torch.ones((), dtype=dec.dtype)
    for i in range(r - 1, -1, -1):
        w[i] = acc.clone()
        acc = acc * dec[r0 + i]
    return acc, w


def replay(D, kt, u, dec, r0, r):
    """S' = c D + K^T[:, r0:r0+r] diag(w) U[r0:r0+r, :] (r = 0: D). Shapes: D [Dk, Dv], kt [Dk, 32], u [32, Dv]."""
    if r == 0:
        return D.clone()
    c, w = replay_weights(dec, r0, r)
    wv = torch.stack([x.to(D.dtype) for x in w])
    return c.to(D.dtype) * D + kt[:, r0 : r0 + r].to(D.dtype) @ (wv[:, None] * u[r0 : r0 + r].to(D.dtype))


def chunk_wy(S, qn, kn, v, beta, dec):
    """One verify chunk in WY form from start state S. qn / kn [T, Dk] (l2-normalized, q scaled), v [T, Dv], beta / dec
    [T]. Returns (O [T, Dv], U [T, Dv]): O[t] = q_t S_t of the serial recurrence, U[t] its delta row."""
    T = qn.shape[0]
    dt = S.dtype
    G = torch.zeros(T, T, dtype=dt)
    E = torch.zeros(T, dtype=dt)
    e = torch.tensor(1.0, dtype=dt)
    for t in range(T):
        e = e * dec[t]
        E[t] = e
        g = torch.tensor(1.0, dtype=dt)
        G[t, t] = 1.0
        for i in range(t - 1, -1, -1):
            g = g * dec[i + 1]
            G[t, i] = g
    KS, QS = kn @ S, qn @ S
    KK, QK = kn @ kn.T, qn @ kn.T
    N = beta[:, None] * KK * (G - torch.eye(T, dtype=dt))
    rhs = beta[:, None] * (v - E[:, None] * KS)
    U = torch.linalg.solve(torch.eye(T, dtype=dt) + N, rhs)
    O = E[:, None] * QS + (QK * G) @ U
    return O, U


def chunk_serial(S, qn, kn, v, beta, dec):
    """The serial gated delta rule the WY form replaces (today's gdn_spec_step token loop). Returns (O [T, Dv],
    states [T, Dk, Dv] after each token)."""
    outs, states = [], []
    for t in range(qn.shape[0]):
        S = S * dec[t]
        delta = beta[t] * (v[t] - kn[t] @ S)
        S = S + torch.outer(kn[t], delta)
        outs.append(qn[t] @ S)
        states.append(S)
    return torch.stack(outs), torch.stack(states)


def parked_slot0_lazy(ring, s, T, par, r0, Kt, Vt):
    """The committed state of a parked row from its host ring copy under K4: ring [D*T, Nv, Dk, Dv] (save_spec_user;
    D devices), s = the accepted index of its last verify, par = the window parity at park time (the stash of that
    verify sits in slot 1 + par), r0 = (row*T) % 32. Returns [D, Nv, Dk, Dv] fp32: D replayed s + 1 tokens (the
    restored row then verifies FRESH, i.e. with the ctrl mi >= T)."""
    Dn = ring.shape[0] // T
    r4 = ring.reshape(Dn, T, *ring.shape[1:]).to(torch.float32)
    out = torch.empty(Dn, *ring.shape[1:], dtype=torch.float32)
    for d in range(Dn):
        for h in range(ring.shape[1]):
            kt, u, dec = stash_from_block(r4[d, 1 + par, h], Kt, Vt)
            out[d, h] = replay(r4[d, 0, h], kt, u, dec, r0, s + 1)
    return out

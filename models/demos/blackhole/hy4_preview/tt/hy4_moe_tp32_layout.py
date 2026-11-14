# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""X40 TP32-sharded routed experts for DECODE: tile layout + a torch emulation of the planned kernel dataflow (pure torch, no ttnn).

Chip c (0..31 over the whole 4x8 mesh) holds the intermediate columns [64 c, 64 c + 64) (2 tiles) of ALL 256 experts: gate/up [6144 x 64], down [64 x 6144].
Inside a chip the stream of every DRAM bank b (0..7, one worker core per bank) is the same size for every expert, so any routing set reads equal bytes per bank:
  gate/up stream of (bank b, expert e): K tiles [24 b, 24 b + 24) of the 6144 hidden rows, per K tile the 4 tiles (g0, g1, u0, u1)  -> 96 tiles = 55,296 B
  down stream of (bank b, expert e): output tiles [24 b, 24 b + 24) of the hidden dim in 6 sub-blocks of 4 tiles; per sub-block s the 2 K tiles x 4 N tiles -> 48 tiles = 27,648 B
bank b stores for all experts: W01[b] = [e][k_local 24][4]  (tile index e * 96 + k_local * 4 + j) and W2[b] = [s 6][e][k 2][4 n] (tile index (s * 256 + e) * 8 + k * 4 + n)
(a W2 read of sub-block s for one expert is 8 tiles = 4,608 B contiguous; every expert's gate/up read is 55,296 B contiguous).
Dataflow emulated: partial gate/up over the bank's K slice -> sum over the 8 banks (exchange) -> act = silu(min(g, L)) * clamp(u, -L, L) * score[t, e] -> down over the bank's output slice
(accumulated over the active experts) -> [T, 768] per bank = this chip's [T, 6144] partial; the 32 chips' partials are summed by the cross-chip reduce.
"""
import torch

HIDDEN, INTER, NEXP, TP, NB = 6144, 2048, 256, 32, 8
KT = HIDDEN // 32  # 192
KB = KT // NB  # 24 K tiles per bank
ICOL = INTER // TP  # 64 columns per chip
TILE_BYTES_BFP4 = 576
LIMIT = 10.0


def tiles(m):
    """[R, C] -> [R/32, C/32, 32, 32] tile grid."""
    r, c = m.shape
    return m.reshape(r // 32, 32, c // 32, 32).permute(0, 2, 1, 3)


def pack_chip(w0, w1, w2, c):
    """w0, w1: [E, 6144, 2048] (in, out); w2: [E, 2048, 6144].  Returns W01 [NB, E*96, 32, 32], W2 [NB, 6*E*8, 32, 32] tile streams of chip c."""
    E = w0.shape[0]
    sl = slice(ICOL * c, ICOL * (c + 1))
    w01 = torch.zeros(NB, E * KB * 4, 32, 32, dtype=w0.dtype)
    w2s = torch.zeros(NB, 6 * E * 8, 32, 32, dtype=w0.dtype)
    for e in range(E):
        g, u, d = tiles(w0[e][:, sl]), tiles(w1[e][:, sl]), tiles(w2[e][sl, :])  # [192, 2], [192, 2], [2, 192]
        for b in range(NB):
            for kl in range(KB):
                k = KB * b + kl
                base = (e * KB + kl) * 4
                w01[b, base + 0], w01[b, base + 1], w01[b, base + 2], w01[b, base + 3] = (
                    g[k, 0],
                    g[k, 1],
                    u[k, 0],
                    u[k, 1],
                )
            for s in range(6):
                for k in range(2):
                    for n in range(4):
                        w2s[b, ((s * E + e) * 8) + k * 4 + n] = d[k, KB * b + 4 * s + n]
    return w01, w2s


def emulate_chip(x, ids, scores, w01, w2s, limit=LIMIT, E=NEXP):
    """x [T, 6144]; ids/scores [T, 8]; returns this chip's partial [T, 6144] (fp32) following the kernel dataflow."""
    T = x.shape[0]
    xt = x.float().reshape(T, KT, 32)  # K tile k, 32 values
    active = sorted(set(ids.flatten().tolist()))
    # phase A: per bank partial over its K slice, [expert][T][4 tiles of 32]
    part = torch.zeros(NB, len(active), T, 4, 32)
    for b in range(NB):
        for ai, e in enumerate(active):
            for kl in range(KB):
                k = KB * b + kl
                for j in range(4):
                    part[b, ai, :, j] += xt[:, k] @ w01[b, (e * KB + kl) * 4 + j].float()
    gu = part.sum(0)  # exchange: sum over the 8 banks
    out = torch.zeros(T, HIDDEN)
    # act per (expert, token) with the routing score (0 when the token did not pick the expert)
    for ai, e in enumerate(active):
        s = torch.zeros(T)
        for t in range(T):
            for k in range(ids.shape[1]):
                if int(ids[t, k]) == e:
                    s[t] = scores[t, k].float()
        g = gu[ai, :, 0:2].reshape(T, 64)
        u = gu[ai, :, 2:4].reshape(T, 64)
        act = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit) * s[:, None]
        at = act.reshape(T, 2, 32)
        for b in range(NB):
            for sb in range(6):
                for n in range(4):
                    acc = torch.zeros(T, 32)
                    for k in range(2):
                        acc += at[:, k] @ w2s[b, ((sb * E + e) * 8) + k * 4 + n].float()
                    col = (KB * b + 4 * sb + n) * 32
                    out[:, col : col + 32] += acc
    return out


def reference(x, ids, scores, w0, w1, w2, limit=LIMIT):
    T = x.shape[0]
    y = torch.zeros(T, HIDDEN)
    for t in range(T):
        for k in range(ids.shape[1]):
            e = int(ids[t, k])
            g = x[t].float() @ w0[e].float()
            u = x[t].float() @ w1[e].float()
            a = torch.nn.functional.silu(g.clamp(max=limit)) * u.clamp(-limit, limit)
            y[t] += scores[t, k].float() * (a @ w2[e].float())
    return y


def bytes_per_chip(distinct_experts):
    return (
        distinct_experts * (KT * 4 + 2 * KT) * TILE_BYTES_BFP4
    )  # per expert: gate/up 768 tiles + down 384 tiles = 1152 tiles

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Operand probes of the fused row matmul: the inputs of each experiment and the arithmetic used to read the answers
(pure torch, no ttnn: the analysis runs without a device).

An experiment is a list of runs (w [32 kt, 32 ct], x [rows, 32 kt]) for one (weight format, x format) pair; the
device's FP32 sums of every run are stored next to the inputs by tests/test_probe.py (one .pt file per experiment and
fidelity) and read by analysis.py.

Reading a weight: x has one non-zero value per token row, so output row r is weight row j(r) times that value as the
multiplier sees both. Reading x: the weight is a 0 / 1 selection matrix (one 1.0 per output column)."""

from __future__ import annotations

import torch

TILE = 32
ROWS = 8


def one_hot_runs(n_k, value=1.0, rows=ROWS, order=None):
    """x of the runs that read every weight row once: run i, token row r has `value` at K index order[rows i + r].
    Returns ([runs, rows, n_k] x, [runs, rows] the K index of each token row)."""
    order = torch.arange(n_k) if order is None else order
    runs = -(-len(order) // rows)
    idx = torch.cat((order, order[: runs * rows - len(order)])).reshape(runs, rows)
    x = torch.zeros(runs, rows, n_k)
    x[torch.arange(runs)[:, None], torch.arange(rows)[None, :], idx] = value
    return x, idx


def gauss_weights(kt, ct, seed=4, scale=0.05):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(TILE * kt, TILE * ct, generator=g) * scale


def sweep_weights(kt, ct, bits, seed=7, pin=True, exponent=-4):
    """Every block of 16 (a row of a face: the unit that shares an exponent in the BFP formats) holds one pin
    (the largest magnitude of the format: all mantissa bits set, so the block's exponent is known) and 15 values
    m * 2^(exponent - bits + 1), m uniform in [0, 2^bits), random signs. bits: 7 for BFP8, 3 for BFP4. Without the
    pin the block's exponent is that of its largest value."""
    g = torch.Generator().manual_seed(seed)
    n_blocks = TILE * kt * 2 * ct
    top = 2**bits
    m = torch.randint(0, top, (n_blocks, 16), generator=g)
    if pin:
        where = torch.randint(0, 16, (n_blocks,), generator=g)
        m[torch.arange(n_blocks), where] = top - 1
    sign = torch.randint(0, 2, (n_blocks, 16), generator=g) * 2 - 1
    w = (sign * m).double() * 2.0 ** (exponent - bits + 1)
    return w.reshape(TILE * kt, 2 * ct * 16).float()


def isolate_rows(w, keep=5, period=16):
    """w with every row zeroed except those with (row % period) == keep: no other non-zero weight shares a face
    column with a kept one."""
    out = torch.zeros_like(w)
    out[keep::period] = w[keep::period]
    return out


def selection(n):
    """[n, n] identity: output column c reads x[:, c]."""
    return torch.eye(n)


def float_grid(sig_bits, exponents, seed=11, shape=(ROWS, 64), density=1.0):
    """Random values with `sig_bits` significant bits (leading one included, the last bit free), exponents drawn from
    `exponents`, random signs; a fraction 1 - density of the entries is zero."""
    g = torch.Generator().manual_seed(seed)
    n = 1
    for d in shape:
        n *= d
    m = torch.randint(2 ** (sig_bits - 1), 2**sig_bits, (n,), generator=g)
    e = torch.tensor(exponents)[torch.randint(0, len(exponents), (n,), generator=g)]
    sign = torch.randint(0, 2, (n,), generator=g) * 2 - 1
    v = (sign * m).double() * 2.0 ** (e - sig_bits + 1).double()
    keep = torch.rand(n, generator=g) < density
    return (v * keep).float().reshape(shape)


# ---------------------------------------------------------------------------------------------------------------
# reading helpers


def decompose(v):
    """|v| = m * 2^e with m in [1, 2): (sign, e, m) as doubles; zeros give (0, 0, 0)."""
    v = v.double()
    sign = torch.sign(v)
    a = v.abs()
    e = torch.floor(torch.log2(torch.where(a > 0, a, torch.ones_like(a))))
    m = torch.where(a > 0, a / 2.0**e, torch.zeros_like(a))
    return sign, e, m


def sig_bits(v, limit=24):
    """Number of significant bits of each value (0 for zero)."""
    _, _, m = decompose(v)
    k = (m * 2.0 ** (limit - 1)).round().long()
    out = torch.zeros_like(k)
    for b in range(limit):
        out = torch.where((k >> b) & 1 == 1, torch.maximum(out, torch.full_like(out, limit - b)), out)
    return out


def trunc_sig(v, bits):
    """v truncated toward zero to `bits` significant bits."""
    sign, e, m = decompose(v)
    q = torch.floor(m * 2.0 ** (bits - 1)) / 2.0 ** (bits - 1)
    return (sign * q * 2.0**e).float()


def round_sig(v, bits):
    """v rounded to nearest to `bits` significant bits, ties away from zero."""
    sign, e, m = decompose(v)
    q = torch.floor(m * 2.0 ** (bits - 1) + 0.5) / 2.0 ** (bits - 1)
    return (sign * q * 2.0**e).float()


def block_exponent(w):
    """Per value, the exponent of the largest magnitude of its block of 16 (a row of a face)."""
    a = w.double().abs().reshape(w.shape[0], -1, 16)
    top = a.max(dim=2, keepdim=True).values
    e = torch.floor(torch.log2(torch.where(top > 0, top, torch.ones_like(top))))
    return e.expand_as(a).reshape(w.shape)

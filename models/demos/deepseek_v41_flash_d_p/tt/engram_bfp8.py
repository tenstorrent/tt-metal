# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Engram rows as bfloat8_b TILE bytes straight from the table's 8-bit bytes: integer operations only, no float.

A table value is an e4m3 byte times its group's e8m0 power of two (32 values a scale). bfloat8_b (tt-metal
blockfloat_common.cpp convert_u32_to_bfp, round to nearest even) keeps, per 16 consecutive values of a row, the
largest fp32 exponent and per value a sign + 7-bit magnitude: the value's 24-bit significand shifted down by its
exponent's gap to the shared one, rounded to 7 bits, capped at 127. An e4m3 value has a 4-bit significand q (8..15
after normalizing), so its magnitude is RNE(q * 2^(3 - gap)): a lookup on (byte, gap). The output bytes equal
ttnn's packing of the same values as floats (tests/test_engram_bfp8.py), so the device sees what
from_torch(dtype=bfloat8_b) would give it.

TILE bytes per 32 x 32 tile: 64 exponent bytes (face 0..3, 16 rows each) then 1024 sign-magnitude bytes (face 0..3,
row-major 16 x 16); tiles row-major over the tensor's [rows / 32, cols / 32].

Packed table rows (ENGRAM_TABLE=resident with ENGRAM_ROWS=bfp8): each table row as 16 exponent bytes then its 256
magnitude bytes (272 B); ``tile_rows`` gathers a chunk's rows from them straight into TILE bytes.
"""

from __future__ import annotations

import torch

_GAPS = 32  # a gap past 10 already rounds every e4m3 significand to 0


def _tables():
    b = torch.arange(256)
    sign = (b >> 7) & 1
    e = (b >> 3) & 0xF
    m = b & 7
    # normal: value = (8 + m) / 8 * 2^(e - 7); subnormal (e = 0): m / 8 * 2^-6 -> normalize m into 8..15
    q = torch.where(e > 0, 8 + m, m)
    exp = torch.where(e > 0, e - 7, torch.full_like(e, -6))  # exponent of q / 8 (before normalizing subnormals)
    for _ in range(3):
        small = (e == 0) & (q > 0) & (q < 8)
        q = torch.where(small, q << 1, q)
        exp = torch.where(small, exp - 1, exp)
    zero = q == 0  # +-0 (e4m3fn has no inf; 0x7f / 0xff are NaN and never stored)
    # magnitude for a gap g: RNE(q * 2^(3 - g)), capped at 127
    g = torch.arange(_GAPS).view(1, -1)
    qq = q.view(-1, 1)
    sh = (g - 3).clamp(min=0)
    num = qq << (3 - g).clamp(min=0)
    half = torch.where(sh > 0, 1 << (sh - 1).clamp(min=0), torch.zeros_like(sh))
    rem = qq & ((1 << sh) - 1)
    mag = num >> sh
    up = (sh > 0) & ((rem > half) | ((rem == half) & ((mag & 1) == 1)))
    mag = (mag + up.long()).clamp(max=127)
    mag = torch.where(zero.view(-1, 1), torch.zeros_like(mag), mag)
    byte = torch.where(mag > 0, (sign.view(-1, 1) << 7) | mag, torch.zeros_like(mag)).to(torch.uint8)
    return exp.to(torch.int16), zero, byte  # exp: fp32 exponent minus 127 minus the e8m0 bias, before the scale


_EXP, _ZERO, _BYTE = _tables()


def _run(threads, kernel, *args):
    """A numba parallel kernel with torch's intra-op thread count kept (FIX.10, DS41-133): numba's omp threading layer
    sets the OpenMP max-threads ICV to NUMBA_NUM_THREADS (every logical CPU: 64 on the Galaxy host) at its first
    launch, and torch reads its pool size from the same ICV (32 -> 64). Every torch op after it then ran 5-20x slower
    on the host (the fused head's per-layer weight layout ~112 s a layer instead of ~6 s; tt/perf_engram_host.py
    test_numba_keeps_torch_threads)."""
    import numba

    n = torch.get_num_threads()
    try:
        if threads:  # (set_num_threads launches the pool too)
            numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
        return kernel(*args)
    finally:
        if torch.get_num_threads() != n:
            torch.set_num_threads(n)


def _numba_kernel():
    import numba
    import numpy as np

    @numba.njit(parallel=True, cache=True, nogil=True)  # (nogil: the background build, FIX.10)
    def pack(w, s, exps, mags, exp_t, zero_t, byte_t):
        n, d = w.shape
        per_scale = d // s.shape[1]
        for r in numba.prange(n):
            for k in range(d // 16):
                sc = np.int32(s[r, (k * 16) // per_scale])
                mx = 0
                for j in range(16):
                    b = w[r, k * 16 + j]
                    if not zero_t[b]:
                        mx = max(mx, exp_t[b] + sc)
                exps[r, k] = mx
                for j in range(16):
                    b = w[r, k * 16 + j]
                    mags[r, k * 16 + j] = 0 if zero_t[b] else byte_t[b, min(mx - (exp_t[b] + sc), _GAPS - 1)]

    return pack


_KERNEL = None


def pack_rows(w: torch.Tensor, s: torch.Tensor, threads: int | None = None):
    """w [N, 256] uint8 e4m3 bytes, s [N, 8] uint8 e8m0 bytes -> (exponents [N, 16] uint8, magnitudes [N, 256] uint8):
    the bfloat8_b shared exponent of each 16-value group and each value's sign + 7-bit magnitude. A compiled loop over
    ``threads`` (QB: 5120-token chunk 63 ms on 1 thread, 8 ms on 8; the torch form below: ~230 ms)."""
    global _KERNEL

    if _KERNEL is None:
        _KERNEL = _numba_kernel()
    n, d = w.shape
    exps = torch.empty(n, d // 16, dtype=torch.uint8)
    mags = torch.empty(n, d, dtype=torch.uint8)
    _run(
        threads,
        _KERNEL,
        w.contiguous().numpy(),
        s.contiguous().numpy(),
        exps.numpy(),
        mags.numpy(),
        _EXP.int().numpy(),
        _ZERO.numpy(),
        _BYTE.numpy(),
    )
    return exps, mags


def pack_rows_torch(w: torch.Tensor, s: torch.Tensor):
    """pack_rows in torch tensor operations (the readable form; the tests check the two agree)."""
    n, d = w.shape
    wi = w.long()
    e = _EXP[wi].int() + s.int().repeat_interleave(d // s.shape[1], dim=1)  # fp32 biased exponent (127 + ...)
    e = torch.where(_ZERO[wi], torch.zeros_like(e), e)
    eg = e.view(n, d // 16, 16)
    shared = eg.amax(-1)
    gap = (shared.unsqueeze(-1) - eg).clamp(max=_GAPS - 1).view(n, d)
    return shared.to(torch.uint8), _BYTE[wi, gap.long()]


def pack_into(w: torch.Tensor, s: torch.Tensor, out, threads: int | None = None):
    """pack_rows of w [N, 256], s [N, 8] written into packed rows out [N, 272] (numpy uint8: exponents then
    magnitudes)."""
    global _KERNEL

    if _KERNEL is None:
        _KERNEL = _numba_kernel()
    g = w.shape[1] // 16
    _run(
        threads, _KERNEL, w.numpy(), s.numpy(), out[:, :g], out[:, g:], _EXP.int().numpy(), _ZERO.numpy(), _BYTE.numpy()
    )


def _tile_kernel():
    import numba

    @numba.njit(parallel=True, cache=True, nogil=True)  # (nogil: the background build, FIX.10)
    def tile(packed, ids, valid, out, n):
        g = packed.shape[1] // 17  # 16-value groups a table row
        slots = ids.shape[1]
        tcols = slots * g // 2  # 32-column tiles across [n, slots * g * 16]
        for t in numba.prange(n):
            tr, ri = t // 32, t % 32
            fr, rr = ri // 16, ri % 16
            for j in range(slots):
                rid = ids[t, j] if t < valid else -1
                for k in range(g):
                    c0 = (j * g + k) * 16
                    f = fr * 2 + (c0 % 32) // 16
                    base = (tr * tcols + c0 // 32) * 1088
                    mo = base + 64 + f * 256 + rr * 16
                    if rid < 0:
                        out[base + f * 16 + rr] = 0
                        for v in range(16):
                            out[mo + v] = 0
                    else:
                        out[base + f * 16 + rr] = packed[rid, k]
                        for v in range(16):
                            out[mo + v] = packed[rid, g + k * 16 + v]

    return tile


_TILE = None


def tile_rows(packed, ids: torch.Tensor, n: int, threads: int | None = None, out=None) -> torch.Tensor:
    """packed table rows [N, 272] (numpy), ids [valid, slots] (valid <= n) -> the bfloat8_b TILE bytes of the chunk
    rows [n, slots * 256] (rows past ``valid`` 0): tile_bytes of those rows' exponents and magnitudes, in one pass.
    ``out``: a uint8 tensor of that size to write into (ENGRAM_ASYNC's staging bytes; every byte is written)."""
    global _TILE

    if _TILE is None:
        _TILE = _tile_kernel()
    w = ids.shape[1] * (packed.shape[1] // 17) * 16
    if out is None:
        out = torch.empty(n * w * 17 // 16, dtype=torch.uint8)
    assert out.dtype == torch.uint8 and out.numel() == n * w * 17 // 16, (out.dtype, out.numel(), n, w)
    _run(threads, _TILE, packed, ids.contiguous().numpy(), ids.shape[0], out.numpy(), n)
    return out


def tile_bytes(exps: torch.Tensor, mags: torch.Tensor) -> torch.Tensor:
    """exps [S, W / 16] uint8, mags [S, W] uint8 (S, W multiples of 32) -> the bfloat8_b TILE bytes of [S, W]."""
    s, w = mags.shape
    tr, tc = s // 32, w // 32
    # mags [tr, fr(2), 16 rows, tc, fc(2), 16 cols] -> [tr, tc, fr, fc, 16, 16]
    m = mags.view(tr, 2, 16, tc, 2, 16).permute(0, 3, 1, 4, 2, 5).reshape(tr, tc, 1024)
    x = exps.view(tr, 2, 16, tc, 2).permute(0, 3, 1, 4, 2).reshape(tr, tc, 64)
    return torch.cat([x, m], dim=-1).reshape(-1)

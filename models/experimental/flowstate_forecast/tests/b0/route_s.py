# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""B0 route S: the LTI convolution as a chunked complex scan in fp32 stock ops (CONTRACT §6 B0).

Time-last layout (B, P, n, c), n = L / c chunks. Intra-chunk x = Lbar^t (cumsum_t (Lbar^-t Bu)) with ttnn.cumsum
(fp32 SFPU, not the tri-matmul: the FPU takes fp32 at tf32 at best); the chunk carry h_j = Lbar^c h_{j-1} + x_end_j
by a log-depth doubling scan over n (|Lbar^(c 2^k)| <= 1, no overflow); then x += Lbar^(t+1) h_{j-1}.
Modes where |Lbar|^-(c-1) max|Bu| would overflow take an 8-tap FIR instead (|Lbar|^8 < 2e-8 for them); they are
zeroed in the scaled tables. All fp32; complex values as (re, im) pairs.
"""

import math

import torch
import ttnn

FIR_TAPS = 8


def tables(device, lbar, c, n, bu_absmax=None, limit=1e30):
    """Host-built per (s, layer): powers by repeated multiplication in complex64 (route S's own construction)."""
    P = lbar.shape[0]
    pw = torch.cumprod(torch.cat((torch.ones(1, P, dtype=torch.complex64), lbar.expand(c, P))), 0)  # 0..c
    inv = torch.cumprod(torch.cat((torch.ones(1, P, dtype=torch.complex64), (1 / lbar).expand(c - 1, P))), 0)
    amax = 1.0 if bu_absmax is None else bu_absmax
    bounded = (-(c - 1) * lbar.abs().double().log() + math.log(max(amax, 1e-30))) >= math.log(limit)
    inv = torch.where(bounded, 0, inv)
    pw_t = torch.where(bounded, 0, pw[:c])
    pw_carry = torch.where(bounded, 0, pw[1:])
    steps = []
    a = pw[c]  # Lbar^c
    d = 1
    while d < n:
        steps.append((d, a.clone()))
        a = a * a
        d *= 2

    def put(t, shape):
        t = t.T.reshape(shape).contiguous()
        return [
            ttnn.from_torch(x.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
            for x in (t.real, t.imag)
        ]

    tb = {
        "inv": put(inv, (1, P, 1, c)),
        "pw": put(pw_t, (1, P, 1, c)),
        "pwc": put(pw_carry, (1, P, 1, c)),
        "steps": [(d, put(a.unsqueeze(0), (1, P, 1))) for d, a in steps],
        "bounded": bounded,
    }
    if bounded.any():
        taps = torch.cumprod(torch.cat((torch.ones(1, P, dtype=torch.complex64), lbar.expand(FIR_TAPS - 1, P))), 0)
        taps = torch.where(bounded, taps, 0)
        tb["fir"] = [put(taps[k : k + 1], (1, P, 1)) for k in range(FIR_TAPS)]
    return tb


def cmul(ar, ai, br, bi):
    return (
        ttnn.subtract(ttnn.multiply(ar, br), ttnn.multiply(ai, bi)),
        ttnn.add(ttnn.multiply(ar, bi), ttnn.multiply(ai, br)),
    )


def shift_last(t, d):
    """t[..., j] -> t[..., j - d], zeros in front (along the last dim)."""
    shape = list(t.shape)
    lead, n = shape[:-1], shape[-1]
    zeros = ttnn.zeros((*lead, d), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=t.device())
    return ttnn.concat([zeros, ttnn.slice(t, [0] * len(shape), [*lead, n - d])], dim=-1)


def conv(bu_re, bu_im, tb, c):
    """bu (L, B, P) fp32 TILE -> x (L, B, P) fp32 TILE. L must be a multiple of c (the caller pads)."""
    L, B, P = bu_re.shape
    n = L // c
    br, bi = [ttnn.reshape(ttnn.permute(t, (1, 2, 0)), (B, P, n, c)) for t in (bu_re, bu_im)]
    sr, si = cmul(br, bi, *tb["inv"])
    sr, si = ttnn.cumsum(sr, dim=-1), ttnn.cumsum(si, dim=-1)
    xr, xi = cmul(sr, si, *tb["pw"])
    # chunk carry: inclusive scan of the chunk ends, then shifted by one chunk
    er = ttnn.reshape(ttnn.slice(xr, (0, 0, 0, c - 1), (B, P, n, c)), (B, P, n))
    ei = ttnn.reshape(ttnn.slice(xi, (0, 0, 0, c - 1), (B, P, n, c)), (B, P, n))
    for d, (ar, ai) in tb["steps"]:
        pr, pi = cmul(shift_last(er, d), shift_last(ei, d), ar, ai)
        er, ei = ttnn.add(er, pr), ttnn.add(ei, pi)
    hr, hi = [ttnn.reshape(shift_last(t, 1), (B, P, n, 1)) for t in (er, ei)]
    cr, ci = cmul(hr, hi, *tb["pwc"])
    xr, xi = ttnn.add(xr, cr), ttnn.add(xi, ci)
    if "fir" in tb:  # bounded modes: x_t = sum_{k < 8} Lbar^k Bu_{t-k}
        fr = ttnn.reshape(br, (B, P, L))
        fi = ttnn.reshape(bi, (B, P, L))
        accr, acci = cmul(fr, fi, *tb["fir"][0])
        for k in range(1, FIR_TAPS):
            pr, pi = cmul(shift_last(fr, k), shift_last(fi, k), *tb["fir"][k])
            accr, acci = ttnn.add(accr, pr), ttnn.add(acci, pi)
        xr = ttnn.add(xr, ttnn.reshape(accr, (B, P, n, c)))
        xi = ttnn.add(xi, ttnn.reshape(acci, (B, P, n, c)))
    return [ttnn.permute(ttnn.reshape(t, (B, P, L)), (2, 0, 1)) for t in (xr, xi)]

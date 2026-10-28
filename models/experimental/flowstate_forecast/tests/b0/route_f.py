# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""B0 route F: the LTI convolution as ttnn.experimental.fft / ifft at a power-of-two N >= 2L - 1 (CONTRACT §6 B0).

Bu arrives as the B projection leaves it, (L, B, 512) fp32 TILE, real and imaginary parts. The chain: permute to
time-last, row-major, zero-pad to N, fft (precise), multiply by the cached spectrum K^ = fft(K, N) broadcast over B,
ifft, slice [:L], back to (L, B, 512) TILE. All fp32.
"""

import torch
import ttnn


def pow2_n(L):
    return 1 << (2 * L - 2).bit_length()


def spectrum(device, K, N):
    """Cached per (s, L, layer): K (L, P) complex64 -> (re, im) each (1, P, N) fp32 row-major on device."""
    Kh = torch.fft.fft(K, n=N, dim=0).T.unsqueeze(0).contiguous()
    put = lambda t: ttnn.from_torch(t.float(), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return put(Kh.real), put(Kh.imag)


def conv(bu_re, bu_im, kh, N, cmul="multiply"):
    """bu_re / bu_im (L, B, P) fp32 TILE -> x_re / x_im (L, B, P) fp32 TILE."""
    L, B, P = bu_re.shape
    parts = []
    for t in (bu_re, bu_im):
        t = ttnn.permute(t, (1, 2, 0))  # (B, P, L)
        t = ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)
        t = ttnn.pad(t, [(0, 0), (0, 0), (0, N - L)], 0.0)
        parts.append(ttnn.reshape(t, (B * P, N)))
    sr, si = ttnn.experimental.fft(parts[0], parts[1])
    sr, si = ttnn.reshape(sr, (B, P, N)), ttnn.reshape(si, (B, P, N))
    kr, ki = kh
    if cmul == "multiply":
        pr = ttnn.subtract(ttnn.multiply(sr, kr), ttnn.multiply(si, ki))
        pi = ttnn.add(ttnn.multiply(sr, ki), ttnn.multiply(si, kr))
    else:  # complex_mul needs equal shapes and rows <= 1024: kh pre-expanded to (B, P * N / 1024, 1024)
        rows = (B, P * N // 1024, 1024)
        pr, pi = ttnn.experimental.complex_mul(ttnn.reshape(sr, rows), ttnn.reshape(si, rows), kr, ki)
    xr, xi = ttnn.experimental.ifft(ttnn.reshape(pr, (B * P, N)), ttnn.reshape(pi, (B * P, N)))
    out = []
    for t in (xr, xi):
        t = ttnn.slice(ttnn.reshape(t, (B, P, N)), (0, 0, 0), (B, P, L))
        t = ttnn.to_layout(t, ttnn.TILE_LAYOUT)
        out.append(ttnn.permute(t, (2, 0, 1)))
    return out

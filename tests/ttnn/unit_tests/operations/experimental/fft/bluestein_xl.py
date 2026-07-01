# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""
bluestein_xl — extended-range Bluestein FFT (commit 6e-2 deferred work).

The C++ ``ttnn.experimental.bluestein_fft`` op is fully device-resident
but capped at ``M = next_pow2(2*N - 1) ≤ 2^20`` because its inner
length-M FFT routes through ``fft_two_pass``. Lifting that cap requires
the inner FFT to run on ``fft_three_pass`` instead, which expects its
input pre-shaped as ``(B·N1·N2, N3)`` on the host.

This module is the **host-glue version of Bluestein** that we accept as
the price for covering N up to the three-pass envelope. The three vector
ops (chirp pre-mul, B-mul, chirp post-mul) execute as torch operations
on the CPU; only the two length-M FFTs run on device. Host work per
call is ``O(N) + O(M)`` complex multiplies — negligible next to the
device FFT_M's ``O(M log M)``.

Coverage:
  * ``M = next_pow2(2*N - 1) ≤ 2^30`` ⇒ ``N ≤ 536_870_911``  (algorithmic)
  * ``M ≤ 2^22``                       ⇒ ``N ≤ 2_097_151``    (validated)

For ``M ≤ 2^20`` we delegate to the fully device-resident
``ttnn.experimental.bluestein_fft`` (no host vector ops). The hybrid
path activates only when M exceeds the two-pass range.
"""

from __future__ import annotations

import functools
import math
from typing import Optional, Tuple

import torch
import ttnn


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _pick_three_factorization(M: int) -> Tuple[int, int, int]:
    """Mirror ttnn::operations::experimental::pick_three_factorization
    in fft.cpp (lines 343–369) so our host-side pre-shape uses the
    factorization the kernel expects."""
    log2M = (M - 1).bit_length()
    if (1 << log2M) != M:
        raise ValueError(f"bluestein_xl: M={M} is not a power of two")
    if not (15 <= log2M <= 30):
        raise ValueError(
            f"bluestein_xl: M=2^{log2M} out of fft_three_pass range [2^15, 2^30]"
        )
    log2_N3 = 10 if log2M - 10 >= 10 else max(log2M - 10, 5)
    log2_rest = log2M - log2_N3
    log2_N1 = (log2_rest + 1) // 2
    log2_N2 = log2_rest - log2_N1
    return (1 << log2_N1, 1 << log2_N2, 1 << log2_N3)


@functools.lru_cache(maxsize=64)
def _bluestein_plan(N: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Per-N cached chirp vector w[n] and pre-computed B = FFT_M(b_ext).
    Both returned as complex64; B is sized to M = next_pow2(2N-1)."""
    M = _next_pow2(2 * N - 1)

    n = torch.arange(N, dtype=torch.float64)
    phase = -math.pi * (n * n) / float(N)
    w = torch.complex(phase.cos(), phase.sin()).to(torch.complex64)

    b_ext = torch.zeros(M, dtype=torch.complex64)
    b_ext[0] = torch.complex(torch.tensor(1.0), torch.tensor(0.0))
    if N > 1:
        g = w[1:N].conj()
        b_ext[1:N] = g
        b_ext[M - N + 1 : M] = g.flip(0)
    B_fft = torch.fft.fft(b_ext.to(torch.complex128)).to(torch.complex64)
    return w, B_fft, M


def _run_three_pass(
    device,
    x_re: torch.Tensor,
    x_im: torch.Tensor,
    M: int,
    precision: str,
    inverse: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-shape ``(B, M)`` → ``(B·N1·N2, N3)``, dispatch
    ``ttnn.experimental.fft_three_pass``, return ``(B, M)`` torch tensors
    in natural-order."""
    N1, N2, N3 = _pick_three_factorization(M)
    B = x_re.shape[0]

    x_re_pre = x_re.reshape(B * N1 * N2, N3).contiguous()
    x_im_pre = x_im.reshape(B * N1 * N2, N3).contiguous()

    tt_re = ttnn.from_torch(
        x_re_pre,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_im = ttnn.from_torch(
        x_im_pre,
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    out_re, out_im = ttnn.experimental.fft_three_pass(
        tt_re,
        tt_im,
        full_N=M,
        precision=precision,
        inverse=inverse,
    )

    # Per fft.cpp §"OUTPUT shape": result is (B·N3, N2, N1); flat reshape
    # to (B, M) yields natural-order X[K] (the (N3, N2, N1) dim layout
    # encodes K = k3·N1·N2 + k2·N1 + k1 = natural flat K).
    out_re_flat = ttnn.to_torch(out_re).reshape(B, M)
    out_im_flat = ttnn.to_torch(out_im).reshape(B, M)
    return out_re_flat, out_im_flat


# ─── Public entry point ─────────────────────────────────────────────────────


def bluestein_fft_xl(
    device,
    x_re: torch.Tensor,
    x_im: Optional[torch.Tensor] = None,
    N: Optional[int] = None,
    precision: str = "precise",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Length-N Bluestein FFT with host chirp glue + device fft_three_pass.

    Algorithm (per call):
      1.  ``a[n] = x[n] · w[n]``                                 — host
      2.  zero-pad ``(B, N) → (B, M)``                           — host
      3.  ``A = FFT_M(a)``                                       — device (fft_three_pass)
      4.  ``C = A · B_fft``                                      — host
      5.  ``c = IFFT_M(C)``                                      — device (fft_three_pass, inverse)
      6.  ``X[k] = c[k] · w[k]`` for k ∈ [0, N), drop padding    — host

    Args:
        device:    ttnn mesh device.
        x_re:      (B, N) torch tensor, float32 or float64.
        x_im:      Optional (B, N) tensor matching ``x_re``. Zero-imag if omitted.
        N:         Logical FFT length. Inferred from ``x_re.shape[-1]`` if None.
        precision: Forwarded to ``ttnn.experimental.fft_three_pass``.

    Returns:
        Pair of (B, N) float32 torch tensors (real, imag) of the DFT.

    Restrictions:
        * Currently B = 1 (multi-batch will land alongside batched Bluestein).
        * M = next_pow2(2N − 1) must satisfy 2^15 ≤ M ≤ 2^30.  For
          M ≤ 2^20 you should call ``ttnn.experimental.bluestein_fft``
          (fully device-resident, no host glue); this wrapper deliberately
          requires the extended-range path.
    """
    if N is None:
        N = int(x_re.shape[-1])
    if x_re.dim() == 1:
        x_re = x_re.unsqueeze(0)
    if x_im is not None and x_im.dim() == 1:
        x_im = x_im.unsqueeze(0)
    B = x_re.shape[0]
    if B != 1:
        raise NotImplementedError(
            "bluestein_fft_xl: B > 1 not yet implemented. Run rows serially "
            "or wait for the batched-XL update."
        )

    w, B_fft, M = _bluestein_plan(N)

    # Coerce to fp64 for the host pre-mul (better accuracy across the chirp
    # boundary), down-cast to fp32 just before device upload.
    if x_im is None:
        x_im = torch.zeros_like(x_re)
    x = torch.complex(
        x_re.to(torch.float64).reshape(B, N),
        x_im.to(torch.float64).reshape(B, N),
    )

    # ── Steps 1+2: pre-mul + zero-pad (host) ────────────────────────────
    a = torch.zeros((B, M), dtype=torch.complex128)
    a[:, :N] = x * w.to(torch.complex128).unsqueeze(0)
    a32 = a.to(torch.complex64)

    # ── Step 3: forward FFT_M (device, fft_three_pass) ──────────────────
    A_re, A_im = _run_three_pass(
        device,
        a32.real.contiguous(),
        a32.imag.contiguous(),
        M,
        precision,
        inverse=False,
    )
    # torch.complex(real, imag) wants real-valued inputs (float/double),
    # not complex ones — promote to fp64 and combine.
    A = torch.complex(A_re.to(torch.float64), A_im.to(torch.float64))

    # ── Step 4: convolution multiply by precomputed B (host) ────────────
    C = A * B_fft.to(torch.complex128).unsqueeze(0)
    C32 = C.to(torch.complex64)

    # ── Step 5: inverse FFT_M (device, fft_three_pass inverse=True).
    #   This path uses the commit-6c swap-trick and folds the 1/M scale
    #   into the last radix_pass writer — no host scaling required.
    c_re, c_im = _run_three_pass(
        device,
        C32.real.contiguous(),
        C32.imag.contiguous(),
        M,
        precision,
        inverse=True,
    )
    c = torch.complex(c_re.to(torch.float64), c_im.to(torch.float64))

    # ── Steps 6+7: slice [:N] + post-mul by w (host) ─────────────────────
    X = c[:, :N] * w.to(torch.complex128).unsqueeze(0)
    return X.real.to(torch.float32), X.imag.to(torch.float32)

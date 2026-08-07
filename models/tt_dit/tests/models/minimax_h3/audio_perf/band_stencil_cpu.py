"""Validate the fused band's math on CPU before writing a line of kernel C++.

FUSED_BAND_DESIGN.md formulates the band as a per-channel 1-D nonlinear stencil that never
materialises the 2x upsampled tensor or the post-activation tensor:

    u[2n]   = sum_j h[2j]   * x[n-j+off]        polyphase upsample, even phase
    u[2n+1] = sum_j h[2j+1] * x[n-j+off]        odd phase
    s       = u + inv_beta * sin(alpha*u)^2     snake, pointwise, per channel
    y[n]    = sum_k g[k] * s[2n-k+off]          stride-2 downsample

If that algebra is wrong the kernel cannot be right, and a CPU check costs nothing next to a 40-minute
rebuild. So this implements the stencil literally -- computing only the `u` values each output needs,
exactly as the kernel will -- and compares against the literal band (build the whole 2x tensor, apply
snake to all of it, filter it down) in float64.

Also pins the trap the design calls out: replicate padding does not decompose per phase, because the
pad region's parity alternates. The literal form pads once at the input; the stencil must reproduce
that, not pad each phase independently.
"""

import numpy as np

from models.tt_dit.layers.audio_ops import _make_kaiser_sinc_kernel_1d

K = 12
RATIO = 2


def snake(u, alpha, inv_beta):
    return u + inv_beta * np.sin(alpha * u) ** 2


def literal_band(x, h, g, alpha, inv_beta):
    """The band as written today: zero-stuff, filter, activate everything, filter down."""
    T, C = x.shape
    pad = K // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")  # replicate, once, at the input

    # upsample: zero-stuff by RATIO then FIR with the ratio-scaled taps
    stuffed = np.zeros(((len(xp)) * RATIO, C))
    stuffed[::RATIO] = xp
    u = np.zeros((len(stuffed) - K + 1, C))
    for n in range(u.shape[0]):
        u[n] = np.tensordot(h[::-1], stuffed[n : n + K], axes=(0, 0))

    s = snake(u, alpha, inv_beta)

    # downsample: stride-2 FIR
    y = np.zeros(((s.shape[0] - K) // RATIO + 1, C))
    for n in range(y.shape[0]):
        y[n] = np.tensordot(g[::-1], s[RATIO * n : RATIO * n + K], axes=(0, 0))
    return u, s, y


def stencil_band(x, h, g, alpha, inv_beta, n_out):
    """The kernel's form: for each output, compute only the `u` values it needs."""
    T, C = x.shape
    pad = K // 2
    xp = np.pad(x, ((pad, pad), (0, 0)), mode="edge")
    # Derivation. u[n] = sum_i h[K-1-i] * stuffed[n+i], and stuffed[m] is xp[m/2] for even m, else 0.
    # So only the i with (n+i) even contribute:
    #   n even -> i even, i=2j -> u[n] = sum_j h[K-1-2j] * xp[n/2 + j]        (odd-indexed taps, reversed)
    #   n odd  -> i odd,  i=2j+1 -> u[n] = sum_j h[K-2-2j] * xp[(n+1)/2 + j]  (even-indexed taps, reversed)
    # The phase-to-tap-subset mapping is the opposite of the obvious guess, which is what the first
    # version of this file got wrong -- and why it is checked here and not in a kernel.
    taps_even_n = h[1::2][::-1]  # h[K-1], h[K-3], ...
    taps_odd_n = h[0::2][::-1]  # h[K-2], h[K-4], ...

    def u_at(m):
        """u[m] straight from x, with no stuffed tensor ever built."""
        if m % RATIO == 0:
            taps, base = taps_even_n, m // RATIO
        else:
            taps, base = taps_odd_n, (m + 1) // RATIO
        acc = np.zeros(C)
        for j, t in enumerate(taps):
            idx = base + j
            if 0 <= idx < len(xp):
                acc += t * xp[idx]
        return acc

    y = np.zeros((n_out, C))
    for n in range(n_out):
        acc = np.zeros(C)
        for k in range(K):
            m = RATIO * n + (K - 1 - k)
            acc += g[k] * snake(u_at(m), alpha, inv_beta)
        y[n] = acc
    return y


def main():
    rng = np.random.default_rng(0)
    T, C = 64, 4
    x = rng.standard_normal((T, C)) * 0.3
    h = np.array(_make_kaiser_sinc_kernel_1d(0.5 / RATIO, 0.6 / RATIO, K), dtype=np.float64) * RATIO
    g = np.array(_make_kaiser_sinc_kernel_1d(0.5 / RATIO, 0.6 / RATIO, K), dtype=np.float64)
    alpha = rng.uniform(0.5, 1.5, C)
    inv_beta = 1.0 / (rng.uniform(0.5, 1.5, C) + 1e-9)

    u, s, y_ref = literal_band(x, h, g, alpha, inv_beta)
    print(f"literal: x{x.shape} -> u{u.shape} -> y{y_ref.shape}")

    y_sten = stencil_band(x, h, g, alpha, inv_beta, y_ref.shape[0])
    d = float(np.abs(y_sten - y_ref).max())
    rel = float(np.sqrt(np.mean((y_sten - y_ref) ** 2)) / np.std(y_ref))
    print(f"stencil vs literal: maxdiff {d:.3e}   rel_rmse {rel:.3e}")
    print("MATCH" if d < 1e-9 else "MISMATCH -- the index algebra is wrong, fix before writing C++")


if __name__ == "__main__":
    main()

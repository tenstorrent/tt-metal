"""CPU check of the Activation1d band decomposition, before implementing it on device.

Reference band:  z = interleave(ph0, ph1);  s = act(z);  y[t] = sum_k tap[k] * replicate_pad(s)[2t+k]

Claim: because `act` is pointwise, the 2x tensor need never exist. Splitting the downsample sum by the
parity of k gives two stride-1 FIRs over half-length phase signals:

    y[t] = sum_a tap[2a] * P0[t+a] + sum_a tap[2a+1] * P1[t+a]

where P0/P1 are the even/odd samples of the replicate-padded interleaved signal. The subtlety this
script exists to check: replicate padding does NOT decompose into per-phase replicate padding. The pad
region is a constant (s[0] or s[-1]) whose parity alternates, so P0's left pad is built from s0[0] --
the first sample of the *other* phase -- not from s1[0].
"""

import torch


def reference(ph0, ph1, taps, act):
    M = ph0.shape[0]
    z = torch.empty(2 * M, dtype=torch.float64)
    z[0::2] = ph0
    z[1::2] = ph1
    s = act(z)
    K = len(taps)
    pad_left, pad_right = K // 2 - 1, K // 2
    s_pad = torch.cat([s[:1].repeat(pad_left), s, s[-1:].repeat(pad_right)])
    T_out = (s_pad.shape[0] - K) // 2 + 1
    y = torch.zeros(T_out, dtype=torch.float64)
    for k, tap in enumerate(taps):
        y += tap * s_pad[k : k + 2 * T_out : 2]
    return y


def fused(ph0, ph1, taps, act):
    s0, s1 = act(ph0), act(ph1)
    T = s0.shape[0]
    first, last = s0[:1], s1[-1:]
    # P0[m] = s_pad[2m], P1[m] = s_pad[2m+1]; see the derivation in the docstring.
    P0 = torch.cat([first.repeat(3), s1, last.repeat(2)])
    P1 = torch.cat([first.repeat(2), s0, last.repeat(3)])
    even = [taps[2 * a] for a in range(len(taps) // 2)]
    odd = [taps[2 * a + 1] for a in range(len(taps) // 2)]
    T_out = T
    y = torch.zeros(T_out, dtype=torch.float64)
    for a, tap in enumerate(even):
        y += tap * P0[a : a + T_out]
    for a, tap in enumerate(odd):
        y += tap * P1[a : a + T_out]
    return y


def main():
    torch.manual_seed(0)
    taps = torch.randn(12, dtype=torch.float64).tolist()
    act = lambda v: v + 0.3 * torch.sin(1.7 * v) ** 2  # pointwise, stands in for snake_beta
    for M in (16, 17, 64, 207):
        ph0 = torch.randn(M, dtype=torch.float64)
        ph1 = torch.randn(M, dtype=torch.float64)
        r = reference(ph0, ph1, taps, act)
        f = fused(ph0, ph1, taps, act)
        ok = r.shape == f.shape and torch.allclose(r, f, atol=1e-12)
        diff = float((r[: min(len(r), len(f))] - f[: min(len(r), len(f))]).abs().max()) if len(r) and len(f) else -1
        print(f"M={M:<5} ref_len={len(r):<5} fused_len={len(f):<5} match={ok} maxdiff={diff:.3e}")


if __name__ == "__main__":
    main()

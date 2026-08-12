"""Standalone check of the depthwise weight layout under a channel multiplier -- no device, no build.

Item 3 step 2 says the weight matrix goes (K*C, C) -> (K*C, k*C) and that "output column k*c + j needs
group c's tap set j", and to check that before touching anything downstream. This models both halves of
the contract in pure torch and compares against F.conv1d(groups=C):

  weight prep   conv_depthwise_weight_bcast_helper's mapping: out[i, j] = w[i, 0, tap(j)]
  compute       for output column o, the activation column read is o // k (the plan's "c / k")

The point of interest is that the existing helper indexes axis 0 by ``original_weight_shape[0]``, which
is out_channels -- it never assumes out_channels == groups. So if that shape is allowed to be k*C, the
prep may already be correct and the multiplier work is the gate plus the activation column index.
"""

import torch
import torch.nn.functional as F

TILE_W = 32


def depthwise_layout(w, repeats=1):
    """Model conv_depthwise_weight_bcast_helper: [O,1,1,kW] -> [O, repeats*taps, 1, 1]."""
    O, one, kh, kw = w.shape
    assert one == 1, f"depthwise prep expects in_channels/groups == 1, got {one}"
    taps = kh * kw
    out = torch.zeros(O, repeats * taps, 1, 1, dtype=w.dtype)
    for i in range(O):
        for j in range(repeats * taps):
            tap = j // repeats  # broadcast_per_tap == repeats
            out[i, j, 0, 0] = w[i, 0, tap // kw, tap % kw]
    return out


def device_model(x, wl, k, kw, repeats=1):
    """Model the depthwise factory's accumulation using only the laid-out weights.

    x is (B, T, C_in) row-major activations, the layout the conv sees. Output column o accumulates
    over taps, reading activation column o // k -- this is the one indexing change item 3 point 4
    describes. Zero-padded at the edges, matching padding_mode="zeros"/kw//2.
    """
    B, T, C_in = x.shape
    O = wl.shape[0]
    assert O == k * C_in, f"expected out_channels {k * C_in}, got {O}"
    y = torch.zeros(B, T, O, dtype=x.dtype)
    pad = kw // 2
    for o in range(O):
        c_in = o // k  # <-- the input column is c/k while the output column stays c
        for tap in range(kw):
            wv = wl[o, tap * repeats, 0, 0]
            for t in range(T):
                s = t + tap - pad
                if 0 <= s < T:
                    y[b_all := slice(None), t, o] += x[b_all, s, c_in] * wv
    return y


def check(C, k, kw, T=9, B=2, repeats=1, seed=0):
    torch.manual_seed(seed)
    O = k * C
    # PyTorch grouped conv with groups == in_channels and out_channels == k*in_channels: group g owns
    # output channels [g*k, (g+1)*k) and input channel g, so output channel o reads input o // k.
    w = torch.randn(O, 1, 1, kw, dtype=torch.float64)
    x_bct = torch.randn(B, C, T, dtype=torch.float64)

    ref = F.conv1d(x_bct, w[:, :, 0, :], groups=C, padding=kw // 2)  # (B, O, T)

    wl = depthwise_layout(w, repeats=repeats)
    got_btc = device_model(x_bct.permute(0, 2, 1).contiguous(), wl, k, kw, repeats=repeats)
    got = got_btc.permute(0, 2, 1)  # -> (B, O, T)

    err = (got - ref).abs().max().item()
    denom = ref.abs().max().item()
    ok = err <= 1e-12 * max(denom, 1.0)
    print(
        f"  C={C:<4} k={k}  kw={kw}  out_channels={O:<5} "
        f"cols {O} (was {C})  max_abs_err {err:.3e}  {'OK' if ok else 'MISMATCH'}"
    )
    return ok


print("depthwise channel multiplier: weight layout + activation column index vs F.conv1d(groups=C)")
print("\nk == 1 (today's supported case -- must stay correct):")
all_ok = True
for C in (8, 32, 224):
    for kw in (1, 3, 7):
        all_ok &= check(C, 1, kw)

print("\nk > 1 (item 3's new case):")
for C, k in ((8, 2), (8, 4), (32, 2), (16, 3), (224, 2)):
    for kw in (3, 7):
        all_ok &= check(C, k, kw)

print("\nwith act_block_h broadcast repeats > 1 (the slab form the factory actually feeds):")
for repeats in (2, 4):
    all_ok &= check(8, 2, 3, repeats=repeats)

# The tile-width question the CB plumbing cares about: k*C must still tile cleanly.
print("\ntile alignment of the widened output (conv2d pads channels to TILE_WIDTH):")
for C, k in ((8, 2), (8, 4), (32, 2), (224, 2)):
    O = k * C
    print(f"  C={C:<4} k={k}  out_channels={O:<5} padded_to_32={((O + 31) // 32) * 32:<5} tiles={((O + 31) // 32)}")

print(f"\n{'ALL CHECKS PASS' if all_ok else 'FAILURES PRESENT'}")

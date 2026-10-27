# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0
"""Opt-in higher-precision subsampling for the TTNN Parakeet backend (off by default).

The device truncates fp32 matmul/conv operands (activations to 10 mantissa bits, weights to 9,
tests/diag_matmul_format.py). On hard inputs the encoder error is seeded in the subsampling stack
(tests/diag_hard_speech.py: feeding the FP32 reference subsampling output drops long+gain1.5 from
.057 to .025). `enable(backend, weights_path)` rebinds the backend's subsampling so every conv2d and
linear there runs as three passes on hi/lo operand splits:

    op(x, w) ~= op(x_hi, w_hi) + [op(x_lo, w_hi) + op(x_hi, w_lo)]

x_hi is the operand at the device width (conv0 input: host RNE to 10 bits; later activations: device
bf16 typecast), w_hi is host RNE to 9 bits, and the lo parts are the fp32 remainders (lo*lo is
dropped). The bias is added once. The encoder blocks and the decoder are unchanged.
"""

import types

from .ttnn_parakeet import FP32_WEIGHT_DROP_BITS, _load_state_dict, rne_drop_bits

ACT_DROP_BITS = 13  # fp32 activation operands keep 23 - 13 = 10 mantissa bits on device


def split(t, bits):
    """Host fp32 tensor -> (hi, lo): hi keeps 23 - bits mantissa bits (RNE), lo is the remainder at that width."""
    hi = rne_drop_bits(t, bits)
    return hi, rne_drop_bits(t.float() - hi, bits)


def enable(bk, weights_path):
    """Switch `bk` (a ttnn_parakeet.Backend) to hi/lo split subsampling; returns bk."""
    import torch

    ttnn, c = bk.ttnn, bk.cfg
    if bk.adtype != ttnn.float32:
        raise ValueError("split subsampling needs fp32 activations")
    sd = _load_state_dict(weights_path)
    pre = "encoder.subsampling."
    convs = sorted({int(k.split(".")[3]) for k in sd if k.startswith(pre + "layers.")})
    wb = FP32_WEIGHT_DROP_BITS
    host = lambda t: ttnn.from_torch(t.contiguous().float(), dtype=ttnn.float32)
    dev = lambda t: bk._dev(t, ttnn.float32)
    host_pair = lambda t: tuple(host(p) for p in split(t, wb))
    dev_pair = lambda t: tuple(dev(p) for p in split(t, wb))

    w0 = sd[pre + f"layers.{convs[0]}.weight"]
    w0p = torch.zeros(w0.shape[0], bk.cin0, *w0.shape[2:])
    w0p[:, : w0.shape[1]] = w0
    bk.ss_conv0 = host_pair(w0p) + (host(bk._rw(sd[pre + f"layers.{convs[0]}.bias"].reshape(1, 1, 1, -1))),)
    bk.ss_stages = []
    for dw, pw in zip(convs[1::2], convs[2::2]):
        wdw = host_pair(sd[pre + f"layers.{dw}.weight"]) + (
            host(bk._rw(sd[pre + f"layers.{dw}.bias"].reshape(1, 1, 1, -1))),
        )
        wpw = dev_pair(sd[pre + f"layers.{pw}.weight"][:, :, 0, 0].t()) + (
            bk._sw(sd[pre + f"layers.{pw}.bias"].reshape(1, -1)),
        )
        bk.ss_stages.append((wdw, wpw))
    lw = sd[pre + "linear.weight"]
    D, C = c.hidden, c.sub_channels
    F = lw.shape[1] // C
    bk.ss_lin = dev_pair(lw.reshape(D, C, F).permute(0, 2, 1).reshape(D, F * C).t()) + (
        bk._sw(sd[pre + "linear.bias"].reshape(1, -1)),
    )
    bk._subsample = types.MethodType(_subsample_split, bk)
    bk.sub_split = True
    bk.precision_policy["exceptions"].append("subsampling conv2d/linear run as 3-pass hi/lo operand splits")
    return bk


def _act_split(bk, x):
    ttnn = bk.ttnn
    hi = ttnn.typecast(ttnn.typecast(x, ttnn.bfloat16), ttnn.float32)
    return hi, ttnn.subtract(x, hi)


def _conv3(bk, xs, wts, batch, h, w, cin, cout, groups):
    (xh, xl), (wh, wl, b) = xs, wts
    y, ho, wo = bk._conv2d(xh, wh, b, batch, h, w, cin, cout, groups)
    corr = bk.ttnn.add(
        bk._conv2d(xl, wh, None, batch, h, w, cin, cout, groups)[0],
        bk._conv2d(xh, wl, None, batch, h, w, cin, cout, groups)[0],
    )
    return bk.ttnn.add(y, corr), ho, wo


def _lin3(bk, x, wts, activation=None):
    ttnn = bk.ttnn
    wh, wl, b = wts
    xh, xl = _act_split(bk, x)
    y = ttnn.add(bk._linear(xh, wh, b), ttnn.add(bk._linear(xl, wh), bk._linear(xh, wl)))
    return ttnn.relu(y) if activation == "relu" else y


def _subsample_split(self, mel_pad, lengths):
    """Same graph as Backend._subsample with every conv2d / linear replaced by its 3-pass split."""
    import numpy as np
    import torch

    ttnn, c = self.ttnn, self.cfg
    B, T, F = mel_pad.shape
    C = c.sub_channels
    x = torch.zeros(B, T, F, self.cin0)
    x[..., 0] = torch.from_numpy(mel_pad)
    up = lambda t: ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device)
    xs = tuple(up(p) for p in split(x, ACT_DROP_BITS))
    lens = np.asarray(lengths, dtype=np.int64)
    pad = (c.sub_kernel - 1) // 2
    nxt = lambda n: (n + 2 * pad - c.sub_kernel) // c.sub_stride + 1
    x, h, w = _conv3(self, xs, self.ss_conv0, B, T, F, self.cin0, C, 1)
    lens = nxt(lens)
    x = ttnn.reshape(x, (B, h * w, C))
    x = ttnn.relu(ttnn.multiply(x, self._time_mask(lens, h, w)))
    for wdw, wpw in self.ss_stages:
        x = ttnn.reshape(x, (1, 1, B * h * w, C))
        x, h, w = _conv3(self, _act_split(self, x), wdw, B, h, w, C, C, C)
        lens = nxt(lens)
        x = ttnn.reshape(x, (B, h * w, C))
        x = _lin3(self, x, wpw, activation="relu")
        x = ttnn.multiply(x, self._time_mask(lens, h, w))
    x = ttnn.reshape(x, (B, h, w * C))
    return _lin3(self, x, self.ss_lin), lens

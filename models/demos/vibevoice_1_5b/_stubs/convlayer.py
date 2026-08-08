# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `convlayer` of microsoft/VibeVoice-1.5B.

Reference submodule: `model.acoustic_tokenizer.encoder.stages.0.0.mixer`, a
`vibevoice.modular.modular_vibevoice_tokenizer.Convlayer` — a thin wrapper
around `SConv1d` (itself `NormConv1d` + causal padding), used as the token-mixer
inside `Block1D` (see `_stubs/block1_d.py`, which implements the same op
inline as `_depthwise_causal_conv1d`).

For this component: depthwise (`groups==dim==32`) causal Conv1d, kernel_size=7,
stride=1, dilation=1, left-pad=6 (constant/zero padding; the stride-1
`get_extra_padding_for_conv1d` term is always 0 here since T is already an
integer number of frames), `norm='none'` (Identity).

Implemented as a shifted tap-accumulate, elementwise per-channel (not matmul,
since groups == channels):
    y[:, t, c] = sum_tap  x_pad[:, t + tap, c] * w[c, 0, tap]   (+ bias)
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs._trace_pad import cached_zeros

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained depthwise conv weight/bias and return a native ttnn forward closure."""
    conv = torch_module.conv.conv.conv  # SConv1d.conv -> NormConv1d.conv -> nn.Conv1d
    w = conv.weight.detach().float()  # [C, 1, K]
    b = conv.bias.detach().float() if conv.bias is not None else None
    dim, _, kernel_size = w.shape

    if conv.groups != dim:
        raise RuntimeError(
            f"convlayer native port expects a depthwise conv (groups==dim); got groups={conv.groups}, dim={dim}"
        )
    if int(conv.stride[0]) != 1 or int(conv.dilation[0]) != 1:
        raise RuntimeError("convlayer native port supports stride=1, dilation=1 only")

    sconv = torch_module.conv  # SConv1d
    pad_left = int(sconv.padding_total)
    if not bool(sconv.causal):
        raise RuntimeError("convlayer native port assumes a causal depthwise conv")

    def _row(t_1d):
        return ttnn.from_torch(
            t_1d.reshape(1, 1, -1).contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device
        )

    taps = [_row(w[:, 0, tap]) for tap in range(kernel_size)]
    bias_row = _row(b) if b is not None else None
    _zc: dict = {}  # cached causal zero-pad buffers (trace-capture safe)

    def _to_f32(t):
        return ttnn.typecast(t, ttnn.float32) if t.get_dtype() != ttnn.float32 else t

    _TILE_H = 32

    def _pad_time(x_tc, pad_l, pad_r):
        if pad_l == 0 and pad_r == 0:
            return x_tc, 0
        B, T, C = x_tc.shape
        # Round the causal left-pad up to a full tile so the zero block's own
        # tile boundary lines up with x_tc's (both multiples of 32). A
        # fractional-tile pad (e.g. pad_l=6) forces ttnn.concat to re-tile the
        # shared boundary tile via TilizeWithValPadding on every call; padding
        # to a whole tile instead makes the concat a plain tile-list splice
        # (no value-padding kernel). The extra leading zero rows are never
        # read (the per-tap slice below is offset to skip them), so the
        # numerics are unchanged -- only the padding buffer's shape grows.
        pad_l_aligned = ((pad_l + _TILE_H - 1) // _TILE_H) * _TILE_H if pad_l else 0
        pieces = []
        if pad_l_aligned:
            pieces.append(cached_zeros(_zc, (B, pad_l_aligned, C), ttnn.float32, ttnn.TILE_LAYOUT, device))
        pieces.append(x_tc)
        if pad_r:
            pieces.append(cached_zeros(_zc, (B, pad_r, C), ttnn.float32, ttnn.TILE_LAYOUT, device))
        return ttnn.concat(pieces, dim=1, memory_config=_DRAM), (pad_l_aligned - pad_l)

    def forward_tc(x_tc, *args, **kwargs):
        # Channels-last core: [B, T, C] -> [B, T, C]. This is where all the real
        # work happens; the depthwise tap-accumulate is naturally channels-last
        # (taps broadcast as [1,1,C]). Block1D calls this directly so the mixer
        # participates in a single channels-last block instead of transposing in
        # and out around it (see block1_d.py) -- killing the layout churn.
        x_tc = _to_f32(x_tc)
        xp, extra_pad = _pad_time(x_tc, pad_left, 0)
        Lp = int(xp.shape[1]) - extra_pad
        t_out = Lp - (kernel_size - 1)
        y = None
        for tap in range(kernel_size):
            off = extra_pad + tap
            xs = ttnn.slice(xp, [0, off, 0], [1, off + t_out, dim])
            term = ttnn.mul(xs, taps[tap], memory_config=_DRAM)
            y = term if y is None else ttnn.add(y, term, memory_config=_DRAM)
        if bias_row is not None:
            y = ttnn.add(y, bias_row, memory_config=_DRAM)
        return y

    def forward(x, *args, **kwargs):
        # x: [B, C, T] channels-first (graduated interface, unchanged).
        x = _to_f32(x)
        x_tc = ttnn.transpose(x, 1, 2)  # [B, T, C]
        y = forward_tc(x_tc)
        return ttnn.transpose(y, 1, 2)  # [B, C, T]

    forward.forward_tc = forward_tc  # channels-last fast path for Block1D
    return forward


def convlayer(*args, **kwargs):
    raise RuntimeError(
        "convlayer requires build(device, torch_module) to bind trained weights; the bare callable has no parameters."
    )

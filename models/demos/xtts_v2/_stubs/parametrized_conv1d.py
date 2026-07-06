# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrized_conv1d` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.waveform_decoder.resblocks.0.convs1.0`, a
weight-norm `Conv1d` (stride 1, groups 1, kernel 3, padding 1, dilation 1, bias),
weight `[256, 256, 3]`. `.weight` returns the reconstructed (weight-norm-folded)
kernel, so we bind it directly. Captured input/output: `[1, 256, 416]` (channels-
first `[B, C, T]`).

Implemented as a **matmul shifted tap-accumulate** conv (the same recipe the
graduated `hifigan_generator` port uses — `ttnn.conv1d`'s halo path OOMs L1_SMALL,
and tap-accumulate lands at PCC ~1.0 in fp32). For a stride-1, groups-1 Conv1d:

    y[:, t, :] = sum_tap  x_pad[:, t + tap*dilation, :] @ W[:, :, tap]^T   (+ bias)

i.e. `k` matmuls of `[1, T, C_in] @ [C_in, C_out]` on the zero-padded input. The
channels-first `[1, C, T]` activation is transposed to `[1, T, C]` for the matmuls
and transposed back to `[1, C, T]` to match the reference output.
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained conv weight/bias and return a native ttnn forward closure."""
    import torch

    m = torch_module
    w = m.weight.detach().float()              # [C_out, C_in, k]
    c_out, c_in, k = w.shape
    stride = int(m.stride[0])
    dil = int(m.dilation[0])
    pad = int(m.padding[0])
    if stride != 1 or m.groups != 1:
        raise RuntimeError(f"parametrized_conv1d native port supports stride-1 groups-1 only (got stride={stride}, groups={m.groups})")

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Per-tap weight in x @ W orientation: W[:, :, tap]^T -> [C_in, C_out].
    taps = [
        ttnn.as_tensor(
            w[:, :, tap].t().contiguous(), dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        for tap in range(k)
    ]
    bias = None
    if m.bias is not None:
        bias = ttnn.as_tensor(
            m.bias.detach().reshape(1, 1, c_out).contiguous().float(),
            dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _pad_L(x, p):
        if p == 0:
            return x
        B, L, C = x.shape
        z = ttnn.zeros((B, p, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.concat([z, x, z], dim=1, memory_config=_DRAM)

    def forward(x, *args, **kwargs):
        # x: [1, C_in, T] channels-first -> [1, T, C_in]
        xtlc = ttnn.transpose(x, 1, 2)
        if xtlc.get_dtype() != ttnn.float32:
            xtlc = ttnn.typecast(xtlc, ttnn.float32)
        xp = _pad_L(xtlc, pad)
        Lp = int(xp.shape[1])
        t_out = Lp - dil * (k - 1)
        y = None
        for tap in range(k):
            s = tap * dil
            xs = ttnn.slice(xp, [0, s, 0], [1, s + t_out, c_in])
            yt = ttnn.matmul(xs, taps[tap], compute_kernel_config=compute_config, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        if bias is not None:
            y = ttnn.add(y, bias, memory_config=_DRAM)
        # [1, T, C_out] -> [1, C_out, T] channels-first
        return ttnn.transpose(y, 1, 2)

    return forward


def parametrized_conv1d(*args, **kwargs):
    raise RuntimeError(
        "parametrized_conv1d requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )

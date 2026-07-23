# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `res_block1` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.waveform_decoder.resblocks.0`, a
`TTS.vocoder.models.hifigan_generator.ResBlock1` (channels 256, kernel 3,
dilations (1, 3, 5)):

    for c1, c2 in zip(convs1, convs2):
        xt = leaky_relu(x, 0.1); xt = c1(xt)
        xt = leaky_relu(xt, 0.1); xt = c2(xt)
        x = xt + x
    return x

`convs1[i]` has dilation `(1,3,5)[i]` (padding = dilation), `convs2[i]` has
dilation 1 (padding 1); weight-norm is folded (`.weight` materialized). Captured
input/output `[1, 256, 416]` (channels-first `[B, C, T]`).

Each `Conv1d` is the stride-1 matmul tap-accumulate used by the graduated
`hifigan_generator` / `parametrized_conv1d` ports (fp32, HiFi4). We transpose the
activation to channels-last `[1, T, C]` once, run the whole block there, and
transpose back to `[1, C, T]`.
"""

from __future__ import annotations

import ttnn

_LRELU = 0.1
_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the resblock's conv weights and return a native ttnn forward closure."""

    m = torch_module

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _prep(c):
        w = c.weight.detach().float()  # [C_out, C_in, k]
        c_out, c_in, k = w.shape
        taps = [
            ttnn.as_tensor(
                w[:, :, tap].t().contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for tap in range(k)
        ]
        bias = None
        if c.bias is not None:
            bias = ttnn.as_tensor(
                c.bias.detach().reshape(1, 1, c_out).contiguous().float(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        return {"taps": taps, "bias": bias, "k": k, "cin": c_in, "dil": int(c.dilation[0]), "pad": int(c.padding[0])}

    convs1 = [_prep(c) for c in m.convs1]
    convs2 = [_prep(c) for c in m.convs2]

    def _pad_L(x, p):
        if p == 0:
            return x
        B, L, C = x.shape
        z = ttnn.zeros((B, p, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.concat([z, x, z], dim=1, memory_config=_DRAM)

    def _conv(x, spec):
        # x: [1, L, C_in] channels-last -> [1, T, C_out]
        xp = _pad_L(x, spec["pad"])
        Lp = int(xp.shape[1])
        t_out = Lp - spec["dil"] * (spec["k"] - 1)
        y = None
        for tap in range(spec["k"]):
            s = tap * spec["dil"]
            xs = ttnn.slice(xp, [0, s, 0], [1, s + t_out, spec["cin"]])
            yt = ttnn.matmul(xs, spec["taps"][tap], compute_kernel_config=compute_config, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        if spec["bias"] is not None:
            y = ttnn.add(y, spec["bias"], memory_config=_DRAM)
        return y

    def forward(x, *args, **kwargs):
        # [1, C, T] -> [1, T, C]
        h = ttnn.transpose(x, 1, 2)
        if h.get_dtype() != ttnn.float32:
            h = ttnn.typecast(h, ttnn.float32)
        for c1, c2 in zip(convs1, convs2):
            xt = ttnn.leaky_relu(h, _LRELU)
            xt = _conv(xt, c1)
            xt = ttnn.leaky_relu(xt, _LRELU)
            xt = _conv(xt, c2)
            h = ttnn.add(xt, h)
        # [1, T, C] -> [1, C, T]
        return ttnn.transpose(h, 1, 2)

    return forward


def res_block1(*args, **kwargs):
    raise RuntimeError(
        "res_block1 requires build(device, torch_module) to bind trained weights; "
        "the bare callable has no parameters."
    )

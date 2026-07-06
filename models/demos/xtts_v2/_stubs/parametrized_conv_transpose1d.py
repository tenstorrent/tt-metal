# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrized_conv_transpose1d` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.waveform_decoder.ups.0`, a weight-norm
`ConvTranspose1d` (stride 8, padding 4, dilation 1, output_padding 0, bias),
weight `[512, 256, 16]` (torch ConvTranspose weight is `[in, out, k]`). `.weight`
returns the reconstructed (weight-norm-folded) kernel. Captured input/output:
`[1, 512, 52]` -> `[1, 256, 416]` (channels-first `[B, C, T]`).

A stride-`s` `ConvTranspose1d` is exactly a stride-1 `Conv1d` on the input with
`s-1` zeros stuffed between samples, using the kernel flipped along `k` and with
`(in, out)` transposed, symmetrically padded by `k - 1 - (k - s)//2` (the inverse
of torch's `padding=(k-s)//2`). The stride-1 conv is then the same matmul
tap-accumulate used by the graduated `hifigan_generator` / `parametrized_conv1d`
ports (fp32, HiFi4 -> PCC ~1.0; avoids `ttnn.conv1d`'s L1_SMALL halo OOM).

    y[:, t, :] = sum_tap  x_stuffed_pad[:, t + tap, :] @ Wc[:, :, tap]^T   (+ bias)

with `Wc = flip(W, k).permute(1,0,2)`  ->  `[out, in, k]`.
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.parametrization_list import build as _build_parametrization_list

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind the trained transpose-conv weight/bias and return a native ttnn forward."""
    import torch

    m = torch_module
    parametrized = hasattr(m, "parametrizations") and "weight" in m.parametrizations
    w_dev = None
    if parametrized:
        # Live weight-norm: reconstruct the conv weight via the parametrization_list
        # leaf instead of reading m.weight. It returns the same [C_in, C_out, k]
        # weight m.weight would materialize, and stays ON DEVICE (no host round-trip),
        # so its output genuinely feeds the taps below.
        _pl_fwd = _build_parametrization_list(device, m.parametrizations.weight)
        w_dev = _pl_fwd()                        # ttnn [C_in, C_out, k]
        c_in, c_out, k = (int(d) for d in w_dev.shape)
    else:
        w = m.weight.detach().float()            # [C_in, C_out, k]
        c_in, c_out, k = w.shape
    stride = int(m.stride[0])
    pad = int(m.padding[0])
    out_pad = int(m.output_padding[0])
    if int(m.dilation[0]) != 1 or m.groups != 1 or out_pad != 0:
        raise RuntimeError(
            f"parametrized_conv_transpose1d native port supports dilation=1, groups=1, "
            f"output_padding=0 only (got dil={m.dilation}, groups={m.groups}, out_pad={out_pad})"
        )

    ext_pad = k - 1 - (k - stride) // 2

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Conv1d-equivalent kernel: the ConvTranspose weight [C_in, C_out, k] flipped
    # along k with (in,out) transposed means tap `t` is exactly w[:, :, k-1-t]
    # (a [C_in, C_out] slice — no transpose needed).
    if parametrized:
        # Extract taps ON DEVICE from the resident reconstructed weight (host-free):
        # permute k to the front, then slice one tap at a time.
        wp = ttnn.permute(w_dev, (2, 0, 1))              # [k, C_in, C_out]
        taps = []
        for tap in range(k):
            idx = k - 1 - tap
            sl = ttnn.slice(wp, [idx, 0, 0], [idx + 1, c_in, c_out])  # [1, C_in, C_out]
            sl = ttnn.reshape(sl, [c_in, c_out])
            taps.append(ttnn.to_layout(sl, ttnn.TILE_LAYOUT))
        ttnn.deallocate(wp)
    else:
        w_conv = torch.flip(w, dims=[-1]).permute(1, 0, 2).contiguous()   # [C_out, C_in, k]
        taps = [
            ttnn.as_tensor(
                w_conv[:, :, tap].t().contiguous(), dtype=ttnn.float32,   # [C_in, C_out]
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

    def _zero_stuff(x):
        # [1, L, C] -> [1, L*stride - (stride-1), C] with stride-1 zeros between samples.
        if stride == 1:
            return x
        B, L, C = x.shape
        xr = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (B, L, 1, C))
        z = ttnn.zeros((B, L, stride - 1, C), dtype=xr.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        s = ttnn.concat([xr, z], dim=2, memory_config=_DRAM)
        s = ttnn.reshape(s, (B, L * stride, C))
        s = ttnn.slice(s, [0, 0, 0], [B, L * stride - (stride - 1), C])
        return ttnn.to_layout(s, ttnn.TILE_LAYOUT)

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
        xs = _zero_stuff(xtlc)
        xp = _pad_L(xs, ext_pad)
        Lp = int(xp.shape[1])
        t_out = Lp - (k - 1)
        y = None
        for tap in range(k):
            xsl = ttnn.slice(xp, [0, tap, 0], [1, tap + t_out, c_in])
            yt = ttnn.matmul(xsl, taps[tap], compute_kernel_config=compute_config, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        if bias is not None:
            y = ttnn.add(y, bias, memory_config=_DRAM)
        # [1, T_out, C_out] -> [1, C_out, T_out] channels-first
        return ttnn.transpose(y, 1, 2)

    return forward


def parametrized_conv_transpose1d(*args, **kwargs):
    raise RuntimeError(
        "parametrized_conv_transpose1d requires build(device, torch_module) to bind "
        "trained weights; the bare callable has no parameters."
    )

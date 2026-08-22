# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `ada_i_n_res_block1` (hexgrad/Kokoro-82M
`decoder.generator.noise_res.0`, a StyleTTS2/ISTFTNet `AdaINResBlock1`).

Reference torch forward (3 layers, each a Snake-activated conv pair with AdaIN
style modulation and a residual add):

    def forward(self, x, s):
        for c1, c2, n1, n2, a1, a2 in zip(convs1, convs2, adain1, adain2,
                                          alpha1, alpha2):
            xt = n1(x, s)
            xt = xt + (1 / a1) * (torch.sin(a1 * xt) ** 2)   # Snake1D
            xt = c1(xt)
            xt = n2(xt, s)
            xt = xt + (1 / a2) * (torch.sin(a2 * xt) ** 2)   # Snake1D
            xt = c2(xt)
            x = xt + x
        return x

Native ttnn building blocks (all fp32 for a clean PCC):
  * AdaIN1d: `s @ fc^T + b -> (gamma, beta)`; InstanceNorm1d over the time axis
    with affine weight/bias; `(1 + gamma) * norm(x) + beta`.
  * Snake1D: `xt + (1/a) * sin(a * xt)^2` with per-channel alpha `[1, C, 1]`.
  * Conv1d (weight-norm folded via `.weight`): matmul shifted tap-accumulate,
    the same recipe as the graduated `parametrized_conv1d` / `hifigan_generator`
    ports (ttnn.conv1d's halo path OOMs L1_SMALL; tap-accumulate is fp32-exact).
"""

from __future__ import annotations

import ttnn
from models.demos.kokoro_82m._stubs._lstm_scan import masked_moments, zero_pad_frames

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind all layer params and return a native ttnn forward closure."""
    import torch

    m = torch_module

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _t(x):
        return ttnn.from_torch(
            x.contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )

    # ---- AdaIN1d params -> a callable closure ----
    def _build_adain(adain):
        eps = float(getattr(adain.norm, "eps", 1e-5))
        c = int(adain.norm.weight.shape[0])
        fc_w_t = _t(adain.fc.weight.detach().t())  # [style_dim, 2C]
        fc_b = _t(adain.fc.bias.detach().reshape(1, -1))  # [1, 2C]
        norm_w = _t(adain.norm.weight.detach().reshape(1, -1, 1))
        norm_b = _t(adain.norm.bias.detach().reshape(1, -1, 1))

        def apply(x, s):
            h = ttnn.linear(s, fc_w_t, bias=fc_b, compute_kernel_config=compute_config)  # [B, 2C]
            b = h.shape[0]
            gamma = ttnn.reshape(ttnn.slice(h, [0, 0], [b, c]), [b, c, 1])
            beta = ttnn.reshape(ttnn.slice(h, [0, c], [b, 2 * c]), [b, c, 1])
            mv = masked_moments(device, x)  # frame-axis masked mean/var under a fixed-capacity trace
            if mv is not None:
                mean, var = mv
                xc = ttnn.subtract(x, mean)
            else:
                mean = ttnn.mean(x, dim=2, keepdim=True)
                xc = ttnn.subtract(x, mean)
                var = ttnn.mean(ttnn.multiply(xc, xc), dim=2, keepdim=True)
            nx = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
            nx = ttnn.add(ttnn.multiply(nx, norm_w), norm_b)
            return ttnn.add(ttnn.multiply(ttnn.add(gamma, 1.0), nx), beta)

        return apply

    # ---- Conv1d (stride-1, groups-1) params -> a tap-accumulate closure ----
    def _build_conv(conv):
        w = conv.weight.detach().float()  # [C_out, C_in, k]
        c_out, c_in, k = w.shape
        stride = int(conv.stride[0])
        dil = int(conv.dilation[0])
        pad = int(conv.padding[0])
        if stride != 1 or conv.groups != 1:
            raise RuntimeError(
                f"ada_i_n_res_block1 conv port supports stride-1 groups-1 only "
                f"(got stride={stride}, groups={conv.groups})"
            )
        taps = [_t(w[:, :, tap].t()) for tap in range(k)]  # each [C_in, C_out]
        bias = _t(conv.bias.detach().reshape(1, 1, c_out)) if conv.bias is not None else None

        def _pad_L(x, p):
            if p == 0:
                return x
            B, L, C = x.shape
            z = ttnn.zeros((B, p, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
            return ttnn.concat([z, x, z], dim=1, memory_config=_DRAM)

        def apply(x):
            # x: [1, C_in, T] -> [1, T, C_in]
            xtlc = ttnn.transpose(x, 1, 2)
            if xtlc.get_dtype() != ttnn.float32:
                xtlc = ttnn.typecast(xtlc, ttnn.float32)
            xp = _pad_L(xtlc, pad)
            Lp = int(xp.shape[1])
            t_out = Lp - dil * (k - 1)
            y = None
            for tap in range(k):
                s0 = tap * dil
                xs = ttnn.slice(xp, [0, s0, 0], [1, s0 + t_out, c_in])
                yt = ttnn.matmul(xs, taps[tap], compute_kernel_config=compute_config, memory_config=_DRAM)
                y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
            if bias is not None:
                y = ttnn.add(y, bias, memory_config=_DRAM)
            return ttnn.transpose(y, 1, 2)  # [1, C_out, T]

        return apply

    n_layers = len(m.convs1)
    layers = []
    for i in range(n_layers):
        a1 = m.alpha1[i].detach().float()  # [1, C, 1]
        a2 = m.alpha2[i].detach().float()
        layers.append(
            {
                "n1": _build_adain(m.adain1[i]),
                "n2": _build_adain(m.adain2[i]),
                "c1": _build_conv(m.convs1[i]),
                "c2": _build_conv(m.convs2[i]),
                "a1": _t(a1),
                "inv_a1": _t(torch.reciprocal(a1)),
                "a2": _t(a2),
                "inv_a2": _t(torch.reciprocal(a2)),
            }
        )

    def _snake(xt, a, inv_a):
        # xt + (1/a) * sin(a * xt)^2
        s = ttnn.sin(ttnn.multiply(xt, a))
        s2 = ttnn.multiply(s, s)
        return ttnn.add(xt, ttnn.multiply(s2, inv_a))

    def _to_ttnn(t):
        if isinstance(t, ttnn.Tensor):
            if t.get_dtype() != ttnn.float32:
                t = ttnn.typecast(t, ttnn.float32)
            return t
        return ttnn.from_torch(
            t.contiguous().float(),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )

    def forward(x, s=None, *args, **kwargs):
        if s is None:
            raise RuntimeError("ada_i_n_res_block1 forward requires style vector `s`")
        x = _to_ttnn(x)  # [B, C, T]
        s = _to_ttnn(s)  # [B, style_dim]
        for lyr in layers:
            xt = lyr["n1"](x, s)
            xt = _snake(xt, lyr["a1"], lyr["inv_a1"])
            xt = lyr["c1"](xt)
            xt = lyr["n2"](xt, s)
            xt = _snake(xt, lyr["a2"], lyr["inv_a2"])
            xt = lyr["c2"](xt)
            x = ttnn.add(xt, x)
        return zero_pad_frames(device, x)

    return forward


def ada_i_n_res_block1(*args, **kwargs):
    raise RuntimeError(
        "ada_i_n_res_block1 requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )

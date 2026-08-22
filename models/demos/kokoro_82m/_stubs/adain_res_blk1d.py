# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `adain_res_blk1d` (hexgrad/Kokoro-82M
`predictor.F0.0`, a StyleTTS2 `AdainResBlk1d`).

Reference torch forward (dim_in == dim_out == 512, upsample='none', so the
shortcut and pool are identities and there is no learned 1x1):

    def _residual(self, x, s):
        x = self.norm1(x, s); x = self.actv(x); x = self.pool(x)   # AdaIN, LeakyReLU
        x = self.conv1(self.dropout(x))                            # Conv1d k3 p1
        x = self.norm2(x, s); x = self.actv(x)
        x = self.conv2(self.dropout(x))                            # Conv1d k3 p1
        return x
    def forward(self, x, s):
        out = self._residual(x, s)
        out = (out + self._shortcut(x)) * torch.rsqrt(torch.tensor(2))
        return out

`actv` is `LeakyReLU(0.2)`, `dropout` is identity in eval, `_shortcut(x) == x`.

Native ttnn building blocks (fp32 for a clean PCC):
  * AdaIN1d: `s @ fc^T + b -> (gamma, beta)`; InstanceNorm1d over the time axis
    with affine weight/bias; `(1 + gamma) * norm(x) + beta`.
  * Conv1d (weight-norm folded via `.weight`): matmul shifted tap-accumulate,
    the same recipe as the graduated `parametrized_conv1d` port.
"""

from __future__ import annotations

import math

import ttnn
from models.demos.kokoro_82m._stubs._lstm_scan import masked_moments, zero_pad_frames

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def build(device, torch_module):
    """Bind all layer params and return a native ttnn forward closure."""

    m = torch_module
    if m.learned_sc or m.upsample_type != "none":
        raise RuntimeError(
            f"adain_res_blk1d native port supports learned_sc=False upsample='none' "
            f"only (got learned_sc={m.learned_sc}, upsample={m.upsample_type})"
        )
    neg_slope = float(getattr(m.actv, "negative_slope", 0.2))
    inv_sqrt2 = float(1.0 / math.sqrt(2.0))

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

    def _build_adain(adain):
        eps = float(getattr(adain.norm, "eps", 1e-5))
        c = int(adain.norm.weight.shape[0])
        fc_w_t = _t(adain.fc.weight.detach().t())
        fc_b = _t(adain.fc.bias.detach().reshape(1, -1))
        norm_w = _t(adain.norm.weight.detach().reshape(1, -1, 1))
        norm_b = _t(adain.norm.bias.detach().reshape(1, -1, 1))

        def apply(x, s):
            h = ttnn.linear(s, fc_w_t, bias=fc_b, compute_kernel_config=compute_config)
            b = h.shape[0]
            gamma = ttnn.reshape(ttnn.slice(h, [0, 0], [b, c]), [b, c, 1])
            beta = ttnn.reshape(ttnn.slice(h, [0, c], [b, 2 * c]), [b, c, 1])
            # InstanceNorm1d over time. Compute Var = E[x^2] - E[x]^2 from SUMS
            # over the TRUE length (x's tile padding is 0, so both sums are exact
            # regardless of tile padding). Using ttnn.mean(x-mean)^2 instead would
            # fold the -mean written into padding slots into the variance, a
            # systematic variance-inflation -> output down-scale that drifts the
            # NSF source phase. See ada_i_n1d for the same fix.
            mv = masked_moments(device, x)  # frame-axis masked mean/var under a fixed-capacity trace
            if mv is not None:
                mean, var = mv
            else:
                n = int(x.shape[-1])
                inv_n = 1.0 / float(n)
                mean = ttnn.multiply(ttnn.sum(x, dim=2, keepdim=True), inv_n)
                mean_x2 = ttnn.multiply(ttnn.sum(ttnn.multiply(x, x), dim=2, keepdim=True), inv_n)
                var = ttnn.subtract(mean_x2, ttnn.multiply(mean, mean))
            xc = ttnn.subtract(x, mean)
            nx = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, eps)))
            nx = ttnn.add(ttnn.multiply(nx, norm_w), norm_b)
            return ttnn.add(ttnn.multiply(ttnn.add(gamma, 1.0), nx), beta)

        return apply

    def _build_conv(conv):
        w = conv.weight.detach().float()  # [C_out, C_in, k]
        c_out, c_in, k = w.shape
        stride = int(conv.stride[0])
        dil = int(conv.dilation[0])
        pad = int(conv.padding[0])
        if stride != 1 or conv.groups != 1:
            raise RuntimeError(
                f"adain_res_blk1d conv port supports stride-1 groups-1 only "
                f"(got stride={stride}, groups={conv.groups})"
            )
        taps = [_t(w[:, :, tap].t()) for tap in range(k)]
        bias = _t(conv.bias.detach().reshape(1, 1, c_out)) if conv.bias is not None else None

        def _pad_L(x, p):
            if p == 0:
                return x
            B, L, C = x.shape
            z = ttnn.zeros((B, p, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
            return ttnn.concat([z, x, z], dim=1, memory_config=_DRAM)

        def apply(x):
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
            return ttnn.transpose(y, 1, 2)

        return apply

    norm1 = _build_adain(m.norm1)
    norm2 = _build_adain(m.norm2)
    conv1 = _build_conv(m.conv1)
    conv2 = _build_conv(m.conv2)

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
            raise RuntimeError("adain_res_blk1d forward requires style vector `s`")
        x = _to_ttnn(x)  # [B, C, T]
        s = _to_ttnn(s)  # [B, style_dim]

        r = norm1(x, s)
        r = ttnn.leaky_relu(r, neg_slope)
        r = conv1(r)
        r = norm2(r, s)
        r = ttnn.leaky_relu(r, neg_slope)
        r = conv2(r)

        out = ttnn.multiply(ttnn.add(r, x), inv_sqrt2)  # shortcut == x
        return zero_pad_frames(device, out)

    return forward


def adain_res_blk1d(*args, **kwargs):
    raise RuntimeError(
        "adain_res_blk1d requires build(device, torch_module) to bind trained "
        "weights; the bare callable has no parameters."
    )

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `hifigan_generator` (coqui/XTTS-v2 `hifigan_decoder.waveform_decoder`).

The submodule is a HiFi-GAN vocoder (`TTS.vocoder.models.hifigan_generator.HifiganGenerator`)
configured by the XTTS `HifiDecoder`:

    conv_pre : Conv1d(1024 -> 512, k=7, pad=3)
    + cond_layer(g)                              # 1x1 conv on the d-vector -> per-channel bias
    for i in 0..3:                               # upsample_factors = [8, 8, 2, 2]
        leaky_relu(0.1)
        ups[i]  : ConvTranspose1d(512//2^i -> 512//2^(i+1), k, stride, pad=(k-s)//2)
        + conds[i](g)                            # 1x1 conv -> per-channel bias (cond_in_each_up_layer)
        o = mean_j resblocks[i*3+j](o)           # MRF, kernels [3, 7, 11]
    leaky_relu(0.01)                             # NOTE: default slope, NOT 0.1
    conv_post : Conv1d(ch -> 1, k=7, pad=3, no bias)
    tanh

Weight-norm is already folded (HifiDecoder loads with eval=True -> remove_weight_norm()),
so every conv here is a plain `nn.Conv1d` / `nn.ConvTranspose1d` with materialized weights.

Convolutions are done as a **matmul shifted tap-accumulate** rather than `ttnn.conv1d`:
the sliding-window/halo path that `ttnn.conv1d` uses OOMs L1_SMALL on the long upsampled
sequences (up to 647k samples) this vocoder produces — the same wall the tt-metal LTX-2
audio decoder hit (`models/tt_dit/layers/audio_ops.py`), which likewise falls back to a
tap-accumulate. For a stride-1, groups-1 `Conv1d`:

    y[:, t, :] = sum_tap  x_pad[:, t + tap*dilation, :] @ W[:, :, tap]^T   (+ bias)

which is `k` matmuls of `[1, T_out, C_in] @ [C_in, C_out]` on zero-padded input — no halo,
fp32 accumulation, so each conv lands at PCC ~1.0. `ConvTranspose1d` is expressed as a
stride-1 `Conv1d` on the stride-zero-stuffed input with the kernel flipped along `k` and
`(in,out)` transposed, symmetrically padded by ``k-1-(k-stride)//2`` — the exact inverse of
torch's ``padding=(k-stride)//2``. All conv hyper-params are read from the torch module.
"""

from __future__ import annotations

import ttnn
from models.demos.vibevoice_1_5b._stubs.parametrized_conv1d import build as _build_conv1d
from models.demos.vibevoice_1_5b._stubs.parametrized_conv_transpose1d import build as _build_conv_transpose1d
from models.demos.vibevoice_1_5b._stubs.res_block1 import build as _build_res_block1

HF_MODEL_ID = "coqui/XTTS-v2"

_LRELU = 0.1  # slope used inside the upsample loop and the ResBlocks


def build(device, torch_module):
    """Bind the trained HiFi-GAN weights and return a native ttnn forward closure."""

    m = torch_module

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )
    DRAM = ttnn.DRAM_MEMORY_CONFIG

    def _taps(w, dtype):
        """torch Conv1d weight [out, in, k] -> list of k device tensors [in, out]."""
        out_ch, in_ch, k = w.shape
        return [
            ttnn.as_tensor(
                w[:, :, t].t().contiguous().float(),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for t in range(k)
        ]

    def _bias(bias, dtype):
        if bias is None:
            return None
        return ttnn.as_tensor(
            bias.detach().reshape(1, 1, -1).contiguous().float(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def prep_conv1d(c, dtype=ttnn.float32):
        pad = c.padding[0] if isinstance(c.padding, (tuple, list)) else c.padding
        return {
            "kind": "conv",
            "taps": _taps(c.weight.detach(), dtype),
            "b": _bias(c.bias, dtype),
            "dtype": dtype,
            "in_ch": c.in_channels,
            "out_ch": c.out_channels,
            "k": c.kernel_size[0],
            "stride": c.stride[0],
            "pad": pad,
            "dil": c.dilation[0],
        }

    # conv_pre / conv_post / ups / resblocks are delegated to the graduated leaf
    # stubs (channels-first [1, C, T]); the cond_layer / conds 1x1 cond-bias convs
    # stay on the local prep_conv1d + do_conv path.
    conv_pre_fwd = _build_conv1d(device, m.conv_pre)
    has_cond = hasattr(m, "cond_layer")
    cond_layer = prep_conv1d(m.cond_layer) if has_cond else None
    ups_fwd = [_build_conv_transpose1d(device, u) for u in m.ups]
    cond_in_each = bool(getattr(m, "cond_in_each_up_layer", False))
    conds = [prep_conv1d(c) for c in m.conds] if cond_in_each else None
    resblock_fwd = [_build_res_block1(device, rb) for rb in m.resblocks]
    conv_post_fwd = _build_conv1d(device, m.conv_post)
    num_kernels = m.num_kernels
    num_upsamples = m.num_upsamples

    def _free(*ts):
        for t in ts:
            if isinstance(t, ttnn.Tensor):
                ttnn.deallocate(t)

    def _pad_L(x, pl, pr):
        if pl == 0 and pr == 0:
            return x, False
        B, L, C = x.shape
        parts = []
        if pl:
            parts.append(ttnn.zeros((B, pl, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device))
        parts.append(x)
        if pr:
            parts.append(ttnn.zeros((B, pr, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device))
        return ttnn.concat(parts, dim=1, memory_config=DRAM), True

    def _zero_stuff(x, stride):
        """Insert stride-1 zeros between samples: length L -> L*stride - (stride-1)."""
        if stride == 1:
            return x
        B, L, C = x.shape
        xr = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        xr = ttnn.reshape(xr, (B, L, 1, C))
        z = ttnn.zeros((B, L, stride - 1, C), dtype=xr.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        s = ttnn.concat([xr, z], dim=2, memory_config=DRAM)
        _free(xr, z)
        s = ttnn.reshape(s, (B, L * stride, C))
        s = ttnn.slice(s, [0, 0, 0], [B, L * stride - (stride - 1), C])
        return ttnn.to_layout(s, ttnn.TILE_LAYOUT)

    def _conv_core(x, taps, bias, k, dil, pl, pr, dtype=ttnn.bfloat16):
        """Stride-1 groups-1 Conv1d via tap-accumulate. `x`: TILE [1, L, C_in] -> [1, T_out, C_out]."""
        Cin = int(x.shape[2])
        xp, made = _pad_L(x, pl, pr)
        if xp.get_dtype() != dtype:
            xp2 = ttnn.typecast(xp, dtype)
            if made:
                _free(xp)
            xp, made = xp2, True
        Lp = int(xp.shape[1])
        T_out = Lp - dil * (k - 1)
        y = None
        for tap in range(k):
            s = tap * dil
            xs = ttnn.slice(xp, [0, s, 0], [1, s + T_out, Cin])
            yt = ttnn.matmul(xs, taps[tap], compute_kernel_config=compute_config, memory_config=DRAM)
            _free(xs)
            if y is None:
                y = yt
            else:
                y2 = ttnn.add(y, yt, memory_config=DRAM)
                _free(y, yt)
                y = y2
        if made:
            _free(xp)
        if bias is not None:
            y2 = ttnn.add(y, bias, memory_config=DRAM)
            _free(y)
            y = y2
        return y

    def do_conv(x, spec):
        """Apply a prepped Conv1d / ConvTranspose1d. `x` (TILE [1, L, C]) is left intact."""
        if spec["kind"] == "conv":
            return _conv_core(
                x, spec["taps"], spec["b"], spec["k"], spec["dil"], spec["pad"], spec["pad"], spec["dtype"]
            )
        xzs = _zero_stuff(x, spec["stride"])
        y = _conv_core(xzs, spec["taps"], spec["b"], spec["k"], 1, spec["ext_pad"], spec["ext_pad"], spec["dtype"])
        if xzs is not x:
            _free(xzs)
        return y

    def cond_bias(g_torch, spec):
        """1x1 conv on the (time-invariant) d-vector g -> per-channel bias [1, 1, out_ch]."""

        g = g_torch
        if isinstance(g, ttnn.Tensor):
            # Host-free: g is a device [1, cond_channels, 1] (or [1, C, T]); take the
            # first time step and move channels to the last axis via ttnn ops only.
            in_ch = spec["in_ch"]
            gg = ttnn.slice(g, [0, 0, 0], [1, in_ch, 1])  # [1, C, 1]
            gt = ttnn.permute(gg, (0, 2, 1))  # [1, 1, C]
            gt = ttnn.to_layout(gt, ttnn.TILE_LAYOUT)
            if gt.get_dtype() != ttnn.float32:
                gt = ttnn.typecast(gt, ttnn.float32)
        else:
            # g: [1, cond_channels, 1] -> device TILE [1, L=1, C=cond_channels]
            g = g.reshape(1, spec["in_ch"], -1)[:, :, :1].permute(0, 2, 1).contiguous().float()
            gt = ttnn.as_tensor(g, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)
        out = do_conv(gt, spec)  # [1, 1, out_ch]
        _free(gt)
        return out

    def _to_1LC(x):
        """Primary input x=[C, T] (or [B, C, T]) -> TILE [1, T, C]; host-free for device x."""
        import torch

        if isinstance(x, ttnn.Tensor):
            # channels-first [1, C, T] (or [C, T]) -> [1, T, C] via ttnn permute only.
            xx = x
            if len(xx.shape) == 2:
                xx = ttnn.reshape(xx, (1, int(xx.shape[0]), int(xx.shape[1])))
            xx = ttnn.permute(xx, (0, 2, 1))  # [1, T, C]
            xx = ttnn.to_layout(xx, ttnn.TILE_LAYOUT)
            if xx.get_dtype() != ttnn.float32:
                xx = ttnn.typecast(xx, ttnn.float32)
            return xx
        t = torch.as_tensor(x)
        if t.dim() == 2:
            t = t.unsqueeze(0)  # [1, C, T]
        t = t.reshape(1, t.shape[-2], t.shape[-1]).permute(0, 2, 1).contiguous().float()
        return ttnn.as_tensor(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=DRAM)

    def forward(x, g=None, *args, **kwargs):
        o = _to_1LC(x)  # [1, T, 1024]
        # conv_pre via the parametrized_conv1d leaf (channels-first [1, C, T]).
        o_cf = ttnn.transpose(o, 1, 2)  # [1, T, 1024] -> [1, 1024, T]
        _free(o)
        o_cf2 = conv_pre_fwd(o_cf)  # [1, 512, T]
        _free(o_cf)
        o = ttnn.transpose(o_cf2, 1, 2)  # [1, 512, T] -> [1, T, 512]
        _free(o_cf2)
        if has_cond:
            cb = cond_bias(g, cond_layer)
            prev = o
            o = ttnn.add(o, cb, memory_config=DRAM)
            _free(prev, cb)
        for i in range(num_upsamples):
            prev = o
            o = ttnn.leaky_relu(o, _LRELU, memory_config=DRAM)
            _free(prev)
            # ups[i] via the parametrized_conv_transpose1d leaf (channels-first).
            o_cf = ttnn.transpose(o, 1, 2)  # [1, T, C] -> [1, C, T]
            _free(o)
            o_cf2 = ups_fwd[i](o_cf)  # [1, C', T_out]
            _free(o_cf)
            o = ttnn.transpose(o_cf2, 1, 2)  # [1, C', T_out] -> [1, T_out, C']
            _free(o_cf2)
            if cond_in_each:
                cb = cond_bias(g, conds[i])
                prev = o
                o = ttnn.add(o, cb, memory_config=DRAM)
                _free(prev, cb)
            # MRF: mean over resblocks via the res_block1 leaf (channels-first);
            # the leaf leaves its input intact so o_cf is shared across kernels.
            o_cf = ttnn.transpose(o, 1, 2)  # [1, T, C] -> [1, C, T]
            _free(o)
            z_sum = None
            for j in range(num_kernels):
                zj = resblock_fwd[i * num_kernels + j](o_cf)  # [1, C, T] channels-first
                if z_sum is None:
                    z_sum = zj
                else:
                    z2 = ttnn.add(z_sum, zj, memory_config=DRAM)
                    _free(z_sum, zj)
                    z_sum = z2
            _free(o_cf)
            o_mean_cf = ttnn.multiply(z_sum, 1.0 / num_kernels, memory_config=DRAM)
            _free(z_sum)
            o = ttnn.transpose(o_mean_cf, 1, 2)  # [1, C, T] -> [1, T, C]
            _free(o_mean_cf)
        prev = o
        o = ttnn.leaky_relu(o, 0.01, memory_config=DRAM)  # final slope is torch's default
        _free(prev)
        # conv_post via the parametrized_conv1d leaf (channels-first [1, C, T]).
        o_cf = ttnn.transpose(o, 1, 2)  # [1, T, C] -> [1, C, T]
        _free(o)
        o_cf2 = conv_post_fwd(o_cf)  # [1, 1, T_out]
        _free(o_cf)
        o = ttnn.transpose(o_cf2, 1, 2)  # [1, 1, T_out] -> [1, T_out, 1]
        _free(o_cf2)
        prev = o
        o = ttnn.tanh(o, memory_config=DRAM)
        _free(prev)
        # Output is [1, T_out, 1] (== torch [1, 1, T_out] up to a squeeze); the PCC
        # comparison squeezes+flattens both, so no giant final reshape is needed
        # (reshaping a [1, 647424, 1] row-major tensor overflows L1 circular buffers).
        return o

    return forward


def hifigan_generator(*args, **kwargs):
    raise RuntimeError(
        "hifigan_generator requires build(device, torch_module) to bind the trained "
        "HiFi-GAN weights; the bare callable has no parameters."
    )

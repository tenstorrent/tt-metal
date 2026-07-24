# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN glue ops for the Kokoro-82M end-to-end pipeline.

These are the small building blocks the graduated stubs do NOT cover — the
leaf convolutions that live inside the (non-graduated) `decoder` / `generator`
containers (F0_conv, N_conv, asr_res, F0_proj, N_proj, noise_convs, the
ConvTranspose1d upsamplers, conv_post) plus the source-module linear/tanh.

Every function here is pure native ttnn (no torch host compute in the returned
closures). Conv1d/ConvTranspose1d use the same fp32 tap-accumulate matmul
recipe the graduated `adain_res_blk1d` / `custom_s_t_f_t` ports use, which is
numerically exact and avoids ttnn.conv1d's L1 halo OOM.
"""
from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


def compute_config(device):
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


# ttnn.matmul with fp32 inputs + HiFi4 still truncates the operand mantissas to
# ~tf32, giving ~1.5e-3 relative error AND a systematic ~0.9987 down-scale
# (round-toward-zero of the operands). Through the deep prosody predictor that
# bias accumulates into a coherent ~0.3% F0 under-estimate, which the NSF source
# integrates (cumsum) into a multi-radian phase drift -> the raw waveform
# decorrelates. Recover near-fp32 by a bf16 hi/lo operand split (3-term
# Dekker-style product): a*b = ah*bh + ah*bl + al*bh (al*bl negligible).
_HP_KEEP = ttnn.matmul  # captured native ttnn matmul (never re-patched)
_HP_LINEAR_KEEP = ttnn.linear


def _split_bf16(t):
    if t.get_dtype() != ttnn.float32:
        t = ttnn.typecast(t, ttnn.float32)
    hi = ttnn.typecast(ttnn.typecast(t, ttnn.bfloat16), ttnn.float32)
    lo = ttnn.subtract(t, hi)
    return hi, lo


_HP_KCHUNK = 64  # summing K in fp32 over <=64-wide chunks lowers the matmul's
# accumulator floor (~4.8e-4 -> ~3e-4) which further shrinks the
# residual F0 phase drift.


def _mm1(a, b, cc, mc):
    """Single near-fp32 matmul via bf16 hi/lo operand split (3-term)."""
    ah, al = _split_bf16(a)
    bh, bl = _split_bf16(b)
    y = _HP_KEEP(ah, bh, compute_kernel_config=cc, dtype=ttnn.float32, memory_config=mc)
    y = ttnn.add(y, _HP_KEEP(ah, bl, compute_kernel_config=cc, dtype=ttnn.float32, memory_config=mc), memory_config=mc)
    y = ttnn.add(y, _HP_KEEP(al, bh, compute_kernel_config=cc, dtype=ttnn.float32, memory_config=mc), memory_config=mc)
    return y


def hp_matmul(a, b, *, compute_kernel_config=None, memory_config=None, **kw):
    """Near-fp32 matmul: bf16 hi/lo operand split + fp32 K-chunk accumulation.
    Only chunks the plain a[...,M,K] @ b[K,N] case (2D weight); otherwise one shot."""
    mc = memory_config or _DRAM
    cc = compute_kernel_config
    try:
        K = int(a.shape[-1])
        b_is_2d = len(b.shape) == 2 and int(b.shape[0]) == K
        if not b_is_2d or K <= _HP_KCHUNK:
            return _mm1(a, b, cc, mc)
        Mrank = len(a.shape)
        y = None
        for k0 in range(0, K, _HP_KCHUNK):
            k1 = min(k0 + _HP_KCHUNK, K)
            a_start = [0] * Mrank
            a_start[-1] = k0
            a_end = [int(s) for s in a.shape]
            a_end[-1] = k1
            asl = ttnn.slice(a, a_start, a_end)
            bsl = ttnn.slice(b, [k0, 0], [k1, int(b.shape[1])])
            p = _mm1(asl, bsl, cc, mc)
            y = p if y is None else ttnn.add(y, p, memory_config=mc)
        return y
    except Exception:
        return _mm1(a, b, cc, mc)


def hp_linear(a, b, *, bias=None, compute_kernel_config=None, memory_config=None, **kw):
    """Near-fp32 linear (matmul + optional bias) via the hi/lo operand split."""
    y = hp_matmul(a, b, compute_kernel_config=compute_kernel_config, memory_config=memory_config)
    if bias is not None:
        y = ttnn.add(y, bias, memory_config=memory_config or _DRAM)
    return y


# Scoped precision override: when active, ttnn.matmul/linear run as a SINGLE native
# bf16 matmul (the caller's HiFi4 + fp32 dest-accumulate config) instead of the 3-term
# hi/lo split, collapsing ~14 dispatched ops/matmul down to ~1. Kokoro is feed-forward
# (no autoregressive argmax), so the near-fp32 emulation is unnecessary: a single bf16
# matmul holds the end-to-end log-spectrogram PCC well above the 0.95 gate (measured 0.98).
# run_tts_fast enables it for the whole traced forward; it stays a scoped flag (default off)
# so the dynamic run_tts / golden path keep the conservative near-fp32 split unchanged.
_HP_BYPASS = False


def set_hp_bypass(on):
    global _HP_BYPASS
    _HP_BYPASS = bool(on)


def hp_bypass_active():
    return _HP_BYPASS


def enable_hp_matmul():
    """Route ALL ttnn.matmul / ttnn.linear calls (this stub tree's stubs call
    them directly) through the near-fp32 hi/lo split. Idempotent."""
    if getattr(ttnn.matmul, "_is_hp", False):
        return

    def _mm(a, b, *args, **kw):
        try:
            if _HP_BYPASS:
                return _HP_KEEP(
                    a,
                    b,
                    compute_kernel_config=kw.get("compute_kernel_config"),
                    memory_config=kw.get("memory_config") or _DRAM,
                    dtype=ttnn.float32,
                )
            return hp_matmul(
                a, b, compute_kernel_config=kw.get("compute_kernel_config"), memory_config=kw.get("memory_config")
            )
        except Exception:
            return _HP_KEEP(a, b, *args, **kw)

    def _lin(a, b, *args, **kw):
        try:
            if _HP_BYPASS:
                y = _HP_KEEP(
                    a,
                    b,
                    compute_kernel_config=kw.get("compute_kernel_config"),
                    memory_config=kw.get("memory_config") or _DRAM,
                    dtype=ttnn.float32,
                )
                bias = kw.get("bias")
                if bias is not None:
                    y = ttnn.add(y, bias, memory_config=kw.get("memory_config") or _DRAM)
                return y
            return hp_linear(
                a,
                b,
                bias=kw.get("bias"),
                compute_kernel_config=kw.get("compute_kernel_config"),
                memory_config=kw.get("memory_config"),
            )
        except Exception:
            return _HP_LINEAR_KEEP(a, b, *args, **kw)

    _mm._is_hp = True
    _lin._is_hp = True
    ttnn.matmul = _mm
    ttnn.linear = _lin


def to_tt(device, x):
    if isinstance(x, ttnn.Tensor):
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        return x
    return ttnn.from_torch(
        x.contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )


def _const(device, t):
    return ttnn.from_torch(
        t.contiguous().float(),
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_DRAM,
    )


def effective_weight(conv):
    """Fold weight_norm (weight_g / weight_v) into a plain weight tensor."""
    import torch

    if hasattr(conv, "weight_g") and hasattr(conv, "weight_v"):
        return torch._weight_norm(conv.weight_v, conv.weight_g, 0)
    if hasattr(conv, "parametrizations") and "weight" in getattr(conv, "parametrizations", {}):
        return conv.weight  # parametrize materializes .weight on access
    return conv.weight


def build_conv1d(device, conv):
    """Native Conv1d (groups=1, any stride/dilation/padding). x:[B,Cin,T]->[B,Cout,Tout]."""

    cc = compute_config(device)
    w = effective_weight(conv).detach().float()  # [Cout, Cin, k]
    c_out, c_in, k = w.shape
    stride = int(conv.stride[0])
    dil = int(conv.dilation[0])
    pad = int(conv.padding[0]) if not isinstance(conv.padding, str) else 0
    if conv.groups != 1:
        raise RuntimeError(f"build_conv1d supports groups=1 (got {conv.groups})")
    taps = [_const(device, w[:, :, tap].t()) for tap in range(k)]  # [Cin,Cout]
    bias = _const(device, conv.bias.detach().reshape(1, 1, c_out)) if conv.bias is not None else None

    def _pad_L(x, p):
        if p == 0:
            return x
        B, L, C = x.shape
        z = ttnn.zeros((B, p, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.concat([z, x, z], dim=1, memory_config=_DRAM)

    def apply(x):
        x = to_tt(device, x)
        xtlc = ttnn.transpose(x, 1, 2)  # [B,T,Cin]
        xp = _pad_L(xtlc, pad)
        B = int(xp.shape[0])
        Lp = int(xp.shape[1])
        t_full = Lp - dil * (k - 1)  # stride-1 output length
        y = None
        for tap in range(k):
            s0 = tap * dil
            xs = ttnn.slice(xp, [0, s0, 0], [B, s0 + t_full, c_in])
            yt = ttnn.matmul(xs, taps[tap], compute_kernel_config=cc, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        if bias is not None:
            y = ttnn.add(y, bias, memory_config=_DRAM)
        y = _subsample_time(device, y, stride) if stride > 1 else y
        return ttnn.transpose(y, 1, 2)  # [B,Cout,Tout]

    return apply


def _subsample_time(device, y, stride):
    """y:[B,L,C] -> take every `stride`-th time step starting at 0."""
    B, L, C = [int(s) for s in y.shape]
    n_out = (L - 1) // stride + 1
    pad_to = (n_out - 1) * stride + 1
    # pad L up to n_out*stride so it reshapes cleanly
    full = n_out * stride
    if full > L:
        z = ttnn.zeros((B, full - L, C), dtype=y.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
        y = ttnn.concat([y, z], dim=1, memory_config=_DRAM)
    y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
    y = ttnn.reshape(y, (B, n_out, stride, C))
    y = ttnn.slice(y, [0, 0, 0, 0], [B, n_out, 1, C])
    y = ttnn.reshape(y, (B, n_out, C))
    return ttnn.to_layout(y, ttnn.TILE_LAYOUT)


def build_conv_transpose1d(device, conv):
    """Native ConvTranspose1d (groups=1 OR depthwise groups=Cin). x:[B,Cin,T]->[B,Cout,Tout].

    torch ConvTranspose1d weight is [Cin, Cout/groups, k]. Output:
      zero-stuff x by stride, full-conv with flipped weight, crop padding.
    """
    import torch

    cc = compute_config(device)
    w = effective_weight(conv).detach().float()  # [Cin, Cout/groups, k]
    c_in, c_out_g, k = w.shape
    stride = int(conv.stride[0])
    pad = int(conv.padding[0])
    out_pad = int(conv.output_padding[0])
    groups = int(conv.groups)
    bias = None
    if conv.bias is not None:
        c_out = c_out_g * groups
        bias = _const(device, conv.bias.detach().reshape(1, 1, c_out))

    if groups == 1:
        # per-tap weight [Cin, Cout] for matmul [B,t,Cin]@[Cin,Cout]
        wc = torch.flip(w, dims=[-1])  # flip kernel -> [Cin,Cout,k]
        taps = [_const(device, wc[:, :, tap]) for tap in range(k)]  # [Cin,Cout]

        def apply(x):
            x = to_tt(device, x)
            xtlc = ttnn.transpose(x, 1, 2)  # [B,T,Cin]
            xs = _zero_stuff(device, xtlc, stride)
            ext = (k - 1) - pad
            xp = _zero_pad_L(device, xs, ext, ext + out_pad)
            B = int(xp.shape[0])
            Lp = int(xp.shape[1])
            t_out = Lp - (k - 1)
            y = None
            for tap in range(k):
                sl = ttnn.slice(xp, [0, tap, 0], [B, tap + t_out, c_in])
                yt = ttnn.matmul(sl, taps[tap], compute_kernel_config=cc, memory_config=_DRAM)
                y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
            if bias is not None:
                y = ttnn.add(y, bias, memory_config=_DRAM)
            return ttnn.transpose(y, 1, 2)

        return apply

    # depthwise (groups == c_in, c_out == c_in): each channel independent 1D deconv.
    if not (groups == c_in and c_out_g == 1):
        raise RuntimeError(f"conv_transpose1d supports groups=1 or depthwise (got groups={groups})")
    wc = torch.flip(w, dims=[-1])  # [Cin,1,k]
    # per-tap channel weight as [1,1,Cin] broadcast multiply
    taps = [_const(device, wc[:, 0, tap].reshape(1, 1, c_in)) for tap in range(k)]

    def apply_dw(x):
        x = to_tt(device, x)
        xtlc = ttnn.transpose(x, 1, 2)  # [B,T,Cin]
        xs = _zero_stuff(device, xtlc, stride)
        ext = (k - 1) - pad
        xp = _zero_pad_L(device, xs, ext, ext + out_pad)
        B = int(xp.shape[0])
        Lp = int(xp.shape[1])
        t_out = Lp - (k - 1)
        y = None
        for tap in range(k):
            sl = ttnn.slice(xp, [0, tap, 0], [B, tap + t_out, c_in])
            yt = ttnn.multiply(sl, taps[tap], memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        if bias is not None:
            y = ttnn.add(y, bias, memory_config=_DRAM)
        return ttnn.transpose(y, 1, 2)

    return apply_dw


def _zero_stuff(device, x, stride):
    """[B,L,C] -> [B,(L-1)*stride+1,C] with stride-1 zeros between samples."""
    if stride == 1:
        return x
    B, L, C = [int(s) for s in x.shape]
    xr = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (B, L, 1, C))
    z = ttnn.zeros((B, L, stride - 1, C), dtype=xr.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    s = ttnn.concat([xr, z], dim=2, memory_config=_DRAM)
    s = ttnn.reshape(s, (B, L * stride, C))
    s = ttnn.slice(s, [0, 0, 0], [B, L * stride - (stride - 1), C])
    return ttnn.to_layout(s, ttnn.TILE_LAYOUT)


def _zero_pad_L(device, x, pl, pr):
    parts = []
    B, L, C = [int(s) for s in x.shape]
    if pl > 0:
        parts.append(ttnn.zeros((B, pl, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device))
    parts.append(x)
    if pr > 0:
        parts.append(ttnn.zeros((B, pr, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device))
    if len(parts) == 1:
        return x
    return ttnn.concat(parts, dim=1, memory_config=_DRAM)


def build_linear(device, lin):
    """nn.Linear -> native ttnn.linear closure. x[...,in]->[...,out]."""
    cc = compute_config(device)
    w_t = _const(device, lin.weight.detach().t())
    bias = _const(device, lin.bias.detach().reshape(1, -1)) if lin.bias is not None else None

    def apply(x):
        x = to_tt(device, x)
        return ttnn.linear(x, w_t, bias=bias, compute_kernel_config=cc, memory_config=_DRAM)

    return apply

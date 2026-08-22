# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `custom_s_t_f_t` (hexgrad/Kokoro-82M
`decoder.generator.stft`, a StyleTTS2/ISTFTNet `CustomSTFT`).

Reference torch forward does a full STFT -> iSTFT reconstruction:

    mag, phase = transform(x)        # conv1d(stride=hop) -> magnitude / atan2 phase
    return inverse(mag, phase, length=x.shape[-1])

Key algebraic simplification: the inverse rebuilds
`real = mag*cos(phase)`, `imag = mag*sin(phase)` from `mag=sqrt(re^2+im^2)`,
`phase=atan2(im,re)` — which collapses exactly back to the STFT's own
`real_out`/`imag_out` (the eps=1e-14 and the atan2 sign correction are
numerically negligible). So the whole pass is just two forward convs and two
transpose convs, with NO magnitude/phase/atan2 needed:

    xp        = replicate_pad(x, n_fft//2)              # center padding
    real_out  = conv1d(xp, W_fwd_real, stride=hop)      # [B, freq, frames]
    imag_out  = conv1d(xp, W_fwd_imag, stride=hop)
    real_rec  = conv_transpose1d(real_out, W_bwd_real, stride=hop)  # [B, 1, time]
    imag_rec  = conv_transpose1d(imag_out, W_bwd_imag, stride=hop)
    waveform  = (real_rec - imag_rec)[..., pad:-pad][..., :length]

Native ttnn: the strided conv1d is a stride-1 matmul tap-accumulate followed by a
time-subsample; the strided conv_transpose1d is the zero-stuff + full-conv
tap-accumulate recipe (as in the graduated `parametrized_conv_transpose1d`
port). All fp32 (HiFi4) for a clean PCC.
"""

from __future__ import annotations

import ttnn

_DRAM = ttnn.DRAM_MEMORY_CONFIG


HF_MODEL_ID = "hexgrad/Kokoro-82M"


def build(device, torch_module):
    """Bind STFT/iSTFT filter banks and return a native ttnn forward closure."""
    import torch

    m = torch_module
    n_fft = int(m.n_fft)
    hop = int(m.hop_length)
    center = bool(m.center)
    pad_len = n_fft // 2

    Wfr = m.weight_forward_real.detach().float()  # [freq, 1, n_fft]
    Wfi = m.weight_forward_imag.detach().float()
    Wbr = m.weight_backward_real.detach().float()  # [freq, 1, n_fft]
    Wbi = m.weight_backward_imag.detach().float()
    freq_bins, _, k = Wfr.shape

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    # Forward-conv taps: per k, W[:, 0, k] as [1, 1, freq] (outer product with x).
    def _fwd_taps(W):
        return [
            ttnn.from_torch(
                W[:, 0, tap].reshape(1, 1, freq_bins).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_DRAM,
            )
            for tap in range(k)
        ]

    fwd_real_taps = _fwd_taps(Wfr)
    fwd_imag_taps = _fwd_taps(Wfi)

    # Transpose-conv taps: Wc = flip(W, k).permute(1,0,2) -> [out=1, in=freq, k];
    # store tap [in=freq, out=1] for matmul [B, t, freq] @ [freq, 1] -> [B, t, 1].
    def _bwd_taps(W):
        wc = torch.flip(W, dims=[-1]).permute(1, 0, 2).contiguous()  # [1, freq, k]
        return [
            ttnn.from_torch(
                wc[:, :, tap].t().contiguous(),
                dtype=ttnn.float32,  # [freq, 1]
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_DRAM,
            )
            for tap in range(k)
        ]

    bwd_real_taps = _bwd_taps(Wbr)
    bwd_imag_taps = _bwd_taps(Wbi)

    ext_pad = (k - 1) - 0  # transpose conv has padding=0 -> full-conv crop = k-1

    def _rm(x):
        return ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    def _tile(x):
        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    def _replicate_pad(x, p):
        # x: [B, L, 1] -> replicate first/last time step p times each side.
        B = x.shape[0]
        left = ttnn.slice(x, [0, 0, 0], [B, 1, 1])
        right = ttnn.slice(x, [0, int(x.shape[1]) - 1, 0], [B, int(x.shape[1]), 1])
        parts = [left] * p + [x] + [right] * p
        return ttnn.concat(parts, dim=1, memory_config=_DRAM)

    def _zero_pad_L(x, p):
        if p == 0:
            return x
        B, L, C = x.shape
        z = ttnn.zeros((B, p, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.concat([z, x, z], dim=1, memory_config=_DRAM)

    def _fwd_conv(xp, taps):
        # xp: [B, Lp, 1]; stride-1 tap-accumulate -> [B, L1, freq], then subsample by hop.
        B = int(xp.shape[0])
        Lp = int(xp.shape[1])
        L1 = Lp - (k - 1)
        y = None
        for tap in range(k):
            xs = ttnn.slice(xp, [0, tap, 0], [B, tap + L1, 1])  # [B, L1, 1]
            yt = ttnn.matmul(xs, taps[tap], compute_kernel_config=compute_config, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        # subsample time by hop: pad L1 up to a multiple of hop, reshape, take index 0.
        n_frames = (L1 - 1) // hop + 1
        need = (n_frames - 1) * hop + 1
        pad_to = n_frames * hop
        if pad_to > L1:
            B2, _, C2 = y.shape
            z = ttnn.zeros((B2, pad_to - L1, C2), dtype=y.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
            y = ttnn.concat([y, z], dim=1, memory_config=_DRAM)
        y = _rm(y)
        y = ttnn.reshape(y, (B, n_frames, hop, freq_bins))
        y = ttnn.slice(y, [0, 0, 0, 0], [B, n_frames, 1, freq_bins])
        y = ttnn.reshape(y, (B, n_frames, freq_bins))
        return _tile(y)

    def _zero_stuff(x):
        # [B, L, C] -> [B, L*hop - (hop-1), C] with hop-1 zeros between samples.
        if hop == 1:
            return x
        B, L, C = x.shape
        xr = ttnn.reshape(_rm(x), (B, L, 1, C))
        z = ttnn.zeros((B, L, hop - 1, C), dtype=xr.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        s = ttnn.concat([xr, z], dim=2, memory_config=_DRAM)
        s = ttnn.reshape(s, (B, L * hop, C))
        s = ttnn.slice(s, [0, 0, 0], [B, L * hop - (hop - 1), C])
        return _tile(s)

    def _bwd_conv(x, taps):
        # x: [B, frames, freq]; strided transpose conv -> [B, t_out, 1].
        xs = _zero_stuff(x)
        xp = _zero_pad_L(xs, ext_pad)
        B = int(xp.shape[0])
        Lp = int(xp.shape[1])
        t_out = Lp - (k - 1)
        y = None
        for tap in range(k):
            xsl = ttnn.slice(xp, [0, tap, 0], [B, tap + t_out, freq_bins])
            yt = ttnn.matmul(xsl, taps[tap], compute_kernel_config=compute_config, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        return y  # [B, t_out, 1]

    def forward(x, *args, **kwargs):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(
                x.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
            )
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)

        shp = list(x.shape)
        if len(shp) == 2:
            B, T = int(shp[0]), int(shp[1])
            x = ttnn.reshape(x, (B, T, 1))
        else:
            B, T = int(shp[0]), int(shp[1])

        xp = _replicate_pad(x, pad_len) if center else x

        real_out = _fwd_conv(xp, fwd_real_taps)  # [B, frames, freq]
        imag_out = _fwd_conv(xp, fwd_imag_taps)

        real_rec = _bwd_conv(real_out, bwd_real_taps)  # [B, time, 1]
        imag_rec = _bwd_conv(imag_out, bwd_imag_taps)
        wav = ttnn.subtract(real_rec, imag_rec)  # [B, time, 1]

        L = int(wav.shape[1])
        start = pad_len if center else 0
        end = (L - pad_len) if center else L
        if start > 0 or end < L:
            wav = ttnn.slice(wav, [0, start, 0], [B, end, 1])
        if int(wav.shape[1]) > T:
            wav = ttnn.slice(wav, [0, 0, 0], [B, T, 1])

        # [B, time, 1] -> [B, 1, time] to match the reference (B, 1, T).
        return ttnn.transpose(wav, 1, 2)

    return forward


def _stft_internals(device, torch_module):
    """Shared filter-bank / tap construction for the separate transform & inverse
    builders below (same recipe as build()'s fused path)."""
    import torch

    m = torch_module
    n_fft = int(m.n_fft)
    hop = int(m.hop_length)
    center = bool(m.center)
    pad_len = n_fft // 2
    Wfr = m.weight_forward_real.detach().float()
    Wfi = m.weight_forward_imag.detach().float()
    Wbr = m.weight_backward_real.detach().float()
    Wbi = m.weight_backward_imag.detach().float()
    freq_bins, _, k = Wfr.shape
    cc = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )

    def _fwd_taps(W):
        return [
            ttnn.from_torch(
                W[:, 0, tap].reshape(1, 1, freq_bins).contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_DRAM,
            )
            for tap in range(k)
        ]

    def _bwd_taps(W):
        wc = torch.flip(W, dims=[-1]).permute(1, 0, 2).contiguous()  # [1,freq,k]
        return [
            ttnn.from_torch(
                wc[:, :, tap].t().contiguous(),
                dtype=ttnn.float32,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=_DRAM,
            )
            for tap in range(k)
        ]

    return {
        "n_fft": n_fft,
        "hop": hop,
        "center": center,
        "pad_len": pad_len,
        "freq_bins": freq_bins,
        "k": k,
        "cc": cc,
        "fwd_real": _fwd_taps(Wfr),
        "fwd_imag": _fwd_taps(Wfi),
        "bwd_real": _bwd_taps(Wbr),
        "bwd_imag": _bwd_taps(Wbi),
        "ext_pad": (k - 1),
    }


def build_stft_transform(device, torch_module):
    """CustomSTFT.transform: waveform [B,T] -> (magnitude, phase) each [B, freq, frames]."""

    I = _stft_internals(device, torch_module)
    freq_bins, k, hop, pad_len, center, cc = (I["freq_bins"], I["k"], I["hop"], I["pad_len"], I["center"], I["cc"])
    fwd_real, fwd_imag = I["fwd_real"], I["fwd_imag"]

    def _replicate_pad(x, p):
        B = int(x.shape[0])
        left = ttnn.slice(x, [0, 0, 0], [B, 1, 1])
        right = ttnn.slice(x, [0, int(x.shape[1]) - 1, 0], [B, int(x.shape[1]), 1])
        return ttnn.concat([left] * p + [x] + [right] * p, dim=1, memory_config=_DRAM)

    def _fwd_conv(xp, taps):
        B = int(xp.shape[0])
        Lp = int(xp.shape[1])
        L1 = Lp - (k - 1)
        y = None
        for tap in range(k):
            xs = ttnn.slice(xp, [0, tap, 0], [B, tap + L1, 1])
            yt = ttnn.matmul(xs, taps[tap], compute_kernel_config=cc, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        n_frames = (L1 - 1) // hop + 1
        pad_to = n_frames * hop
        if pad_to > L1:
            z = ttnn.zeros((B, pad_to - L1, freq_bins), dtype=y.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
            y = ttnn.concat([y, z], dim=1, memory_config=_DRAM)
        y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        y = ttnn.reshape(y, (B, n_frames, hop, freq_bins))
        y = ttnn.slice(y, [0, 0, 0, 0], [B, n_frames, 1, freq_bins])
        y = ttnn.reshape(y, (B, n_frames, freq_bins))
        return ttnn.to_layout(y, ttnn.TILE_LAYOUT)

    def transform(x):
        if not isinstance(x, ttnn.Tensor):
            x = ttnn.from_torch(
                x.contiguous().float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
            )
        if x.get_dtype() != ttnn.float32:
            x = ttnn.typecast(x, ttnn.float32)
        shp = list(x.shape)
        B, T = int(shp[0]), int(shp[-1])
        x = ttnn.reshape(x, (B, T, 1))
        xp = _replicate_pad(x, pad_len) if center else x
        real_out = _fwd_conv(xp, fwd_real)  # [B, frames, freq]
        imag_out = _fwd_conv(xp, fwd_imag)
        mag = ttnn.sqrt(ttnn.add(ttnn.add(ttnn.multiply(real_out, real_out), ttnn.multiply(imag_out, imag_out)), 1e-14))
        phase = ttnn.atan2(imag_out, real_out)
        # -> [B, freq, frames] channels-first
        return ttnn.transpose(mag, 1, 2), ttnn.transpose(phase, 1, 2)

    return transform


def build_stft_inverse(device, torch_module):
    """CustomSTFT.inverse: (magnitude, phase) [B,freq,frames] -> waveform [B,1,T]."""
    I = _stft_internals(device, torch_module)
    freq_bins, k, hop, pad_len, center, cc, ext_pad = (
        I["freq_bins"],
        I["k"],
        I["hop"],
        I["pad_len"],
        I["center"],
        I["cc"],
        I["ext_pad"],
    )
    bwd_real, bwd_imag = I["bwd_real"], I["bwd_imag"]

    def _zero_stuff(x):
        if hop == 1:
            return x
        B, L, C = [int(v) for v in x.shape]
        xr = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), (B, L, 1, C))
        z = ttnn.zeros((B, L, hop - 1, C), dtype=xr.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        s = ttnn.concat([xr, z], dim=2, memory_config=_DRAM)
        s = ttnn.reshape(s, (B, L * hop, C))
        s = ttnn.slice(s, [0, 0, 0], [B, L * hop - (hop - 1), C])
        return ttnn.to_layout(s, ttnn.TILE_LAYOUT)

    def _zero_pad_L(x, p):
        if p == 0:
            return x
        B, L, C = [int(v) for v in x.shape]
        z = ttnn.zeros((B, p, C), dtype=x.get_dtype(), layout=ttnn.TILE_LAYOUT, device=device)
        return ttnn.concat([z, x, z], dim=1, memory_config=_DRAM)

    def _bwd_conv(x, taps):
        xs = _zero_stuff(x)
        xp = _zero_pad_L(xs, ext_pad)
        B = int(xp.shape[0])
        Lp = int(xp.shape[1])
        t_out = Lp - (k - 1)
        y = None
        for tap in range(k):
            xsl = ttnn.slice(xp, [0, tap, 0], [B, tap + t_out, freq_bins])
            yt = ttnn.matmul(xsl, taps[tap], compute_kernel_config=cc, memory_config=_DRAM)
            y = yt if y is None else ttnn.add(y, yt, memory_config=_DRAM)
        return y  # [B, t_out, 1]

    def inverse(magnitude, phase, length=None):
        # inputs [B, freq, frames] -> [B, frames, freq]
        mag = ttnn.transpose(magnitude, 1, 2)
        ph = ttnn.transpose(phase, 1, 2)
        real_part = ttnn.multiply(mag, ttnn.cos(ph))
        imag_part = ttnn.multiply(mag, ttnn.sin(ph))
        real_rec = _bwd_conv(real_part, bwd_real)
        imag_rec = _bwd_conv(imag_part, bwd_imag)
        wav = ttnn.subtract(real_rec, imag_rec)  # [B, time, 1]
        B = int(wav.shape[0])
        L = int(wav.shape[1])
        if center:
            wav = ttnn.slice(wav, [0, pad_len, 0], [B, L - pad_len, 1])
        if length is not None and int(wav.shape[1]) > length:
            wav = ttnn.slice(wav, [0, 0, 0], [B, length, 1])
        return ttnn.transpose(wav, 1, 2)  # [B, 1, time]

    return inverse


def custom_s_t_f_t(*args, **kwargs):
    raise RuntimeError(
        "custom_s_t_f_t requires build(device, torch_module) to bind the STFT "
        "filter banks; the bare callable has no parameters."
    )

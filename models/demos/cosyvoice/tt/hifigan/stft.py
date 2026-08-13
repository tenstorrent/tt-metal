# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Forward STFT of the NSF excitation -- the transposed twin of istft.py.

HiFT's `decode` opens by transforming the harmonic excitation:

    s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)     # [B, 18, T]

which is `torch.stft(x, 16, 4, 16, window=hann)` -- so `center=True` and
`pad_mode='reflect'` by default.

At n_fft = 16 this decomposes the same way the inverse does, and lands on the
same 18-row shape:

    X[k,t] = sum_n frames[n,t]*w[n] * exp(-2*pi*i*k*n/N)

so with C[k,n] = cos(2*pi*k*n/N) and S[k,n] = sin(2*pi*k*n/N),

    real = C @ (windowed frames),  imag = -S @ (windowed frames)

and stacking `[C ; -S]` as one `[18, 16]` matrix produces `[real ; imag]`
**directly in the concatenated layout `decode` wants** -- the concat is free, it
is just how the matmul comes out.

Framing plus windowing is a single strided conv1d with a diagonal kernel
`W[o,0,k] = delta(o==k) * hann[k]`, exactly mirroring the transposed convolution
that istft.py uses for overlap-add.

Net: reflect-pad -> conv1d -> one matmul. Same op inventory as the inverse, and
no FFT anywhere.
"""
from __future__ import annotations

import numpy as np
import torch
from loguru import logger

import ttnn

from .istft import periodic_hann


def stft_basis(n_fft: int = 16, dtype=np.float32) -> np.ndarray:
    """The [2*bins, n_fft] matrix [C ; -S] mapping windowed frames to [real; imag].

    Note this is NOT the transpose of istft_basis: the inverse folds Hermitian
    weights (2x for bins 1..N/2-1) and a 1/N scale into its matrix, and the
    forward transform does neither.
    """
    bins = n_fft // 2 + 1
    k = np.arange(bins, dtype=np.float64).reshape(-1, 1)
    n = np.arange(n_fft, dtype=np.float64).reshape(1, -1)
    ang = 2.0 * np.pi * k * n / n_fft
    return np.concatenate([np.cos(ang), -np.sin(ang)], axis=0).astype(dtype)


class TtStft:
    """Device-resident forward STFT. Output is `[B, 2*bins, T]` (real over imag)."""

    def __init__(
        self,
        device,
        n_fft: int = 16,
        hop: int = 4,
        window: torch.Tensor | np.ndarray | None = None,
        dtype=ttnn.bfloat16,
        center: bool = True,
    ):
        self.device = device
        self.n_fft, self.hop = n_fft, hop
        self.bins = n_fft // 2 + 1
        self.dtype = dtype
        self.center = center

        if window is None:
            window = periodic_hann(n_fft)
        if isinstance(window, torch.Tensor):
            window = window.detach().cpu().numpy()
        self.window = window.astype(np.float32)

        # [1, 2*bins, n_fft] so batched matmul broadcasts.
        self.basis = ttnn.from_torch(
            torch.from_numpy(stft_basis(n_fft)).unsqueeze(0), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )

        # Diagonal framing kernel with the window folded in, [out_ch, in_ch, k].
        w = torch.zeros(n_fft, 1, n_fft, dtype=torch.float32)
        w[torch.arange(n_fft), 0, torch.arange(n_fft)] = torch.from_numpy(self.window)
        self.frame_weight = ttnn.from_torch(w, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self.conv_config = ttnn.Conv1dConfig(weights_dtype=dtype, deallocate_activation=False)

        # The same kernel as OIHW, which is what `prepare_conv_weights` wants. Kept
        # alongside the [O, I, K] form so the unprepared path still works when
        # preparation is unavailable -- see `_prepared`.
        w4 = torch.zeros(n_fft, 1, 1, n_fft, dtype=torch.float32)
        w4[torch.arange(n_fft), 0, 0, torch.arange(n_fft)] = torch.from_numpy(self.window)
        self._weight_4d = ttnn.from_torch(w4, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._prep_cache: dict[tuple[int, int], ttnn.Tensor] = {}

        # Reversal operator for reflect padding. TTNN has no `flip`, and
        # `ttnn.pad` offers only Replicate and Zeros -- no reflect mode -- so the
        # reflection is composed from an exchange (anti-diagonal) matrix:
        # reversing a length-p slice along the time axis is J[p,p] @ slice[B,p,1].
        # At n_fft=16 that is an 8x8 constant, so the whole thing is one tiny matmul
        # per side rather than a host round-trip.
        p = n_fft // 2
        self.pad = p
        J = torch.flip(torch.eye(p), dims=[0]).unsqueeze(0)  # [1, p, p]
        self.exchange = ttnn.from_torch(J, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    def n_frames(self, length: int) -> int:
        padded = length + (self.n_fft if self.center else 0)
        return (padded - self.n_fft) // self.hop + 1

    _warned = False

    def _prepared(self, x, length: int, batch_size: int):
        """Pre-tilized framing weight, cached per input geometry.

        Same reasoning as `TtConv1d._prepared`, and the same consequence: handing
        `ttnn.conv1d` a host-layout weight makes it redo the layout transform on every
        call, which is host traffic, which is what a trace forbids. Until this was
        hoisted out, the vocoder could not be captured at all -- capture died on
        `Writes are not supported during trace capture` inside this one op.

        It buys nothing on its own (measured `1.00x` untraced). Its whole value is
        making `TtHiFTGenerator.decode` traceable, which is worth `3.3x` on Blackhole
        and `1.6x` on Wormhole. Output is bit-identical either way (`max|d| 0.000e+00`).

        `length` is the *padded* length, since that is what the conv sees.
        """
        key = (length, batch_size)
        if key in self._prep_cache:
            return self._prep_cache[key]
        try:
            w = ttnn.prepare_conv_weights(
                weight_tensor=self._weight_4d,
                weights_format="OIHW",
                has_bias=False,
                input_memory_config=x.memory_config(),
                input_layout=x.layout,
                in_channels=1,
                out_channels=self.n_fft,
                batch_size=batch_size,
                input_height=1,
                input_width=length,
                kernel_size=(1, self.n_fft),
                stride=(1, self.hop),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=self.device,
                input_dtype=self.dtype,
                conv_config=self.conv_config,
            )
        except Exception as e:  # noqa: BLE001
            # Logged rather than swallowed: a silent fallback here looks like a working
            # fast path and the only symptom is a trace capture that fails elsewhere.
            if not TtStft._warned:
                TtStft._warned = True
                logger.warning(f"stft prepare_conv_weights unavailable, stays untraceable: {str(e)[:200]}")
            w = self.frame_weight
        self._prep_cache[key] = w
        return w

    def __call__(self, x, length: int, batch_size: int = 1):
        """x: ttnn [B, L, 1] (channels-last, single channel) -> [B, 2*bins, T]."""
        # center=True reflect-pads n_fft//2 on each side. torch's reflect pad emits
        #   left  = reverse(x[1 : p+1])
        #   right = reverse(x[L-p-1 : L-1])
        # i.e. the edge sample itself is not repeated. Reversal is the exchange
        # matmul built in __init__ -- a host round-trip here would defeat the point
        # of keeping the vocoder on device.
        if self.center:
            p = self.pad
            ls = ttnn.slice(x, [0, 1, 0], [batch_size, p + 1, 1])
            rs = ttnn.slice(x, [0, length - p - 1, 0], [batch_size, length - 1, 1])
            left = ttnn.matmul(self.exchange, ls)
            right = ttnn.matmul(self.exchange, rs)
            ttnn.deallocate(ls)
            ttnn.deallocate(rs)
            x = ttnn.concat([left, x, right], dim=1)
            ttnn.deallocate(left)
            ttnn.deallocate(right)
            length = length + 2 * p

        frames, n_frames = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self._prepared(x, length, batch_size),
            device=self.device,
            in_channels=1,
            out_channels=self.n_fft,
            batch_size=batch_size,
            input_length=length,
            kernel_size=self.n_fft,
            stride=self.hop,
            padding=0,
            dilation=1,
            groups=1,
            conv_config=self.conv_config,
            dtype=self.dtype,
            return_output_dim=True,
        )
        # conv1d yields [B, T, n_fft]; the DFT wants [B, n_fft, T].
        frames = ttnn.reshape(frames, (batch_size, n_frames, self.n_fft))
        frames = ttnn.permute(frames, (0, 2, 1))
        spec = ttnn.matmul(self.basis, frames)  # [B, 2*bins, T]
        ttnn.deallocate(frames)
        return spec, n_frames

    @staticmethod
    def torch_reference(
        x: torch.Tensor, window: torch.Tensor, n_fft: int = 16, hop: int = 4, center: bool = True
    ) -> torch.Tensor:
        """The same sequence in torch, returning the concatenated [real; imag]."""
        b, length = x.shape
        if center:
            p = n_fft // 2
            x = torch.nn.functional.pad(x.unsqueeze(1), (p, p), mode="reflect").squeeze(1)
        W = torch.zeros(n_fft, 1, n_fft, dtype=x.dtype)
        W[torch.arange(n_fft), 0, torch.arange(n_fft)] = window.to(x.dtype)
        frames = torch.nn.functional.conv1d(x.unsqueeze(1), W, stride=hop)  # [B, n_fft, T]
        M = torch.from_numpy(stft_basis(n_fft)).to(x.dtype)
        return M @ frames

# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Inverse STFT for the HiFT vocoder, built entirely from ops TTNN already has.

TTNN has no FFT of any kind (verified absent across ttnn/ and tt_metal/). This
module removes the need for one, by exploiting the fact that CosyVoice's HiFT
uses `n_fft = 16`.

The identity
------------
At n_fft = 16 the inverse DFT of 9 one-sided bins is a fixed 16x9 real matrix pair
-- smaller than one 32x32 tile. Folding in Hermitian symmetry:

    x[n] = (1/16)*( X0.re + 2*sum_{k=1..7}(Xk.re*cos(2pi k n/16) - Xk.im*sin(2pi k n/16))
                    + X8.re*cos(pi n) )

so with weights w0 = w8 = 1, w1..7 = 2 folded into C and S, and S[:,0] = S[:,8] = 0
to reproduce irfft's treatment of the DC and Nyquist bins:

    frames[16,T] = C[16,9] @ real[9,T] - S[16,9] @ imag[9,T]

Those two matmuls collapse into ONE by concatenating along the bin axis:

    frames[16,T] = M[16,18] @ concat(real, imag)[18,T],   M = [C | -S]

which is not just a micro-optimisation: HiFT's `conv_post` already emits 18 channels
(n_fft + 2) that are split into magnitude and phase, so the 18-row stacked form is
the natural shape of the data at this boundary rather than an imposed one.

Windowing and overlap-add then fuse into a single transposed convolution. OLA is
`out[t*hop + j] += frames[j,t] * w[j]`, and conv_transpose1d computes
`out[o, t*s + k] += in[i,t] * W[i,o,k]`. Setting one output channel and a diagonal
kernel `W[i,0,k] = delta(i==k) * hann[k]` makes them the same operation.

Finally torch.istft divides by the overlap-added window^2 (the NOLA envelope), which
depends only on T -- so it is precomputed on host and applied as one multiply by its
reciprocal, and `center=True` trims n_fft//2 samples from each end.

Net: matmul + conv_transpose2d + multiply. No new C++, no host round-trip.

Verified against the real vocoder tensors captured in tests/golden/hift.istft.npz
(magnitude spanning 1.06e-13 .. 1.21e+01):  PCC 1.0000000000 in fp32,
0.9999765688 with bfloat16-rounded inputs.
"""
from __future__ import annotations

import numpy as np
import torch
from loguru import logger

import ttnn


def periodic_hann(n_fft: int) -> np.ndarray:
    """Exactly scipy.signal.get_window("hann", n_fft, fftbins=True).

    Spelled out rather than imported so this package does not pull in scipy: the
    reference builds its window with that call, and fftbins=True means *periodic*
    (sym=False), i.e. 0.5 - 0.5*cos(2*pi*n/N) over n = 0..N-1 -- not the symmetric
    variant, which would be divided by N-1 and would break NOLA reconstruction.
    """
    n = np.arange(n_fft, dtype=np.float64)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * n / n_fft)).astype(np.float32)


def istft_basis(n_fft: int = 16, dtype=np.float32) -> np.ndarray:
    """The [n_fft, 2*bins] matrix M = [C | -S] that inverts a one-sided rFFT.

    Host-side and device-free, so it is unit-testable on its own.
    """
    bins = n_fft // 2 + 1
    n = np.arange(n_fft, dtype=np.float64).reshape(-1, 1)
    k = np.arange(bins, dtype=np.float64).reshape(1, -1)
    ang = 2.0 * np.pi * n * k / n_fft

    w = np.full((bins,), 2.0)
    w[0] = 1.0
    if n_fft % 2 == 0:
        w[-1] = 1.0  # Nyquist bin is its own conjugate, so it is not doubled

    C = (np.cos(ang) * w) / n_fft
    S = (np.sin(ang) * w) / n_fft
    S[:, 0] = 0.0  # DC has no imaginary part in a real signal's transform
    if n_fft % 2 == 0:
        S[:, -1] = 0.0  # nor does Nyquist
    return np.concatenate([C, -S], axis=1).astype(dtype)


def nola_envelope(window: np.ndarray, n_frames: int, hop: int) -> np.ndarray:
    """Overlap-added window^2, i.e. what torch.istft divides by.

    Depends only on the frame count, so it is a constant per utterance length.
    """
    n_fft = window.shape[0]
    length = (n_frames - 1) * hop + n_fft
    env = np.zeros(length, dtype=np.float64)
    sq = window.astype(np.float64) ** 2
    for t in range(n_frames):
        env[t * hop : t * hop + n_fft] += sq
    return env


class TtIStft:
    """Device-resident iSTFT. Constant tensors are built once, per utterance length.

    Any tensor created inside a captured trace is pinned to a fixed device address
    (CLAUDE.md ss.7), so every constant here is allocated up front and the T-dependent
    envelope is cached rather than rebuilt inside the forward path.
    """

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
        self.n_fft = n_fft
        self.hop = hop
        self.bins = n_fft // 2 + 1
        self.dtype = dtype
        self.center = center

        if window is None:
            window = periodic_hann(n_fft)
        if isinstance(window, torch.Tensor):
            window = window.detach().cpu().numpy()
        self.window = window.astype(np.float32)

        # M = [C | -S], shape [n_fft, 2*bins]; batched matmul wants a leading dim.
        basis = istft_basis(n_fft)
        self.basis = ttnn.from_torch(
            torch.from_numpy(basis).unsqueeze(0),  # [1, n_fft, 2*bins]
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        # Diagonal OLA kernel in TTNN's (C, O/G, K_H, K_W) weight order.
        w = torch.zeros(n_fft, 1, 1, n_fft, dtype=torch.float32)
        w[torch.arange(n_fft), 0, 0, torch.arange(n_fft)] = torch.from_numpy(self.window)
        self.ola_weight = ttnn.from_torch(w, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)

        # prepare_conv_transpose2d_weights asserts conv_config.weights_dtype.has_value(),
        # so the config exists for preparation's sake -- see the same note in upsample.py.
        self.conv_config = ttnn.Conv2dConfig(weights_dtype=dtype)

        self._env_cache: dict[int, ttnn.Tensor] = {}
        self._prep_cache: dict[tuple[int, int], tuple] = {}

    # -- constants ---------------------------------------------------------
    def _envelope(self, n_frames: int):
        """1 / NOLA envelope for this frame count, as [1, 1, L_out, 1]."""
        if n_frames not in self._env_cache:
            env = nola_envelope(self.window, n_frames, self.hop)
            inv = (1.0 / np.maximum(env, 1e-11)).astype(np.float32)
            if self.center:
                inv = inv[self.n_fft // 2 : -(self.n_fft // 2)]
            t = torch.from_numpy(inv).reshape(1, 1, -1, 1)
            self._env_cache[n_frames] = ttnn.from_torch(
                t, dtype=self.dtype, layout=ttnn.TILE_LAYOUT, device=self.device
            )
        return self._env_cache[n_frames]

    # -- forward -----------------------------------------------------------
    _warned = False

    def _prepared(self, nhwc, n_frames: int, batch_size: int):
        """Pre-tilized overlap-add weight, cached per input geometry.

        The mirror of `TtStft._prepared`, and for the same reason: `conv_transpose2d`
        prepares a host-layout weight on every call, and that host traffic is what a
        trace forbids. These two convolutions were the only things keeping
        `TtHiFTGenerator.decode` from being captured.

        Returns `(weight, conv_config)`. The config is `None` on the fallback path
        because the unprepared call never passed one, and adding it there would change
        an op that is working.
        """
        key = (n_frames, batch_size)
        if key in self._prep_cache:
            return self._prep_cache[key]
        try:
            w = ttnn.prepare_conv_transpose2d_weights(
                weight_tensor=self.ola_weight,
                weights_format="IOHW",
                has_bias=False,
                input_memory_config=nhwc.memory_config(),
                input_layout=nhwc.layout,
                in_channels=self.n_fft,
                out_channels=1,
                batch_size=batch_size,
                input_height=1,
                input_width=n_frames,
                kernel_size=(1, self.n_fft),
                stride=(1, self.hop),
                padding=(0, 0),
                dilation=(1, 1),
                groups=1,
                device=self.device,
                input_dtype=self.dtype,
                conv_config=self.conv_config,
            )
            entry = (w, self.conv_config)
        except Exception as e:  # noqa: BLE001
            if not TtIStft._warned:
                TtIStft._warned = True
                logger.warning(f"istft prepare_conv_transpose2d_weights unavailable, stays untraceable: {str(e)[:200]}")
            entry = (self.ola_weight, None)
        self._prep_cache[key] = entry
        return entry

    def __call__(self, real, imag):
        """real, imag: ttnn [B, bins, T] -> waveform ttnn [B, 1, L, 1] (NHWC).

        Returns NHWC because that is what conv_transpose2d produces and what the
        surrounding vocoder convolutions consume; the caller squeezes it once at the
        very end rather than permuting mid-graph.
        """
        b, bins, n_frames = real.shape
        assert bins == self.bins, f"expected {self.bins} bins, got {bins}"

        # --- 1. inverse DFT as a single matmul -----------------------------
        ri = ttnn.concat([real, imag], dim=1)  # [B, 2*bins, T]
        frames = ttnn.matmul(self.basis, ri)  # [B, n_fft, T]
        ttnn.deallocate(ri)

        # --- 2. window + overlap-add as one transposed convolution ---------
        # to NHWC: [B, n_fft, T] -> [B, 1, T, n_fft]
        nhwc = ttnn.permute(frames, (0, 2, 1))  # [B, T, n_fft]
        ttnn.deallocate(frames)
        nhwc = ttnn.reshape(nhwc, (b, 1, n_frames, self.n_fft))

        weight, conv_config = self._prepared(nhwc, n_frames, b)
        conv_kw = dict(
            input_tensor=nhwc,
            weight_tensor=weight,
            device=self.device,
            in_channels=self.n_fft,
            out_channels=1,
            batch_size=b,
            input_height=1,
            input_width=n_frames,
            kernel_size=(1, self.n_fft),
            stride=(1, self.hop),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            dtype=self.dtype,
        )
        if conv_config is not None:
            conv_kw["conv_config"] = conv_config
        out = ttnn.conv_transpose2d(**conv_kw)
        if isinstance(out, (tuple, list)):
            out = out[0]
        ttnn.deallocate(nhwc)

        # --- 3. center=True trim, then NOLA normalisation -------------------
        length = (n_frames - 1) * self.hop + self.n_fft
        out = ttnn.reshape(out, (b, 1, length, 1))
        if self.center:
            out = ttnn.slice(out, [0, 0, self.n_fft // 2, 0], [b, 1, length - self.n_fft // 2, 1])
        out = ttnn.multiply(out, self._envelope(n_frames))
        return out

    # -- host reference, for tests and for deriving the constants -----------
    @staticmethod
    def torch_reference(
        real: torch.Tensor, imag: torch.Tensor, window: torch.Tensor, n_fft: int = 16, hop: int = 4, center: bool = True
    ):
        """The same sequence in torch. Used to separate 'the identity is wrong'
        from 'the TTNN op behaved unexpectedly' when a PCC test fails."""
        b, _, n_frames = real.shape
        M = torch.from_numpy(istft_basis(n_fft)).to(real.dtype)
        frames = M @ torch.cat([real, imag], dim=1)

        W = torch.zeros(n_fft, 1, n_fft, dtype=real.dtype)
        W[torch.arange(n_fft), 0, torch.arange(n_fft)] = window.to(real.dtype)
        out = torch.nn.functional.conv_transpose1d(frames, W, stride=hop)

        env = torch.from_numpy(nola_envelope(window.numpy(), n_frames, hop)).to(real.dtype)
        out = out / env.clamp_min(1e-11)
        if center:
            out = out[..., n_fft // 2 : -(n_fft // 2)]
        return out.squeeze(1)

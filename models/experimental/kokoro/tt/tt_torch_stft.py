# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.istftnet.TorchSTFT`.

``TorchSTFT`` wraps ``torch.stft`` / ``torch.istft`` (hann window, ``center=True``, reflect pad).
TT has no native FFT op, but for a fixed input length / hop / ``n_fft`` the STFT and iSTFT are
just sparse linear maps, so we precompute them on the host (numpy) once at construction time
and reduce inference to two ``ttnn.matmul`` per direction plus a handful of pointwise ops.

Math summary (real input, ``n_fft`` even, ``K = n_fft // 2 + 1``)
-----------------------------------------------------------------
* Reflection pad ``x`` by ``p = n_fft // 2`` on each side (linear -> matrix ``P``).
* For each frame ``f`` (``F = L // hop + 1``) and bin ``k``:

    X_real[k, f] = Σ_n window[n] · cos(2πkn/N) · x_padded[f·hop + n]
    X_imag[k, f] = −Σ_n window[n] · sin(2πkn/N) · x_padded[f·hop + n]

  Fold the pad map into the STFT matrices ``A_real``, ``A_imag`` of shape ``[L_in, K·F]``;
  the runtime is just ``X = x @ A``.

* iSTFT inverts the same path: hermitian-extended iFFT coefficients (``2/N`` mid bins, ``1/N``
  DC + Nyquist, sign-alternating at Nyquist), multiplied by the synthesis window, scattered
  to the right position via OLA, then divided by ``Σ_f window[n]²`` for the COLA normalisation.
  We pre-divide each column by that denominator and trim the pad so the iSTFT collapses to a
  single matmul + one elementwise add per direction:

    y = X_real_flat @ B_real + X_imag_flat @ B_imag   # [B, output_length]

PyTorch and NumPy are used only at module construction; execution path is pure TTNN ops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import ttnn


@dataclass(frozen=True)
class TTTorchSTFTParams:
    """Pre-computed STFT/iSTFT matrices (device-resident) for a fixed input length."""

    # STFT: y_real_flat = x @ stft_real, y_imag_flat = x @ stft_imag (each ``[B, K*F]``).
    stft_real: ttnn.Tensor  # [L_in, K*F]
    stft_imag: ttnn.Tensor  # [L_in, K*F]

    # iSTFT: y_out = X_real_flat @ istft_real + X_imag_flat @ istft_imag (``[B, output_length]``).
    istft_real: ttnn.Tensor  # [K*F, output_length]
    istft_imag: ttnn.Tensor  # [K*F, output_length]

    filter_length: int  # n_fft
    hop_length: int
    win_length: int
    input_length: int  # L_in
    output_length: int  # round-trip output length
    K: int  # n_fft // 2 + 1
    F: int  # number of frames


def _hann_window(win_length: int) -> np.ndarray:
    """Match ``torch.hann_window(win_length, periodic=True)`` in float64."""
    return torch.hann_window(win_length, periodic=True, dtype=torch.float64).numpy()


def _reflection_pad_matrix(L: int, p: int) -> np.ndarray:
    """``[L + 2p, L]`` matrix that applies the same reflection ``torch.stft`` uses (``mode="reflect"``)."""
    L_padded = L + 2 * p
    P = np.zeros((L_padded, L), dtype=np.float64)
    for i in range(p):
        P[i, p - i] = 1.0
    for i in range(L):
        P[p + i, i] = 1.0
    for i in range(p):
        P[p + L + i, L - 2 - i] = 1.0
    return P


def _build_stft_matrices(L: int, n_fft: int, hop: int, window: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(A_real, A_imag, F)`` with ``A_*`` shaped ``[L, K*F]`` (k-major flat)."""
    assert n_fft % 2 == 0, "n_fft must be even for hermitian half-spectrum"
    K = n_fft // 2 + 1
    p = n_fft // 2
    L_padded = L + 2 * p
    F = L // hop + 1

    P = _reflection_pad_matrix(L, p)

    M_real = np.zeros((K * F, L_padded), dtype=np.float64)
    M_imag = np.zeros((K * F, L_padded), dtype=np.float64)
    for f in range(F):
        for n in range(n_fft):
            m = f * hop + n
            if m >= L_padded:
                continue
            for k in range(K):
                phase = 2.0 * np.pi * k * n / n_fft
                M_real[k * F + f, m] = window[n] * np.cos(phase)
                M_imag[k * F + f, m] = -window[n] * np.sin(phase)

    A_real = (M_real @ P).T.astype(np.float32)  # [L, K*F]
    A_imag = (M_imag @ P).T.astype(np.float32)
    return A_real, A_imag, F


def _build_istft_matrices(L: int, n_fft: int, hop: int, window: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(B_real, B_imag, output_length)`` with ``B_*`` shaped ``[K*F, output_length]``.

    The matrices already include synthesis windowing, OLA scatter, COLA normalisation and the
    pad-trim so the runtime path is one matmul per direction.
    """
    assert n_fft % 2 == 0, "n_fft must be even"
    K = n_fft // 2 + 1
    p = n_fft // 2
    F = L // hop + 1
    output_length = (F - 1) * hop
    L_padded = output_length + n_fft  # one frame past the last valid window

    N = n_fft
    iFFT_real = np.zeros((K, N), dtype=np.float64)
    iFFT_imag = np.zeros((K, N), dtype=np.float64)
    for n in range(N):
        iFFT_real[0, n] = 1.0 / N
        iFFT_real[K - 1, n] = (1.0 / N) * ((-1.0) ** n)  # Nyquist (n_fft even => K-1 is Nyquist)
        for k in range(1, K - 1):
            iFFT_real[k, n] = (2.0 / N) * np.cos(2.0 * np.pi * k * n / N)
            iFFT_imag[k, n] = -(2.0 / N) * np.sin(2.0 * np.pi * k * n / N)

    B_real = np.zeros((K * F, L_padded), dtype=np.float64)
    B_imag = np.zeros((K * F, L_padded), dtype=np.float64)
    denom = np.zeros(L_padded, dtype=np.float64)
    for f in range(F):
        for n in range(N):
            m = f * hop + n
            if m >= L_padded:
                continue
            w_n = window[n]
            denom[m] += w_n * w_n
            for k in range(K):
                B_real[k * F + f, m] = w_n * iFFT_real[k, n]
                B_imag[k * F + f, m] = w_n * iFFT_imag[k, n]

    denom_trim = denom[p : p + output_length]
    inv = 1.0 / np.maximum(denom_trim, 1e-11)
    B_real_out = (B_real[:, p : p + output_length] * inv).astype(np.float32)  # [K*F, output_length]
    B_imag_out = (B_imag[:, p : p + output_length] * inv).astype(np.float32)
    return B_real_out, B_imag_out, output_length


def _upload(arr: np.ndarray, device, *, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.from_numpy(arr.astype(np.float32)),
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def preprocess_tt_torch_stft(
    *,
    filter_length: int,
    hop_length: int,
    win_length: int,
    input_length: int,
    device: ttnn.Device,
    weights_dtype=ttnn.bfloat16,
) -> TTTorchSTFTParams:
    """Build STFT/iSTFT matrices on the host and upload them to device."""
    if win_length != filter_length:
        raise ValueError(f"Only win_length == filter_length is supported (got {win_length} vs {filter_length})")
    if input_length % hop_length != 0:
        raise ValueError(
            "input_length must be a multiple of hop_length for clean round-trip "
            f"(got input_length={input_length}, hop_length={hop_length})"
        )

    window = _hann_window(win_length)
    A_real, A_imag, F = _build_stft_matrices(input_length, filter_length, hop_length, window)
    B_real, B_imag, output_length = _build_istft_matrices(input_length, filter_length, hop_length, window)
    K = filter_length // 2 + 1

    return TTTorchSTFTParams(
        stft_real=_upload(A_real, device, dtype=weights_dtype),
        stft_imag=_upload(A_imag, device, dtype=weights_dtype),
        istft_real=_upload(B_real, device, dtype=weights_dtype),
        istft_imag=_upload(B_imag, device, dtype=weights_dtype),
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        input_length=input_length,
        output_length=output_length,
        K=K,
        F=F,
    )


class TTTorchSTFT:
    """TT port of :class:`TorchSTFT` (STFT/iSTFT via precomputed dense matrices)."""

    def __init__(self, device: ttnn.Device, params: TTTorchSTFTParams) -> None:
        self.device = device
        self.params = params
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

    def _matmul_to_bkf(self, x_bL: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """``x_bL @ weight`` followed by a reshape to ``[B, K, F]``."""
        p = self.params
        y = ttnn.matmul(
            x_bL,
            weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        B = int(x_bL.shape[0])
        return ttnn.reshape(y, [B, p.K, p.F], memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def transform(self, x_bL: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            x_bL: ``[B, L]`` (TILE layout) where ``L == params.input_length``.

        Returns:
            ``(magnitude, phase)`` each ``[B, K, F]`` (TILE layout).
        """
        L_in = int(x_bL.shape[-1])
        if L_in != self.params.input_length:
            raise ValueError(f"input length mismatch: got {L_in}, expected {self.params.input_length}")

        X_real = self._matmul_to_bkf(x_bL, self.params.stft_real)
        X_imag = self._matmul_to_bkf(x_bL, self.params.stft_imag)

        mag_sq = ttnn.add(
            ttnn.multiply(X_real, X_real, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.multiply(X_imag, X_imag, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        magnitude = ttnn.sqrt(mag_sq, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mag_sq)
        phase = ttnn.atan2(X_imag, X_real, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        ttnn.deallocate(X_real)
        ttnn.deallocate(X_imag)
        return magnitude, phase

    def inverse(self, magnitude: ttnn.Tensor, phase: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            magnitude: ``[B, K, F]``.
            phase: ``[B, K, F]``.

        Returns:
            ``[B, 1, output_length]`` (matches ``TorchSTFT.inverse``'s trailing ``unsqueeze(-2)``).
        """
        p = self.params

        cos_phase = ttnn.cos(phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin_phase = ttnn.sin(phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        X_real = ttnn.multiply(magnitude, cos_phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        X_imag = ttnn.multiply(magnitude, sin_phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cos_phase)
        ttnn.deallocate(sin_phase)

        B = int(magnitude.shape[0])
        X_real_flat = ttnn.reshape(X_real, [B, p.K * p.F], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        X_imag_flat = ttnn.reshape(X_imag, [B, p.K * p.F], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(X_real)
        ttnn.deallocate(X_imag)

        y_real = ttnn.matmul(
            X_real_flat,
            p.istft_real,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        y_imag = ttnn.matmul(
            X_imag_flat,
            p.istft_imag,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
        )
        ttnn.deallocate(X_real_flat)
        ttnn.deallocate(X_imag_flat)

        y = ttnn.add(y_real, y_imag, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(y_real)
        ttnn.deallocate(y_imag)

        # Reshape to [B, 1, output_length] to match ``TorchSTFT.inverse``.
        while len(y.shape) > 2:
            y = ttnn.squeeze(y, 0)
        out = ttnn.reshape(y, [B, 1, p.output_length], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out

    def forward(self, x_bL: ttnn.Tensor) -> ttnn.Tensor:
        """STFT → iSTFT round trip (matches ``TorchSTFT.forward``)."""
        mag, phase = self.transform(x_bL)
        y = self.inverse(mag, phase)
        ttnn.deallocate(mag)
        ttnn.deallocate(phase)
        return y

    __call__ = forward

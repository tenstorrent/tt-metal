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
    # Conv2d STFT projection kernels (old TT path, used for transform only).
    conv_stft_real: ttnn.Tensor  # [K, 1, n_fft, 1], ROW_MAJOR
    conv_stft_imag: ttnn.Tensor  # [K, 1, n_fft, 1], ROW_MAJOR
    conv_pad_len: int


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


def _upload_rm(arr: np.ndarray, *, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(
        torch.from_numpy(arr.astype(np.float32)),
        dtype=dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def _build_conv_stft_kernels(n_fft: int, win_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Build conv2d-compatible forward STFT kernels ``[K, 1, n_fft, 1]``."""
    window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
    if win_length < n_fft:
        window = torch.nn.functional.pad(window, (0, n_fft - win_length))
    elif win_length > n_fft:
        window = window[:n_fft]
    w = window.numpy()

    k = np.arange(n_fft // 2 + 1)
    n = np.arange(n_fft)
    angle = 2.0 * np.pi * np.outer(k, n) / n_fft
    real = (np.cos(angle) * w)[:, None, :, None].astype(np.float32)
    imag = ((-np.sin(angle)) * w)[:, None, :, None].astype(np.float32)
    return real, imag


def _time_slice_n1tc(x: ttnn.Tensor, t0: int, t1: int, *, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    b = int(x.shape[0])
    c = int(x.shape[3])
    return ttnn.slice(x, [0, 0, t0, 0], [b, 1, t1, c], [1, 1, 1, 1], memory_config=memory_config)


def _reflect_pad_1d_dim2(x_n1tc: ttnn.Tensor, time_len: int, pad: int) -> ttnn.Tensor:
    """Reflect pad along dim=2 for shape ``[B, 1, T, 1]``."""
    if pad <= 0:
        return x_n1tc
    mc = ttnn.DRAM_MEMORY_CONFIG
    left_parts = []
    for i in range(pad):
        # left reflect: x[pad-i]
        left_parts.append(_time_slice_n1tc(x_n1tc, pad - i, pad - i + 1, memory_config=mc))
    right_parts = []
    for i in range(pad):
        # right reflect: x[T-2-i]
        idx = time_len - 2 - i
        right_parts.append(_time_slice_n1tc(x_n1tc, idx, idx + 1, memory_config=mc))
    out = ttnn.concat([*left_parts, x_n1tc, *right_parts], dim=2, memory_config=mc)
    for t in left_parts:
        ttnn.deallocate(t)
    for t in right_parts:
        ttnn.deallocate(t)
    return out


class _StridedStftConv:
    """Strided conv2d projection for STFT real/imag branches."""

    def __init__(self, device: ttnn.Device, weight_rm: ttnn.Tensor, hop_length: int):
        self.device = device
        self.weight_rm = weight_rm
        self.weight_prepared = weight_rm
        self._prep_key: tuple[int, int] | None = None
        self.hop_length = int(hop_length)
        self.out_channels = int(weight_rm.shape[0])
        self.in_channels = int(weight_rm.shape[1])
        self.kernel_size = int(weight_rm.shape[2])
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            config_tensors_in_dram=True,
            reshard_if_not_optimal=False,
            enable_kernel_stride_folding=False,
            force_split_reader=False,
            transpose_shards=False,
            enable_activation_reuse=False,
            full_inner_dim=False,
        )
        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.slice_cfg = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=8)

    def __call__(self, x_n1tc: ttnn.Tensor, batch_size: int, input_height: int) -> ttnn.Tensor:
        x_rm = ttnn.to_layout(x_n1tc, ttnn.ROW_MAJOR_LAYOUT)
        key = (batch_size, input_height)
        if self._prep_key != key:
            self.weight_prepared = ttnn.prepare_conv_weights(
                weight_tensor=self.weight_rm,
                input_memory_config=x_rm.memory_config(),
                input_layout=x_rm.layout,
                weights_format="OIHW",
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                input_height=input_height,
                input_width=1,
                kernel_size=(self.kernel_size, 1),
                stride=(self.hop_length, 1),
                padding=(0, 0),
                dilation=(1, 1),
                has_bias=False,
                groups=1,
                device=self.device,
                input_dtype=x_rm.dtype,
                conv_config=self.conv_config,
                compute_config=self.compute_cfg,
                slice_config=self.slice_cfg,
            )
            self._prep_key = key
        result, [oh, _ow], wpair = ttnn.conv2d(
            input_tensor=x_rm,
            weight_tensor=self.weight_prepared,
            device=self.device,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=1,
            kernel_size=(self.kernel_size, 1),
            stride=(self.hop_length, 1),
            padding=(0, 0),
            bias_tensor=None,
            conv_config=self.conv_config,
            compute_config=self.compute_cfg,
            slice_config=self.slice_cfg,
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        self.weight_prepared = wpair[0]
        out = ttnn.reshape(result, [batch_size, int(oh), self.out_channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.permute(out, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out


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
    conv_real, conv_imag = _build_conv_stft_kernels(filter_length, win_length)
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
        conv_stft_real=_upload_rm(conv_real, dtype=ttnn.float32),
        conv_stft_imag=_upload_rm(conv_imag, dtype=ttnn.float32),
        conv_pad_len=filter_length // 2,
    )


class TTTorchSTFT:
    """TT port of :class:`TorchSTFT` (STFT/iSTFT via precomputed dense matrices)."""

    def __init__(self, device: ttnn.Device, params: TTTorchSTFTParams) -> None:
        self.device = device
        self.params = params
        self.eps = 1e-11
        self._conv_real = _StridedStftConv(device, params.conv_stft_real, params.hop_length)
        self._conv_imag = _StridedStftConv(device, params.conv_stft_imag, params.hop_length)
        # HiFi4 + fp32_dest_acc_en empirically beats HiFi3 on WH for STFT-of-small-signals:
        # cos(phase) PCC 0.78 vs 0.77 and near-zero sign match 0.701 vs 0.686 (the WH runtime
        # warns about HiFi4+fp32 but the warning concerns a different op pattern). Mag PCC is
        # 1.0 in both modes.
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _matmul_to_bkf(self, x_bL: ttnn.Tensor, weight: ttnn.Tensor) -> ttnn.Tensor:
        """``x_bL @ weight`` followed by a reshape to ``[B, K, F]``."""
        p = self.params
        y = ttnn.matmul(
            x_bL,
            weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config,
            dtype=x_bL.dtype,
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
        if x_bL.dtype != ttnn.float32:
            x_bL = ttnn.typecast(x_bL, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        X_real = self._matmul_to_bkf(x_bL, self.params.stft_real)
        X_imag = self._matmul_to_bkf(x_bL, self.params.stft_imag)

        # Match tt_old stabilization: sqrt(real^2 + imag^2 + eps).
        mag_sq = ttnn.add(
            ttnn.multiply(X_real, X_real, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.multiply(X_imag, X_imag, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        eps_t = ttnn.full_like(X_real, self.eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mag_sq = ttnn.add(mag_sq, eps_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(eps_t)
        magnitude = ttnn.sqrt(mag_sq, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(mag_sq)
        phase = ttnn.atan2(X_imag, X_real, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        # Match old STFT edge-case behavior for branch cut at negative real axis.
        corr_mask = ttnn.logical_and(
            ttnn.eq(X_imag, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            ttnn.lt(X_real, 0.0, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        pi_fill = ttnn.full_like(phase, np.pi, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        phase = ttnn.where(corr_mask, pi_fill, phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(corr_mask)
        ttnn.deallocate(pi_fill)

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
        if magnitude.dtype != ttnn.float32:
            magnitude = ttnn.typecast(magnitude, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if phase.dtype != ttnn.float32:
            phase = ttnn.typecast(phase, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

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

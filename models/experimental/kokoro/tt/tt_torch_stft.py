# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.istftnet.TorchSTFT`.

``TorchSTFT`` wraps ``torch.stft`` / ``torch.istft`` (hann window, ``center=True``, reflect pad).
TT has no native FFT op.  For a fixed input length / hop / ``n_fft`` the iSTFT is a dense linear
map, so we pre-compute it once at construction and reduce iSTFT to two matmuls + one add.

STFT forward path (conv2d)
--------------------------
The STFT uses strided conv2d: the windowed DFT kernels are stored as ``[K, 1, n_fft, 1]`` weight
tensors and applied with stride ``= hop_length``.  On Blackhole hardware, all compute ops
internally round float32 inputs to BF16 before the MAC unit regardless of HiFi setting.  This
limits the phase accuracy of near-zero STFT bins — bins whose true value is smaller than the
BF16 noise floor (~signal × 1e-2) will have sign-random phase.  Trained Kokoro is designed to
tolerate this: the noise_conv was learned to absorb harmonic-source phase noise, and empirical
testing shows the conv2d BF16 error pattern achieves cos(phase) PCC ≈ 0.64 and full-forward
PCC ≈ 0.58.  This is the practical ceiling on current BH hardware without CPU fallback.

iSTFT path
----------
Pre-computed dense matrices (device-resident, small inputs only):

    y = X_real_flat @ B_real + X_imag_flat @ B_imag   # [B, output_length]

B_real / B_imag already encode synthesis windowing, OLA and COLA normalisation, so the
runtime is two matmuls + one add.

When the iSTFT matrix would exceed ``_ISTFT_MATRIX_BYTES_LIMIT`` bytes (default 1 GiB) we avoid
materializing the full matrix and instead run iSTFT as a sequence of on-device chunked matmuls.
This keeps large-sequence synthesis fully on TT without torch runtime fallbacks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

import ttnn

# iSTFT matrix bytes limit: if [K*F, output_length] float32 would exceed this, skip
# precomputing the matrices and use torch.istft on CPU instead.
_ISTFT_MATRIX_BYTES_LIMIT = 1_073_741_824  # 1 GiB


@dataclass(frozen=True)
class TTTorchSTFTParams:
    """Pre-computed STFT/iSTFT parameters (device-resident) for a fixed input length."""

    # STFT forward: strided conv2d kernels (k-th output channel = k-th DFT bin).
    conv_stft_real: ttnn.Tensor  # [K, 1, n_fft, 1], ROW_MAJOR
    conv_stft_imag: ttnn.Tensor  # [K, 1, n_fft, 1], ROW_MAJOR
    conv_pad_len: int  # n_fft // 2

    # iSTFT: y = X_real_flat @ istft_real + X_imag_flat @ istft_imag
    # None when the full matrix exceeds _ISTFT_MATRIX_BYTES_LIMIT; inverse uses chunked TT matmuls.
    istft_real: Optional[ttnn.Tensor]  # [K*F, output_length]
    istft_imag: Optional[ttnn.Tensor]  # [K*F, output_length]
    istft_chunk_size: int

    filter_length: int  # n_fft
    hop_length: int
    win_length: int
    input_length: int  # L_in
    output_length: int
    K: int  # n_fft // 2 + 1
    F: int  # number of frames


def _hann_window(win_length: int) -> np.ndarray:
    """Match ``torch.hann_window(win_length, periodic=True)`` in float64."""
    return torch.hann_window(win_length, periodic=True, dtype=torch.float64).numpy()


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
    L_padded = output_length + n_fft

    N = n_fft
    iFFT_real = np.zeros((K, N), dtype=np.float64)
    iFFT_imag = np.zeros((K, N), dtype=np.float64)
    for n in range(N):
        iFFT_real[0, n] = 1.0 / N
        iFFT_real[K - 1, n] = (1.0 / N) * ((-1.0) ** n)
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
    B_real_out = (B_real[:, p : p + output_length] * inv).astype(np.float32)
    B_imag_out = (B_imag[:, p : p + output_length] * inv).astype(np.float32)
    return B_real_out, B_imag_out, output_length


def _build_istft_inverse_denom_trim(L: int, n_fft: int, hop: int, window: np.ndarray) -> np.ndarray:
    """Return COLA normalization inverse for trimmed output positions ``[output_length]``."""
    output_length = (L // hop) * hop
    p = n_fft // 2
    L_padded = output_length + n_fft
    denom = np.zeros(L_padded, dtype=np.float64)
    for f in range(L // hop + 1):
        base = f * hop
        for n in range(n_fft):
            m = base + n
            if m < L_padded:
                w_n = window[n]
                denom[m] += w_n * w_n
    denom_trim = denom[p : p + output_length]
    return (1.0 / np.maximum(denom_trim, 1e-11)).astype(np.float32)


def _build_istft_matrix_chunk(
    L: int,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    inv_denom_trim: np.ndarray,
    col_start: int,
    col_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one iSTFT chunk ``[K*F, col_end-col_start]`` for trimmed output columns."""
    K = n_fft // 2 + 1
    F = L // hop + 1
    output_length = (F - 1) * hop
    p = n_fft // 2
    N = n_fft
    chunk_w = col_end - col_start
    padded_start = p + col_start
    padded_end = p + col_end

    iFFT_real = np.zeros((K, N), dtype=np.float64)
    iFFT_imag = np.zeros((K, N), dtype=np.float64)
    for n in range(N):
        iFFT_real[0, n] = 1.0 / N
        iFFT_real[K - 1, n] = (1.0 / N) * ((-1.0) ** n)
        for k in range(1, K - 1):
            angle = 2.0 * np.pi * k * n / N
            iFFT_real[k, n] = (2.0 / N) * np.cos(angle)
            iFFT_imag[k, n] = -(2.0 / N) * np.sin(angle)

    B_real = np.zeros((K * F, chunk_w), dtype=np.float32)
    B_imag = np.zeros((K * F, chunk_w), dtype=np.float32)
    for f in range(F):
        frame_base = f * hop
        for n in range(N):
            m = frame_base + n
            if m < padded_start or m >= padded_end:
                continue
            out_col = m - padded_start
            w_n = float(window[n])
            row_base = f
            for k in range(K):
                row = k * F + row_base
                B_real[row, out_col] = np.float32(w_n * iFFT_real[k, n])
                B_imag[row, out_col] = np.float32(w_n * iFFT_imag[k, n])

    norm = inv_denom_trim[col_start:col_end][None, :]
    B_real *= norm
    B_imag *= norm
    return B_real, B_imag


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
        left_parts.append(_time_slice_n1tc(x_n1tc, pad - i, pad - i + 1, memory_config=mc))
    right_parts = []
    for i in range(pad):
        idx = time_len - 2 - i
        right_parts.append(_time_slice_n1tc(x_n1tc, idx, idx + 1, memory_config=mc))
    out = ttnn.concat([*left_parts, x_n1tc, *right_parts], dim=2, memory_config=mc)
    for t in left_parts:
        ttnn.deallocate(t)
    for t in right_parts:
        ttnn.deallocate(t)
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
    """Build STFT/iSTFT parameters on the host and upload them to device.

    The iSTFT full matrix is skipped when it would exceed ``_ISTFT_MATRIX_BYTES_LIMIT``.
    In that case ``TTTorchSTFT.inverse`` uses chunked on-device matmuls.
    """
    if win_length != filter_length:
        raise ValueError(f"Only win_length == filter_length is supported (got {win_length} vs {filter_length})")
    if input_length % hop_length != 0:
        raise ValueError(
            "input_length must be a multiple of hop_length for clean round-trip "
            f"(got input_length={input_length}, hop_length={hop_length})"
        )

    K = filter_length // 2 + 1
    F = input_length // hop_length + 1
    output_length = (F - 1) * hop_length  # == input_length when input_length % hop_length == 0

    matrix_bytes = K * F * output_length * 4  # float32 bytes for one matrix
    skip_istft_precompute = matrix_bytes > _ISTFT_MATRIX_BYTES_LIMIT

    istft_chunk_size = 1024
    if skip_istft_precompute:
        istft_real_t: Optional[ttnn.Tensor] = None
        istft_imag_t: Optional[ttnn.Tensor] = None
    else:
        window = _hann_window(win_length)
        B_real, B_imag, output_length = _build_istft_matrices(input_length, filter_length, hop_length, window)
        istft_real_t = _upload(B_real, device, dtype=weights_dtype)
        istft_imag_t = _upload(B_imag, device, dtype=weights_dtype)

    conv_real, conv_imag = _build_conv_stft_kernels(filter_length, win_length)

    return TTTorchSTFTParams(
        conv_stft_real=_upload_rm(conv_real, dtype=ttnn.float32),
        conv_stft_imag=_upload_rm(conv_imag, dtype=ttnn.float32),
        conv_pad_len=filter_length // 2,
        istft_real=istft_real_t,
        istft_imag=istft_imag_t,
        istft_chunk_size=istft_chunk_size,
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        input_length=input_length,
        output_length=output_length,
        K=K,
        F=F,
    )


class _StridedStftConv:
    """Strided conv2d projection for one STFT branch (real or imag).

    When ``use_torch_fallback=True``, the convolution runs on CPU via
    ``torch.nn.functional.conv2d`` (float32 MACs) and the result is uploaded back to device.
    This bypasses the BH BF16 MAC ceiling while keeping everything else in TTNN.
    """

    def __init__(
        self, device: ttnn.Device, weight_rm: ttnn.Tensor, hop_length: int, *, use_torch_fallback: bool = False
    ):
        self.device = device
        self.weight_rm = weight_rm
        self.weight_prepared = weight_rm
        self._prep_key: tuple[int, int] | None = None
        self.hop_length = int(hop_length)
        self.out_channels = int(weight_rm.shape[0])
        self.in_channels = int(weight_rm.shape[1])
        self.kernel_size = int(weight_rm.shape[2])
        self.use_torch_fallback = use_torch_fallback
        self.conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.float32,  # float32 storage improves DFT precision vs bfloat16 (0.58 vs 0.34 PCC)
            output_layout=ttnn.TILE_LAYOUT,
            deallocate_activation=False,
            reallocate_halo_output=False,
            enable_act_double_buffer=False,
            enable_weights_double_buffer=False,
            config_tensors_in_dram=True,
            reshard_if_not_optimal=False,
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

    def _torch_forward(self, x_n1tc: ttnn.Tensor, batch_size: int, input_height: int) -> ttnn.Tensor:
        """CPU fallback: float32 ``F.conv2d`` → upload result to device."""
        import torch.nn.functional as F_torch

        x_cpu = ttnn.to_torch(x_n1tc).float().reshape(batch_size, 1, input_height, 1)
        w_cpu = ttnn.to_torch(self.weight_rm).float()  # [K, 1, n_fft, 1]
        with torch.no_grad():
            out = F_torch.conv2d(x_cpu, w_cpu, stride=(self.hop_length, 1), padding=0)
        out = out.squeeze(-1).contiguous()  # [B, K, F]
        return ttnn.from_torch(
            out, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

    def __call__(self, x_n1tc: ttnn.Tensor, batch_size: int, input_height: int) -> ttnn.Tensor:
        if self.use_torch_fallback:
            return self._torch_forward(x_n1tc, batch_size, input_height)
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


class TTTorchSTFT:
    """TT port of :class:`TorchSTFT` (STFT/iSTFT via precomputed dense matrices).

    ``use_torch_stft_fallback=True`` applies to :meth:`transform` only.
    iSTFT always runs on TT (full matrix or chunked matmuls).
    """

    def __init__(
        self,
        device: ttnn.Device,
        params: TTTorchSTFTParams,
        *,
        use_torch_stft_fallback: bool = False,
    ) -> None:
        self.device = device
        self.params = params
        self.eps = 1e-11
        # Phase is undefined for near-zero bins. In no-fallback TT STFT those bins are most
        # sensitive to BF16 rounding; clamp their phase to 0 to reduce random phase jitter.
        self.phase_zero_floor = float(os.getenv("KOKORO_STFT_PHASE_ZERO_FLOOR", "1e-8"))
        self._use_torch_stft_fallback = use_torch_stft_fallback
        # _StridedStftConv is used only when use_torch_stft_fallback=False.
        self._conv_real = _StridedStftConv(device, params.conv_stft_real, params.hop_length)
        self._conv_imag = _StridedStftConv(device, params.conv_stft_imag, params.hop_length)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._istft_window = _hann_window(params.win_length)
        self._istft_inv_denom_trim = _build_istft_inverse_denom_trim(
            params.input_length, params.filter_length, params.hop_length, self._istft_window
        )

    def _transform_torch_fallback(self, x_bL: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """CPU float32 STFT transform using ``torch.stft`` (exact match to reference TorchSTFT).

        Why this is needed: BH hardware rounds float32 inputs to BF16 before ALL compute ops —
        including the SFPU that evaluates ``atan2``.  Near-zero DFT bins (true value ~1e-5) are
        rounded to zero or sign-flipped after BF16 rounding, so ``atan2(X_imag, X_real)`` gives
        sign-random phase even when X_real/X_imag were computed precisely on CPU.  Moving the
        entire transform (conv2d + atan2 + sqrt) to CPU float32 eliminates this.

        Matches reference TorchSTFT.transform exactly: same window, same center=True default,
        same n_fft/hop/win_length.
        """
        p = self.params
        B = int(x_bL.shape[0])
        x_cpu = ttnn.to_torch(x_bL).float().reshape(B, p.input_length)
        window = torch.hann_window(p.win_length, periodic=True, dtype=torch.float32)
        with torch.no_grad():
            z = torch.stft(
                x_cpu,
                p.filter_length,
                p.hop_length,
                p.win_length,
                window=window,
                return_complex=True,
            )  # [B, K, F] complex float32
        magnitude = torch.abs(z).contiguous()
        phase = torch.angle(z).contiguous()
        mc = ttnn.DRAM_MEMORY_CONFIG
        mag_tt = ttnn.from_torch(
            magnitude, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=mc
        )
        phase_tt = ttnn.from_torch(
            phase, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=self.device, memory_config=mc
        )
        return mag_tt, phase_tt

    def transform(self, x_bL: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            x_bL: ``[B, L]`` (TILE layout) where ``L == params.input_length``.

        Returns:
            ``(magnitude, phase)`` each ``[B, K, F]`` (TILE layout).

        When ``use_torch_stft_fallback=True`` the entire transform runs on CPU via ``torch.stft``
        (see :meth:`_transform_torch_fallback`).  This is necessary because BH rounds float32
        inputs to BF16 before ALL ops — including the SFPU that evaluates ``atan2`` — so moving
        only the conv2d to CPU still leaves ``atan2`` degraded.  The TTNN path (fallback=False)
        applies BH BF16 throughout and achieves ~0.58 PCC on full-forward.
        """
        if self._use_torch_stft_fallback:
            return self._transform_torch_fallback(x_bL)

        L_in = int(x_bL.shape[-1])
        if L_in != self.params.input_length:
            raise ValueError(f"input length mismatch: got {L_in}, expected {self.params.input_length}")
        if x_bL.dtype != ttnn.float32:
            x_bL = ttnn.typecast(x_bL, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        p = self.params
        B = int(x_bL.shape[0])

        x_bL_rm = ttnn.to_layout(x_bL, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_n1lc = ttnn.reshape(x_bL_rm, [B, 1, L_in, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(x_bL_rm)

        x_padded = _reflect_pad_1d_dim2(x_n1lc, L_in, p.conv_pad_len)
        ttnn.deallocate(x_n1lc)
        L_padded = L_in + 2 * p.conv_pad_len

        X_real = self._conv_real(x_padded, B, L_padded)  # [B, K, F]
        X_imag = self._conv_imag(x_padded, B, L_padded)  # [B, K, F]
        ttnn.deallocate(x_padded)

        mc = ttnn.DRAM_MEMORY_CONFIG
        mag_sq = ttnn.add(
            ttnn.multiply(X_real, X_real, memory_config=mc),
            ttnn.multiply(X_imag, X_imag, memory_config=mc),
            memory_config=mc,
        )
        eps_t = ttnn.full_like(X_real, self.eps, memory_config=mc)
        mag_sq = ttnn.add(mag_sq, eps_t, memory_config=mc)
        ttnn.deallocate(eps_t)
        magnitude = ttnn.sqrt(mag_sq, memory_config=mc)
        phase = ttnn.atan2(X_imag, X_real, memory_config=mc)
        corr_mask = ttnn.logical_and(
            ttnn.eq(X_imag, 0.0, memory_config=mc),
            ttnn.lt(X_real, 0.0, memory_config=mc),
            memory_config=mc,
        )
        pi_fill = ttnn.full_like(phase, np.pi, memory_config=mc)
        phase = ttnn.where(corr_mask, pi_fill, phase, memory_config=mc)
        ttnn.deallocate(corr_mask)
        ttnn.deallocate(pi_fill)
        near_zero_mask = ttnn.lt(mag_sq, self.phase_zero_floor, memory_config=mc)
        zero_phase = ttnn.full_like(phase, 0.0, memory_config=mc)
        phase = ttnn.where(near_zero_mask, zero_phase, phase, memory_config=mc)
        ttnn.deallocate(near_zero_mask)
        ttnn.deallocate(zero_phase)
        ttnn.deallocate(mag_sq)
        ttnn.deallocate(X_real)
        ttnn.deallocate(X_imag)
        return magnitude, phase

    def _inverse_chunked(self, X_real_flat: ttnn.Tensor, X_imag_flat: ttnn.Tensor) -> ttnn.Tensor:
        """Chunked iSTFT matmul path for long sequences (TT-only, no torch runtime fallback)."""
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG
        parts: list[ttnn.Tensor] = []
        for col_start in range(0, p.output_length, p.istft_chunk_size):
            col_end = min(col_start + p.istft_chunk_size, p.output_length)
            B_real_np, B_imag_np = _build_istft_matrix_chunk(
                p.input_length,
                p.filter_length,
                p.hop_length,
                self._istft_window,
                self._istft_inv_denom_trim,
                col_start,
                col_end,
            )
            B_real_tt = _upload(B_real_np, self.device, dtype=ttnn.float32)
            B_imag_tt = _upload(B_imag_np, self.device, dtype=ttnn.float32)
            y_real = ttnn.matmul(
                X_real_flat, B_real_tt, memory_config=mc, compute_kernel_config=self.compute_kernel_config
            )
            y_imag = ttnn.matmul(
                X_imag_flat, B_imag_tt, memory_config=mc, compute_kernel_config=self.compute_kernel_config
            )
            y_chunk = ttnn.add(y_real, y_imag, memory_config=mc)
            ttnn.deallocate(y_real)
            ttnn.deallocate(y_imag)
            ttnn.deallocate(B_real_tt)
            ttnn.deallocate(B_imag_tt)
            parts.append(y_chunk)
        if len(parts) == 1:
            return parts[0]
        out = parts[0]
        for i in range(1, len(parts)):
            cat = ttnn.concat([out, parts[i]], dim=1, memory_config=mc)
            ttnn.deallocate(out)
            ttnn.deallocate(parts[i])
            out = cat
        return out

    def inverse(self, magnitude: ttnn.Tensor, phase: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            magnitude: ``[B, K, F]``.
            phase: ``[B, K, F]``.

        Returns:
            ``[B, 1, output_length]`` (matches ``TorchSTFT.inverse``'s trailing ``unsqueeze(-2)``).

        When iSTFT full matrices are not precomputed (input too large for
        ``_ISTFT_MATRIX_BYTES_LIMIT``), this method uses chunked TT matmuls.
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

        if p.istft_real is not None and p.istft_imag is not None:
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
            y = ttnn.add(y_real, y_imag, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(y_real)
            ttnn.deallocate(y_imag)
        else:
            y = self._inverse_chunked(X_real_flat, X_imag_flat)
        ttnn.deallocate(X_real_flat)
        ttnn.deallocate(X_imag_flat)

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

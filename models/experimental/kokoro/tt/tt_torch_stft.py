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

The dense iSTFT (:meth:`TTTorchSTFT.inverse`) runs entirely in bf16: the incoming
``magnitude``/``phase`` are NOT typecast to float32 first.  The polar->rectangular conversion
(cos/sin/multiply) and the well-conditioned dense matmul against the bf16 ``B_real``/``B_imag``
tolerate bf16 (inverse-vs-reference PCC 0.999995, full kModel fallback PCC unchanged at ~0.874),
unlike the forward atan2 path which amplifies near-zero-bin noise.  Dropping the two BF16=>FP32
typecasts removes two dispatches on this op-gap-bound path (9 device ops vs 11, device time
~23µs vs ~28µs).  A single fused matmul against the row-stacked ``[B_real; B_imag]`` was tried
and rejected: concatenating along the K=11 (non-tile-aligned) axis forces an
untilize->concat->tilize round-trip that costs more than the matmul + add it replaces.

When the iSTFT matrix would exceed ``_ISTFT_MATRIX_BYTES_LIMIT`` bytes (default 1 GiB) the
full matrix is not materialised.  Instead iSTFT uses ``conv_transpose2d``:

    y_ola = conv_transpose2d(X_real, synth_real, stride=hop)
          + conv_transpose2d(X_imag, synth_imag, stride=hop)   # [B, L_padded, 1]
    y = trim(y_ola) * inv_denom                                 # COLA normalisation

``conv_transpose2d`` is the adjoint of the forward STFT strided conv2d, so it computes the
windowed overlap-add (OLA) exactly in a single on-device pass.  There is no CPU involvement
in the forward path.

Long sequences (single-pass, device-only)
------------------------------------------
Both directions run in a single device pass at any length — no chunking, no CPU fallback.  The
strided conv2d (forward) and conv_transpose2d OLA (iSTFT) stream the signal through bounded L1
circular buffers via ``dram_height_slice_config`` (DRAM height slicing), which fits the full
Kokoro range (forward input_height ~635k, iSTFT F ~127k) at the default 256-slice budget.  The
only other op that would overflow L1 at large length is the 2D ROW_MAJOR reshape
(``[B, L] -> [B, 1, L, 1]`` and the ``[B, W, 1] -> [B, W]`` COLA squeeze), whose circular buffer
scales with the full width; these are routed through TILE layout (tile-local, bounded) instead.

The ``torch.stft`` / ``torch.istft`` paths now run only when explicitly requested via the
``use_torch_*`` flags.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from loguru import logger

import ttnn

from models.experimental.kokoro.tt.tt_conv import dram_height_slice_config, dram_height_slice_num_slices

# iSTFT matrix bytes limit: if [K*F, output_length] float32 would exceed this, skip
# precomputing the matrices and use conv_transpose2d OLA instead.
_ISTFT_MATRIX_BYTES_LIMIT = 1_073_741_824  # 1 GiB


def _to_fp32_if_needed(x: ttnn.Tensor, memory_config: ttnn.MemoryConfig) -> tuple[ttnn.Tensor, bool]:
    if x.dtype == ttnn.float32:
        return x, False
    return ttnn.typecast(x, ttnn.float32, memory_config=memory_config), True


@dataclass(frozen=True)
class TTTorchSTFTParams:
    """Pre-computed STFT/iSTFT parameters (device-resident) for a fixed input length."""

    # STFT forward: strided conv2d kernels (k-th output channel = k-th DFT bin).
    conv_stft_real: ttnn.Tensor  # [K, 1, n_fft, 1], ROW_MAJOR
    conv_stft_imag: ttnn.Tensor  # [K, 1, n_fft, 1], ROW_MAJOR
    conv_pad_len: int  # n_fft // 2

    # iSTFT full matrix (None when it would exceed _ISTFT_MATRIX_BYTES_LIMIT).
    # y = X_real_flat @ istft_real + X_imag_flat @ istft_imag
    istft_real: Optional[ttnn.Tensor]  # [K*F, output_length]
    istft_imag: Optional[ttnn.Tensor]  # [K*F, output_length]

    # iSTFT conv-transpose OLA (used when full matrix is not precomputed).
    # Synthesis kernels w[n]*iFFT_real/imag[k,n] — adjoint of the STFT forward conv.
    synth_real: ttnn.Tensor  # [K, 1, n_fft, 1], ROW_MAJOR, host
    synth_imag: ttnn.Tensor  # [K, 1, n_fft, 1], ROW_MAJOR, host
    synth_combined: ttnn.Tensor  # [2K, 1, n_fft, 1], ROW_MAJOR, host — fused real+imag conv weight
    inv_denom_tt: ttnn.Tensor  # [1, output_length], TILE, device — COLA normalisation

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
    """Return COLA normalisation inverse for trimmed output positions ``[output_length]``."""
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


def _build_synth_kernels(n_fft: int, win_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Build per-frame synthesis kernels ``[K, 1, n_fft, 1]`` (IOHW) for conv_transpose2d OLA.

    synth_real/imag[k, 0, n, 0] = window[n] * iFFT_real/imag[k, n].
    conv_transpose2d with these kernels and stride=hop computes the windowed OLA sum exactly:
    the operation is the adjoint of the forward STFT strided conv2d.
    """
    window = _hann_window(win_length)
    K = n_fft // 2 + 1
    k = np.arange(K)
    n = np.arange(n_fft)
    angle = 2.0 * np.pi * np.outer(k, n) / n_fft

    iFFT_real = np.zeros((K, n_fft), dtype=np.float64)
    iFFT_imag = np.zeros((K, n_fft), dtype=np.float64)
    iFFT_real[0, :] = 1.0 / n_fft
    iFFT_real[K - 1, :] = ((-1.0) ** n) / n_fft
    iFFT_real[1 : K - 1, :] = (2.0 / n_fft) * np.cos(angle[1 : K - 1, :])
    iFFT_imag[1 : K - 1, :] = -(2.0 / n_fft) * np.sin(angle[1 : K - 1, :])

    synth_real = (window * iFFT_real).astype(np.float32)[:, None, :, None]  # [K, 1, n_fft, 1]
    synth_imag = (window * iFFT_imag).astype(np.float32)[:, None, :, None]
    return synth_real, synth_imag


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
    In that case ``TTTorchSTFT`` uses conv_transpose2d OLA iSTFT.
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

    window = _hann_window(win_length)

    matrix_bytes = K * F * output_length * 4  # float32 bytes for one matrix
    if matrix_bytes > _ISTFT_MATRIX_BYTES_LIMIT:
        istft_real_t: Optional[ttnn.Tensor] = None
        istft_imag_t: Optional[ttnn.Tensor] = None
    else:
        B_real, B_imag, output_length = _build_istft_matrices(input_length, filter_length, hop_length, window)
        istft_real_t = _upload(B_real, device, dtype=weights_dtype)
        istft_imag_t = _upload(B_imag, device, dtype=weights_dtype)

    # Synthesis kernels for conv_transpose2d OLA iSTFT (always precomputed; tiny).
    synth_real_np, synth_imag_np = _build_synth_kernels(filter_length, win_length)
    synth_real_t = ttnn.from_torch(torch.from_numpy(synth_real_np), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    synth_imag_t = ttnn.from_torch(torch.from_numpy(synth_imag_np), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT)
    # Channel-stacked weight [2K, 1, n_fft, 1] for the fused real+imag conv_transpose: because
    # conv_transpose2d sums over input channels, conv([X_real|X_imag], [W_real|W_imag]) equals
    # conv(X_real, W_real) + conv(X_imag, W_imag) — one conv instead of two, and no trailing add.
    synth_combined_np = np.concatenate([synth_real_np, synth_imag_np], axis=0)  # [2K, 1, n_fft, 1]
    synth_combined_t = ttnn.from_torch(
        torch.from_numpy(synth_combined_np), dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT
    )

    # COLA normalisation on device for conv_transpose OLA path.
    inv_denom_np = _build_istft_inverse_denom_trim(input_length, filter_length, hop_length, window)
    inv_denom_t = _upload(inv_denom_np.reshape(1, -1), device, dtype=ttnn.float32)  # [1, output_length]

    conv_real, conv_imag = _build_conv_stft_kernels(filter_length, win_length)

    return TTTorchSTFTParams(
        conv_stft_real=_upload_rm(conv_real, dtype=ttnn.float32),
        conv_stft_imag=_upload_rm(conv_imag, dtype=ttnn.float32),
        conv_pad_len=filter_length // 2,
        istft_real=istft_real_t,
        istft_imag=istft_imag_t,
        synth_real=synth_real_t,
        synth_imag=synth_imag_t,
        synth_combined=synth_combined_t,
        inv_denom_tt=inv_denom_t,
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
        self.slice_cfg = dram_height_slice_config(2048)

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
        use_dram_slice = input_height > 2048
        slice_cfg = dram_height_slice_config(input_height)
        if use_dram_slice:
            conv_config = ttnn.Conv2dConfig(
                weights_dtype=ttnn.float32,
                deallocate_activation=True,
                config_tensors_in_dram=True,
                enable_act_double_buffer=False,
                enable_weights_double_buffer=False,
            )
        else:
            conv_config = self.conv_config
        x_rm = ttnn.to_layout(x_n1tc, ttnn.ROW_MAJOR_LAYOUT)
        key = (batch_size, input_height, slice_cfg.num_slices, use_dram_slice)
        if self._prep_key != key:
            self.slice_cfg = slice_cfg
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
                conv_config=conv_config,
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
            conv_config=conv_config,
            compute_config=self.compute_cfg,
            slice_config=self.slice_cfg,
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        self.weight_prepared = wpair[0]
        out = ttnn.reshape(result, [batch_size, int(oh), self.out_channels], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if use_dram_slice and out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        out = ttnn.permute(out, (0, 2, 1), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return out


class TTTorchSTFT:
    """TT port of :class:`TorchSTFT` (STFT/iSTFT via precomputed dense matrices).

    ``use_torch_stft_fallback=True`` runs the entire :meth:`transform` on CPU (``torch.stft``).
    ``use_torch_stft_conv_fallback=True`` runs only the strided conv on CPU float32; magnitude and
    phase stay on TT.  On trained harmonic waveforms cos(phase) PCC remains well below full fallback;
    use ``use_torch_stft_fallback=True`` for cos(phase) PCC > 0.99 vs reference.
    """

    def __init__(
        self,
        device: ttnn.Device,
        params: TTTorchSTFTParams,
        *,
        use_torch_stft_fallback: bool = False,
        use_torch_stft_conv_fallback: bool = False,
    ) -> None:
        self.device = device
        self.params = params
        self.eps = 1e-11
        self.phase_zero_floor = float(os.getenv("KOKORO_STFT_PHASE_ZERO_FLOOR", "1e-8"))
        self._use_torch_stft_fallback = use_torch_stft_fallback
        self._use_torch_stft_conv_fallback = use_torch_stft_conv_fallback and not use_torch_stft_fallback
        conv_fb = self._use_torch_stft_conv_fallback
        self._conv_real = _StridedStftConv(device, params.conv_stft_real, params.hop_length, use_torch_fallback=conv_fb)
        self._conv_imag = _StridedStftConv(device, params.conv_stft_imag, params.hop_length, use_torch_fallback=conv_fb)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self._synth_real_prep: Optional[ttnn.Tensor] = None
        self._synth_imag_prep: Optional[ttnn.Tensor] = None
        self._synth_prep_key: Optional[tuple] = None

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

    def _forward_stft_conv(self, x_bL: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """Strided conv STFT branches → ``(X_real, X_imag)`` each ``[B, K, F]``.

        Single device pass at any length: the strided conv2d streams the signal through bounded L1
        via DRAM height slicing.  The ``[B, L] -> [B, 1, L, 1]`` reshape is routed through TILE
        layout (the 2D ROW_MAJOR reshape would overflow L1 above ~95k samples; the TILE reshape +
        untilize, whose untilized width is 1, is tile-local and bounded).
        """
        L_in = int(x_bL.shape[-1])
        if L_in != self.params.input_length:
            raise ValueError(f"input length mismatch: got {L_in}, expected {self.params.input_length}")
        if x_bL.dtype != ttnn.float32:
            x_bL = ttnn.typecast(x_bL, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG
        B = int(x_bL.shape[0])

        # Reshape [B, L] -> [B, 1, L, 1] via TILE to avoid the 2D ROW_MAJOR reshape L1 overflow.
        if x_bL.layout != ttnn.TILE_LAYOUT:
            x_bL = ttnn.to_layout(x_bL, ttnn.TILE_LAYOUT, memory_config=mc)
        x_n1lc_tile = ttnn.reshape(x_bL, [B, 1, L_in, 1], memory_config=mc)
        x_n1lc = ttnn.to_layout(x_n1lc_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        ttnn.deallocate(x_n1lc_tile)

        x_padded = _reflect_pad_1d_dim2(x_n1lc, L_in, p.conv_pad_len)
        ttnn.deallocate(x_n1lc)
        L_padded = L_in + 2 * p.conv_pad_len

        X_real = self._conv_real(x_padded, B, L_padded)
        X_imag = self._conv_imag(x_padded, B, L_padded)
        ttnn.deallocate(x_padded)
        return X_real, X_imag

    def _magnitude_phase_from_xy(self, X_real: ttnn.Tensor, X_imag: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``sqrt(real^2+imag^2)`` and ``atan2`` on TT in fp32 (SFPU atan2 still rounds on BH)."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        X_real, owns_r = _to_fp32_if_needed(X_real, mc)
        X_imag, owns_i = _to_fp32_if_needed(X_imag, mc)
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
        if owns_r:
            ttnn.deallocate(X_real)
        if owns_i:
            ttnn.deallocate(X_imag)
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
        return magnitude, phase

    def transform(self, x_bL: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """
        Args:
            x_bL: ``[B, L]`` (TILE layout) where ``L == params.input_length``.

        Returns:
            ``(magnitude, phase)`` each ``[B, K, F]`` (TILE layout).

        Any ``input_length`` stays on device in a single DRAM-sliced conv2d pass
        (:meth:`_forward_stft_conv`); there is no length-based CPU fallback.

        Fallback dispatch:
        - ``use_torch_stft_fallback=True``: entire transform on CPU via ``torch.stft`` (highest PCC).
        - ``use_torch_stft_conv_fallback=True``: only the strided conv runs on CPU float32;
          atan2 + sqrt stay on device.
        - No fallbacks: BH BF16 throughout; cos(phase) PCC ~0.64 for Kokoro harmonic input.
        """
        if self._use_torch_stft_fallback:
            return self._transform_torch_fallback(x_bL)

        X_real, X_imag = self._forward_stft_conv(x_bL)
        return self._magnitude_phase_from_xy(X_real, X_imag)

    def _dense_istft_two_matmul(self, X_real: ttnn.Tensor, X_imag: ttnn.Tensor, B: int) -> ttnn.Tensor:
        """Dense iSTFT via two ``[B, K*F] @ [K*F, output_length]`` matmuls summed.

        ``y = X_real_flat @ istft_real + X_imag_flat @ istft_imag``.  Consumes its inputs.

        (A single fused matmul against the row-stacked ``[B_real; B_imag]`` matrix was tried but is
        slower: concatenating ``X_real``/``X_imag`` along the K=11 axis is not tile-aligned, so it
        falls back to ROW_MAJOR and injects an untilize→concat→tilize round-trip — three extra program
        dispatches that cost more than the matmul + add they replace.)
        """
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG
        X_real_flat = ttnn.reshape(X_real, [B, p.K * p.F], memory_config=mc)
        X_imag_flat = ttnn.reshape(X_imag, [B, p.K * p.F], memory_config=mc)
        ttnn.deallocate(X_real)
        ttnn.deallocate(X_imag)
        y_real = ttnn.matmul(
            X_real_flat, p.istft_real, memory_config=mc, compute_kernel_config=self.compute_kernel_config
        )
        y_imag = ttnn.matmul(
            X_imag_flat, p.istft_imag, memory_config=mc, compute_kernel_config=self.compute_kernel_config
        )
        y = ttnn.add(y_real, y_imag, memory_config=mc)
        ttnn.deallocate(y_real)
        ttnn.deallocate(y_imag)
        ttnn.deallocate(X_real_flat)
        ttnn.deallocate(X_imag_flat)
        while len(y.shape) > 2:
            y = ttnn.squeeze(y, 0)
        return ttnn.reshape(y, [B, 1, p.output_length], memory_config=mc)

    def _to_nhwc_rm(self, x_bkf: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, K, F]`` (TILE) → ``[B, 1, F, K]`` ROW_MAJOR for conv_transpose2d NHWC input.

        ROW_MAJOR so the frame axis (dim=2) can be sliced at arbitrary (non-tile-aligned)
        boundaries by the chunked iSTFT path.
        """
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG
        B = int(x_bkf.shape[0])
        x_bfk = ttnn.permute(x_bkf, (0, 2, 1), memory_config=mc)
        x_rm = ttnn.to_layout(x_bfk, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        ttnn.deallocate(x_bfk)
        return ttnn.reshape(x_rm, [B, 1, p.F, p.K], memory_config=mc)

    def _ct_ola_run(self, x_nhwc: ttnn.Tensor, n_frames: int) -> ttnn.Tensor:
        """Fused conv_transpose2d OLA on ``n_frames`` frames → raw windowed-sum ``[B, L_local, 1]`` ROW_MAJOR.

        ``x_nhwc`` is the channel-stacked ``[B, 1, n_frames, 2K]`` (real then imag bins) and the weight
        is the stacked ``synth_combined`` ``[2K, 1, n_fft, 1]``.  Because conv_transpose2d sums over
        input channels, this single conv yields ``y_real + y_imag`` directly — no second conv, no add.
        ``L_local = (n_frames - 1) * hop + n_fft``.  No reflect-pad trim, no COLA — the caller
        slices/trims and applies COLA.  Consumes (deallocates) ``x_nhwc``.
        """
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG
        B = int(x_nhwc.shape[0])
        in_channels = 2 * p.K

        # Match tt_conv_transpose1d_nlc dram-sliced path: no output_layout=TILE (avoids L1 OOM on BH).
        conv_cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.float32)
        conv_cfg.config_tensors_in_dram = True
        conv_cfg.deallocate_activation = True
        try:
            conv_cfg.enable_act_double_buffer = False
        except Exception:
            pass
        # HiFi4 kept: at this conv_transpose shape the op is data-movement/reader-bound, so HiFi2
        # halves MAC FLOPs but yields no device-time gain — not worth the (small) reconstruction-PCC
        # risk on the thin kModel fallback margin.
        compute_cfg = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        # Scale DRAM height slices with the chunk frame count.  The generic budget assumes
        # channels=1 and caps long sequences at 128 rows/slice, but this conv_transpose has only
        # K (=11) input channels, so far more frames fit per L1 slice.  Passing the real channel
        # count and a larger per-slice row budget (KOKORO_ISTFT_CT_ROWS_PER_SLICE) collapses the
        # 64 slices/direction down to a handful, cutting the per-slice
        # PaddedSlice/Halo/Move/Conv2d/SliceWrite op fan-out by the same factor.
        # ``target_rows_per_slice`` is the real L1 guard: num_slices >= ceil(n_frames/rows_per_slice),
        # so per-slice rows (hence activation footprint = rows * 2K * 2B) is bounded by it regardless
        # of the slice count.  The generic helper also floors at min_slices=8, which only pads short
        # sequences (production F~127k -> 64 slices is unaffected); drop that floor so this conv uses
        # the minimum slices its row budget allows, halving the per-slice op fan-out again.
        rows_per_slice = int(os.getenv("KOKORO_ISTFT_CT_ROWS_PER_SLICE", "2048"))
        num_slices = dram_height_slice_num_slices(n_frames, min_slices=1, target_rows_per_slice=rows_per_slice)
        slice_cfg = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=num_slices)
        logger.debug(
            f"TTTorchSTFT iSTFT fused conv_transpose OLA: n_frames={n_frames}, in_channels={in_channels}, "
            f"rows_per_slice={rows_per_slice} -> {num_slices} slices"
        )

        y, out_hw = ttnn.conv_transpose2d(
            input_tensor=x_nhwc,
            weight_tensor=p.synth_combined,
            device=self.device,
            in_channels=in_channels,
            out_channels=1,
            batch_size=B,
            input_height=n_frames,
            input_width=1,
            kernel_size=(p.filter_length, 1),
            stride=(p.hop_length, 1),
            padding=(0, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            groups=1,
            bias_tensor=None,
            conv_config=conv_cfg,
            compute_config=compute_cfg,
            dram_slice_config=slice_cfg,
            mirror_kernel=True,
            return_output_dim=True,
        )
        ttnn.deallocate(x_nhwc)
        oh, ow = int(out_hw[0]), int(out_hw[1])
        # y is in the conv_transpose2d internal layout; reshape to [B, L_local, C_out].
        y_sum = ttnn.reshape(y, (y.shape[0], oh * ow, y.shape[-1]), memory_config=mc)

        # ROW_MAJOR so the caller can slice/concat along the sample axis at arbitrary boundaries.
        if y_sum.layout != ttnn.ROW_MAJOR_LAYOUT:
            y_sum = ttnn.to_layout(y_sum, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        return y_sum

    def _inverse_conv_transpose(self, X_real: ttnn.Tensor, X_imag: ttnn.Tensor) -> ttnn.Tensor:
        """Conv-transpose OLA iSTFT — single device pass at any F, no CPU involvement.

        conv_transpose2d with synthesis kernels w[n]*iFFT[k,n] and stride=hop is the adjoint of the
        forward STFT strided conv2d: it computes the windowed OLA sum on device, streaming through
        bounded L1 via DRAM height slicing (:meth:`_ct_ola_run`).  After the OLA the reflect-pad
        margins are trimmed and the COLA normalisation (precomputed on device) is applied pointwise.
        """
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG
        B = int(X_real.shape[0])
        pad = p.filter_length // 2

        # Channel-stack [X_real | X_imag] -> [B, 1, F, 2K] so a single fused conv_transpose computes
        # y_real + y_imag.  The concat is along the last (channel) axis in ROW_MAJOR — no tilize/
        # untilize round-trip (unlike a TILE concat along the non-aligned K axis).
        xr_nhwc = self._to_nhwc_rm(X_real)  # [B, 1, F, K] ROW_MAJOR
        xi_nhwc = self._to_nhwc_rm(X_imag)  # [B, 1, F, K] ROW_MAJOR
        x_comb = ttnn.concat([xr_nhwc, xi_nhwc], dim=3, memory_config=mc)  # [B, 1, F, 2K]
        ttnn.deallocate(xr_nhwc)
        ttnn.deallocate(xi_nhwc)
        y_raw = self._ct_ola_run(x_comb, p.F)
        y_trim = ttnn.slice(y_raw, [0, pad, 0], [B, pad + p.output_length, 1], [1, 1, 1], memory_config=mc)
        ttnn.deallocate(y_raw)

        # [B, output_length, 1] → [B, output_length] → COLA multiply.  Tilize BEFORE the squeeze:
        # the 2D ROW_MAJOR reshape path allocates an L1 circular buffer scaled to the full output
        # width and overflows BH L1 above ~131072 samples, whereas the TILE reshape is tile-local.
        y_trim_tile = ttnn.to_layout(y_trim, ttnn.TILE_LAYOUT, memory_config=mc)
        ttnn.deallocate(y_trim)
        y_flat = ttnn.reshape(y_trim_tile, [B, p.output_length], memory_config=mc)
        ttnn.deallocate(y_trim_tile)
        y_norm = ttnn.multiply(y_flat, p.inv_denom_tt, memory_config=mc)
        ttnn.deallocate(y_flat)
        return ttnn.reshape(y_norm, [B, 1, p.output_length], memory_config=mc)

    def inverse(self, magnitude: ttnn.Tensor, phase: ttnn.Tensor) -> ttnn.Tensor:
        """
        Args:
            magnitude: ``[B, K, F]``.
            phase: ``[B, K, F]``.

        Returns:
            ``[B, 1, output_length]`` (matches ``TorchSTFT.inverse``'s trailing ``unsqueeze(-2)``).

        When the dense iSTFT matrix was not precomputed (``istft_real is None``, large F) this runs
        a single-pass DRAM-sliced conv_transpose2d OLA on device (:meth:`_inverse_conv_transpose`).
        No chunking, no CPU fallback.
        """
        p = self.params

        # No fp32 typecast of the inputs: cos/sin/multiply and the matmul (against a bf16 iSTFT
        # matrix) run directly on the incoming dtype.  The polar->rect conversion and the
        # well-conditioned dense matmul tolerate bf16 here (unlike the forward atan2 path), so the
        # two BF16=>FP32 typecast dispatches are pure overhead on this op-gap-bound path.
        cos_phase = ttnn.cos(phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        sin_phase = ttnn.sin(phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        X_real = ttnn.multiply(magnitude, cos_phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        X_imag = ttnn.multiply(magnitude, sin_phase, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(cos_phase)
        ttnn.deallocate(sin_phase)

        B = int(magnitude.shape[0])

        if p.istft_real is not None and p.istft_imag is not None:
            logger.info(f"TTTorchSTFT.inverse: dense two-matmul path (K*F={p.K * p.F}, F={p.F})")
            return self._dense_istft_two_matmul(X_real, X_imag, B)

        # Matrix skipped but F fits BH conv_transpose slice budget — on-device OLA.
        logger.info(f"TTTorchSTFT.inverse: conv_transpose2d OLA path (F={p.F}, dense matrix skipped)")
        y = self._inverse_conv_transpose(X_real, X_imag)
        ttnn.deallocate(X_real)
        ttnn.deallocate(X_imag)
        return y

    def _forward_stft_to_istft(self, x_bL: ttnn.Tensor) -> ttnn.Tensor:
        """STFT conv → iSTFT without mag/phase roundtrip.

        Skipping atan2 + cos/sin avoids BH BF16 phase errors that amplify near-zero bin
        noise: mag * cos(atan2_BF16(y, x)) can diverge from x by ~100× when phase is off by π.
        Direct X_real/X_imag passthrough keeps the error at the conv2d noise floor.
        """
        X_real, X_imag = self._forward_stft_conv(x_bL)
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG

        if X_real.dtype != ttnn.float32:
            X_real = ttnn.typecast(X_real, ttnn.float32, memory_config=mc)
        if X_imag.dtype != ttnn.float32:
            X_imag = ttnn.typecast(X_imag, ttnn.float32, memory_config=mc)

        B = int(X_real.shape[0])

        if p.istft_real is not None and p.istft_imag is not None:
            logger.info(f"TTTorchSTFT.forward: dense two-matmul iSTFT path (K*F={p.K * p.F}, F={p.F})")
            return self._dense_istft_two_matmul(X_real, X_imag, B)

        # Long sequences: conv_transpose2d OLA with DRAM height slicing (fully on device).
        logger.info(f"TTTorchSTFT.forward: conv_transpose2d OLA iSTFT path (F={p.F}, dense matrix skipped)")
        y = self._inverse_conv_transpose(X_real, X_imag)
        ttnn.deallocate(X_real)
        ttnn.deallocate(X_imag)
        return y

    def forward(self, x_bL: ttnn.Tensor) -> ttnn.Tensor:
        """STFT → iSTFT round trip (matches ``TorchSTFT.forward``)."""
        if self._use_torch_stft_fallback:
            mag, phase = self._transform_torch_fallback(x_bL)
            y = self.inverse(mag, phase)
            ttnn.deallocate(mag)
            ttnn.deallocate(phase)
            return y
        return self._forward_stft_to_istft(x_bL)

    __call__ = forward

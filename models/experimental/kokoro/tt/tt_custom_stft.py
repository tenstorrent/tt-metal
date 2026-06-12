# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN port of Kokoro :class:`~models.experimental.kokoro.reference.custom_stft.CustomSTFT`.

``CustomSTFT`` is the ``disable_complex=True`` STFT used by the Kokoro istftnet ``Generator``.
Unlike :class:`~models.experimental.kokoro.reference.istftnet.TorchSTFT` (which wraps
``torch.stft`` / ``torch.istft``), ``CustomSTFT`` is built entirely from ``conv1d`` /
``conv_transpose1d`` so it exports cleanly to ONNX.  This module ports that exact formulation
to device — there is **no** CPU fallback and **no** ``torch.stft`` anywhere.

Implementation
--------------
The strided 1-D DFT is realised as a strided **conv2d** (kernel ``[K, 1, n_fft, 1]``, stride
``hop`` along the height axis) and the iSTFT as the adjoint **conv_transpose2d**.  This is the
STFT primitive validated on Blackhole for the Kokoro ``n_fft=20`` config (conv2d + HiFi3 +
float32 storage); the generic ``conv1d`` NLC helper does not handle the tiny
(``in=1, out=K=11``) STFT kernel correctly.

Forward (``transform``)
    1. ``center=True`` pad of ``n_fft // 2`` on each side using **replicate** padding (``CustomSTFT``
       uses replicate, not the reflect padding of ``TorchSTFT``).
    2. Two strided conv2d projections give ``X_real = cos(2πkn/N)·w``, ``X_imag = -sin(2πkn/N)·w``
       → each ``[B, K, F]``.
    3. ``magnitude = sqrt(X_real² + X_imag² + 1e-14)``; ``phase = atan2(X_imag, X_real)`` with the
       ``(X_imag == 0) & (X_real < 0) -> π`` correction matching the reference / PyTorch atan2.

Inverse (``inverse``)
    ``CustomSTFT.inverse`` is an *approximate* real iFFT (uniform ``1/N`` scale, no DC/Nyquist
    half-scaling, no COLA normalisation — the trained Kokoro weights absorb the difference):

        real_part = magnitude · cos(phase);  imag_part = magnitude · sin(phase)
        y = conv_transpose2d(real_part, w·cos(2πkn/N)/N)
          - conv_transpose2d(imag_part, w·sin(2πkn/N)/N)      # note the minus
        y = y[..., pad : -pad]                                # undo center pad

Precision note (Blackhole)
--------------------------
BH rounds float32 inputs to BF16 before every MAC.  Near-zero DFT bins fall below that noise
floor and get sign-random phase; this is the documented BH STFT ceiling
(see ``feedback_bh_bf16_stft_ceiling`` in memory).  Weights / activations are kept in float32 to
push that floor as low as the hardware allows — there is no software fallback to escape it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import ttnn

from models.experimental.kokoro.tt.tt_conv import dram_height_slice_config, dram_height_slice_target_rows
from models.experimental.kokoro.tt.tt_torch_stft import _build_conv_stft_kernels


@dataclass(frozen=True)
class TTCustomSTFTParams:
    """Device-resident ``CustomSTFT`` conv kernels for a fixed (n_fft, hop, win) configuration.

    The kernels are length-agnostic, so unlike :class:`TTTorchSTFTParams` these do **not** bake an
    input length — the same params serve any ``transform`` / ``inverse`` frame count.
    """

    conv_fwd_real: ttnn.Tensor  # [K, 1, n_fft, 1] ROW_MAJOR — forward DFT real branch
    conv_fwd_imag: ttnn.Tensor  # [K, 1, n_fft, 1] ROW_MAJOR — forward DFT imag branch
    synth_real: ttnn.Tensor  # [K, 1, n_fft, 1] ROW_MAJOR — inverse real branch (w·cos/N)
    synth_imag: ttnn.Tensor  # [K, 1, n_fft, 1] ROW_MAJOR — inverse imag branch (w·sin/N)

    filter_length: int  # n_fft
    hop_length: int
    win_length: int
    K: int  # n_fft // 2 + 1
    pad_len: int  # n_fft // 2 (center pad)
    center: bool


def _build_custom_synth_kernels(n_fft: int, win_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Build ``CustomSTFT`` inverse kernels ``[K, 1, n_fft, 1]`` (IOHW) for conv_transpose2d.

    ``backward_real/imag[k, 0, n, 0] = w[n] · cos/sin(2πkn/N) / N`` — uniform ``1/N`` scale, no
    DC/Nyquist half-scaling (matches the reference ``CustomSTFT`` exactly).
    """
    window = torch.hann_window(win_length, periodic=True, dtype=torch.float32)
    if win_length < n_fft:
        window = torch.nn.functional.pad(window, (0, n_fft - win_length))
    elif win_length > n_fft:
        window = window[:n_fft]
    w = window.numpy().astype(np.float64)

    freq_bins = n_fft // 2 + 1
    k = np.arange(freq_bins)
    n = np.arange(n_fft)
    angle = 2.0 * np.pi * np.outer(k, n) / n_fft  # [K, n_fft]
    inv_window = w / n_fft

    backward_real = (np.cos(angle) * inv_window).astype(np.float32)[:, None, :, None]  # [K, 1, n_fft, 1]
    backward_imag = (np.sin(angle) * inv_window).astype(np.float32)[:, None, :, None]
    return backward_real, backward_imag


def _upload_rm(arr: np.ndarray, *, dtype: ttnn.DataType) -> ttnn.Tensor:
    return ttnn.from_torch(torch.from_numpy(arr.astype(np.float32)), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)


def preprocess_tt_custom_stft(
    *,
    filter_length: int,
    hop_length: int,
    win_length: int,
    weights_dtype: ttnn.DataType = ttnn.float32,
    center: bool = True,
) -> TTCustomSTFTParams:
    """Build ``CustomSTFT`` conv kernels on the host and upload them (ROW_MAJOR, float32 default).

    float32 storage keeps the small Kokoro harmonic-source DFT bins above the BH BF16 noise floor
    as far as the hardware allows.
    """
    if win_length != filter_length:
        raise ValueError(f"Only win_length == filter_length is supported (got {win_length} vs {filter_length})")

    # Forward kernels: cos·w and -sin·w (shared with the TorchSTFT conv builder — same forward DFT).
    fwd_real, fwd_imag = _build_conv_stft_kernels(filter_length, win_length)  # each [K, 1, n_fft, 1]
    bwd_real, bwd_imag = _build_custom_synth_kernels(filter_length, win_length)

    return TTCustomSTFTParams(
        conv_fwd_real=_upload_rm(fwd_real, dtype=weights_dtype),
        conv_fwd_imag=_upload_rm(fwd_imag, dtype=weights_dtype),
        synth_real=_upload_rm(bwd_real, dtype=weights_dtype),
        synth_imag=_upload_rm(bwd_imag, dtype=weights_dtype),
        filter_length=filter_length,
        hop_length=hop_length,
        win_length=win_length,
        K=filter_length // 2 + 1,
        pad_len=filter_length // 2,
        center=center,
    )


def _stft_conv_slice_config(input_height: int) -> ttnn.Conv2dSliceConfig:
    """Pick DRAM height slices for the forward STFT conv2d.

    Each slice still runs an L1 height-sharded conv internally; ``num_slices`` controls how
    much height is staged from DRAM per pass.  The generic helper defaults to ``min_slices=8``
    even when the full height fits, which adds unnecessary padded-slice / halo / move overhead
    on Kokoro-scale inputs.

    ``Conv2dL1FullSliceConfig`` (pure L1 chunking, no DRAM slice glue) is not usable here:
    this STFT geometry (1→11 ch, 20×1 kernel, stride 5, width 1) hits a BH conv2d program
    factory limit (``reader_indices`` CB sizing) for both L1Full and single-slice DRAM paths.
    """
    target = dram_height_slice_target_rows(input_height, channels=1, activation_dtype=ttnn.float32)
    if input_height <= target:
        return ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=2)
    return dram_height_slice_config(input_height, channels=1, activation_dtype=ttnn.float32)


def _time_slice_n1tc(x: ttnn.Tensor, t0: int, t1: int, *, memory_config: ttnn.MemoryConfig) -> ttnn.Tensor:
    b = int(x.shape[0])
    c = int(x.shape[3])
    return ttnn.slice(x, [0, 0, t0, 0], [b, 1, t1, c], [1, 1, 1, 1], memory_config=memory_config)


def _replicate_pad_dim2(x_n1tc: ttnn.Tensor, time_len: int, pad: int) -> ttnn.Tensor:
    """Replicate-pad along dim=2 for shape ``[B, 1, T, 1]`` (matches ``F.pad(mode='replicate')``)."""
    if pad <= 0:
        return x_n1tc
    mc = ttnn.DRAM_MEMORY_CONFIG
    first = _time_slice_n1tc(x_n1tc, 0, 1, memory_config=mc)  # [B, 1, 1, C]
    last = _time_slice_n1tc(x_n1tc, time_len - 1, time_len, memory_config=mc)
    left = ttnn.repeat(first, [1, 1, pad, 1], memory_config=mc)
    right = ttnn.repeat(last, [1, 1, pad, 1], memory_config=mc)
    ttnn.deallocate(first)
    ttnn.deallocate(last)
    out = ttnn.concat([left, x_n1tc, right], dim=2, memory_config=mc)
    ttnn.deallocate(left)
    ttnn.deallocate(right)
    return out


class _StridedStftConv:
    """Strided conv2d projection for one STFT branch (real or imag)."""

    def __init__(self, device: ttnn.Device, weight_rm: ttnn.Tensor, hop_length: int) -> None:
        self.device = device
        self.weight_rm = weight_rm
        self.weight_prepared = weight_rm
        self._prep_key: tuple | None = None
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

    def __call__(self, x_n1tc: ttnn.Tensor, batch_size: int, input_height: int) -> ttnn.Tensor:
        mc = ttnn.DRAM_MEMORY_CONFIG
        slice_cfg = _stft_conv_slice_config(input_height)
        if x_n1tc.layout == ttnn.ROW_MAJOR_LAYOUT:
            x_rm = x_n1tc
        else:
            x_rm = ttnn.to_layout(x_n1tc, ttnn.ROW_MAJOR_LAYOUT)
        key = (batch_size, input_height, slice_cfg.num_slices)
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
                slice_config=slice_cfg,
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
            slice_config=slice_cfg,
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        self.weight_prepared = wpair[0]
        out = ttnn.reshape(result, [batch_size, int(oh), self.out_channels], memory_config=mc)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT, memory_config=mc)
        return ttnn.permute(out, (0, 2, 1), memory_config=mc)


class TTCustomSTFT:
    """TT port of :class:`CustomSTFT` — conv2d STFT / conv_transpose2d iSTFT, no fallback.

    ``transform(x)`` returns ``(magnitude, phase)`` each ``[B, K, F]``.
    ``inverse(mag, phase)`` returns ``[B, 1, output_length]``.
    ``forward(x)`` is the ``transform`` → ``inverse`` round trip.
    """

    _EPS = 1e-14  # matches reference CustomSTFT.transform

    def __init__(self, device: ttnn.Device, params: TTCustomSTFTParams) -> None:
        self.device = device
        self.params = params
        self._conv_real = _StridedStftConv(device, params.conv_fwd_real, params.hop_length)
        self._conv_imag = _StridedStftConv(device, params.conv_fwd_imag, params.hop_length)
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ------------------------------------------------------------------
    # forward STFT
    # ------------------------------------------------------------------

    def transform(self, x_bL: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``[B, L]`` waveform → ``(magnitude, phase)`` each ``[B, K, F]`` (all on device)."""
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG
        L_in = int(x_bL.shape[-1])
        B = int(x_bL.shape[0])

        if x_bL.dtype != ttnn.float32:
            x_bL = ttnn.typecast(x_bL, ttnn.float32, memory_config=mc)

        # [B, L] -> [B, 1, L, 1] via TILE to avoid the 2D ROW_MAJOR reshape L1 overflow at large L.
        if x_bL.layout != ttnn.TILE_LAYOUT:
            x_bL = ttnn.to_layout(x_bL, ttnn.TILE_LAYOUT, memory_config=mc)
        x_n1lc_tile = ttnn.reshape(x_bL, [B, 1, L_in, 1], memory_config=mc)
        x_n1lc = ttnn.to_layout(x_n1lc_tile, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        ttnn.deallocate(x_n1lc_tile)

        if p.center:
            x_padded = _replicate_pad_dim2(x_n1lc, L_in, p.pad_len)
            ttnn.deallocate(x_n1lc)
            L_padded = L_in + 2 * p.pad_len
        else:
            x_padded = x_n1lc
            L_padded = L_in

        X_real = self._conv_real(x_padded, B, L_padded)
        X_imag = self._conv_imag(x_padded, B, L_padded)
        ttnn.deallocate(x_padded)

        return self._magnitude_phase(X_real, X_imag)

    def _magnitude_phase(self, X_real: ttnn.Tensor, X_imag: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``sqrt(r²+i²+eps)`` and ``atan2(i, r)`` with the ``atan2(0, neg) -> π`` correction."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        if X_real.dtype != ttnn.float32:
            X_real = ttnn.typecast(X_real, ttnn.float32, memory_config=mc)
        if X_imag.dtype != ttnn.float32:
            X_imag = ttnn.typecast(X_imag, ttnn.float32, memory_config=mc)

        mag_sq = ttnn.add(
            ttnn.multiply(X_real, X_real, memory_config=mc),
            ttnn.multiply(X_imag, X_imag, memory_config=mc),
            memory_config=mc,
        )
        magnitude = ttnn.sqrt(ttnn.add(mag_sq, self._EPS, memory_config=mc), memory_config=mc)
        phase = ttnn.atan2(X_imag, X_real, memory_config=mc)
        ttnn.deallocate(mag_sq)
        # (imag == 0) & (real < 0) -> π  (matches reference correction_mask)
        corr_mask = ttnn.logical_and(
            ttnn.eq(X_imag, 0.0, memory_config=mc),
            ttnn.lt(X_real, 0.0, memory_config=mc),
            memory_config=mc,
        )
        pi_fill = ttnn.full_like(phase, float(np.pi), memory_config=mc)
        phase = ttnn.where(corr_mask, pi_fill, phase, memory_config=mc)
        ttnn.deallocate(corr_mask)
        ttnn.deallocate(pi_fill)
        ttnn.deallocate(X_real)
        ttnn.deallocate(X_imag)
        return magnitude, phase

    # ------------------------------------------------------------------
    # inverse iSTFT
    # ------------------------------------------------------------------

    def _to_nhwc_rm(self, x_bkf: ttnn.Tensor) -> ttnn.Tensor:
        """``[B, K, F]`` (TILE) → ``[B, 1, F, K]`` ROW_MAJOR for conv_transpose2d NHWC input."""
        mc = ttnn.DRAM_MEMORY_CONFIG
        B = int(x_bkf.shape[0])
        x_bfk = ttnn.permute(x_bkf, (0, 2, 1), memory_config=mc)
        x_rm = ttnn.to_layout(x_bfk, ttnn.ROW_MAJOR_LAYOUT, memory_config=mc)
        ttnn.deallocate(x_bfk)
        out = ttnn.reshape(x_rm, [B, 1, int(x_bkf.shape[2]), int(x_bkf.shape[1])], memory_config=mc)
        ttnn.deallocate(x_rm)
        return out

    def _conv_transpose_branch(self, x_nhwc: ttnn.Tensor, synth_w: ttnn.Tensor, n_frames: int) -> ttnn.Tensor:
        """conv_transpose2d for one branch → ``[B, L_full, 1]`` TILE.  Consumes ``x_nhwc``."""
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG
        B = int(x_nhwc.shape[0])

        conv_cfg = ttnn.Conv2dConfig(weights_dtype=ttnn.float32)
        conv_cfg.config_tensors_in_dram = True
        conv_cfg.deallocate_activation = True
        try:
            conv_cfg.enable_act_double_buffer = False
        except Exception:
            pass
        compute_cfg = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi3,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        # Single-slice (no DRAM height slicing): the multi-slice conv_transpose corrupts the
        # windowed overlap-add across slice boundaries (the iFFT kernel spans n_fft samples, far
        # more than the per-slice row count, so each slice drops its neighbours' contributions).
        # A single slice is exact and fits BH L1 across the practical Kokoro range — measured PCC
        # 1.0 up to F≈60k (≈300k output samples).  Beyond ~F≈130k (max-phoneme silence padding)
        # the single-slice program OOMs; that extreme regime uses the TorchSTFT/torch.istft path
        # (``disable_complex=False``) instead.
        slice_cfg = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=1)

        y, out_hw = ttnn.conv_transpose2d(
            input_tensor=x_nhwc,
            weight_tensor=synth_w,
            device=self.device,
            in_channels=p.K,
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
        y = ttnn.reshape(y, (y.shape[0], oh * ow, y.shape[-1]), memory_config=mc)
        if y.layout != ttnn.TILE_LAYOUT:
            y = ttnn.to_layout(y, ttnn.TILE_LAYOUT, memory_config=mc)
        return y

    def inverse(self, magnitude: ttnn.Tensor, phase: ttnn.Tensor) -> ttnn.Tensor:
        """``(magnitude, phase)`` each ``[B, K, F]`` → ``[B, 1, output_length]`` (on device)."""
        p = self.params
        mc = ttnn.DRAM_MEMORY_CONFIG

        # conv_transpose2d output staging assumes a single batch; process B>1 per item.
        B0 = int(magnitude.shape[0])
        if B0 > 1:
            K, F0 = int(magnitude.shape[1]), int(magnitude.shape[2])
            outs = []
            for b in range(B0):
                mag_b = ttnn.slice(magnitude, [b, 0, 0], [b + 1, K, F0], [1, 1, 1], memory_config=mc)
                ph_b = ttnn.slice(phase, [b, 0, 0], [b + 1, K, F0], [1, 1, 1], memory_config=mc)
                outs.append(self.inverse(mag_b, ph_b))
                ttnn.deallocate(mag_b)
                ttnn.deallocate(ph_b)
            out = ttnn.concat(outs, dim=0, memory_config=mc)
            for t in outs:
                ttnn.deallocate(t)
            return out

        if magnitude.dtype != ttnn.float32:
            magnitude = ttnn.typecast(magnitude, ttnn.float32, memory_config=mc)
        if phase.dtype != ttnn.float32:
            phase = ttnn.typecast(phase, ttnn.float32, memory_config=mc)

        B = int(magnitude.shape[0])
        F = int(magnitude.shape[2])

        cos_ph = ttnn.cos(phase, memory_config=mc)
        sin_ph = ttnn.sin(phase, memory_config=mc)
        real_bct = ttnn.multiply(magnitude, cos_ph, memory_config=mc)  # [B, K, F]
        imag_bct = ttnn.multiply(magnitude, sin_ph, memory_config=mc)
        ttnn.deallocate(cos_ph)
        ttnn.deallocate(sin_ph)

        real_rec = self._conv_transpose_branch(self._to_nhwc_rm(real_bct), p.synth_real, F)
        ttnn.deallocate(real_bct)
        imag_rec = self._conv_transpose_branch(self._to_nhwc_rm(imag_bct), p.synth_imag, F)
        ttnn.deallocate(imag_bct)

        # y = real_rec - imag_rec  (the real iFFT minus on the imaginary branch).
        y_nlc = ttnn.subtract(real_rec, imag_rec, memory_config=mc)  # [B, L_full, 1]
        ttnn.deallocate(real_rec)
        ttnn.deallocate(imag_rec)

        # center=True trim: drop pad_len from each end along the length axis.
        L_full = int(y_nlc.shape[1])
        if p.center:
            y_trim = ttnn.slice(y_nlc, [0, p.pad_len, 0], [B, L_full - p.pad_len, 1], [1, 1, 1], memory_config=mc)
            ttnn.deallocate(y_nlc)
        else:
            y_trim = y_nlc

        # NLC [B, T, 1] -> BCT [B, 1, T] (matches reference ``conv_transpose`` channel-first output).
        y_out = ttnn.permute(y_trim, (0, 2, 1), memory_config=mc)
        ttnn.deallocate(y_trim)
        return y_out

    # ------------------------------------------------------------------
    # round trip
    # ------------------------------------------------------------------

    def forward(self, x_bL: ttnn.Tensor) -> ttnn.Tensor:
        """STFT → iSTFT round trip (matches ``CustomSTFT.forward``)."""
        mag, phase = self.transform(x_bL)
        y = self.inverse(mag, phase)
        ttnn.deallocate(mag)
        ttnn.deallocate(phase)
        return y

    __call__ = forward

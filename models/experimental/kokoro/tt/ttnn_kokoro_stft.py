# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN STFT / iSTFT using fixed DFT + strided conv2d (matches repo `CustomSTFT`)."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

import ttnn


def _compute_cfg(device):
    # NOTE: TTNN runtime warns HiFi4 + fp32 accumulation can be LESS accurate on WH B0 due to a HW bug.
    # Empirically, however, the STFT conv2d / conv_transpose2d outputs match PyTorch's CPU conv1d more
    # closely with HiFi4 here than HiFi3 — switching this site to HiFi3 regressed the istftnet e2e
    # waveform PCC by ~0.03. We keep HiFi4 only for these specific conv DFT projections.
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )


# Per-slice input height the STFT analysis conv2d can safely keep in L1 alongside an L1-resident
# sibling output (``real_o`` while ``fwd_i`` runs). Empirically 3000 matches the working baseline
# at num_slices=8 for ~24k-sample audio; longer utterances scale ``num_slices`` proportionally.
_STFT_FWD_TARGET_SLICE_H = 3000
# Frame count at which to switch the iSTFT synthesis path to DRAM-input + DRAM-sliced
# conv_transpose2d (matches ``_UpsConvTranspose1d`` strategy for long sequences). Below this we
# keep the original "L1 input, no slice config" path which works for short, wide STFTs
# (e.g. filter_length=800 / freq_bins=401 / frames=25).
_STFT_INV_DRAM_FRAMES_THRESHOLD = 256


def _stft_num_slices(height: int, target: int, min_slices: int = 8, max_slices: int = 512) -> int:
    """Pick a num_slices that keeps each conv2d slice ≤ ``target`` rows tall."""
    if height <= 0:
        return min_slices
    n = (int(height) + int(target) - 1) // int(target)
    return max(int(min_slices), min(int(max_slices), int(n)))


def _time_slice_time(
    x: ttnn.Tensor,
    t0: int,
    t1: int,
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """Slice time as [t0:t1). Supports (B, 1, T) or (B, 1, T, C)."""
    b = int(x.shape[0])
    rank = len(x.shape)
    kw: dict[str, Any] = {}
    if memory_config is not None:
        kw["memory_config"] = memory_config
    if rank == 3:
        return ttnn.slice(x, [0, 0, t0], [b, 1, t1], **kw)
    c = int(x.shape[3])
    return ttnn.slice(x, [0, 0, t0, 0], [b, 1, t1, c], **kw)


def _replicate_pad_1d(x: ttnn.Tensor, time_len: int, pad: int) -> ttnn.Tensor:
    """Replicate-pad time axis (dim=2) for shape (B, 1, T, 1)."""
    if pad <= 0:
        return x
    first = _time_slice_time(x, 0, 1)
    last = _time_slice_time(x, time_len - 1, time_len)
    left = ttnn.repeat(first, ttnn.Shape([1, 1, pad, 1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    right = ttnn.repeat(last, ttnn.Shape([1, 1, pad, 1]), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x_dram = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(x)
    out = ttnn.concat([left, x_dram, right], dim=2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(first)
    ttnn.deallocate(last)
    ttnn.deallocate(left)
    ttnn.deallocate(right)
    ttnn.deallocate(x_dram)
    out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
    return out


def _narrow_time_dim2_b1t(
    x: ttnn.Tensor,
    start: int,
    length: int,
    *,
    memory_config: Optional[ttnn.MemoryConfig] = None,
) -> ttnn.Tensor:
    """Narrow along dim=2 to [start : start+length) via `ttnn.slice` (may alias input; do not deallocate `x`)."""
    return _time_slice_time(x, start, start + length, memory_config=memory_config)


class _StridedStftConv:
    def __init__(self, device, weight_rm: ttnn.Tensor, hop_length: int):
        self.device = device
        self.weight_rm = weight_rm
        self.weight_prepared = weight_rm
        self._prep_key: Optional[tuple[int, int]] = None
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
        self.compute_cfg = _compute_cfg(device)
        self.dram_slice_config = ttnn.Conv2dSliceConfig(slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=8)

    def __call__(self, x_n1lc: ttnn.Tensor, batch_size: int, input_height: int) -> ttnn.Tensor:
        x_rm = ttnn.to_layout(x_n1lc, ttnn.ROW_MAJOR_LAYOUT)
        key = (batch_size, input_height)
        if self._prep_key != key:
            # Scale num_slices with input_height so per-slice static CBs don't clash with the
            # L1-resident sibling output (``real_o`` while ``fwd_i`` runs) on long utterances.
            num_slices = _stft_num_slices(input_height, _STFT_FWD_TARGET_SLICE_H)
            self.dram_slice_config = ttnn.Conv2dSliceConfig(
                slice_type=ttnn.Conv2dDRAMSliceHeight, num_slices=num_slices
            )
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
                slice_config=self.dram_slice_config,
            )
            self._prep_key = key
        result, [oh, _ow], weight_pair = ttnn.conv2d(
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
            slice_config=self.dram_slice_config,
            return_weights_and_bias=True,
            return_output_dim=True,
        )
        self.weight_prepared = weight_pair[0]
        oh_i = int(oh)
        result = ttnn.reshape(result, [batch_size, oh_i, self.out_channels], memory_config=ttnn.L1_MEMORY_CONFIG)
        result = ttnn.permute(result, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
        return ttnn.to_memory_config(result, ttnn.L1_MEMORY_CONFIG)


class _StridedIstftConvTranspose:
    def __init__(
        self,
        device,
        weight_rm: ttnn.Tensor,
        hop_length: int,
        freq_bins: int,
        *,
        pad_h: int = 0,
    ):
        self.device = device
        self.weight = weight_rm
        self.hop_length = int(hop_length)
        self.freq_bins = int(freq_bins)
        self.kernel_size = int(weight_rm.shape[2])
        self.pad_h = int(pad_h)
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
        self.compute_cfg = _compute_cfg(device)

    def __call__(self, x_bcl: ttnn.Tensor, batch_size: int, frames: int) -> ttnn.Tensor:
        # Stage input layout/dtype in DRAM (cheap, no L1 pressure) — conv_transpose2d streams from
        # DRAM input and picks its own L1 path. Forcing the input into L1 ahead of time leaves no
        # room for the conv's static circular buffers on long synthesis.
        x_bcl = ttnn.to_memory_config(x_bcl, ttnn.DRAM_MEMORY_CONFIG)
        if x_bcl.dtype != ttnn.float32:
            x_bcl = ttnn.typecast(x_bcl, ttnn.float32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_bcl = ttnn.permute(x_bcl, [0, 2, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_n1lc = ttnn.reshape(x_bcl, [batch_size, 1, frames, self.freq_bins], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result, [oh, ow], wpair = ttnn.conv_transpose2d(
            input_tensor=x_n1lc,
            weight_tensor=self.weight,
            in_channels=self.freq_bins,
            out_channels=1,
            device=self.device,
            bias_tensor=None,
            kernel_size=(self.kernel_size, 1),
            stride=(self.hop_length, 1),
            # ``padding=pad_h`` absorbs the otherwise-needed post-conv center trim (avoids a
            # ``ttnn.slice`` whose static CB region blew through L1 on long synthesis).
            padding=(self.pad_h, 0),
            output_padding=(0, 0),
            dilation=(1, 1),
            batch_size=batch_size,
            input_height=frames,
            input_width=1,
            conv_config=self.conv_config,
            compute_config=self.compute_cfg,
            groups=1,
            mirror_kernel=True,
            return_output_dim=True,
            return_weights_and_bias=True,
            dtype=ttnn.float32,
        )
        self.weight = wpair[0]
        flat = int(oh) * int(ow)
        result = ttnn.reshape(result, [batch_size, flat, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        result = ttnn.permute(result, [0, 2, 1], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        return result


def _to_conv2d_input(x_b1t: ttnn.Tensor, batch_size: int, time_len: int) -> ttnn.Tensor:
    x_b1t = ttnn.to_memory_config(x_b1t, ttnn.L1_MEMORY_CONFIG)
    if x_b1t.dtype != ttnn.float32:
        x_b1t = ttnn.typecast(x_b1t, ttnn.float32)
    x_b1t = ttnn.permute(x_b1t, [0, 2, 1], memory_config=ttnn.L1_MEMORY_CONFIG)
    return ttnn.reshape(x_b1t, [batch_size, 1, time_len, 1], memory_config=ttnn.L1_MEMORY_CONFIG)


class KokoroConvStft:
    """
    STFT / iSTFT on device: strided conv2d analysis and conv_transpose2d synthesis.

    Matches `CustomSTFT` in `models/experimental/kokoro/reference/kokoro_istftnet.py`
    (same DFT matrices, Hann window, replicate center pad, magnitude/phase and iSTFT merge).
    """

    def __init__(self, device, parameters: dict[str, Any]):
        self.device = device
        self.n_fft = int(parameters["n_fft"])
        self.hop_length = int(parameters["hop_length"])
        self.freq_bins = int(parameters["freq_bins"])
        self.center = bool(parameters["center"])
        self.pad_len = int(parameters["pad_len"])
        self.eps = float(parameters["eps"])
        # Absorb the inverse "center trim" into the conv_transpose2d padding instead of running a
        # post-conv ``ttnn.slice`` (see ``_StridedIstftConvTranspose.__call__`` comment).
        inv_pad = self.pad_len if self.center else 0
        self.fwd_r = _StridedStftConv(device, parameters["weight_forward_real"], self.hop_length)
        self.fwd_i = _StridedStftConv(device, parameters["weight_forward_imag"], self.hop_length)
        self.inv_r = _StridedIstftConvTranspose(
            device, parameters["weight_backward_real"], self.hop_length, self.freq_bins, pad_h=inv_pad
        )
        self.inv_i = _StridedIstftConvTranspose(
            device, parameters["weight_backward_imag"], self.hop_length, self.freq_bins, pad_h=inv_pad
        )

    def transform(self, waveform_b1t: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Return magnitude and phase, shape (B, freq_bins, num_frames)."""
        l1 = ttnn.L1_MEMORY_CONFIG
        dram = ttnn.DRAM_MEMORY_CONFIG
        waveform_b1t = ttnn.to_memory_config(waveform_b1t, l1)
        if waveform_b1t.dtype != ttnn.float32:
            waveform_b1t = ttnn.typecast(waveform_b1t, ttnn.float32)
        batch_size = int(waveform_b1t.shape[0])
        time_len = int(waveform_b1t.shape[2])
        x_n1lc = _to_conv2d_input(waveform_b1t, batch_size, time_len)
        if self.center:
            x_n1lc = _replicate_pad_1d(x_n1lc, time_len, self.pad_len)
            time_pad = time_len + 2 * self.pad_len
        else:
            time_pad = time_len
            x_n1lc = ttnn.to_memory_config(x_n1lc, dram)
        # Stage ``real_o`` in DRAM while ``fwd_i`` runs: keeping it L1-resident leaves no room for
        # ``fwd_i``'s static circular buffers on long utterances (clash at conv2d program load).
        real_o = self.fwd_r(x_n1lc, batch_size, time_pad)
        real_o = ttnn.to_memory_config(real_o, dram)
        imag_o = self.fwd_i(x_n1lc, batch_size, time_pad)
        imag_o = ttnn.to_memory_config(imag_o, dram)
        ttnn.deallocate(x_n1lc)
        # Run the magnitude / phase math against DRAM operands; ttnn elementwise ops can read/write
        # DRAM directly so we stay clear of L1 for any unbounded-length intermediate.
        r2 = ttnn.multiply(real_o, real_o, memory_config=dram)
        i2 = ttnn.multiply(imag_o, imag_o, memory_config=dram)
        eps_t = ttnn.full_like(real_o, self.eps, memory_config=dram)
        sq = ttnn.add(ttnn.add(r2, i2, memory_config=dram), eps_t, memory_config=dram)
        ttnn.deallocate(r2)
        ttnn.deallocate(i2)
        ttnn.deallocate(eps_t)
        mag = ttnn.sqrt(sq, memory_config=dram)
        ttnn.deallocate(sq)

        # Phase via ``atan2`` + ``imag==0 & real<0 → π`` correction, matching ``CustomSTFT.transform``.
        # NOTE: STFT phase PCC bottoms out at ~0.87 vs PyTorch regardless of trig op (verified by swapping
        # ``atan2``↔``acos``↔host-side ``torch.atan2``: identical PCC). Root cause is sub-ULP differences
        # between TTNN conv2d and PyTorch conv1d output at noise-floor magnitudes flipping the sign of
        # ``imag``, which atan2 amplifies to ±π by definition. Downstream ``noise_conv`` weights absorb this
        # noise — ``noise_conv[i]`` PCC stays at 0.999997+ on real inputs — so the 0.87 phase PCC is not
        # a true e2e bottleneck.
        phase = ttnn.atan2(imag_o, real_o, memory_config=dram)
        corr_mask = ttnn.logical_and(
            ttnn.eq(imag_o, 0.0, memory_config=dram),
            ttnn.lt(real_o, 0.0, memory_config=dram),
            memory_config=dram,
        )
        pi_fill = ttnn.full_like(phase, math.pi, memory_config=dram)
        phase = ttnn.where(corr_mask, pi_fill, phase, memory_config=dram)
        ttnn.deallocate(corr_mask)
        ttnn.deallocate(pi_fill)
        ttnn.deallocate(real_o)
        ttnn.deallocate(imag_o)
        return (
            ttnn.to_memory_config(mag, l1),
            ttnn.to_memory_config(phase, l1),
        )

    def inverse(
        self,
        magnitude: ttnn.Tensor,
        phase: ttnn.Tensor,
        length: Optional[int] = None,
    ) -> ttnn.Tensor:
        """Synthesize waveform, shape (B, 1, num_samples)."""
        l1 = ttnn.L1_MEMORY_CONFIG
        dram = ttnn.DRAM_MEMORY_CONFIG
        # On long synthesis (e.g. ~36k+ output samples), the intermediate cos/sin/real/imag tensors
        # plus the two inv_r/inv_i static circular buffer regions exceed per-core L1. Keep the math
        # in DRAM so only the conv slicer touches L1.
        frames_hint = int(magnitude.shape[2])
        work_mc = dram if frames_hint > _STFT_INV_DRAM_FRAMES_THRESHOLD else l1
        magnitude = ttnn.to_memory_config(magnitude, work_mc)
        phase = ttnn.to_memory_config(phase, work_mc)
        if magnitude.dtype != ttnn.float32:
            magnitude = ttnn.typecast(magnitude, ttnn.float32, memory_config=work_mc)
        if phase.dtype != ttnn.float32:
            phase = ttnn.typecast(phase, ttnn.float32, memory_config=work_mc)
        batch_size = int(magnitude.shape[0])
        frames = int(magnitude.shape[2])
        cos_phase = ttnn.cos(phase, memory_config=work_mc)
        sin_phase = ttnn.sin(phase, memory_config=work_mc)
        # Do not deallocate magnitude/phase here: they are caller-owned (PCC tests read mag/phase after
        # inverse). KokoroGenerator already deallocates spec/phase after stft.inverse returns.
        real_part = ttnn.multiply(magnitude, cos_phase, memory_config=work_mc)
        imag_part = ttnn.multiply(magnitude, sin_phase, memory_config=work_mc)
        ttnn.deallocate(cos_phase)
        ttnn.deallocate(sin_phase)
        # Stage ``real_rec`` in DRAM while ``inv_i`` runs (mirror of the ``fwd_r``/``fwd_i`` fix):
        # keeping it L1-resident leaves no room for ``inv_i``'s static circular buffers on long
        # synthesis. Also free ``real_part`` before ``inv_i`` to keep L1 headroom.
        real_rec = self.inv_r(real_part, batch_size, frames)
        ttnn.deallocate(real_part)
        real_rec = ttnn.to_memory_config(real_rec, work_mc)
        imag_rec = self.inv_i(imag_part, batch_size, frames)
        ttnn.deallocate(imag_part)
        imag_rec = ttnn.to_memory_config(imag_rec, work_mc)
        # Long synthesised waveforms (multi-second audio → 100k+ samples) don't fit in L1 alongside the
        # slice op's static circular buffers on a single core, so keep the trim/narrow path in DRAM.
        waveform = ttnn.subtract(real_rec, imag_rec, memory_config=dram)
        ttnn.deallocate(real_rec)
        ttnn.deallocate(imag_rec)
        # The center trim is absorbed by ``conv_transpose2d`` padding in ``_StridedIstftConvTranspose``
        # — no slice needed here. The length trim only runs if the caller asked for a shorter output
        # than the conv produced.
        if length is not None:
            cur = int(waveform.shape[2])
            take = min(int(length), cur)
            if take < cur:
                # ROW_MAJOR slice handles byte-aligned offsets with smaller static CBs than TILE slice.
                waveform = ttnn.to_layout(waveform, ttnn.ROW_MAJOR_LAYOUT)
                waveform = _narrow_time_dim2_b1t(waveform, 0, take, memory_config=dram)
        return waveform

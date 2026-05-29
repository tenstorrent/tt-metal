# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 vocoder bandwidth-extension wrapper (Stage C).

Mirror of ``_STFTFn``, ``MelSTFT`` and ``VocoderWithBWE`` from
``LTX-2/packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py``
lines 419-594.

Pipeline:

    mel (B, 2, T, mel_bins)
        ──> main LTXVocoder ──> waveform_24k (B, 2, T*160)
            ├── pad to multiple of hop_length (=80)
            ├── LTXMelSTFT  (B*2, T_pad) -> log-mel (B*2, 64, T_frames)
            ├── reshape to (B, 2, T_frames, 64)
            ├── bwe_generator (LTXVocoder w/ BWE config) ──> residual (B, 2, T_out)
            └── resampler (Hann-window sinc, ratio=2) ──> skip (B, 2, T_out)
        ──> clamp(residual + skip, -1, 1)[..., :output_length]

The whole forward runs in fp32 - the reference wraps it in
``torch.autocast(dtype=float32)``; on device every conv is constructed
with ``dtype=ttnn.float32`` which routes to HiFi4 + ``fp32_dest_acc``.
"""

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.module import Module, Parameter
from .vocoder_ltx import LTXVocoder, _replicate_pad_t, _zero_pad_t, _zero_stuff_t

# ---------------------------------------------------------------------------
# LTX_STFTFn — STFT expressed as a host-side ``unfold`` + on-device matmul.
#
# The reference stores ``forward_basis`` of shape ``(n_freqs*2, 1, win_length)``
# = ``(514, 1, 512)`` where the first ``n_freqs`` rows are the real part and
# the next ``n_freqs`` rows are the imaginary part. We reshape it to
# ``(win_length, n_freqs*2)`` and matmul against the windowed waveform.
# ---------------------------------------------------------------------------


class LTX_STFTFn(Module):
    """Causal STFT expressed as a single matmul on a windowed waveform.

    Mathematically identical to convolving ``y`` (shape ``(B, T)``) with
    ``forward_basis`` (shape ``(n_freqs*2, 1, win_length)``) at
    ``stride=hop_length`` after left-padding by ``win_length - hop_length``
    samples (causal — no lookahead).

    We DO NOT use ``Conv1dViaConv3d`` for this step: the conv3d kernel
    enforces ``C_in_block=32`` in fp32 mode, so a 512-tap kernel with C_in=1
    blows up the static circular-buffer allocation to ~4 MB per core
    (``patch_size = kernel_volume * C_in_block = 512*32 = 16384`` fp32 = 64
    KB per CB page × 32 pages × N cores > L1). Instead we host-side ``unfold``
    the (padded) waveform into ``(B, T_frames, win_length)`` and matmul
    against the basis reshaped to ``(win_length, n_freqs*2)``. The matmul
    is fp32 end-to-end (HiFi4 + ``fp32_dest_acc`` for matmuls), which gives
    the same bit-level fidelity the conv1d would have produced.

    The unfold + upload only materializes ``win_length × T_frames`` floats
    per channel — about 256 KB per stereo channel at 24 kHz / 80 hop / 512
    win, which is well within bandwidth budget.

    Layout: input is ``(B, T, 1)`` ROW_MAJOR on device. Output magnitude
    and phase are ``(B, T_frames, n_freqs)`` ROW_MAJOR.

    The ``forward_basis`` buffer is loaded from the checkpoint as a
    Parameter — we do not recompute it on host. ``inverse_basis`` is part
    of the checkpoint but unused (it belongs to the iSTFT path) and silently
    dropped.
    """

    def __init__(
        self,
        *,
        filter_length: int,
        hop_length: int,
        win_length: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_freqs = filter_length // 2 + 1
        self.left_pad = max(0, win_length - hop_length)
        self.mesh_device = mesh_device
        self.dtype = dtype

        # Basis stored as a (1, win_length, n_freqs*2) Parameter for matmul.
        # We keep a host copy as well so we can hand-load it without doing
        # an extra device→host roundtrip in ``_prepare_torch_state``.
        self.forward_basis = Parameter(
            total_shape=[1, win_length, self.n_freqs * 2],
            device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Reshape ``forward_basis (n_freqs*2, 1, win_length)`` into the
        matmul-friendly ``(1, win_length, n_freqs*2)`` form.
        """
        if "forward_basis" in state:
            w = state.pop("forward_basis")
            assert w.dim() == 3 and tuple(w.shape) == (
                self.n_freqs * 2,
                1,
                self.win_length,
            ), (
                f"forward_basis shape mismatch: expected "
                f"({self.n_freqs * 2}, 1, {self.win_length}), got {tuple(w.shape)}"
            )
            # (Cout=n*2, 1, K) -> (K, Cout) -> (1, K, Cout)
            state["forward_basis"] = w.squeeze(1).t().contiguous().unsqueeze(0).float()
        # ``inverse_basis`` is part of the checkpoint but is only used by the
        # iSTFT path which we don't implement here. Drop it silently.
        state.pop("inverse_basis", None)

    def forward(self, y_BTC: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``y_BTC``: ``(B, T, 1)`` ROW_MAJOR (waveform, one channel).

        Returns ``(magnitude, phase)`` each of shape ``(B, T_frames, n_freqs)``
        ROW_MAJOR.

        We currently host-side ``unfold`` the input: device → host →
        ``F.pad`` left + ``unfold(T, win, hop)`` → upload as ``(B, T_frames,
        win_length)``. The matmul against the basis stays on device.
        """
        assert y_BTC.layout == ttnn.ROW_MAJOR_LAYOUT, f"expected ROW_MAJOR, got {y_BTC.layout}"
        assert y_BTC.shape[2] == 1, f"STFT input must have C=1, got {y_BTC.shape[2]}"

        # Pull the waveform back to host. We're going to re-upload it as
        # a windowed view; we can't easily unfold on device.
        y_host = ttnn.to_torch(ttnn.get_device_tensors(y_BTC)[0])
        # (B, T, 1) -> (B, T)
        y_host = y_host.squeeze(-1).float().contiguous()
        # Left-pad by ``win_length - hop_length`` zeros (causal).
        y_padded = torch.nn.functional.pad(y_host, (self.left_pad, 0))
        # Unfold to (B, T_frames, win_length).
        y_windowed = y_padded.unfold(dimension=-1, size=self.win_length, step=self.hop_length)
        y_windowed = y_windowed.contiguous().float()
        B, T_frames, win_length = y_windowed.shape
        assert win_length == self.win_length

        # Upload as TILE (the matmul wants tiled inputs).
        y_tile = ttnn.from_torch(
            y_windowed,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
        )

        # Matmul with basis (1, win_length, n_freqs*2). Result: (B, T_frames, n_freqs*2).
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        spec_tile = ttnn.matmul(
            y_tile,
            self.forward_basis.data,
            compute_kernel_config=compute_kernel_config,
        )
        ttnn.deallocate(y_tile)

        # Slice real [0:n_freqs] and imag [n_freqs:2*n_freqs] along the
        # last axis. ttnn.slice prefers ROW_MAJOR for non-tile-aligned slices.
        spec = ttnn.to_layout(spec_tile, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(spec_tile)
        real = ttnn.slice(spec, [0, 0, 0], [B, T_frames, self.n_freqs])
        imag = ttnn.slice(spec, [0, 0, self.n_freqs], [B, T_frames, self.n_freqs * 2])
        ttnn.deallocate(spec)

        # magnitude = sqrt(real^2 + imag^2);  phase = atan2(imag, real).
        # All these eltwise ops run in fp32 since Parameters / outputs are
        # fp32 and the compute kernel for eltwise honors fp32_dest_acc.
        real_sq = ttnn.multiply(real, real)
        imag_sq = ttnn.multiply(imag, imag)
        mag_sq = ttnn.add(real_sq, imag_sq)
        ttnn.deallocate(real_sq)
        ttnn.deallocate(imag_sq)
        magnitude = ttnn.sqrt(mag_sq)
        ttnn.deallocate(mag_sq)

        phase = ttnn.atan2(imag, real)
        ttnn.deallocate(real)
        ttnn.deallocate(imag)
        return magnitude, phase


# ---------------------------------------------------------------------------
# LTXMelSTFT — wraps the STFT with a mel filterbank projection.
# ---------------------------------------------------------------------------


class LTXMelSTFT(Module):
    """Causal log-mel spectrogram:

        magnitude, _ = stft_fn(y)
        mel         = mel_basis @ magnitude                # (B, T, n_freqs) -> (B, T, n_mels)
        log_mel     = log(clamp(mel, min=1e-5))

    The mel filterbank ``mel_basis`` is loaded from the checkpoint as a
    Parameter (shape ``(n_mel_channels, n_freqs)``).

    Layout: input is laid out as ``(B, T, 1)`` ROW_MAJOR. The output log-mel
    is returned as ``(B, T_frames, n_mels)`` ROW_MAJOR — channel-on-the-end
    matches the rest of the audio_ops convention.
    """

    def __init__(
        self,
        *,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.n_freqs = filter_length // 2 + 1
        self.mesh_device = mesh_device
        self.dtype = dtype

        self.stft_fn = LTX_STFTFn(
            filter_length=filter_length,
            hop_length=hop_length,
            win_length=win_length,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        # The mel filterbank lives as a (n_mels, n_freqs) matmul weight.
        # We store it transposed to (n_freqs, n_mels) so that
        # ``ttnn.matmul(magnitude_BTF, mel_basis_T)`` computes
        # ``magnitude @ mel_basis.T`` which is the channel-last form of
        # the reference's ``mel_basis @ magnitude``.
        self.mel_basis = Parameter(
            total_shape=[1, self.n_freqs, self.n_mel_channels],
            device=mesh_device,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "mel_basis" in state:
            mb = state.pop("mel_basis").float()
            assert mb.dim() == 2 and tuple(mb.shape) == (
                self.n_mel_channels,
                self.n_freqs,
            ), (
                f"mel_basis shape mismatch: expected " f"({self.n_mel_channels}, {self.n_freqs}), got {tuple(mb.shape)}"
            )
            # Transpose to (n_freqs, n_mels) and add the leading batch axis to
            # match the Parameter's total_shape.
            state["mel_basis"] = mb.t().contiguous().unsqueeze(0)

    def forward(self, y_BT: ttnn.Tensor) -> ttnn.Tensor:
        """``y_BT``: ``(B, T, 1)`` ROW_MAJOR waveform. Returns log-mel
        ``(B, T_frames, n_mels)`` ROW_MAJOR.
        """
        magnitude, phase = self.stft_fn(y_BT)
        ttnn.deallocate(phase)  # not needed for log-mel

        # The matmul wants TILE-layout inputs.
        mag_tile = ttnn.to_layout(magnitude, ttnn.TILE_LAYOUT)
        ttnn.deallocate(magnitude)
        mel_basis_tile = self.mel_basis.data  # already TILE (Parameter default)
        # (B, T, n_freqs) x (1, n_freqs, n_mels) -> (B, T, n_mels)
        mel = ttnn.matmul(mag_tile, mel_basis_tile)
        ttnn.deallocate(mag_tile)

        # log(clamp(mel, min=1e-5))
        clamped = ttnn.clamp(mel, 1e-5, float("inf"))
        ttnn.deallocate(mel)
        log_mel = ttnn.log(clamped)
        ttnn.deallocate(clamped)

        # Return ROW_MAJOR for consistency with the rest of the pipeline.
        return ttnn.to_layout(log_mel, ttnn.ROW_MAJOR_LAYOUT)


# ---------------------------------------------------------------------------
# Hann-window sinc UpSample1d — used by the BWE resampler (24 kHz -> 48 kHz).
#
# Stage B's LTXUpSample1d uses kaiser-sinc and a different pad/crop recipe.
# The Hann-window variant in the reference has its own kernel construction
# and its own pad/crop amounts (see vocoder.py:94-106, 121-126). We mirror
# them here, reusing the same shifted multiply-accumulate pattern as Stage B
# for the depthwise lowpass step.
# ---------------------------------------------------------------------------


def _make_hann_sinc_kernel_1d(*, ratio: int) -> tuple[torch.Tensor, int, int, int, int]:
    """Return ``(kernel, kernel_size, pad, pad_left, pad_right)`` for the
    Hann-window sinc resampler variant from the reference (vocoder.py:94-106).

    Hyperparams (constants in the reference):
        rolloff = 0.99
        lowpass_filter_width = 6
    """
    rolloff = 0.99
    lowpass_filter_width = 6
    width = math.ceil(lowpass_filter_width / rolloff)
    kernel_size = 2 * width * ratio + 1
    pad = width
    pad_left = 2 * width * ratio
    pad_right = kernel_size - ratio

    time_axis = (torch.arange(kernel_size, dtype=torch.float64) / ratio - width) * rolloff
    time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
    window = torch.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
    sinc_filter = torch.sinc(time_axis) * window * rolloff / ratio
    return sinc_filter.float().reshape(kernel_size), kernel_size, pad, pad_left, pad_right


class LTXHannUpSample1d(Module):
    """Hann-window sinc upsampler.

    Mirrors the reference ``UpSample1d(window_type="hann")``:

        x_pad = F.pad(x, (pad, pad), mode='replicate')
        y     = ratio * F.conv_transpose1d(x_pad, filter, stride=ratio, groups=C)
        y     = y[..., pad_left:-pad_right]

    On device we implement the conv_transpose as zero-stuff + zero-pad +
    depthwise conv (shifted multiply-accumulate) — same trick as Stage B's
    ``LTXUpSample1d``, just with the Hann-specific kernel and pad amounts.

    The filter is not stored in the checkpoint (``persistent=False`` in the
    reference) — we always rebuild it on host at construction time.
    """

    def __init__(
        self,
        *,
        ratio: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio
        self.mesh_device = mesh_device
        self.dtype = dtype

        kernel, k, pad, pad_left_crop, pad_right_crop = _make_hann_sinc_kernel_1d(ratio=ratio)
        self.kernel_size = k
        self.pad = pad
        self.pad_left_crop = pad_left_crop
        self.pad_right_crop = pad_right_crop
        self._taps_cpu = kernel.tolist()

        # The kernel is a constant derived from the window formula — not a
        # learned weight, not in any checkpoint. Store as a plain device tensor,
        # not a Parameter, so it never appears in named_parameters() and
        # load_torch_state_dict never expects it as a mandatory key.
        k_tt = ttnn.from_torch(kernel.reshape(1, 1, k), device=mesh_device, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT)
        self._filter_tt = k_tt

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # The reference resampler's filter is not persistent and won't be in
        # any state_dict we ever load. If somebody does pass one in, sync our
        # taps and the on-device buffer to match.
        if "filter" in state:
            t = state.pop("filter")
            assert tuple(t.shape) == (1, 1, self.kernel_size)
            self._taps_cpu = t.reshape(self.kernel_size).float().tolist()
            # No-op for the Parameter — leave the host-baked filter in place.
            # We pop the key so the loader doesn't complain about "missing".
            # (If we wanted to honor the checkpoint, we'd load_torch_tensor
            # here too; but the reference says these are non-persistent.)

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        B, T, C = x_BTC.shape

        # Replicate-pad along T with ``pad`` each side.
        x_pad = _replicate_pad_t(x_BTC, self.pad, self.pad, self.mesh_device)
        # Zero-stuff to length ratio * T_pad - (ratio - 1).
        x_zs = _zero_stuff_t(x_pad, stride=self.stride, mesh_device=self.mesh_device)
        ttnn.deallocate(x_pad)
        # Zero-pad ``kernel_size - 1`` each side so the equivalent Conv1d
        # produces the same length as the reference's ConvTranspose1d.
        x_padded = _zero_pad_t(x_zs, self.kernel_size - 1, self.kernel_size - 1, self.mesh_device)
        ttnn.deallocate(x_zs)

        T_full = x_padded.shape[1]
        T_out = T_full - self.kernel_size + 1
        y = None
        for j in range(self.kernel_size):
            w = float(self._taps_cpu[j]) * float(self.ratio)
            slice_j = ttnn.slice(x_padded, [0, j, 0], [B, j + T_out, C])
            scaled = ttnn.multiply(slice_j, w)
            ttnn.deallocate(slice_j)
            if y is None:
                y = scaled
            else:
                y_new = ttnn.add(y, scaled)
                ttnn.deallocate(y)
                ttnn.deallocate(scaled)
                y = y_new
        ttnn.deallocate(x_padded)

        # Crop pad_left_crop from front, pad_right_crop from back.
        T_y = y.shape[1]
        y_cropped = ttnn.slice(y, [0, self.pad_left_crop, 0], [B, T_y - self.pad_right_crop, C])
        ttnn.deallocate(y)
        return y_cropped


# ---------------------------------------------------------------------------
# LTXVocoderWithBWE — full bandwidth extension wrapper.
# ---------------------------------------------------------------------------


class LTXVocoderWithBWE(Module):
    """Vocoder + bandwidth extension. Mirrors ``VocoderWithBWE``.

    Constructor takes the two LTXVocoder instances (main + bwe_generator),
    the LTXMelSTFT, plus the sampling-rate / hop-length config. We build the
    Hann-window resampler internally — it's not stored in the checkpoint.

    The forward pass runs entirely in fp32: every op below is invoked on
    tensors uploaded with ``dtype=ttnn.float32`` and every conv is wired to
    HiFi4 + ``fp32_dest_acc``.
    """

    def __init__(
        self,
        *,
        vocoder: LTXVocoder,
        bwe_generator: LTXVocoder,
        mel_stft: LTXMelSTFT,
        input_sampling_rate: int,
        output_sampling_rate: int,
        hop_length: int,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        self.mesh_device = mesh_device
        self.dtype = dtype

        # Hann-window resampler. ratio = output / input (e.g. 48000 / 24000 = 2).
        ratio = output_sampling_rate // input_sampling_rate
        assert (
            ratio * input_sampling_rate == output_sampling_rate
        ), "output_sampling_rate must be an integer multiple of input_sampling_rate"
        self.resampler = LTXHannUpSample1d(ratio=ratio, mesh_device=mesh_device, dtype=dtype)

    # The reference module exposes a couple of properties that proxy to the
    # main vocoder's conv_pre/post. We don't need them on device, but keep
    # the interface shape for callers that might walk the module tree.
    @property
    def conv_pre(self):
        return self.vocoder.conv_pre

    @property
    def conv_post(self):
        return self.vocoder.conv_post

    def _compute_mel_device(self, x_BCT_torch: torch.Tensor) -> torch.Tensor:
        """Compute log-mel from waveform on device.

        Args:
            x_BCT_torch: torch waveform of shape ``(B, C, T)`` with C=2 stereo.

        Returns:
            torch log-mel ``(B, C, n_mels, T_frames)`` on host (so the caller
            can transpose and feed it to the bwe_generator which itself
            uploads its input).
        """
        B, C, T = x_BCT_torch.shape
        # Flatten stereo: (B, C, T) -> (B*C, T).
        flat = x_BCT_torch.reshape(B * C, T).float().contiguous()
        # Upload as (B*C, T, 1) ROW_MAJOR.
        flat_BTC = flat.unsqueeze(-1)  # (B*C, T, 1)
        y_dev = ttnn.from_torch(
            flat_BTC,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
        )
        log_mel_dev = self.mel_stft(y_dev)  # (B*C, T_frames, n_mels)
        ttnn.deallocate(y_dev)
        log_mel_host = ttnn.to_torch(ttnn.get_device_tensors(log_mel_dev)[0])
        ttnn.deallocate(log_mel_dev)
        # (B*C, T_frames, n_mels) -> (B, C, n_mels, T_frames)
        T_frames = log_mel_host.shape[1]
        n_mels = log_mel_host.shape[2]
        log_mel_host = log_mel_host.reshape(B, C, T_frames, n_mels).transpose(2, 3).contiguous()
        return log_mel_host

    def _resample_device(self, x_BCT_torch: torch.Tensor) -> torch.Tensor:
        """Run the Hann-window resampler on a torch ``(B, C, T)`` waveform.

        Returns the resampled waveform as a torch ``(B, C, T*ratio)``.
        """
        B, C, T = x_BCT_torch.shape
        # Upload as (B, T, C) ROW_MAJOR.
        x_BTC = x_BCT_torch.transpose(1, 2).float().contiguous()
        x_dev = ttnn.from_torch(x_BTC, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)
        y_dev = self.resampler(x_dev)
        ttnn.deallocate(x_dev)
        y_host = ttnn.to_torch(ttnn.get_device_tensors(y_dev)[0])
        ttnn.deallocate(y_dev)
        # (B, T_out, C) -> (B, C, T_out)
        return y_host.transpose(1, 2).contiguous()

    def _bwe_from_waveform(self, x: torch.Tensor, *, input_dtype: torch.dtype, output_length: int) -> torch.Tensor:
        """Run only the BWE half of the pipeline starting from a precomputed
        24 kHz waveform ``x`` of shape ``(B, C, T_low)``.

        Mirrors lines 580-594 of the reference forward (everything after the
        main vocoder). Returns ``(B, C, T_out)`` 48 kHz waveform clamped to
        [-1, 1], trimmed to ``output_length`` samples.
        """
        assert x.dim() == 3, f"x must be (B, C, T), got {tuple(x.shape)}"
        B, C, length_low_rate = x.shape

        # Pad to multiple of hop_length on the right (zero pad).
        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            pad_right = self.hop_length - remainder
            x = torch.nn.functional.pad(x, (0, pad_right))

        # Compute log-mel on device, return host (B, C, n_mels, T_frames).
        mel = self._compute_mel_device(x)

        # bwe_generator expects (B, C, T_frames, n_mels) — transpose.
        mel_for_bwe = mel.transpose(2, 3).contiguous()
        residual = self.bwe_generator(mel_for_bwe)

        # Resample x (24kHz) to 48kHz.
        skip = self._resample_device(x)
        assert residual.shape == skip.shape, f"residual {tuple(residual.shape)} != skip {tuple(skip.shape)}"

        out = torch.clamp(residual + skip, -1.0, 1.0)
        out = out[..., :output_length]
        return out.to(input_dtype)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """End-to-end forward.

        Args:
            mel_spec: ``(B, 2, T, mel_bins)`` for stereo. Matches the
                reference's call-site convention. Returned waveform has the
                same dtype as the input.

        Returns:
            ``(B, 2, T_out)`` waveform clamped to ``[-1, 1]``, where
            ``T_out = T * 160 * output_sampling_rate // input_sampling_rate``.
        """
        input_dtype = mel_spec.dtype

        # Main vocoder → 24 kHz waveform (B, 2, T_low).
        x = self.vocoder(mel_spec.float())
        B, C, length_low_rate = x.shape
        output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate

        return self._bwe_from_waveform(x, input_dtype=input_dtype, output_length=output_length)

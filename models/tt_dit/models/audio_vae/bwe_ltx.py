# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 vocoder bandwidth-extension wrapper (Stage C).

Pipeline:

    mel (B, 2, T, mel_bins)
        ──> main LTXVocoder ──> waveform_24k (B, 2, T*160)
            ├── pad to multiple of hop_length
            ├── LTXMelSTFT  -> log-mel
            ├── bwe_generator ──> residual (B, 2, T_out)
            └── resampler (Hann-window sinc, ratio=2) ──> skip (B, 2, T_out)
        ──> clamp(residual + skip, -1, 1)[..., :output_length]

fp32 throughout: every conv is constructed with ``dtype=ttnn.float32`` (HiFi4 +
``fp32_dest_acc``).
"""

from __future__ import annotations

import math

import torch

import ttnn

from ...layers.audio_ops import depthwise_tap_filter
from ...layers.module import Module, Parameter
from .vocoder_ltx import LTXVocoder, _replicate_pad_t, _zero_pad_t, _zero_stuff_t


class LTX_STFTFn(Module):
    """Causal STFT expressed as a host-side ``unfold`` + on-device matmul.

    We avoid ``Conv1dViaConv3d`` here: the conv3d kernel forces ``C_in_block=32``
    in fp32, so a 512-tap kernel with C_in=1 blows the static CB allocation past
    L1. Instead we unfold the (causally left-padded) waveform into
    ``(B, T_frames, win_length)`` and matmul against the basis reshaped to
    ``(win_length, n_freqs*2)`` — fp32 end-to-end, same fidelity.

    Input is ``(B, T, 1)`` ROW_MAJOR; output magnitude/phase are
    ``(B, T_frames, n_freqs)`` ROW_MAJOR. ``forward_basis`` is a Parameter loaded
    from the checkpoint; ``inverse_basis`` (iSTFT path) is dropped.
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
            state["forward_basis"] = w.squeeze(1).t().contiguous().unsqueeze(0).float()
        state.pop("inverse_basis", None)

    def forward(self, y_BTC: ttnn.Tensor) -> tuple[ttnn.Tensor, ttnn.Tensor]:
        """``y_BTC``: ``(B, T, 1)`` ROW_MAJOR waveform → ``(magnitude, phase)``,
        each ``(B, T_frames, n_freqs)`` ROW_MAJOR.
        """
        assert y_BTC.layout == ttnn.ROW_MAJOR_LAYOUT, f"expected ROW_MAJOR, got {y_BTC.layout}"
        assert y_BTC.shape[2] == 1, f"STFT input must have C=1, got {y_BTC.shape[2]}"

        # Unfold on host (no device unfold), re-upload as a windowed view.
        y_host = ttnn.to_torch(ttnn.get_device_tensors(y_BTC)[0])
        y_host = y_host.squeeze(-1).float().contiguous()
        # Causal left-pad by win_length - hop_length.
        y_padded = torch.nn.functional.pad(y_host, (self.left_pad, 0))
        y_windowed = y_padded.unfold(dimension=-1, size=self.win_length, step=self.hop_length)
        y_windowed = y_windowed.contiguous().float()
        B, T_frames, win_length = y_windowed.shape
        assert win_length == self.win_length

        y_tile = ttnn.from_torch(
            y_windowed,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=self.dtype,
        )

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

        # Split real [0:n_freqs] / imag [n_freqs:2*n_freqs]. ttnn.slice prefers
        # ROW_MAJOR for non-tile-aligned slices.
        spec = ttnn.to_layout(spec_tile, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(spec_tile)
        real = ttnn.slice(spec, [0, 0, 0], [B, T_frames, self.n_freqs])
        imag = ttnn.slice(spec, [0, 0, self.n_freqs], [B, T_frames, self.n_freqs * 2])
        ttnn.deallocate(spec)

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


class LTXMelSTFT(Module):
    """Causal log-mel spectrogram: ``log(clamp(mel_basis @ |STFT(y)|, min=1e-5))``.

    Input ``(B, T, 1)`` ROW_MAJOR → log-mel ``(B, T_frames, n_mels)`` ROW_MAJOR.
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

        # Stored transposed to (n_freqs, n_mels) so ``matmul(magnitude_BTF, .)``
        # gives the channel-last form of ``mel_basis @ magnitude``.
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
            state["mel_basis"] = mb.t().contiguous().unsqueeze(0)

    def forward(self, y_BT: ttnn.Tensor) -> ttnn.Tensor:
        """``y_BT``: ``(B, T, 1)`` ROW_MAJOR → log-mel ``(B, T_frames, n_mels)`` ROW_MAJOR."""
        magnitude, phase = self.stft_fn(y_BT)
        ttnn.deallocate(phase)

        mag_tile = ttnn.to_layout(magnitude, ttnn.TILE_LAYOUT)
        ttnn.deallocate(magnitude)
        mel_basis_tile = self.mel_basis.data
        mel = ttnn.matmul(mag_tile, mel_basis_tile)
        ttnn.deallocate(mag_tile)

        clamped = ttnn.clamp(mel, 1e-5, float("inf"))
        ttnn.deallocate(mel)
        log_mel = ttnn.log(clamped)
        ttnn.deallocate(clamped)

        return ttnn.to_layout(log_mel, ttnn.ROW_MAJOR_LAYOUT)


def _make_hann_sinc_kernel_1d(*, ratio: int) -> tuple[torch.Tensor, int, int, int, int]:
    """Return ``(kernel, kernel_size, pad, pad_left, pad_right)`` for the
    Hann-window sinc resampler variant.
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
    """Hann-window sinc upsampler (zero-stuff + zero-pad + depthwise conv).

    The filter is non-persistent (not in any checkpoint) — always rebuilt on host
    at construction time.
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
        self._conv1d_cache: dict = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # The filter is non-persistent, but sync taps if a checkpoint provides one
        # and pop the key so the loader doesn't flag it missing.
        if "filter" in state:
            t = state.pop("filter")
            assert tuple(t.shape) == (1, 1, self.kernel_size)
            self._taps_cpu = t.reshape(self.kernel_size).float().tolist()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        B, T, C = x_BTC.shape

        x_pad = _replicate_pad_t(x_BTC, self.pad, self.pad, self.mesh_device)
        x_zs = _zero_stuff_t(x_pad, stride=self.stride, mesh_device=self.mesh_device)
        ttnn.deallocate(x_pad)
        x_padded = _zero_pad_t(x_zs, self.kernel_size - 1, self.kernel_size - 1, self.mesh_device)
        ttnn.deallocate(x_zs)

        # Fold the ratio scale into the kernel taps.
        y = depthwise_tap_filter(
            x_padded,
            [t * self.ratio for t in self._taps_cpu],
            1,
            mesh_device=self.mesh_device,
            dtype=self.dtype,
            cache=self._conv1d_cache,
        )
        ttnn.deallocate(x_padded)

        T_y = y.shape[1]
        y_cropped = ttnn.slice(y, [0, self.pad_left_crop, 0], [B, T_y - self.pad_right_crop, C])
        ttnn.deallocate(y)
        return y_cropped


class LTXVocoderWithBWE(Module):
    """Vocoder + bandwidth extension. fp32 throughout."""

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

        ratio = output_sampling_rate // input_sampling_rate
        assert (
            ratio * input_sampling_rate == output_sampling_rate
        ), "output_sampling_rate must be an integer multiple of input_sampling_rate"
        self.resampler = LTXHannUpSample1d(ratio=ratio, mesh_device=mesh_device, dtype=dtype)

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
        flat = x_BCT_torch.reshape(B * C, T).float().contiguous()
        flat_BTC = flat.unsqueeze(-1)
        y_dev = ttnn.from_torch(
            flat_BTC,
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=self.dtype,
        )
        log_mel_dev = self.mel_stft(y_dev)
        ttnn.deallocate(y_dev)
        log_mel_host = ttnn.to_torch(ttnn.get_device_tensors(log_mel_dev)[0])
        ttnn.deallocate(log_mel_dev)
        T_frames = log_mel_host.shape[1]
        n_mels = log_mel_host.shape[2]
        log_mel_host = log_mel_host.reshape(B, C, T_frames, n_mels).transpose(2, 3).contiguous()
        return log_mel_host

    def _resample_device(self, x_BCT_torch: torch.Tensor) -> torch.Tensor:
        """Run the Hann-window resampler on a torch ``(B, C, T)`` waveform.

        Returns the resampled waveform as a torch ``(B, C, T*ratio)``.
        """
        B, C, T = x_BCT_torch.shape
        x_BTC = x_BCT_torch.transpose(1, 2).float().contiguous()
        x_dev = ttnn.from_torch(x_BTC, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)
        y_dev = self.resampler(x_dev)
        ttnn.deallocate(x_dev)
        y_host = ttnn.to_torch(ttnn.get_device_tensors(y_dev)[0])
        ttnn.deallocate(y_dev)
        return y_host.transpose(1, 2).contiguous()

    def _bwe_from_waveform(self, x: torch.Tensor, *, input_dtype: torch.dtype, output_length: int) -> torch.Tensor:
        """Run the BWE half of the pipeline from a precomputed low-rate waveform
        ``x`` ``(B, C, T_low)``. Returns ``(B, C, T_out)`` clamped to [-1, 1],
        trimmed to ``output_length``.
        """
        assert x.dim() == 3, f"x must be (B, C, T), got {tuple(x.shape)}"
        B, C, length_low_rate = x.shape

        remainder = length_low_rate % self.hop_length
        if remainder != 0:
            pad_right = self.hop_length - remainder
            x = torch.nn.functional.pad(x, (0, pad_right))

        mel = self._compute_mel_device(x)

        # bwe_generator expects (B, C, T_frames, n_mels).
        mel_for_bwe = mel.transpose(2, 3).contiguous()
        residual = self.bwe_generator(mel_for_bwe)

        skip = self._resample_device(x)
        assert residual.shape == skip.shape, f"residual {tuple(residual.shape)} != skip {tuple(skip.shape)}"

        out = torch.clamp(residual + skip, -1.0, 1.0)
        out = out[..., :output_length]
        return out.to(input_dtype)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """``mel_spec``: ``(B, 2, T, mel_bins)`` stereo → ``(B, 2, T_out)`` waveform
        clamped to ``[-1, 1]``, same dtype as input.
        """
        input_dtype = mel_spec.dtype

        x = self.vocoder(mel_spec.float())
        B, C, length_low_rate = x.shape
        output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate

        return self._bwe_from_waveform(x, input_dtype=input_dtype, output_length=output_length)

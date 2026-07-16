# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 vocoder bandwidth-extension wrapper (Stage C): main Vocoder → BWE generator
residual + sinc-resampled skip, clamped to [-1, 1]. fp32 throughout (every conv is
``dtype=ttnn.float32``, HiFi4 + ``fp32_dest_acc``).
"""

from __future__ import annotations

import torch

import ttnn

from ...layers.audio_resample import UpSample1d
from ...layers.module import Module, Parameter
from .vocoder_ltx import Vocoder


class _STFTFn(Module):
    """Causal STFT expressed as a host-side ``unfold`` + on-device matmul.

    We avoid ``Conv1dViaConv3d`` here: the conv3d kernel forces ``C_in_block=32``
    in fp32, so a 512-tap kernel with C_in=1 blows the static CB allocation past
    L1. Instead we unfold the (causally left-padded) waveform into
    ``(B, T_frames, win_length)`` and matmul against the basis reshaped to
    ``(win_length, n_freqs*2)`` — fp32 end-to-end, same fidelity. The cost is a
    device→host→device round-trip per call (no device-side unfold op exists);
    permanent fix is a device unfold to keep the waveform resident.

    Input is ``(B, T, 1)`` ROW_MAJOR; output magnitude is
    ``(B, T_frames, n_freqs)`` ROW_MAJOR. ``forward_basis`` is a Parameter loaded
    from the checkpoint; ``inverse_basis`` (iSTFT path) and phase are dropped.
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

    def forward(self, y_BTC: ttnn.Tensor) -> ttnn.Tensor:
        """``y_BTC``: ``(B, T, 1)`` ROW_MAJOR waveform → ``magnitude``,
        ``(B, T_frames, n_freqs)`` ROW_MAJOR.
        """
        assert y_BTC.layout == ttnn.ROW_MAJOR_LAYOUT, f"expected ROW_MAJOR, got {y_BTC.layout}"
        assert y_BTC.shape[2] == 1, f"STFT input must have C=1, got {y_BTC.shape[2]}"

        y_host = ttnn.to_torch(ttnn.get_device_tensors(y_BTC)[0])
        y_host = y_host.squeeze(-1).float().contiguous()
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

        # ttnn.slice prefers ROW_MAJOR for non-tile-aligned slices.
        spec = ttnn.to_layout(spec_tile, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(spec_tile)
        real = ttnn.slice(spec, [0, 0, 0], [B, T_frames, self.n_freqs])
        imag = ttnn.slice(spec, [0, 0, self.n_freqs], [B, T_frames, self.n_freqs * 2])
        ttnn.deallocate(spec)

        real_sq = ttnn.multiply(real, real)
        imag_sq = ttnn.multiply(imag, imag)
        ttnn.deallocate(real)
        ttnn.deallocate(imag)
        mag_sq = ttnn.add(real_sq, imag_sq)
        ttnn.deallocate(real_sq)
        ttnn.deallocate(imag_sq)
        magnitude = ttnn.sqrt(mag_sq)
        ttnn.deallocate(mag_sq)

        return magnitude


class MelSTFT(Module):
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

        self.stft_fn = _STFTFn(
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
        magnitude = self.stft_fn(y_BT)

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


class VocoderWithBWE(Module):
    """Vocoder + bandwidth extension. fp32 throughout."""

    def __init__(
        self,
        *,
        vocoder: Vocoder,
        bwe_generator: Vocoder,
        mel_stft: MelSTFT,
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
        self.resampler = UpSample1d(ratio=ratio, window="hann", mesh_device=mesh_device, dtype=dtype)

        # When set, each generator runs via capture-once/replay (forward_traced), removing
        # per-op host dispatch (~5x on its device graph). use_trace_bwe is separate so the
        # BWE generator can be trace-gated independently of the main vocoder and validated
        # against eager.
        self.use_trace = False
        self.use_trace_bwe = False

    def release_trace(self) -> None:
        """Free both generators' captured traces; safe to call when none is active."""
        self.vocoder.release_trace()
        self.bwe_generator.release_trace()

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
        residual = (
            self.bwe_generator.forward_traced(mel_for_bwe) if self.use_trace_bwe else self.bwe_generator(mel_for_bwe)
        )

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

        x = self.vocoder.forward_traced(mel_spec.float()) if self.use_trace else self.vocoder(mel_spec.float())
        B, C, length_low_rate = x.shape
        output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate

        return self._bwe_from_waveform(x, input_dtype=input_dtype, output_length=output_length)

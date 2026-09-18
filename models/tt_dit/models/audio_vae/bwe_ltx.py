# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 vocoder bandwidth-extension wrapper (Stage C): main Vocoder → BWE generator
residual + sinc-resampled skip, clamped to [-1, 1]. fp32 throughout (every conv is
``dtype=ttnn.float32``, HiFi4 + ``fp32_dest_acc``).
"""

from __future__ import annotations

import math
import os

import torch

import ttnn

from ...layers.audio_resample import UpSample1d
from ...layers.module import Module, Parameter
from ...utils.tracing import traced_function
from .vocoder_ltx import Vocoder


class _STFTFn(Module):
    """Causal windowing followed by an on-device STFT basis matmul.

    We avoid ``Conv1dViaConv3d`` here: the conv3d kernel forces ``C_in_block=32``
    in fp32, so a 512-tap kernel with C_in=1 blows the static CB allocation past
    L1. Instead we unfold the (causally left-padded) waveform into
    ``(B, T_frames, win_length)`` and matmul against the basis reshaped to
    ``(win_length, n_freqs*2)`` — fp32 end-to-end, same fidelity. By default the
    windows use host ``unfold``. ``LTX_STFT_DEVICE_FRAMING=1`` selects a device
    gather using immutable indices uploaded during warmup, before trace capture.

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
        self.device_framing = os.environ.get("LTX_STFT_DEVICE_FRAMING", "0") == "1"
        self._window_indices: dict[tuple[int, int], ttnn.Tensor] = {}
        self._gather_grid = None

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

    def prepare_device_windows(self, batch: int, length: int) -> None:
        """Upload immutable gather indices before capture; no waveform values visit the host."""
        # Native RM gather double-buffers an entire waveform row on every core.
        # Bound this first experiment to the production clip and its uncropped
        # extent; a streaming window reader is needed before extending duration.
        if (self.win_length, self.hop_length, self.left_pad) != (512, 80, 432) or not 80 <= length <= 96640:
            raise ValueError("device STFT framing requires win512/hop80 and 80..96640 waveform samples")
        if self._gather_grid is None:
            grid = self.mesh_device.compute_with_storage_grid_size()
            if grid.x * grid.y < 64:
                raise ValueError("device STFT framing requires at least 64 workers")
            # The native RM factory uses unrounded CB page sizes. A 512-wide
            # window split across 64 workers gives 32B-aligned index/output
            # slices; splitting across all 120 BH workers need not do so.
            self._gather_grid = ttnn.num_cores_to_corerangeset(64, grid, row_wise=True)
        key = (batch, length)
        if key in self._window_indices:
            return
        frames = (length + self.left_pad - self.win_length) // self.hop_length + 1
        if frames <= 0:
            raise ValueError("STFT input is shorter than one causal window")
        indices = torch.arange(frames, dtype=torch.int64)[:, None] * self.hop_length
        indices = indices + torch.arange(self.win_length, dtype=torch.int64)[None, :]
        indices = indices.reshape(1, 1, 1, -1).expand(1, 1, batch, -1).contiguous()
        self._window_indices[key] = ttnn.from_torch(
            indices.to(torch.int32),
            device=self.mesh_device,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.uint32,
        )

    def _frame_device(self, y_BTC: ttnn.Tensor) -> ttnn.Tensor:
        """Exact copy-only equivalent of causal pad/unfold, returning ROW_MAJOR windows."""
        batch, length, channels = tuple(y_BTC.shape)
        assert channels == 1 and y_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        assert y_BTC.dtype == ttnn.float32, "device framing currently supports FP32 waveforms"
        indices = self._window_indices[(batch, length)]
        flat = ttnn.reshape(y_BTC, (1, 1, batch, length))
        # Align the input CB page as well. Added trailing zeros are never indexed.
        tail = (-(length + self.left_pad)) % 32
        padded = ttnn.pad(flat, [(0, 0), (0, 0), (0, 0), (self.left_pad, tail)], 0.0)
        gathered = ttnn.gather(padded, 3, indices, sub_core_grids=self._gather_grid)
        frames = indices.shape[-1] // self.win_length
        return ttnn.reshape(gathered, (batch, frames, self.win_length))

    def forward(self, y_BTC: ttnn.Tensor) -> ttnn.Tensor:
        """``y_BTC``: ``(B, T, 1)`` ROW_MAJOR waveform → ``magnitude``,
        ``(B, T_frames, n_freqs)`` ROW_MAJOR.
        """
        assert y_BTC.layout == ttnn.ROW_MAJOR_LAYOUT, f"expected ROW_MAJOR, got {y_BTC.layout}"
        assert y_BTC.shape[2] == 1, f"STFT input must have C=1, got {y_BTC.shape[2]}"

        if self.device_framing:
            self.prepare_device_windows(y_BTC.shape[0], y_BTC.shape[1])
            windows = self._frame_device(y_BTC)
            B, T_frames, _ = tuple(windows.shape)
            y_tile = ttnn.to_layout(windows, ttnn.TILE_LAYOUT)
        else:
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
        self.device_chain = os.environ.get("LTX_AUDIO_DEVICE_CHAIN", "0") == "1"
        if self.device_chain:
            assert self.dtype == ttnn.float32, "device audio chain requires FP32"
            self.mel_stft.stft_fn.device_framing = True

    def release_trace(self) -> None:
        """Free both generators' captured traces; safe to call when none is active."""
        self.vocoder.release_trace()
        self.bwe_generator.release_trace()
        for tracer in type(self)._forward_device_chain._tracers_keyed.get(self, {}).values():
            tracer.release_trace()
        self.mel_stft.stft_fn._window_indices.clear()

    @traced_function(device=lambda self: self.mesh_device, prep_run=False, clone_prep_inputs=False)
    def _forward_device_chain(self, mel_BTC: ttnn.Tensor, length_low: int, output_length: int) -> ttnn.Tensor:
        """One trace owns all temporaries, so no output crosses an older independent trace.

        The ordinary eager pipeline warmup must materialize lazy conv/CCL state first.
        Both generators are deliberately called eagerly *inside* this combined capture.
        """
        batch = mel_BTC.shape[0]
        channels = self.vocoder.out_channels
        wave = self.vocoder._forward_device(mel_BTC)
        wave = ttnn.slice(wave, (0, 0, 0), (batch, length_low, channels))
        pad_right = (-length_low) % self.hop_length
        if pad_right:
            wave = ttnn.pad(wave, [(0, 0), (0, pad_right), (0, 0)], 0.0)
        padded_length = length_low + pad_right

        # Preserve the host path's B,C,T ordering before flattening stereo into batch.
        mono = ttnn.permute(wave, (0, 2, 1))
        mono = ttnn.reshape(mono, (batch * channels, padded_length, 1))
        mel = self.mel_stft(mono)
        frames, bins = mel.shape[1], mel.shape[2]
        mel = ttnn.reshape(mel, (batch, channels, frames, bins))
        mel = ttnn.permute(mel, (0, 2, 1, 3))
        mel = ttnn.reshape(mel, (batch, frames, channels * bins))
        if self.bwe_generator._t_pad:
            mel = ttnn.pad(mel, [(0, 0), (0, self.bwe_generator._t_pad), (0, 0)], 0.0)
        residual = self.bwe_generator._forward_device(mel)
        full_length = padded_length * self.output_sampling_rate // self.input_sampling_rate
        residual = ttnn.slice(residual, (0, 0, 0), (batch, full_length, channels))
        skip = self.resampler(wave)
        assert tuple(residual.shape) == tuple(skip.shape)
        mixed = ttnn.add(residual, skip)
        clipped = ttnn.clamp(mixed, -1.0, 1.0)
        return ttnn.slice(clipped, (0, 0, 0), (batch, output_length, channels))

    def _device_chain_from_mel(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """One upload and one final download; cache construction is outside capture."""
        assert mel_spec.ndim == 4 and mel_spec.shape[1] == 2, "device chain currently requires stereo mel"
        traces = type(self)._forward_device_chain._tracers_keyed.get(self, {})
        shape_key = tuple(mel_spec.shape)
        requested_trace = traces.get(shape_key)
        if (requested_trace is None or not requested_trace.trace_captured) and any(
            t.trace_captured for t in traces.values()
        ):
            raise RuntimeError("release audio traces and eagerly warm the new shape before capturing it")
        length_low = mel_spec.shape[2] * math.prod(self.vocoder.upsample_rates)
        padded_length = length_low + (-length_low) % self.hop_length
        self.mel_stft.stft_fn.prepare_device_windows(mel_spec.shape[0] * 2, padded_length)
        frames = padded_length // self.hop_length
        pc = self.bwe_generator.parallel_config
        factor = pc.factor if pc is not None else 1
        self.bwe_generator._t_pad = max(math.ceil(frames / factor), 32) * factor - frames if factor > 1 else 0
        output_length = length_low * self.output_sampling_rate // self.input_sampling_rate
        mel_dev = self.vocoder._host_to_device(mel_spec.float())
        output = self._forward_device_chain(
            mel_dev,
            length_low,
            output_length,
            traced=self.use_trace or self.use_trace_bwe,
            tracer_trace_key=shape_key,
        )
        host = ttnn.to_torch(ttnn.get_device_tensors(output)[0])
        return host.transpose(1, 2).contiguous().to(mel_spec.dtype)

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

        if self.device_chain:
            return self._device_chain_from_mel(mel_spec)

        x = self.vocoder.forward_traced(mel_spec.float()) if self.use_trace else self.vocoder(mel_spec.float())
        B, C, length_low_rate = x.shape
        output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate

        return self._bwe_from_waveform(x, input_dtype=input_dtype, output_length=output_length)

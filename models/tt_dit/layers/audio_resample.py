# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Anti-aliased 1D resampling layers for LTX-2 audio (BigVGAN-v2 style).

Device equivalents of the reference's ``LowPassFilter1d`` / ``UpSample1d`` /
``DownSample1d`` / ``Activation1d``. The transposed-conv upsample is expressed as
zero-stuff + zero-pad + a depthwise tap filter (bit-equivalent to
``F.conv_transpose1d`` for the symmetric sinc kernel). ``UpSample1d`` is shared
by the vocoder (kaiser window, T-shard/halo aware) and the BWE resampler (hann
window, single-device).
"""

from __future__ import annotations

import math

import torch

import ttnn

from ..parallel.config import ParallelFactor
from ..parallel.manager import CCLManager
from .audio_ops import (
    _make_kaiser_sinc_kernel_1d,
    _replicate_pad_t,
    _t_neighbor_pad,
    _zero_pad_t,
    _zero_stuff_t,
    depthwise_tap_filter,
)
from .module import Module


def _make_hann_sinc_kernel_1d(*, ratio: int) -> tuple[torch.Tensor, int, int, int, int]:
    """Return ``(kernel, kernel_size, pad, pad_left_crop, pad_right_crop)`` for the
    Hann-window sinc resampler (torchaudio-equivalent), used by the BWE skip path."""
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


class LowPassFilter1d(Module):
    """Depthwise low-pass conv1d with a fixed kaiser-sinc kernel.

    The kernel is constant (baked at __init__), so ``_prepare_torch_state`` only
    absorbs a checkpoint-provided kernel if present (BigVGAN convention).
    """

    def __init__(
        self,
        *,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        kernel_size: int = 12,
        padding: bool = True,
        padding_mode: str = "replicate",
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        if cutoff < 0.0 or cutoff > 0.5:
            raise ValueError("cutoff must be in [0, 0.5]")
        if padding_mode not in ("replicate", "zeros"):
            raise ValueError(f"padding_mode must be replicate or zeros, got {padding_mode!r}")
        sharded = parallel_config is not None and parallel_config.factor > 1
        if sharded:
            assert ccl_manager is not None, "T-sharding requires ccl_manager"
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        kernel = _make_kaiser_sinc_kernel_1d(cutoff, half_width, kernel_size)
        self._taps_cpu = kernel.tolist()
        self._conv1d_cache: dict = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "filter" in state:
            t = state.pop("filter")
            assert tuple(t.shape) == (1, 1, self.kernel_size)
            self._taps_cpu = t.reshape(self.kernel_size).float().tolist()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        """``x_BTC``: ``(B, T, C)`` ROW_MAJOR. Returns ``(B, T_out, C)``.

        When ``parallel_config.factor > 1``, ``T`` is the *per-device* extent and
        the replicate/zero pad becomes a halo exchange via ``_t_neighbor_pad``.
        """
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        sharded = self.parallel_config is not None and self.parallel_config.factor > 1

        if self.padding:
            if sharded:
                x = _t_neighbor_pad(
                    x_BTC,
                    pad_left=self.pad_left,
                    pad_right=self.pad_right,
                    parallel_config=self.parallel_config,
                    ccl_manager=self.ccl_manager,
                    padding_mode=self.padding_mode,
                )
            elif self.padding_mode == "replicate":
                x = _replicate_pad_t(x_BTC, self.pad_left, self.pad_right, self.mesh_device)
            else:
                x = _zero_pad_t(x_BTC, self.pad_left, self.pad_right, self.mesh_device)
        else:
            x = x_BTC

        return depthwise_tap_filter(
            x, self._taps_cpu, self.stride, mesh_device=self.mesh_device, dtype=self.dtype, cache=self._conv1d_cache
        )


class UpSample1d(Module):
    """Anti-aliased sinc upsampler (zero-stuff + depthwise lowpass).

    ``window="kaiser"`` is the BigVGAN anti-alias upsampler (T-shard/halo aware);
    ``window="hann"`` is the torchaudio-equivalent resampler for the BWE skip path
    (single-device). The depthwise Conv1d formulation is bit-equivalent to
    ``F.conv_transpose1d`` for the symmetric sinc kernel.
    """

    def __init__(
        self,
        *,
        ratio: int = 2,
        window: str = "kaiser",
        kernel_size: int | None = None,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        sharded = parallel_config is not None and parallel_config.factor > 1
        if sharded:
            assert ccl_manager is not None, "T-sharding requires ccl_manager"
        self.ratio = ratio
        self.stride = ratio
        self.window = window
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        if window == "hann":
            kernel, self.kernel_size, self.pad, self.pad_left_crop, self.pad_right_crop = _make_hann_sinc_kernel_1d(
                ratio=ratio
            )
            self._taps_cpu = kernel.tolist()
        elif window == "kaiser":
            self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            self.pad = self.kernel_size // ratio - 1
            self.pad_left_crop = self.pad * self.stride + (self.kernel_size - self.stride) // 2
            self.pad_right_crop = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
            kernel = _make_kaiser_sinc_kernel_1d(
                cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size
            )
            self._taps_cpu = kernel.tolist()
        else:
            raise ValueError(f"window must be kaiser or hann, got {window!r}")
        self._conv1d_cache: dict = {}
        self._use_polyphase = (self.kernel_size % ratio) == 0
        if self._use_polyphase:
            self._poly_K_sub = self.kernel_size // ratio

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "filter" in state:
            t = state.pop("filter")
            assert tuple(t.shape) == (1, 1, self.kernel_size)
            self._taps_cpu = t.reshape(self.kernel_size).float().tolist()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        B, T, C = x_BTC.shape
        sharded = self.parallel_config is not None and self.parallel_config.factor > 1

        poly2 = self._use_polyphase and self.ratio == 2
        # ratio-2 polyphase reads only x_pad[2:T_pad-2]; pad two fewer rows per side when possible.
        crop = 2 if (poly2 and self.pad >= 2) else 0
        eff_pad = self.pad - crop

        if sharded and eff_pad > 0:
            x_pad = _t_neighbor_pad(
                x_BTC,
                pad_left=eff_pad,
                pad_right=eff_pad,
                parallel_config=self.parallel_config,
                ccl_manager=self.ccl_manager,
                padding_mode="replicate",
            )
        else:
            x_pad = _replicate_pad_t(x_BTC, eff_pad, eff_pad, self.mesh_device)

        if poly2:
            B_, T_pad, C_ = x_pad.shape
            # Zero-pad sub-taps (sub0 trailing, sub1 leading) so both phases convolve the same input.
            if crop:
                base = x_pad
            else:
                base = ttnn.slice(x_pad, [0, 2, 0], [B_, T_pad - 2, C_])
            scaled_taps = [t * self.ratio for t in self._taps_cpu]
            sub0 = [scaled_taps[2 * j + 0] for j in range(self._poly_K_sub)] + [0.0]
            sub1 = [0.0] + [scaled_taps[2 * j + 1] for j in range(self._poly_K_sub)]
            ph0 = depthwise_tap_filter(
                base, sub0, 1, mesh_device=self.mesh_device, dtype=self.dtype, cache=self._conv1d_cache
            )
            ph1 = depthwise_tap_filter(
                base, sub1, 1, mesh_device=self.mesh_device, dtype=self.dtype, cache=self._conv1d_cache
            )
            if base is not x_pad:
                ttnn.deallocate(base)
            T_out = ph0.shape[1]
            ph0_b = ttnn.reshape(ph0, (B_, T_out, 1, C_))
            ph1_b = ttnn.reshape(ph1, (B_, T_out, 1, C_))
            stacked = ttnn.concat([ph0_b, ph1_b], dim=2)
            return ttnn.reshape(stacked, (B_, T_out * 2, C_))

        x_zs = _zero_stuff_t(x_pad, stride=self.stride, mesh_device=self.mesh_device)
        x_padded = _zero_pad_t(x_zs, self.kernel_size - 1, self.kernel_size - 1, self.mesh_device)

        y = depthwise_tap_filter(
            x_padded,
            [t * self.ratio for t in self._taps_cpu],
            1,
            mesh_device=self.mesh_device,
            dtype=self.dtype,
            cache=self._conv1d_cache,
        )

        T_y = y.shape[1]
        y_cropped = ttnn.slice(y, [0, self.pad_left_crop, 0], [B, T_y - self.pad_right_crop, C])
        ttnn.deallocate(y)
        return y_cropped


class DownSample1d(Module):
    """Strided kaiser-sinc lowpass downsampler wrapping ``LowPassFilter1d``."""

    def __init__(
        self,
        *,
        ratio: int = 2,
        kernel_size: int | None = None,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
            padding=True,
            padding_mode="replicate",
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        return self.lowpass(x_BTC)


class Activation1d(Module):
    """Anti-aliased activation: ``UpSample1d(2x) → activation → DownSample1d(2x)``."""

    def __init__(
        self,
        *,
        channels: int,
        activation: Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.act = activation
        self.upsample = UpSample1d(
            ratio=up_ratio,
            window="kaiser",
            kernel_size=up_kernel_size,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.downsample = DownSample1d(
            ratio=down_ratio,
            kernel_size=down_kernel_size,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        y = self.upsample(x_BTC)
        y = self.act(y)
        if y.layout != ttnn.ROW_MAJOR_LAYOUT:
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        y = self.downsample(y)
        return y

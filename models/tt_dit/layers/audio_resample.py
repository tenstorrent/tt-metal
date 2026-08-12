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
import os

import torch

import ttnn

from ..parallel.config import ParallelFactor
from ..parallel.manager import CCLManager
from .audio_ops import (
    SnakeBeta,
    _make_kaiser_sinc_kernel_1d,
    _replicate_pad_t,
    _t_neighbor_pad,
    _zero_pad_t,
    _zero_stuff_t,
    depthwise_tap_filter,
    depthwise_tap_filter_snake,
    fuse_snake_into_conv_enabled,
)
from .module import Module

# L1, not correctness: the parameter CB costs two tiles per channel-tile, so C=512 wants 32 tiles
# (128 KB) and conv1d's DRAM auto-slice then reports it "requires more memory than available even
# with maximum slicing" against 1395840 bytes of L1. C=256 fits and is exact (5.459e-08). The C=512
# stage is the shortest-T one, so excluding it costs the least of any stage.
SNAKE_CONV_MAX_CHANNELS = int(os.environ.get("MINIMAX_H3_AUDIO_SNAKE_CONV_MAX_C", "256"))


def fuse_band_enabled() -> bool:
    """Whether `Activation1d` runs as one fused band, from ``MINIMAX_H3_AUDIO_FUSE_BAND`` (default off).

    The band is ``up2 -> activation -> down2``. Run literally, the 2x upsampled tensor is written to
    DRAM and read back by the interleave concat, the activation's layout round trip, the downsampler's
    replicate pad and the downsampler itself. Since the activation is pointwise, none of that is
    necessary: see `Activation1d._forward_fused`. Off restores the literal form, which is what the
    fused path is checked against.

    **Exact, and not worth switching on.** rel_rmse against the literal form is 8.5e-08 at every
    production shape -- fp32 round-off, so the algebra is right -- but per band it is a wash to a loss:

        shape              unfused    fused
        s3 C64  T20701      6.99 ms   6.53 ms
        s4 C32  T41403      7.08 ms   7.36 ms
        s5 C16  T82806      9.47 ms   9.53 ms
        s6 C8   T165606    18.84 ms  21.11 ms

    End to end it measures 1.200 / 2.312 / 3.672 s at 5/10/15 s, i.e. inside the run-to-run spread of
    the unfused path -- neutral, not a win. Removing the 2x tensor roughly doubles the band's op count
    (two activations, two concats and two FIRs over half-length signals instead of one of each over the
    full length), and this stage is as sensitive to op count as to bytes. Kept off because neutral and
    more complex loses; kept at all because the decomposition is the one a real fused kernel wants, and
    it is proven correct here.

    **That verdict is now out of date, in this band's favour.** What made it a wash was the extra op
    count, and `MINIMAX_H3_AUDIO_FUSE_SNAKE_CONV` deletes the largest part of it: the two per-phase
    activations fold into the convs that produce the phases, taking their layout round trips with them.
    With both on, the decode measures **1.002 s against the 1.113 s default (1.110x)** at 207 latents,
    PSNR 68.6 dB against the default's output -- two orders inside the 49.45 dB the decode already
    carries against CPU. Turn the two on together; on its own this one is still a wash.
    """
    return os.environ.get("MINIMAX_H3_AUDIO_FUSE_BAND", "0") == "1"


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

    def _can_fuse(self) -> bool:
        """Whether the band can run without ever materialising the 2x upsampled tensor.

        Requires the ratio-2 polyphase upsampler, a ratio-2 even-kernel downsampler with the standard
        ``(K//2 - 1, K//2)`` replicate pad, and no T-sharding (the halo exchange owns the padding
        there, and duplicating that invariant is how it gets wrong).
        """
        up, low = self.upsample, self.downsample.lowpass
        return (
            fuse_band_enabled()
            and up._use_polyphase
            and up.ratio == 2
            and up.pad >= 2
            and low.stride == 2
            and low.padding
            and low.padding_mode == "replicate"
            and low.kernel_size % 2 == 0
            and low.pad_left == low.kernel_size // 2 - 1
            and low.pad_right == low.kernel_size // 2
            and (up.parallel_config is None or up.parallel_config.factor <= 1)
            and (low.parallel_config is None or low.parallel_config.factor <= 1)
        )

    def _snake_conv_params(self):
        """``(alpha, inv_beta)`` as CPU vectors if the snake can ride the phase conv, else None.

        Only SnakeBeta qualifies: the kernel computes `v + inv_beta * sin(alpha*v)^2` literally, and
        plain `Snake` is a different expression. `beta` already carries the `+eps` that
        `SnakeBeta._prepare_torch_state` folded in, so the reciprocal here needs no epsilon of its own.
        """
        if not fuse_snake_into_conv_enabled() or not isinstance(self.act, SnakeBeta):
            return None
        # Declines under T-sharding because `_forward_fused` builds its boundary padding from the
        # *local* tensor's first/last rows (`_replicate_pad_t` plus the slice/concat pad split below),
        # and on a T-shard those are not the global first/last rows. Lifting this gate needs that
        # padding moved onto `_t_neighbor_pad`; it is not just a flag.
        if self.act.parallel_config is not None and getattr(self.act.parallel_config, "factor", 1) > 1:
            return None
        # Wider than one tile of channel used to be silently wrong: the reader fetched only column 0
        # and the compute kernel read CB tiles 0 and 1 whatever output column it was on, so C=64
        # reapplied channels 0-31's parameters to channels 32-63 (rel_rmse 2.6e-01 against the unfused
        # band, versus exact at C<=32). The CB now carries every column and compute indexes by its own
        # output column, so width is no longer a correctness limit -- C=64/128/256 all measure exact.
        # `SNAKE_CONV_MAX_CHANNELS` is now purely the L1 backstop; see its definition.
        if self.channels > SNAKE_CONV_MAX_CHANNELS:
            return None
        cached = getattr(self, "_snake_conv_ab", None)
        if cached is None:
            # These parameters are replicated across the mesh, so on any multi-device mesh a bare
            # `ttnn.to_torch` hits `buffers.size() == 1` (pytensor.cpp:299) and the whole decode dies
            # -- fusion plus a mesh was an unreachable combination before this. Read one shard, the
            # same shape as `Vocoder._device_to_host` and `_project_latents_device`. Note the gate
            # above means only the unsharded-on-a-mesh case gets here today, which is exactly the
            # t_factor=1 baseline of `test_audio_decode_t_parallel`.
            def _param(t: ttnn.Tensor) -> torch.Tensor:
                shards = ttnn.get_device_tensors(t)
                return ttnn.to_torch(shards[0] if len(shards) > 1 else t).float().reshape(-1)

            cached = (_param(self.act.alpha.data), 1.0 / _param(self.act.beta.data))
            self._snake_conv_ab = cached
        return cached

    def _forward_fused(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        """``up2 -> act -> down2`` computed on half-length phase signals.

        The upsampler's output is ``z[2m] = ph0[m]``, ``z[2m+1] = ph1[m]``, and the activation is
        pointwise, so ``act(z)`` interleaves ``act(ph0)`` and ``act(ph1)``. Splitting the downsample
        sum by the parity of the tap index then gives

            y[t] = sum_a tap[2a] * P0[t+a] + sum_a tap[2a+1] * P1[t+a]

        with ``P0[m] = s_pad[2m]``, ``P1[m] = s_pad[2m+1]`` -- two stride-1 depthwise FIRs over
        half-length signals. The interleave concat and every op that would have run at 2x length
        disappear; nothing is approximated (verified exact against the unfused form).

        The one trap: replicate padding does **not** decompose into per-phase replicate padding. The
        pad region is the constant ``s[0]`` (or ``s[-1]``) whose parity alternates, so P0's left pad is
        built from ``s0[0]`` -- the first sample of the *other* phase -- not from ``s1[0]``.
        """
        up, low = self.upsample, self.downsample.lowpass
        B, T, C = int(x_BTC.shape[0]), int(x_BTC.shape[1]), int(x_BTC.shape[2])

        # --- upsample phases, without interleaving them ---
        crop = 2
        x_pad = _replicate_pad_t(x_BTC, up.pad - crop, up.pad - crop, up.mesh_device)
        scaled = [t * up.ratio for t in up._taps_cpu]
        sub0 = [scaled[2 * j + 0] for j in range(up._poly_K_sub)] + [0.0]
        sub1 = [0.0] + [scaled[2 * j + 1] for j in range(up._poly_K_sub)]
        fir = lambda src, taps: depthwise_tap_filter(
            src, taps, 1, mesh_device=up.mesh_device, dtype=up.dtype, cache=up._conv1d_cache
        )

        # --- upsample phase + pointwise activation, in one op per phase where possible ---
        #
        # Stacking the phases along C to halve the activation count was tried and reverted: it did not
        # get faster, and it moved s6 (C=8) from 8.6e-08 to 4.8e-04, so `channel_repeat` combined with
        # the tile-fold is wrong somewhere. Two separate activations are exact.
        def activate(v):
            v = self.act(v)
            return v if v.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(v, ttnn.ROW_MAJOR_LAYOUT)

        if self._snake_conv_params() is not None:
            # The snake rides the phase conv's own output, so `activate` never runs and neither does
            # the tilize/untilize around it: two ops per band become none.
            alpha, inv_beta = self._snake_conv_params()
            sfir = lambda taps, tag: depthwise_tap_filter_snake(
                x_pad,
                taps,
                alpha=alpha,
                inv_beta=inv_beta,
                mesh_device=up.mesh_device,
                dtype=up.dtype,
                cache=up._conv1d_cache,
                cache_tag=tag,
            )
            s0, s1 = sfir(sub0, "sub0"), sfir(sub1, "sub1")
        else:
            ph0, ph1 = fir(x_pad, sub0), fir(x_pad, sub1)
            s0, s1 = activate(ph0), activate(ph1)
        ttnn.deallocate(x_pad)
        M = int(s0.shape[1])

        # --- even/odd samples of the replicate-padded interleaved signal ---
        needed = M + low.kernel_size // 2 - 1
        l0, l1 = (low.pad_left + 1) // 2, low.pad_left // 2
        r0, r1 = needed - M - l0, needed - M - l1
        assert r0 >= 0 and r1 >= 0, f"pad split negative: {l0=} {l1=} {r0=} {r1=} {needed=} {M=}"
        first = ttnn.slice(s0, [0, 0, 0], [B, 1, C])
        last = ttnn.slice(s1, [0, M - 1, 0], [B, M, C])
        p0 = ttnn.concat([first] * l0 + [s1] + [last] * r0, dim=1)
        p1 = ttnn.concat([first] * l1 + [s0] + [last] * r1, dim=1)
        ttnn.deallocate(first)
        ttnn.deallocate(last)

        half = low.kernel_size // 2
        even = [low._taps_cpu[2 * a] for a in range(half)]
        odd = [low._taps_cpu[2 * a + 1] for a in range(half)]
        dfir = lambda src, taps: depthwise_tap_filter(
            src, taps, 1, mesh_device=low.mesh_device, dtype=low.dtype, cache=low._conv1d_cache
        )
        out = ttnn.add(dfir(p0, even), dfir(p1, odd))
        ttnn.deallocate(p0)
        ttnn.deallocate(p1)
        return out

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        if self._can_fuse():
            return self._forward_fused(x_BTC)
        y = self.upsample(x_BTC)
        y = self.act(y)
        if y.layout != ttnn.ROW_MAJOR_LAYOUT:
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        y = self.downsample(y)
        return y

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""LTX-2 audio vocoder (Stage B).

Mirror of the torch reference ``Vocoder`` from
``LTX-2/packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py``.

The production LTX-2.3 22B distilled config consumes mel-spectrogram-like
features ``(B, 2, T_frames, mel_bins=64)`` and produces a waveform
``(B, 2, T_frames * 160)``. The architecture is BigVGAN-v2 with AMP1 blocks:

- ``conv_pre``: Conv1d k=7 (128 → 1536)
- 6 upsample stages with rates [5, 2, 2, 2, 2, 2] (total factor 160):
  - each: ``ConvTranspose1d`` (channel halve) → 3 parallel ``AMPBlock1``, mean
- ``act_post``: ``Activation1d(SnakeBeta(24))``
- ``conv_post``: Conv1d k=7 (24 → 2), no bias
- ``clamp(-1, 1)``

**fp32 mandatory throughout.** Per the reference comment in
``vocoder.py:553-573``, bf16 accumulation degrades spectral metrics 40-90 %
through the 108-conv chain. Every Conv1d, every Snake, every anti-alias
filter runs at ``dtype=ttnn.float32`` which routes to the HiFi4 +
``fp32_dest_acc_en`` + ``packer_l1_acc`` path in
``Conv1dViaConv3d.compute_kernel_config``.

Layout: vocoder works on ``(B, C, T)`` torch tensors. We convert to
``(B, T, C)`` ROW_MAJOR at the device boundary to match the
``Conv1dViaConv3d`` expectation.
"""

from __future__ import annotations

import math
from typing import List, Sequence

import torch

import ttnn

from ...layers.audio_ops import Conv1dViaConv3d, Snake, SnakeBeta, _t_neighbor_pad
from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import ParallelFactor
from ...parallel.manager import CCLManager
from ...utils.conv3d import aligned_channels

# ---------------------------------------------------------------------------
# Kaiser-sinc kernel construction (host-side, baked into module at __init__).
# Mirrors ``kaiser_sinc_filter1d`` in the reference vocoder.py:30.
# ---------------------------------------------------------------------------


def _make_kaiser_sinc_kernel_1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Return a shape-``(kernel_size,)`` kaiser-windowed sinc filter.

    Bit-identical with the reference ``kaiser_sinc_filter1d`` in
    ``vocoder.py:30-48`` (modulo the leading ``(1, 1, ...)`` axes).
    """
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    if even:
        time = torch.arange(-half_size, half_size) + 0.5
    else:
        time = torch.arange(kernel_size) - half_size
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = (
            2
            * cutoff
            * window
            * torch.where(
                time == 0,
                torch.tensor(1.0, dtype=time.dtype),
                torch.sin(math.pi * 2 * cutoff * time) / (math.pi * 2 * cutoff * time),
            )
        )
        filter_ = filter_ / filter_.sum()
    return filter_.float().reshape(kernel_size)


# ---------------------------------------------------------------------------
# Device-side tensor utilities for ROW_MAJOR (B, T, C) tensors.
# ---------------------------------------------------------------------------


def _replicate_pad_t(x_BTC: ttnn.Tensor, pad_left: int, pad_right: int, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Replicate-pad along the T axis: append copies of the first/last sample.

    Mirrors ``F.pad(x, (pad_left, pad_right), mode='replicate')`` for a
    ``(B, T, C)`` ROW_MAJOR tensor.
    """
    if pad_left == 0 and pad_right == 0:
        return x_BTC
    B, T, C = x_BTC.shape
    pieces = []
    if pad_left > 0:
        first = ttnn.slice(x_BTC, [0, 0, 0], [B, 1, C])
        # Replicate the first row `pad_left` times.
        pieces.extend([first] * pad_left)
    pieces.append(x_BTC)
    if pad_right > 0:
        last = ttnn.slice(x_BTC, [0, T - 1, 0], [B, T, C])
        pieces.extend([last] * pad_right)
    return ttnn.concat(pieces, dim=1)


def _zero_pad_t(x_BTC: ttnn.Tensor, pad_left: int, pad_right: int, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Zero-pad along the T axis."""
    if pad_left == 0 and pad_right == 0:
        return x_BTC
    B, T, C = x_BTC.shape
    pieces = []
    dtype = x_BTC.get_dtype()
    if pad_left > 0:
        zeros = ttnn.zeros((B, pad_left, C), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        pieces.append(zeros)
    pieces.append(x_BTC)
    if pad_right > 0:
        zeros = ttnn.zeros((B, pad_right, C), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
        pieces.append(zeros)
    return ttnn.concat(pieces, dim=1)


def _pad_channels_to_aligned(x_BTC: ttnn.Tensor, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Pad C up to ``aligned_channels(C)`` with zeros. No-op if already aligned."""
    B, T, C = x_BTC.shape
    aligned = aligned_channels(C)
    if aligned == C:
        return x_BTC
    pad_c = aligned - C
    dtype = x_BTC.get_dtype()
    zeros = ttnn.zeros((B, T, pad_c), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    return ttnn.concat([x_BTC, zeros], dim=2)


def _state_pad_out_channels(state: dict, key: str, real_out: int, aligned_out: int) -> None:
    """Zero-pad the ``out`` axis of a torch Conv1d-style weight tensor in state.

    Conv1dViaConv3d's ``out_channels = max(32, out)`` rule (without
    ``aligned_channels``) lets non-32-multiple values like ``48`` slip through
    to ``ttnn.experimental.conv3d``, which can produce buffers whose page size
    does not divide the buffer length. We pre-pad weights to
    ``aligned_channels(out)`` and force ``Conv1dViaConv3d`` to use that count.
    The padded ``out`` rows are zeros so the extra output channels are 0 — we
    discard them downstream (or rely on the next layer's ``in_channels`` mask).
    """
    if key in state:
        w = state[key]
        if w.shape[0] != real_out:
            return
        if real_out == aligned_out:
            return
        pad = aligned_out - real_out
        # Pad along axis 0 (out) with zeros.
        pad_tuple = [0, 0] * (w.ndim - 1) + [0, pad]
        state[key] = torch.nn.functional.pad(w, pad_tuple)


def _zero_stuff_t(x_BTC: ttnn.Tensor, *, stride: int, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Insert ``stride - 1`` zeros between input samples along T.

    For ``stride=s`` and input length ``T``, the output has length
    ``T*s - (s - 1)``: every original sample ``x[t]`` lands at output position
    ``t * s`` and the gaps in between are zeros. This is the canonical
    zero-stuffing used to express ``ConvTranspose1d`` as a regular ``Conv1d``.

    Implementation: stack the input with zero blocks along a new axis,
    reshape to interleave, then trim the trailing ``(stride-1)`` zeros. This
    is O(1) ttnn ops rather than O(T).
    """
    if stride == 1:
        return x_BTC
    B, T, C = x_BTC.shape
    dtype = x_BTC.get_dtype()
    # Reshape input to (B, T, 1, C), build zero block (B, T, stride-1, C),
    # concat along axis 2 to get (B, T, stride, C), then reshape to
    # (B, T*stride, C).
    x_btoc = ttnn.reshape(x_BTC, (B, T, 1, C))
    zero_block = ttnn.zeros((B, T, stride - 1, C), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    stacked = ttnn.concat([x_btoc, zero_block], dim=2)  # (B, T, stride, C)
    interleaved = ttnn.reshape(stacked, (B, T * stride, C))  # (B, T*stride, C)
    # Trim trailing (stride - 1) zeros so the output has length T*stride - (stride-1).
    out_len = T * stride - (stride - 1)
    return ttnn.slice(interleaved, [0, 0, 0], [B, out_len, C])


# ---------------------------------------------------------------------------
# Depthwise fixed-kernel filter (kaiser-sinc lowpass), implemented via the
# "shifted multiply-accumulate" pattern: for a K-tap depthwise convolution,
# the output is a weighted sum of K shifted slices of the (already-padded)
# input. Works for arbitrary in_channels because the kernel is the same
# scalar at every channel.
# ---------------------------------------------------------------------------


class LTXLowPassFilter1d(Module):
    """Depthwise low-pass conv1d with a fixed kaiser-sinc kernel.

    Mirrors the reference ``LowPassFilter1d``:

      pad_left = K // 2 - int(even);  pad_right = K // 2
      y[b, t_out, c] = sum_{j=0..K-1} kernel[j] * x_padded[b, t_out * stride + j, c]
      x_padded = F.pad(x, (pad_left, pad_right), mode='replicate')

    The kernel is constant — baked at __init__ — so there is no learned weight
    and ``_prepare_torch_state`` is a no-op (it just absorbs the kernel from
    the checkpoint if present, which is the BigVGAN convention).
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
            # stride > 1 sharded path: T_per_device must be divisible by stride so
            # the per-chip output length sums correctly across the mesh. The
            # Activation1d pattern (UpSample → activation → DownSample) preserves
            # this because UpSample doubles T and DownSample halves it.
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

        # Bake the kernel on the host. Stored as Parameter so it shows up in
        # state-dict iteration (matches the checkpoint key ``...filter`` with
        # shape (1, 1, K)). Also expose as a Python list of scalars used by
        # the multiply-accumulate path.
        kernel = _make_kaiser_sinc_kernel_1d(cutoff, half_width, kernel_size)
        self._taps_cpu = kernel.tolist()
        # Parameter shape matches the reference: (1, 1, K).
        self.filter = Parameter(total_shape=[1, 1, kernel_size], device=mesh_device, dtype=dtype)
        # Self-load so the module is usable even without state-dict load.
        self.filter.load_torch_tensor(kernel.reshape(1, 1, kernel_size))

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # If a checkpoint provides the filter, we overwrite our baked default.
        # Use the checkpoint value as the source of truth for ``self._taps_cpu``.
        if "filter" in state:
            t = state["filter"]
            assert tuple(t.shape) == (
                1,
                1,
                self.kernel_size,
            ), f"filter shape mismatch: expected (1, 1, {self.kernel_size}), got {tuple(t.shape)}"
            self._taps_cpu = t.reshape(self.kernel_size).float().tolist()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        """``x_BTC``: ``(B, T, C)`` ROW_MAJOR. Returns ``(B, T_out, C)``.

        When ``parallel_config.factor > 1``, ``T`` is the *per-device* extent
        and the replicate/zero pad becomes a halo exchange via the shared
        ``_t_neighbor_pad`` helper.
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

        B = x.shape[0]
        C = x.shape[2]
        T_padded = x.shape[1]
        # T_out = floor((T_padded - K) / stride) + 1
        T_out = (T_padded - self.kernel_size) // self.stride + 1
        assert T_out > 0, f"T_out={T_out}, T_padded={T_padded}, K={self.kernel_size}, stride={self.stride}"

        # Shifted multiply-accumulate over the K taps.
        y = None
        for j in range(self.kernel_size):
            w = float(self._taps_cpu[j])
            # slice_j = x[:, j : j + T_out*stride : stride, :]  (length T_out)
            # For stride==1 this is a contiguous slice; for stride>1 we use
            # the strided form of ttnn.slice.
            if self.stride == 1:
                slice_j = ttnn.slice(x, [0, j, 0], [B, j + T_out, C])
            else:
                slice_j = ttnn.slice(
                    x,
                    [0, j, 0],
                    [B, j + (T_out - 1) * self.stride + 1, C],
                    [1, self.stride, 1],
                )
            scaled = ttnn.multiply(slice_j, w)
            if y is None:
                y = scaled
            else:
                y_new = ttnn.add(y, scaled)
                ttnn.deallocate(y)
                ttnn.deallocate(scaled)
                y = y_new
            ttnn.deallocate(slice_j)
        return y


# ---------------------------------------------------------------------------
# UpSample1d / DownSample1d / Activation1d (anti-aliased BigVGAN v2 pattern).
# ---------------------------------------------------------------------------


class LTXUpSample1d(Module):
    """Anti-aliased ``2*ratio×`` insert-zeros + kaiser-sinc lowpass upsampler.

    Mirrors the reference ``UpSample1d``:

      x = F.pad(x, (pad, pad), mode='replicate')
      y = ratio * F.conv_transpose1d(x_pad, filter, stride=ratio, groups=C)
      y = y[..., pad_left:-pad_right]

    Implementation: replicate-pad → zero-stuff by ``ratio`` → zero-pad
    by ``K - 1`` each side → depthwise Conv1d with the kaiser-sinc kernel
    (using the shifted-multiply-accumulate pattern) → scale by ``ratio`` →
    crop ``pad_left`` from front and ``pad_right`` from back.

    The depthwise Conv1d formulation is bit-equivalent to the reference's
    ``F.conv_transpose1d`` for symmetric kernels (which kaiser-sinc is).
    """

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
        sharded = parallel_config is not None and parallel_config.factor > 1
        if sharded:
            assert ccl_manager is not None, "T-sharding requires ccl_manager"
        self.ratio = ratio
        self.stride = ratio
        self.kernel_size = int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        self.pad = self.kernel_size // ratio - 1
        self.pad_left_crop = self.pad * self.stride + (self.kernel_size - self.stride) // 2
        self.pad_right_crop = self.pad * self.stride + (self.kernel_size - self.stride + 1) // 2
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        kernel = _make_kaiser_sinc_kernel_1d(cutoff=0.5 / ratio, half_width=0.6 / ratio, kernel_size=self.kernel_size)
        self._taps_cpu = kernel.tolist()
        # Match the checkpoint key ``upsample.filter`` with shape (1, 1, K).
        self.filter = Parameter(total_shape=[1, 1, self.kernel_size], device=mesh_device, dtype=dtype)
        self.filter.load_torch_tensor(kernel.reshape(1, 1, self.kernel_size))

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "filter" in state:
            t = state["filter"]
            assert tuple(t.shape) == (1, 1, self.kernel_size)
            self._taps_cpu = t.reshape(self.kernel_size).float().tolist()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        B, T, C = x_BTC.shape
        sharded = self.parallel_config is not None and self.parallel_config.factor > 1

        # Replicate-pad along T with ``pad`` each side. When sharded, halo brings
        # ``pad`` samples from neighbors; ``padding_mode="replicate"`` makes
        # boundary chips replicate their own first/last sample (matches the
        # reference's ``F.pad(..., mode='replicate')``).
        if sharded and self.pad > 0:
            x_pad = _t_neighbor_pad(
                x_BTC,
                pad_left=self.pad,
                pad_right=self.pad,
                parallel_config=self.parallel_config,
                ccl_manager=self.ccl_manager,
                padding_mode="replicate",
            )
        else:
            x_pad = _replicate_pad_t(x_BTC, self.pad, self.pad, self.mesh_device)
        # Zero-stuff to length ratio * T_pad - (ratio - 1).
        x_zs = _zero_stuff_t(x_pad, stride=self.stride, mesh_device=self.mesh_device)
        # Pad zeros (kernel_size - 1) each side so the equivalent Conv1d
        # produces the same length as the reference's ConvTranspose1d.
        x_padded = _zero_pad_t(x_zs, self.kernel_size - 1, self.kernel_size - 1, self.mesh_device)

        # Shifted multiply-accumulate depthwise conv (stride=1, no pad).
        T_full = x_padded.shape[1]
        T_out = T_full - self.kernel_size + 1
        y = None
        for j in range(self.kernel_size):
            # scale by ratio in advance: combine with the kernel tap.
            w = float(self._taps_cpu[j]) * float(self.ratio)
            slice_j = ttnn.slice(x_padded, [0, j, 0], [B, j + T_out, C])
            scaled = ttnn.multiply(slice_j, w)
            if y is None:
                y = scaled
            else:
                y_new = ttnn.add(y, scaled)
                ttnn.deallocate(y)
                ttnn.deallocate(scaled)
                y = y_new
            ttnn.deallocate(slice_j)

        # Crop pad_left_crop from front, pad_right_crop from back.
        T_y = y.shape[1]
        y_cropped = ttnn.slice(y, [0, self.pad_left_crop, 0], [B, T_y - self.pad_right_crop, C])
        ttnn.deallocate(y)
        return y_cropped


class LTXDownSample1d(Module):
    """Strided kaiser-sinc lowpass downsampler — wraps ``LTXLowPassFilter1d``.

    Mirrors the reference ``DownSample1d`` which uses ``LowPassFilter1d``
    with ``stride=ratio``. State-dict key is ``downsample.lowpass.filter``.
    """

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
        self.lowpass = LTXLowPassFilter1d(
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


class LTXVocoderActivation1d(Module):
    """Anti-aliased activation: ``UpSample1d(2×) → activation → DownSample1d(2×)``.

    Mirrors the reference ``Activation1d`` in ``vocoder.py:145``. State-dict
    layout matches: ``upsample.filter``, ``act.alpha`` (+ ``act.beta`` for
    SnakeBeta), ``downsample.lowpass.filter``.
    """

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
        self.upsample = LTXUpSample1d(
            ratio=up_ratio,
            kernel_size=up_kernel_size,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.downsample = LTXDownSample1d(
            ratio=down_ratio,
            kernel_size=down_kernel_size,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        # Upsample produces a ROW_MAJOR tensor.
        y = self.upsample(x_BTC)
        # Snake / SnakeBeta multiplications upcast to TILE internally.
        # Pull back to ROW_MAJOR before the downsample, which expects RM.
        y = self.act(y)
        if y.layout != ttnn.ROW_MAJOR_LAYOUT:
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)
        y = self.downsample(y)
        return y


# ---------------------------------------------------------------------------
# Conv1d wrapper that aligns out_channels up to a 32-multiple (the base
# ``Conv1dViaConv3d`` uses ``max(32, out)`` which is wrong for non-aligned
# values such as 48 or 24 — ``ttnn.experimental.conv3d`` produces a buffer
# whose page size does not divide the buffer length in that case).
# ---------------------------------------------------------------------------


class _AlignedOutConv1d(Conv1dViaConv3d):
    """Conv1dViaConv3d variant that uses ``aligned_channels(out)`` internally.

    Same call signature as the base class. If the user-requested
    ``out_channels`` is not a 32-multiple, we round it up to one, pad the
    loaded weight/bias on the ``out`` axis with zeros, and trim the output
    back to the real channel count at the end of forward.

    Forward also pads input C to ``aligned_channels(in)`` before invoking
    the parent so the conv3d sees an aligned tensor on the runtime side.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding_mode: str = "zeros",
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            padding_mode=padding_mode,
            bias=bias,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        aligned_out = aligned_channels(self.unpadded_out_channels)
        if aligned_out != self.out_channels:
            self.out_channels = aligned_out
            d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
            self.weight = Parameter(
                total_shape=[d, self.out_channels],
                device=self.mesh_device,
                pad_value=0,
                dtype=self.dtype,
            )
            if self.bias is not None:
                self.bias = Parameter(
                    total_shape=[1, self.out_channels],
                    device=self.mesh_device,
                    pad_value=0,
                    dtype=self.dtype,
                )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        # Pad input C to aligned before invoking parent's forward.
        x_BTC = _pad_channels_to_aligned(x_BTC, self.mesh_device)
        y = super().forward(x_BTC)
        # Trim back to the real ``out`` channel count.
        if self.unpadded_out_channels < self.out_channels:
            B, T, C = y.shape
            y = ttnn.slice(y, [0, 0, 0], [B, T, self.unpadded_out_channels])
        return y


# ---------------------------------------------------------------------------
# ConvTranspose1d substitute.
# ---------------------------------------------------------------------------


class LTXConvTranspose1d(Module):
    """Substitute for ``torch.nn.ConvTranspose1d`` with ``padding=(k-stride)//2``.

    Mathematically equivalent to ``Conv1d`` on the zero-stuffed input with
    the weight flipped along the kernel axis and transposed:

      torch: out = ConvTranspose1d(in_ch, out_ch, k, stride, pad=(k-s)//2)(x)
      ours:  zs   = zero_stuff(x, stride)                        # length s*T - (s-1)
             pad  = zero_pad(zs, p_each_side=k-1-(k-s)//2)
             w'   = flip(W, axis=-1).transpose(0, 1)             # (out, in, k)
             out  = Conv1d(in_ch, out_ch, k, stride=1, pad=0)(pad, w')

    The padding amount ``p = k - 1 - (k-s)//2`` is the unique value that
    matches the reference output length ``(L_in - 1)*stride - 2*pad + k = L_in*stride``
    (for the common case ``pad = (k-s)//2``, which produces an exact
    ``stride×`` upsample). Verified for ``k=11, s=5`` (p=7) and ``k=4, s=2``
    (p=2) — see derivation in the doc.

    State-dict layout: ``weight`` (in_ch, out_ch, k) + optional ``bias``
    (out_ch,) — matches torch's ``ConvTranspose1d``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        # External pad on the zero-stuffed input that, combined with the
        # standard kernel, produces an exact stride× upsample.
        self.external_pad_each = kernel_size - 1 - (kernel_size - stride) // 2
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        # NOTE: when sharded, the underlying conv stays UNSHARDED because the
        # transposed-conv math (zero-stuff with stride > 1 + asymmetric local
        # zero-pad) is awkward to halo cleanly on the time axis (see
        # wiki/AUDIO_TSHARD_PLAN.md). The forward gathers the sharded input
        # across T, runs the existing unsharded transposed-conv pipeline on
        # the full sequence, then mesh-partitions the output back. There are
        # only 6 of these per vocoder (one per upsample stage) so the
        # gather/partition overhead is small compared to the AMPBlock1 chain.
        self.conv = _AlignedOutConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding_mode="causal",
            bias=bias,
            mesh_device=mesh_device,
            dtype=dtype,
            # Conv runs unsharded — see comment above.
            parallel_config=None,
            ccl_manager=None,
        )
        # Disable the front pad — we'll do our own symmetric padding.
        self.conv.external_pad_front = 0

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Reshape ConvTranspose1d weight ``(in, out, k)`` to Conv1d weight ``(out, in, k)``.

        The flip-along-k step is required because Conv1d implements
        cross-correlation while ConvTranspose1d's equivalent zero-stuff form
        requires a flipped kernel.
        """
        # Migrate keys from this module's namespace down to "conv.*" so the
        # base Conv1dViaConv3d._prepare_torch_state can pick them up.
        if "weight" in state:
            w = state.pop("weight")
            assert w.dim() == 3 and tuple(w.shape) == (self.in_channels, self.out_channels, self.kernel_size), (
                f"expected ConvTranspose1d weight shape ({self.in_channels}, {self.out_channels}, "
                f"{self.kernel_size}), got {tuple(w.shape)}"
            )
            # Flip along the kernel axis, then permute (in, out, k) -> (out, in, k).
            w_flipped = torch.flip(w, dims=[-1])
            w_conv1d = w_flipped.permute(1, 0, 2).contiguous()
            state["conv.weight"] = w_conv1d
        if "bias" in state:
            state["conv.bias"] = state.pop("bias")

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        sharded = self.parallel_config is not None and self.parallel_config.factor > 1

        if sharded:
            # Gather the sharded T-fractured input to a full sequence on every
            # chip, run the unsharded zero-stuff + zero-pad + conv pipeline,
            # then mesh-partition the output back to sharded T. See class
            # docstring for the rationale (only 6 of these per vocoder).
            x_BTC = ttnn.to_layout(x_BTC, ttnn.TILE_LAYOUT)
            x_BTC = self.ccl_manager.all_gather_persistent_buffer(
                x_BTC, dim=1, mesh_axis=self.parallel_config.mesh_axis
            )
            x_BTC = ttnn.to_layout(x_BTC, ttnn.ROW_MAJOR_LAYOUT)

        # Pad C up to aligned width if needed — Conv1dViaConv3d weight is
        # allocated for the aligned-C size, so the runtime input must match.
        x_BTC = _pad_channels_to_aligned(x_BTC, self.mesh_device)
        # Zero-stuff: length s*T - (s-1).
        x_zs = _zero_stuff_t(x_BTC, stride=self.stride, mesh_device=self.mesh_device)
        # Zero-pad ``external_pad_each`` each side.
        x_padded = _zero_pad_t(x_zs, self.external_pad_each, self.external_pad_each, self.mesh_device)
        # Conv1d with stride=1, no internal padding (configured in __init__).
        y = self.conv(x_padded)

        if sharded:
            # Repartition output along T across the mesh.
            y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
            y = ttnn.mesh_partition(y, dim=1, cluster_axis=self.parallel_config.mesh_axis)
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)

        return y


# ---------------------------------------------------------------------------
# AMPBlock1: 3 parallel branches with Activation1d → Conv1d (dilated) →
# Activation1d → Conv1d (dilation=1). Residual is summed in.
# ---------------------------------------------------------------------------


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    """Mirrors the reference ``get_padding`` (same-pad for odd kernels)."""
    return int((kernel_size * dilation - dilation) / 2)


class LTXDilatedConv1d(_AlignedOutConv1d):
    """Dilated 1D conv that directly passes ``dilation`` to ``ttnn.experimental.conv3d``.

    ``Conv1dViaConv3d.forward`` does not forward the ``dilation`` argument to
    ``ttnn.experimental.conv3d`` (the conv3d kernel itself supports it). We
    keep the parent's weight allocation (sized for the original ``kernel_size``,
    not an effectively-expanded one) and override ``forward`` to issue the
    underlying conv3d call ourselves with ``dilation=(d, 1, 1)``.

    Inherits from ``_AlignedOutConv1d`` which rounds ``out_channels`` up to
    the next 32-multiple to avoid the page-size mismatch that
    ``ttnn.experimental.conv3d`` produces for non-32-aligned ``out_channels``
    (e.g. 48, 24).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding_mode="causal",
            bias=bias,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        # Disable the parent's front pad — we add symmetric pad in forward.
        self.external_pad_front = 0
        self.true_kernel_size = kernel_size
        self.true_dilation = dilation
        # "Same" pad amount for a dilated conv with odd kernel size matches
        # the reference's ``get_padding(k, d) = (k*d - d) // 2 = (k-1)*d // 2``.
        self.same_pad = (kernel_size - 1) * dilation // 2

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        x_BTC = _pad_channels_to_aligned(x_BTC, self.mesh_device)
        sharded = self.parallel_config is not None and self.parallel_config.factor > 1
        # External symmetric zero-pad / halo for "same" output length.
        if sharded:
            x_padded = _t_neighbor_pad(
                x_BTC,
                pad_left=self.same_pad,
                pad_right=self.same_pad,
                parallel_config=self.parallel_config,
                ccl_manager=self.ccl_manager,
                padding_mode="zeros",
            )
        else:
            x_padded = _zero_pad_t(x_BTC, self.same_pad, self.same_pad, self.mesh_device)

        B, T_pad, C = x_padded.shape
        x_5d = ttnn.reshape(x_padded, (B, T_pad, 1, 1, C))
        out_5d = ttnn.experimental.conv3d(
            input_tensor=x_5d,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data if self.bias is not None else None,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=(0, 0, 0),
            dilation=(self.true_dilation, 1, 1),
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        y = ttnn.reshape(out_5d, (out_5d.shape[0], out_5d.shape[1], out_5d.shape[4]))
        # Trim back to the real ``out`` channel count.
        if self.unpadded_out_channels < self.out_channels:
            B, T, _ = y.shape
            y = ttnn.slice(y, [0, 0, 0], [B, T, self.unpadded_out_channels])
        return y


class LTXAMPBlock1(Module):
    """Three parallel residual branches with anti-aliased SnakeBeta activations.

    Mirrors the reference ``AMPBlock1`` in ``vocoder.py:211-268``:

      for c1, c2, a1, a2 in zip(convs1, convs2, acts1, acts2):
          xt = a1(x)
          xt = c1(xt)             # dilated conv
          xt = a2(xt)
          xt = c2(xt)             # dilation=1 conv
          x = x + xt
      return x

    State-dict layout matches: ``convs1.{0..2}.weight/bias``,
    ``convs2.{0..2}.weight/bias``, ``acts1.{0..2}.act.alpha/beta``,
    ``acts1.{0..2}.upsample.filter``, etc.
    """

    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        dilation: Sequence[int] = (1, 3, 5),
        activation: str = "snakebeta",
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_branches = len(dilation)
        self.mesh_device = mesh_device

        act_cls = SnakeBeta if activation == "snakebeta" else Snake

        self.convs1 = ModuleList(
            [
                LTXDilatedConv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=dilation[i],
                    bias=True,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for i in range(self.num_branches)
            ]
        )
        self.convs2 = ModuleList(
            [
                LTXDilatedConv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    dilation=1,
                    bias=True,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for i in range(self.num_branches)
            ]
        )
        # Per-branch anti-aliased activation: SnakeBeta (or Snake), wrapped
        # in Activation1d. ``alpha_logscale=True`` for AMP — the checkpoint
        # stores log α / log β and Snake/SnakeBeta collapses at load time.
        self.acts1 = ModuleList(
            [
                LTXVocoderActivation1d(
                    channels=channels,
                    activation=act_cls(channels, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype),
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for _ in range(self.num_branches)
            ]
        )
        self.acts2 = ModuleList(
            [
                LTXVocoderActivation1d(
                    channels=channels,
                    activation=act_cls(channels, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype),
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for _ in range(self.num_branches)
            ]
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        for i in range(self.num_branches):
            xt = self.acts1[i](x_BTC)
            xt = self.convs1[i](xt)
            xt = self.acts2[i](xt)
            xt = self.convs2[i](xt)
            x_new = ttnn.add(x_BTC, xt)
            ttnn.deallocate(xt)
            if i > 0:
                ttnn.deallocate(x_BTC)
            x_BTC = x_new
        return x_BTC


# ---------------------------------------------------------------------------
# Top-level LTXVocoder.
# ---------------------------------------------------------------------------


class LTXVocoder(Module):
    """BigVGAN-v2 AMP1 vocoder for LTX-2 audio decode (Stage B).

    Maps mel ``(B, 2, T_frames, mel_bins)`` to a waveform
    ``(B, 2, T_frames * prod(upsample_rates))`` via:

      conv_pre → for i in 6 stages:
          ups[i](x)                               # ConvTranspose1d substitute
          mean over 3 parallel AMPBlock1 outputs
      act_post(SnakeBeta) → conv_post → clamp(-1, 1)

    fp32 mandatory everywhere — see module-level docstring.
    """

    def __init__(
        self,
        *,
        resblock_kernel_sizes: List[int] | None = None,
        upsample_rates: List[int] | None = None,
        upsample_kernel_sizes: List[int] | None = None,
        resblock_dilation_sizes: List[List[int]] | None = None,
        upsample_initial_channel: int = 1536,
        resblock: str = "AMP1",
        activation: str = "snakebeta",
        use_tanh_at_final: bool = False,
        apply_final_activation: bool = True,
        use_bias_at_final: bool = False,
        in_channels: int = 128,
        out_channels: int = 2,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config: ParallelFactor | None = None,
        ccl_manager: CCLManager | None = None,
    ) -> None:
        super().__init__()

        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if upsample_rates is None:
            upsample_rates = [5, 2, 2, 2, 2, 2]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [11, 4, 4, 4, 4, 4]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        if resblock != "AMP1":
            raise NotImplementedError(f"only AMP1 is supported, got {resblock!r}")

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.use_tanh_at_final = use_tanh_at_final
        self.apply_final_activation = apply_final_activation
        self.use_bias_at_final = use_bias_at_final
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_rates = list(upsample_rates)
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        self.conv_pre = _AlignedOutConv1d(
            in_channels=in_channels,
            out_channels=upsample_initial_channel,
            kernel_size=7,
            stride=1,
            padding_mode="zeros",
            bias=True,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        # Upsamplers: 1 per stage, channel-halving.
        self.ups = ModuleList(
            [
                LTXConvTranspose1d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=upsample_kernel_sizes[i],
                    stride=upsample_rates[i],
                    bias=True,
                    mesh_device=mesh_device,
                    dtype=dtype,
                    parallel_config=parallel_config,
                    ccl_manager=ccl_manager,
                )
                for i in range(self.num_upsamples)
            ]
        )

        # 3 × num_upsamples AMP blocks, in row-major over (stage, branch).
        self.resblocks = ModuleList()
        for i in range(self.num_upsamples):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for ks, ds in zip(resblock_kernel_sizes, resblock_dilation_sizes, strict=True):
                self.resblocks.append(
                    LTXAMPBlock1(
                        channels=ch,
                        kernel_size=ks,
                        dilation=ds,
                        activation=activation,
                        mesh_device=mesh_device,
                        dtype=dtype,
                        parallel_config=parallel_config,
                        ccl_manager=ccl_manager,
                    )
                )

        final_channels = upsample_initial_channel // (2**self.num_upsamples)
        self.final_channels = final_channels

        self.act_post = LTXVocoderActivation1d(
            channels=final_channels,
            activation=SnakeBeta(final_channels, alpha_logscale=True, mesh_device=mesh_device, dtype=dtype),
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

        self.conv_post = _AlignedOutConv1d(
            in_channels=final_channels,
            out_channels=out_channels,
            kernel_size=7,
            stride=1,
            padding_mode="zeros",
            bias=use_bias_at_final,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # Reference Vocoder has no extra non-parameter buffers we don't model.
        # All children are registered with matching names.
        pass

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """End-to-end forward.

        Args:
            mel_spec: Torch tensor of shape ``(B, 2, T_frames, mel_bins)`` for
                stereo, or ``(B, T_frames, mel_bins)`` for mono. Mirrors the
                reference ``Vocoder.forward``.

        Returns:
            Torch tensor of shape ``(B, out_channels, T_frames * prod(rates))``.

        The whole computation runs on device — we accept torch input/output to
        match the reference's call-site convention, with a host→device
        boundary at the front and a device→host boundary at the end.
        """
        # Reference layout: (B, C, T, F) → transpose(2, 3) → (B, C, F, T).
        x_t = mel_spec.transpose(2, 3) if mel_spec.dim() == 4 else mel_spec.transpose(1, 2).unsqueeze(1)
        if x_t.dim() == 4:
            assert x_t.shape[1] == 2, f"stereo input must have 2 channels, got {x_t.shape[1]}"
            # (B, 2, F, T) -> (B, 2*F, T)
            B, S, F, T = x_t.shape
            x_t = x_t.reshape(B, S * F, T)
        # Now x_t is (B, C, T) torch.
        B, C, T = x_t.shape
        assert C == self.in_channels, f"expected {self.in_channels} input channels, got {C}"

        # Upload as (B, T, C) ROW_MAJOR fp32.
        x_BTC_torch = x_t.transpose(1, 2).float().contiguous()

        sharded = self.parallel_config is not None and self.parallel_config.factor > 1
        # Pad T to a multiple of (TILE_HEIGHT * factor) so that mesh_partition
        # produces tile-aligned per-chip shards. The "extras" propagate through
        # the chain at upsampled length and get cropped from the final
        # waveform.
        t_pad = 0
        if sharded:
            factor = self.parallel_config.factor
            tile_h = 32
            align = tile_h * factor
            rem = x_BTC_torch.shape[1] % align
            if rem != 0:
                t_pad = align - rem
                x_BTC_torch = torch.nn.functional.pad(x_BTC_torch, (0, 0, 0, t_pad))

        x_dev = ttnn.from_torch(x_BTC_torch, device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=self.dtype)

        if sharded:
            # The from_torch above replicated the tensor across the mesh.
            # mesh_partition along T fractures it for the sharded forward.
            x_dev = ttnn.to_layout(x_dev, ttnn.TILE_LAYOUT)
            x_dev = ttnn.mesh_partition(x_dev, dim=1, cluster_axis=self.parallel_config.mesh_axis)
            x_dev = ttnn.to_layout(x_dev, ttnn.ROW_MAJOR_LAYOUT)

        # conv_pre.
        x_dev = self.conv_pre(x_dev)

        # Upsample stages.
        for i in range(self.num_upsamples):
            x_dev = self.ups[i](x_dev)
            start = i * self.num_kernels
            # Mean over the num_kernels (=3) parallel AMP branches.
            block_outputs = []
            for idx in range(start, start + self.num_kernels):
                block_outputs.append(self.resblocks[idx](x_dev))
            ttnn.deallocate(x_dev)
            # Sum and divide.
            acc = block_outputs[0]
            for k in range(1, self.num_kernels):
                new_acc = ttnn.add(acc, block_outputs[k])
                ttnn.deallocate(acc)
                ttnn.deallocate(block_outputs[k])
                acc = new_acc
            x_dev = ttnn.multiply(acc, 1.0 / self.num_kernels)
            ttnn.deallocate(acc)

        # act_post → conv_post.
        x_dev = self.act_post(x_dev)
        x_dev = self.conv_post(x_dev)

        # Optional clamp/tanh. Reference: ``apply_final_activation`` defaults
        # True, ``use_tanh_at_final`` is False for production → clamp(-1, 1).
        if self.apply_final_activation:
            if self.use_tanh_at_final:
                x_dev = ttnn.tanh(x_dev)
            else:
                x_dev = ttnn.clamp(x_dev, -1.0, 1.0)

        if sharded:
            # All-gather sharded T back to a full sequence on every chip so the
            # host download below sees the full waveform on chip 0.
            x_dev = ttnn.to_layout(x_dev, ttnn.TILE_LAYOUT)
            x_dev = self.ccl_manager.all_gather_persistent_buffer(
                x_dev, dim=1, mesh_axis=self.parallel_config.mesh_axis
            )
            x_dev = ttnn.to_layout(x_dev, ttnn.ROW_MAJOR_LAYOUT)

        # Device → host. Convert (B, T_out, C_out) → (B, C_out, T_out).
        x_host = ttnn.to_torch(ttnn.get_device_tensors(x_dev)[0])
        # Trim padded out channels in case the conv didn't (e.g. when
        # ``out_channels < 32`` and ``aligned == 32`` already inside the conv).
        x_host = x_host[..., : self.out_channels]
        # Crop the upsampled-by-prod(rates) image of the input T-padding.
        if t_pad > 0:
            prod_rates = 1
            for r in self.upsample_rates:
                prod_rates *= r
            x_host = x_host[:, : x_host.shape[1] - t_pad * prod_rates, :]
        x_host = x_host.transpose(-1, -2).contiguous()
        return x_host

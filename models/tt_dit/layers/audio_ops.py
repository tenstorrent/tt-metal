# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Audio-flavored conv primitives for LTX-2 audio decode.

These wrap ``ttnn.experimental.conv3d`` with degenerate axes to express 1D/2D
convs on the small spectrogram/waveform tensors the mel-VAE and vocoder use.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from loguru import logger

# (C, T_pad, K, stride) shapes allowed to use ttnn.conv1d's depthwise kernel
# instead of the MAC shifted-accumulate. Empty: conv1d diverges from MAC by
# ~0.9 dB PSNR on the vocoder anti-alias filters (high-frequency amplitude
# compression), so MAC is the numerical baseline. The conv1d path below stays
# for a future halo-aware depthwise kernel that matches MAC.
_CONV1D_SAFE_SHAPES: set = set()

import ttnn

from ..layers.module import Module, Parameter
from ..parallel.config import AudioTCParallelConfig, AudioTParallelConfig, ParallelFactor
from ..parallel.manager import CCLManager
from ..utils.conv3d import _ntuple, aligned_channels, get_conv3d_config


def channel_axis(parallel_config) -> int | None:
    """Mesh axis the channel (C) dim is tensor-parallel over, or None.

    Channel-TP is sound on a 2D mesh (channels have no sequence boundary), so it
    only lives on :class:`AudioTCParallelConfig`; every other config replicates C.
    """
    if isinstance(parallel_config, AudioTCParallelConfig) and parallel_config.channel_parallel.factor > 1:
        return parallel_config.channel_parallel.mesh_axis
    return None


def channel_factor(parallel_config) -> int:
    """Channel-TP factor (per-chip C-shard count = C / factor), or 1 if no C-TP."""
    if channel_axis(parallel_config) is None:
        return 1
    return parallel_config.channel_parallel.factor


def channel_align_unit(parallel_config) -> int:
    """Channel alignment unit. ``factor * TILE_WIDTH`` under channel-TP so each
    per-chip shard (= unit / factor) is itself a TILE_WIDTH multiple; else 32."""
    from ..utils.conv3d import ALIGNMENT

    return ALIGNMENT * channel_factor(parallel_config)


def partition_channel(x: ttnn.Tensor, parallel_config, *, dim: int) -> ttnn.Tensor:
    """Local slice of a channel-axis-replicated tensor to its per-chip C-shard.

    ``mesh_partition`` of a replicated tensor is local (no communication). No-op
    unless ``parallel_config`` carries a >1 channel factor.
    """
    axis = channel_axis(parallel_config)
    if axis is None:
        return x
    return ttnn.mesh_partition(x, dim=dim, cluster_axis=axis)


def all_gather_channel(ccl_manager, x: ttnn.Tensor, parallel_config, *, dim: int) -> ttnn.Tensor:
    """All-gather a channel-sharded tensor back to full C on the channel axis."""
    axis = channel_axis(parallel_config)
    if axis is None:
        return x
    return ccl_manager.all_gather_persistent_buffer(x, dim=dim, mesh_axis=axis)


def gather_channel_to_full(ccl_manager, x_BTC: ttnn.Tensor, parallel_config) -> ttnn.Tensor:
    """Gather a ``(B, T, C)`` C-shard back to full C_in for a channel-mixing conv.

    No-op without channel-TP. CCL needs TILE, conv3d needs ROW_MAJOR.
    """
    if channel_axis(parallel_config) is None:
        return x_BTC
    x_BTC = ttnn.to_layout(x_BTC, ttnn.TILE_LAYOUT)
    x_BTC = all_gather_channel(ccl_manager, x_BTC, parallel_config, dim=2)
    return ttnn.to_layout(x_BTC, ttnn.ROW_MAJOR_LAYOUT)


def prepare_conv3d_weight_state(
    state: dict,
    w_5d: torch.Tensor,
    *,
    conv_config,
    mesh_device: ttnn.MeshDevice,
    dtype: ttnn.DataType,
    unpadded_out: int,
    out_channels: int,
    unpadded_in: int | None = None,
    in_channels: int | None = None,
) -> None:
    """Zero-pad the 5D conv weight (C_out, and C_in when given) to its aligned size,
    run ``prepare_conv3d_weights``, and write the result to ``state['weight']``;
    ``state['bias']`` is padded to match C_out. C_in padding is opt-in (channel-TP)."""
    if out_channels != unpadded_out:
        pad_co = out_channels - unpadded_out
        w_5d = torch.nn.functional.pad(w_5d, (0, 0, 0, 0, 0, 0, 0, 0, 0, pad_co))
        if "bias" in state:
            state["bias"] = torch.nn.functional.pad(state["bias"], (0, pad_co))
    if in_channels is not None and in_channels != unpadded_in:
        w_5d = torch.nn.functional.pad(w_5d, (0, 0, 0, 0, 0, 0, 0, in_channels - unpadded_in))
    weight_tt = ttnn.from_torch(w_5d, dtype=dtype, pad_value=0)
    prepared = ttnn.experimental.prepare_conv3d_weights(
        weight_tensor=weight_tt, C_in_block=conv_config.C_in_block, device=mesh_device
    )
    state["weight"] = ttnn.to_torch(ttnn.get_device_tensors(prepared)[0])


def _t_neighbor_pad(
    x_BTC: ttnn.Tensor,
    *,
    pad_left: int,
    pad_right: int,
    parallel_config: "ParallelFactor | AudioTParallelConfig",
    ccl_manager: CCLManager,
    padding_mode: str = "zeros",
) -> ttnn.Tensor:
    """Halo exchange on the T axis (dim 1 in BTC layout).

    Supports both single-axis sharding (ParallelFactor) and combined 2-axis
    sharding (AudioTParallelConfig). For 2D sharding the halo is issued as a
    single two-entry neighbor_pad_async call — both axes exchanged atomically.
    """
    if pad_left == 0 and pad_right == 0:
        return x_BTC
    if parallel_config is None or parallel_config.factor <= 1:
        B, T, C = x_BTC.shape
        if pad_left > 0:
            zl = ttnn.zeros(
                (B, pad_left, C), dtype=x_BTC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=x_BTC.device()
            )
            x_BTC = ttnn.concat([zl, x_BTC], dim=1)
        if pad_right > 0:
            zr = ttnn.zeros(
                (B, pad_right, C), dtype=x_BTC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=x_BTC.device()
            )
            x_BTC = ttnn.concat([x_BTC, zr], dim=1)
        return x_BTC

    outer_dims = x_BTC.shape[0]
    num_links = max(1, min(outer_dims, ccl_manager.num_links))

    if isinstance(parallel_config, AudioTParallelConfig):
        # Two-axis halo: neighbor_pad_async requires distinct pad dims, so issue
        # one call per mesh axis (axis 0 then axis 1).
        sem0 = ccl_manager.get_np_ping_pong_semaphore(parallel_config.axis0.mesh_axis)
        x_BTC = ccl_manager.neighbor_pad_persistent_buffer(
            x_BTC,
            dims=[1],
            pad_left=[pad_left],
            pad_right=[pad_right],
            padding_mode=padding_mode,
            axes=[parallel_config.axis0.mesh_axis],
            neighbor_sems=[sem0],
            num_links=[num_links],
        )
        sem1 = ccl_manager.get_np_ping_pong_semaphore(parallel_config.axis1.mesh_axis)
        return ccl_manager.neighbor_pad_persistent_buffer(
            x_BTC,
            dims=[1],
            pad_left=[pad_left],
            pad_right=[pad_right],
            padding_mode=padding_mode,
            axes=[parallel_config.axis1.mesh_axis],
            neighbor_sems=[sem1],
            num_links=[num_links],
        )

    sem = ccl_manager.get_np_ping_pong_semaphore(parallel_config.mesh_axis)
    return ccl_manager.neighbor_pad_persistent_buffer(
        x_BTC,
        dims=[1],
        pad_left=[pad_left],
        pad_right=[pad_right],
        padding_mode=padding_mode,
        axes=[parallel_config.mesh_axis],
        neighbor_sems=[sem],
        num_links=[num_links],
    )


def depthwise_tap_filter(x_BTC, taps, stride, *, mesh_device, dtype, cache):
    """Valid depthwise filter (same K taps on every channel):
    ``y[b, t, c] = sum_{j<K} taps[j] * x[b, t*stride + j, c]`` on an
    already-padded ``(B, T_pad, C)`` ROW_MAJOR input.

    A single ``ttnn.conv1d`` (groups=C) is ~12x faster than the K-tap shifted
    multiply-accumulate, but its HEIGHT_SHARDED reader buffers the sequence in
    L1 and OOMs once T·C is large. So per shape we try conv1d once; on any
    failure we cache a MAC fallback for that shape. The prepared conv weight is
    cached per channel count and reused (preparing it per call is what makes
    conv1d look slow). `cache` is an opaque per-instance dict the caller owns.
    """
    B, T_pad, C = int(x_BTC.shape[0]), int(x_BTC.shape[1]), int(x_BTC.shape[2])
    K = len(taps)
    T_out = (T_pad - K) // stride + 1
    shape_key = (C, T_pad, stride)

    if (C, T_pad, K, stride) not in _CONV1D_SAFE_SHAPES:
        cache[shape_key] = "mac"

    if cache.get(shape_key) != "mac":
        try:
            wprep = cache.get(("w", C))
            weight = wprep
            if weight is None:
                wt = torch.tensor(taps, dtype=torch.float32).reshape(1, 1, K).expand(C, 1, K).contiguous()
                weight = ttnn.from_torch(wt, device=mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=dtype)
            if "cc" not in cache:
                cache["cc"] = ttnn.init_device_compute_kernel_config(
                    mesh_device.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
                )
            conv_config = ttnn.Conv1dConfig(weights_dtype=dtype, shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED)
            out, _, (weight, _bias) = ttnn.conv1d(
                input_tensor=ttnn.reshape(x_BTC, (B, T_pad, 1, C)),
                weight_tensor=weight,
                device=mesh_device,
                in_channels=C,
                out_channels=C,
                batch_size=B,
                input_length=T_pad,
                kernel_size=K,
                stride=stride,
                padding=0,
                dilation=1,
                groups=C,
                dtype=dtype,
                conv_config=conv_config,
                compute_config=cache["cc"],
                return_output_dim=True,
                return_weights_and_bias=True,
            )
            cache[("w", C)] = weight
            cache[shape_key] = "conv1d"
            # conv1d emits HEIGHT_SHARDED TILE; the MAC path and all downstream ops
            # (T-halo neighbor_pad, convs) expect interleaved ROW_MAJOR — match it.
            out = ttnn.to_memory_config(out, ttnn.DRAM_MEMORY_CONFIG)
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
            return ttnn.reshape(out, (B, T_out, C))
        except Exception:
            # conv1d doesn't fit this shape's L1 (or is otherwise unsupported);
            # the MAC loop below is the always-correct fallback.
            cache[shape_key] = "mac"

    y = None
    for j in range(K):
        w = float(taps[j])
        if stride == 1:
            slice_j = ttnn.slice(x_BTC, [0, j, 0], [B, j + T_out, C])
        else:
            slice_j = ttnn.slice(x_BTC, [0, j, 0], [B, j + (T_out - 1) * stride + 1, C], [1, stride, 1])
        scaled = ttnn.multiply(slice_j, w)
        ttnn.deallocate(slice_j)
        if y is None:
            y = scaled
        else:
            y_new = ttnn.add(y, scaled)
            ttnn.deallocate(y)
            ttnn.deallocate(scaled)
            y = y_new
    return y


def _all_gather_t(ccl_manager, x: "ttnn.Tensor", parallel_config) -> "ttnn.Tensor":
    """All-gather the T-sharded tensor to full T on every chip."""
    if isinstance(parallel_config, AudioTParallelConfig):
        x = ccl_manager.all_gather_persistent_buffer(x, dim=1, mesh_axis=parallel_config.axis1.mesh_axis)
        x = ccl_manager.all_gather_persistent_buffer(x, dim=1, mesh_axis=parallel_config.axis0.mesh_axis)
    else:
        x = ccl_manager.all_gather_persistent_buffer(x, dim=1, mesh_axis=parallel_config.mesh_axis)
    return x


def _partition_t(x: "ttnn.Tensor", parallel_config) -> "ttnn.Tensor":
    """Partition T across the mesh (inverse of _all_gather_t)."""
    if isinstance(parallel_config, AudioTParallelConfig):
        x = ttnn.mesh_partition(x, dim=1, cluster_axis=parallel_config.axis0.mesh_axis)
        x = ttnn.mesh_partition(x, dim=1, cluster_axis=parallel_config.axis1.mesh_axis)
    else:
        x = ttnn.mesh_partition(x, dim=1, cluster_axis=parallel_config.mesh_axis)
    return x


def _make_kaiser_sinc_kernel_1d(cutoff: float, half_width: float, kernel_size: int) -> torch.Tensor:
    """Return a shape-``(kernel_size,)`` kaiser-windowed sinc filter."""
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


def _replicate_pad_t(x_BTC: ttnn.Tensor, pad_left: int, pad_right: int, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Replicate-pad along the T axis for a ``(B, T, C)`` ROW_MAJOR tensor."""
    if pad_left == 0 and pad_right == 0:
        return x_BTC
    B, T, C = x_BTC.shape
    pieces = []
    if pad_left > 0:
        first = ttnn.slice(x_BTC, [0, 0, 0], [B, 1, C])
        pieces.extend([first] * pad_left)
    pieces.append(x_BTC)
    if pad_right > 0:
        last = ttnn.slice(x_BTC, [0, T - 1, 0], [B, T, C])
        pieces.extend([last] * pad_right)
    return ttnn.concat(pieces, dim=1)


def _tpad_mask(mesh_device, parallel_config, dtype, global_T, tpad_image, cache):
    """Cached sharded validity mask ``M`` and its complement ``inv``, each ``(1, T, 1)``: ``M`` is
    1.0 for real rows and 0.0 for the trailing ``tpad_image`` rows, ``inv`` the reverse. Sharded
    across T so each chip masks its own rows; the zeros land on the last shard(s), where the
    tile-align pad image lives. Both built on host (0/1 are exact in bf16) so replicate fill is a
    cache fetch, not a per-call device complement."""
    key = (global_T, tpad_image, dtype)
    cached = cache.get(key)
    if cached is None:
        m = torch.ones(1, global_T, 1, dtype=torch.float32)
        m[:, global_T - tpad_image :, :] = 0.0
        pair = []
        for t in (m, 1.0 - m):
            mt = ttnn.from_torch(t, device=mesh_device, layout=ttnn.TILE_LAYOUT, dtype=dtype)
            mt = _partition_t(mt, parallel_config)
            pair.append(ttnn.to_layout(mt, ttnn.ROW_MAJOR_LAYOUT))
        cached = tuple(pair)
        cache[key] = cached
    return cached


def _set_tpad_tail(x_BTC, tpad_image, *, mode, mesh_device, parallel_config, cache):
    """Materialize the tile-align pad image — the trailing ``tpad_image`` rows at the global
    sequence tail — to what the *next* op's own padding produces on the unsharded path, so the
    sharded global tail bit-matches unsharded:

    - ``mode="zeros"``: zero the rows. For zeros-pad convs, and the upsamplers (which gather
      T to full and zero-pad internally, so they must see zeros there).
    - ``mode="replicate"``: fill with the last real row, for the replicate-pad activations.

    CCL-free: a cached validity mask zeros the pad image (body rows multiply by exactly 1.0,
    so they stay bit-identical); replicate adds the real-last row, sliced at a uniform local
    index. Called ~100x per forward, so a gather here would dominate runtime.
    """
    if tpad_image <= 0 or parallel_config is None or getattr(parallel_config, "factor", 0) <= 1:
        return x_BTC
    local_T = x_BTC.shape[1]
    M, inv = _tpad_mask(
        mesh_device, parallel_config, x_BTC.get_dtype(), local_T * parallel_config.factor, tpad_image, cache
    )
    xm = ttnn.multiply(x_BTC, M)
    if mode == "zeros":
        return xm
    if mode != "replicate":
        raise ValueError(f"unknown mode {mode!r}")
    # Local index of the real-last row, uniform across shards. When the pad image spans more
    # than one shard (high factor + short mel), the real->pad boundary is on an interior shard,
    # not the last; mod local_T lands the slice on it. Shards fully past the boundary get a
    # garbage fill, but they are trimmed at output and sit beyond every replicate consumer's
    # (local, halo-bounded) receptive field, and every gather op re-masks them to zeros first.
    global_T = local_T * parallel_config.factor
    assert tpad_image < global_T, f"pad image ({tpad_image}) leaves no real rows (global T {global_T})"
    idx = (global_T - tpad_image - 1) % local_T
    last = ttnn.slice(x_BTC, [0, idx, 0], [x_BTC.shape[0], idx + 1, x_BTC.shape[2]])
    fill = ttnn.multiply(last, inv)
    ttnn.deallocate(last)
    out = ttnn.add(xm, fill)
    ttnn.deallocate(xm)
    ttnn.deallocate(fill)
    return out


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


def _pad_channels_to_aligned(x_BTC: ttnn.Tensor, mesh_device: ttnn.MeshDevice, channel_align: int = 32) -> ttnn.Tensor:
    """Pad C up to ``aligned_channels(C, channel_align)`` with zeros. No-op if aligned."""
    B, T, C = x_BTC.shape
    aligned = aligned_channels(C, channel_align)
    if aligned == C:
        return x_BTC
    pad_c = aligned - C
    dtype = x_BTC.get_dtype()
    zeros = ttnn.zeros((B, T, pad_c), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    return ttnn.concat([x_BTC, zeros], dim=2)


def _zero_stuff_t(x_BTC: ttnn.Tensor, *, stride: int, mesh_device: ttnn.MeshDevice) -> ttnn.Tensor:
    """Insert ``stride - 1`` zeros between input samples along T, output length
    ``T*s - (s-1)``. Expresses ``ConvTranspose1d`` as a regular ``Conv1d``.

    Implemented as concat + reshape (O(1) ttnn ops) rather than O(T).
    """
    if stride == 1:
        return x_BTC
    B, T, C = x_BTC.shape
    dtype = x_BTC.get_dtype()
    x_btoc = ttnn.reshape(x_BTC, (B, T, 1, C))
    zero_block = ttnn.zeros((B, T, stride - 1, C), dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device)
    stacked = ttnn.concat([x_btoc, zero_block], dim=2)
    interleaved = ttnn.reshape(stacked, (B, T * stride, C))
    out_len = T * stride - (stride - 1)
    return ttnn.slice(interleaved, [0, 0, 0], [B, out_len, C])


class Conv2dViaConv3d(Module):
    """2D conv via ``ttnn.experimental.conv3d`` with kernel ``(1, kh, kw)`` on a
    ``(B, 1, H, W, C)`` ROW_MAJOR tensor. Single-device only (no halo).

    Padding modes: ``"zeros"`` (symmetric internal), ``"causal_height"`` /
    ``"causal_width"`` (external ``k-1`` front pad on the causal axis).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | Sequence[int],
        stride: int | Sequence[int] = 1,
        padding_mode: str = "zeros",
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.bfloat16,
    ) -> None:
        super().__init__()

        if padding_mode not in ("zeros", "causal_height", "causal_width"):
            raise ValueError(f"padding_mode must be zeros/causal_height/causal_width, got {padding_mode!r}")

        self.unpadded_in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.in_channels = aligned_channels(in_channels)
        self.out_channels = max(32, out_channels)
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        kh, kw = _ntuple(kernel_size, 2)
        sh, sw = _ntuple(stride, 2)
        self.kernel_size = (1, kh, kw)
        self.stride = (1, sh, sw)
        self.padding_mode = padding_mode
        self.pad_h = kh - 1
        self.pad_w = kw - 1
        self.mesh_device = mesh_device
        self.dtype = dtype

        # Causal modes pad externally on the causal axis (internal pad=0 there);
        # zeros mode lets the conv3d kernel pad symmetrically.
        if padding_mode == "zeros":
            self.internal_padding = (0, kh // 2, kw // 2)
        elif padding_mode == "causal_height":
            self.internal_padding = (0, 0, kw // 2)
        else:  # causal_width
            self.internal_padding = (0, kh // 2, 0)

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
        )

        from models.common.utility_functions import is_blackhole

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4
            if (is_blackhole() or dtype == ttnn.float32)
            else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Map ``Conv2d.weight (Cout, Cin, kh, kw)`` → conv3d-prepared ``(d, Cout)``."""
        if "conv.weight" in state:
            state["weight"] = state.pop("conv.weight")
        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias")

        if "weight" in state:
            w = state["weight"]
            assert w.dim() == 4, f"expected 4D Conv2d weight, got {tuple(w.shape)}"
            prepare_conv3d_weight_state(
                state,
                w.unsqueeze(2).contiguous(),  # (Cout, Cin, kh, kw) → (Cout, Cin, 1, kh, kw)
                conv_config=self.conv_config,
                mesh_device=self.mesh_device,
                dtype=self.dtype,
                unpadded_out=self.unpadded_out_channels,
                out_channels=self.out_channels,
            )
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x_BHWC: ttnn.Tensor) -> ttnn.Tensor:
        """``x_BHWC``: ``(B, H, W, C)`` ROW_MAJOR. Internally pads to 5D and calls conv3d."""
        assert x_BHWC.layout == ttnn.ROW_MAJOR_LAYOUT, f"expected ROW_MAJOR, got {x_BHWC.layout}"
        B, H, W, C = x_BHWC.shape

        if self.padding_mode == "causal_height" and self.pad_h > 0:
            B_, H_, W_, C_ = x_BHWC.shape
            pad_tensor_shape = (B_, self.pad_h, W_, C_)
            zero_pad = ttnn.zeros(
                pad_tensor_shape, dtype=x_BHWC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
            )
            x_BHWC = ttnn.concat([zero_pad, x_BHWC], dim=1)
        elif self.padding_mode == "causal_width" and self.pad_w > 0:
            B_, H_, W_, C_ = x_BHWC.shape
            pad_tensor_shape = (B_, H_, self.pad_w, C_)
            zero_pad = ttnn.zeros(
                pad_tensor_shape, dtype=x_BHWC.get_dtype(), layout=ttnn.ROW_MAJOR_LAYOUT, device=self.mesh_device
            )
            x_BHWC = ttnn.concat([zero_pad, x_BHWC], dim=2)

        x_5d = ttnn.reshape(x_BHWC, (x_BHWC.shape[0], 1, x_BHWC.shape[1], x_BHWC.shape[2], x_BHWC.shape[3]))

        out_5d = ttnn.experimental.conv3d(
            input_tensor=x_5d,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.internal_padding,
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )

        out = ttnn.reshape(out_5d, (out_5d.shape[0], out_5d.shape[2], out_5d.shape[3], out_5d.shape[4]))
        return out


class Conv1dViaConv3d(Module):
    """1D conv via ``ttnn.experimental.conv3d`` with kernel ``(k, 1, 1)`` on a
    ``(B, T, 1, 1, C)`` ROW_MAJOR tensor.

    Padding modes: ``"zeros"`` (internal symmetric, i.e. "same"), ``"causal"``
    (external ``k-1`` front pad on T). When ``parallel_config`` shards T, the
    pad becomes a halo exchange via ``ccl_manager``.
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
        channel_shard_output: bool = True,
    ) -> None:
        super().__init__()

        if padding_mode not in ("zeros", "causal"):
            raise ValueError(f"padding_mode must be zeros/causal, got {padding_mode!r}")
        if dilation != 1:
            # "same" padding with dilation uses effective_kernel = (k-1)*d+1 below.
            pass

        sharded = parallel_config is not None and parallel_config.factor > 1
        if sharded:
            assert ccl_manager is not None, "T-sharding requires ccl_manager"
        # Channel-TP gathers C_in then computes only this chip's C_out slice.
        self.channel_shard_output = channel_shard_output
        if channel_axis(parallel_config) is not None:
            assert ccl_manager is not None, "channel-TP requires ccl_manager"

        # Channel-TP rounds C to factor*TILE_WIDTH so each shard is TILE-legal.
        self.channel_align = channel_align_unit(parallel_config)
        self.unpadded_in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.in_channels = aligned_channels(in_channels, self.channel_align)
        self.out_channels = aligned_channels(max(32, out_channels), self.channel_align)
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size = (kernel_size, 1, 1)
        self.stride = (stride, 1, 1)
        self.dilation = dilation
        self.padding_mode = padding_mode
        self.bias_enabled = bias
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        eff_k = (kernel_size - 1) * dilation + 1

        if sharded:
            # Zero internal T pad; halo exchange in forward() supplies context.
            self.internal_padding = (0, 0, 0)
            self.external_pad_front = 0
            if padding_mode == "zeros":
                self.halo_pad_left = eff_k // 2
                self.halo_pad_right = eff_k // 2
            else:  # causal
                self.halo_pad_left = eff_k - 1
                self.halo_pad_right = 0
        elif padding_mode == "zeros":
            self.internal_padding = (eff_k // 2, 0, 0)
            self.external_pad_front = 0
            self.halo_pad_left = 0
            self.halo_pad_right = 0
        else:  # causal — pad k-1 at front externally
            self.internal_padding = (0, 0, 0)
            self.external_pad_front = eff_k - 1
            self.halo_pad_left = 0
            self.halo_pad_right = 0

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
            h_factor=1,
            w_factor=1,
        )

        # Column-parallel channel-TP: each chip owns out_channels/factor output
        # channels (the weight is sharded on C_out at load); C_in stays full (gathered).
        self.out_channels_shard = self.out_channels // channel_factor(parallel_config)
        if channel_factor(parallel_config) > 1:
            # Reuse the full config's C_in_block: the weight is prepared once with it
            # and C_in stays full, so only C_out_block shrinks to the per-chip slice.
            self.conv_config_shard = ttnn.Conv3dConfig(
                weights_dtype=self.conv_config.weights_dtype,
                output_layout=ttnn.ROW_MAJOR_LAYOUT,
                T_out_block=self.conv_config.T_out_block,
                W_out_block=self.conv_config.W_out_block,
                H_out_block=self.conv_config.H_out_block,
                C_out_block=min(self.conv_config.C_out_block, self.out_channels_shard),
                C_in_block=self.conv_config.C_in_block,
                compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._alloc_weight_bias()

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Map torch ``Conv1d.weight (Cout, Cin, k)`` → conv3d-prepared ``(d, Cout)``."""
        if "conv.weight" in state:
            state["weight"] = state.pop("conv.weight")
        if "conv.bias" in state:
            state["bias"] = state.pop("conv.bias")

        if "weight" in state:
            w = state["weight"]
            assert w.dim() == 3, f"expected 3D Conv1d weight, got {tuple(w.shape)}"
            # C_in is padded to the aligned count (divisible by conv_config C_in_block;
            # channel-TP rounds C_in up to factor*32).
            prepare_conv3d_weight_state(
                state,
                w.unsqueeze(-1).unsqueeze(-1).contiguous(),  # (Cout, Cin, k) → (Cout, Cin, k, 1, 1)
                conv_config=self.conv_config,
                mesh_device=self.mesh_device,
                dtype=self.dtype,
                unpadded_out=self.unpadded_out_channels,
                out_channels=self.out_channels,
                unpadded_in=self.unpadded_in_channels,
                in_channels=self.in_channels,
            )
        if "bias" in state and self.bias is not None:
            state["bias"] = state["bias"].reshape(1, -1)

    def _is_col_parallel(self) -> bool:
        return channel_axis(self.parallel_config) is not None and self.channel_shard_output

    def _alloc_weight_bias(self) -> None:
        """Allocate weight/bias. Column-parallel shards C_out across the channel axis
        at load (each chip stores only its slice); otherwise replicated."""
        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        mesh_axes = [None, channel_axis(self.parallel_config)] if self._is_col_parallel() else None
        self.weight = Parameter(
            total_shape=[d, self.out_channels],
            device=self.mesh_device,
            pad_value=0,
            dtype=self.dtype,
            mesh_axes=mesh_axes,
        )
        self.bias = (
            Parameter(
                total_shape=[1, self.out_channels],
                device=self.mesh_device,
                pad_value=0,
                dtype=self.dtype,
                mesh_axes=mesh_axes,
            )
            if self.bias_enabled
            else None
        )

    def _conv_args(self):
        """``(weight, bias, conv_config, output_channels)``. The weight is already the
        per-chip C_out shard when column-parallel, so the output is C-sharded directly."""
        bias = self.bias.data if self.bias is not None else None
        if self._is_col_parallel():
            return self.weight.data, bias, self.conv_config_shard, self.out_channels_shard
        return self.weight.data, bias, self.conv_config, self.out_channels

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        """``x_BTC``: ``(B, T, C)`` ROW_MAJOR → ``(B, T_out, C_out)``.

        When sharded, ``T`` is the per-device extent and the halo exchange adds
        neighbor-chip boundary context before the conv.
        """
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT

        # Channel-TP: gather C_in to full (the conv mixes channels), then column-parallel
        # produces only this chip's C_out slice (already C-sharded, no scatter).
        x_BTC = gather_channel_to_full(self.ccl_manager, x_BTC, self.parallel_config)
        weight, bias, conv_config, out_channels = self._conv_args()

        if self.parallel_config is not None and self.parallel_config.factor > 1:
            x_BTC = _t_neighbor_pad(
                x_BTC,
                pad_left=self.halo_pad_left,
                pad_right=self.halo_pad_right,
                parallel_config=self.parallel_config,
                ccl_manager=self.ccl_manager,
                padding_mode="zeros",
            )
        elif self.external_pad_front > 0:
            B, T, C = x_BTC.shape
            zero_pad = ttnn.zeros(
                (B, self.external_pad_front, C),
                dtype=x_BTC.get_dtype(),
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.mesh_device,
            )
            x_BTC = ttnn.concat([zero_pad, x_BTC], dim=1)

        x_5d = ttnn.reshape(x_BTC, (x_BTC.shape[0], x_BTC.shape[1], 1, 1, x_BTC.shape[2]))

        out_5d = ttnn.experimental.conv3d(
            input_tensor=x_5d,
            weight_tensor=weight,
            bias_tensor=bias,
            config=conv_config,
            output_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.internal_padding,
            dilation=(self.dilation, 1, 1),
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        # Column-parallel output is already the C-shard; no scatter needed.
        return ttnn.reshape(out_5d, (out_5d.shape[0], out_5d.shape[1], out_5d.shape[4]))


class _AlignedOutConv1d(Conv1dViaConv3d):
    """Conv1dViaConv3d variant that rounds ``out_channels`` to a 32-multiple.

    The base ``max(32, out)`` rule lets non-32-multiples (48, 24) reach
    ``ttnn.experimental.conv3d``, which then produces a buffer whose page size
    does not divide its length. We round up, zero-pad weight/bias on the ``out``
    axis, pad input C to aligned in forward, and trim back to the real count.
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
        channel_shard_output: bool = True,
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
            channel_shard_output=channel_shard_output,
        )

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        # Under channel-TP the input is a C-shard; super().forward gathers it to
        # full in_channels, so skip the (replicated-tensor) local pad here.
        if channel_axis(self.parallel_config) is None:
            x_BTC = _pad_channels_to_aligned(x_BTC, self.mesh_device, channel_align=self.channel_align)
        y = super().forward(x_BTC)
        # Column-parallel output is the per-chip C_out slice of the (padded) channels
        # — can't trim to real C_out per chip; the trim happens once at the output.
        if not self._is_col_parallel() and self.unpadded_out_channels < self.out_channels:
            B, T, C = y.shape
            y = ttnn.slice(y, [0, 0, 0], [B, T, self.unpadded_out_channels])
        return y


class ConvTranspose1dViaConv3d(Module):
    """Substitute for ``torch.nn.ConvTranspose1d`` with ``padding=(k-stride)//2``.

    Equivalent to ``Conv1d`` on the zero-stuffed input with the weight flipped
    along the kernel axis and transposed. The external pad
    ``p = k - 1 - (k-s)//2`` is the unique value yielding an exact stride x
    upsample.
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
        self.external_pad_each = kernel_size - 1 - (kernel_size - stride) // 2
        self.mesh_device = mesh_device
        self.dtype = dtype
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        # When sharded the underlying conv stays UNSHARDED: the transposed-conv
        # math (zero-stuff stride > 1 + asymmetric local zero-pad) is awkward to
        # halo cleanly on T. Forward gathers across T, runs the unsharded
        # pipeline, then mesh-partitions the output back. Only 6 per vocoder, so
        # the gather/partition overhead is small.
        self.conv = _AlignedOutConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding_mode="causal",
            bias=bias,
            mesh_device=mesh_device,
            dtype=dtype,
            parallel_config=None,
            ccl_manager=None,
        )
        # We supply our own symmetric padding instead.
        self.conv.external_pad_front = 0

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        """Reshape ConvTranspose1d weight ``(in, out, k)`` → Conv1d ``(out, in, k)``.

        The flip-along-k is required because Conv1d is cross-correlation but the
        zero-stuff equivalent of ConvTranspose1d needs a flipped kernel. Keys are
        migrated to ``conv.*`` so the base class can pick them up.
        """
        if "weight" in state:
            w = state.pop("weight")
            assert w.dim() == 3 and tuple(w.shape) == (self.in_channels, self.out_channels, self.kernel_size), (
                f"expected ConvTranspose1d weight shape ({self.in_channels}, {self.out_channels}, "
                f"{self.kernel_size}), got {tuple(w.shape)}"
            )
            w_flipped = torch.flip(w, dims=[-1])
            w_conv1d = w_flipped.permute(1, 0, 2).contiguous()
            state["conv.weight"] = w_conv1d
        if "bias" in state:
            state["conv.bias"] = state.pop("bias")

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        assert x_BTC.layout == ttnn.ROW_MAJOR_LAYOUT
        sharded = self.parallel_config is not None and self.parallel_config.factor > 1

        # Channel-TP: the inner conv runs UNSHARDED (parallel_config=None), so gather
        # C_in to full here and scatter C_out back at the end.
        ch_axis = channel_axis(self.parallel_config)
        if ch_axis is not None:
            x_BTC = gather_channel_to_full(self.ccl_manager, x_BTC, self.parallel_config)
            # Gathered C is unit-aligned (factor*32); drop the pad channels so the
            # aligned-32 inner conv sees its real C_in.
            x_BTC = ttnn.slice(x_BTC, [0, 0, 0], [x_BTC.shape[0], x_BTC.shape[1], self.in_channels])

        if sharded:
            x_BTC = ttnn.to_layout(x_BTC, ttnn.TILE_LAYOUT)
            x_BTC = _all_gather_t(self.ccl_manager, x_BTC, self.parallel_config)
            x_BTC = ttnn.to_layout(x_BTC, ttnn.ROW_MAJOR_LAYOUT)

        # The runtime input C must match the aligned-C the conv weight was
        # allocated for.
        x_BTC = _pad_channels_to_aligned(x_BTC, self.mesh_device)
        x_zs = _zero_stuff_t(x_BTC, stride=self.stride, mesh_device=self.mesh_device)
        x_padded = _zero_pad_t(x_zs, self.external_pad_each, self.external_pad_each, self.mesh_device)
        y = self.conv(x_padded)

        if sharded:
            y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
            y = _partition_t(y, self.parallel_config)
            y = ttnn.to_layout(y, ttnn.ROW_MAJOR_LAYOUT)

        if ch_axis is not None:
            # Re-pad real C_out to unit so the per-chip C-shard is TILE-legal.
            y = _pad_channels_to_aligned(y, self.mesh_device, channel_align=channel_align_unit(self.parallel_config))
            y = partition_channel(y, self.parallel_config, dim=2)

        return y


class Snake(Module):
    """``y = x + (1 / (α + ε)) · sin(α · x)²``. α has shape ``(1, 1, C)`` per channel."""

    def __init__(
        self,
        channels: int,
        *,
        alpha_logscale: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config=None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale  # If True, learned param is log(α); collapse at load time.
        self.eps = 1e-9
        self.parallel_config = parallel_config
        # Under channel-TP, pad α to the C-shard unit so it partitions to match the padded activation.
        unit = channel_align_unit(parallel_config) if channel_axis(parallel_config) is not None else 1
        self._aligned_channels = aligned_channels(channels, unit)
        self.alpha = Parameter(total_shape=[1, 1, self._aligned_channels], device=mesh_device, dtype=dtype)
        self._alpha_shard = None  # cached per-chip C-shard of the (static) α

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "alpha" in state:
            a = state["alpha"]
            if self.alpha_logscale:
                a = torch.exp(a)
            a = a.reshape(1, 1, -1)
            if a.shape[-1] < self._aligned_channels:
                a = torch.nn.functional.pad(a, (0, self._aligned_channels - a.shape[-1]))
            state["alpha"] = a.contiguous()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        # α is per-channel; slice it to the activation's C-shard when channel-TP.
        if self._alpha_shard is None:
            self._alpha_shard = partition_channel(self.alpha.data, self.parallel_config, dim=2)
        a = self._alpha_shard
        ax = ttnn.multiply(x_BTC, a)
        s = ttnn.sin(ax)
        s2 = ttnn.multiply(s, s)
        inv = ttnn.reciprocal(ttnn.add(a, self.eps))
        return ttnn.add(x_BTC, ttnn.multiply(s2, inv))


class SnakeBeta(Module):
    """``y = x + (1 / (β + ε)) · sin(α · x)²``. α, β both ``(1, 1, C)`` learned."""

    def __init__(
        self,
        channels: int,
        *,
        alpha_logscale: bool = False,
        mesh_device: ttnn.MeshDevice,
        dtype: ttnn.DataType = ttnn.float32,
        parallel_config=None,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.alpha_logscale = alpha_logscale
        self.eps = 1e-9
        self.parallel_config = parallel_config
        # Under channel-TP, pad α/β to the C-shard unit to match the padded activation.
        unit = channel_align_unit(parallel_config) if channel_axis(parallel_config) is not None else 1
        self._aligned_channels = aligned_channels(channels, unit)
        self.alpha = Parameter(total_shape=[1, 1, self._aligned_channels], device=mesh_device, dtype=dtype)
        self.beta = Parameter(total_shape=[1, 1, self._aligned_channels], device=mesh_device, dtype=dtype)
        self._ab_shard = None  # cached per-chip C-shards of the (static) α, β

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        for name in ("alpha", "beta"):
            if name in state:
                t = state[name]
                if self.alpha_logscale:
                    t = torch.exp(t)
                t = t.reshape(1, 1, -1)
                if t.shape[-1] < self._aligned_channels:
                    t = torch.nn.functional.pad(t, (0, self._aligned_channels - t.shape[-1]))
                state[name] = t.contiguous()

    def forward(self, x_BTC: ttnn.Tensor) -> ttnn.Tensor:
        # α, β are per-channel; slice to the activation's C-shard when channel-TP.
        if self._ab_shard is None:
            self._ab_shard = (
                partition_channel(self.alpha.data, self.parallel_config, dim=2),
                partition_channel(self.beta.data, self.parallel_config, dim=2),
            )
        a, b = self._ab_shard
        ax = ttnn.multiply(x_BTC, a)
        s = ttnn.sin(ax)
        s2 = ttnn.multiply(s, s)
        inv = ttnn.reciprocal(ttnn.add(b, self.eps))
        return ttnn.add(x_BTC, ttnn.multiply(s2, inv))

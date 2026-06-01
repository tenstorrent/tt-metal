# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Audio-flavored conv primitives for LTX-2 audio decode.

These wrap ``ttnn.experimental.conv3d`` with degenerate axes to express 1D/2D
convs on the small spectrogram/waveform tensors the mel-VAE and vocoder use.
"""

from __future__ import annotations

from typing import Sequence

import torch
from loguru import logger

# (C, T_pad, K, stride) for which ttnn.conv1d's depthwise kernel builds (its
# in0_block_w % K assert holds only for some C) and fits L1. The assert is a fatal,
# uncatchable crash, so conv1d is opt-in per shape; every other shape uses MAC.
_CONV1D_SAFE_SHAPES: set = {
    (256, 6155, 12, 2),
    (384, 5131, 12, 2),
    (384, 5161, 12, 1),
    (768, 2571, 12, 2),
    (768, 2601, 12, 1),
}

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
            padding_mode="zeros",
            dtype=self.dtype,
            compute_kernel_config=self.compute_kernel_config,
        )
        # Column-parallel output is already the C-shard; no scatter needed.
        return ttnn.reshape(out_5d, (out_5d.shape[0], out_5d.shape[1], out_5d.shape[4]))


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

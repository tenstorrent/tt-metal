# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType

import torch
from loguru import logger

import ttnn
from models.common.utility_functions import is_blackhole
from models.tt_dit.layers.conv2d import Conv2d
from models.tt_dit.layers.linear import ColParallelLinear, RowParallelLinear, _apply_activation_fn
from models.tt_dit.layers.module import Module, ModuleList, Parameter
from models.tt_dit.layers.normalization import DistributedGroupNorm, DistributedRMSNorm, GroupNorm, RMSNorm
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.utils import tensor
from models.tt_dit.utils.conv3d import aligned_channels, get_conv3d_config
from models.tt_dit.utils.substate import rename_substate
from models.tt_dit.utils.tensor import local_device_to_torch

VAE_A_MAX_SLICES = 8  # DRAM slices per conv the norm-fusion statistics page block covers


@dataclass(frozen=True)
class VaeContext:
    """Runtime context for VAE blocks.

    Channel tensor-parallelism is configured via ``tp_axis`` (None disables).
    Spatial parallelism shards the activation on H and/or W:
      * ``h_factor > 1`` shards H across ``h_mesh_axis``.
      * ``w_factor > 1`` shards W across ``w_mesh_axis``.
    Both default to ``factor=1`` (disabled) so existing TP-only callers are unchanged.
    The H/W mesh axes must differ from ``tp_axis``.
    """

    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None
    h_mesh_axis: int | None = None
    h_factor: int = 1
    w_mesh_axis: int | None = None
    w_factor: int = 1
    use_conv3d: bool = False
    # 2D spatial decode (no TP): 3x3 convs pad their borders from a boundary exchange instead of NeighborPad,
    # a resnet block's norm2 comes from conv1's epilogue statistics (its affine + SiLU in conv2's halo prologue), and
    # the mid-block attention norms the local shard
    external_halo: bool = False
    norm_fusion: bool = False
    norm_prologue: bool = False
    local_attention: bool = False


@dataclass(frozen=True)
class VaeNormDescRms:
    eps: float


@dataclass(frozen=True)
class VaeNormDescGroup:
    eps: float
    num_groups: int


VaeNormDesc = VaeNormDescRms | VaeNormDescGroup


# ---------------------------------------------------------------------------
# Spatial-parallel helpers (no-op when h_factor == w_factor == 1)
# ---------------------------------------------------------------------------


def _all_gather_hw(ctx: VaeContext, x: ttnn.Tensor) -> ttnn.Tensor:
    """All-gather H then W to reconstruct full spatial: [N, H/h, W/w, C] → [N, H, W, C].

    Returns TILE_LAYOUT (the gather kernel requires it). No-op when both factors are 1.
    """
    if ctx.h_factor <= 1 and ctx.w_factor <= 1:
        return x
    assert ctx.ccl_manager is not None
    if x.layout != ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    if ctx.h_factor > 1:
        x = ctx.ccl_manager.all_gather(x, dim=1, mesh_axis=ctx.h_mesh_axis, use_hyperparams=True)
    if ctx.w_factor > 1:
        x = ctx.ccl_manager.all_gather(x, dim=2, mesh_axis=ctx.w_mesh_axis, use_hyperparams=True)
    return x


def _partition_hw(ctx: VaeContext, x: ttnn.Tensor) -> ttnn.Tensor:
    """Partition full spatial back to sharded: [N, H, W, C] → [N, H/h, W/w, C].

    Restores the input layout (mesh_partition requires ROW_MAJOR internally).
    No-op when both factors are 1.
    """
    if ctx.h_factor <= 1 and ctx.w_factor <= 1:
        return x
    original_layout = x.layout
    if x.layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    if ctx.h_factor > 1:
        x = ttnn.mesh_partition(x, dim=1, cluster_axis=ctx.h_mesh_axis, memory_config=x.memory_config())
    if ctx.w_factor > 1:
        x = ttnn.mesh_partition(x, dim=2, cluster_axis=ctx.w_mesh_axis, memory_config=x.memory_config())
    if original_layout != ttnn.ROW_MAJOR_LAYOUT:
        x = ttnn.to_layout(x, original_layout)
    return x


def _get_neighbor_pad_num_links(ccl_manager: CCLManager, input_tensor: ttnn.Tensor, dim: int) -> int:
    """Cap links at the product of dimensions above ``dim`` (kernel constraint)."""
    upper_dims = 1
    for i in range(dim):
        upper_dims *= input_tensor.shape[i]
    return min(upper_dims, ccl_manager.num_links)


class VaeRmsNorm(Module):
    def __init__(self, num_channels: int, *, eps: float, ctx: VaeContext, activation_fn: str | None = None) -> None:
        super().__init__()

        # Neither RMSNorm path here fuses an activation, so it always runs as a separate op.
        self.activation_fn = activation_fn

        tp_axis_size = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1

        # https://github.com/tenstorrent/tt-metal/issues/31216
        self._use_rms_workaround = num_channels % (tp_axis_size * 32) != 0

        if self._use_rms_workaround:
            self.gamma = Parameter(total_shape=[num_channels], mesh_axes=[ctx.tp_axis], device=ctx.device)
        else:
            self.inner = (
                DistributedRMSNorm(
                    num_channels,
                    norm_eps=eps,
                    bias=False,
                    mesh_axis=ctx.tp_axis,
                    mesh_device=ctx.device,
                    ccl_manager=ctx.ccl_manager,
                )
                if tp_axis_size != 1
                else RMSNorm(
                    num_channels,
                    norm_eps=eps,
                    bias=False,
                    mesh_device=ctx.device,
                )
            )

        self._eps = eps
        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager
        self._device = ctx.device
        self._tp_axis_size = tp_axis_size

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "gamma" in state:
            if self._use_rms_workaround:
                state["gamma"] = state["gamma"].reshape([-1])
            else:
                state["inner.weight"] = state.pop("gamma").reshape([-1])

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if not self._use_rms_workaround:
            # Both norms take the 4D tensor as is. Flattening it would be a costly tiled reshape
            # whenever w is not a multiple of the tile size.
            return _apply_activation_fn(self.inner.forward(x), self.activation_fn)

        norm = ttnn.mean(ttnn.pow(x, 2), dim=-1, keepdim=True)

        if self._tp_axis_size != 1:
            assert self._ccl_manager is not None

            n = self._tp_axis_size
            tensor_rank = len(x.shape)
            # repeat the tensor since we do not have an all-reduce op yet
            norm = ttnn.repeat(norm, [1] * (tensor_rank - 1) + [n])
            norm = self._ccl_manager.reduce_scatter(norm, dim=3, mesh_axis=self._tp_axis)
            norm = norm * (1 / n)

        norm = ttnn.rsqrt(norm + self._eps)
        return _apply_activation_fn(x * (norm * self.gamma.data), self.activation_fn)


class _VaeConv2dConv3d(Module):
    """Conv2d implemented via conv3d with T=1. SP-aware (receives pre-padded input from
    VaeConv2d), but no TP sharding — used only when ctx.tp_axis is None or tensor_parallel
    is False.  Weights are stored in the prepared conv3d format (shape [k*k*C_in, C_out]).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int,
        padding: tuple[int, int, int, int],
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        in_channels = aligned_channels(in_channels)
        self._aligned_in_channels = in_channels

        self.out_channels = out_channels
        kernel_3d = (1, kernel_size, kernel_size)
        self._kernel_3d = kernel_3d
        self._stride_3d = (1, stride, stride)

        # conv3d pads symmetrically, so asymmetric padding is applied to the input instead.
        top, bottom, left, right = padding
        if top == bottom and left == right:
            self._input_padding = None
            self._padding_3d = (0, top, left)
        else:
            self._input_padding = [(0, 0), (top, bottom), (left, right), (0, 0)]
            self._padding_3d = (0, 0, 0)

        self.conv_config = get_conv3d_config(
            in_channels,
            out_channels,
            kernel_3d,
            ttnn.bfloat16,
            grid_size=ctx.device.compute_with_storage_grid_size(),
            h_factor=ctx.h_factor,
            w_factor=ctx.w_factor,
            T=1,
            H=0,
            W=0,
        )

        d = kernel_size * kernel_size * in_channels
        self.weight = Parameter(total_shape=[d, out_channels], device=ctx.device, pad_value=0)
        self.bias = Parameter(total_shape=[1, out_channels], device=ctx.device, pad_value=0)

        self._ctx = ctx
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            ctx.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4 if is_blackhole() else ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            w = state["weight"]
            if w.dim() == 2:
                w = w.reshape(w.shape[0], w.shape[1], 1, 1)
            w = w.unsqueeze(2)  # [C_out, C_in, kH, kW] → [C_out, C_in, 1, kH, kW]
            weight_tt = ttnn.from_torch(w, dtype=ttnn.bfloat16, pad_value=0)
            prepared = ttnn.experimental.prepare_conv3d_weights(
                weight_tensor=weight_tt,
                C_in_block=self.conv_config.C_in_block,
                device=self._ctx.device,
            )
            state["weight"] = local_device_to_torch(prepared)
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

        if self._input_padding is not None:
            x = ttnn.pad(x, self._input_padding, value=0.0)

        if x.shape[-1] != self._aligned_in_channels:
            x = tensor.pad_single(x, dim=-1, back=self._aligned_in_channels - x.shape[-1])

        n, h, w, c = x.shape
        x = ttnn.reshape(x, (n, 1, h, w, c))

        x = ttnn.experimental.conv3d(
            input_tensor=x,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self._ctx.device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self._kernel_3d,
            stride=self._stride_3d,
            padding=self._padding_3d,
            padding_mode="zeros",
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        # [N, 1, H_out, W_out, C_out] → [N, H_out, W_out, C_out]
        # Tile-aligned channels: reshape is metadata-only (free). Non-aligned (e.g. C=3):
        # fall back to indexing which gathers into a fresh contiguous buffer.
        if self.out_channels % 32 == 0:
            _, _, h2, w2, c2 = x.shape
            x = ttnn.reshape(x, (n, h2, w2, c2))
        else:
            x = x[:, 0, :, :, :]

        return ttnn.to_layout(x, ttnn.TILE_LAYOUT)


class VaeConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int | Sequence[int] = 0,
        tensor_parallel: bool = True,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        # Shard bigger dimension to minimize communication. If both are equal, shard rows to
        # minimize memory requirements.
        out_is_greater = out_channels > in_channels

        top, bottom, left, right = (padding,) * 4 if isinstance(padding, int) else padding
        self._full_padding = (top, bottom, left, right)
        self._kernel_size, self._stride = kernel_size, stride

        # SP halo: on a sharded axis the conv's own padding becomes 0 and a neighbor_pad halo
        # exchange supplies the missing rows/cols from adjacent devices, and zeros at the edges of
        # the image. Channel TP (in_mesh_axis / out_mesh_axis) is orthogonal — runs on a different
        # mesh axis.
        self._h_sharded = ctx.h_factor > 1 and (top > 0 or bottom > 0)
        self._w_sharded = ctx.w_factor > 1 and (left > 0 or right > 0)
        self._needs_neighbor_pad = self._h_sharded or self._w_sharded

        if self._h_sharded:
            self._h_halo = (top, bottom)
            top = bottom = 0
        if self._w_sharded:
            self._w_halo = (left, right)
            left = right = 0

        # Use conv3d when requested and no TP is in play for this conv.
        no_tp = ctx.tp_axis is None or not tensor_parallel
        self._use_conv3d = ctx.use_conv3d and no_tp

        if self._use_conv3d:
            self.inner = _VaeConv2dConv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(top, bottom, left, right),
                ctx=ctx,
            )
        else:
            self.inner = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=(top, bottom, left, right),
                mesh_device=ctx.device,
                in_mesh_axis=ctx.tp_axis if tensor_parallel and not out_is_greater else None,
                out_mesh_axis=ctx.tp_axis if tensor_parallel and out_is_greater else None,
                ccl_manager=ctx.ccl_manager,
            )
        self._ctx = ctx
        ks = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self._a3 = (
            ctx.external_halo
            and self._needs_neighbor_pad
            and not self._use_conv3d
            and ks == 3
            and stride == 1
            and self._full_padding == (1, 1, 1, 1)
        )
        self._a3_refused: set[tuple[int, int]] = set()  # shapes whose external-halo conv failed

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "", "inner")

    def _a3_eligible(self, h: int, w: int) -> bool:
        """The external halo runs on the conv's height-sharded path (every part of a cout split)."""
        inner = self.inner
        key = (h, w, inner.in_channels, inner.out_channels)
        n_split = inner.cout_parts(h, w)
        layout = inner.split_shard_layout if n_split > 1 else inner.shard_layout
        level_key = (*key, n_split) if n_split > 1 else key
        if (h, w) in self._a3_refused or inner._fallback_level.get(level_key, 0) >= 2:
            return False
        padded_key = (*self.padded_input_shape(h, w), inner.in_channels, inner.out_channels)
        return (
            layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
            and key not in inner.no_shard
            and padded_key not in inner.no_shard
        )

    def padded_input_shape(self, h: int, w: int) -> tuple[int, int]:
        """The neighbour-padded conv input shape for a local (h, w); the slice table is keyed on it."""
        if self._h_sharded:
            h += sum(self._h_halo)
        if self._w_sharded:
            w += sum(self._w_halo)
        return h, w

    def _a3_num_slices(self, h: int, w: int):
        inner = self.inner
        slices = inner.slice_params.get(tuple(inner.mesh_device.shape), inner.slice_default)
        return slices.get((*self.padded_input_shape(h, w), inner.in_channels, inner.out_channels))

    def _exchange_boundary(self, x: ttnn.Tensor, prologue=None) -> ttnn.Tensor:
        """The neighbour devices' boundary sticks for this device's border padding, in the ExternalHaloSpec
        layout [top: W + 2][bottom: W + 2][left: H][right: H] (corners in the rows), ROW_MAJOR DRAM, zeros at the image
        edge. One-row / one-column slices (32-column tile bands for the columns, so the activation is never converted
        whole), exchanged along H, then the H-extended columns along W (which carries the corners). With `prologue`
        (scale, shift) the outgoing slices get silu(x * scale + shift) first, so the halo matches the affined
        interior and the missing-neighbour sides stay exact zeros."""
        ccl = self._ctx.ccl_manager
        assert ccl is not None
        n, h, w, c = (int(d) for d in x.shape)
        dram = ttnn.DRAM_MEMORY_CONFIG

        def act(t):
            if prologue is None:
                return t
            t = ttnn.to_layout(t, ttnn.TILE_LAYOUT) if t.layout != ttnn.TILE_LAYOUT else t
            t = ttnn.multiply(t, prologue[0])
            return ttnn.add(t, prologue[1], activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])

        def rm(t):
            return t if t.layout == ttnn.ROW_MAJOR_LAYOUT else ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)

        def exchange(edges, dim, axis):
            e5 = ttnn.unsqueeze(edges, 0)
            out = ccl.neighbor_pad(
                e5,
                dims=[dim],
                pad_left=[1],
                pad_right=[1],
                padding_mode="zeros",
                axes=[axis],
                neighbor_sems=[ccl.get_np_ping_pong_semaphore(axis)],
                num_links=[_get_neighbor_pad_num_links(ccl, e5, dim)],
            )
            return ttnn.squeeze(out, 0)

        # rows: H is not a tiled dim, so one-row slices of a TILE tensor are cheap
        top = rm(act(ttnn.slice(x, [0, 0, 0, 0], [n, 1, w, c])))
        bot = rm(act(ttnn.slice(x, [0, h - 1, 0, 0], [n, h, w, c])))
        if self._h_sharded:
            ph = exchange(ttnn.concat([top, bot], dim=1), 2, self._ctx.h_mesh_axis)  # [n, 4, w, c]
            halo_top = ttnn.slice(ph, [0, 0, 0, 0], [n, 1, w, c])
            halo_bot = ttnn.slice(ph, [0, 3, 0, 0], [n, 4, w, c])
        else:
            halo_top, halo_bot = ttnn.zeros_like(top), ttnn.zeros_like(bot)
        # columns of the H-extended block: tile-aligned 32-column bands, then the edge column
        band = min(32, w)
        lb = rm(act(ttnn.slice(x, [0, 0, 0, 0], [n, h, band, c])))
        rb = rm(act(ttnn.slice(x, [0, 0, w - band, 0], [n, h, w, c])))
        left = ttnn.slice(lb, [0, 0, 0, 0], [n, h, 1, c])
        right = ttnn.slice(rb, [0, 0, band - 1, 0], [n, h, band, c])
        left_ext = ttnn.concat(
            [ttnn.slice(halo_top, [0, 0, 0, 0], [n, 1, 1, c]), left, ttnn.slice(halo_bot, [0, 0, 0, 0], [n, 1, 1, c])],
            dim=1,
        )
        right_ext = ttnn.concat(
            [
                ttnn.slice(halo_top, [0, 0, w - 1, 0], [n, 1, w, c]),
                right,
                ttnn.slice(halo_bot, [0, 0, w - 1, 0], [n, 1, w, c]),
            ],
            dim=1,
        )
        if self._w_sharded:
            pw = exchange(ttnn.concat([left_ext, right_ext], dim=2), 3, self._ctx.w_mesh_axis)  # [n, h + 2, 4, c]
            halo_left = ttnn.slice(pw, [0, 0, 0, 0], [n, h + 2, 1, c])
            halo_right = ttnn.slice(pw, [0, 0, 3, 0], [n, h + 2, 4, c])
        else:
            halo_left, halo_right = ttnn.zeros_like(left_ext), ttnn.zeros_like(right_ext)

        def row(t, r):  # [n, 1, 1, c] corner stick from a column
            return ttnn.slice(t, [0, r, 0, 0], [n, r + 1, 1, c])

        top_seq = ttnn.concat([row(halo_left, 0), halo_top, row(halo_right, 0)], dim=2)  # [n, 1, w + 2, c]
        bot_seq = ttnn.concat([row(halo_left, h + 1), halo_bot, row(halo_right, h + 1)], dim=2)
        left_seq = ttnn.reshape(ttnn.slice(halo_left, [0, 1, 0, 0], [n, h + 1, 1, c]), (n, 1, h, c))
        right_seq = ttnn.reshape(ttnn.slice(halo_right, [0, 1, 0, 0], [n, h + 1, 1, c]), (n, 1, h, c))
        boundary = ttnn.concat([top_seq, bot_seq, left_seq, right_seq], dim=2, memory_config=dram)  # [n, 1, B, c]
        return ttnn.reshape(boundary, (1, 1, n * (2 * (w + 2) + 2 * h), c))

    def a3_takes(self, hw: tuple[int, int]) -> bool:
        """Whether a (h, w) input goes down the external-halo path (TILE input, halo prologue honoured)."""
        return bool(self._a3) and hw not in self._a3_refused and self._a3_eligible(*hw)

    @staticmethod
    def _apply_prologue(x: ttnn.Tensor, prologue) -> ttnn.Tensor:
        """The affine + SiLU as two eltwise passes, for the paths whose conv input is not TILE (no halo prologue)."""
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        t = ttnn.multiply(x, prologue[0])
        return ttnn.add(t, prologue[1], activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])

    def forward(self, x: ttnn.Tensor, prologue=None, *, epilogue_stats=None) -> ttnn.Tensor:
        if self._a3 and self._a3_eligible(int(x.shape[1]), int(x.shape[2])):
            boundary = self._exchange_boundary(x, prologue)
            if self._a3_num_slices(int(x.shape[1]), int(x.shape[2])) not in (None, 1) and x.is_sharded():
                x = ttnn.sharded_to_interleaved(x, ttnn.DRAM_MEMORY_CONFIG)  # the DRAM-sliced conv reads DRAM
            saved = self.inner.padding
            self.inner.padding = self._full_padding  # the conv pads the border itself, from the boundary
            out = None
            try:
                out = self.inner.forward(
                    x,
                    use_persistent_buffer=False,
                    halo_boundary=boundary,
                    slice_shape=self.padded_input_shape(int(x.shape[1]), int(x.shape[2])),
                    epilogue_stats=epilogue_stats,
                    **({"prologue": prologue} if prologue is not None else {}),
                )
            except RuntimeError as e:  # e.g. the conv fell back off the height-sharded L1 path
                self._a3_refused.add((int(x.shape[1]), int(x.shape[2])))
                logger.warning(f"VAE: external halo refused for {tuple(x.shape)}, using NeighborPad: {str(e)[:160]}")
            finally:
                self.inner.padding = saved
                ttnn.deallocate(boundary)
            if out is not None:
                return out
        if (
            prologue is not None
        ):  # no halo prologue off the external-halo path: apply it before the pad (the halo sees the result)
            x = self._apply_prologue(x, prologue)
            prologue = None
        if self._needs_neighbor_pad:
            ccl = self._ctx.ccl_manager
            assert ccl is not None
            # neighbor_pad_async requires ROW_MAJOR; conv handles either layout.
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            # Unsqueeze to matches the neighbor_pad kernel's expected rank for HW exchange. dims
            # shift up by 1.
            x = ttnn.unsqueeze(x, 0)  # [1, N, H_local, W_local, C]
            dims, pad_left, pad_right, axes, sems, links = [], [], [], [], [], []
            if self._h_sharded:
                dims.append(2)
                pad_left.append(self._h_halo[0])
                pad_right.append(self._h_halo[1])
                axes.append(self._ctx.h_mesh_axis)
                sems.append(ccl.get_np_ping_pong_semaphore(self._ctx.h_mesh_axis))
                links.append(_get_neighbor_pad_num_links(ccl, x, 2))
            if self._w_sharded:
                dims.append(3)
                pad_left.append(self._w_halo[0])
                pad_right.append(self._w_halo[1])
                axes.append(self._ctx.w_mesh_axis)
                sems.append(ccl.get_np_ping_pong_semaphore(self._ctx.w_mesh_axis))
                links.append(_get_neighbor_pad_num_links(ccl, x, 3))
            x = ccl.neighbor_pad(
                x,
                dims=dims,
                pad_left=pad_left,
                pad_right=pad_right,
                padding_mode="zeros",
                axes=axes,
                neighbor_sems=sems,
                num_links=links,
            )
            x = ttnn.squeeze(x, 0)  # back to [N, H_local+pad, W_local+pad, C]

        if self._use_conv3d:
            assert epilogue_stats is None, "epilogue statistics need the 2D conv"
            return self.inner.forward(x)
        return self.inner.forward(x, use_persistent_buffer=False, epilogue_stats=epilogue_stats)


class VaeUpsampler(Module):
    # SP-safe by composition: nearest-neighbor upsample is per-device (each pixel → 2x2 block),
    # so a sharded [N, H/h, W/w, C] input becomes a sharded [N, 2H/h, 2W/w, C] output with
    # the same shard layout. The downstream VaeConv2d already handles SP neighbor-padding.
    def __init__(self, *, in_channels: int, out_channels: int, ctx: VaeContext) -> None:
        super().__init__()
        self.conv = VaeConv2d(in_channels, out_channels, kernel_size=3, padding=1, ctx=ctx)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = tensor.upsample(x, scale_factor=2)
        return self.conv.forward(x)


class VaeResnetBlock(Module):
    def __init__(self, *, in_channels: int, out_channels: int, norm: VaeNormDesc, ctx: VaeContext) -> None:
        super().__init__()

        self.norm1 = _norm(norm, num_channels=in_channels, ctx=ctx, activation_fn="silu")
        self.conv1 = VaeConv2d(in_channels, out_channels, kernel_size=3, padding=1, ctx=ctx)
        self.norm2 = _norm(norm, num_channels=out_channels, ctx=ctx, activation_fn="silu")
        self.conv2 = VaeConv2d(out_channels, out_channels, kernel_size=3, padding=1, ctx=ctx)

        self.conv_shortcut = (
            RowParallelLinear(
                in_channels,
                out_channels,
                mesh_axis=ctx.tp_axis,
                mesh_device=ctx.device,
                ccl_manager=ctx.ccl_manager,
            )
            if in_channels != out_channels
            else None
        )
        self._ctx = ctx
        self._a_fused = (
            ctx.norm_fusion and isinstance(self.norm2, _VaeSpatial2dGroupNorm) and not self.conv1._use_conv3d
        )
        self._a_pages: dict[tuple[int, ...], ttnn.Tensor] = {}
        self._a_prologue = self._a_fused and ctx.norm_prologue

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "conv_shortcut.weight" in state:
            state["conv_shortcut.weight"] = state["conv_shortcut.weight"].squeeze(2, 3)

    def _a_parts(self, h: ttnn.Tensor) -> int:
        """conv1's output-channel split on this input (1 = unsplit)."""
        return self.conv1.inner.cout_parts(*self.conv1.padded_input_shape(int(h.shape[1]), int(h.shape[2])))

    def _conv1_fused_norm2(self, h: ttnn.Tensor, as_prologue: bool = False):
        """conv1 with its output statistics in the epilogue, then norm2 as one per-channel affine + SiLU from those
        statistics (gn_affine_from_stats). A split conv writes one page block per output-channel part; the groups split the same way, so each
        part's affine comes from its own pages and the parts are concatenated along the channels."""
        from models.tt_dit.models.vae.gn_stats import gn_affine_from_stats

        ctx = self._ctx
        norm = self.norm2
        c = norm.num_channels
        n_parts = self._a_parts(h)
        c_part = c // n_parts
        grid = ctx.device.compute_with_storage_grid_size()
        cores_max = grid.x * grid.y
        stride = cores_max * 2 * (c_part // 32)  # pages per DRAM slice (per part)
        key = (*(int(d) for d in h.shape), n_parts)
        pages = self._a_pages.get(key)
        if pages is None:
            # zeroed once: every call rewrites the pages of the slices and cores it uses, the others stay zero
            pages = [
                tensor.from_torch(
                    torch.zeros(1, 1, VAE_A_MAX_SLICES * stride * 32, 32),
                    device=ctx.device,
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_axes=[None, None, None, None],
                )
                for _ in range(n_parts)
            ]
            self._a_pages[key] = pages
        stats_arg = (pages[0], stride) if n_parts == 1 else [(pg, stride) for pg in pages]
        out = self.conv1.forward(h, epilogue_stats=stats_arg)
        _, oh, ow, _ = (int(d) for d in out.shape)
        count = oh * ctx.h_factor * ow * ctx.w_factor * (c // norm.num_groups)
        scales, shifts = [], []
        for i in range(n_parts):
            if n_parts == 1:
                gamma, beta = norm.weight.data, norm.bias.data
            else:
                gamma = ttnn.slice(norm.weight.data, [0, 0, 0, i * c_part], [1, 1, 1, (i + 1) * c_part])
                beta = ttnn.slice(norm.bias.data, [0, 0, 0, i * c_part], [1, 1, 1, (i + 1) * c_part])
            sc, sh = gn_affine_from_stats(
                pages[i],
                VAE_A_MAX_SLICES * cores_max,
                c_part,
                norm.num_groups // n_parts,
                gamma,
                beta,
                norm.eps,
                ctx.h_mesh_axis,
                ctx.w_mesh_axis,
                ctx.ccl_manager,
                count,
            )
            scales.append(sc)
            shifts.append(sh)
        scale = scales[0] if n_parts == 1 else ttnn.concat(scales, dim=3)
        shift = shifts[0] if n_parts == 1 else ttnn.concat(shifts, dim=3)
        if out.layout != ttnn.TILE_LAYOUT:
            out = ttnn.to_layout(out, ttnn.TILE_LAYOUT)
        if as_prologue:
            return out, scale, shift
        scaled = ttnn.multiply(out, scale)
        ttnn.deallocate(out)
        res = ttnn.add(scaled, shift, activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU)])
        ttnn.deallocate(scaled)
        return res

    def _conv1_out_shape(self, h: ttnn.Tensor) -> tuple[int, int]:
        """conv1's output (h, w) per device: the 3x3 / pad 1 conv keeps the local size."""
        _, hh, ww, _ = (int(d) for d in h.shape)
        return hh, ww

    def _a_eligible(self, h: ttnn.Tensor) -> bool:
        """The epilogue statistics exist on the height-sharded conv (unsplit or every split part) and not on a conv
        that fell back to no shard layout; the groups must split evenly over the parts."""
        inner = self.conv1.inner
        _, hh, ww, _ = (int(d) for d in h.shape)
        # a ragged DRAM width slice computes junk columns the epilogue would sum: the local width must be tile-aligned
        if ww % 32:
            return False
        hh, ww = self.conv1.padded_input_shape(hh, ww)
        n_parts = inner.cout_parts(hh, ww)
        layout = inner.split_shard_layout if n_parts > 1 else inner.shard_layout
        if layout != ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
            return False
        if self.norm2.num_groups % n_parts or (self.norm2.num_channels // n_parts) % 32:
            return False
        key = (hh, ww, inner.in_channels, inner.out_channels)
        level_key = (*key, n_parts) if n_parts > 1 else key
        return key not in inner.no_shard and inner._fallback_level.get(level_key, 0) < 2

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = x

        h = self.norm1.forward(h)
        if self._a_fused and self._a_eligible(h):
            if self._a_prologue and self.conv2.a3_takes(self._conv1_out_shape(h)):
                out, scale, shift = self._conv1_fused_norm2(h, as_prologue=True)
                h = self.conv2.forward(out, prologue=(scale, shift))
            else:
                h = self._conv1_fused_norm2(h)
                h = self.conv2.forward(h)
        else:
            h = self.conv1.forward(h)
            h = self.norm2.forward(h)
            h = self.conv2.forward(h)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut.forward(x)

        return x + h


class VaeAttention(Module):
    # SDPA chunk sizes keyed by (is_blackhole, num_channels).
    # Resolution priority: map > constructor args > default.
    sdpa_chunk_size_map: Mapping[tuple[bool, int], tuple[int, int]] = MappingProxyType(
        {
            (False, 640): (64, 64),
            (True, 640): (64, 64),
            (False, 1024): (64, 64),
            (True, 1024): (64, 64),
        }
    )
    default_sdpa_chunk_size: tuple[int, int] = (128, 128)
    # Attend only for this chip's query rows instead of the full sequence replicated on every chip; per-row
    # results are identical (full K/V, one head, no mask). Supports H sharding and TP, not W sharding.
    local_queries: bool = False

    def __init__(
        self,
        *,
        num_channels: int,
        norm: VaeNormDesc,
        ctx: VaeContext,
        q_chunk_size: int | None = None,
        k_chunk_size: int | None = None,
    ) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        linear_args = dict(mesh_axis=ctx.tp_axis, mesh_device=ctx.device, ccl_manager=ctx.ccl_manager)

        self._attn_local = ctx.local_attention and ctx.tp_axis is None and (ctx.h_factor > 1 or ctx.w_factor > 1)
        if self._attn_local:
            # its own parameter name: the weight cache keys tensors by module path and layout, and the plain norm's
            # cached ROW_MAJOR 'norm.weight' cannot load into the 2D norm's TILE parameters
            self.norm_local = _norm(norm, num_channels=num_channels, ctx=ctx)
        else:
            # Norm runs on the AG'd full-spatial tensor inside forward(), so the inner norm should NOT
            # AG/partition again. Bypass the _VaeSpatialGroupNorm wrap by handing _norm a TP-only ctx.
            norm_ctx = replace(ctx, h_factor=1, w_factor=1, h_mesh_axis=None, w_mesh_axis=None)
            self.norm = _norm(norm, num_channels=num_channels, ctx=norm_ctx)
        # Fused QKV: one ColParallelLinear with chunks=3 replaces three RowParallelLinear+RS+AG
        # round-trips. Each output is fractured on out_dim (channels) and gets AG'd before SDPA.
        self.to_qkv = ColParallelLinear(num_channels, 3 * num_channels, chunks=3, **linear_args)
        self.to_out = ColParallelLinear(num_channels, num_channels, **linear_args)

        tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        self._tp_factor = tp_factor
        resolved_q_chunk, resolved_k_chunk = self.sdpa_chunk_size_map.get(
            (is_blackhole(), num_channels),
            (
                q_chunk_size if q_chunk_size is not None else self.default_sdpa_chunk_size[0],
                k_chunk_size if k_chunk_size is not None else self.default_sdpa_chunk_size[1],
            ),
        )

        grid_size = ctx.device.compute_with_storage_grid_size()
        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=resolved_q_chunk,
            k_chunk_size=resolved_k_chunk,
        )
        self._sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            ctx.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self._mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            ctx.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager
        self._ctx = ctx

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")
        rename_substate(state, "group_norm", "norm_local" if self._attn_local else "norm")

        # Merge per-tensor to_q/to_k/to_v into a fused to_qkv with per-device Q|K|V interleaving,
        # so that splitting the output dim on tp_axis gives each device a contiguous Q chunk,
        # K chunk, V chunk that ColParallelLinear.chunks=3 can return as three tensors.
        q_w = state.pop("to_q.weight", None)
        k_w = state.pop("to_k.weight", None)
        v_w = state.pop("to_v.weight", None)
        if q_w is not None and k_w is not None and v_w is not None:
            state["to_qkv.weight"] = self._merge_qkv(q_w, k_w, v_w)

        q_b = state.pop("to_q.bias", None)
        k_b = state.pop("to_k.bias", None)
        v_b = state.pop("to_v.bias", None)
        if q_b is not None and k_b is not None and v_b is not None:
            state["to_qkv.bias"] = self._merge_qkv(q_b, k_b, v_b)

    def _merge_qkv(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Merge three [out, in] (or [out]) tensors into [3*out, in] (or [3*out]) with
        per-device Q|K|V interleaving on the output dim."""
        is_1d = q.dim() == 1
        if is_1d:
            q = q.unsqueeze(-1)
            k = k.unsqueeze(-1)
            v = v.unsqueeze(-1)
        out, inn = q.shape
        n_dev = self._tp_factor
        assert out % n_dev == 0, f"out_features {out} must be divisible by tp_factor {n_dev}"
        head_dim = out // n_dev

        # [out, in] → [n_dev, head_dim, in]
        q = q.reshape(n_dev, head_dim, inn)
        k = k.reshape(n_dev, head_dim, inn)
        v = v.reshape(n_dev, head_dim, inn)

        # Stack Q|K|V per device: [n_dev, 3, head_dim, in] → [3*out, in]
        qkv = torch.stack([q, k, v], dim=1).reshape(n_dev * 3 * head_dim, inn)

        return qkv.squeeze(-1) if is_1d else qkv

    def _forward_local(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Decode item 1: norm and Q on the local shard; K / V on the gathered flat sequence (own tokens' order)."""
        identity = x
        n, h_l, w_l, c = (int(d) for d in x.shape)
        ctx = self._ctx
        ccl = ctx.ccl_manager
        assert ccl is not None
        xn = self.norm_local.forward(x)  # [n, h_l, w_l, c], statistics over the mesh
        if xn.layout != ttnn.TILE_LAYOUT:
            xn = ttnn.to_layout(xn, ttnn.TILE_LAYOUT)
        local = ttnn.reshape(xn, (n, 1, h_l * w_l, c))  # a view: w_l is tile-aligned on the 2D decode
        # the flat sequences of every device, H shards first then W shards; the token order is a fixed permutation of
        # the raster, the same for K, V and the queries of every device
        full = local
        if ctx.h_factor > 1:
            full = ccl.all_gather(full, dim=2, mesh_axis=ctx.h_mesh_axis, use_hyperparams=True)
        if ctx.w_factor > 1:
            full = ccl.all_gather(full, dim=2, mesh_axis=ctx.w_mesh_axis, use_hyperparams=True)
        q, _k, _v = self.to_qkv.forward(local, compute_kernel_config=self._mm_compute_kernel_config)
        ttnn.deallocate(_k)
        ttnn.deallocate(_v)
        _q, k, v = self.to_qkv.forward(full, compute_kernel_config=self._mm_compute_kernel_config)
        ttnn.deallocate(_q)
        if full is not local:
            ttnn.deallocate(full)
        ttnn.deallocate(local)
        program_config = self._sdpa_program_config
        rows = int(q.shape[2])
        if rows % program_config.q_chunk_size:
            program_config = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=program_config.compute_with_storage_grid_size,
                q_chunk_size=max(d for d in range(32, program_config.q_chunk_size + 1, 32) if rows % d == 0),
                k_chunk_size=program_config.k_chunk_size,
                exp_approx_mode=program_config.exp_approx_mode,
            )
        o = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )
        for t in (q, k, v):
            ttnn.deallocate(t)
        o = self.to_out.forward(o, compute_kernel_config=self._mm_compute_kernel_config)
        return ttnn.reshape(o, (n, h_l, w_l, c)) + identity

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._attn_local:
            return self._forward_local(x)
        identity = x  # [N, H/h, W/w, C/tp] when SP+TP active; identity captured pre-AG

        # SDPA over the full spatial extent → AG H+W once (no-op when SP inactive).
        x = _all_gather_hw(self._ctx, x)

        n, h, w, _ = x.shape  # h, w are full-spatial after AG

        # convert to 1d sequence and insert head dimension; there is only one head
        x = x.reshape([n, 1, h * w, -1])

        x = self.norm.forward(x)

        # ColParallelLinear expects replicated input on K. AG manually here (rather than
        # passing parallel_config and letting the linear gather) because on Ring topology
        # that would route through all_gather_minimal_matmul_async, whose K_block_size
        # constraint VAE mid-block shapes (e.g. K=512 / tp=8 → 2 tiles) don't satisfy.
        if self._ccl_manager is not None and self._tp_axis is not None:
            x = self._ccl_manager.all_gather(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        # Fused QKV matmul (1 kernel) returns three TP-fractured outputs.
        q, k, v = self.to_qkv.forward(x, compute_kernel_config=self._mm_compute_kernel_config)

        # AG each to full channels per device (single-head SDPA needs full head_dim = C).
        if self._ccl_manager is not None and self._tp_axis is not None:
            q = self._ccl_manager.all_gather(q, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            k = self._ccl_manager.all_gather(k, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            v = self._ccl_manager.all_gather(v, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        del x

        program_config = self._sdpa_program_config
        h_local = h // self._ctx.h_factor
        tp_split = self._tp_factor > 1 and h_local % self._tp_factor == 0 and h_local // self._tp_factor * w % 32 == 0
        # 2D SP: the W mesh axis splits the local query rows like a TP axis would; the rows are gathered back and
        # re-partitioned by columns after the projection
        w_factor = self._ctx.w_factor
        w_split = w_factor > 1 and h_local % w_factor == 0 and h_local // w_factor * w % 32 == 0
        local = self.local_queries and (
            tp_split or w_split or (w_factor == 1 and self._ctx.h_factor > 1 and h_local * w % 32 == 0)
        )
        if local:
            q = ttnn.to_layout(q, ttnn.ROW_MAJOR_LAYOUT)
            if self._ctx.h_factor > 1:
                q = ttnn.mesh_partition(q, dim=2, cluster_axis=self._ctx.h_mesh_axis, memory_config=q.memory_config())
            if tp_split:
                q = ttnn.mesh_partition(q, dim=2, cluster_axis=self._tp_axis, memory_config=q.memory_config())
            if w_split:
                q = ttnn.mesh_partition(q, dim=2, cluster_axis=self._ctx.w_mesh_axis, memory_config=q.memory_config())
            q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
            rows = q.shape[2]
            if rows % program_config.q_chunk_size:
                program_config = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=program_config.compute_with_storage_grid_size,
                    q_chunk_size=max(d for d in range(32, program_config.q_chunk_size + 1, 32) if rows % d == 0),
                    k_chunk_size=program_config.k_chunk_size,
                    exp_approx_mode=program_config.exp_approx_mode,
                )

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=program_config,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        if local:
            if tp_split:
                x = self._ccl_manager.all_gather(x, dim=2, mesh_axis=self._tp_axis, use_hyperparams=True)
            x = self.to_out.forward(x, compute_kernel_config=self._mm_compute_kernel_config)
            if w_split:
                x = self._ccl_manager.all_gather(x, dim=2, mesh_axis=self._ctx.w_mesh_axis, use_hyperparams=True)
                x = ttnn.to_layout(x.reshape([n, h_local, w, -1]), ttnn.ROW_MAJOR_LAYOUT)
                x = ttnn.mesh_partition(x, dim=2, cluster_axis=self._ctx.w_mesh_axis, memory_config=x.memory_config())
                return ttnn.to_layout(x, ttnn.TILE_LAYOUT) + identity
            return x.reshape([n, h_local, w, -1]) + identity

        x = self.to_out.forward(x, compute_kernel_config=self._mm_compute_kernel_config)

        # convert back to 2d
        x = x.reshape([n, h, w, -1])

        # Re-partition H+W back to local so the residual add aligns with identity (no-op when SP inactive).
        x = _partition_hw(self._ctx, x)

        return x + identity


class VaeMidBlock(Module):
    def __init__(
        self,
        *,
        num_channels: int,
        num_layers: int = 1,
        norm: VaeNormDesc,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            VaeResnetBlock(in_channels=num_channels, out_channels=num_channels, norm=norm, ctx=ctx)
            for _ in range(num_layers + 1)
        )
        self.attentions = ModuleList(
            VaeAttention(num_channels=num_channels, norm=norm, ctx=ctx) for _ in range(num_layers)
        )

        self._num_layers = num_layers

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for i in range(self._num_layers + 1):
            x = self.resnets[i].forward(x)
            if i < self._num_layers:
                x = self.attentions[i].forward(x)

        return x


class VaeUpBlock(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        upsampler_out_channels: int | None = None,
        num_layers: int,
        upsample: bool,
        norm: VaeNormDesc,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            VaeResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm=norm,
                ctx=ctx,
            )
            for i in range(num_layers)
        )

        self.upsampler = (
            VaeUpsampler(in_channels=out_channels, out_channels=upsampler_out_channels or out_channels, ctx=ctx)
            if upsample
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "upsamplers.0", "upsampler")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for block in self.resnets:
            x = block.forward(x)

        if self.upsampler is not None:
            x = self.upsampler.forward(x)

        return x


class VaeDownsampler(Module):
    def __init__(self, *, num_channels: int, ctx: VaeContext) -> None:
        super().__init__()
        self.conv = VaeConv2d(num_channels, num_channels, kernel_size=3, stride=2, padding=(0, 1, 0, 1), ctx=ctx)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.conv.forward(x)


class VaeDownBlock(Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        downsample: bool,
        norm: VaeNormDesc,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            VaeResnetBlock(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                norm=norm,
                ctx=ctx,
            )
            for i in range(num_layers)
        )

        self.downsampler = VaeDownsampler(num_channels=out_channels, ctx=ctx) if downsample else None

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        for block in self.resnets:
            x = block.forward(x)

        if self.downsampler is not None:
            x = self.downsampler.forward(x)

        return x


class _VaeSpatialGroupNorm(Module):
    """GroupNorm for spatially sharded VAE activations.

    ``DistributedGroupNorm`` all-gathers Welford stats on one mesh axis (H if H is
    sharded, otherwise W). Activations stay sharded on that axis. If both H and W
    are sharded, activations are all-gathered on the *other* spatial dim only so the
    fused stats reduction still sees the full spatial extent, then re-partitioned.

    Channel TP is preserved via ``mesh_axis=ctx.tp_axis``. HiFi4 matches Wan/Mochi norms.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float,
        ctx: VaeContext,
        activation_fn: str | None = None,
    ) -> None:
        super().__init__()
        h_sp = ctx.h_factor > 1
        w_sp = ctx.w_factor > 1
        # Stats AG prefers H when both are sharded (matches Flux2 H+TP).
        cluster_axis = ctx.h_mesh_axis if h_sp else ctx.w_mesh_axis
        # 2-D SP: gather the non-cluster spatial dim of the activations, not both.
        self._gather_w = h_sp and w_sp
        self.inner = DistributedGroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            mesh_axis=ctx.tp_axis,
            mesh_device=ctx.device,
            cluster_axis=cluster_axis,
            ccl_manager=ctx.ccl_manager,
            activation_fn=activation_fn,
        )
        self._ctx = ctx
        self._compute_kernel_config = ttnn.init_device_compute_kernel_config(
            ctx.device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "", "inner")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        if self._gather_w:
            assert self._ctx.ccl_manager is not None
            x = self._ctx.ccl_manager.all_gather(x, dim=2, mesh_axis=self._ctx.w_mesh_axis, use_hyperparams=True)

        x = self.inner.forward(x, compute_kernel_config=self._compute_kernel_config)

        if self._gather_w:
            original_layout = x.layout
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.mesh_partition(x, dim=2, cluster_axis=self._ctx.w_mesh_axis, memory_config=x.memory_config())
            if original_layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, original_layout)

        return x


class _VaeSpatial2dGroupNorm(Module):
    """GroupNorm on an activation sharded over both mesh axes with full channels per device.

    The fused distributed op gathers its statistics over one mesh axis only. Here the per-group sums are formed
    locally, all-gathered over both axes (h x w x G scalars) and applied as one per-channel affine pass with the
    activation folded into the add, like the MiniMax-H3 encoder norm.
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        *,
        eps: float,
        ctx: VaeContext,
        activation_fn: str | None = None,
    ) -> None:
        super().__init__()
        assert ctx.tp_axis is None, "2D spatial GroupNorm expects full channels per device"
        assert ctx.ccl_manager is not None
        assert num_channels % num_groups == 0
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self._ctx = ctx
        self._fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if activation_fn == "silu" else None
        self.activation_fn = None if activation_fn == "silu" else activation_fn
        self.weight = Parameter(total_shape=[1, 1, 1, num_channels], device=ctx.device)
        self.bias = Parameter(total_shape=[1, 1, 1, num_channels], device=ctx.device)
        # fp32: it contracts the fp32 channel sums
        self.to_groups = Parameter(total_shape=[1, 1, num_channels, num_groups], device=ctx.device, dtype=ttnn.float32)
        self.from_groups = Parameter(total_shape=[1, 1, num_groups, num_channels], device=ctx.device)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        state["weight"] = state["weight"].reshape(1, 1, 1, self.num_channels)
        state["bias"] = state["bias"].reshape(1, 1, 1, self.num_channels)
        per_group = self.num_channels // self.num_groups
        selector = torch.zeros(self.num_channels, self.num_groups)
        for group in range(self.num_groups):
            selector[group * per_group : (group + 1) * per_group, group] = 1.0
        state["to_groups"] = selector.reshape(1, 1, self.num_channels, self.num_groups)
        state["from_groups"] = selector.t().contiguous().reshape(1, 1, self.num_groups, self.num_channels)

    def _group_sums(self, x_flat: ttnn.Tensor) -> ttnn.Tensor:
        """(N, 1, HW_local, C) -> per-group sums over every device, (N, 1, 1, G)."""
        sums = ttnn.matmul(ttnn.sum(x_flat, dim=2, keepdim=True), self.to_groups.data)
        ccl = self._ctx.ccl_manager
        sums = ccl.all_gather(sums, dim=1, mesh_axis=self._ctx.h_mesh_axis, use_hyperparams=False)
        sums = ccl.all_gather(sums, dim=1, mesh_axis=self._ctx.w_mesh_axis, use_hyperparams=False)
        return ttnn.sum(sums, dim=1, keepdim=True)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        n, h, w, c = (int(d) for d in x.shape)
        if x.layout != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x = ttnn.reshape(x, (n, 1, h * w, c))
        scale = 1.0 / (c // self.num_groups * h * w * self._ctx.h_factor * self._ctx.w_factor)
        x32 = ttnn.typecast(x, ttnn.float32)
        mean_g = ttnn.multiply(self._group_sums(x32), scale)
        sq = ttnn.multiply(x32, x32)
        ttnn.deallocate(x32)
        var_g = ttnn.subtract(ttnn.multiply(self._group_sums(sq), scale), ttnn.multiply(mean_g, mean_g))
        ttnn.deallocate(sq)
        inv_g = ttnn.typecast(ttnn.rsqrt(ttnn.add(var_g, self.eps)), x.dtype)
        gamma = ttnn.multiply(ttnn.matmul(inv_g, self.from_groups.data), self.weight.data)
        mean_c = ttnn.matmul(ttnn.typecast(mean_g, x.dtype), self.from_groups.data)
        beta = ttnn.subtract(self.bias.data, ttnn.multiply(mean_c, gamma))
        scaled = ttnn.multiply(x, gamma)
        if self._fused_activation is not None:
            out = ttnn.add(scaled, beta, activations=[self._fused_activation])
        else:
            out = _apply_activation_fn(ttnn.add(scaled, beta), self.activation_fn)
        ttnn.deallocate(scaled)
        return ttnn.reshape(out, (n, h, w, c))


def _norm(
    norm: VaeNormDesc,
    num_channels: int,
    *,
    ctx: VaeContext,
    activation_fn: str | None = None,
) -> GroupNorm | _VaeSpatialGroupNorm | _VaeSpatial2dGroupNorm | VaeRmsNorm:
    """Build the norm module.

    ``activation_fn`` is applied after the norm; the module folds it into the fused GN op
    when that path is used and runs it separately otherwise. Callers just call ``forward``.
    """
    if isinstance(norm, VaeNormDescGroup):
        if ctx.h_factor > 1 and ctx.w_factor > 1 and ctx.tp_axis is None:
            return _VaeSpatial2dGroupNorm(
                num_groups=norm.num_groups,
                num_channels=num_channels,
                eps=norm.eps,
                ctx=ctx,
                activation_fn=activation_fn,
            )
        # SP-active: fused DistributedGroupNorm (stats AG on one spatial axis).
        if ctx.h_factor > 1 or ctx.w_factor > 1:
            return _VaeSpatialGroupNorm(
                num_groups=norm.num_groups,
                num_channels=num_channels,
                eps=norm.eps,
                ctx=ctx,
                activation_fn=activation_fn,
            )
        return GroupNorm(
            num_groups=norm.num_groups,
            num_channels=num_channels,
            eps=norm.eps,
            mesh_axis=ctx.tp_axis,
            mesh_device=ctx.device,
            activation_fn=activation_fn,
        )
    if isinstance(norm, VaeNormDescRms):
        # VaeRmsNorm normalizes per spatial position over the channel dim → SP-safe as-is.
        return VaeRmsNorm(num_channels=num_channels, eps=norm.eps, ctx=ctx, activation_fn=activation_fn)

    msg = f"invalid VaeNormDesc: {norm}"
    raise ValueError(msg)


def fold_quant_conv_mean(
    state: dict[str, torch.Tensor],
    *,
    conv_out_prefix: str,
    quant_conv_prefix: str,
    latent_channels: int,
) -> None:
    """Fold the mean half of a KL VAE's 1x1 ``quant_conv`` into the encoder's ``conv_out``.

    The encoder then returns the mean of the latent distribution directly. ``quant_conv`` is
    removed from ``state``.
    """
    quant_weight = state.pop(f"{quant_conv_prefix}weight").flatten(1).float()[:latent_channels]
    quant_bias = state.pop(f"{quant_conv_prefix}bias").float()[:latent_channels]
    out_weight = state[f"{conv_out_prefix}weight"].float()
    out_bias = state[f"{conv_out_prefix}bias"].float()
    state[f"{conv_out_prefix}weight"] = torch.einsum("oq,qihw->oihw", quant_weight, out_weight)
    state[f"{conv_out_prefix}bias"] = quant_weight @ out_bias + quant_bias

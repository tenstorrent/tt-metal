# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

import ttnn
from models.common.utility_functions import is_blackhole

from ..layers.conv2d import Conv2d
from ..layers.linear import ColParallelLinear, RowParallelLinear
from ..layers.module import Module, ModuleList, Parameter
from ..layers.normalization import DistributedRMSNorm, GroupNorm, RMSNorm
from ..parallel.manager import CCLManager
from ..utils import tensor
from ..utils.conv3d import get_conv3d_config
from ..utils.substate import rename_substate
from ..utils.tensor import local_device_to_torch


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
        x = ctx.ccl_manager.all_gather_persistent_buffer(x, dim=1, mesh_axis=ctx.h_mesh_axis)
    if ctx.w_factor > 1:
        x = ctx.ccl_manager.all_gather_persistent_buffer(x, dim=2, mesh_axis=ctx.w_mesh_axis)
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
    def __init__(self, num_channels: int, *, eps: float, ctx: VaeContext) -> None:
        super().__init__()

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
            bs, h, w, c = x.shape
            x = ttnn.reshape(x, [1, 1, bs * h * w, c])
            x = self.inner.forward(x)
            return ttnn.reshape(x, [bs, h, w, c])

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
        return x * (norm * self.gamma.data)


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
        h_pad: int,
        w_pad: int,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        self.out_channels = out_channels
        kernel_3d = (1, kernel_size, kernel_size)
        self._kernel_3d = kernel_3d
        self._padding_3d = (0, h_pad, w_pad)

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

        self._tile_aligned = (in_channels % 32 == 0) and (out_channels % 32 == 0)
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
        n, h, w, c = x.shape
        if x.layout != ttnn.ROW_MAJOR_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (n, 1, h, w, c))

        x = ttnn.experimental.conv3d(
            input_tensor=x,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data,
            device=self._ctx.device,
            config=self.conv_config,
            output_channels=self.out_channels,
            kernel_size=self._kernel_3d,
            stride=(1, 1, 1),
            padding=self._padding_3d,
            padding_mode="zeros",
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        # [N, 1, H_out, W_out, C_out] → [N, H_out, W_out, C_out]
        # Tile-aligned channels: reshape is metadata-only (free). Non-aligned (e.g. C=3):
        # fall back to indexing which gathers into a fresh contiguous buffer.
        if self._tile_aligned:
            _, _, h2, w2, c2 = x.shape
            return ttnn.reshape(x, (n, h2, w2, c2))
        return x[:, 0, :, :, :]


class VaeConv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int = 0,
        tensor_parallel: bool = True,
        ctx: VaeContext,
    ) -> None:
        super().__init__()

        # Shard bigger dimension to minimize communication. If both are equal, shard rows to
        # minimize memory requirements.
        out_is_greater = out_channels > in_channels

        # SP halo: for 3×3 padding=1 convs on a sharded axis, the conv's own padding becomes 0
        # and a neighbor_pad halo exchange supplies the missing rows/cols from adjacent devices.
        # Channel TP (in_mesh_axis / out_mesh_axis) is orthogonal — runs on a different mesh axis.
        is_padded_3x3 = kernel_size == 3 and padding == 1
        self._h_sharded = ctx.h_factor > 1 and is_padded_3x3
        self._w_sharded = ctx.w_factor > 1 and is_padded_3x3
        self._needs_neighbor_pad = self._h_sharded or self._w_sharded

        h_pad = 0 if self._h_sharded else padding
        w_pad = 0 if self._w_sharded else padding
        actual_padding = (h_pad, w_pad) if (h_pad != w_pad) else h_pad

        # Use conv3d when requested and no TP is in play for this conv.
        no_tp = ctx.tp_axis is None or not tensor_parallel
        self._use_conv3d = ctx.use_conv3d and no_tp

        if self._use_conv3d:
            self.inner = _VaeConv2dConv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                h_pad=h_pad,
                w_pad=w_pad,
                ctx=ctx,
            )
        else:
            self.inner = Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=actual_padding,
                mesh_device=ctx.device,
                in_mesh_axis=ctx.tp_axis if tensor_parallel and not out_is_greater else None,
                out_mesh_axis=ctx.tp_axis if tensor_parallel and out_is_greater else None,
                ccl_manager=ctx.ccl_manager,
            )
        self._ctx = ctx
        self._padding = padding

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "", "inner")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        if self._needs_neighbor_pad:
            ccl = self._ctx.ccl_manager
            assert ccl is not None
            # neighbor_pad_async requires ROW_MAJOR; conv handles either layout.
            if x.layout != ttnn.ROW_MAJOR_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            # Squeeze the leading batch dim (N=1 for VAE) — matches vae_flux2_new and the
            # neighbor_pad kernel's expected rank for HW exchange. dims shift down by 1.
            x = ttnn.squeeze(x, 0)  # [H_local, W_local, C]
            dims, pad_left, pad_right, axes, sems, links = [], [], [], [], [], []
            if self._h_sharded:
                dims.append(0)
                pad_left.append(self._padding)
                pad_right.append(self._padding)
                axes.append(self._ctx.h_mesh_axis)
                sems.append(ccl.get_np_ping_pong_semaphore(self._ctx.h_mesh_axis))
                links.append(_get_neighbor_pad_num_links(ccl, x, 0))
            if self._w_sharded:
                dims.append(1)
                pad_left.append(self._padding)
                pad_right.append(self._padding)
                axes.append(self._ctx.w_mesh_axis)
                sems.append(ccl.get_np_ping_pong_semaphore(self._ctx.w_mesh_axis))
                links.append(_get_neighbor_pad_num_links(ccl, x, 1))
            x = ccl.neighbor_pad_persistent_buffer(
                x,
                dims=dims,
                pad_left=pad_left,
                pad_right=pad_right,
                padding_mode="zeros",
                axes=axes,
                neighbor_sems=sems,
                num_links=links,
            )
            x = ttnn.unsqueeze(x, 0)  # back to [N=1, H_local+pad, W_local+pad, C]

        if self._use_conv3d:
            result = self.inner.forward(x)
            if result.layout != ttnn.TILE_LAYOUT:
                result = ttnn.to_layout(result, ttnn.TILE_LAYOUT)
            return result
        return self.inner.forward(x, use_persistent_buffer=False)


class VaeUpsampler(Module):
    # SP-safe by composition: nearest-neighbor upsample is per-device (each pixel → 2×2 block),
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

        self.norm1 = _norm(norm, num_channels=in_channels, ctx=ctx)
        self.conv1 = VaeConv2d(in_channels, out_channels, kernel_size=3, padding=1, ctx=ctx)
        self.norm2 = _norm(norm, num_channels=out_channels, ctx=ctx)
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

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "conv_shortcut.weight" in state:
            state["conv_shortcut.weight"] = state["conv_shortcut.weight"].squeeze(2, 3)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = x

        h = self.norm1.forward(h)
        h = ttnn.silu(h)
        h = self.conv1.forward(h)

        h = self.norm2.forward(h)
        h = ttnn.silu(h)
        h = self.conv2.forward(h)

        if self.conv_shortcut is not None:
            x = self.conv_shortcut.forward(x)

        return x + h


class VaeAttention(Module):
    # SDPA chunk sizes keyed by (is_blackhole, h_factor, w_factor, tp_factor). Empty by default;
    # callers populate per-config tuning. Resolution priority: map > constructor args > default.
    sdpa_chunk_size_map: dict[tuple, tuple[int, int]] = {}
    default_sdpa_chunk_size: tuple[int, int] = (128, 128)

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
            (is_blackhole(), ctx.h_factor, ctx.w_factor, tp_factor),
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
            exp_approx_mode=False,
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
        rename_substate(state, "group_norm", "norm")

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

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        identity = x  # [N, H/h, W/w, C/tp] when SP+TP active; identity captured pre-AG

        # SDPA over the full spatial extent → AG H+W once (no-op when SP inactive).
        x = _all_gather_hw(self._ctx, x)

        x = self.norm.forward(x)

        # ColParallelLinear expects replicated input on K. AG manually here (rather than
        # passing parallel_config and letting the linear gather) because on Ring topology
        # that would route through all_gather_minimal_matmul_async, whose K_block_size
        # constraint VAE mid-block shapes (e.g. K=512 / tp=8 → 2 tiles) don't satisfy.
        if self._ccl_manager is not None:
            x = self._ccl_manager.all_gather(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        # Fused QKV matmul (1 kernel) returns three TP-fractured outputs.
        q, k, v = self.to_qkv.forward(x, compute_kernel_config=self._mm_compute_kernel_config)

        # AG each to full channels per device (single-head SDPA needs full head_dim = C).
        if self._ccl_manager is not None:
            q = self._ccl_manager.all_gather(q, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            k = self._ccl_manager.all_gather(k, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            v = self._ccl_manager.all_gather(v, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        del x

        n, h, w, c = q.shape  # h, w are full-spatial after AG

        # convert to 1d sequence and insert head dimension; there is only one head
        q = q.reshape([n, 1, h * w, c])
        k = k.reshape([n, 1, h * w, c])
        v = v.reshape([n, 1, h * w, c])

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=self._sdpa_program_config,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

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


class _VaeSpatialGroupNorm(Module):
    """GroupNorm wrapper that AGs sharded H/W for correct stats and repartitions after.

    Uses HiFi4 compute config (matching Wan/Mochi convention for norms) and trims any
    tile-padding rows added by mesh_partition before computing statistics, restoring them
    after so that _partition_hw receives the same padded shape as before.
    Channel TP is preserved on the inner ``GroupNorm`` via ``mesh_axis=ctx.tp_axis``.
    """

    def __init__(self, num_groups: int, num_channels: int, *, eps: float, ctx: VaeContext) -> None:
        super().__init__()
        self.inner = GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            mesh_axis=ctx.tp_axis,
            mesh_device=ctx.device,
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
        # logical_h_full: unpadded full-H = local H * h_factor. Used to detect and strip
        # zero-padding rows that mesh_partition may have added for tile alignment.
        logical_h_full = x.shape[1] * self._ctx.h_factor

        x = _all_gather_hw(self._ctx, x)  # → [N, H_gathered, W, C], TILE_LAYOUT

        padded_h = x.shape[1]
        needs_trim = padded_h > logical_h_full
        if needs_trim:
            x = x[:, :logical_h_full, :, :]

        x = self.inner.forward(x, compute_kernel_config=self._compute_kernel_config)

        # Restore padding rows so _partition_hw receives the same padded shape.
        if needs_trim:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
            x = ttnn.pad(x, [(0, 0), (0, padded_h - logical_h_full), (0, 0), (0, 0)], value=0.0)

        return _partition_hw(self._ctx, x)


def _norm(norm: VaeNormDesc, num_channels: int, *, ctx: VaeContext) -> GroupNorm | _VaeSpatialGroupNorm | VaeRmsNorm:
    if isinstance(norm, VaeNormDescGroup):
        # SP-active: wrap to AG H+W for correct stats; SP-inactive: plain GroupNorm.
        if ctx.h_factor > 1 or ctx.w_factor > 1:
            return _VaeSpatialGroupNorm(
                num_groups=norm.num_groups,
                num_channels=num_channels,
                eps=norm.eps,
                ctx=ctx,
            )
        return GroupNorm(
            num_groups=norm.num_groups,
            num_channels=num_channels,
            eps=norm.eps,
            mesh_axis=ctx.tp_axis,
            mesh_device=ctx.device,
        )
    if isinstance(norm, VaeNormDescRms):
        # VaeRmsNorm normalizes per spatial position over the channel dim → SP-safe as-is.
        return VaeRmsNorm(num_channels=num_channels, eps=norm.eps, ctx=ctx)

    msg = f"invalid VaeNormDesc: {norm}"
    raise ValueError(msg)

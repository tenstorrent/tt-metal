# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from ..layers.conv2d import Conv2d
from ..layers.linear import ColParallelLinear, RowParallelLinear
from ..layers.module import Module, ModuleList, Parameter
from ..layers.normalization import DistributedRMSNorm, GroupNorm, RMSNorm
from ..parallel.manager import CCLManager
from ..utils import tensor
from ..utils.substate import rename_substate

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class VaeContext:
    device: ttnn.MeshDevice
    tp_axis: int | None
    ccl_manager: CCLManager | None


@dataclass(frozen=True)
class VaeNormDescRms:
    eps: float


@dataclass(frozen=True)
class VaeNormDescGroup:
    eps: float
    num_groups: int


VaeNormDesc = VaeNormDescRms | VaeNormDescGroup


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

        self.inner = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            mesh_device=ctx.device,
            in_mesh_axis=ctx.tp_axis if tensor_parallel and not out_is_greater else None,
            out_mesh_axis=ctx.tp_axis if tensor_parallel and out_is_greater else None,
            ccl_manager=ctx.ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "", "inner")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.inner.forward(x, use_persistent_buffer=False)


class VaeUpsampler(Module):
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
    def __init__(self, *, num_channels: int, norm: VaeNormDesc, ctx: VaeContext) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        linear_args = dict(mesh_axis=ctx.tp_axis, mesh_device=ctx.device, ccl_manager=ctx.ccl_manager)

        self.norm = _norm(norm, num_channels=num_channels, ctx=ctx)
        self.to_q = RowParallelLinear(num_channels, num_channels, **linear_args)
        self.to_k = RowParallelLinear(num_channels, num_channels, **linear_args)
        self.to_v = RowParallelLinear(num_channels, num_channels, **linear_args)
        self.to_out = ColParallelLinear(num_channels, num_channels, **linear_args)

        grid_size = ctx.device.compute_with_storage_grid_size()
        self._sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=grid_size,
            q_chunk_size=128,
            k_chunk_size=128,
            exp_approx_mode=False,
        )
        self._sdpa_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "to_out.0", "to_out")
        rename_substate(state, "group_norm", "norm")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        identity = x

        x = self.norm.forward(x)

        # The call to reduces-scatter in RowParallelLinear writes to persistent buffer so we need to
        # perform all-gather before the next invocation.
        q = self.to_q.forward(x)
        if self._ccl_manager is not None:
            q = self._ccl_manager.all_gather(q, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        k = self.to_k.forward(x)
        if self._ccl_manager is not None:
            k = self._ccl_manager.all_gather(k, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        v = self.to_v.forward(x)
        if self._ccl_manager is not None:
            v = self._ccl_manager.all_gather(v, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        del x

        n, h, w, c = q.shape

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

        x = self.to_out.forward(x)

        # convert back to 2d
        x = x.reshape([n, h, w, -1])

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


def _norm(norm: VaeNormDesc, num_channels: int, *, ctx: VaeContext) -> GroupNorm | VaeRmsNorm:
    if isinstance(norm, VaeNormDescGroup):
        return GroupNorm(
            num_groups=norm.num_groups,
            num_channels=num_channels,
            eps=norm.eps,
            mesh_axis=ctx.tp_axis,
            mesh_device=ctx.device,
        )
    if isinstance(norm, VaeNormDescRms):
        return VaeRmsNorm(num_channels=num_channels, eps=norm.eps, ctx=ctx)

    msg = f"invalid VaeNormDesc: {norm}"
    raise ValueError(msg)

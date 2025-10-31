# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

import ttnn

from ...layers.conv2d import Conv2d
from ...layers.linear import ColParallelLinear, Linear, RowParallelLinear, prepare_chunked_linear_output
from ...layers.module import Module, ModuleList, Parameter
from ...parallel.config import VAEParallelConfig
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.substate import pop_substate, rename_substate

if TYPE_CHECKING:
    from collections.abc import Sequence

    import torch


@dataclass
class QwenImageVaeContext:
    tp_axis: int | None
    device: ttnn.MeshDevice
    ccl_manager: CCLManager | None


class QwenImageConv(Module):
    """Qwen-Image causal convolution without temporal dimension.

    The original QwenImage VAE supports video so the convolution is three-dimensional. Since this is
    not needed for image generation, the temporal dimension is removed here.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        padding: int = 0,
        tensor_parallel: bool = True,
        ctx: QwenImageVaeContext,
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
        # remove temporal dimension and rename
        if "weight" in state:
            state["inner.weight"] = state.pop("weight")[:, :, -1, :, :]
        if "bias" in state:
            state["inner.bias"] = state.pop("bias")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return self.inner.forward(x)


class QwenImageRmsNorm(Module):
    def __init__(self, dim: int, *, eps: float = 1e-12, ctx: QwenImageVaeContext) -> None:
        super().__init__()

        # Using DistributedRMSNorm leads to a drop in accuracy: RMSE increases by a factor of 8
        # from RMSE = 0.022 * σ to RMSE = 0.18 * σ. So we implement the operation here directly.

        # self.norm = (
        #     DistributedRMSNorm(
        #         dim,
        #         norm_eps=eps,
        #         bias=False,
        #         mesh_axis=ctx.tp_axis,
        #         mesh_device=ctx.device,
        #         ccl_manager=ctx.ccl_manager,
        #     )
        #     if ctx.tp_axis is not None
        #     else RMSNorm(
        #         dim,
        #         norm_eps=eps,
        #         bias=False,
        #         mesh_device=ctx.device,
        #     )
        # )

        self.gamma = Parameter(total_shape=[dim], mesh_axes=[ctx.tp_axis], device=ctx.device)

        self.dim = dim
        self.eps = eps

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager
        self._device = ctx.device
        self._tp_axis_size = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        # if "gamma" in state:
        #     state["norm.weight"] = state.pop("gamma").reshape([-1])

        if "gamma" in state:
            state["gamma"] = state["gamma"].reshape([-1])

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # return self.norm.forward(x)

        norm = ttnn.mean(ttnn.pow(x, 2), dim=-1, keepdim=True)

        if self._tp_axis_size != 1:
            n = self._tp_axis_size
            tensor_rank = len(x.shape)
            # repeat the tensor since we do not have an all-reduce op yet
            norm = ttnn.repeat(norm, [1] * (tensor_rank - 1) + [n])
            norm = self._ccl_manager.reduce_scatter_persistent_buffer(norm, dim=-1, mesh_axis=self._tp_axis)
            norm = norm * (1 / n)

        norm = ttnn.sqrt(norm + self.eps)
        return ttnn.div(x, norm) * self.gamma.data


class QwenImageResample(Module):
    def __init__(self, *, dim: int, mode: str, ctx: QwenImageVaeContext) -> None:
        super().__init__()

        if mode in {"upsample2d", "upsample3d"}:
            self.conv = Conv2d(
                dim,
                dim // 2,
                kernel_size=3,
                padding=1,
                in_mesh_axis=ctx.tp_axis,
                mesh_device=ctx.device,
                ccl_manager=ctx.ccl_manager,
            )
        else:
            msg = f"unsupported resample mode '{mode}'"
            raise ValueError(msg)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        pop_substate(state, "time_conv")  # only needed for temporal upsampling
        rename_substate(state, "resample.1", "conv")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = tensor.upsample(x, scale_factor=2)
        return self.conv.forward(x)


class QwenImageResidualBlock(Module):
    def __init__(self, *, in_dim: int, out_dim: int, non_linearity: str, ctx: QwenImageVaeContext) -> None:
        super().__init__()

        assert non_linearity == "silu"
        self._nonlinearity = ttnn.silu

        self.norm1 = QwenImageRmsNorm(in_dim, ctx=ctx)
        self.conv1 = QwenImageConv(in_dim, out_dim, kernel_size=3, padding=1, ctx=ctx)
        self.norm2 = QwenImageRmsNorm(out_dim, ctx=ctx)
        self.conv2 = QwenImageConv(out_dim, out_dim, kernel_size=3, padding=1, ctx=ctx)
        self.conv_shortcut = (
            RowParallelLinear(
                in_dim, out_dim, mesh_axis=ctx.tp_axis, mesh_device=ctx.device, ccl_manager=ctx.ccl_manager
            )
            if in_dim != out_dim
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "conv_shortcut.weight" in state:
            state["conv_shortcut.weight"] = state["conv_shortcut.weight"].flatten(1)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        h = self.conv_shortcut(x) if self.conv_shortcut is not None else x

        x = self.norm1.forward(x)
        x = self._nonlinearity(x)
        x = self.conv1.forward(x)

        x = self.norm2.forward(x)
        x = self._nonlinearity(x)
        x = self.conv2.forward(x)

        return x + h


class QwenImageAttentionBlock(Module):
    def __init__(self, *, dim: int, ctx: QwenImageVaeContext) -> None:
        super().__init__()

        if ctx.tp_axis is not None:
            assert ctx.ccl_manager is not None

        self.norm = QwenImageRmsNorm(dim, ctx=ctx)
        self.to_qkv = RowParallelLinear(
            dim, dim * 3, mesh_axis=ctx.tp_axis, mesh_device=ctx.device, ccl_manager=ctx.ccl_manager
        )
        self.proj = ColParallelLinear(
            dim, dim, mesh_axis=ctx.tp_axis, mesh_device=ctx.device, ccl_manager=ctx.ccl_manager
        )

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
        self._tp_factor = ctx.device.shape[ctx.tp_axis] if ctx.tp_axis is not None else 1
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "to_qkv.weight" in state:
            state["to_qkv.weight"] = state["to_qkv.weight"].flatten(1)
        if "proj.weight" in state:
            state["proj.weight"] = state["proj.weight"].flatten(1)

        prepare_chunked_linear_output(state, prefix="to_qkv", device_count=self._tp_factor, chunks=3)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        identity = x

        x = self.norm.forward(x)

        batch_size, height, width, channels = x.shape

        # convert to 1d sequence and insert head dimension; there is only one head
        x = x.reshape([batch_size, 1, height * width, channels])

        x = self.to_qkv.forward(x)

        q, k, v = ttnn.chunk(x, 3, dim=-1)  # batch_size, 1, height * width, head_size

        if self._tp_axis is not None:
            q = self._ccl_manager.all_gather_persistent_buffer(q, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            k = self._ccl_manager.all_gather_persistent_buffer(k, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)
            v = self._ccl_manager.all_gather_persistent_buffer(v, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = ttnn.transformer.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=False,
            program_config=self._sdpa_program_config,
            compute_kernel_config=self._sdpa_compute_kernel_config,
        )

        x = self.proj.forward(x)

        # convert back to 2d
        x = x.reshape([batch_size, height, width, -1])

        return x + identity


class QwenImageMidBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        non_linearity: str,
        num_layers: int = 1,
        ctx: QwenImageVaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList(
            QwenImageResidualBlock(in_dim=dim, out_dim=dim, non_linearity=non_linearity, ctx=ctx)
            for _ in range(num_layers + 1)
        )
        self.attentions = ModuleList(QwenImageAttentionBlock(dim=dim, ctx=ctx) for _ in range(num_layers))

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        first_resnet, *other_resnets = self.resnets

        x = first_resnet.forward(x)

        for attn, resnet in zip(self.attentions, other_resnets, strict=True):
            x = attn.forward(x)
            x = resnet.forward(x)

        return x


class QwenImageUpBlock(Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None,
        non_linearity: str,
        ctx: QwenImageVaeContext,
    ) -> None:
        super().__init__()

        self.resnets = ModuleList([])
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            self.resnets.append(
                QwenImageResidualBlock(
                    in_dim=current_dim,
                    out_dim=out_dim,
                    non_linearity=non_linearity,
                    ctx=ctx,
                )
            )
            current_dim = out_dim

        self.upsampler = (
            QwenImageResample(dim=out_dim, mode=upsample_mode, ctx=ctx) if upsample_mode is not None else None
        )

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "upsamplers.0", "upsampler")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Run the model forward.

        May or may not return tensor sharded on the mesh depending on input size.
        """
        for resnet in self.resnets:
            x = resnet.forward(x)

        if self.upsampler is not None:
            x = self.upsampler.forward(x)

        return x


class QwenImageVaeDecoder(Module):
    """Qwen-Image VAE decoder without support for temporal dimension."""

    def __init__(
        self,
        *,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        temperal_downsample: Sequence[bool] = (False, True, True),
        non_linearity: str = "silu",
        parallel_config: VAEParallelConfig | None,
        device: ttnn.MeshDevice,
        ccl_manager: CCLManager | None,
    ) -> None:
        super().__init__()

        ctx = QwenImageVaeContext(
            tp_axis=parallel_config.tensor_parallel.mesh_axis if parallel_config is not None else None,
            device=device,
            ccl_manager=ccl_manager,
        )

        if ctx.tp_axis is not None and ctx.ccl_manager is None:
            msg = "ccl_manager must be provided if tensor parallelism is used"
            raise ValueError(msg)

        assert non_linearity == "silu"
        self._nonlinearity = ttnn.silu

        dims = [base_dim * u for u in [dim_mult[-1], *dim_mult[::-1]]]

        self.post_quant_conv = Linear(z_dim, z_dim, mesh_device=device)
        self.conv_in = QwenImageConv(z_dim, dims[0], kernel_size=3, padding=1, ctx=ctx)
        self.mid_block = QwenImageMidBlock(dim=dims[0], non_linearity=non_linearity, num_layers=1, ctx=ctx)

        self.up_blocks = ModuleList([])
        for i, (in_dim, out_dim) in enumerate(itertools.pairwise(dims)):
            if i == len(dim_mult) - 1:
                upsample_mode = None
            else:
                upsample_mode = "upsample3d" if temperal_downsample[-i - 1] else "upsample2d"

            up_block = QwenImageUpBlock(
                in_dim=in_dim // 2 if i > 0 else in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                upsample_mode=upsample_mode,
                non_linearity=non_linearity,
                ctx=ctx,
            )
            self.up_blocks.append(up_block)

        self.norm_out = QwenImageRmsNorm(out_dim, ctx=ctx)
        self.conv_out = QwenImageConv(out_dim, 3, kernel_size=3, padding=1, tensor_parallel=False, ctx=ctx)

        self._tp_axis = ctx.tp_axis
        self._ccl_manager = ctx.ccl_manager

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "post_quant_conv.weight" in state:
            state["post_quant_conv.weight"] = state["post_quant_conv.weight"].flatten(1)

        rename_substate(state, "decoder", "")

        # remove encoder state
        pop_substate(state, "quant_conv")
        pop_substate(state, "encoder")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.post_quant_conv.forward(x)
        x = self.conv_in.forward(x)
        x = self.mid_block.forward(x)

        for block in self.up_blocks:
            x = block.forward(x)

        x = self.norm_out.forward(x)
        x = self._nonlinearity(x)

        if self._tp_axis is not None:
            x = self._ccl_manager.all_gather_persistent_buffer(x, dim=-1, mesh_axis=self._tp_axis, use_hyperparams=True)

        x = self.conv_out.forward(x)

        return ttnn.clamp(x, min=-1.0, max=1.0)

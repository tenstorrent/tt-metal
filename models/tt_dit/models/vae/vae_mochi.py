# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import ttnn

from ...layers.conv3d import ContextParallelConv3d
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import GroupNorm
from ...parallel.config import MochiVAEParallelConfig, vae_all_gather, vae_neighbor_pad, vae_slice_reshard
from ...parallel.manager import CCLManager
from ...utils.substate import rename_substate

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


def get_padded_size(numerator, denominator):
    return ((numerator + denominator - 1) // denominator) * denominator


class Conv1x1(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        bias: bool = True,
        swizzle_weight: Callable[[torch.Tensor], torch.Tensor] | None = None,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        """
        A 1x1 convolution implemented as a linear operation for ttnn.

        Can be instantiated from either:
        - nn.Conv3d with kernel_size=(1,1,1)
        - Conv1x1 (which is implemented as nn.Linear)

        Args:
            mesh_device: TTNN mesh device
            state_dict: Dictionary containing weights
            state_dict_prefix: Prefix for loading weights from state_dict
            in_channels: Number of input channels
            out_channels: Number of output channels
            bias: Whether to include bias
            swizzle_weight: Function to swizzle weights, useful for channel expansion
        """
        super().__init__()

        self.weight = Parameter(total_shape=[in_channels, out_channels], device=mesh_device)
        self.bias = Parameter(total_shape=[1, out_channels], device=mesh_device) if bias else None

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.swizzle_weight = swizzle_weight
        self.mesh_device = mesh_device

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.get("weight")
        if weight is not None:
            # Load weights - supports both Conv3d(1,1,1) and Conv1x1 format
            if weight.ndim == 5:  # Conv3d weight
                # Convert from (out_channels, in_channels, 1, 1, 1) to (out_channels, in_channels)
                weight = weight.squeeze()
            weight = weight.transpose(0, 1)  # (out_channels, in_channels) -> (in_channels, out_channels)
            if self.swizzle_weight:
                weight = self.swizzle_weight(weight)
            state["weight"] = weight

        bias = state.get("bias")
        if bias is not None:
            if self.swizzle_weight:
                bias = self.swizzle_weight(bias)
            state["bias"] = bias.reshape(1, -1)

    def forward(self, x_NTHWC):
        """
        Forward pass for Conv1x1.

        Args:
            x_NTHWC: Input tensor in NTHWC layout

        Returns:
            Output tensor in NTHWC layout
        """
        # Convert to tile layout for efficient computation
        x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
        ttnn.deallocate(x_NTHWC)

        # Apply linear transformation
        x_tile_NTHWO = ttnn.linear(
            x_tile_NTHWC,
            self.weight.data,
            bias=self.bias.data if self.bias is not None else None,
            compute_kernel_config=self.compute_kernel_config,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            core_grid=self.mesh_device.core_grid,
        )
        ttnn.deallocate(x_tile_NTHWC)

        # Convert back to row major layout
        x_NTHWO = ttnn.to_layout(x_tile_NTHWO, ttnn.ROW_MAJOR_LAYOUT)
        ttnn.deallocate(x_tile_NTHWO)

        return x_NTHWO


class ResBlock(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        causal: bool = True,
        padding_mode: str = "replicate",
        bias: bool = True,
        mesh_device: ttnn.MeshDevice,
        parallel_config: MochiVAEParallelConfig,
        ccl_manager: CCLManager,
    ) -> None:
        super().__init__()

        self.core_grid_y_map = {
            # large latent
            768: {
                8: 4,  # 28 padded up to 32, divided by 8
                4: 7,  # 28/4
            },
        }
        self.num_out_blocks_map = {
            # small latent, large latent
            768: {
                40 * 50: 2,
                60 * 106: 8,
            },
            512: {
                80 * 100: 4,
                120 * 212: 10,
            },
            256: {
                160 * 200: 15,
                240 * 424: 40,
            },
            128: {
                320 * 400: 50,
                480 * 848: 140,
            },
        }
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        grid_size_x = mesh_device.core_grid.x
        grid_size_y = (
            self.core_grid_y_map[768][self.parallel_config.time_parallel.factor]
            if in_channels == 768
            else mesh_device.core_grid.y
        )
        self.grid_size = ttnn.CoreGrid(y=grid_size_y, x=grid_size_x)

        self.norm1 = GroupNorm(
            in_channels,
            num_groups=32,
            mesh_device=mesh_device,
            mesh_axis=None,
            core_grid=self.grid_size,
        )
        self.norm2 = GroupNorm(
            out_channels,
            num_groups=32,
            mesh_device=mesh_device,
            mesh_axis=None,
            core_grid=self.grid_size,
        )
        self.conv1 = ContextParallelConv3d(
            in_channels,
            out_channels,
            mesh_device=mesh_device,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding_mode=padding_mode,
            bias=bias,
            causal=causal,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )
        self.conv2 = ContextParallelConv3d(
            out_channels,
            out_channels,
            mesh_device=mesh_device,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding_mode=padding_mode,
            bias=bias,
            causal=causal,
            parallel_config=parallel_config,
            ccl_manager=ccl_manager,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "norm1.norm_layer", "norm1")
        rename_substate(state, "norm2.norm_layer", "norm2")
        rename_substate(state, "conv1.conv", "conv1")
        rename_substate(state, "conv2.conv", "conv2")

    def get_tensor_shapes(self, x):
        return x.shape

    def reshape_tilize(self, x, shape):
        N, T, H, W, C = shape
        output = ttnn.tilize_with_zero_padding(
            ttnn.reshape(x, [N * T, 1, H * W, C]),
            use_multicore=True,
        )
        return output

    def pre_all_gather_reshape_norm_1(self, x, shape):
        N, T, H, W, C = shape
        dim_2_1 = ttnn.reshape(x, [N * T, 1, H * W, C])
        residual = ttnn.tilize_with_zero_padding(
            dim_2_1,
            use_multicore=True,
        )
        assert not (C % 32)
        if not (H * W * (C // 32) % 32):
            output = ttnn.reshape(residual, [N * T, 1, H * W * (C // 32), 32])
            gather_dim = 2
        else:
            output = ttnn.reshape(residual, [1, 1, N * T, H * W * C])
            gather_dim = 3
        ttnn.deallocate(dim_2_1)
        return gather_dim, residual, output

    def pre_all_gather_reshape_norm_2(self, x, shape):
        N, T, H, W, C = shape
        assert not (C % 32)
        if not (H * W * (C // 32) % 32):
            x = ttnn.reshape(x, (N * T, 1, H * W * (C // 32), 32))
            gather_dim = 2
        else:
            x = ttnn.reshape(x, (1, 1, N * T, H * W * C))
            gather_dim = 3
        return gather_dim, x

    def sharded_reshape_untilize_tilize(self, x, input_shapes, output_shapes, deallocate_input):
        N, T, H, W, C = input_shapes
        dim0, dim1, dim2, dim3 = output_shapes
        if C == 768:
            x = ttnn.reshape(x, (1, 1, N * T * H, W * C))
        output = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        if deallocate_input:
            ttnn.deallocate(x)
        output = ttnn.reshape(output, [dim0, dim1, dim2, dim3])
        output = ttnn.tilize_with_zero_padding(
            output,
            use_multicore=True,
        )
        return output

    def untilize_reshape(self, x, shapes):
        N, T, H, W, C = shapes
        output = ttnn.reshape(ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT), [N, T, H, W, C])
        return output

    def forward(self, x_NTHWC: ttnn.Tensor) -> ttnn.Tensor:
        shapes = self.get_tensor_shapes(x_NTHWC)
        N, T, H, W, C = shapes
        if self.parallel_config.w_parallel.factor > 1:
            gather_dim, residual_tiled_NTHWC, x_tiled_NTHWC = self.pre_all_gather_reshape_norm_1(x_NTHWC, shapes)
            ttnn.deallocate(x_NTHWC)
            residual_tiled_NTHWC = ttnn.reallocate(residual_tiled_NTHWC)
            all_gather_output = vae_all_gather(
                self.ccl_manager,
                x_tiled_NTHWC,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                dim=gather_dim,
                reshape=False,
            )
            ttnn.deallocate(x_tiled_NTHWC)
            x_tiled_NTHWC = self.sharded_reshape_untilize_tilize(
                all_gather_output,
                (N, T, H * self.parallel_config.h_parallel.factor, W * self.parallel_config.w_parallel.factor, C),
                (N * T, 1, H * W * self.parallel_config.h_parallel.factor * self.parallel_config.w_parallel.factor, C),
                True,
            )
            ttnn.deallocate(all_gather_output)
        else:
            x_tiled_NTHWC = self.reshape_tilize(x_NTHWC, shapes)
            ttnn.deallocate(x_NTHWC)
            residual_tiled_NTHWC = x_tiled_NTHWC
        gathered_shapes = (
            shapes[0],
            shapes[1],
            shapes[2] * self.parallel_config.h_parallel.factor,
            shapes[3] * self.parallel_config.w_parallel.factor,
            shapes[4],
        )

        HW = x_tiled_NTHWC.shape[2]
        C = x_tiled_NTHWC.shape[3]
        num_out_blocks = self.num_out_blocks_map[C][HW]

        x_norm_tiled_NTHWC = self.norm1(x_tiled_NTHWC, num_out_blocks)

        if self.parallel_config.w_parallel.factor > 1:
            ttnn.deallocate(x_tiled_NTHWC)
        x_norm_tiled_NTHWC = ttnn.silu(x_norm_tiled_NTHWC, output_tensor=x_norm_tiled_NTHWC)  # in-place
        x_NTHWC = self.untilize_reshape(x_norm_tiled_NTHWC, gathered_shapes)
        ttnn.deallocate(x_norm_tiled_NTHWC)

        if self.parallel_config.w_parallel.factor > 1:
            x_NTHWC = ttnn.reshape(
                x_NTHWC,
                (N, T, self.parallel_config.h_parallel.factor * self.parallel_config.w_parallel.factor, H, W, C),
            )

            x_NTHWC = ttnn.mesh_partition(
                x_NTHWC,
                2,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                memory_config=x_NTHWC.memory_config(),
            )
            x_NTHWC = ttnn.squeeze(x_NTHWC, 0)  # Get rid of N
            x_NTHWC = ttnn.squeeze(x_NTHWC, 1)  # Get rid of HW dim
            x_NTHWC = vae_neighbor_pad(
                self.ccl_manager,
                x_NTHWC,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                dim=2,
                padding_left=1,
                padding_right=1,
                padding_mode="replicate",
                secondary_cluster_axis=1,
                secondary_mesh_shape=(self.parallel_config.h_parallel.factor, self.parallel_config.w_parallel.factor),
            )
            if self.parallel_config.h_parallel.factor > 1:
                x_NTHWC = vae_neighbor_pad(
                    self.ccl_manager,
                    x_NTHWC,
                    cluster_axis=self.parallel_config.h_parallel.mesh_axis,
                    dim=1,
                    padding_left=1,
                    padding_right=1,
                    padding_mode="replicate",
                    secondary_cluster_axis=0,
                    secondary_mesh_shape=(
                        self.parallel_config.h_parallel.factor,
                        self.parallel_config.w_parallel.factor,
                    ),
                )
            x_NTHWC = ttnn.unsqueeze(x_NTHWC, 0)
        elif self.parallel_config.h_parallel.factor > 1:
            raise NotImplementedError()

        x_conv1_NTHWC = self.conv1(x_NTHWC)
        ttnn.deallocate(x_NTHWC)
        x_conv1_tiled_NTHWC = self.reshape_tilize(x_conv1_NTHWC, shapes)

        if self.parallel_config.w_parallel.factor > 1:
            gather_dim, x_conv1_tiled_NTHWC = self.pre_all_gather_reshape_norm_2(x_conv1_tiled_NTHWC, shapes)
            ttnn.deallocate(x_conv1_NTHWC)
            x_conv1_tiled_NTHWC = ttnn.reallocate(x_conv1_tiled_NTHWC)
            x_conv1_tiled_NTHWC = vae_all_gather(
                self.ccl_manager,
                x_conv1_tiled_NTHWC,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                dim=gather_dim,
                reshape=False,
            )
            x_conv1_tiled_NTHWC = self.sharded_reshape_untilize_tilize(
                x_conv1_tiled_NTHWC,
                (N, T, H * self.parallel_config.h_parallel.factor, W * self.parallel_config.w_parallel.factor, C),
                (N * T, 1, H * W * self.parallel_config.h_parallel.factor * self.parallel_config.w_parallel.factor, C),
                True,
            )
        else:
            ttnn.deallocate(x_conv1_NTHWC)

        HW = x_conv1_tiled_NTHWC.shape[2]
        C = x_conv1_tiled_NTHWC.shape[3]
        num_out_blocks = self.num_out_blocks_map[C][HW]
        x_tiled_NTHWC = self.norm2(x_conv1_tiled_NTHWC, num_out_blocks)
        ttnn.deallocate(x_conv1_tiled_NTHWC)
        x_tiled_NTHWC = ttnn.silu(x_tiled_NTHWC, output_tensor=x_tiled_NTHWC)  # in-place
        x_NTHWC = self.untilize_reshape(x_tiled_NTHWC, gathered_shapes)
        ttnn.deallocate(x_tiled_NTHWC)

        if self.parallel_config.w_parallel.factor > 1:
            x_NTHWC = ttnn.reshape(
                x_NTHWC,
                (N, T, self.parallel_config.h_parallel.factor * self.parallel_config.w_parallel.factor, H, W, C),
            )
            x_NTHWC = ttnn.mesh_partition(
                x_NTHWC,
                2,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                memory_config=x_NTHWC.memory_config(),
            )
            x_NTHWC = ttnn.squeeze(x_NTHWC, 0)  # Get rid of N
            x_NTHWC = ttnn.squeeze(x_NTHWC, 1)  # Get rid of HW dim
            x_NTHWC = vae_neighbor_pad(
                self.ccl_manager,
                x_NTHWC,
                cluster_axis=self.parallel_config.w_parallel.mesh_axis,
                dim=2,
                padding_left=1,
                padding_right=1,
                padding_mode="replicate",
                secondary_cluster_axis=1,
                secondary_mesh_shape=(self.parallel_config.h_parallel.factor, self.parallel_config.w_parallel.factor),
            )
            if self.parallel_config.h_parallel.factor > 1:
                x_NTHWC = vae_neighbor_pad(
                    self.ccl_manager,
                    x_NTHWC,
                    cluster_axis=self.parallel_config.h_parallel.mesh_axis,
                    dim=1,
                    padding_left=1,
                    padding_right=1,
                    padding_mode="replicate",
                    secondary_cluster_axis=0,
                    secondary_mesh_shape=(
                        self.parallel_config.h_parallel.factor,
                        self.parallel_config.w_parallel.factor,
                    ),
                )
            x_NTHWC = ttnn.unsqueeze(x_NTHWC, 0)
        elif self.parallel_config.h_parallel.factor > 1:
            raise NotImplementedError()

        x_conv2_NTHWC = self.conv2(x_NTHWC)
        ttnn.deallocate(x_NTHWC)
        x_conv2_tiled_NTHWC = self.reshape_tilize(x_conv2_NTHWC, shapes)
        ttnn.deallocate(x_conv2_NTHWC)

        x_tiled_NTHWC = ttnn.add(x_conv2_tiled_NTHWC, residual_tiled_NTHWC)
        x_NTHWC = self.untilize_reshape(x_tiled_NTHWC, shapes)
        ttnn.deallocate(x_conv2_tiled_NTHWC)
        ttnn.deallocate(residual_tiled_NTHWC)
        ttnn.deallocate(x_tiled_NTHWC)
        return x_NTHWC


class CausalUpsampleBlock(Module):
    def __init__(
        self,
        mesh_device: ttnn.MeshDevice,
        in_channels: int,
        out_channels: int,
        num_res_blocks: int = 0,
        *,
        parallel_config: MochiVAEParallelConfig,
        ccl_manager: CCLManager,
        temporal_expansion: int = 2,
        spatial_expansion: int = 2,
        temporal_offset: int = 0,
        has_attention: bool = False,
        causal: bool = True,
        prune_bottleneck: bool = False,
        padding_mode: str = "replicate",
        bias: bool = True,
    ) -> None:
        super().__init__()

        assert causal
        assert not prune_bottleneck
        assert not has_attention
        self.mesh_device = mesh_device
        self.resnets = ModuleList(
            ResBlock(
                in_channels,
                in_channels,
                mesh_device=mesh_device,
                causal=causal,
                padding_mode=padding_mode,
                bias=bias,
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for _ in range(num_res_blocks)
        )

        self.temporal_expansion = temporal_expansion
        self.spatial_expansion = spatial_expansion
        self.temporal_offset = temporal_offset
        self.out_channels = out_channels
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.reshard_time_map = {
            120 * 212: 84,
            240 * 424: 168,
            480 * 848: 168,
        }

        # Swizzle conv1x1 weights
        def swizzle_weight(w):
            # X (C texp sexp sexp) -> X (texp sexp sexp C)
            w = w.reshape(-1, out_channels, temporal_expansion, spatial_expansion, spatial_expansion)
            w = w.permute(0, 2, 3, 4, 1)
            w = w.reshape(-1, temporal_expansion * spatial_expansion * spatial_expansion * out_channels)
            return w.squeeze()

        self.proj = Conv1x1(
            mesh_device=mesh_device,
            in_channels=in_channels,
            out_channels=out_channels * temporal_expansion * (spatial_expansion**2),
            bias=bias,
            swizzle_weight=swizzle_weight,
        )

    def depth_to_spacetime(self, x_NTHWC):
        texp, sexp = self.temporal_expansion, self.spatial_expansion
        if self.parallel_config.time_parallel.factor == 1:
            B, T, H, W, C = x_NTHWC.shape
            x = ttnn.reshape(x_NTHWC, [B, T, H, W, texp, sexp, sexp, self.out_channels])
            x = ttnn.permute(x, [0, 1, 4, 2, 5, 3, 6, 7])  # (B T texp H sexp W sexp C_out)

            x = ttnn.reshape(x, [B, T * texp, H * sexp, W * sexp, self.out_channels])
            if texp > 1:
                # Drop the first temporal_offset frames.
                x = ttnn.slice(
                    x, [0, self.temporal_offset, 0, 0, 0], [B, T * texp, H * sexp, W * sexp, self.out_channels]
                )
            return x
        else:
            # Workaround for 1) issue #17535 for multi-device reshape,
            # and 2) slicing only the first shard.
            B, T, H, W, C = x_NTHWC.shape
            x_NTHWC = ttnn.reshape(x_NTHWC, [B, T, H, W, texp, sexp, sexp, self.out_channels])
            x_NTHWC = ttnn.permute(x_NTHWC, [0, 1, 4, 2, 5, 3, 6, 7])  # (B T texp H sexp W sexp C_out)

            x_NTHWC = ttnn.reshape(x_NTHWC, [B, T * texp, H * sexp, W * sexp, self.out_channels])

            return x_NTHWC

    def reshard_output(self, x_NTHWC):
        N, T, H, W, C = x_NTHWC.shape
        num_devices = self.parallel_config.time_parallel.factor
        expected_T = self.reshard_time_map[
            H * W * self.parallel_config.h_parallel.factor * self.parallel_config.w_parallel.factor
        ]
        input_is_padded = T * num_devices != expected_T
        if (self.temporal_offset > 0 or input_is_padded) and (
            self.parallel_config.time_parallel.factor > 1 and self.temporal_expansion > 1
        ):
            padded_T = ((expected_T + num_devices - 1) // num_devices) * num_devices
            x_NTHWC = ttnn.squeeze(x_NTHWC, 0)
            x_NTHWC = vae_slice_reshard(
                self.ccl_manager,
                x_NTHWC,
                cluster_axis=self.parallel_config.time_parallel.mesh_axis,
                dim=0,
                output_shape=padded_T,
                output_offset=self.temporal_offset,
            )
            x_NTHWC = ttnn.unsqueeze(x_NTHWC, 0)
        return x_NTHWC

    def forward(self, x_NTHWC: ttnn.Tensor) -> ttnn.Tensor:
        N, T, H, W, C = x_NTHWC.shape
        for block in self.resnets:
            x_NTHWC = block(x_NTHWC)
        x_NTHWO = self.proj(x_NTHWC)
        x_NTHWC = self.depth_to_spacetime(x_NTHWO)
        ttnn.deallocate(x_NTHWO)
        x_NTHWC = self.reshard_output(x_NTHWC)
        return x_NTHWC


class MochiVAEDecoder(Module):
    def __init__(
        self,
        out_channels: int,
        *,
        mesh_device: ttnn.MeshDevice,
        parallel_config: MochiVAEParallelConfig,
        ccl_manager: CCLManager,
        base_channels: int = 128,
        channel_multipliers: Sequence[int] = (1, 2, 4, 6),
        temporal_expansions: Sequence[int] = (1, 2, 3),
        spatial_expansions: Sequence[int] = (2, 2, 2),
        num_res_blocks: Sequence[int] = (3, 3, 4, 6, 3),
        latent_dim: int = 12,
        has_attention: Sequence[bool] = (False, False, False, False, False),
        nonlinearity: str = "silu",
        output_nonlinearity: str = "silu",
        causal: bool = True,
        latents_mean=None,
        latents_std=None,
        scaling_factor=1.0,
    ) -> None:
        """
        TTNN implementation of the VAE Decoder.
        """
        super().__init__()

        self.input_channels = latent_dim
        self.output_channels = out_channels
        self.base_channels = base_channels
        self.channel_multipliers = channel_multipliers
        self.num_res_blocks = num_res_blocks
        self.temporal_expansions = temporal_expansions
        self.spatial_expansions = spatial_expansions
        self.output_nonlinearity = output_nonlinearity
        self.mesh_device = mesh_device
        self.config = lambda: None
        self.config.latents_mean = latents_mean
        self.config.latents_std = latents_std
        self.config.scaling_factor = scaling_factor
        assert nonlinearity == "silu"
        assert causal
        assert not any(has_attention), "Attention is not supported in the decoder"

        # Calculate channels for each level
        ch = [mult * base_channels for mult in channel_multipliers]
        self.num_up_blocks = len(ch) - 1
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager
        assert len(num_res_blocks) == self.num_up_blocks + 2

        assert len(temporal_expansions) == len(spatial_expansions) == self.num_up_blocks
        assert len(num_res_blocks) == len(has_attention) == self.num_up_blocks + 2

        # Create the initial projection from latent space
        self.input_proj = Conv1x1(latent_dim, ch[-1], mesh_device=mesh_device)

        # First set of residual blocks
        self.first_blocks = ModuleList(
            ResBlock(
                ch[-1],
                ch[-1],
                mesh_device=mesh_device,
                causal=causal,
                padding_mode="replicate",
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for _ in range(num_res_blocks[-1])
        )

        # Create upsampling blocks
        self.up_blocks = ModuleList(
            CausalUpsampleBlock(
                mesh_device=mesh_device,
                in_channels=ch[-i - 1],
                out_channels=ch[-i - 2],
                num_res_blocks=num_res_blocks[-i - 2],
                temporal_expansion=temporal_expansions[-i - 1],
                spatial_expansion=spatial_expansions[-i - 1],
                causal=causal,
                padding_mode="replicate",
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for i in range(len(ch) - 1)
        )

        # Last set of residual blocks
        self.last_blocks = ModuleList(
            ResBlock(
                ch[0],
                ch[0],
                mesh_device=mesh_device,
                causal=causal,
                padding_mode="replicate",
                parallel_config=parallel_config,
                ccl_manager=ccl_manager,
            )
            for _ in range(num_res_blocks[0])
        )

        # Final output projection
        self.output_proj = Conv1x1(ch[0], out_channels, mesh_device=mesh_device)

        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "conv_in", "input_proj")
        rename_substate(state, "proj_out", "output_proj")

        rename_substate(state, "block_in.resnets", "first_blocks")
        rename_substate(state, "block_out.resnets", "last_blocks")

    def dealloc(self):
        self.input_proj.dealloc()
        for block in self.first_blocks:
            block.dealloc()
        for block in self.up_blocks:
            block.dealloc()
        for block in self.last_blocks:
            block.dealloc()
        self.output_proj.dealloc()

    def prepare_input(self, x_NCTHW):
        N, C, T, H, W = x_NCTHW.shape
        x_NTHWC = x_NCTHW.permute(0, 2, 3, 4, 1)  # [N, T, H, W, C]

        num_devices_T = self.mesh_device.shape[self.parallel_config.time_parallel.mesh_axis]
        if T % num_devices_T:
            padded_T = get_padded_size(T, num_devices_T)
            T_padding = padded_T - T
            x_NTHWC = torch.nn.functional.pad(x_NTHWC, pad=(0, 0, 0, 0, 0, 0, 0, T_padding))
        else:
            padded_T = T
        num_devices_W = self.parallel_config.w_parallel.factor
        if W % num_devices_W:
            padded_W = get_padded_size(W, num_devices_W)
            W_padding = padded_W - W
            x_NTHWC = torch.nn.functional.pad(x_NTHWC, pad=(0, 0, 0, W_padding))
        else:
            padded_W = W
        num_devices_H = self.parallel_config.h_parallel.factor
        if H % num_devices_H:
            padded_H = get_padded_size(H, num_devices_H)
            H_padding = padded_H - H
            x_NTHWC = torch.nn.functional.pad(x_NTHWC, pad=(0, 0, 0, 0, 0, H_padding))
        else:
            padded_H = H

        x_NTHWC = torch.reshape(
            x_NTHWC,
            (N, padded_T, num_devices_H, padded_H // num_devices_H, num_devices_W, padded_W // num_devices_W, C),
        )
        x_NTHWC = x_NTHWC.permute(0, 1, 2, 4, 3, 5, 6)
        x_NTHWC = torch.reshape(
            x_NTHWC,
            (N, padded_T, num_devices_H * num_devices_W, padded_H // num_devices_H, padded_W // num_devices_W, C),
        )

        dims = [0, 0]
        dims[self.parallel_config.time_parallel.mesh_axis] = 1
        dims[self.parallel_config.w_parallel.mesh_axis] = 2

        tt_x_NTHWC = ttnn.from_torch(
            x_NTHWC,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims),
        )

        tt_x_NTHWC = ttnn.squeeze(tt_x_NTHWC, 2)
        return tt_x_NTHWC

    def postprocess_output(self, tt_x_NTHWC, input_shape):
        N, C, T, H, W = input_shape
        tt_x_NTHWC = ttnn.unsqueeze(tt_x_NTHWC, 2)

        dims = [0, 0]
        dims[self.parallel_config.time_parallel.mesh_axis] = 1
        dims[self.parallel_config.w_parallel.mesh_axis] = 2

        # Convert TT output to torch tensor
        x_NTHWC_torch = ttnn.to_torch(
            tt_x_NTHWC,
            mesh_composer=ttnn.ConcatMesh2dToTensor(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=dims
            ),
        )
        ttnn.deallocate(tt_x_NTHWC)

        num_devices_T = self.mesh_device.shape[self.parallel_config.time_parallel.mesh_axis]
        num_devices_W = self.parallel_config.w_parallel.factor
        num_devices_H = self.parallel_config.h_parallel.factor

        # unpad tt output
        expected_T = T * math.prod(self.temporal_expansions)
        expected_padded_T = get_padded_size(expected_T, num_devices_T)
        expected_H = H * math.prod(self.spatial_expansions) // num_devices_H
        expected_W = W * math.prod(self.spatial_expansions) // num_devices_W
        x_NTHWC_torch = torch.reshape(
            x_NTHWC_torch,
            (N, expected_padded_T, num_devices_H, num_devices_W, expected_H, expected_W, self.output_channels),
        )
        x_NTHWC_torch = x_NTHWC_torch.permute(0, 1, 2, 4, 3, 5, 6)
        x_NTHWC_torch = torch.reshape(
            x_NTHWC_torch,
            (N, expected_padded_T, num_devices_H * expected_H, num_devices_W * expected_W, self.output_channels),
        )
        x_NCTHW_torch = x_NTHWC_torch.permute(0, 4, 1, 2, 3)  # [N, C, T, H, W]
        return x_NCTHW_torch

    def forward(self, x_NTHWC: ttnn.Tensor) -> ttnn.Tensor:
        """
        Forward pass for the decoder.

        Args:
            x_NTHWC: Input tensor in NTHWC layout

        Returns:
            Output tensor in NTHWC layout
        """
        # Initial projection
        x_NTHWC = self.input_proj(x_NTHWC)

        # First set of residual blocks
        for block in self.first_blocks:
            x_res_NTHWC = block(x_NTHWC)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC

        # Upsampling blocks
        for block in self.up_blocks:
            x_res_NTHWC = block(x_NTHWC)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC

        # Last set of residual blocks
        for block in self.last_blocks:
            x_res_NTHWC = block(x_NTHWC)
            ttnn.deallocate(x_NTHWC)
            x_NTHWC = x_res_NTHWC

        # Apply output nonlinearity if needed
        if self.output_nonlinearity == "silu":
            x_tile_NTHWC = ttnn.to_layout(x_NTHWC, ttnn.TILE_LAYOUT)
            ttnn.deallocate(x_NTHWC)
            x_tile_NTHWC = ttnn.silu(x_tile_NTHWC, output_tensor=x_tile_NTHWC)  # in-place
            x_NTHWC = ttnn.to_layout(x_tile_NTHWC, ttnn.ROW_MAJOR_LAYOUT)
            ttnn.deallocate(x_tile_NTHWC)
        else:
            assert not self.output_nonlinearity  # StyleGAN3 omits the to-RGB nonlinearity.

        # Final projection
        x_NTHWC = self.output_proj(x_NTHWC)

        return x_NTHWC

    def decode(self, x_NCTHW, return_dict):
        input_shape = x_NCTHW.shape
        tt_x_NTHWC = self.prepare_input(x_NCTHW)

        tt_x_NTHWC = self(tt_x_NTHWC)

        x_NCTHW_torch = self.postprocess_output(tt_x_NTHWC, input_shape)

        return [x_NCTHW_torch]

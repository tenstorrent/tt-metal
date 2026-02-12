# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from loguru import logger

import ttnn

from ...layers.linear import Linear
from ...layers.module import Module, ModuleList, Parameter
from ...layers.normalization import RMSNorm
from ...parallel.config import VaeHWParallelConfig
from ...parallel.manager import CCLManager
from ...utils.conv3d import _ntuple, aligned_channels, count_convs, get_conv3d_config, prepare_conv3d_weights
from ...utils.substate import pop_substate, rename_substate

if TYPE_CHECKING:
    from collections.abc import Sequence

CACHE_T = 2


def conv3d_to_linear_weight(state):
    weight = state["weight"]
    out_c, in_c, kt, kh, kw = weight.shape
    assert kt == kh == kw == 1
    weight = weight.reshape(out_c, in_c)
    padded_out_c = aligned_channels(out_c)
    padded_in_c = aligned_channels(in_c)
    weight = torch.nn.functional.pad(weight, (0, padded_in_c - in_c, 0, padded_out_c - out_c))
    bias = state["bias"]
    bias = torch.nn.functional.pad(bias, (0, padded_out_c - out_c))
    state["weight"] = weight
    state["bias"] = bias
    return state


class WanAttentionBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        self.norm = RMSNorm(
            embedding_dim=dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.to_qkv = Linear(
            in_features=dim,
            out_features=dim * 3,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.proj = Linear(
            in_features=dim,
            out_features=dim,
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
        )
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=32,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )
        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def permute_conv2d_weights(weight):
            out_c, in_c, kh, kw = weight.shape
            assert kh == kw == 1
            weight = weight.permute(0, 2, 3, 1).reshape(out_c, in_c)
            return weight

        if "to_qkv.weight" in state:
            state["to_qkv.weight"] = permute_conv2d_weights(state["to_qkv.weight"])

        if "proj.weight" in state:
            state["proj.weight"] = permute_conv2d_weights(state["proj.weight"])

        if "norm.gamma" in state:
            state["norm.weight"] = state.pop("norm.gamma").squeeze()

    def forward(self, x_BTHWC: ttnn.Tensor, logical_h: int) -> ttnn.Tensor:
        """
        x_BTHWC: (B, T, H, W, C) fractured on H and W

        returns: (B, T, H, W, C) fractured on H and W
        """
        assert len(x_BTHWC.shape) == 5
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT
        residual_BTHWC = x_BTHWC

        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)

        # Gather height and width for replicated attention
        if self.parallel_config.height_parallel.factor > 1:
            x_BTHWC = ttnn.experimental.all_gather_async(
                x_BTHWC,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    x_BTHWC.shape, 2, self.parallel_config.height_parallel.mesh_axis, dtype=x_BTHWC.dtype
                ),
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.height_parallel.mesh_axis,
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.height_parallel.mesh_axis,
            )

        if self.parallel_config.width_parallel.factor > 1:
            x_BTHWC = ttnn.experimental.all_gather_async(
                x_BTHWC,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    x_BTHWC.shape,
                    3,
                    self.parallel_config.width_parallel.mesh_axis,
                    dtype=x_BTHWC.dtype,
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.width_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.width_parallel.mesh_axis,
            )

        x_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        padded_h = x_BTHWC.shape[2]
        if padded_h != logical_h:
            """
            H is padded, so slice it out before attention
            """
            x_BTHWC = x_BTHWC[:, :, :logical_h, :, :]
        B, T, H, W, C = x_BTHWC.shape
        x_TNC = ttnn.reshape(x_BTHWC, (B * T, H * W, C))
        x_TNC = ttnn.to_layout(x_TNC, ttnn.TILE_LAYOUT)
        x_TNC = self.norm(x_TNC, compute_kernel_config=self.hifi4_compute_kernel_config)
        default_block_size = (2, 2, 2) if x_TNC.dtype == ttnn.DataType.FLOAT32 else (8, 8, 8)
        x_TND = self.to_qkv(
            x_TNC, compute_kernel_config=self.mm_compute_kernel_config, default_block_size=default_block_size
        )
        q_THNC, k_THNC, v_THNC = ttnn.transformer.split_query_key_value_and_split_heads(
            x_TND, num_heads=1, transpose_key=False
        )
        out_THNC = ttnn.transformer.scaled_dot_product_attention(
            ttnn.typecast(q_THNC, ttnn.bfloat16) if q_THNC.dtype != ttnn.bfloat16 else q_THNC,
            ttnn.typecast(k_THNC, ttnn.bfloat16) if k_THNC.dtype != ttnn.bfloat16 else k_THNC,
            ttnn.typecast(v_THNC, ttnn.bfloat16) if v_THNC.dtype != ttnn.bfloat16 else v_THNC,
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        out_THNC = ttnn.typecast(out_THNC, q_THNC.dtype) if out_THNC.dtype != q_THNC.dtype else out_THNC
        out_TNC = ttnn.transformer.concatenate_heads(out_THNC)
        out_TND = self.proj(
            out_TNC, compute_kernel_config=self.mm_compute_kernel_config, default_block_size=default_block_size
        )
        out_TND = ttnn.to_layout(out_TND, ttnn.ROW_MAJOR_LAYOUT)

        if padded_h != logical_h:
            """
            H should be padded to divide by H parallel factor.
            """
            # NOTE: Workaround. I'd prefer to pad after reshaping H out of HW, but
            # ttnn.pad only works on <=4 dims. Do padding while tensor has 3 dims.
            out_TND = ttnn.pad(out_TND, [(0, 0), (0, W * (padded_h - logical_h)), (0, 0)], value=0.0)

        # H after optionally padding
        H = out_TND.shape[1] // W

        out_BTHWC = ttnn.reshape(out_TND, (B, T, H, W, C))

        # Scatter height and width
        if self.parallel_config.height_parallel.factor > 1:
            out_BTHWC = ttnn.mesh_partition(
                out_BTHWC, dim=2, cluster_axis=self.parallel_config.height_parallel.mesh_axis
            )
        if self.parallel_config.width_parallel.factor > 1:
            out_BTHWC = ttnn.mesh_partition(
                out_BTHWC, dim=3, cluster_axis=self.parallel_config.width_parallel.mesh_axis
            )

        result_BTHWC = out_BTHWC + residual_BTHWC

        return result_BTHWC


class WanCausalConv3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int = 1,
        padding: Sequence[int] | int = 0,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        self.unpadded_in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.TILE_WIDTH = 32
        self.in_channels = aligned_channels(in_channels)
        if self.in_channels != self.unpadded_in_channels:
            logger.warning(f"Padding in_channels from {self.unpadded_in_channels} to {self.in_channels}")
        self.out_channels = self.TILE_WIDTH if out_channels < self.TILE_WIDTH else out_channels
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        padding = _ntuple(padding, 3)
        external_padding = list(padding)
        internal_padding = list(padding)
        # t padding is handled explicitly and depends on the cache.
        external_padding[0] = 2 * padding[0]
        internal_padding[0] = 0
        # HW padding may be handled by the halo CCL if the model is parallelized
        if self.parallel_config.height_parallel.factor > 1:
            external_padding[1] = padding[1]
            internal_padding[1] = 0
        if self.parallel_config.width_parallel.factor > 1:
            external_padding[2] = padding[2]
            internal_padding[2] = 0
        self.external_padding = tuple(external_padding)
        self.internal_padding = tuple(internal_padding)

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,  # Do not use HiFi4.
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)

        self.mask_cache = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def maybe_pad_out_channels(weight, bias):
            if self.out_channels != self.unpadded_out_channels:
                weight = torch.nn.functional.pad(
                    weight, (0, 0, 0, 0, 0, 0, 0, 0, 0, self.out_channels - self.unpadded_out_channels)
                )
                bias = torch.nn.functional.pad(bias, (0, self.out_channels - self.unpadded_out_channels))
            return weight, bias

        if "weight" in state and "bias" in state:
            weight, bias = maybe_pad_out_channels(state["weight"], state["bias"])
            state["weight"], state["bias"] = prepare_conv3d_weights(weight, bias, self.conv_config)

    def get_cached_mask(self, x_BTHWC, logical_h):
        sharded_h = x_BTHWC.shape[2]
        key = (sharded_h, logical_h)
        if key not in self.mask_cache:
            padded_h = sharded_h * self.parallel_config.height_parallel.factor
            mask_shape = (1, 1, padded_h, 1, 1)
            mask = torch.ones(mask_shape)
            mask[:, :, logical_h:, :, :] = 0.0
            mapper_dims = [None, None]
            mapper_dims[self.parallel_config.height_parallel.mesh_axis] = 2
            mask = ttnn.from_torch(
                mask,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=mapper_dims
                ),
            )
            self.mask_cache[key] = mask
        return self.mask_cache[key]

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        cache_x_BTHWC: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        x_BTHWC: (B, T, H, W, C) fractured on H and W
        cache_x_BTHWC: (B, T1, H, W, C) fractured on H and W

        returns: (B, T, H, W, C) fractured on H and W
        """
        # NOTE: T padding is handled explicitly and depends on the cache
        t_front_padding = self.external_padding[0]
        if cache_x_BTHWC is not None and t_front_padding > 0:
            # concat on T
            x_BTHWC = ttnn.concat([cache_x_BTHWC, x_BTHWC], dim=1)
            t_front_padding -= cache_x_BTHWC.shape[1]
        if t_front_padding > 0:
            # Padding only works on the lowest 3 dims. reshape input.
            B, T, H, W, C = x_BTHWC.shape
            x_BTNC = ttnn.reshape(x_BTHWC, (B, T, H * W, C))
            x_BTNC = ttnn.pad(x_BTNC, [(0, 0), (t_front_padding, 0), (0, 0), (0, 0)], value=0.0)
            x_BTHWC = ttnn.reshape(x_BTNC, (B, T + t_front_padding, H, W, C))

        if x_BTHWC.shape[2] * self.parallel_config.height_parallel.factor != logical_h:
            """
            H is padded to divide by H parallel factor. Must zero out padded portion of H.
            """
            mask = self.get_cached_mask(x_BTHWC, logical_h)
            x_BTHWC = ttnn.mul(x_BTHWC, mask)

        # Height halo
        if self.external_padding[1] > 0 and self.parallel_config.height_parallel.factor > 1:
            ttnn.synchronize_device(x_BTHWC.device())
            x_BTHWC = ttnn.experimental.neighbor_pad_async(
                x_BTHWC,
                dim=2,
                padding_left=self.external_padding[1],
                padding_right=self.external_padding[1],
                padding_mode="zeros",
                cluster_axis=self.parallel_config.height_parallel.mesh_axis,
                # neighbor_pad only needs one final semaphore, so index into ag semahpore list
                final_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.height_parallel.mesh_axis
                )[0],
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(
                    self.parallel_config.height_parallel.mesh_axis
                ),
                num_links=get_neighbor_pad_num_links(self.ccl_manager, x_BTHWC, 2),
                topology=self.ccl_manager.topology,
            )
            ttnn.synchronize_device(x_BTHWC.device())

        # Width halo
        if self.external_padding[2] > 0 and self.parallel_config.width_parallel.factor > 1:
            # TODO: Fix validation in neighbor_pad_async to allow halo on dim3
            x_THWC = ttnn.squeeze(x_BTHWC, dim=0)
            ttnn.synchronize_device(x_THWC.device())
            x_THWC = ttnn.experimental.neighbor_pad_async(
                x_THWC,
                dim=2,
                padding_left=self.external_padding[2],
                padding_right=self.external_padding[2],
                padding_mode="zeros",
                cluster_axis=self.parallel_config.width_parallel.mesh_axis,
                # neighbor_pad only needs one final semaphore, so index into ag semahpore list
                final_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.width_parallel.mesh_axis
                )[0],
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(self.parallel_config.width_parallel.mesh_axis),
                num_links=get_neighbor_pad_num_links(self.ccl_manager, x_THWC, 2),
                # memory_config=mem_config_output,
                topology=self.ccl_manager.topology,
            )
            ttnn.synchronize_device(x_THWC.device())
            x_BTHWC = ttnn.unsqueeze(x_THWC, dim=0)

        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
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

        if x_BTHWC.shape[2] * self.parallel_config.height_parallel.factor != logical_h:
            """
            H is padded to divide by H parallel factor. Must zero out padded portion of H.
            """
            mask = self.get_cached_mask(x_BTHWC, logical_h)
            x_BTHWC = ttnn.mul(x_BTHWC, mask)
        return x_BTHWC


class WanResidualBlock(Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mesh_device = mesh_device

        self.norm1 = RMSNorm(
            embedding_dim=in_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.conv1 = WanCausalConv3d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )
        self.norm2 = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.conv2 = WanCausalConv3d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        if in_dim != out_dim:
            self.conv_shortcut = Linear(
                in_features=in_dim,
                out_features=out_dim,
                mesh_device=mesh_device,
                dtype=dtype,
            )
        else:
            self.conv_shortcut = None

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,  # Do not use HiFi4.
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm1.gamma" in state:
            state["norm1.weight"] = state.pop("norm1.gamma").squeeze()

        if "norm2.gamma" in state:
            state["norm2.weight"] = state.pop("norm2.gamma").squeeze()

        def conv_1d_to_matmul_weight(weight):
            out_c, in_c, kt, kh, kw = weight.shape
            assert kt == kh == kw == 1
            weight = weight.reshape(out_c, in_c)
            return weight

        if "conv_shortcut.weight" in state:
            state["conv_shortcut.weight"] = conv_1d_to_matmul_weight(state["conv_shortcut.weight"])

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
    ) -> ttnn.Tensor:
        x_tile_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
        h_tile_BTHWC = (
            self.conv_shortcut(x_tile_BTHWC, compute_kernel_config=self.hifi4_compute_kernel_config)
            if self.conv_shortcut is not None
            else x_tile_BTHWC
        )
        x_norm_tile_BTHWC = self.norm1(x_tile_BTHWC, compute_kernel_config=self.hifi4_compute_kernel_config)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        # Cached conv
        if feat_cache is not None:
            # Prepare to cache the current activation for future use
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

            x_conv_BTHWC = self.conv1(x_BTHWC, logical_h, feat_cache[idx])
            # NOTE: Should deallocate feat_cache[idx] after it's reassigned
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_conv_BTHWC = self.conv1(x_BTHWC, logical_h)

        x_tile_BTHWC = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
        x_norm_tile_BTHWC = self.norm2(x_tile_BTHWC, compute_kernel_config=self.hifi4_compute_kernel_config)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        # Cached conv
        if feat_cache is not None:
            # Prepare to cache the current activation for future use
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

            x_conv_BTHWC = self.conv2(x_BTHWC, logical_h, feat_cache[idx])
            # NOTE: Should deallocate feat_cache[idx] after it's reassigned
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_conv_BTHWC = self.conv2(x_BTHWC, logical_h)

        # Add residual
        x_tile_BTHWC = ttnn.to_layout(x_conv_BTHWC, ttnn.TILE_LAYOUT)
        x_tile_BTHWC = ttnn.add(h_tile_BTHWC, x_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
        return x_BTHWC


class WanMidBlock(Module):
    def __init__(
        self,
        *,
        dim: int,
        num_layers: int = 1,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        resnets.append(
            WanResidualBlock(
                in_dim=dim,
                out_dim=dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
            )
        )

        for _ in range(num_layers):
            attentions.append(
                WanAttentionBlock(
                    dim=dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    dtype=dtype,
                )
            )
            resnets.append(
                WanResidualBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    dtype=dtype,
                )
            )

        self.resnets = resnets
        self.attentions = attentions

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
    ) -> ttnn.Tensor:
        x_res_BTHWC = self.resnets[0](x_BTHWC, logical_h, feat_cache, feat_idx)
        x_BTHWC = x_res_BTHWC
        for i in range(len(self.attentions)):
            x_attn_BTHWC = self.attentions[i](x_BTHWC, logical_h)
            x_BTHWC = self.resnets[i + 1](x_attn_BTHWC, logical_h, feat_cache, feat_idx)
        return x_BTHWC


class WanConv2d(Module):
    """
    A conv2d implemented with conv3d.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int = 1,
        padding: Sequence[int] | int = 0,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.unpadded_out_channels = out_channels
        self.TILE_WIDTH = 32
        self.out_channels = self.TILE_WIDTH if out_channels < self.TILE_WIDTH else out_channels
        if self.out_channels != self.unpadded_out_channels:
            logger.warning(f"Padding out_channels from {self.unpadded_out_channels} to {self.out_channels}")

        self.kernel_size = _ntuple(kernel_size, 3)
        self.stride = _ntuple(stride, 3)
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        padding = _ntuple(padding, 3)
        external_padding = list(padding)
        internal_padding = list(padding)
        # t padding is handled explicitly and depends on the cache.
        external_padding[0] = 2 * padding[0]
        internal_padding[0] = 0
        # HW padding may be handled by the halo CCL if the model is parallelized
        if self.parallel_config.height_parallel.factor > 1:
            external_padding[1] = padding[1]
            internal_padding[1] = 0
        if self.parallel_config.width_parallel.factor > 1:
            external_padding[2] = padding[2]
            internal_padding[2] = 0
        self.external_padding = tuple(external_padding)
        self.internal_padding = tuple(internal_padding)

        self.conv_config = get_conv3d_config(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            dtype,
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
        )
        logger.info(f"Loaded conv_config: {self.conv_config}")

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        d = self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2] * self.in_channels
        self.weight = Parameter(total_shape=[d, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)
        self.bias = Parameter(total_shape=[1, self.out_channels], device=mesh_device, pad_value=0, dtype=dtype)

        self.mask_cache = {}

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state and "bias" in state:
            state["weight"], state["bias"] = prepare_conv3d_weights(
                state["weight"].unsqueeze(2), state["bias"], self.conv_config
            )

    def get_cached_mask(self, x_BTHWC, logical_h):
        sharded_h = x_BTHWC.shape[2]
        key = (sharded_h, logical_h)
        if key not in self.mask_cache:
            padded_h = sharded_h * self.parallel_config.height_parallel.factor
            mask_shape = (1, 1, padded_h, 1, 1)
            mask = torch.ones(mask_shape)
            mask[:, :, logical_h:, :, :] = 0.0
            mapper_dims = [None, None]
            mapper_dims[self.parallel_config.height_parallel.mesh_axis] = 2
            mask = ttnn.from_torch(
                mask,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                dtype=self.dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=mapper_dims
                ),
            )
            self.mask_cache[key] = mask
        return self.mask_cache[key]

    def forward(self, x_BTHWC: ttnn.Tensor, logical_h: int) -> ttnn.Tensor:
        if x_BTHWC.shape[2] * self.parallel_config.height_parallel.factor != logical_h:
            """
            H is padded to divide by H parallel factor. Must zero out padded portion of H.
            """
            mask = self.get_cached_mask(x_BTHWC, logical_h)
            x_BTHWC = ttnn.mul(x_BTHWC, mask)

        # Height halo
        if self.external_padding[1] > 0 and self.parallel_config.height_parallel.factor > 1:
            ttnn.synchronize_device(x_BTHWC.device())
            x_BTHWC = ttnn.experimental.neighbor_pad_async(
                x_BTHWC,
                dim=2,
                padding_left=self.external_padding[1],
                padding_right=self.external_padding[1],
                padding_mode="zeros",
                cluster_axis=self.parallel_config.height_parallel.mesh_axis,
                # neighbor_pad only needs one final semaphore, so index into ag semahpore list
                final_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.height_parallel.mesh_axis
                )[0],
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(
                    self.parallel_config.height_parallel.mesh_axis
                ),
                num_links=get_neighbor_pad_num_links(self.ccl_manager, x_BTHWC, 2),
                # memory_config=mem_config_output,
                topology=self.ccl_manager.topology,
            )
            ttnn.synchronize_device(x_BTHWC.device())
        # Width halo
        if self.external_padding[2] > 0 and self.parallel_config.width_parallel.factor > 1:
            # TODO: Fix validation in neighbor_pad_async to allow halo on dim3
            x_THWC = ttnn.squeeze(x_BTHWC, dim=0)
            ttnn.synchronize_device(x_THWC.device())
            x_THWC = ttnn.experimental.neighbor_pad_async(
                x_THWC,
                dim=2,
                padding_left=self.external_padding[2],
                padding_right=self.external_padding[2],
                padding_mode="zeros",
                cluster_axis=self.parallel_config.width_parallel.mesh_axis,
                # neighbor_pad only needs one final semaphore, so index into ag semahpore list
                final_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.width_parallel.mesh_axis
                )[0],
                barrier_semaphore=self.ccl_manager.get_barrier_semaphore(self.parallel_config.width_parallel.mesh_axis),
                num_links=get_neighbor_pad_num_links(self.ccl_manager, x_THWC, 2),
                # memory_config=mem_config_output,
                topology=self.ccl_manager.topology,
            )
            ttnn.synchronize_device(x_THWC.device())
            x_BTHWC = ttnn.unsqueeze(x_THWC, dim=0)

        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
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

        if x_BTHWC.shape[2] * self.parallel_config.height_parallel.factor != logical_h:
            """
            H is padded to divide by H parallel factor. Must zero out padded portion of H.
            """
            mask = self.get_cached_mask(x_BTHWC, logical_h)
            x_BTHWC = ttnn.mul(x_BTHWC, mask)
        return x_BTHWC


class WanResample(Module):
    def __init__(
        self,
        *,
        dim: int,
        mode: str,
        resample_out_dim: int | None = None,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.mode = mode
        self.mesh_device = mesh_device
        resample_out_dim = resample_out_dim or (dim // 2 if "upsample" in mode else dim)

        assert mode in ["upsample2d", "upsample3d", "downsample2d", "downsample3d"]

        self.conv = WanConv2d(
            in_channels=dim,
            out_channels=resample_out_dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        self.is_upsample = "upsample" in mode
        self.is_3d = "3d" in mode

        if self.is_3d:
            self.time_conv = WanCausalConv3d(
                in_channels=dim,
                out_channels=dim * 2 if self.is_upsample else dim,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0) if self.is_upsample else (0, 0, 0),
                stride=(1, 1, 1) if self.is_upsample else (2, 1, 1),
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "resample.1", "conv")

    def forward(
        self,
        x_BTHWC,
        logical_h,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
    ) -> tuple[ttnn.Tensor, int]:
        B, T, H, W, C = x_BTHWC.shape
        if self.is_3d and self.is_upsample:  # upsample3d
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    t_start = x_BTHWC.shape[1] - CACHE_T
                    cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
                    is_rep = isinstance(feat_cache[idx], str) and feat_cache[idx] == "Rep"
                    assert not (
                        isinstance(feat_cache[idx], str) and not is_rep
                    ), "If feat_cache[idx] is a string, it must be 'Rep'"
                    if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None and not is_rep:
                        cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)

                    if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None and is_rep:
                        # When feat_cache[idx] is "Rep", we need to pad the cache_x_BTHWC with zeros
                        # Padding only works on the lowest 3 dims
                        cache_x_B1NC = ttnn.reshape(cache_x_BTHWC, (B, 1, H * W, C))
                        cache_x_BTNC = ttnn.pad(cache_x_B1NC, [(0, 0), (1, 0), (0, 0), (0, 0)], value=0.0)
                        cache_x_BTHWC = ttnn.reshape(cache_x_BTNC, (B, 2, H, W, C))

                    if is_rep:
                        x_time_BTHWU = self.time_conv(x_BTHWC, logical_h)
                    else:
                        x_time_BTHWU = self.time_conv(x_BTHWC, logical_h, feat_cache[idx])
                    x_BTHWU = x_time_BTHWU
                    feat_cache[idx] = cache_x_BTHWC
                    feat_idx[0] += 1

                    T1 = x_BTHWU.shape[1]
                    x_BTHW2C = ttnn.reshape(x_BTHWU, (B, T1, H, W, 2, C))
                    x_BT2HWC = ttnn.permute(x_BTHW2C, (0, 1, 4, 2, 3, 5))
                    x_BTHWC = ttnn.reshape(x_BT2HWC, (B, T1 * 2, H, W, C))
            else:
                raise ValueError("feat_cache cannot be None")

        if self.is_upsample:
            T2 = x_BTHWC.shape[1]
            x_NHWC = ttnn.reshape(x_BTHWC, (B * T2, H, W, C))
            x_upsamped_NHWC = ttnn.upsample(x_NHWC, scale_factor=2)
            logical_h *= 2
            H2, W2 = x_upsamped_NHWC.shape[1], x_upsamped_NHWC.shape[2]
            x_BTHWC = ttnn.reshape(x_upsamped_NHWC, (B, T2, H2, W2, C))
            x_conv_BTHWC = self.conv(x_BTHWC, logical_h)
        else:
            x_conv_BTHWC = self.conv(x_BTHWC, logical_h)
            x_conv_BTHWC = x_conv_BTHWC[
                :, :, 1::2, 1::2, :
            ]  # 2x2 strided convolution output to support patched conv, with only right and bottom padding
            logical_h //= 2

        # Handle downsample3d
        if self.is_3d and not self.is_upsample:  # downsample3d
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = ttnn.clone(x_conv_BTHWC)
                    feat_idx[0] += 1
                else:
                    cache_x_BTHWC = ttnn.clone(x_conv_BTHWC[:, -1:, :, :, :])
                    x_conv_BTHWC = self.time_conv(
                        ttnn.concat([feat_cache[idx][:, -1:, :, :, :], x_conv_BTHWC], dim=1), logical_h
                    )
                    feat_cache[idx] = cache_x_BTHWC
                    feat_idx[0] += 1
            else:
                raise ValueError("feat_cache cannot be None")
        return x_conv_BTHWC, logical_h


class WanUpBlock(Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None = None,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_res_blocks = num_res_blocks
        self.upsample_mode = upsample_mode
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        assert upsample_mode in ["upsample2d", "upsample3d"] or upsample_mode is None

        resnets = ModuleList()
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(
                    in_dim=current_dim,
                    out_dim=out_dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                    dtype=dtype,
                )
            )
            current_dim = out_dim
        self.resnets = resnets

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = WanResample(
                dim=out_dim,
                mode=upsample_mode,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
            )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        rename_substate(state, "upsamplers.0", "upsamplers")

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
    ) -> tuple[ttnn.Tensor, int]:
        for resnet in self.resnets:
            x_res_BTHWC = resnet(x_BTHWC, logical_h, feat_cache, feat_idx)
            x_BTHWC = x_res_BTHWC
        if self.upsamplers is not None:
            x_upsampled_BTHWC, logical_h = self.upsamplers(x_BTHWC, logical_h, feat_cache, feat_idx)
            x_BTHWC = x_upsampled_BTHWC
        return x_BTHWC, logical_h


class WanDecoder3d(Module):
    def __init__(
        self,
        *,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temperal_upsample: Sequence[bool] = (False, True, True),
        out_channels: int = 3,
        is_residual: bool = False,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        assert not is_residual, "is_residual is not supported"
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        # init block
        self.conv_in = WanCausalConv3d(
            z_dim,
            dims[0],
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        # middle blocks
        self.mid_block = WanMidBlock(
            dim=dims[0],
            num_layers=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        # upsample blocks
        self.up_blocks = ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i > 0 and not is_residual:
                # wan vae 2.1
                in_dim = in_dim // 2

            # determine if we need upsampling
            up_flag = i != len(dim_mult) - 1
            # determine upsampling mode, if not upsampling, set to None
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"
            # Create and add the upsampling block
            # NOTE: Different codepath if is_residual. Not implemented yet.
            up_block = WanUpBlock(
                in_dim=in_dim,
                out_dim=out_dim,
                num_res_blocks=num_res_blocks,
                upsample_mode=upsample_mode,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
                dtype=dtype,
            )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.conv_out = WanCausalConv3d(
            out_dim,
            out_channels,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm_out.gamma" in state:
            state["norm_out.weight"] = state.pop("norm_out.gamma").squeeze()

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
        first_chunk: bool = False,
    ) -> tuple[ttnn.Tensor, int]:
        # NOTE: first_chunk is not used. It would be needed for WanResidualUpBlock.
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_in(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_in(x_BTHWC, logical_h)

        ## middle
        x_BTHWC = self.mid_block(x_BTHWC, logical_h, feat_cache, feat_idx)

        ## upsamples
        for up_block in self.up_blocks:
            x_BTHWC, logical_h = up_block(x_BTHWC, logical_h, feat_cache, feat_idx)

        ## head
        x_tile_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
        x_norm_tile_BTHWC = self.norm_out(x_tile_BTHWC)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_out(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_out(x_BTHWC, logical_h)
        return x_BTHWC, logical_h


class WanDecoder(Module):
    def __init__(
        self,
        *,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales=(),
        temperal_downsample: Sequence[bool] = (False, True, True),
        latents_mean: Sequence[float] = (
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ),
        latents_std: Sequence[float] = (
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ),
        is_residual: bool = False,
        in_channels: int = 3,
        out_channels: int = 3,
        mesh_device: ttnn.MeshDevice,
        parallel_config: VaeHWParallelConfig,
        ccl_manager: CCLManager,
        dtype: ttnn.DataType = ttnn.DataType.FLOAT32,
    ) -> None:
        super().__init__()

        assert not is_residual, "is_residual is not supported"
        self.z_dim = z_dim
        self.temperal_upsample = temperal_downsample[::-1]
        self.out_channels = out_channels
        decoder_base_dim = decoder_base_dim or base_dim

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # Linear for post_quant_conv
        self.post_quant_conv = Linear(
            in_features=aligned_channels(z_dim),
            out_features=aligned_channels(z_dim),
            mesh_device=mesh_device,
            dtype=dtype,
        )

        self.decoder = WanDecoder3d(
            dim=decoder_base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_upsample=self.temperal_upsample,
            out_channels=out_channels,
            is_residual=is_residual,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        self.cached_conv_count = count_convs(self.decoder)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "post_quant_conv.weight" in state and "post_quant_conv.bias" in state:
            state_sub = conv3d_to_linear_weight(pop_substate(state, "post_quant_conv"))
            state["post_quant_conv.weight"] = state_sub["weight"]
            state["post_quant_conv.bias"] = state_sub["bias"]

        pop_substate(state, "encoder")
        pop_substate(state, "quant_conv")

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_cache = [None] * self.cached_conv_count

    def forward(self, z_BTHWC: ttnn.Tensor, logical_h: int) -> tuple[ttnn.Tensor, int]:
        B, T, H, W, C = z_BTHWC.shape

        self.clear_cache()
        z_tile_BTHWC = ttnn.to_layout(z_BTHWC, ttnn.TILE_LAYOUT)
        x_tile_BTHWC = self.post_quant_conv(z_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        output_BCTHW = None
        for i in range(T):
            # Process one frame at a time
            self._conv_idx = [0]
            out_BTHWC, new_logical_h = self.decoder(
                x_BTHWC[:, i : i + 1, :, :, :], logical_h, feat_cache=self._feat_cache, feat_idx=self._conv_idx
            )
            # Channels first
            out_BCTHW = ttnn.permute(out_BTHWC, (0, 4, 1, 2, 3))
            # Trim padding on output channels
            out_BCTHW = out_BCTHW[:, : self.out_channels, :, :, :]
            if output_BCTHW is None:
                output_BCTHW = out_BCTHW
            else:
                output_BCTHW = ttnn.concat([output_BCTHW, out_BCTHW], dim=2)

        output_tile_BCTHW = ttnn.to_layout(output_BCTHW, ttnn.TILE_LAYOUT)
        output_BCTHW = ttnn.clamp(output_tile_BCTHW, min=-1.0, max=1.0)
        output_BCTHW = ttnn.to_layout(output_BCTHW, ttnn.ROW_MAJOR_LAYOUT)
        self.clear_cache()
        return (output_BCTHW, new_logical_h)


class WanEncoder3D(Module):
    def __init__(
        self,
        in_channels: int = 3,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        is_residual: bool = False,  # wan 2.2 vae use a residual downblock
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ) -> None:
        super().__init__()

        assert not is_residual, "is_residual is not supported"
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv_in = WanCausalConv3d(
            in_channels,
            dims[0],
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        # downsample blocks.
        self.down_blocks = ModuleList()
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                self.down_blocks.append(
                    WanResidualBlock(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        mesh_device=mesh_device,
                        ccl_manager=ccl_manager,
                        parallel_config=parallel_config,
                        dtype=dtype,
                    )
                )
                if scale in attn_scales:
                    self.down_blocks.append(
                        WanAttentionBlock(
                            dim=out_dim,
                            mesh_device=mesh_device,
                            ccl_manager=ccl_manager,
                            parallel_config=parallel_config,
                            dtype=dtype,
                        )
                    )
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                self.down_blocks.append(
                    WanResample(
                        dim=out_dim,
                        mode=mode,
                        mesh_device=mesh_device,
                        ccl_manager=ccl_manager,
                        parallel_config=parallel_config,
                        dtype=dtype,
                    )
                )
                scale /= 2.0

        # middle blocks
        self.mid_block = WanMidBlock(
            dim=out_dim,
            num_layers=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

        # output blocks
        self.norm_out = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-12,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.conv_out = WanCausalConv3d(
            out_dim,
            z_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "norm_out.gamma" in state:
            state["norm_out.weight"] = state.pop("norm_out.gamma").squeeze()

    def forward(
        self,
        x_BTHWC: ttnn.Tensor,
        logical_h: int,
        feat_cache: list[ttnn.Tensor] | None = None,
        feat_idx: list[int] = [0],
    ) -> tuple[ttnn.Tensor, int]:
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_in(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_in(x_BTHWC, logical_h)

        ## downsamples
        for down_block in self.down_blocks:
            if isinstance(down_block, WanResample):
                x_BTHWC, logical_h = down_block(x_BTHWC, logical_h, feat_cache, feat_idx)
            elif isinstance(down_block, WanResidualBlock):
                x_BTHWC = down_block(x_BTHWC, logical_h, feat_cache, feat_idx)
            elif isinstance(down_block, WanAttentionBlock):
                x_BTHWC = down_block(x_BTHWC, logical_h)
            else:
                raise ValueError(f"Unsupported downblock type: {type(down_block)}")

        ## middle
        x_BTHWC = self.mid_block(x_BTHWC, logical_h, feat_cache, feat_idx)

        ## head
        x_tile_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
        x_norm_tile_BTHWC = self.norm_out(x_tile_BTHWC)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)
        x_BTHWC = ttnn.to_layout(x_silu_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)

        if feat_cache is not None:
            idx = feat_idx[0]
            t_start = x_BTHWC.shape[1] - CACHE_T
            cache_x_BTHWC = x_BTHWC[:, t_start:, :, :, :]
            if cache_x_BTHWC.shape[1] < 2 and feat_cache[idx] is not None:
                # Current activation is too short, so append the cached activation as well
                cache_x_BTHWC = ttnn.concat([feat_cache[idx][:, -1:, :, :, :], cache_x_BTHWC], dim=1)
            x_BTHWC = self.conv_out(x_BTHWC, logical_h, feat_cache[idx])
            feat_cache[idx] = cache_x_BTHWC
            feat_idx[0] += 1
        else:
            x_BTHWC = self.conv_out(x_BTHWC, logical_h)
        return x_BTHWC, logical_h


class WanEncoder(Module):
    def __init__(
        self,
        base_dim=96,
        in_channels: int = 3,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        is_residual: bool = False,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ) -> None:
        super().__init__()

        self.z_dim = z_dim
        self.out_channels = z_dim * 2  # Mean and logvar
        self.encoder = WanEncoder3D(
            in_channels=in_channels,
            dim=base_dim,
            z_dim=self.out_channels,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            is_residual=is_residual,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
            dtype=dtype,
        )
        # Linear for quant_conv
        self.quant_conv = Linear(
            in_features=aligned_channels(self.out_channels),
            out_features=aligned_channels(self.out_channels),
            mesh_device=mesh_device,
            dtype=dtype,
        )
        self.cached_conv_count = count_convs(self.encoder)

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "quant_conv.weight" in state and "quant_conv.bias" in state:
            state_sub = conv3d_to_linear_weight(pop_substate(state, "quant_conv"))
            state["quant_conv.weight"] = state_sub["weight"]
            state["quant_conv.bias"] = state_sub["bias"]

        pop_substate(state, "decoder")
        pop_substate(state, "post_quant_conv")

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_cache = [None] * self.cached_conv_count

    def forward(self, x_BTHWC: ttnn.Tensor, logical_h: int) -> tuple[ttnn.Tensor, int]:
        B, T, H, W, C = x_BTHWC.shape

        self.clear_cache()

        output_BTHWC = None
        T_encoded = 1 + (T - 1) // 4
        for i in range(T_encoded):
            # Process one frame at a time
            self._conv_idx = [0]
            if i == 0:
                x_BTHWC_chunk = x_BTHWC[:, :1, :, :, :]
            else:
                x_BTHWC_chunk = x_BTHWC[:, 1 + 4 * (i - 1) : 1 + 4 * i, :, :, :]

            out_BTHWC, new_logical_h = self.encoder(
                x_BTHWC_chunk, logical_h, feat_cache=self._feat_cache, feat_idx=self._conv_idx
            )

            if output_BTHWC is None:
                output_BTHWC = out_BTHWC
            else:
                output_BTHWC = ttnn.concat([output_BTHWC, out_BTHWC], dim=1)

        output_tile_BTHWC = ttnn.to_layout(output_BTHWC, ttnn.TILE_LAYOUT)
        output_tile_BTHWC = self.quant_conv(output_tile_BTHWC)
        output_BTHWC = ttnn.to_layout(output_tile_BTHWC, ttnn.ROW_MAJOR_LAYOUT)
        # Permute to channel second expected by torch
        output_BCTHW = ttnn.permute(output_BTHWC, (0, 4, 1, 2, 3))
        # Trim padding on output channels
        output_BCTHW = output_BCTHW[:, : self.z_dim, :, :, :]  # Get the mean
        self.clear_cache()
        return (output_BCTHW, new_logical_h)


def get_neighbor_pad_num_links(ccl_manager, input_tensor, dim):
    """
    Neighbor pad can use no more links than the product of the upper dimensions of the input tensor.
    """
    upper_dims = 1
    for i in range(dim):
        upper_dims *= input_tensor.shape[i]
    return min(upper_dims, ccl_manager.num_links)

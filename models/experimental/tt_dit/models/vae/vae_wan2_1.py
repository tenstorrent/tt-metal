# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from loguru import logger
from ...layers.normalization import RMSNorm
from ...layers.linear import Linear
from ...utils.conv3d import _ntuple, get_conv3d_config, prepare_conv3d_weights, count_convs, aligned_channels
from ...utils.substate import substate, indexed_substates
from ...utils.tensor import bf16_tensor

CACHE_T = 2


class WanAttentionBlock:
    def __init__(
        self,
        dim,
        mesh_device,
        parallel_config,
        ccl_manager,
    ):
        self.dim = dim
        self.mesh_device = mesh_device
        self.parallel_config = parallel_config
        self.ccl_manager = ccl_manager

        self.norm = RMSNorm(
            embedding_dim=dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.to_qkv = Linear(
            in_features=dim,
            out_features=dim * 3,
            mesh_device=mesh_device,
        )
        self.proj = Linear(
            in_features=dim,
            out_features=dim,
            mesh_device=mesh_device,
        )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
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
            packer_l1_acc=True,
        )
        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def load_state_dict(self, state_dict):
        def permute_conv2d_weights(weight):
            out_c, in_c, kh, kw = weight.shape
            assert kh == kw == 1
            weight = weight.permute(0, 2, 3, 1).reshape(out_c, in_c)
            return weight

        self.to_qkv.load_state_dict(
            {
                "weight": permute_conv2d_weights(state_dict["to_qkv.weight"]),
                "bias": state_dict["to_qkv.bias"],
            }
        )
        self.proj.load_state_dict(
            {
                "weight": permute_conv2d_weights(state_dict["proj.weight"]),
                "bias": state_dict["proj.bias"],
            }
        )

        self.norm.load_state_dict(
            {
                "weight": state_dict["norm.gamma"].squeeze(),
            }
        )

    def __call__(self, x_BTHWC, logical_h):
        """
        x_BTHWC: (B, T, H, W, C) fractured on H and W

        returns: (B, T, H, W, C) fractured on H and W
        """
        assert len(x_BTHWC.shape) == 5
        assert x_BTHWC.layout == ttnn.ROW_MAJOR_LAYOUT
        residual_BTHWC = x_BTHWC

        # Gather height and width for replicated attention
        if self.parallel_config.height_parallel.factor > 1:
            x_BTHWC = ttnn.experimental.all_gather_async(
                x_BTHWC,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    x_BTHWC.shape, 2, self.parallel_config.height_parallel.mesh_axis
                ),
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.height_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.height_parallel.mesh_axis,
            )
        if self.parallel_config.width_parallel.factor > 1:
            x_BTHWC = ttnn.experimental.all_gather_async(
                x_BTHWC,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    x_BTHWC.shape, 3, self.parallel_config.width_parallel.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                    self.parallel_config.width_parallel.mesh_axis
                ),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.parallel_config.width_parallel.mesh_axis,
            )

        if logical_h % self.parallel_config.height_parallel.factor != 0:
            """
            H is padded, so slice it out before attention
            """
            padded_h = x_BTHWC.shape[2]
            x_BTHWC = x_BTHWC[:, :, :logical_h, :, :]
        B, T, H, W, C = x_BTHWC.shape
        x_TNC = ttnn.reshape(x_BTHWC, (B * T, H * W, C))
        x_TNC = ttnn.to_layout(x_TNC, ttnn.TILE_LAYOUT)
        x_TNC = self.norm(x_TNC, compute_kernel_config=self.hifi4_compute_kernel_config)
        x_TND = self.to_qkv(x_TNC, compute_kernel_config=self.mm_compute_kernel_config, core_grid=self.core_grid)
        q_THNC, k_THNC, v_THNC = ttnn.transformer.split_query_key_value_and_split_heads(
            x_TND, num_heads=1, transpose_key=False
        )
        out_THNC = ttnn.transformer.scaled_dot_product_attention(
            q_THNC,
            k_THNC,
            v_THNC,
            is_causal=False,
            program_config=self.sdpa_program_config,
            compute_kernel_config=self.sdpa_compute_kernel_config,
        )
        out_TNC = ttnn.transformer.concatenate_heads(out_THNC)
        out_TND = self.proj(out_TNC, compute_kernel_config=self.mm_compute_kernel_config, core_grid=self.core_grid)
        out_TND = ttnn.to_layout(out_TND, ttnn.ROW_MAJOR_LAYOUT)

        if logical_h % self.parallel_config.height_parallel.factor != 0:
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

        return out_BTHWC + residual_BTHWC


class WanCausalConv3d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mesh_device,
        stride=1,
        padding=0,
        ccl_manager=None,
        parallel_config=None,
    ):
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
            self.stride,
            self.internal_padding,
            padding_mode="zeros",
            grid_size=self.mesh_device.compute_with_storage_grid_size(),
        )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        self.mask_cache = {}

    def load_state_dict(self, state_dict):
        def maybe_pad_out_channels(weight, bias):
            if self.out_channels != self.unpadded_out_channels:
                weight = torch.nn.functional.pad(
                    weight, (0, 0, 0, 0, 0, 0, 0, 0, 0, self.out_channels - self.unpadded_out_channels)
                )
                bias = torch.nn.functional.pad(bias, (0, self.out_channels - self.unpadded_out_channels))
            return weight, bias

        padded_weight, padded_bias = maybe_pad_out_channels(state_dict["weight"], state_dict["bias"])

        self.conv_weight, self.conv_bias = prepare_conv3d_weights(
            self.mesh_device,
            padded_weight,
            padded_bias,
            self.conv_config,
        )

    def get_cached_mask(self, x_BTHWC, logical_h):
        sharded_h = x_BTHWC.shape[2]
        key = (sharded_h, logical_h)
        if key not in self.mask_cache:
            padded_h = sharded_h * self.parallel_config.height_parallel.factor
            mask_shape = (1, 1, padded_h, 1, 1)
            mask = torch.ones(mask_shape)
            mask[:, :, logical_h:, :, :] = 0.0
            mask = bf16_tensor(
                mask,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_axis=self.parallel_config.height_parallel.mesh_axis,
                shard_dim=2,
            )
            self.mask_cache[key] = mask
        return self.mask_cache[key]

    def __call__(self, x_BTHWC, logical_h, cache_x_BTHWC=None):
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

        if logical_h % self.parallel_config.height_parallel.factor != 0:
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
                num_links=1,  # Forcing to 1 because on 6U, not enough work to split among links
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
                num_links=1,  # Forcing to 1 because on 6U, not enough work to split among links
                # memory_config=mem_config_output,
                topology=self.ccl_manager.topology,
            )
            ttnn.synchronize_device(x_THWC.device())
            x_BTHWC = ttnn.unsqueeze(x_THWC, dim=0)

        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.conv_weight,
            bias_tensor=self.conv_bias,
            config=self.conv_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        if logical_h % self.parallel_config.height_parallel.factor != 0:
            """
            H is padded to divide by H parallel factor. Must zero out padded portion of H.
            """
            mask = self.get_cached_mask(x_BTHWC, logical_h)
            x_BTHWC = ttnn.mul(x_BTHWC, mask)
        return x_BTHWC


class WanResidualBlock:
    def __init__(
        self,
        in_dim,
        out_dim,
        mesh_device,
        ccl_manager=None,
        parallel_config=None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mesh_device = mesh_device

        self.norm1 = RMSNorm(
            embedding_dim=in_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.conv1 = WanCausalConv3d(
            in_channels=in_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )
        self.norm2 = RMSNorm(
            embedding_dim=out_dim,
            norm_eps=1e-6,
            norm_elementwise_affine=True,
            bias=False,
            mesh_device=mesh_device,
        )
        self.conv2 = WanCausalConv3d(
            in_channels=out_dim,
            out_channels=out_dim,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        if in_dim != out_dim:
            self.conv_shortcut = Linear(
                in_features=in_dim,
                out_features=out_dim,
                mesh_device=mesh_device,
            )
        else:
            self.conv_shortcut = None

        self.hifi4_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

    def load_state_dict(self, state_dict):
        def rename_norm_state(state):
            return {"weight": state["gamma"].squeeze()}

        self.norm1.load_state_dict(rename_norm_state(substate(state_dict, "norm1")))
        self.norm2.load_state_dict(rename_norm_state(substate(state_dict, "norm2")))
        self.conv1.load_state_dict(substate(state_dict, "conv1"))
        self.conv2.load_state_dict(substate(state_dict, "conv2"))

        def conv_1d_to_matmul_weight(weight):
            out_c, in_c, kt, kh, kw = weight.shape
            assert kt == kh == kw == 1
            weight = weight.reshape(out_c, in_c)
            return weight

        if self.conv_shortcut is not None:
            self.conv_shortcut.load_state_dict(
                {
                    "weight": conv_1d_to_matmul_weight(state_dict["conv_shortcut.weight"]),
                    "bias": state_dict["conv_shortcut.bias"],
                }
            )

    def __call__(self, x_BTHWC, logical_h, feat_cache=None, feat_idx=[0]):
        x_tile_BTHWC = ttnn.to_layout(x_BTHWC, ttnn.TILE_LAYOUT)
        h_tile_BTHWC = (
            self.conv_shortcut(
                x_tile_BTHWC, compute_kernel_config=self.hifi4_compute_kernel_config, core_grid=self.core_grid
            )
            if self.conv_shortcut is not None
            else x_tile_BTHWC
        )
        x_norm_tile_BTHWC = self.norm1(x_tile_BTHWC, compute_kernel_config=self.hifi4_compute_kernel_config)
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)  # NOTE: potential correctness issue
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
        x_silu_tile_BTHWC = ttnn.silu(x_norm_tile_BTHWC)  # NOTE: potential correctness issue
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


class WanMidBlock:
    def __init__(
        self,
        dim,
        mesh_device,
        ccl_manager=None,
        parallel_config=None,
        num_layers=1,
    ):
        self.dim = dim
        self.mesh_device = mesh_device
        resnets = []
        attentions = []

        resnets.append(
            WanResidualBlock(
                in_dim=dim,
                out_dim=dim,
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )
        )

        for _ in range(num_layers):
            attentions.append(
                WanAttentionBlock(
                    dim=dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                )
            )
            resnets.append(
                WanResidualBlock(
                    in_dim=dim,
                    out_dim=dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
                )
            )

        self.resnets = resnets
        self.attentions = attentions

    def load_state_dict(self, state_dict):
        for i in range(len(self.resnets)):
            self.resnets[i].load_state_dict(substate(state_dict, f"resnets.{i}"))
        for i in range(len(self.attentions)):
            self.attentions[i].load_state_dict(substate(state_dict, f"attentions.{i}"))

    def __call__(self, x_BTHWC, logical_h, feat_cache=None, feat_idx=[0]):
        x_res_BTHWC = self.resnets[0](x_BTHWC, logical_h, feat_cache, feat_idx)
        x_BTHWC = x_res_BTHWC
        for i in range(len(self.attentions)):
            x_attn_BTHWC = self.attentions[i](x_BTHWC, logical_h)
            x_BTHWC = self.resnets[i + 1](x_attn_BTHWC, logical_h, feat_cache, feat_idx)
        return x_BTHWC


class WanConv2d:
    """
    A conv2d implemented with conv3d.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        mesh_device,
        ccl_manager=None,
        parallel_config=None,
        stride=1,
        padding=0,
    ):
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
            self.stride,
            self.internal_padding,
            padding_mode="zeros",
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

        self.mask_cache = {}

    def load_state_dict(self, state_dict):
        def conv2d_to_conv3d_weight(weight):
            weight = weight.unsqueeze(2)
            return weight

        reshaped_weight = conv2d_to_conv3d_weight(state_dict["weight"])

        self.conv_weight, self.conv_bias = prepare_conv3d_weights(
            self.mesh_device,
            reshaped_weight,
            state_dict["bias"],
            self.conv_config,
        )

    def get_cached_mask(self, x_BTHWC, logical_h):
        sharded_h = x_BTHWC.shape[2]
        key = (sharded_h, logical_h)
        if key not in self.mask_cache:
            padded_h = sharded_h * self.parallel_config.height_parallel.factor
            mask_shape = (1, 1, padded_h, 1, 1)
            mask = torch.ones(mask_shape)
            mask[:, :, logical_h:, :, :] = 0.0
            mask = bf16_tensor(
                mask,
                device=self.mesh_device,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                mesh_axis=self.parallel_config.height_parallel.mesh_axis,
                shard_dim=2,
            )
            self.mask_cache[key] = mask
        return self.mask_cache[key]

    def __call__(self, x_BTHWC, logical_h):
        if logical_h % self.parallel_config.height_parallel.factor != 0:
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
                num_links=1,  # Forcing to 1 because on 6U, not enough work to split among links
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
                num_links=1,  # Forcing to 1 because on 6U, not enough work to split among links
                # memory_config=mem_config_output,
                topology=self.ccl_manager.topology,
            )
            ttnn.synchronize_device(x_THWC.device())
            x_BTHWC = ttnn.unsqueeze(x_THWC, dim=0)

        x_BTHWC = ttnn.experimental.conv3d(
            input_tensor=x_BTHWC,
            weight_tensor=self.conv_weight,
            bias_tensor=self.conv_bias,
            config=self.conv_config,
            compute_kernel_config=self.compute_kernel_config,
        )

        if logical_h % self.parallel_config.height_parallel.factor != 0:
            """
            H is padded to divide by H parallel factor. Must zero out padded portion of H.
            """
            mask = self.get_cached_mask(x_BTHWC, logical_h)
            x_BTHWC = ttnn.mul(x_BTHWC, mask)
        return x_BTHWC


class WanResample:
    def __init__(
        self,
        dim,
        mode,
        mesh_device,
        ccl_manager=None,
        parallel_config=None,
        upsample_out_dim=None,
    ):
        self.dim = dim
        self.mode = mode
        self.mesh_device = mesh_device
        upsample_out_dim = upsample_out_dim or dim // 2

        assert mode in ["upsample2d", "upsample3d"]

        self.conv = WanConv2d(
            in_channels=dim,
            out_channels=upsample_out_dim,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

        if mode == "upsample3d":
            self.time_conv = WanCausalConv3d(
                in_channels=dim,
                out_channels=dim * 2,
                kernel_size=(3, 1, 1),
                padding=(1, 0, 0),
                mesh_device=mesh_device,
                ccl_manager=ccl_manager,
                parallel_config=parallel_config,
            )

    def load_state_dict(self, state_dict):
        self.conv.load_state_dict(substate(state_dict, "resample.1"))

        if self.mode == "upsample3d":
            self.time_conv.load_state_dict(substate(state_dict, "time_conv"))

    def __call__(self, x_BTHWC, logical_h, feat_cache=None, feat_idx=[0]):
        B, T, H, W, C = x_BTHWC.shape
        if self.mode == "upsample3d":
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

        T2 = x_BTHWC.shape[1]
        x_NHWC = ttnn.reshape(x_BTHWC, (B * T2, H, W, C))
        x_upsamped_NHWC = ttnn.upsample(x_NHWC, scale_factor=2)
        logical_h *= 2
        H2, W2 = x_upsamped_NHWC.shape[1], x_upsamped_NHWC.shape[2]
        x_BTHWC = ttnn.reshape(x_upsamped_NHWC, (B, T2, H2, W2, C))
        x_conv_BTHWC = self.conv(x_BTHWC, logical_h)
        return x_conv_BTHWC, logical_h


class WanUpBlock:
    def __init__(
        self,
        in_dim,
        out_dim,
        num_res_blocks,
        mesh_device,
        ccl_manager=None,
        parallel_config=None,
        upsample_mode=None,
    ):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_res_blocks = num_res_blocks
        self.upsample_mode = upsample_mode
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        assert upsample_mode in ["upsample2d", "upsample3d"] or upsample_mode is None

        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(
                WanResidualBlock(
                    in_dim=current_dim,
                    out_dim=out_dim,
                    mesh_device=mesh_device,
                    ccl_manager=ccl_manager,
                    parallel_config=parallel_config,
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
            )

    def load_state_dict(self, state_dict):
        for i in range(len(self.resnets)):
            self.resnets[i].load_state_dict(substate(state_dict, f"resnets.{i}"))
        if self.upsamplers is not None:
            self.upsamplers.load_state_dict(substate(state_dict, "upsamplers.0"))

    def __call__(self, x_BTHWC, logical_h, feat_cache=None, feat_idx=[0]):
        for resnet in self.resnets:
            x_res_BTHWC = resnet(x_BTHWC, logical_h, feat_cache, feat_idx)
            x_BTHWC = x_res_BTHWC
        if self.upsamplers is not None:
            x_upsampled_BTHWC, logical_h = self.upsamplers(x_BTHWC, logical_h, feat_cache, feat_idx)
            x_BTHWC = x_upsampled_BTHWC
        return x_BTHWC, logical_h


class WanDecoder3d:
    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        out_channels: int = 3,
        is_residual: bool = False,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ):
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
        )

        # middle blocks
        self.mid_block = WanMidBlock(
            dims[0], num_layers=1, mesh_device=mesh_device, ccl_manager=ccl_manager, parallel_config=parallel_config
        )

        # upsample blocks
        self.up_blocks = []
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
            )
            self.up_blocks.append(up_block)

        # output blocks
        self.norm_out = RMSNorm(
            embedding_dim=out_dim, norm_eps=1e-6, norm_elementwise_affine=True, bias=False, mesh_device=mesh_device
        )
        self.conv_out = WanCausalConv3d(
            out_dim,
            out_channels,
            kernel_size=3,
            padding=1,
            mesh_device=mesh_device,
            ccl_manager=ccl_manager,
            parallel_config=parallel_config,
        )

    def load_state_dict(self, state_dict):
        self.conv_in.load_state_dict(substate(state_dict, "conv_in"))
        self.mid_block.load_state_dict(substate(state_dict, "mid_block"))
        for i, state in enumerate(indexed_substates(state_dict, "up_blocks")):
            self.up_blocks[i].load_state_dict(state)

        def rename_norm_state(state):
            return {"weight": state["gamma"].squeeze()}

        self.norm_out.load_state_dict(rename_norm_state(substate(state_dict, "norm_out")))
        self.conv_out.load_state_dict(substate(state_dict, "conv_out"))

    def __call__(self, x_BTHWC, logical_h, feat_cache=None, feat_idx=[0], first_chunk=False):
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
        # DEBUG
        # return x_BTHWC

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


class WanDecoder:
    def __init__(
        self,
        base_dim=96,
        decoder_base_dim=None,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        latents_mean=[
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
        ],
        latents_std=[
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
        ],
        is_residual=False,
        in_channels=3,
        out_channels=3,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
    ):
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
        )

        self.cached_conv_count = count_convs(self.decoder)

    def load_state_dict(self, state_dict):
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

        self.post_quant_conv.load_state_dict(conv3d_to_linear_weight(substate(state_dict, "post_quant_conv")))
        self.decoder.load_state_dict(substate(state_dict, "decoder"))

    def clear_cache(self):
        self._conv_idx = [0]
        self._feat_cache = [None] * self.cached_conv_count

    def __call__(self, z_BTHWC, logical_h):
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

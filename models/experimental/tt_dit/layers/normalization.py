# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ..utils.tensor import bf16_tensor
from loguru import logger
import math


class RMSNorm:
    def __init__(
        self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True, mesh_device=None, init=False
    ):
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = bias
        self.weight = None
        self.bias = None
        if norm_elementwise_affine and init:
            self.weight = bf16_tensor(torch.randn(1, embedding_dim), device=self.mesh_device)
            if bias:
                self.bias = bf16_tensor(torch.randn(1, embedding_dim), device=self.mesh_device)

    def load_state_dict(self, state_dict):
        if self.norm_elementwise_affine:
            self.weight = bf16_tensor(state_dict["weight"].unsqueeze(0), device=self.mesh_device)
            if self.use_bias:
                self.bias = bf16_tensor(state_dict["bias"].unsqueeze(0), device=self.mesh_device)

    def __call__(self, x):
        return ttnn.rms_norm(x, weight=self.weight, bias=self.bias, epsilon=self.norm_eps)


class LayerNorm:
    def __init__(
        self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True, mesh_device=None, init=False
    ):
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = bias
        self.weight = None
        self.bias = None
        if norm_elementwise_affine and init:
            self.weight = bf16_tensor(torch.randn(1, embedding_dim), device=self.mesh_device)
            if bias:
                self.bias = bf16_tensor(torch.randn(1, embedding_dim), device=self.mesh_device)

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def load_state_dict(self, state_dict):
        if self.norm_elementwise_affine:
            self.weight = bf16_tensor(state_dict["weight"].unsqueeze(0), device=self.mesh_device)
            if self.use_bias:
                self.bias = bf16_tensor(state_dict["bias"].unsqueeze(0), device=self.mesh_device)

    def __call__(self, x):
        return ttnn.layer_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.norm_eps,
            compute_kernel_config=self.compute_kernel_config,
        )


class DistributedLayerNorm:
    """
    Implements LayerNorm on an activation sharded on the reduction dimension.

    Requires gamma and beta, which will be created if not provided.
    """

    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=True,
        mesh_axis=0,
        mesh_device=None,
        ccl_manager=None,
        init=False,
    ):
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = bias
        self.mesh_axis = mesh_axis
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.weight = None
        self.bias = None
        self.mesh_width = tuple(mesh_device.shape)[mesh_axis]
        self.TILE_SIZE = 32
        if init or not (norm_elementwise_affine and bias):
            if not (norm_elementwise_affine and bias):
                pass  # TODO: make logging less noisy
                # logger.debug(
                #     "DistributedLayerNorm initialized with norm_elementwise_affine=False. Creating gamma and beta tensors to meet op requirements."
                # )
            weight = torch.ones(1, embedding_dim)
            weight = weight.reshape([-1, self.TILE_SIZE * self.mesh_width])
            bias = torch.zeros(1, embedding_dim)
            bias = bias.reshape([-1, self.TILE_SIZE * self.mesh_width])
            self.weight = bf16_tensor(
                weight, device=self.mesh_device, mesh_axis=mesh_axis, shard_dim=-1, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            self.bias = bf16_tensor(
                bias, device=self.mesh_device, mesh_axis=mesh_axis, shard_dim=-1, layout=ttnn.ROW_MAJOR_LAYOUT
            )

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def load_state_dict(self, state_dict):
        if self.norm_elementwise_affine:
            weight = state_dict["weight"]
            weight = (
                weight.reshape(self.mesh_width, -1, self.TILE_SIZE)
                .permute(1, 0, 2)
                .reshape(-1, self.TILE_SIZE * self.mesh_width)
            )
            self.weight = bf16_tensor(
                weight, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            if self.use_bias:
                bias = state_dict["bias"]
                bias = (
                    bias.reshape(self.mesh_width, -1, self.TILE_SIZE)
                    .permute(1, 0, 2)
                    .reshape(-1, self.TILE_SIZE * self.mesh_width)
                )
                self.bias = bf16_tensor(
                    bias, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1, layout=ttnn.ROW_MAJOR_LAYOUT
                )

    def __call__(self, x):
        assert (
            self.weight is not None and self.bias is not None
        ), "weight and bias must be initialized before calling __call__"
        stats = ttnn.layer_norm_pre_all_gather(x)

        stats_gathered = ttnn.experimental.all_gather_async(
            stats,
            dim=len(x.shape) - 1,
            cluster_axis=self.mesh_axis,
            mesh_device=x.device(),
            topology=self.ccl_manager.topology,
            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(),
            persistent_output_tensor=self.ccl_manager.get_ag_ping_pong_buffer(
                stats.shape, len(stats.shape) - 1, self.mesh_axis
            ),
            num_links=self.ccl_manager.num_links,
        )

        x = ttnn.layer_norm_post_all_gather(
            x,
            stats_gathered,
            weight=self.weight,
            bias=self.bias,
            epsilon=self.norm_eps,
            compute_kernel_config=self.compute_kernel_config,
        )
        return x


"""
Groupnorm that supports data parallel computation.
The number of channels and groups will be updated to match the distribution of the data across the mesh.
Set mesh_axis to None to disable data parallelism.
"""


# TODO: Add helper to assert torch reference
class GroupNorm:
    default_num_out_blocks = {
        # (Batch, Height, Width, Channels): num_out_blocks
    }  # used to overrride the num_out_blocks computed based on the input shape.

    def __init__(
        self,
        num_channels=None,
        num_groups=None,
        eps=None,
        mesh_device=None,
        mesh_axis=None,
        core_grid=None,
        num_out_blocks=None,
        torch_ref=None,
    ):
        self.eps = eps or torch_ref.eps
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.num_devices = tuple(mesh_device.shape)[mesh_axis] if mesh_axis is not None else 1
        self.num_channels = (num_channels or torch_ref.num_channels) // self.num_devices
        self.num_groups = (num_groups or torch_ref.num_groups) // self.num_devices
        self.num_out_blocks = num_out_blocks
        self.weight = None
        self.bias = None
        self.mask = None
        self.core_grid = core_grid or self.mesh_device.core_grid
        self.num_block_devisor = 256 * 256 * (self.core_grid.x * self.core_grid.y)

        # Assert group norm parameters
        assert (
            self.num_channels % 32 == 0 == self.num_channels % self.num_groups
        ), f"num_channels must be divisible by 32 and num_groups"

        if torch_ref is not None:
            self.load_state_dict(torch_ref.state_dict())

    @classmethod
    def from_torch(cls, torch_ref, num_output_blocks=None, mesh_device=None, mesh_axis=None, core_grid=None):
        layer = cls(
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            core_grid=core_grid,
            num_out_blocks=num_output_blocks,
            torch_ref=torch_ref,
        )
        return layer

    def group_norm_weight_bias_rm_sharded(self, tensor):
        if self.num_devices > 1:
            torch_sharded_lst = [
                ttnn.create_group_norm_weight_bias_rm(t, self.num_channels, self.core_grid.y)
                for t in tensor.chunk(self.num_devices)
            ]
            tensor_to_shard = torch.cat(torch_sharded_lst, dim=0)
            shard_dim = 0
        else:
            tensor_to_shard = ttnn.create_group_norm_weight_bias_rm(tensor, self.num_channels, self.core_grid.y)
            shard_dim = None

        return tensor_to_shard, shard_dim

    def load_state_dict(self, state_dict):
        [self.weight, self.bias], self.mask = ttnn.group_norm_params_from_torch(
            torch_params=[state_dict["weight"], state_dict["bias"]],
            channels_per_device=self.num_channels,
            groups_per_device=self.num_groups,
            device=self.mesh_device,
            mesh_axis=self.mesh_axis,
            return_mask=True,
        )

    # @classmethod
    def get_num_out_blocks(self, x_shape):
        logger.info(
            f"x_shape: {x_shape}, product: {x_shape[1] * x_shape[2] * x_shape[3]}, num_block_devisor: {self.num_block_devisor}, div: {((x_shape[1] * x_shape[2] * x_shape[3]) // self.num_block_devisor)}"
        )
        req = ((x_shape[1] * x_shape[2] * x_shape[3]) // self.num_block_devisor) or 1
        num_blocks = 2 ** (math.ceil(math.log2(req)))
        return self.default_num_out_blocks.setdefault(x_shape, num_blocks)

    def __call__(self, x):
        # num_out_blocks = -1 #min(256, self.num_out_blocks or self.get_num_out_blocks(tuple(x.shape)))
        # logger.info(f"shape: {x.shape} , num_out_blocks: {num_out_blocks}")

        batch_size, height, width, channels = x.shape
        x = x.reshape([batch_size, 1, width * height, channels])
        x = ttnn.group_norm(
            x,
            weight=self.weight,
            bias=self.bias,
            input_mask=self.mask,
            num_groups=self.num_groups,
            epsilon=self.eps,
            core_grid=self.core_grid,
            inplace=False,
            num_out_blocks=-1,
            output_layout=ttnn.TILE_LAYOUT,
        )
        x = x.reshape([batch_size, height, width, channels])

        return x

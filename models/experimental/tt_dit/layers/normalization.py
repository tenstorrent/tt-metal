# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from ..utils.tensor import bf16_tensor


class RMSNorm:
    def __init__(self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True, mesh_device=None):
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = bias
        self.weight = None
        self.bias = None

    def to_cached_state_dict(self, path_prefix, path_suffix=".tensorbin"):
        cache_dict = {}

        # Cache weight
        if self.weight is not None:
            weight_path = path_prefix + "weight" + path_suffix
            ttnn.dump_tensor(weight_path, self.weight)
            cache_dict["weight"] = weight_path

        # Cache bias if it exists
        if self.bias is not None:
            bias_path = path_prefix + "bias" + path_suffix
            ttnn.dump_tensor(bias_path, self.bias)
            cache_dict["bias"] = bias_path

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        if "weight" in cache_dict:
            self.weight = ttnn.load_tensor(cache_dict["weight"], device=self.mesh_device)
        if "bias" in cache_dict:
            self.bias = ttnn.load_tensor(cache_dict["bias"], device=self.mesh_device)

    def load_state_dict(self, state_dict):
        if self.norm_elementwise_affine:
            self.weight = bf16_tensor(state_dict["weight"].unsqueeze(0), device=self.mesh_device)
            if self.use_bias:
                self.bias = bf16_tensor(state_dict["bias"].unsqueeze(0), device=self.mesh_device)

    def __call__(self, x, compute_kernel_config=None):
        return ttnn.rms_norm(
            x, weight=self.weight, bias=self.bias, epsilon=self.norm_eps, compute_kernel_config=compute_kernel_config
        )


class LayerNorm:
    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=True,
        mesh_device=None,
        use_row_major_workaround=False,  # Issue #20789
    ):
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = bias
        self.use_row_major_workaround = use_row_major_workaround
        self.weight = None
        self.bias = None
        # When using the row-major workaround, ensure that dummy weight/bias are created
        if use_row_major_workaround:
            if use_row_major_workaround:
                self.weight = bf16_tensor(
                    torch.ones(1, embedding_dim).reshape(-1, 32), device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
                )
            else:
                self.weight = bf16_tensor(torch.ones(1, embedding_dim), device=self.mesh_device)
            if bias:
                if use_row_major_workaround:
                    self.bias = bf16_tensor(
                        torch.zeros(1, embedding_dim).reshape(-1, 32),
                        device=self.mesh_device,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    )
                else:
                    self.bias = bf16_tensor(torch.zeros(1, embedding_dim), device=self.mesh_device)

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    def to_cached_state_dict(self, path_prefix, path_suffix=".tensorbin"):
        cache_dict = {}

        if self.weight is not None:
            # Cache weight
            weight_path = path_prefix + "weight" + path_suffix
            ttnn.dump_tensor(weight_path, self.weight)
            cache_dict["weight"] = weight_path

        if self.bias is not None:
            # Cache bias
            bias_path = path_prefix + "bias" + path_suffix
            ttnn.dump_tensor(bias_path, self.bias)
            cache_dict["bias"] = bias_path

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        if "weight" in cache_dict:
            self.weight = ttnn.load_tensor(cache_dict["weight"], device=self.mesh_device)
        if "bias" in cache_dict:
            self.bias = ttnn.load_tensor(cache_dict["bias"], device=self.mesh_device)

    def load_state_dict(self, state_dict):
        if self.norm_elementwise_affine:
            if self.use_row_major_workaround:
                self.weight = bf16_tensor(
                    state_dict["weight"].reshape(-1, 32), device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
                )
            else:
                self.weight = bf16_tensor(state_dict["weight"].unsqueeze(0), device=self.mesh_device)
            if self.use_bias:
                if self.use_row_major_workaround:
                    self.bias = bf16_tensor(
                        state_dict["bias"].reshape(-1, 32), device=self.mesh_device, layout=ttnn.ROW_MAJOR_LAYOUT
                    )
                else:
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
        if not (norm_elementwise_affine and bias):
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

    def to_cached_state_dict(self, path_prefix, path_suffix=".tensorbin"):
        cache_dict = {}

        # Cache weight
        if self.weight is not None:
            weight_path = path_prefix + "weight" + path_suffix
            ttnn.dump_tensor(weight_path, self.weight)
            cache_dict["weight"] = weight_path

        # Cache bias
        if self.bias is not None:
            bias_path = path_prefix + "bias" + path_suffix
            ttnn.dump_tensor(bias_path, self.bias)
            cache_dict["bias"] = bias_path

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        if "weight" in cache_dict:
            self.weight = ttnn.load_tensor(cache_dict["weight"], device=self.mesh_device)
        if "bias" in cache_dict:
            self.bias = ttnn.load_tensor(cache_dict["bias"], device=self.mesh_device)

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
            multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.mesh_axis),
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
        num_out_blocks=-1,
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
        self.core_grid = core_grid or ttnn.CoreGrid(x=8, y=8)  # self.mesh_device.core_grid # Issue on 6U 8x9 grid

        # Assert group norm parameters
        assert (
            self.num_channels % 32 == 0 == self.num_channels % self.num_groups
        ), f"num_channels must be divisible by 32 and num_groups"

        if torch_ref is not None:
            self.load_state_dict(torch_ref.state_dict())

    @classmethod
    def from_torch(cls, torch_ref, num_output_blocks=-1, mesh_device=None, mesh_axis=None, core_grid=None):
        layer = cls(
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            core_grid=core_grid,
            num_out_blocks=num_output_blocks,
            torch_ref=torch_ref,
        )
        return layer

    def load_state_dict(self, state_dict):
        [self.weight, self.bias], self.mask = ttnn.dram_group_norm_params_from_torch(
            torch_params=[state_dict["weight"], state_dict["bias"]],
            channels_per_device=self.num_channels,
            groups_per_device=self.num_groups,
            device=self.mesh_device,
            mesh_axis=self.mesh_axis,
            return_mask=True,
        )

    def __call__(self, x):
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
            num_out_blocks=self.num_out_blocks,
            output_layout=ttnn.TILE_LAYOUT,
        )
        x = x.reshape([batch_size, height, width, channels])

        return x

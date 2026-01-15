# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import math

import torch
import ttnn

from .module import Module, Parameter


class RMSNorm(Module):
    def __init__(self, embedding_dim, norm_eps=1e-5, norm_elementwise_affine=True, bias=True, mesh_device=None):
        super().__init__()

        # https://github.com/tenstorrent/tt-metal/issues/31216
        assert embedding_dim % 32 == 0, "embedding_dim must be divisible by tile size"

        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = norm_elementwise_affine and bias

        if norm_elementwise_affine:
            self.weight = Parameter(total_shape=[1, embedding_dim], device=mesh_device)
            self.bias = Parameter(total_shape=[1, embedding_dim], device=mesh_device) if bias else None
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
        return ttnn.rms_norm(
            x,
            weight=self.weight.data if self.weight is not None else None,
            bias=self.bias.data if self.bias is not None else None,
            epsilon=self.norm_eps,
            compute_kernel_config=compute_kernel_config,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].unsqueeze(0)

        if "bias" in state:
            state["bias"] = state["bias"].unsqueeze(0)


class LayerNorm(Module):
    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=True,
        mesh_device=None,
        use_row_major_workaround=False,  # Issue #20789
    ):
        super().__init__()

        assert embedding_dim % 32 == 0, "embedding_dim must be divisible by tile size"

        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_device = mesh_device
        self.use_bias = norm_elementwise_affine and bias
        self.use_row_major_workaround = use_row_major_workaround

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        shape = [embedding_dim // 32, 32] if use_row_major_workaround else [1, embedding_dim]
        layout = ttnn.ROW_MAJOR_LAYOUT if self.use_row_major_workaround else ttnn.TILE_LAYOUT

        self.weight = (
            Parameter(total_shape=shape, layout=layout, device=mesh_device)
            if norm_elementwise_affine or self.use_row_major_workaround
            else None
        )
        self.bias = Parameter(total_shape=shape, layout=layout, device=mesh_device) if self.use_bias else None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.pop("weight", None)
        bias = state.pop("bias", None)

        # When using the row-major workaround, ensure that dummy weight/bias are created
        if self.use_row_major_workaround:
            assert self.norm_elementwise_affine == (weight is not None)
            assert self.use_bias == (bias is not None)

            if weight is None:
                weight = torch.ones(self.embedding_dim)
            if self.use_bias and bias is None:
                bias = torch.zeros(self.embedding_dim)

        if weight is not None:
            state["weight"] = weight.reshape(-1, 32) if self.use_row_major_workaround else weight.unsqueeze(0)

        if bias is not None:
            state["bias"] = bias.reshape(-1, 32) if self.use_row_major_workaround else bias.unsqueeze(0)

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(
            x,
            weight=self.weight.data if self.weight is not None else None,
            bias=self.bias.data if self.bias is not None else None,
            epsilon=self.norm_eps,
            compute_kernel_config=self.compute_kernel_config,
        )


class DistributedRMSNorm(Module):
    """
    Implements RMSNorm on an activation sharded on the reduction dimension.
    """

    def __init__(
        self,
        embedding_dim,
        norm_eps=1e-5,
        norm_elementwise_affine=True,
        bias=False,
        mesh_axis=0,
        mesh_device=None,
        ccl_manager=None,
    ):
        super().__init__()

        assert not bias, "bias is not supported for DistributedRMSNorm"
        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.mesh_axis = mesh_axis
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.mesh_width = tuple(mesh_device.shape)[mesh_axis]
        self.TILE_SIZE = 32

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        n = self.TILE_SIZE * self.mesh_width

        # https://github.com/tenstorrent/tt-metal/issues/31216
        assert embedding_dim % n == 0, "embedding_dim must be divisible by tile size times mesh width"

        self.weight = (
            Parameter(
                total_shape=[1, embedding_dim],
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_axes=[None, mesh_axis],
            )
            if norm_elementwise_affine
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].reshape(1, self.embedding_dim)

    def forward(
        self,
        x: ttnn.Tensor,
        num_heads_per_device=1,
        compute_kernel_config=None,
        rope_cos=None,
        rope_sin=None,
        trans_mat=None,
    ) -> ttnn.Tensor:
        expected_dim = self.embedding_dim // self.mesh_width
        if x.shape[-1] != expected_dim:
            msg = (
                f"last dimension of input tensor with shape {tuple(x.shape)} should match "
                f"embedding_dim / mesh_width = {expected_dim}"
            )
            raise ValueError(msg)

        stats = ttnn.experimental.wan_fused_rmsnorm_pre_allgather(
            x, compute_kernel_config=compute_kernel_config or self.compute_kernel_config
        )

        if tuple(self.mesh_device.shape)[self.mesh_axis] > 1:
            stats = ttnn.experimental.all_gather_async(
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

        x = ttnn.experimental.wan_fused_rmsnorm_post_allgather(
            x,
            stats,
            epsilon=self.norm_eps,
            num_heads_per_device=num_heads_per_device,
            weight=self.weight.data if self.weight is not None else None,
            compute_kernel_config=compute_kernel_config or self.compute_kernel_config,
            transformation_mat=trans_mat,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
        )
        return x


class DistributedLayerNorm(Module):
    """
    Implements LayerNorm on an activation sharded on the reduction dimension.
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
        super().__init__()

        self.embedding_dim = embedding_dim
        self.norm_eps = norm_eps
        self.norm_elementwise_affine = norm_elementwise_affine
        self.use_bias = norm_elementwise_affine and bias
        self.mesh_axis = mesh_axis
        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.mesh_width = tuple(mesh_device.shape)[mesh_axis]
        self.TILE_SIZE = 32

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        n = self.TILE_SIZE * self.mesh_width
        shape = [embedding_dim // n, n]

        assert embedding_dim % n == 0, "embedding_dim must be divisible by tile size times mesh width"

        self.weight = (
            Parameter(total_shape=shape, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, mesh_axis], device=mesh_device)
            if self.norm_elementwise_affine
            else None
        )
        self.bias = (
            Parameter(total_shape=shape, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_axes=[None, mesh_axis], device=mesh_device)
            if self.use_bias
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.pop("weight", None)
        bias = state.pop("bias", None)
        assert (weight is not None) == self.norm_elementwise_affine
        assert (bias is not None) == self.use_bias

        if self.norm_elementwise_affine:
            state["weight"] = (
                weight.reshape(self.mesh_width, -1, self.TILE_SIZE)
                .permute(1, 0, 2)
                .reshape(-1, self.TILE_SIZE * self.mesh_width)
            )

        if self.use_bias:
            state["bias"] = (
                bias.reshape(self.mesh_width, -1, self.TILE_SIZE)
                .permute(1, 0, 2)
                .reshape(-1, self.TILE_SIZE * self.mesh_width)
            )

    def forward(
        self, x: ttnn.Tensor, dynamic_weight=None, dynamic_bias=None, compute_kernel_config=None
    ) -> ttnn.Tensor:
        assert (dynamic_weight is None) == (
            dynamic_bias is None
        ), "dynamic_weight and dynamic_bias must be either both provided or both None"
        if dynamic_weight is not None:
            assert (
                not self.norm_elementwise_affine
            ), "Module must not have weight and bias parameters when dynamic_weight and dynamic_bias are provided"

            weight = dynamic_weight
            bias = dynamic_bias
        else:
            weight = self.weight.data if self.weight is not None else None
            bias = self.bias.data if self.bias is not None else None

        stats = ttnn.experimental.dit_layernorm_pre_allgather(
            x,
            compute_kernel_config=compute_kernel_config or self.compute_kernel_config,
        )

        stats = self.ccl_manager.all_gather_persistent_buffer(
            stats,
            dim=len(x.shape) - 1,
            mesh_axis=self.mesh_axis,
        )

        x = ttnn.experimental.dit_layernorm_post_allgather(
            x,
            stats,
            weight=weight,
            bias=bias,
            epsilon=self.norm_eps,
            compute_kernel_config=compute_kernel_config or self.compute_kernel_config,
        )
        return x


"""
Groupnorm that supports data parallel computation.
The number of channels and groups will be updated to match the distribution of the data across the mesh.
Set mesh_axis to None to disable data parallelism.
"""


# TODO: Add helper to assert torch reference
class GroupNorm(Module):
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
        torch_ref=None,
    ):
        super().__init__()

        """
        Args:
            num_channels: Number of channels in the input tensor.
            num_groups: Number of groups.
            eps: Epsilon value for numerical stability.
            mesh_device: The device to use.
            mesh_axis: The mesh axis to use for sharding.
            core_grid: The core grid to use.
            num_out_blocks: The number of output blocks to use.
            torch_ref: The torch reference layer.
        """
        self.eps = eps or torch_ref.eps
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.num_devices = tuple(mesh_device.shape)[mesh_axis] if mesh_axis is not None else 1
        self.num_channels = (num_channels or torch_ref.num_channels) // self.num_devices
        self.num_groups = (num_groups or torch_ref.num_groups) // self.num_devices
        self.core_grid = core_grid or ttnn.CoreGrid(x=8, y=8)  # self.mesh_device.core_grid # Issue on 6U 8x9 grid
        self.num_virtual_cols = ttnn.operations.normalization.dram_group_norm_virtual_columns(
            self.mesh_device.core_grid, self.num_channels, self.num_groups
        )

        # Assert group norm parameters
        assert (
            self.num_channels % 32 == 0 == self.num_channels % self.num_groups
        ), f"num_channels must be divisible by 32 and num_groups"

        weight_shape = [
            self.num_devices,
            1,
            math.ceil(self.num_channels // self.num_virtual_cols / 32) * self.num_virtual_cols,
            32,
        ]
        block_wt = ttnn.operations.normalization.find_max_tile_span(
            self.num_channels, self.num_channels // self.num_groups, 32
        )
        mask_shape = [1, self.num_groups, 32, 32 * block_wt]

        self.weight = Parameter(
            total_shape=weight_shape,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[mesh_axis, None, None, None],
            device=self.mesh_device,
        )
        self.bias = Parameter(
            total_shape=weight_shape,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_axes=[mesh_axis, None, None, None],
            device=self.mesh_device,
        )
        self.mask = Parameter(total_shape=mask_shape, device=self.mesh_device)

        if torch_ref is not None:
            self.load_torch_state_dict(torch_ref.state_dict())

    @classmethod
    def from_torch(cls, torch_ref, mesh_device=None, mesh_axis=None, core_grid=None):
        layer = cls(
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            core_grid=core_grid,
            torch_ref=torch_ref,
        )
        return layer

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = self._prepare_param(state["weight"])
        if "bias" in state:
            state["bias"] = self._prepare_param(state["bias"])

        input_mask = ttnn.create_group_norm_input_mask(self.num_channels, self.num_groups, self.num_virtual_cols)
        state["mask"] = ttnn.to_torch(input_mask)

    def _prepare_param(self, param: torch.Tensor) -> torch.Tensor:
        expected_shape = (self.num_channels * self.num_devices,)
        assert param.shape == expected_shape, f"expected shape {expected_shape}, got {param.shape}"

        torch_sharded_lst = [
            ttnn.create_group_norm_weight_bias_rm(t, self.num_channels, self.num_virtual_cols)
            for t in param.chunk(self.num_devices)
        ]
        return torch.cat(torch_sharded_lst, dim=0)

    def forward(self, x: ttnn.Tensor, num_out_blocks=-1) -> ttnn.Tensor:
        batch_size, height, width, channels = x.shape
        x = x.reshape([batch_size, 1, width * height, channels])
        x = ttnn.group_norm(
            x,
            weight=self.weight.data,
            bias=self.bias.data,
            input_mask=self.mask.data,
            num_groups=self.num_groups,
            epsilon=self.eps,
            core_grid=self.core_grid,
            inplace=False,
            num_out_blocks=num_out_blocks,
            output_layout=ttnn.TILE_LAYOUT,
        )
        x = x.reshape([batch_size, height, width, channels])

        return x

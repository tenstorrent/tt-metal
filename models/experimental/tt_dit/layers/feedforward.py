# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

from .linear import ColParallelLinear, Linear, RowParallelLinear
from .module import Module


class FeedForward(Module):
    """
    Linear layer with replicated weights
    """

    def __init__(
        self,
        dim: int,
        dim_out=None,
        mult: int = 4,
        activation_fn: str = "gelu",
        inner_dim=None,
        bias: bool = True,
        mesh_device=None,
    ):
        super().__init__()

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.mesh_device = mesh_device
        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.activation_fn = activation_fn
        self.bias = bias

        self.ff1 = Linear(dim, inner_dim, bias=bias, mesh_device=mesh_device, activation_fn=activation_fn)
        self.ff2 = Linear(inner_dim, dim_out, bias=bias, mesh_device=mesh_device)

    def forward(self, x: ttnn.Tensor, core_grid=None, compute_kernel_config=None) -> ttnn.Tensor:
        ff1_out = self.ff1(x, core_grid=core_grid, compute_kernel_config=compute_kernel_config)
        return self.ff2(ff1_out, core_grid=core_grid, compute_kernel_config=compute_kernel_config)


class ParallelFeedForward(Module):
    """
    Linear layer implementing megatron-style parallelism.
    """

    def __init__(
        self,
        dim: int,
        dim_out=None,
        mult: int = 4,
        activation_fn: str = "gelu",
        inner_dim=None,
        bias: bool = True,
        mesh_device=None,
        mesh_axis=0,
        fsdp_mesh_axis=None,
        ccl_manager=None,
    ):
        super().__init__()

        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.mesh_device = mesh_device
        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.activation_fn = activation_fn
        self.bias = bias
        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis

        self.ff1 = ColParallelLinear(
            dim,
            inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            activation_fn=activation_fn,
            mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.ff2 = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

    def forward(self, x: ttnn.Tensor, core_grid=None, compute_kernel_config=None) -> ttnn.Tensor:
        """
        Expects x to be replicated.
        Return output fractured on columns.
        """
        ff1_out = self.ff1(x, core_grid=core_grid, compute_kernel_config=compute_kernel_config)
        return self.ff2(ff1_out, core_grid=core_grid, compute_kernel_config=compute_kernel_config)

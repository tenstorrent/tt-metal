# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from .linear import Linear, ColParallelLinear, RowParallelLinear
from ..utils.substate import substate


class FeedForward:
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
        init=False,
    ):
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.mesh_device = mesh_device
        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.activation_fn = activation_fn
        self.bias = bias

        self.ff1 = Linear(dim, inner_dim, bias=bias, mesh_device=mesh_device, activation_fn=activation_fn, init=init)
        self.ff2 = Linear(inner_dim, dim_out, bias=bias, mesh_device=mesh_device, init=init)

    def to_cached_state_dict(self, path_prefix):
        ff1_cache = self.ff1.to_cached_state_dict(path_prefix + "ff1.")
        ff2_cache = self.ff2.to_cached_state_dict(path_prefix + "ff2.")
        cache_dict = {}
        # Add ff1. prefix to all keys from ff1_cache
        for key, value in ff1_cache.items():
            cache_dict[f"ff1.{key}"] = value
        # Add ff2. prefix to all keys from ff2_cache
        for key, value in ff2_cache.items():
            cache_dict[f"ff2.{key}"] = value
        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.ff1.from_cached_state_dict(substate(cache_dict, "ff1"))
        self.ff2.from_cached_state_dict(substate(cache_dict, "ff2"))

    def load_state_dict(self, state_dict):
        self.ff1.load_state_dict(substate(state_dict, "ff1"))
        self.ff2.load_state_dict(substate(state_dict, "ff2"))

    def __call__(self, x, core_grid=None, compute_kernel_config=None):
        ff1_out = self.ff1(x, core_grid=core_grid, compute_kernel_config=compute_kernel_config)
        return self.ff2(ff1_out, core_grid=core_grid, compute_kernel_config=compute_kernel_config)


class ParallelFeedForward:
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
        init=False,
    ):
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
            init=init,
        )
        self.ff2 = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
            init=init,
        )

    def to_cached_state_dict(self, path_prefix):
        ff1_cache = self.ff1.to_cached_state_dict(path_prefix + "ff1.")
        ff2_cache = self.ff2.to_cached_state_dict(path_prefix + "ff2.")
        cache_dict = {}
        # Add ff1. prefix to all keys from ff1_cache
        for key, value in ff1_cache.items():
            cache_dict[f"ff1.{key}"] = value
        # Add ff2. prefix to all keys from ff2_cache
        for key, value in ff2_cache.items():
            cache_dict[f"ff2.{key}"] = value
        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.ff1.from_cached_state_dict(substate(cache_dict, "ff1"))
        self.ff2.from_cached_state_dict(substate(cache_dict, "ff2"))

    def load_state_dict(self, state_dict):
        self.ff1.load_state_dict(substate(state_dict, "ff1"))
        self.ff2.load_state_dict(substate(state_dict, "ff2"))

    def __call__(self, x, core_grid=None, compute_kernel_config=None):
        """
        Expects x to be replicated.
        Return output fractured on columns.
        """
        ff1_out = self.ff1(x, core_grid=core_grid, compute_kernel_config=compute_kernel_config)
        return self.ff2(ff1_out, core_grid=core_grid, compute_kernel_config=compute_kernel_config)

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from .linear import ColParallelLinear, RowParallelLinear
from ..utils.substate import substate
import ttnn


class FeedForward:
    """
    Linear layer with replicated weights
    """

    def __init__(
        self,
        dim: int,
        dim_out=None,
        mult: int = 4,
        activation_fn: str = "geglu",
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

        self.ff1 = Linear(dim, inner_dim, bias=bias, mesh_device=mesh_device, init=init)
        self.ff2 = Linear(inner_dim, dim_out, bias=bias, mesh_device=mesh_device, init=init)

    def load_state_dict(self, state_dict, transform=None):
        assert transform is None, "Haven't figured out how to pass two transformations yet"

        has_fc_keys = any(k.startswith("fc1.") or k.startswith("fc2.") for k in state_dict.keys())
        has_ff_keys = any(k.startswith("ff1.") or k.startswith("ff2.") for k in state_dict.keys())

        if has_fc_keys:
            # CLIP format: fc1, fc2
            self.ff1.load_state_dict(substate(state_dict, "fc1"))
            self.ff2.load_state_dict(substate(state_dict, "fc2"))
        else:
            # standard format: ff1, ff2
            self.ff1.load_state_dict(substate(state_dict, "ff1"))
            self.ff2.load_state_dict(substate(state_dict, "ff2"))

    def __call__(self, x):
        return self.ff2(self.ff1(x))


class ParallelFeedForward:
    """
    Linear layer implementing megatron-style parallelism.
    """

    def __init__(
        self,
        dim: int,
        dim_out=None,
        mult: int = 4,
        activation_fn: str = "geglu",
        inner_dim=None,
        bias: bool = True,
        mesh_device=None,
        mesh_axis=0,
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

        self.ff1 = ColParallelLinear(dim, inner_dim, bias=bias, mesh_device=mesh_device, init=init)
        self.ff2 = RowParallelLinear(
            inner_dim,
            dim_out,
            bias=bias,
            mesh_device=mesh_device,
            mesh_axis=mesh_axis,
            ccl_manager=ccl_manager,
            init=init,
        )

    def load_state_dict(self, state_dict, transform=None):
        assert transform is None, "Haven't figured out how to pass two transformations yet"

        # check which key format is present
        has_fc_keys = any(k.startswith("fc1.") or k.startswith("fc2.") for k in state_dict.keys())
        has_ff_keys = any(k.startswith("ff1.") or k.startswith("ff2.") for k in state_dict.keys())

        if has_fc_keys:
            # CLIP format: fc1, fc2
            self.ff1.load_state_dict(substate(state_dict, "fc1"))
            self.ff2.load_state_dict(substate(state_dict, "fc2"))
        else:
            # standard format: ff1, ff2
            self.ff1.load_state_dict(substate(state_dict, "ff1"))
            self.ff2.load_state_dict(substate(state_dict, "ff2"))

    def __call__(self, x, parallel_manager=None):
        """
        Expects x to be replicated.
        Return output fractured on columns.
        """
        logger.info(f"Starting ParallelFeedForward, input shape: {x.shape}")

        logger.info("Computing FF1 (first linear layer)...")
        ff1_output = self.ff1(x)
        logger.info(f"FF1 done, shape: {ff1_output.shape}")

        logger.info(f"Activation function: {self.activation_fn}")
        if self.activation_fn == "gelu":
            logger.info("Activation function: gelu")
            # TODO: Should we use approximate gelu?
            ff1_output = ttnn.gelu(ff1_output)
        elif self.activation_fn == "quick_gelu":
            logger.info("Activation function: quick_gelu")
            ff1_output = ttnn.quick_gelu(ff1_output)
        else:
            raise ValueError(f"Unsupported activation function: {self.activation_fn}")

        logger.info("Computing FF2 (second linear layer)...")
        result = self.ff2(ff1_output)
        logger.info(f"ParallelFeedForward completed, output shape: {result.shape}")

        return result

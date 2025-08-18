# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from .linear import Linear, ColParallelLinear, RowParallelLinear
from ..utils.substate import substate
from ..utils.padding import pad_weight_tensor, pad_bias_tensor
from ..utils.tensor import bf16_tensor
import ttnn
import torch
from loguru import logger


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

        self.ff1 = Linear(dim, inner_dim, bias=bias, mesh_device=mesh_device, activation=activation_fn, init=init)
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

            if self.padding_config and self.padding_config.is_padding_needed():
                logger.info(
                    f"Padding ff1 and ff2 weights from {self.padding_config.original_dim} to {self.padding_config.target_dim}"
                )

                # Convert sharded ttnn tensors to torch tensors for padding
                # Weights are column-sharded, so we need to concatenate along the last dimension
                mesh_composer = ttnn.ConcatMesh2dToTensor(
                    self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=(0, -1)
                )

                ff1_torch = ttnn.to_torch(self.ff1.weight, mesh_composer=mesh_composer)
                ff2_torch = ttnn.to_torch(self.ff2.weight, mesh_composer=mesh_composer)

                # Apply padding
                ff1_padded = pad_weight_tensor(ff1_torch, self.padding_config, pad_input_dim=True, pad_output_dim=True)
                ff2_padded = pad_weight_tensor(ff2_torch, self.padding_config, pad_input_dim=True, pad_output_dim=False)

                # Convert back to sharded ttnn tensors
                # For ColParallelLinear (ff1): shard on output dimension (last)
                ff1_combined = torch.cat(ff1_padded_tensors, dim=-1)  # Concat along output dim
                self.ff1.weight = bf16_tensor(
                    ff1_combined, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1
                )

                # For RowParallelLinear (ff2): shard on input dimension (first)
                ff2_combined = torch.cat(ff2_padded_tensors, dim=0)  # Concat along input dim
                self.ff2.weight = bf16_tensor(
                    ff2_combined, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-2
                )

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
        activation_fn: str = "geglu",
        inner_dim=None,
        bias: bool = True,
        mesh_device=None,
        mesh_axis=0,
        ccl_manager=None,
        init=False,
        padding_config=None,
    ):
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.dim = dim
        self.dim_out = dim_out
        self.inner_dim = inner_dim
        self.activation_fn = activation_fn
        self.bias = bias
        self.padding_config = padding_config
        self.ff1 = ColParallelLinear(
            dim,
            inner_dim,
            bias=bias,
            mesh_device=mesh_device,
            activation_fn=activation_fn,
            mesh_axis=mesh_axis,
            init=init,
        )
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

        has_fc_keys = any(k.startswith("fc1.") or k.startswith("fc2.") for k in state_dict.keys())
        has_ff_keys = any(k.startswith("ff1.") or k.startswith("ff2.") for k in state_dict.keys())

        # fc1.weight: torch.Size([3072, 768])
        # fc2.weight: torch.Size([768, 3072])

        if has_fc_keys:
            # CLIP format: fc1, fc2
            self.ff1.load_state_dict(substate(state_dict, "fc1"))  #
            self.ff2.load_state_dict(substate(state_dict, "fc2"))

            if self.padding_config and self.padding_config.is_padding_needed():
                logger.info(
                    f"Padding ff1 and ff2 weights from {self.padding_config.original_dim} to {self.padding_config.target_dim}"
                )
                # breakpoint()
                # convert sharded ttnn tensors to torch tensors for padding
                # different concat strategies for ColParallel vs RowParallel

                # ff1 (ColParallel): sharded on output dim (last), concat on last dim
                ff1_mesh_composer = ttnn.ConcatMesh2dToTensor(
                    self.mesh_device,
                    mesh_shape=tuple(self.mesh_device.shape),
                    dims=(0, -1),  # concat along last dimension
                )

                # ff2 (RowParallel): sharded on input dim (first), concat on first dim
                ff2_mesh_composer = ttnn.ConcatMesh2dToTensor(
                    self.mesh_device,
                    mesh_shape=tuple(self.mesh_device.shape),
                    dims=(-1, 0),  # concat along first dimension
                )

                # skip mesh composition - work with per-device tensors directly
                # each device has [768, 768] weights, pad to [1024, 1024]
                padding_amount = self.padding_config.dim_padding  # 256

                # get per-device tensors and pad each one individually
                ff1_device_tensors = ttnn.get_device_tensors(self.ff1.weight)
                ff2_device_tensors = ttnn.get_device_tensors(self.ff2.weight)

                ff1_padded_tensors = []
                ff2_padded_tensors = []

                for device_tensor in ff1_device_tensors:
                    tensor_torch = ttnn.to_torch(device_tensor)  # [768, 768]
                    # pad to [1024, 1024]: add 256 to both dimensions
                    padded = torch.cat(
                        [
                            tensor_torch,
                            torch.zeros(
                                padding_amount,
                                tensor_torch.shape[1],
                                dtype=tensor_torch.dtype,
                                device=tensor_torch.device,
                            ),
                        ],
                        dim=0,
                    )
                    padded = torch.cat(
                        [
                            padded,
                            torch.zeros(
                                padded.shape[0], padding_amount, dtype=tensor_torch.dtype, device=tensor_torch.device
                            ),
                        ],
                        dim=1,
                    )
                    ff1_padded_tensors.append(padded)

                for device_tensor in ff2_device_tensors:
                    tensor_torch = ttnn.to_torch(device_tensor)  # [768, 768]
                    # pd to [1024, 1024]: add 256 to both dimensions
                    padded = torch.cat(
                        [
                            tensor_torch,
                            torch.zeros(
                                padding_amount,
                                tensor_torch.shape[1],
                                dtype=tensor_torch.dtype,
                                device=tensor_torch.device,
                            ),
                        ],
                        dim=0,
                    )
                    padded = torch.cat(
                        [
                            padded,
                            torch.zeros(
                                padded.shape[0], padding_amount, dtype=tensor_torch.dtype, device=tensor_torch.device
                            ),
                        ],
                        dim=1,
                    )
                    ff2_padded_tensors.append(padded)

                # convert back to sharded ttnn tensors
                # for ColParallelLinear (ff1): shard on output dimension (last)
                ff1_combined = torch.cat(ff1_padded_tensors, dim=-1)  # Concat along output dim
                self.ff1.weight = bf16_tensor(
                    ff1_combined, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1
                )

                # For RowParallelLinear (ff2): shard on input dimension (first)
                ff2_combined = torch.cat(ff2_padded_tensors, dim=0)  # Concat along input dim
                self.ff2.weight = bf16_tensor(
                    ff2_combined, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-2
                )

                # pad bias tensors using pad_bias_tensor utility
                if self.ff1.bias is not None:
                    # ff1 bias: [768] per device -> [1024] per device using pad_bias_tensor
                    ff1_bias_tensors = ttnn.get_device_tensors(self.ff1.bias)
                    ff1_bias_padded = []
                    for bias_tensor in ff1_bias_tensors:
                        bias_torch = ttnn.to_torch(bias_tensor)  # [1, 768]
                        bias_torch = bias_torch.squeeze(0)  # [768]
                        bias_padded = pad_bias_tensor(bias_torch, self.padding_config)  # [1024]
                        ff1_bias_padded.append(bias_padded)

                    # concatenate padded biases to recreate full bias: 4×[1024] → [4096]
                    # this is needed because bf16_tensor() expects a full tensor to shard
                    ff1_bias_combined = torch.cat(ff1_bias_padded, dim=-1)  # [4096]
                    self.ff1.bias = bf16_tensor(
                        ff1_bias_combined, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1
                    )

                if self.ff2.bias is not None:
                    # ff2 bias: [768] -> [1024] using pad_bias_tensor
                    # ff2 is RowParallelLinear, so bias is replicated (not sharded)
                    ff2_bias_torch = ttnn.to_torch(
                        ttnn.get_device_tensors(self.ff2.bias)[0]
                    )  # get first device tensor [768] or [1, 768]
                    if len(ff2_bias_torch.shape) > 1:
                        ff2_bias_torch = ff2_bias_torch.squeeze()  # Ensure 1D: [768]
                    ff2_bias_padded = pad_bias_tensor(ff2_bias_torch, self.padding_config)  # [1024]
                    self.ff2.bias = bf16_tensor(ff2_bias_padded, device=self.mesh_device, layout=ttnn.TILE_LAYOUT)
        else:
            # standard format: ff1, ff2
            self.ff1.load_state_dict(substate(state_dict, "ff1"))
            self.ff2.load_state_dict(substate(state_dict, "ff2"))

    def __call__(self, x, core_grid=None, compute_kernel_config=None):
        """
        Expects x to be replicated.
        Return output fractured on columns.
        """
        ff1_out = self.ff1(x, core_grid=core_grid, compute_kernel_config=compute_kernel_config)
        return self.ff2(ff1_out, core_grid=core_grid, compute_kernel_config=compute_kernel_config)

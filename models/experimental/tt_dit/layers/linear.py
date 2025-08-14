# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ..utils.tensor import bf16_tensor


class Linear:
    """
    Linear layer with replicated weights
    """

    def __init__(self, in_features, out_features, bias=True, activation=None, mesh_device=None, init=False):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.mesh_device = mesh_device
        if init:
            self.weight = bf16_tensor(torch.randn(in_features, out_features), device=self.mesh_device)
            if bias:
                self.bias = bf16_tensor(torch.randn(1, out_features), device=self.mesh_device)
            else:
                self.bias = None

        """
        NOTE: This is the special config which attains good correctness
        HiFi2 + packer_l1_acc + bf16 acc in a fused linear (matmul + bias) with unfused non-approx activation
        """
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def load_state_dict(self, state_dict, transform=None):
        """
        Loads the state dict into the layer.
        transform is a lambda that takes two tensors and returns two transformed tensors.
        """
        weight = state_dict["weight"].transpose(0, 1)
        bias = state_dict.get("bias", None)

        if transform is not None:
            weight, bias = transform(weight, bias)
        self.weight = bf16_tensor(weight, device=self.mesh_device)
        if bias is not None:
            bias = bias.reshape(1, -1)
            self.bias = bf16_tensor(bias, device=self.mesh_device)
        else:
            self.bias = None

    def __call__(self, x, core_grid=None, compute_kernel_config=None):
        output = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            core_grid=core_grid,
            compute_kernel_config=compute_kernel_config or self.compute_config,
        )
        if self.activation is not None:
            assert self.activation == "gelu"
            output = ttnn.gelu(output, fast_and_approximate_mode=False)
        return output


class ColParallelLinear:
    """
    Linear layer with column parallel weights
    """

    def __init__(
        self, in_features, out_features, bias=True, activation=None, mesh_device=None, mesh_axis=0, init=False
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        if init:
            self.weight = bf16_tensor(
                torch.randn(in_features, out_features), device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1
            )
            if bias:
                self.bias = bf16_tensor(
                    torch.randn(1, out_features), device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1
                )
            else:
                self.bias = None

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def load_state_dict(self, state_dict, transform=None):
        """
        Loads the state dict into the layer.
        transform is a lambda that takes two tensors and returns two transformed tensors.
        """
        weight = state_dict["weight"].transpose(0, 1)
        bias = state_dict.get("bias", None)

        if transform is not None:
            weight, bias = transform(weight, bias)
        self.weight = bf16_tensor(weight, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1)
        if bias is not None:
            bias = bias.reshape(1, -1)
            self.bias = bf16_tensor(bias, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1)
        else:
            self.bias = None

    def __call__(self, x, core_grid=None, compute_kernel_config=None):
        """
        Expects x to be replicated.
        Return output fractured on columns.
        """
        output = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            core_grid=core_grid,
            compute_kernel_config=compute_kernel_config or self.compute_config,
        )
        if self.activation is not None:
            assert self.activation == "gelu"
            output = ttnn.gelu(output, fast_and_approximate_mode=False)
        return output


class RowParallelLinear:
    """
    Linear layer with row parallel weights
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation=None,
        mesh_device=None,
        mesh_axis=0,
        ccl_manager=None,
        init=False,
    ):
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.ccl_manager = ccl_manager
        if init:
            self.weight = bf16_tensor(
                torch.randn(in_features, out_features), device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-2
            )
            if bias:
                # row-parallel bias must not be replicated across mesh_devices
                rand_bias = torch.randn(1, out_features)
                if tuple(mesh_device.shape)[mesh_axis] > 1:
                    zero_bias = torch.zeros(1, out_features * (tuple(mesh_device.shape)[mesh_axis] - 1))
                    rand_bias = torch.cat([rand_bias, zero_bias], dim=-1)
                self.bias = bf16_tensor(rand_bias, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1)
            else:
                self.bias = None

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def load_state_dict(self, state_dict, transform=None):
        """
        Loads the state dict into the layer.
        transform is a lambda that takes two tensors and returns two transformed tensors.
        """
        weight = state_dict["weight"].transpose(0, 1)
        bias = state_dict.get("bias", None)

        if transform is not None:
            weight, bias = transform(weight, bias)
        self.weight = bf16_tensor(weight, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-2)
        if bias is not None:
            bias = bias.reshape(1, -1)
            if tuple(self.mesh_device.shape)[self.mesh_axis] > 1:
                zero_bias = torch.zeros(1, bias.shape[1] * (tuple(self.mesh_device.shape)[self.mesh_axis] - 1))
                bias = torch.cat([bias, zero_bias], dim=-1)
            self.bias = bf16_tensor(bias, device=self.mesh_device, mesh_axis=self.mesh_axis, shard_dim=-1)
        else:
            self.bias = None

    def __call__(self, x, core_grid=None, compute_kernel_config=None):
        """
        Expects x to be column fractured.
        Return output fractured on columns.
        """
        output = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            core_grid=core_grid,
            compute_kernel_config=compute_kernel_config or self.compute_config,
        )

        if tuple(self.mesh_device.shape)[self.mesh_axis] > 1:
            output = ttnn.experimental.reduce_scatter_minimal_async(
                output,
                persistent_output_buffers=self.ccl_manager.get_rs_ping_pong_buffer(
                    output.shape, dim=3, mesh_axis=self.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(),
                num_links=self.ccl_manager.num_links,
                memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
                topology=self.ccl_manager.topology,
                cluster_axis=self.mesh_axis,
                **self.ccl_manager.get_rs_hyperparams(output.shape),
            )

        if self.activation is not None:
            assert self.activation == "gelu"
            output = ttnn.gelu(output, fast_and_approximate_mode=False)

        return output

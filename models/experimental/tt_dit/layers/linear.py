# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch
import ttnn

from .module import Module, Parameter
from ..utils.matmul import get_matmul_config


class Linear(Module):
    """
    Linear layer with replicated weights
    """

    def __init__(self, in_features, out_features, bias=True, activation_fn=None, mesh_device=None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if activation_fn == "swiglu":
            # Double out features for fused swiglu activation
            self.out_features = self.out_features * 2
        self.activation_fn = activation_fn
        self.fused_activation_fn = None
        if self.activation_fn == "gelu":
            self.activation_fn = None
            self.fused_activation_fn = (ttnn.UnaryOpType.GELU, False)
        self.mesh_device = mesh_device

        """
        NOTE: This is the special config which attains good correctness
        HiFi2 + packer_l1_acc + bf16 acc in a fused linear (matmul + bias) with unfused non-approx activation
        """
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.weight = Parameter(total_shape=[self.in_features, self.out_features], device=mesh_device)
        self.bias = Parameter(total_shape=[1, self.out_features], device=mesh_device) if bias else None

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].transpose(0, 1)
        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
        M, K, N = x.padded_shape[-2], x.padded_shape[-1], self.weight.data.padded_shape[-1]
        core_grid = self.mesh_device.compute_with_storage_grid_size()
        matmul_config = get_matmul_config(M, K, N, core_grid)
        output = ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=self.weight.data,
            bias_tensor=self.bias.data if self.bias is not None else None,
            config=matmul_config,
            fused_activation=self.fused_activation_fn,
            compute_kernel_config=compute_kernel_config or self.compute_config,
        )
        if self.activation_fn == "decomposed_gelu":
            output = gelu_decomposed(output)
        elif self.activation_fn == "quick_gelu":
            output = output * ttnn.sigmoid_accurate(1.702 * output)  # quick approx gelu
        elif self.activation_fn == "swiglu":
            output, gate = ttnn.chunk(output, 2, -1)
            output = output * ttnn.silu(gate)
        else:
            assert self.activation_fn is None, f"Unsupported activation: {self.activation_fn}"
        return output


def gelu_decomposed(x: ttnn.Tensor) -> ttnn.Tensor:
    # GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))
    # ttnn.gelu is the same, but avoiding for potential issues (see ttnn.layernorm)
    sqrt_2 = math.sqrt(2.0)
    x_div_sqrt2 = ttnn.multiply(x, 1.0 / sqrt_2)
    erf_x = ttnn.erf(x_div_sqrt2)
    one_plus_erf = ttnn.add(erf_x, 1.0)
    x_times_bracket = ttnn.multiply(x, one_plus_erf)
    return ttnn.multiply(x_times_bracket, 0.5)


class ColParallelLinear(Module):
    """
    Linear layer with column parallel weights
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation_fn=None,
        mesh_device=None,
        mesh_axis=0,
        fsdp_mesh_axis=None,
        ccl_manager=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        if activation_fn == "swiglu":
            # Double out features for fused swiglu activation
            self.out_features = self.out_features * 2
        self.fused_activation_fn = None
        if self.activation_fn == "gelu":
            self.activation_fn = None
            self.fused_activation_fn = (ttnn.UnaryOpType.GELU, False)
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis
        self.ccl_manager = ccl_manager

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis
            assert self.ccl_manager is not None

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.weight = Parameter(
            total_shape=[self.in_features, self.out_features], mesh_axes=[fsdp_mesh_axis, mesh_axis], device=mesh_device
        )
        self.bias = (
            Parameter(total_shape=[1, self.out_features], mesh_axes=[None, mesh_axis], device=mesh_device)
            if bias
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.pop("weight", None)
        bias = state.pop("bias", None)

        def permute_for_swiglu(tensor):
            assert self.activation_fn == "swiglu"
            ndev = self.mesh_device.shape[self.mesh_axis]
            tensor = tensor.reshape(-1, 2, ndev, tensor.shape[-1] // 2 // ndev)
            tensor = tensor.permute(0, 2, 1, 3)
            tensor = tensor.reshape(-1, self.out_features)
            assert tensor.shape[0] in [1, self.in_features]
            return tensor

        if weight is not None:
            weight = weight.transpose(0, 1)
            if self.activation_fn == "swiglu":
                weight = permute_for_swiglu(weight)
            state["weight"] = weight
        if bias is not None:
            bias = bias.reshape(1, -1)
            if self.activation_fn == "swiglu":
                bias = permute_for_swiglu(bias)
            state["bias"] = bias

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
        """
        Expects x to be replicated.
        Return output fractured on columns.
        """
        if self.fsdp_mesh_axis is not None and self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=2, mesh_axis=self.fsdp_mesh_axis
            )

            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = self.weight.data

        M, K, N = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
        core_grid = self.mesh_device.compute_with_storage_grid_size()
        matmul_config = get_matmul_config(M, K, N, core_grid)
        output = ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=self.bias.data if self.bias is not None else None,
            config=matmul_config,
            fused_activation=self.fused_activation_fn,
            compute_kernel_config=compute_kernel_config or self.compute_config,
        )
        if self.activation_fn == "decomposed_gelu":
            output = gelu_decomposed(output)
        elif self.activation_fn == "quick_gelu":
            output = output * ttnn.sigmoid_accurate(1.702 * output)  # quick approx gelu
        elif self.activation_fn == "swiglu":
            output, gate = ttnn.chunk(output, 2, -1)
            output = output * ttnn.silu(gate)
        elif self.activation_fn is None:
            pass
        else:
            raise ValueError(f"Activation function {self.activation_fn} not supported")
        return output


class RowParallelLinear(Module):
    """
    Linear layer with row parallel weights
    """

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        mesh_device=None,
        mesh_axis=0,
        fsdp_mesh_axis=None,
        ccl_manager=None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis
        self.ccl_manager = ccl_manager

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        ndev = tuple(self.mesh_device.shape)[self.mesh_axis]

        self.weight = Parameter(
            total_shape=[self.in_features, self.out_features], mesh_axes=[mesh_axis, fsdp_mesh_axis], device=mesh_device
        )
        self.bias = (
            Parameter(total_shape=[1, self.out_features * ndev], mesh_axes=[None, mesh_axis], device=mesh_device)
            if bias
            else None
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].transpose(0, 1)

        bias = state.pop("bias", None)
        if bias is not None:
            bias = bias.reshape(1, -1)
            if tuple(self.mesh_device.shape)[self.mesh_axis] > 1:
                zero_bias = torch.zeros(1, bias.shape[1] * (tuple(self.mesh_device.shape)[self.mesh_axis] - 1))
                bias = torch.cat([bias, zero_bias], dim=-1)
            state["bias"] = bias

    def forward(self, x: ttnn.Tensor, compute_kernel_config=None) -> ttnn.Tensor:
        """
        Expects x to be column fractured.
        Return output fractured on columns.
        """
        if self.fsdp_mesh_axis is not None and self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight.data)
            weight = self.ccl_manager.all_gather_persistent_buffer(
                unsqueezed_weight, dim=3, mesh_axis=self.fsdp_mesh_axis
            )

            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = self.weight.data

        M, K, N = x.padded_shape[-2], x.padded_shape[-1], weight.padded_shape[-1]
        core_grid = self.mesh_device.compute_with_storage_grid_size()
        matmul_config = get_matmul_config(M, K, N, core_grid)
        output = ttnn.experimental.minimal_matmul(
            input_tensor=x,
            weight_tensor=weight,
            bias_tensor=self.bias.data if self.bias is not None else None,
            config=matmul_config,
            compute_kernel_config=compute_kernel_config or self.compute_config,
        )

        if tuple(self.mesh_device.shape)[self.mesh_axis] > 1:
            needs_reshape = len(output.shape) <= 3
            if needs_reshape:
                output = ttnn.unsqueeze(output, 0)

            output = ttnn.experimental.reduce_scatter_minimal_async(
                output,
                persistent_output_buffers=self.ccl_manager.get_rs_ping_pong_buffer(
                    output.shape, dim=3, mesh_axis=self.mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_rs_ping_pong_semaphore(self.mesh_axis),
                num_links=self.ccl_manager.num_links,
                memory_config=ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM),
                topology=self.ccl_manager.topology,
                cluster_axis=self.mesh_axis,
                **self.ccl_manager.get_rs_hyperparams(output.shape),
            )

            if needs_reshape:
                output = ttnn.squeeze(output, 0)

        return output

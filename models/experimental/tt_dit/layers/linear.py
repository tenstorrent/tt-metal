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

        return _apply_activation_fn(output, self.activation_fn)


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

        self._mesh_axis_size = self.mesh_device.shape[self.mesh_axis] if self.mesh_axis is not None else 1

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        weight = state.pop("weight", None)
        bias = state.pop("bias", None)

        def permute_for_swiglu(tensor):
            assert self.activation_fn == "swiglu"
            ndev = self._mesh_axis_size
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

        return _apply_activation_fn(output, self.activation_fn)


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

        ndev = self.mesh_device.shape[self.mesh_axis] if self.mesh_axis is not None else 1

        self.weight = Parameter(
            total_shape=[self.in_features, self.out_features], mesh_axes=[mesh_axis, fsdp_mesh_axis], device=mesh_device
        )
        self.bias = (
            Parameter(total_shape=[1, self.out_features * ndev], mesh_axes=[None, mesh_axis], device=mesh_device)
            if bias
            else None
        )

        self._mesh_axis_size = ndev

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"] = state["weight"].transpose(0, 1)

        bias = state.pop("bias", None)
        if bias is not None:
            bias = bias.reshape(1, -1)
            if self._mesh_axis_size > 1:
                zero_bias = torch.zeros(1, bias.shape[1] * (self._mesh_axis_size - 1))
                bias = torch.cat([bias, zero_bias], dim=-1)
            state["bias"] = bias

    def forward(
        self,
        x: ttnn.Tensor,
        *,
        compute_kernel_config=None,
        use_persistent_buffer: bool = True,
    ) -> ttnn.Tensor:
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

        if self._mesh_axis_size > 1:
            needs_reshape = len(output.shape) <= 3
            if needs_reshape:
                output = ttnn.unsqueeze(output, 0)

            output = self.ccl_manager.reduce_scatter(
                output, dim=3, mesh_axis=self.mesh_axis, use_persistent_buffer=use_persistent_buffer
            )

            if needs_reshape:
                output = ttnn.squeeze(output, 0)

        return output


def _apply_activation_fn(t: ttnn.Tensor, activation_fn: str | None) -> ttnn.Tensor:
    if activation_fn is None:
        return t
    if activation_fn == "silu":
        return ttnn.silu(t)
    if activation_fn == "decomposed_gelu":
        return gelu_decomposed(t)
    if activation_fn == "quick_gelu":
        return t * ttnn.sigmoid_accurate(1.702 * t)  # quick approx gelu
    if activation_fn == "swiglu":
        t, gate = ttnn.chunk(t, 2, -1)
        return t * ttnn.silu(gate)

    msg = f"Activation function {activation_fn} not supported"
    raise ValueError(msg)


def prepare_chunked_linear_output(
    state: dict[str, torch.Tensor], *, prefix: str, device_count: int, chunks: int
) -> None:
    weight_key = f"{prefix}.weight"
    bias_key = f"{prefix}.bias"

    weight = state.get(weight_key)
    bias = state.get(bias_key)

    if weight is not None:
        _, in_dim = weight.shape
        weight = weight.reshape([chunks, device_count, -1, in_dim]).transpose(0, 1).reshape([-1, in_dim])
        state[weight_key] = weight

    if bias is not None:
        bias = state[bias_key].reshape([chunks, device_count, -1]).transpose(0, 1).reshape([-1])
        state[bias_key] = bias

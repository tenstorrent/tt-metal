# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn

from .module import Module, Parameter


class Linear(Module):
    """
    Linear layer with replicated weights
    """

    def __init__(self, in_features, out_features, bias=True, activation_fn=None, mesh_device=None, init=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        if activation_fn == "swiglu":
            # Double out features for fused swiglu activation
            self.out_features = self.out_features * 2
        self.activation_fn = activation_fn
        self.mesh_device = mesh_device

        self.weight = Parameter(shape=[in_features, out_features], device=mesh_device, init=init)
        self.bias = Parameter(shape=[1, out_features], device=mesh_device, init=init) if bias else None

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

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"].transpose_(0, 1)

        if "bias" in state:
            state["bias"] = state["bias"].reshape(1, -1)

    def forward(self, x: ttnn.Tensor, core_grid=None, compute_kernel_config=None) -> ttnn.Tensor:
        output = ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            core_grid=core_grid,
            compute_kernel_config=compute_kernel_config or self.compute_config,
        )
        if self.activation_fn == "gelu":
            output = ttnn.gelu(output, fast_and_approximate_mode=False)
        elif self.activation_fn == "swiglu":
            output, gate = ttnn.chunk(output, 2, -1)
            output = output * ttnn.silu(gate)
        else:
            assert self.activation_fn is None, f"Unsupported activation: {self.activation_fn}"
        return output


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
        init=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        if activation_fn == "swiglu":
            # Double out features for fused swiglu activation
            self.out_features = self.out_features * 2
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis
        self.ccl_manager = ccl_manager

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis
            assert self.ccl_manager is not None

        self.weight = Parameter(
            shape=[in_features, out_features],
            mesh_mapping={mesh_axis: 1, fsdp_mesh_axis: 0},
            device=mesh_device,
            init=init,
        )
        self.bias = (
            Parameter(shape=[1, out_features], mesh_mapping={mesh_axis: 1}, device=mesh_device, init=init)
            if bias
            else None
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        def permute_for_swiglu(tensor):
            assert self.activation_fn == "swiglu"
            ndev = self.mesh_device.shape[self.mesh_axis]
            tensor = tensor.reshape(-1, 2, ndev, tensor.shape[-1] // 2 // ndev)
            tensor = tensor.permute(0, 2, 1, 3)
            tensor = tensor.reshape(-1, self.out_features)
            assert tensor.shape[0] in [1, self.in_features]
            return tensor

        if "weight" in state:
            weight = state["weight"].transpose(0, 1)

            if self.activation_fn == "swiglu":
                weight = permute_for_swiglu(weight)

            state["weight"] = weight

        if "bias" in state:
            bias = state["bias"].reshape(1, -1)

            if self.activation_fn == "swiglu":
                bias = permute_for_swiglu(bias)

            state["bias"] = bias

    def forward(self, x: ttnn.Tensor, core_grid=None, compute_kernel_config=None) -> ttnn.Tensor:
        """
        Expects x to be replicated.
        Return output fractured on columns.
        """
        if self.fsdp_mesh_axis is not None and self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight)
            weight = ttnn.experimental.all_gather_async(
                unsqueezed_weight,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    unsqueezed_weight.shape, 2, self.fsdp_mesh_axis
                ),
                dim=2,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.fsdp_mesh_axis),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.fsdp_mesh_axis,
                # **self.ccl_manager.get_ag_hyperparams(unsqueezed_weight.shape),
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = self.weight

        output = ttnn.linear(
            x,
            weight,
            bias=self.bias,
            core_grid=core_grid,
            compute_kernel_config=compute_kernel_config or self.compute_config,
        )
        if self.activation_fn == "gelu":
            output = ttnn.gelu(output, fast_and_approximate_mode=False)
        elif self.activation_fn == "quick_gelu":
            output = output * ttnn.sigmoid(1.702 * output)  # quick approx gelu
        elif self.activation_fn == "silu":
            output = ttnn.silu(output, fast_and_approximate_mode=False)
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
        activation_fn=None,
        mesh_device=None,
        mesh_axis=0,
        fsdp_mesh_axis=None,
        ccl_manager=None,
        init=False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.activation_fn = activation_fn
        self.mesh_device = mesh_device
        self.mesh_axis = mesh_axis
        self.fsdp_mesh_axis = fsdp_mesh_axis
        self.ccl_manager = ccl_manager

        if self.fsdp_mesh_axis is not None:
            assert self.mesh_axis != self.fsdp_mesh_axis

        if init:
            # row-parallel bias must not be replicated across mesh_devices
            bias_init = torch.randn([1, out_features])
            if tuple(mesh_device.shape)[mesh_axis] > 1:
                zero_bias = torch.zeros(1, out_features * (tuple(mesh_device.shape)[mesh_axis] - 1))
                bias_init = torch.cat([bias_init, zero_bias], dim=-1)
        else:
            bias_init = False

        self.weight = Parameter(
            shape=[in_features, out_features],
            mesh_mapping={mesh_axis: 0, fsdp_mesh_axis: 1},
            device=mesh_device,
            init=init,
        )
        self.bias = (
            Parameter(shape=[1, out_features], mesh_mapping={mesh_axis: 1}, device=mesh_device, init=bias_init)
            if bias
            else None
        )

        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

    def _prepare_torch_state(self, state: dict[str, torch.Tensor]) -> None:
        if "weight" in state:
            state["weight"].transpose_(0, 1)

        if "bias" in state:
            bias = state["bias"].reshape(1, -1)

            if tuple(self.mesh_device.shape)[self.mesh_axis] > 1:
                zero_bias = torch.zeros(1, bias.shape[1] * (tuple(self.mesh_device.shape)[self.mesh_axis] - 1))
                bias = torch.cat([bias, zero_bias], dim=-1)

            state["bias"] = bias

    def forward(self, x: ttnn.Tensor, core_grid=None, compute_kernel_config=None) -> ttnn.Tensor:
        """
        Expects x to be column fractured.
        Return output fractured on columns.
        """
        if self.fsdp_mesh_axis is not None and self.mesh_device.shape[self.fsdp_mesh_axis] > 1:
            unsqueezed_weight = ttnn.unsqueeze_to_4D(self.weight)
            weight = ttnn.experimental.all_gather_async(
                unsqueezed_weight,
                persistent_output_buffer=self.ccl_manager.get_ag_ping_pong_buffer(
                    unsqueezed_weight.shape, 3, self.fsdp_mesh_axis
                ),
                dim=3,
                multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(self.fsdp_mesh_axis),
                num_links=self.ccl_manager.num_links,
                topology=self.ccl_manager.topology,
                cluster_axis=self.fsdp_mesh_axis,
                # **self.ccl_manager.get_ag_hyperparams(unsqueezed_weight.shape),
            )
            weight = ttnn.reshape(weight, (weight.shape[-2], weight.shape[-1]))
        else:
            weight = self.weight

        output = ttnn.linear(
            x,
            weight,
            bias=self.bias,
            core_grid=core_grid,
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

        if self.activation_fn is not None:
            assert self.activation_fn == "gelu"
            output = ttnn.gelu(output, fast_and_approximate_mode=False)

        return output

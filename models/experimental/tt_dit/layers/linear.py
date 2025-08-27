# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ..utils.tensor import bf16_tensor, bf16_tensor_2dshard


class Linear:
    """
    Linear layer with replicated weights
    """

    def __init__(self, in_features, out_features, bias=True, activation_fn=None, mesh_device=None, init=False):
        self.in_features = in_features
        self.out_features = out_features
        if activation_fn == "swiglu":
            # Double out features for fused swiglu activation
            self.out_features = self.out_features * 2
        self.activation_fn = activation_fn
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

    def to_cached_state_dict(self, path_prefix):
        weight_path = path_prefix + "weight"
        bias_path = path_prefix + "bias"
        ttnn.dump_tensor(weight_path, self.weight)
        if self.bias is not None:
            ttnn.dump_tensor(bias_path, self.bias)
        cache_dict = {"weight": weight_path}
        if self.bias is not None:
            cache_dict["bias"] = bias_path
        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.weight = ttnn.load_tensor(cache_dict["weight"], device=self.mesh_device)
        if "bias" in cache_dict:
            self.bias = ttnn.load_tensor(cache_dict["bias"], device=self.mesh_device)
        else:
            self.bias = None

    def load_state_dict(self, state_dict):
        """
        Loads the state dict into the layer.
        """
        weight = state_dict["weight"].transpose(0, 1)
        bias = state_dict.get("bias", None)
        if bias is not None:
            bias = bias.reshape(1, -1)

        self.weight = bf16_tensor(weight, device=self.mesh_device)
        if bias is not None:
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
        if self.activation_fn == "gelu":
            output = ttnn.gelu(output, fast_and_approximate_mode=False)
        elif self.activation_fn == "swiglu":
            output, gate = ttnn.chunk(output, 2, -1)
            output = output * ttnn.silu(gate)
        else:
            assert self.activation_fn is None, f"Unsupported activation: {self.activation_fn}"
        return output


class ColParallelLinear:
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

        if init:
            if fsdp_mesh_axis is not None:
                self.weight = bf16_tensor_2dshard(
                    torch.randn(in_features, out_features),
                    device=self.mesh_device,
                    shard_mapping={mesh_axis: 1, fsdp_mesh_axis: 0},
                )
            else:
                self.weight = bf16_tensor(
                    torch.randn(in_features, out_features),
                    device=self.mesh_device,
                    mesh_axis=self.mesh_axis,
                    shard_dim=-1,
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

    def to_cached_state_dict(self, path_prefix):
        weight_path = path_prefix + "weight"
        bias_path = path_prefix + "bias"
        ttnn.dump_tensor(weight_path, self.weight)
        if self.bias is not None:
            ttnn.dump_tensor(bias_path, self.bias)
        cache_dict = {"weight": weight_path}
        if self.bias is not None:
            cache_dict["bias"] = bias_path
        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.weight = ttnn.load_tensor(cache_dict["weight"], device=self.mesh_device)
        if "bias" in cache_dict:
            self.bias = ttnn.load_tensor(cache_dict["bias"], device=self.mesh_device)
        else:
            self.bias = None

    def load_state_dict(self, state_dict):
        """
        Loads the state dict into the layer.
        """
        weight = state_dict["weight"].transpose(0, 1)
        bias = state_dict.get("bias", None)

        def permute_for_swiglu(tensor):
            assert self.activation_fn == "swiglu"
            ndev = self.mesh_device.shape[self.mesh_axis]
            tensor = tensor.reshape(-1, 2, ndev, tensor.shape[-1] // 2 // ndev)
            tensor = tensor.permute(0, 2, 1, 3)
            tensor = tensor.reshape(-1, self.out_features)
            assert tensor.shape[0] in [1, self.in_features]
            return tensor

        if self.activation_fn == "swiglu":
            weight = permute_for_swiglu(weight)
            if bias is not None:
                bias = permute_for_swiglu(bias)

        if self.fsdp_mesh_axis is not None:
            self.weight = bf16_tensor_2dshard(
                weight, device=self.mesh_device, shard_mapping={self.mesh_axis: 1, self.fsdp_mesh_axis: 0}
            )
        else:
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
        elif self.activation_fn == "swiglu":
            output, gate = ttnn.chunk(output, 2, -1)
            output = output * ttnn.silu(gate)
        elif self.activation_fn is None:
            pass
        else:
            raise ValueError(f"Activation function {self.activation_fn} not supported")
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
        activation_fn=None,
        mesh_device=None,
        mesh_axis=0,
        fsdp_mesh_axis=None,
        ccl_manager=None,
        init=False,
    ):
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
            if self.fsdp_mesh_axis is not None:
                self.weight = bf16_tensor_2dshard(
                    torch.randn(in_features, out_features),
                    device=self.mesh_device,
                    shard_mapping={self.mesh_axis: 0, self.fsdp_mesh_axis: 1},
                )
            else:
                self.weight = bf16_tensor(
                    torch.randn(in_features, out_features),
                    device=self.mesh_device,
                    mesh_axis=self.mesh_axis,
                    shard_dim=-2,
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

    def to_cached_state_dict(self, path_prefix):
        weight_path = path_prefix + "weight"
        bias_path = path_prefix + "bias"
        ttnn.dump_tensor(weight_path, self.weight)
        if self.bias is not None:
            ttnn.dump_tensor(bias_path, self.bias)
        cache_dict = {"weight": weight_path}
        if self.bias is not None:
            cache_dict["bias"] = bias_path
        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.weight = ttnn.load_tensor(cache_dict["weight"], device=self.mesh_device)
        if "bias" in cache_dict:
            self.bias = ttnn.load_tensor(cache_dict["bias"], device=self.mesh_device)
        else:
            self.bias = None

    def load_state_dict(self, state_dict):
        """
        Loads the state dict into the layer.
        """
        weight = state_dict["weight"].transpose(0, 1)
        bias = state_dict.get("bias", None)

        if self.fsdp_mesh_axis is not None:
            self.weight = bf16_tensor_2dshard(
                weight, device=self.mesh_device, shard_mapping={self.mesh_axis: 0, self.fsdp_mesh_axis: 1}
            )
        else:
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

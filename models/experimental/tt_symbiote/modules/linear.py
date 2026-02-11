# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Linear layer implementations for TTNN."""

from torch import nn
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, deallocate_weights_after, run_on_devices, DeviceArch
from models.experimental.tt_symbiote.core.run_config import trace_enabled


@trace_enabled
class TTNNLinear(TTNNModule):
    """TTNN-accelerated linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    @classmethod
    def from_parameters(cls, weight, bias=None):
        """Create TTNNLinear from a weight parameter."""
        new_linear = cls(
            in_features=weight.shape[1],
            out_features=weight.shape[0],
        )
        new_linear.weight = weight
        new_linear.bias = bias
        new_linear.preprocess_weights()
        del new_linear.weight
        del new_linear.bias
        return new_linear

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        """Create TTNNLinear from PyTorch Linear layer."""
        new_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_linear._fallback_torch_layer = linear
        new_linear.weight = linear.weight
        new_linear.bias = linear.bias
        return new_linear

    @property
    def _parameters(self):
        return self.torch_layer._parameters

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(self.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    def move_weights_to_device_impl(self):
        """Move weights to TTNN device."""
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None

    def deallocate_weights_impl(self):
        """Deallocate weights from device."""
        ttnn.deallocate(self.tt_weight)
        if self.tt_bias is not None:
            ttnn.deallocate(self.tt_bias)
        super().deallocate_weights_impl()

    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(input_tensor, self.tt_weight, bias=self.tt_bias, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [self.out_features])
        return tt_output


class TTNNLinearInputShardedWeightSharded(TTNNLinear):
    """TTNN-accelerated linear layer."""

    def __init__(self, in_features, out_features, input_dim, weight_dim) -> None:
        super().__init__(in_features, out_features)
        self.input_dim = input_dim
        self.weight_dim = weight_dim
        assert (
            self.input_dim == -1
        ), f"Only input sharding on second to last dimension is supported, got {self.input_dim}."
        assert self.weight_dim == -2, f"Only weight sharding on last dimension is supported, got {self.weight_dim}."

    def preprocess_weights_impl(self):
        self.tt_bias_host = self.bias
        self.tt_weight_host = self.weight

    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
            )
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.input_dim),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None


class TTNNLinearIColShardedWRowSharded(TTNNLinearInputShardedWeightSharded):
    """TTNN-accelerated linear layer with input and weight sharded on last dimension."""

    def __init__(self, in_features, out_features) -> None:
        super().__init__(in_features, out_features, input_dim=-1, weight_dim=-2)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        if len(input_tensor.tensor_topology().placements()) == 1:
            assert (
                input_tensor.tensor_topology().placements()[0].dim == self.input_dim
            ), f"Input tensor must be sharded on dimension {self.input_dim}."
        elif len(input_tensor.tensor_topology().placements()) == 2:
            assert (
                input_tensor.tensor_topology().placements()[0].dim == 0
            ), f"Input tensor must be sharded on batch dim (0)."
            assert (
                input_tensor.tensor_topology().placements()[1].dim == self.input_dim
            ), f"Input tensor must be sharded on dimension {self.input_dim}."
        else:
            raise RuntimeError(
                f"Input tensor must be sharded on either batch dim (0) or input dim ({self.input_dim}), but got tensor with placements: {input_tensor.tensor_topology().placements()}"
            )
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.experimental.reduce_scatter_minimal_async(
            tt_output,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.device_state.ccl_manager.get_and_cycle_rs_semaphore_handles(1),
            barrier_semaphore=self.device_state.ccl_manager.get_and_cycle_barrier_semaphore_handle(1),
            num_links=1,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            intermediate_memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
            chunks_per_sync=10,
            num_workers_per_link=2,
            num_buffers_per_channel=2,
        )
        if self.tt_bias is not None:
            tt_output += self.tt_bias
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


class TTNNLinearLLama(TTNNLinear):
    """TTNN Linear layer optimized for LLaMA models using bfloat8."""

    def preprocess_weights_impl(self):
        """Preprocess linear weights with bfloat8 precision."""
        self.tt_weight_host = preprocess_linear_weight(self.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        self.tt_bias_host = None
        if self.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(self.bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


class TTNNLinearLLamaIColShardedWRowSharded(TTNNLinearIColShardedWRowSharded):
    """TTNN Linear layer optimized for LLaMA models using bfloat8."""

    def move_weights_to_device_impl(self):
        if isinstance(self.tt_weight_host, torch.Tensor):
            self.tt_weight_host = preprocess_linear_weight(
                self.tt_weight_host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
            )
        if isinstance(self.tt_bias_host, torch.Tensor):
            self.tt_bias_host = preprocess_linear_bias(
                self.tt_bias_host,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.input_dim),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None

    @deallocate_weights_after
    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


class TTNNLinearLLamaBFloat16(TTNNLinear):
    """TTNN Linear layer optimized for LLaMA models using bfloat16."""

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


class PytorchLinearActivation(nn.Module):
    def __init__(self, dense, act_fn) -> None:
        super().__init__()
        self.dense = dense
        self.intermediate_act_fn = act_fn

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


@trace_enabled
class TTNNLinearActivation(TTNNModule):
    """Linear layer with activation using TTNN."""

    @classmethod
    def from_parameters(cls, weight, linear_class, ttnn_act_fn, nn_act_fn, bias=None):
        new_linear = cls()
        new_linear.dense = linear_class.from_parameters(weight=weight, bias=bias)
        new_linear.activation = ttnn_act_fn
        return new_linear

    @classmethod
    def from_torch(cls, linear: nn.Linear, linear_class, ttnn_act_fn, nn_act_fn):
        new_linear = cls()
        new_linear._fallback_torch_layer = PytorchLinearActivation(dense=linear, act_fn=nn_act_fn)
        new_linear.dense = linear_class.from_torch(linear)
        new_linear.activation = ttnn_act_fn
        return new_linear

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states.to_ttnn)
        return hidden_states


class TTNNLinearGelu:
    """Linear layer with GELU activation using TTNN."""

    @classmethod
    def from_parameters(cls, weight, bias=None, linear_class=TTNNLinear):
        new_linear = TTNNLinearActivation.from_parameters(weight, linear_class, ttnn.gelu, nn.GELU(), bias)
        return new_linear

    @classmethod
    def from_torch(cls, linear: nn.Linear, linear_class=TTNNLinear):
        new_linear = TTNNLinearActivation.from_torch(linear, linear_class, ttnn.gelu, nn.GELU())
        return new_linear


class TTNNLinearSilu:
    """SiLU activated Linear module with TTNN acceleration."""

    @classmethod
    def from_parameters(cls, weight, bias=None, linear_class=TTNNLinear):
        new_linear = TTNNLinearActivation.from_parameters(weight, linear_class, ttnn.silu, nn.SiLU(), bias)
        return new_linear

    @classmethod
    def from_torch(cls, linear: nn.Linear, linear_class=TTNNLinear):
        new_linear = TTNNLinearActivation.from_torch(linear, linear_class, ttnn.silu, nn.SiLU())
        return new_linear


class TTNNViTIntermediate(TTNNLinearGelu):
    """ViT Intermediate module with TTNN acceleration."""

    @classmethod
    def from_torch(cls, torch_vit_intermediate: "ViTIntermediate"):
        assert (
            torch_vit_intermediate.intermediate_act_fn.__class__.__name__ == "GELUActivation"
        ), "Only GELU activation is supported."
        new_intermediate = cls()
        new_intermediate._fallback_torch_layer = torch_vit_intermediate
        new_intermediate.dense = TTNNLinear.from_torch(torch_vit_intermediate.dense)
        return new_intermediate

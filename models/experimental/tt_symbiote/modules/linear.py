# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Linear layer implementations for TTNN."""

from torch import nn
import torch
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight
import ttnn
from models.experimental.tt_symbiote.core.module import (
    TTNNModule,
    deallocate_weights_after,
    run_on_devices,
    DeviceArch,
)
from models.experimental.tt_symbiote.core.run_config import trace_enabled, trace_disabled


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
        tt_output = ttnn.reduce_scatter(
            tt_output,
            dim=3,
            num_links=1,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )
        if self.tt_bias is not None:
            tt_output += self.tt_bias
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


class TTNNLinearIColShardedWAllReduced(TTNNLinearIColShardedWRowSharded):
    def move_weights_to_device_impl(self):
        # Keep weight row-sharded, but bias replicated because output is all-reduced.
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
                weights_mesh_mapper=ttnn.ReplicateTensorToMesh(self.device),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass: matmul + all_reduce.

        The input is column-sharded across devices. After matmul each device
        holds a partial sum.  all_reduce sums the partials so every device
        gets the full output (replicated).
        """
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
                f"Input tensor must be sharded on either batch dim (0) or input dim ({self.input_dim}), "
                f"but got tensor with placements: {input_tensor.tensor_topology().placements()}"
            )

        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        if len(input_shape) == 2:
            input_shape.insert(0, 1)  # Add batch dimension if missing
        if len(input_shape) == 3:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)

        # Matmul: partial sum on each device
        tt_output = ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        tt_output = ttnn.all_reduce(
            tt_output,
            num_links=1,
            topology=ttnn.Topology.Ring,
            cluster_axis=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        if self.tt_bias is not None:
            tt_output += self.tt_bias

        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


@trace_disabled
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


@trace_disabled
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


class TTNNLinearInputReplicatedWeightSharded(TTNNLinear):
    """TTNN-accelerated linear layer."""

    def __init__(self, in_features, out_features, weight_dim) -> None:
        super().__init__(in_features, out_features)
        self.weight_dim = weight_dim
        assert self.weight_dim == -1, f"Only weight sharding on last dimension is supported, got {self.weight_dim}."

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
                weights_mesh_mapper=ttnn.shard_tensor_to_mesh_mapper(self.device, dim=self.weight_dim),
            )
        self.tt_weight = ttnn.to_device(self.tt_weight_host, self.device)
        self.tt_bias = ttnn.to_device(self.tt_bias_host, self.device) if self.tt_bias_host is not None else None


class TTNNLinearIReplicatedWColSharded(TTNNLinearInputReplicatedWeightSharded):
    """TTNN-accelerated linear layer with input and weight sharded on last dimension."""

    def __init__(self, in_features, out_features) -> None:
        super().__init__(in_features, out_features, weight_dim=-1)

    @run_on_devices(DeviceArch.T3K)
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through linear layer."""
        if input_tensor.layout != ttnn.TILE_LAYOUT:
            input_tensor = ttnn.to_layout(input_tensor, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        input_tensor_shape = list(input_tensor.shape)
        input_shape = list(input_tensor_shape)
        while len(input_shape) < 4:
            input_shape.insert(1, 1)  # Add batch dimensions if needed
        input_tensor = ttnn.reshape(input_tensor, input_shape)
        tt_output = ttnn.linear(input_tensor, self.tt_weight, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        if self.tt_bias is not None:
            tt_output += self.tt_bias
        tt_output = ttnn.reshape(tt_output, input_tensor_shape[:-1] + [-1])
        return tt_output


@trace_disabled
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
        hidden_states = self.activation(hidden_states)
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
    def from_torch(cls, torch_vit_intermediate: object):
        assert (
            torch_vit_intermediate.intermediate_act_fn.__class__.__name__ == "GELUActivation"
        ), "Only GELU activation is supported."
        new_intermediate = cls()
        new_intermediate._fallback_torch_layer = torch_vit_intermediate
        new_intermediate.dense = TTNNLinear.from_torch(torch_vit_intermediate.dense)
        return new_intermediate


class TTNNQwen3OmniVisionMLP(TTNNModule):
    """TTNN implementation of Qwen3OmniMoeVisionMLP (fc1 -> act -> fc2)."""

    def __init__(self):
        super().__init__()
        self.hidden_size = None
        self.intermediate_size = None

        self.linear_fc1 = None
        self.linear_fc2 = None

        # Store activation name as lower-case string (e.g. "silu", "gelu").
        self.act_fn = None

    @classmethod
    def from_torch(cls, torch_mlp):
        module = cls()
        module._fallback_torch_layer = torch_mlp

        module.hidden_size = torch_mlp.hidden_size
        module.intermediate_size = torch_mlp.intermediate_size

        # Use tensor-parallel MLP linears to reduce DRAM pressure on mesh:
        # fc1: replicated input + column-sharded weight (sharded intermediate)
        # fc2: column-sharded input + row-sharded weight + all-reduce (replicated output)
        module.linear_fc1 = TTNNLinearIReplicatedWColSharded.from_torch(torch_mlp.linear_fc1)
        module.linear_fc2 = TTNNLinearIColShardedWAllReduced.from_torch(torch_mlp.linear_fc2)

        act_fn = getattr(torch_mlp, "act_fn", None)
        act_name = getattr(act_fn, "__name__", None) or str(act_fn)
        module.act_fn = act_name.lower()

        return module

    def preprocess_weights_impl(self):
        self.linear_fc1.preprocess_weights()
        self.linear_fc2.preprocess_weights()

    def move_weights_to_device_impl(self):
        self.linear_fc1.move_weights_to_device()
        self.linear_fc2.move_weights_to_device()

    def deallocate_weights_impl(self):
        self.linear_fc1.deallocate_weights()
        self.linear_fc2.deallocate_weights()

    @run_on_devices(DeviceArch.T3K)
    def forward(self, hidden_states):
        # Most TTNN paths already provide a ttnn.Tensor; keep this conversion as a safety net.
        if not isinstance(hidden_states, ttnn.Tensor):
            hidden_states = ttnn.from_torch(
                hidden_states, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
            )

        # Vision MLP expects full hidden width. In distributed runs we may see either:
        # - width-sharded input (e.g. 144), which must be gathered, or
        # - concatenated width (e.g. 9216), which must be clamped.
        in_width = int(hidden_states.shape[-1])
        if in_width > int(self.hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)
        elif in_width < int(self.hidden_size):
            hidden_states = ttnn.all_gather(
                hidden_states,
                dim=-1,
                cluster_axis=1,
                num_links=1,
                topology=ttnn.Topology.Linear,
            )

        if hidden_states.layout != ttnn.TILE_LAYOUT:
            hidden_states = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = self.linear_fc1(hidden_states)

        # HF Qwen3OmniMoeVisionMLP uses ACT2FN[config.hidden_act] (typically "silu" or "gelu").
        if self.act_fn in {"silu", "swish"}:
            hidden_states = ttnn.silu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        elif self.act_fn in {"gelu", "gelu_new"}:
            hidden_states = ttnn.gelu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Default to silu to avoid hard failure if an unexpected activation is encountered.
            hidden_states = ttnn.silu(hidden_states, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        hidden_states = self.linear_fc2(hidden_states)

        # Some distributed paths may return concatenated width shards after FC2.
        # Keep only the logical hidden width expected by the residual add.
        out_width = int(hidden_states.shape[-1])
        if out_width != int(self.hidden_size) and out_width > int(self.hidden_size):
            rank = len(hidden_states.shape)
            starts = [0] * rank
            ends = [int(s) for s in hidden_states.shape]
            ends[-1] = int(self.hidden_size)
            hidden_states = ttnn.slice(hidden_states, starts, ends)

        # Match MoE/attention patterns: when distributed tensors surface with a
        # widened logical width at Torch boundary, compose + clamp before residual add.
        if self.device is not None and self.device.get_num_devices() > 1:
            composed = ttnn.to_torch(hidden_states, mesh_composer=ttnn.ConcatMeshToTensor(self.device, dim=-1))
            if int(composed.shape[-1]) != int(self.hidden_size):
                return composed[..., : int(self.hidden_size)]

        return hidden_states

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Linear layer implementations for TTNN."""

from torch import nn
from ttnn.model_preprocessing import preprocess_linear_bias, preprocess_linear_weight

import ttnn
from models.experimental.tt_symbiote.core.module import TTNNModule, deallocate_weights_after


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
    def from_torch(cls, linear: nn.Linear):
        """Create TTNNLinear from PyTorch Linear layer."""
        new_linear = TTNNLinear(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_linear._fallback_torch_layer = linear
        return new_linear

    def preprocess_weights_impl(self):
        """Preprocess linear weights for TTNN."""
        if self.torch_layer is None:
            self._fallback_torch_layer = nn.Linear(self.in_features, self.out_features)
        self.tt_weight_host = preprocess_linear_weight(
            self.torch_layer.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_host = None
        if self.torch_layer.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.torch_layer.bias, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
            )

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


class TTNNLinearLLama(TTNNLinear):
    """TTNN Linear layer optimized for LLaMA models using bfloat8."""

    def preprocess_weights_impl(self):
        """Preprocess linear weights with bfloat8 precision."""
        if self.torch_layer is None:
            self._fallback_torch_layer = nn.Linear(self.in_features, self.out_features)
        self.tt_weight_host = preprocess_linear_weight(
            self.torch_layer.weight, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
        )
        self.tt_bias_host = None
        if self.torch_layer.bias is not None:
            self.tt_bias_host = preprocess_linear_bias(
                self.torch_layer.bias, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT
            )

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        """Create TTNNLinearLLama from PyTorch Linear layer."""
        new_linear = TTNNLinearLLama(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_linear._fallback_torch_layer = linear
        return new_linear

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


class TTNNLinearLLamaBFloat16(TTNNLinear):
    """TTNN Linear layer optimized for LLaMA models using bfloat16."""

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        """Create TTNNLinearLLama from PyTorch Linear layer."""
        new_linear = TTNNLinearLLamaBFloat16(
            in_features=linear.in_features,
            out_features=linear.out_features,
        )
        new_linear._fallback_torch_layer = linear
        return new_linear

    @deallocate_weights_after
    def forward(self, input_tensor: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass with automatic weight deallocation."""
        return super().forward(input_tensor)


class PytorchLinearGelu(nn.Module):
    def __init__(self, dense) -> None:
        super().__init__()
        self.dense = dense
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class TTNNLinearGelu(TTNNModule):
    """ViT Intermediate module with TTNN acceleration."""

    @classmethod
    def from_torch(cls, linear: nn.Linear):
        new_intermediate = TTNNViTIntermediate()
        new_intermediate._fallback_torch_layer = PytorchLinearGelu(
            dense=linear,
        )
        new_intermediate.dense = TTNNLinear.from_torch(linear)
        return new_intermediate

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = ttnn.gelu(hidden_states.to_ttnn)
        return hidden_states


class TTNNViTIntermediate(TTNNLinearGelu):
    """ViT Intermediate module with TTNN acceleration."""

    @classmethod
    def from_torch(cls, torch_vit_intermediate: "ViTIntermediate"):
        assert (
            torch_vit_intermediate.intermediate_act_fn.__class__.__name__ == "GELUActivation"
        ), "Only GELU activation is supported."
        new_intermediate = TTNNViTIntermediate()
        new_intermediate._fallback_torch_layer = torch_vit_intermediate
        new_intermediate.dense = TTNNLinear.from_torch(torch_vit_intermediate.dense)
        return new_intermediate

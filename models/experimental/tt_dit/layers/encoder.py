# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from ..models.transformers.attention_encoders import CLIPAttentionParameters, CLIPAttention
from ..parallel.config import EncoderParallelManager
from ..utils.tensor import bf16_tensor
from ..utils.substate import substate
from .feedforward import ParallelFeedForward
import ttnn


class CLIPEncoderLayer:
    def __init__(self, parameters, config, parallel_manager: EncoderParallelManager = None) -> None:
        self.config = config
        self._parallel_manager = parallel_manager
        self.mesh_device = getattr(parameters, "mesh_device", None)

        # will be created in load_state_dict
        self._self_attn = None
        self._mlp = None

        # layer norm parameters
        self._layer_norm1 = parameters.layer_norm1_weight
        self._layer_norm1_bias = parameters.layer_norm1_bias
        self._layer_norm2 = parameters.layer_norm2_weight
        self._layer_norm2_bias = parameters.layer_norm2_bias
        self._layer_norm_eps = config.layer_norm_eps

    def load_state_dict(self, state_dict):
        # layer norm weights
        self._layer_norm1 = bf16_tensor(
            state_dict["layer_norm1.weight"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )
        self._layer_norm1_bias = bf16_tensor(
            state_dict["layer_norm1.bias"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )
        self._layer_norm2 = bf16_tensor(
            state_dict["layer_norm2.weight"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )
        self._layer_norm2_bias = bf16_tensor(
            state_dict["layer_norm2.bias"], device=self.mesh_device, layout=ttnn.TILE_LAYOUT
        )

        attn_params = CLIPAttentionParameters.from_torch(
            substate(state_dict, "self_attn"),
            config=self.config,
            mesh_device=self.mesh_device,
            parallel_manager=self._parallel_manager,
        )
        self._self_attn = CLIPAttention(attn_params, self.config, self._parallel_manager)

        # MLP
        if self.config.hidden_act == "gelu":
            activation_fn = "gelu"
        elif self.config.hidden_act == "quick_gelu":
            activation_fn = "quick_gelu"
        else:
            raise ValueError(f"Unsupported activation function: {self.config.hidden_act}")

        self.mlp = ParallelFeedForward(
            dim=dim,
            dim_out=dim,
            activation_fn=activation_fn,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=ccl_manager,  # TODO: add ccl_manager
            init=init,
        )

        self.mlp.load_state_dict(rename_ff_state(substate(state_dict, "mlp")))

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        parallel_manager: EncoderParallelManager = None,
    ) -> ttnn.Tensor:
        # self attention block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self._layer_norm1, bias=self._layer_norm1_bias, epsilon=self._layer_norm_eps
        )

        attn_output = self._self_attn(hidden_states, causal_attention_mask, parallel_manager=parallel_manager)

        hidden_states = residual + attn_output

        # MLP block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self._layer_norm2, bias=self._layer_norm2_bias, epsilon=self._layer_norm_eps
        )

        mlp_output = self._mlp(hidden_states, parallel_manager=parallel_manager)

        hidden_states = residual + mlp_output

        return hidden_states

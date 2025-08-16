# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
from ..models.transformers.attention_encoders import EncoderAttention

# from ..parallel.config import EncoderParallelManager
from ..utils.tensor import bf16_tensor
from ..utils.substate import substate
from .feedforward import ParallelFeedForward
import ttnn


class CLIPEncoderLayer:
    def __init__(
        self,
        parameters,
        config,
        mesh_device: ttnn.Device,
        parallel_manager=None,
        ccl_manager=None,
        parallel_config=None,
    ) -> None:
        self.config = config
        self.parallel_manager = parallel_manager
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config
        self.mesh_device = mesh_device

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

        self._self_attn = EncoderAttention(
            self.config, self.mesh_device, ccl_manager=self.ccl_manager, parallel_config=self.parallel_config
        )

        # load attention weights
        attn_state = substate(state_dict, "self_attn")
        self._self_attn.load_state_dict(attn_state)

        # create and load MLP
        if self.config.hidden_act == "gelu":
            activation_fn = "gelu"
        elif self.config.hidden_act == "quick_gelu":
            activation_fn = "quick_gelu"
        else:
            raise ValueError(f"Unsupported activation function: {self.config.hidden_act}")
        # breakpoint()
        self._mlp = ParallelFeedForward(
            dim=self.config.hidden_size,
            dim_out=self.config.hidden_size,
            activation_fn=activation_fn,
            mesh_device=self.mesh_device,
            mesh_axis=self.parallel_config.tensor_parallel.mesh_axis,
            ccl_manager=self.ccl_manager,
        )

        # load MLP weights
        mlp_state = substate(state_dict, "mlp")
        self._mlp.load_state_dict(mlp_state)

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        ccl_manager=None,
        parallel_config=None,
    ) -> ttnn.Tensor:
        logger.info(f"Starting CLIPEncoderLayer forward pass, input shape: {hidden_states.shape}")

        # self attention block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self._layer_norm1, bias=self._layer_norm1_bias, epsilon=self._layer_norm_eps
        )
        logger.info(f" Layer norm 1 done, shape: {hidden_states.shape}")

        attn_output = self._self_attn(
            hidden_states, causal_attention_mask, ccl_manager=ccl_manager, parallel_config=parallel_config
        )
        logger.info(f"Self-attention done, output shape: {attn_output.shape}")

        hidden_states = residual + attn_output
        logger.info(f"Residual added, shape: {hidden_states.shape}")

        # MLP block
        residual = hidden_states
        hidden_states = ttnn.layer_norm(
            hidden_states, weight=self._layer_norm2, bias=self._layer_norm2_bias, epsilon=self._layer_norm_eps
        )
        logger.info(f"Layer norm 2 done, shape: {hidden_states.shape}")

        mlp_output = self._mlp(hidden_states)
        logger.info(f"MLP done, output shape: {mlp_output.shape}")

        hidden_states_shape = list(mlp_output.shape)
        logger.debug(f"Saving original shape: {hidden_states_shape}")

        breakpoint()
        # AllReduce output
        logger.debug(f"Starting reduce_scatter_minimal_async...")
        mlp_output_scattered = ttnn.experimental.reduce_scatter_minimal_async(
            mlp_output,
            dim=3,
            multi_device_global_semaphore=ccl_manager.get_rs_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ccl_manager.topology,
            cluster_axis=parallel_config.tensor_parallel.mesh_axis,
        )
        logger.debug(f"reduce_scatter completed, shape: {mlp_output_scattered.shape}")

        logger.debug(f"Starting all_gather_async...")
        mlp_output = ttnn.experimental.all_gather_async(
            mlp_output_scattered,
            dim=3,
            cluster_axis=parallel_config.tensor_parallel.mesh_axis,
            mesh_device=ccl_manager.mesh_device,
            topology=ccl_manager.topology,
            multi_device_global_semaphore=ccl_manager.get_ping_pong_semaphore(),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        logger.debug(f"all_gather completed, shape: {mlp_output.shape}")

        logger.debug(f"Reshaping back to original shape...")
        mlp_output = ttnn.reshape(mlp_output, hidden_states_shape)
        logger.debug(f"Target shape: {hidden_states_shape}, current shape: {mlp_output.shape}")
        logger.debug(f"Final MLP output shape: {mlp_output.shape}")

        logger.info("Adding final residual connection...")
        hidden_states = residual + mlp_output
        logger.info(f"CLIPEncoderLayer completed, final shape: {hidden_states.shape}")

        return hidden_states

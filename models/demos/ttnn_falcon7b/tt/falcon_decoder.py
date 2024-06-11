# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import ttnn

from .falcon_attention import TtFalconAttention
from .falcon_mlp import TtFalconMLP


class TtFalconDecoderLayer:
    def __init__(
        self,
        device,
        config,
        model_config,
        parameters,
    ):
        self.parameters = parameters
        self.hidden_size = config.hidden_size
        self.device = device
        self.model_config = model_config

        assert config.parallel_attn, "Path for config.parallel_attn=False is not implemented in TtFalconDecoderLayer!"

        self.self_attn = TtFalconAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=config.max_position_embeddings,
            model_config=model_config,
            parameters=parameters.self_attention,
            core_grid=device.core_grid if isinstance(device, ttnn.Device) else device.get_devices()[0].core_grid,
        )

        self.mlp = TtFalconMLP(model_config, parameters=parameters.mlp)

        self.input_layernorm_weight = parameters.input_layernorm.weight
        self.input_layernorm_bias = parameters.input_layernorm.bias
        self.layernorm_eps = config.layer_norm_epsilon

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        alibi: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[ttnn.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor, Optional[Tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """Input shape: [batch, 1, seq_len, hidden_size]"""

        assert not output_attentions

        residual = hidden_states
        layernorm_output = ttnn.layer_norm(
            hidden_states,
            epsilon=self.layernorm_eps,
            memory_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
        )
        layernorm_output = ttnn.mul(
            layernorm_output,
            self.input_layernorm_weight,
            memory_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
        )
        layernorm_output = ttnn.add(
            layernorm_output,
            self.input_layernorm_bias,
            memory_config=self.model_config["INPUT_LAYERNORM_OUTPUT_MEMCFG"],
        )

        # Self Attention
        attn_outputs = self.self_attn(
            hidden_states=layernorm_output,
            alibi=alibi,
            attention_mask=attention_mask,
            llm_mode=llm_mode,
            user_id=user_id,
            layer_past=layer_past,
            layer_past_len=layer_past_len,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        attention_output, outputs = attn_outputs[0], attn_outputs[1:]

        mlp_output = self.mlp(layernorm_output)
        ttnn.deallocate(layernorm_output)

        output = ttnn.add(
            mlp_output,
            attention_output,
            memory_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
        )
        ttnn.deallocate(mlp_output)
        ttnn.deallocate(attention_output)

        output = ttnn.add(
            output,
            residual,
            memory_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
        )

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (
                output,
                (),
            )  # Ignore past-cache

        return outputs

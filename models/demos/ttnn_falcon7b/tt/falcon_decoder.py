# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
from torch import nn
from typing import Optional, Tuple

import tt_lib
import ttnn

from models.demos.ttnn_falcon7b.tt.falcon_attention import TtFalconAttention
from models.demos.ttnn_falcon7b.tt.falcon_mlp import TtFalconMLP
from models.utility_functions import pad_by_zero


class TtFalconDecoderLayer(nn.Module):
    def __init__(
        self,
        device,
        config,
        max_position_embeddings,
        model_config,
        parameters,
    ):
        super().__init__()
        self.parameters = parameters
        self.hidden_size = config.hidden_size
        self.device = device
        self.max_position_embeddings = max_position_embeddings
        self.model_config = model_config

        assert config.parallel_attn, "Path for config.parallel_attn=False is not implemented in TtFalconDecoderLayer!"

        self.self_attn = TtFalconAttention(
            device=device,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            model_config=model_config,
            parameters=parameters.self_attention,
        )

        self.mlp = TtFalconMLP(
            device=device,
            model_config=model_config,
            parameters=parameters.mlp,
        )

        self.input_layernorm_weight = parameters.input_layernorm.weight
        self.input_layernorm_bias = parameters.input_layernorm.bias

        self.layernorm_eps = config.layer_norm_epsilon

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        alibi: torch.Tensor,
        attention_mask: torch.Tensor,
        llm_mode: str,
        user_id: int = 0,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor]] = None,
        layer_past_len: int = 0,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[tt_lib.tensor.Tensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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

        # MLP
        # mlp will deallocate layernorm_output
        mlp_output = self.mlp(layernorm_output)
        ttnn.deallocate(layernorm_output)

        output = ttnn.add(
            mlp_output,
            attention_output,
            memory_config=self.model_config["PARALLEL_ATTN_ADD_OUTPUT_MEMCFG"],
        )
        ttnn.deallocate(mlp_output)
        ttnn.deallocate(attention_output)

        # dropout_add
        # For inference, this is just add
        output = ttnn.add(
            output,
            residual,
            memory_config=self.model_config["DROPOUT_ADD_OUTPUT_MEMCFG"],
        )
        ttnn.deallocate(residual)

        if use_cache:
            outputs = (output,) + outputs
        else:
            outputs = (
                output,
                (),
            )  # Outputs should be empty if we ignore layer_past as well

        return outputs

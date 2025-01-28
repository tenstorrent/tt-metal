# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from torch import nn
from typing import Any, Optional, Tuple, Union

from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_sdpa_attention import ttnn_CLIPSdpaAttention
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_mlp import ttnn_CLIPMLP

CLIP_ATTENTION_CLASSES = {
    # "eager": CLIPAttention,
    "sdpa": ttnn_CLIPSdpaAttention,
    # "flash_attention_2": CLIPFlashAttention2,
}


class ttnn_CLIPEncoderLayer:
    def __init__(self, config):
        self.embed_dim = config.hidden_size
        self.self_attn = CLIP_ATTENTION_CLASSES[config._attn_implementation](config)
        self.layer_norm1 = ttnn.layer_norm
        self.mlp = ttnn_CLIPMLP(config)
        self.layer_norm2 = ttnn.layer_norm

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: ttnn.Tensor,
        causal_attention_mask: ttnn.Tensor,
        output_attentions: Optional[bool] = False,
        parameters=None,
    ) -> Tuple[ttnn.Tensor]:
        residual = hidden_states

        hidden_states = self.layer_norm1(
            hidden_states,
            weight=parameters["layer_norm1"]["weight"],
            bias=parameters["layer_norm1"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            parameters=parameters["self_attn"],
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(
            hidden_states,
            weight=parameters["layer_norm2"]["weight"],
            bias=parameters["layer_norm2"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        hidden_states = self.mlp(hidden_states, parameters=parameters["mlp"])
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

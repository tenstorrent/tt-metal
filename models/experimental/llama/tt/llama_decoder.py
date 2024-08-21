# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
from torch import nn

import ttnn
from loguru import logger
from typing import List, Optional, Tuple, Union
from models.experimental.llama.tt.llama_mlp import TtLlamaMLP
from models.experimental.llama.tt.llama_attention import TtLlamaAttention
from models.experimental.llama.tt.llama_layer_norm import TtLlamaRMSNorm


class TtLlamaDecoderLayer(nn.Module):
    def __init__(self, device, state_dict, base_url, decoder_idx, max_position_embeddings, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.state_dict = state_dict
        self.base_url = base_url
        self.device = device
        self.decoder_idx = decoder_idx
        self.max_position_embeddings = max_position_embeddings

        self.self_attn = TtLlamaAttention(
            self.device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=decoder_idx,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            max_position_embeddings=self.max_position_embeddings,
        )

        self.mlp = TtLlamaMLP(
            self.device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=decoder_idx,
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )

        self.input_layernorm = TtLlamaRMSNorm(
            self.device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=decoder_idx,
            layer_position="input_layernorm",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.post_attention_layernorm = TtLlamaRMSNorm(
            self.device,
            state_dict=self.state_dict,
            base_url=self.base_url,
            layer_num=decoder_idx,
            layer_position="post_attention_layernorm",
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        logger.debug(f"Decoder ID: {self.decoder_idx}")
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_ids=position_ids,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = ttnn.add(residual, hidden_states)

        # Fully Connected
        residual = hidden_states

        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states = self.mlp(hidden_states)

        hidden_states = ttnn.add(residual, hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

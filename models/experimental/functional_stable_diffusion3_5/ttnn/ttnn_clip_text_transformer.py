# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn

from typing import Optional, Union, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPooling
from models.experimental.functional_stable_diffusion3_5.reference.clip_text_embeddings import CLIPTextEmbeddings
from models.experimental.functional_stable_diffusion3_5.reference.clip_encoder import CLIPEncoder
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask

from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_text_embeddings import ttnn_CLIPTextEmbeddings
from models.experimental.functional_stable_diffusion3_5.ttnn.ttnn_clip_encoder import ttnn_CLIPEncoder


class ttnn_CLIPTextTransformer:
    def __init__(self, config, parameters):
        self.config = config
        embed_dim = config.hidden_size
        self.embeddings = ttnn_CLIPTextEmbeddings(config, parameters)
        self.encoder = ttnn_CLIPEncoder(config)
        self.final_layer_norm = ttnn.layer_norm

        # For `pooled_output` computation
        self.eos_token_id = config.eos_token_id

        # For attention mask, it differs between `flash_attention_2` and other attention implementations
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

    def __call__(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        position_ids: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        parameters=None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.shape
        input_ids = ttnn.reshape(input_ids, (-1, input_shape[-1]))

        hidden_states = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, parameters=parameters["embeddings"]
        )

        causal_attention_mask = _create_4d_causal_attention_mask(input_shape, torch.bfloat16, torch.device("cpu"))

        device = hidden_states.device()
        causal_attention_mask = ttnn.from_torch(
            causal_attention_mask, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.L1_MEMORY_CONFIG, device=device
        )
        # expand attention_mask
        if attention_mask is not None and not self._use_flash_attention_2:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            parameters=parameters["encoder"],
        )

        last_hidden_state = encoder_outputs.last_hidden_state
        last_hidden_state = self.final_layer_norm(
            last_hidden_state,
            weight=parameters["final_layer_norm"]["weight"],
            bias=parameters["final_layer_norm"]["bias"],
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )

        last_hidden_state = ttnn.to_torch(last_hidden_state)
        input_ids = ttnn.to_torch(input_ids)

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        last_hidden_state = ttnn.from_torch(last_hidden_state, layout=ttnn.TILE_LAYOUT, device=device)
        input_ids = ttnn.from_torch(input_ids, layout=ttnn.TILE_LAYOUT, device=device)
        pooled_output = ttnn.from_torch(pooled_output, layout=ttnn.TILE_LAYOUT, device=device)

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import math
from loguru import logger
from typing import Optional, Tuple
from dataclasses import dataclass

import ttnn
from tt_lib import fallback_ops

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.trocr.tt.trocr_configuration import TtTrOCRConfig
from models.experimental.trocr.tt.trocr_decoder_layer import TtTrOCRDecoderLayer
from models.experimental.trocr.tt.trocr_embed_tokens import TtTrOCREmbedTokens
from models.experimental.trocr.tt.trocr_learned_positional_embeddings import (
    TtTrOCRLearnedPositionalEmbedding,
)
from models.experimental.trocr.trocr_utils import (
    _make_causal_mask,
    _expand_mask,
    create_custom_forward,
)


@dataclass
class TtBaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: ttnn.Tensor = None
    past_key_values: Optional[Tuple[Tuple[ttnn.Tensor]]] = None
    hidden_states: Optional[Tuple[ttnn.Tensor]] = None
    attentions: Optional[Tuple[ttnn.Tensor]] = None
    cross_attentions: Optional[Tuple[ttnn.Tensor]] = None


class TtTrOCRDecoder(nn.Module):
    def __init__(
        self,
        config: TtTrOCRConfig,
        state_dict=None,
        base_address=None,
        device=None,
        host=None,
    ):
        super().__init__()

        self.host = host
        self.state_dict = state_dict
        self.config = config
        self.device = device
        self.base_address = base_address
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.embed_scale = math.sqrt(config.hidden_size) if config.scale_embedding else 1.0
        self.embed_tokens = TtTrOCREmbedTokens(
            config,
            self.device,
            state_dict=state_dict,
            base_address=f"{base_address}.model.decoder.embed_tokens",
        )

        if config.use_learned_position_embeddings:
            self.embed_positions = TtTrOCRLearnedPositionalEmbedding(
                config.max_position_embeddings,
                config.hidden_size,
                config=config,
                base_address=f"{base_address}.model.decoder.embed_positions",
                state_dict=state_dict,
                device=device,
            )
        else:
            # This is not used/supported yet
            self.embed_positions = TrOCRSinusoidalPositionalEmbedding(
                config.max_position_embeddings + self.padding_idx + 1,
                config.hidden_size,
                self.padding_idx,
            )

        if config.layernorm_embedding:
            self.layernorm_embedding_weight = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.model.decoder.layernorm_embedding.weight"],
                self.device,
                put_on_device=True,
            )

            self.layernorm_embedding_bias = torch_to_tt_tensor_rm(
                state_dict[f"{base_address}.model.decoder.layernorm_embedding.bias"],
                self.device,
                put_on_device=True,
            )

            self.layernorm_embedding = ttnn.layer_norm
        else:
            self.layernorm_embedding = None

        self.layers = nn.ModuleList(
            [
                TtTrOCRDecoderLayer(
                    config,
                    base_address=f"{base_address}.model.decoder.layers.{layer}",
                    host=host,
                    state_dict=state_dict,
                    device=device,
                )
                for layer in range(config.decoder_layers)
            ]
        )

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        dtype = tt_to_torch_tensor(inputs_embeds).dtype
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        cross_attn_head_mask: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[ttnn.Tensor]]] = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> ttnn.Tensor:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input = input_ids
            input_ids = fallback_ops.reshape(input_ids, 1, 1, -1, input.shape.with_tile_padding()[-1])

        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            input = inputs_embeds[:, :, -1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = ttnn.multiply(self.embed_tokens(input_ids), self.embed_scale)

        if self.config.use_learned_position_embeddings:
            embed_pos = self.embed_positions(input, past_key_values_length=past_key_values_length)
            embed_pos = torch_to_tt_tensor_rm(embed_pos, self.device, put_on_device=False)
        else:
            embed_pos = self.embed_positions(input_ids, past_key_values_length=past_key_values_length)

        hidden_states = ttnn.add(inputs_embeds, embed_pos)

        if self.layernorm_embedding is not None:
            hidden_states = self.layernorm_embedding(
                hidden_states,
                epsilon=1e-05,
                weight=self.layernorm_embedding_weight,
                bias=self.layernorm_embedding_bias,
            )

        input_shape = input.shape.with_tile_padding()

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache = True` is incompatible with gradient checkpointing. Setting `use_cache = False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for"
                        f" {head_mask.size()[0]}."
                    )
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.training:
                dropout_probability = torch.rand([])
                if dropout_probability < self.layerdrop:
                    continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return TtBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

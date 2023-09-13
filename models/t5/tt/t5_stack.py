# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import tt_lib
from typing import Optional
from dataclasses import dataclass
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.t5.tt.t5_block import TtT5Block
from models.t5.tt.t5_layer_norm import TtT5LayerNorm


@dataclass
class BaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: Optional[tt_lib.tensor.Tensor] = None
    past_key_values: Optional[tt_lib.tensor.Tensor] = None
    hidden_states: Optional[tt_lib.tensor.Tensor] = None
    attentions: Optional[tt_lib.tensor.Tensor] = None
    cross_attentions: Optional[tt_lib.tensor.Tensor] = None


class TtT5Stack(nn.Module):
    def __init__(self, config, state_dict, base_address, device, embed_tokens=None):
        super().__init__()

        self.out_mem_config_l1 = tt_lib.tensor.MemoryConfig(
            True, tt_lib.tensor.BufferType.L1
        )

        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.config_use_cache = config.use_cache
        self.config_output_attentions = config.output_attentions
        self.config_output_hidden_states = config.output_hidden_states
        self.config_use_return_dict = config.use_return_dict
        self.device = device
        self.block = nn.ModuleList()
        self.main_input_name = "input_ids"

        for i in range(config.num_layers):
            tmp_block = TtT5Block(
                config,
                state_dict,
                f"{base_address}.block.{i}",
                device,
                has_relative_attention_bias=bool(i == 0),
            )
            self.block.append(tmp_block)

        self.final_layer_norm = TtT5LayerNorm(
            config, state_dict, f"{base_address}.final_layer_norm", device
        )

        self.cached_extended_attention_mask = None
        self.cached_encoder_extended_attention_mask = None

        # Model parallel
        self.model_parallel = False
        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def create_extended_attention_mask_for_decoder(self, input_shape, attention_mask):
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length)
        causal_mask = (
            seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
            <= seq_ids[None, :, None]
        )

        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = tt_lib.tensor.concat(
                [
                    tt_lib.tensor.ones(
                        (batch_size, seq_length, prefix_seq_len),
                        dtype=causal_mask.dtype,
                        output_mem_config=self.out_mem_config_l1,
                    ),
                    causal_mask,
                ],
                axis=-1,
                output_mem_config=self.out_mem_config_l1,
            )

        extended_attention_mask = (
            causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        )
        return extended_attention_mask

    def get_extended_attention_mask(self, attention_mask, input_shape):
        if self.cached_extended_attention_mask is not None:
            if (
                input_shape[0] == self.cached_extended_attention_mask.shape[0]
                and input_shape[1] == self.cached_extended_attention_mask.shape[3]
            ):
                return self.cached_extended_attention_mask

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                extended_attention_mask = (
                    self.create_extended_attention_mask_for_decoder(
                        input_shape, attention_mask
                    )
                )
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)
        extended_attention_mask = (
            (1.0 - extended_attention_mask) * torch.finfo(torch.float16).min / 2
        )
        self.cached_extended_attention_mask = extended_attention_mask

        return extended_attention_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"

        head_mask = head_mask.to(dtype=self.dtype)
        return head_mask

    def get_head_mask(
        self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
    ):
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def invert_attention_mask(self, encoder_attention_mask):
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=torch.float16
        )
        encoder_extended_attention_mask = (
            (1.0 - encoder_extended_attention_mask) * torch.finfo(torch.float16).min / 2
        )
        return encoder_extended_attention_mask

    def get_encoder_extended_attention_mask(
        self, encoder_attention_mask, encoder_batch_size, encoder_sequence_length
    ):
        if self.cached_encoder_extended_attention_mask is not None:
            if (
                encoder_batch_size
                == self.cached_encoder_extended_attention_mask.shape[0]
            ):
                if (
                    encoder_sequence_length
                    == self.cached_encoder_extended_attention_mask.shape[3]
                ):
                    return self.cached_encoder_extended_attention_mask

        encoder_extended_attention_mask = self.invert_attention_mask(
            encoder_attention_mask
        )
        self.cached_encoder_extended_attention_mask = encoder_extended_attention_mask

        return encoder_extended_attention_mask

    def forward(
        self,
        input_ids: Optional[tt_lib.tensor.Tensor] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        encoder_hidden_states: Optional[tt_lib.tensor.Tensor] = None,
        encoder_attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        cross_attn_head_mask: Optional[tt_lib.tensor.Tensor] = None,
        past_key_values: Optional[tt_lib.tensor.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[tt_lib.tensor.Tensor] = None,
        output_hidden_states: Optional[tt_lib.tensor.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> tt_lib.tensor.Tensor:
        use_cache = use_cache if use_cache is not None else self.config_use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config_output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config_output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config_use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = (inputs_embeds.shape()[1], inputs_embeds.shape()[2])
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            assert (
                self.embed_tokens is not None
            ), "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

            inputs_embeds = inputs_embeds.unsqueeze(1)
            inputs_embeds = torch_to_tt_tensor_rm(inputs_embeds, self.device)

        batch_size, seq_length = input_shape

        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            assert (
                self.is_decoder
            ), f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length)

        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        if self.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                _,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.shape()
            encoder_extended_attention_mask = self.get_encoder_extended_attention_mask(
                encoder_attention_mask, encoder_batch_size, encoder_sequence_length
            )

        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Copy mask to Torch since get_head_mask is not ported to Tt
        if head_mask is not None:
            head_mask = tt_to_torch_tensor(head_mask)

        if cross_attn_head_mask is not None:
            cross_attn_head_mask = tt_to_torch_tensor(cross_attn_head_mask)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = inputs_embeds

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return tuple(module(*inputs, use_cache, output_attentions))

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            position_bias = layer_outputs[2]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]

            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        hidden_states = self.final_layer_norm(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

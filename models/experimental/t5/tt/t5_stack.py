# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn

from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)
from models.experimental.t5.tt.t5_block import TtT5Block
from models.experimental.t5.tt.t5_layer_norm import TtT5LayerNorm


class BaseModelOutputWithPastAndCrossAttentions:
    """
    Base class for model's outputs that may also contain a past key/values (to speed up sequential decoding).
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
            If `past_key_values` is used only the last hidden-state of the sequences of shape `(batch_size, 1,
            hidden_size)` is output.
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            `config.is_encoder_decoder=True` 2 additional tensors of shape `(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            `config.is_encoder_decoder=True` in the cross-attention blocks) that can be used (see `past_key_values`
            input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
    """

    def __init__(
        self,
        last_hidden_state,
        past_key_values,
        hidden_states,
        attentions,
        cross_attentions,
    ):
        self.last_hidden_state = last_hidden_state  # FloatTensor
        self.past_key_values = past_key_values  # Optional[Tuple[Tuple[FloatTensor]]]
        self.hidden_states = hidden_states  # Optional[Tuple[FloatTensor]]
        self.attentions = attentions  # Optional[Tuple[FloatTensor]]
        self.cross_attentions = cross_attentions  # Optional[Tuple[FloatTensor]]


class TtT5Stack(nn.Module):
    def __init__(self, config, state_dict, base_address, device, embed_tokens=None):
        super().__init__()

        self.config = config
        self.embed_tokens = embed_tokens
        self.is_decoder = config["is_decoder"]
        self.config_use_cache = config["use_cache"] if "use_cache" in config else False
        self.config_output_attentions = config["output_attentions"] if "output_attentions" in config else False
        self.config_output_hidden_states = config["output_hidden_states"] if "output_hidden_states" in config else False
        self.config_use_return_dict = config["use_return_dict"] if "use_return_dict" in config else False
        self.device = device
        self.block = nn.ModuleList()
        self.main_input_name = "input_ids"

        for i in range(config["num_layers"]):
            tmp_block = TtT5Block(
                config,
                state_dict,
                f"{base_address}.block.{i}",
                device,
                has_relative_attention_bias=bool(i == 0),
            )
            self.block.append(tmp_block)

        self.final_layer_norm = TtT5LayerNorm(config, state_dict, f"{base_address}.final_layer_norm", device)
        self.dropout = nn.Dropout(config["dropout_rate"])

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
        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(attention_mask.dtype)

        if causal_mask.shape[1] < attention_mask.shape[1]:
            prefix_seq_len = attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, seq_length, prefix_seq_len),
                        dtype=causal_mask.dtype,
                    ),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(self, attention_mask, input_shape):
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.
        Arguments:
            attention_mask:
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape:
                The shape of the input to the model [batch_size, seq_length].
        Returns:
            The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """

        # if we have cached_extended_attention_mask return it
        if self.cached_extended_attention_mask is not None:
            if (
                input_shape[0] == self.cached_extended_attention_mask.shape[0]
                and input_shape[1] == self.cached_extended_attention_mask.shape[3]
            ):
                return self.cached_extended_attention_mask

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config["is_decoder"]:
                extended_attention_mask = self.create_extended_attention_mask_for_decoder(input_shape, attention_mask)
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        # Check what dtype is self.dtype, attention_mask.dtype?
        # Added "/ 2" bec of Tt device preccision
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(torch.float16).min / 2
        self.cached_extended_attention_mask = extended_attention_mask

        return extended_attention_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"

        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def get_head_mask(self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False):
        """
        Prepare the head mask if needed.
        Args:
            head_mask (Tensor with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.
        Returns:
            Tensor with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).
        Args:
            encoder_attention_mask: An attention mask.
        Returns:
            The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]

        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))

        # dtype fixed to torch.float16 (instead of self.dtype)
        # Added "/ 2" bec of Tt device preccision
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=torch.float16)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(torch.float16).min / 2
        return encoder_extended_attention_mask

    def get_encoder_extended_attention_mask(self, encoder_attention_mask, encoder_batch_size, encoder_sequence_length):
        # Take from cache if we have it
        if self.cached_encoder_extended_attention_mask is not None:
            if encoder_batch_size == self.cached_encoder_extended_attention_mask.shape[0]:
                if encoder_sequence_length == self.cached_encoder_extended_attention_mask.shape[3]:
                    return self.cached_encoder_extended_attention_mask

        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_batch_size, encoder_sequence_length, dtype=torch.long)

        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        self.cached_encoder_extended_attention_mask = encoder_extended_attention_mask

        return encoder_extended_attention_mask

    def forward(
        self,
        input_ids=None,  # Input is pytorch tensor since it is long data
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config_use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config_output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config_output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config_use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = (inputs_embeds.shape.with_tile_padding()[1], inputs_embeds.shape.with_tile_padding()[2])
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

            # Make it broadcastable to num_heads
            inputs_embeds = inputs_embeds.unsqueeze(1)
            inputs_embeds = torch2tt_tensor(inputs_embeds, self.device)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length)

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        # Do not copy for now. Copy later.
        # Copy data to Tt device
        # extended_attention_mask = torch2tt_tensor(extended_attention_mask, self.device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            (
                encoder_batch_size,
                _,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.shape.with_tile_padding()
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
            head_mask = tt2torch_tensor(head_mask)

        if cross_attn_head_mask is not None:
            cross_attn_head_mask = tt2torch_tensor(cross_attn_head_mask)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config["num_layers"])
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config["num_layers"])
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        # hidden_states = self.dropout(inputs_embeds)
        hidden_states = inputs_embeds

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            # Model parallel
            # if self.model_parallel:
            #     # Ensure that attention_mask is always on the same device as hidden_states
            #     if attention_mask is not None:
            #         attention_mask = attention_mask.to(hidden_states.device)
            #     if position_bias is not None:
            #         position_bias = position_bias.to(hidden_states.device)
            #     if encoder_hidden_states is not None:
            #         encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            #     if encoder_extended_attention_mask is not None:
            #         encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            #     if encoder_decoder_position_bias is not None:
            #         encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            #     if layer_head_mask is not None:
            #         layer_head_mask = layer_head_mask.to(hidden_states.device)
            #     if cross_attn_layer_head_mask is not None:
            #         cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)

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
                    None,  # past_key_value is always None with gradient checkpointing
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

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            # if self.model_parallel:
            #     for k, v in self.device_map.items():
            #         if i == v[-1] and "cuda:" + str(k) != self.last_device:
            #             hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)

        # Dropout not supported
        # hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,  # FloatTensor
                    present_key_value_states,  # Optional[Tuple[Tuple[FloatTensor]]]
                    all_hidden_states,  # Optional[Tuple[FloatTensor]]
                    all_attentions,  # Optional[Tuple[FloatTensor]]
                    all_cross_attentions,  # Optional[Tuple[FloatTensor]]
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,  # FloatTensor
            past_key_values=present_key_value_states,  # Optional[Tuple[Tuple[FloatTensor]]]
            hidden_states=all_hidden_states,  # Optional[Tuple[FloatTensor]]
            attentions=all_attentions,  # Optional[Tuple[FloatTensor]]
            cross_attentions=all_cross_attentions,  # Optional[Tuple[FloatTensor]]
        )

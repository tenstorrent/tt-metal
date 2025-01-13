# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import ttnn
from models.experimental.roberta.tt.roberta_encoder import TtRobertaEncoder
from models.experimental.roberta.tt.roberta_pooler import TtRobertaPooler
from models.experimental.roberta.tt.roberta_embeddings import PytorchEmbeddings
from models.utility_functions import (
    tt2torch_tensor,
)
from models.experimental.roberta.roberta_common import torch2tt_tensor


@dataclass
class TtBaseModelOutputWithPoolingAndCrossAttentions:
    last_hidden_state: ttnn.Tensor = None
    pooler_output: ttnn.Tensor = None
    past_key_values: ttnn.Tensor = None
    hidden_states: ttnn.Tensor = None
    attentions: ttnn.Tensor = None
    cross_attentions: ttnn.Tensor = None


class TtRobertaModel(nn.Module):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    # Copied from transformers.models.bert.modeling_bert.BertModel.__init__ with Bert->Roberta
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
        reference_model,
        add_pooling_layer=True,
    ):
        super().__init__()
        self.mem_config = ttnn.L1_MEMORY_CONFIG
        self.config = config
        self.device = device

        self.embeddings = PytorchEmbeddings(reference_model)

        if base_address != "":
            base_address = base_address + "."

        self.encoder = TtRobertaEncoder(
            config,
            state_dict,
            f"{base_address}" + "encoder",
            device,
        )

        self.pooler = (
            TtRobertaPooler(
                config,
                state_dict,
                f"{base_address}" + "pooler",
                device,
            )
            if add_pooling_layer
            else None
        )

        self.dtype = torch.float32
        self.dtype_min_const = torch.finfo(self.dtype).min

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def create_extended_attention_mask_for_decoder(self, input_shape, torch_attention_mask):
        """
        Using torch implementation bc of missing ops.
        This function is only used in decoder mode. For preprocessing attention masks.
        """
        batch_size, seq_length = input_shape
        seq_ids = torch.arange(seq_length)

        causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
        # in case past_key_values are used we need to add a prefix ones mask to the causal mask
        # causal and attention masks must have same type with pytorch version < 1.3
        causal_mask = causal_mask.to(torch_attention_mask.dtype)

        if causal_mask.shape[1] < torch_attention_mask.shape[1]:
            prefix_seq_len = torch_attention_mask.shape[1] - causal_mask.shape[1]
            causal_mask = torch.cat(
                [
                    torch.ones(
                        (batch_size, seq_length, prefix_seq_len),
                        device=device,
                        dtype=causal_mask.dtype,
                    ),
                    causal_mask,
                ],
                axis=-1,
            )

        extended_attention_mask = causal_mask[:, None, :, :] * torch_attention_mask[:, None, None, :]
        return extended_attention_mask

    def get_extended_attention_mask(self, attention_mask: ttnn.Tensor, input_shape: Tuple[int]) -> ttnn.Tensor:
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (`Tuple[int]`):
                The shape of the input to the model.

        Returns:
            `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        torch_attention_mask = tt2torch_tensor(attention_mask).squeeze(0).squeeze(0)

        if len(torch_attention_mask.size()) == 3:
            torch_extended_attention_mask = torch_attention_mask[:, None, :, :]

        elif len(torch_attention_mask.size()) == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                torch_extended_attention_mask = self.create_extended_attention_mask_for_decoder(
                    input_shape, torch_attention_mask
                )
            else:
                torch_extended_attention_mask = torch_attention_mask[:, None, None, :]

        else:
            raise ValueError(
                f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
            )

        extended_attention_mask = torch2tt_tensor(torch_extended_attention_mask, self.device)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and the dtype's smallest value for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        self.ones_const = ttnn.full(extended_attention_mask.shape.with_tile_padding(), 1.0)
        self.mul_const = ttnn.full(extended_attention_mask.shape.with_tile_padding(), self.dtype_min_const)
        extended_attention_mask = ttnn.sub(self.ones_const, extended_attention_mask, memory_config=self.mem_config)

        extended_attention_mask = ttnn.mul(extended_attention_mask, self.mul_const, memory_config=self.mem_config)
        return extended_attention_mask

    def invert_attention_mask(self, encoder_attention_mask: ttnn.Tensor) -> ttnn.Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        torch_encoder_attention_mask = tt2torch_tensor(encoder_attention_mask)

        if len(encoder_attention_mask.shape.with_tile_padding()) == 3:
            torch_encoder_extended_attention_mask = torch_encoder_attention_mask[:, None, :, :]
        if len(encoder_attention_mask.shape.with_tile_padding()) == 2:
            torch_encoder_extended_attention_mask = torch_encoder_attention_mask[:, None, None, :]

        encoder_extended_attention_mask = torch2tt_tensor(torch_encoder_extended_attention_mask, self.device)

        self.ones_const = ttnn.full(encoder_extended_attention_mask.shape.with_tile_padding(), 1.0)
        self.mul_const = ttnn.full(encoder_extended_attention_mask.shape.with_tile_padding(), self.dtype_min_const)

        encoder_extended_attention_mask = ttnn.sub(
            self.ones_const,
            encoder_extended_attention_mask,
            output_mem_config=self.mem_config,
        ).to(self.device)
        encoder_extended_attention_mask = ttnn.mul(
            encoder_extended_attention_mask,
            self.mul_const,
            memory_config=self.mem_config,
        )

        return encoder_extended_attention_mask

    def _convert_head_mask_to_5d(self, head_mask, num_hidden_layers):
        """Implemented in PyTorch for now"""
        """-> [num_hidden_layers x batch x num_heads x seq_length x seq_length]"""
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        assert head_mask.dim() == 5, f"head_mask.dim != 5, instead {head_mask.dim()}"
        head_mask = head_mask.to(dtype=self.dtype)  # switch to float if need + fp16 compatibility
        return head_mask

    def get_head_mask(
        self,
        head_mask: Optional[ttnn.Tensor],
        num_hidden_layers: int,
        is_attention_chunked: bool = False,
    ) -> ttnn.Tensor:
        """
        Prepare the head mask if needed.

        Args:
            head_mask (`torch.Tensor` with shape `[num_heads]` or `[num_hidden_layers x num_heads]`, *optional*):
                The mask indicating if we should keep the heads or not (1.0 for keep, 0.0 for discard).
            num_hidden_layers (`int`):
                The number of hidden layers in the model.
            is_attention_chunked: (`bool`, *optional*, defaults to `False`):
                Whether or not the attentions scores are computed by chunks or not.

        Returns:
            `torch.Tensor` with shape `[num_hidden_layers x batch x num_heads x seq_length x seq_length]` or list with
            `[None]` for each layer.
        """
        if head_mask is not None:
            torch_head_mask = tt2torch_tensor(head_mask)
        else:
            torch_head_mask = None

        if torch_head_mask is not None:
            torch_head_mask = self._convert_head_mask_to_5d(torch_head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                torch_head_mask = torch_head_mask.unsqueeze(-1)
            head_mask = torch2tt_tensor(torch_head_mask, self.device)
        else:
            head_mask = [
                None,
            ] * num_hidden_layers

        return head_mask

    """
    TODO: Implement method if needed. Not used for now for this config roberta-base.
    """
    # def _prune_heads(self, heads_to_prune):
    #     """
    #     Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
    #     class PreTrainedModel
    #     """
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)

    # Copied from transformers.models.bert.modeling_bert.BertModel.forward

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        encoder_hidden_states: Optional[ttnn.Tensor] = None,
        encoder_attention_mask: Optional[ttnn.Tensor] = None,
        past_key_values: Optional[List[ttnn.Tensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[ttnn.Tensor], TtBaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = list(input_ids.size())
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.shape.with_tile_padding()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        while len(input_shape) < 4:
            input_shape.insert(0, 1)
        _, _, batch_size, seq_length = input_shape

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape.with_tile_padding()[2] if past_key_values is not None else 0
        )

        if attention_mask is None:
            attention_mask = ttnn.full((1, 1, batch_size, seq_length + past_key_values_length), 0.0)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            (
                _,
                encoder_batch_size,
                encoder_sequence_length,
                _,
            ) = encoder_hidden_states.shape.with_tile_padding()
            encoder_hidden_shape = (1, 1, encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = ttnn.full(encoder_hidden_shape, 1.1)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        torch_embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=tt2torch_tensor(inputs_embeds) if inputs_embeds is not None else None,
            past_key_values_length=past_key_values_length,
        )
        torch_embedding_output = torch_embedding_output.squeeze(0)
        embedding_output = torch2tt_tensor(torch_embedding_output, self.device)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return TtBaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )

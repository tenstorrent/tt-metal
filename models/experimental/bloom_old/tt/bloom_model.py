# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch.nn import functional as F

import models.experimental.bloom_old.bloom_utils as bloom_utils
import models.experimental.bloom_old.tt.bloom_block as bloom_block

from fused_ops.layernorm import Layernorm as TtLayernorm
from typing import Optional, Tuple, Union


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    """
    Make causal mask used for self-attention.
    """
    batch_size, target_length = input_ids_shape
    mask = torch.empty(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )
    # ONNX doesn't support `torch.Tensor.triu` properly, thus we use this workaround
    seq_ids = torch.arange(target_length, device=device)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(batch_size, 1, target_length, target_length + past_key_values_length)
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    """
    Expands attention_mask from `[batch_size, src_length]` to `[batch_size, 1, tgt_length, src_length]`.
    """
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def build_alibi_tensor(attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
    """
    Link to paper: https://arxiv.org/abs/2108.12409 Alibi tensor is not causal as the original paper mentions, it
    relies on a translation invariance of softmax for quick implementation: with l being a tensor, and a fixed value
    `softmax(l+a) = softmax(l)`. Based on
    https://github.com/ofirpress/attention_with_linear_biases/blob/a35aaca144e0eb6b789dfcb46784c4b8e31b7983/fairseq/models/transformer.py#L742
    TODO @thomasw21 this doesn't work as nicely due to the masking strategy, and so masking varies slightly.
    Args:
    Returns tensor shaped (batch_size * num_heads, 1, max_seq_len)
        attention_mask (`torch.Tensor`):
            Token-wise attention mask, this should be of shape (batch_size, max_seq_len).
        num_heads (`int`, *required*):
            number of heads
        dtype (`torch.dtype`, *optional*, default=`torch.bfloat16`):
            dtype of the output tensor
    """
    batch_size, seq_length = attention_mask.shape
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        device=attention_mask.device,
        dtype=torch.float32,
    )

    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            device=attention_mask.device,
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
            device=attention_mask.device,
            dtype=torch.int32,
        )
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    # Note: alibi will added to the attention bias that will be applied to the query, key product of attention
    # => therefore alibi will have to be of shape (batch_size, num_heads, query_length, key_length)
    # => here we set (batch_size=1, num_heads=num_heads, query_length=1, key_length=max_length)
    # => the query_length dimension will then be broadcasted correctly
    # This is more or less identical to T5's relative position bias:
    # https://github.com/huggingface/transformers/blob/f681437203baa7671de3174b0fa583c349d9d5e1/src/transformers/models/t5/modeling_t5.py#L527
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    alibi = alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)
    alibi = alibi.repeat(1, seq_length, 1)

    return alibi


# class BloomModel(BloomPreTrainedModel):
#     def __init__(self, config: BloomConfig):
#         super().__init__(config)

#         self.embed_dim = config.hidden_size
#         self.num_heads = config.n_head

#         # Embedding + LN Embedding
#         self.word_embeddings = nn.Embedding(config.vocab_size, self.embed_dim)
#         self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

#         # Transformer blocks
#         self.h = nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])

#         # Final Layer Norm
#         self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

#         self.gradient_checkpointing = False

#         # Initialize weights and apply final processing
#         self.post_init()

#     def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
#         return build_alibi_tensor(attention_mask, num_heads, dtype)

#     def get_input_embeddings(self):
#         return self.word_embeddings

#     def _prepare_attn_mask(
#         self, attention_mask: torch.Tensor, input_shape: Tuple[int, int], past_key_values_length: int
#     ) -> torch.BoolTensor:
#         # create causal mask
#         # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
#         combined_attention_mask = None
#         device = attention_mask.device
#         _, src_length = input_shape

#         if src_length > 1:
#             combined_attention_mask = _make_causal_mask(
#                 input_shape, device=device, past_key_values_length=past_key_values_length
#             )

#         # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
#         expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
#         combined_attention_mask = (
#             expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
#         )

#         return combined_attention_mask

#     def set_input_embeddings(self, new_embeddings: torch.Tensor):
#         self.word_embeddings = new_embeddings

#     @add_start_docstrings_to_model_forward(BLOOM_INPUTS_DOCSTRING)
#     @add_code_sample_docstrings(
#         checkpoint=_CHECKPOINT_FOR_DOC,
#         output_type=BaseModelOutputWithPastAndCrossAttentions,
#         config_class=_CONFIG_FOR_DOC,
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.LongTensor] = None,
#         inputs_embeds: Optional[torch.LongTensor] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#         **deprecated_arguments,
#     ) -> Union[Tuple[torch.Tensor, ...], BaseModelOutputWithPastAndCrossAttentions]:
#         if deprecated_arguments.pop("position_ids", False) is not False:
#             # `position_ids` could have been `torch.Tensor` or `None` so defaulting pop to `False` allows to detect if users were passing explicitly `None`
#             warnings.warn(
#                 "`position_ids` have no functionality in BLOOM and will be removed in v5.0.0. You can safely ignore"
#                 " passing `position_ids`.",
#                 FutureWarning,
#             )
#         if len(deprecated_arguments) > 0:
#             raise ValueError(f"Got unexpected arguments: {deprecated_arguments}")

#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         use_cache = use_cache if use_cache is not None else self.config.use_cache
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             batch_size, seq_length = input_ids.shape
#         elif inputs_embeds is not None:
#             batch_size, seq_length, _ = inputs_embeds.shape
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         if past_key_values is None:
#             past_key_values = tuple([None] * len(self.h))

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape batch_size x num_heads x N x N
#         # head_mask has shape n_layer x batch x num_heads x N x N
#         head_mask = self.get_head_mask(head_mask, self.config.n_layer)

#         if inputs_embeds is None:
#             inputs_embeds = self.word_embeddings(input_ids)

#         hidden_states = self.word_embeddings_layernorm(inputs_embeds)

#         presents = () if use_cache else None
#         all_self_attentions = () if output_attentions else None
#         all_hidden_states = () if output_hidden_states else None

#         if self.gradient_checkpointing and self.training:
#             if use_cache:
#                 logger.warning_once(
#                     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
#                 )
#                 use_cache = False

#         # Compute alibi tensor: check build_alibi_tensor documentation
#         seq_length_with_past = seq_length
#         past_key_values_length = 0
#         if past_key_values[0] is not None:
#             past_key_values_length = past_key_values[0][0].shape[2]
#             seq_length_with_past = seq_length_with_past + past_key_values_length
#         if attention_mask is None:
#             attention_mask = torch.ones((batch_size, seq_length_with_past), device=hidden_states.device)
#         else:
#             attention_mask = attention_mask.to(hidden_states.device)

#         alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

#         causal_mask = self._prepare_attn_mask(
#             attention_mask,
#             input_shape=(batch_size, seq_length),
#             past_key_values_length=past_key_values_length,
#         )

#         for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             if self.gradient_checkpointing and self.training:

#                 def create_custom_forward(module):
#                     def custom_forward(*inputs):
#                         # None for past_key_value
#                         return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

#                     return custom_forward

#                 outputs = torch.utils.checkpoint.checkpoint(
#                     create_custom_forward(block),
#                     hidden_states,
#                     alibi,
#                     causal_mask,
#                     layer_past,
#                     head_mask[i],
#                 )
#             else:
#                 outputs = block(
#                     hidden_states,
#                     layer_past=layer_past,
#                     attention_mask=causal_mask,
#                     head_mask=head_mask[i],
#                     use_cache=use_cache,
#                     output_attentions=output_attentions,
#                     alibi=alibi,
#                 )

#             hidden_states = outputs[0]
#             if use_cache is True:
#                 presents = presents + (outputs[1],)

#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

#         # Add last hidden state
#         hidden_states = self.ln_f(hidden_states)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

#         return BaseModelOutputWithPastAndCrossAttentions(
#             last_hidden_state=hidden_states,
#             past_key_values=presents,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         )


class TtBloomModel(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.n_layer = config.num_hidden_layers

        # Embedding + LN Embedding
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings.weight = torch.nn.Parameter(state_dict[f"{base_address}.word_embeddings.weight"])

        self.word_embeddings_layernorm_bias = bloom_utils.tt_load_layer_weights(
            f"{base_address}.word_embeddings_layernorm.bias", state_dict
        )
        self.word_embeddings_layernorm_weight = bloom_utils.tt_load_layer_weights(
            f"{base_address}.word_embeddings_layernorm.weight", state_dict
        )

        self.word_embeddings_layernorm = TtLayernorm(
            self.word_embeddings_layernorm_weight,
            self.word_embeddings_layernorm_bias,
            config.layer_norm_epsilon,
            config.hidden_size,
            config.hidden_size,
            device,
            1,
        )

        # Transformer blocks
        blocks = []

        for i in range(self.n_layer):
            block = bloom_block.TtBloomBlock(config, state_dict, f"{base_address}.h.{i}", device)
            blocks.append(block)

        self.h = torch.nn.ModuleList(blocks)

        self.ln_f_bias = bloom_utils.tt_load_layer_weights(f"{base_address}.ln_f.bias", state_dict)
        self.ln_f_weight = bloom_utils.tt_load_layer_weights(f"{base_address}.ln_f.weight", state_dict)

        # Final Layer Norm
        self.ln_f = TtLayernorm(
            self.ln_f_weight,
            self.ln_f_bias,
            config.layer_norm_epsilon,
            config.hidden_size,
            config.hidden_size,
            device,
            1,
        )

    def build_alibi_tensor(self, attention_mask: torch.Tensor, num_heads: int, dtype: torch.dtype) -> torch.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype)

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        # create causal mask
        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        # [batch_size, seq_length] -> [batch_size, 1, tgt_length, src_length]
        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

    def set_input_embeddings(self, new_embeddings: torch.Tensor):
        self.word_embeddings = new_embeddings

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

    def forward(
        self,
        device,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape batch_size x num_heads x N x N
        # head_mask has shape n_layer x batch x num_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        inputs_embeds = bloom_utils.torch2tt_tensor(inputs_embeds, device)

        hidden_states = self.word_embeddings_layernorm(inputs_embeds)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        # Compute alibi tensor: check build_alibi_tensor documentation
        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None and past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length_with_past))

        alibi = self.build_alibi_tensor(attention_mask, self.num_heads, dtype=torch.float32)
        alibi = bloom_utils.torch2tt_tensor(alibi, device)

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        print(f"Num blocks {self.n_layer}")
        i = 0

        for block in self.h:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            print(f"Running block {i}")

            # if self.gradient_checkpointing and self.training:

            #     def create_custom_forward(module):
            #         def custom_forward(*inputs):
            #             # None for past_key_value
            #             return module(*inputs, use_cache=use_cache, output_attentions=output_attentions)

            #         return custom_forward

            #     outputs = torch.utils.checkpoint.checkpoint(
            #         create_custom_forward(block),
            #         hidden_states,
            #         alibi,
            #         causal_mask,
            #         layer_past,
            #         head_mask[i],
            #     )
            # else:
            outputs = block(
                device,
                hidden_states,
                layer_past=None,  # layer_past,
                attention_mask=causal_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
                alibi=alibi,
            )

            hidden_states = outputs[0]

            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)

            i = i + 1

        # Add last hidden state
        hidden_states = self.ln_f(hidden_states, overrideH=hidden_states.get_legacy_shape()[-2])

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    presents,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

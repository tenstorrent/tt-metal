"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import torch
import torch.nn as nn
import math

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)

from models.bloom.tt.bloom_block import TtBloomBlock
import tt_lib
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class TtBaseModelOutputWithPastAndCrossAttentions:
    last_hidden_state: tt_lib.tensor.Tensor = None
    past_key_values: Optional[Tuple[Tuple[tt_lib.tensor.Tensor]]] = None
    hidden_states: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None
    cross_attentions: Optional[Tuple[tt_lib.tensor.Tensor]] = None


def _make_causal_mask(
    input_ids_shape: torch.Size, device: torch.device, past_key_values_length: int
) -> torch.BoolTensor:
    batch_size, target_length = input_ids_shape
    mask = torch.empty(
        (target_length, target_length + past_key_values_length),
        dtype=torch.bool,
        device=device,
    )

    seq_ids = tt_lib.tensor.arange(0, target_length, 1)
    seq_ids = tt_to_torch_tensor(seq_ids).squeeze(0).squeeze(0).squeeze(0)
    mask[:, past_key_values_length:] = seq_ids[:, None] < seq_ids[None, :]

    if past_key_values_length > 0:
        mask[:, :past_key_values_length] = False

    expanded_mask = mask[None, None, :, :].expand(
        batch_size, 1, target_length, target_length + past_key_values_length
    )
    return expanded_mask


def _expand_mask(mask: torch.Tensor, tgt_length: int) -> torch.BoolTensor:
    batch_size, src_length = mask.shape
    tgt_length = tgt_length if tgt_length is not None else src_length

    expanded_mask = ~(mask[:, None, None, :].to(torch.bool))
    return expanded_mask.expand(batch_size, 1, tgt_length, src_length)


def build_alibi_tensor(
    attention_mask: tt_lib.tensor.Tensor, num_heads: int, dtype: torch.dtype, device
) -> tt_lib.tensor.Tensor:
    _, _, batch_size, seq_length = attention_mask.shape()
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))),
        dtype=torch.float32,
    )

    powers = tt_lib.tensor.arange(1, 1 + closest_power_of_2, 1)
    powers = tt_to_torch_tensor(powers)
    powers = powers.type(torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))),
            dtype=torch.float32,
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = tt_lib.tensor.arange(
            1,
            1 + 2 * num_remaining_heads,
            2,
        )
        extra_powers = tt_to_torch_tensor(extra_powers).type(torch.int32)
        temp = torch.pow(extra_base, extra_powers)
        temp = torch_to_tt_tensor_rm(temp, device)
        slopes = torch_to_tt_tensor_rm(slopes, device)
        slopes = tt_lib.tensor.concat(slopes, temp, dim=0)

    attention_mask = tt_to_torch_tensor(attention_mask).squeeze(0).squeeze(0)
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    alibi = torch_to_tt_tensor_rm(alibi.squeeze(0), device)
    alibi = tt_lib.tensor.reshape(alibi, 1, batch_size * num_heads, 1, seq_length)
    alibi = tt_to_torch_tensor(alibi).squeeze(0)
    alibi = alibi.repeat(1, seq_length, 1)
    alibi = torch_to_tt_tensor_rm(alibi, device, put_on_device=True)

    return alibi


class TtBloomModel(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.config = config
        self.state_dict = state_dict
        self.base_address = base_address
        self.device = device

        self.embed_dim = config.hidden_size
        self.num_heads = config.n_head
        self.n_layer = config.num_hidden_layers

        self.word_embedding_weight = state_dict[
            f"{base_address}.word_embeddings.weight"
        ]

        self.word_embeddings = nn.Embedding(
            config.vocab_size,
            self.embed_dim,
            _weight=self.word_embedding_weight,
        )

        self.word_embeddings_layernorm_gamma = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.word_embeddings_layernorm.weight"], self.device
        )
        self.word_embeddings_layernorm_beta = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.word_embeddings_layernorm.bias"], self.device
        )

        self.word_embeddings_layernorm = tt_lib.tensor.layernorm

        self.h = nn.ModuleList(
            [
                TtBloomBlock(
                    config,
                    state_dict=state_dict,
                    base_address=f"{base_address}.h.{i}",
                    device=device,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.ln_f_gamma = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.ln_f.weight"], self.device
        )
        self.ln_f_beta = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.ln_f.bias"], self.device
        )

        self.ln_f = tt_lib.tensor.layernorm

    def build_alibi_tensor(
        self,
        attention_mask: tt_lib.tensor.Tensor,
        num_heads: int,
        dtype: torch.dtype,
        device,
    ) -> tt_lib.tensor.Tensor:
        return build_alibi_tensor(attention_mask, num_heads, dtype, device)

    def get_input_embeddings(self):
        return self.word_embeddings

    def _prepare_attn_mask(
        self,
        attention_mask: tt_lib.tensor.Tensor,
        input_shape: Tuple[int, int],
        past_key_values_length: int,
    ) -> torch.BoolTensor:
        attention_mask = tt_to_torch_tensor(attention_mask).squeeze(0).squeeze(0)
        combined_attention_mask = None
        device = attention_mask.device
        _, src_length = input_shape

        if src_length > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                device=device,
                past_key_values_length=past_key_values_length,
            )

        expanded_attn_mask = _expand_mask(attention_mask, tgt_length=src_length)
        combined_attention_mask = (
            expanded_attn_mask
            if combined_attention_mask is None
            else expanded_attn_mask | combined_attention_mask
        )

        return combined_attention_mask

        def set_input_embeddings(self, new_embeddings: torch.Tensor):
            self.word_embeddings = new_embeddings

    def get_head_mask(
        self, head_mask, num_hidden_layers: int, is_attention_chunked: bool = False
    ) -> torch.tensor:
        if head_mask is not None:
            head_mask = self._convert_head_mask_to_5d(head_mask, num_hidden_layers)
            if is_attention_chunked is True:
                head_mask = head_mask.unsqueeze(-1)
        else:
            head_mask = [None] * num_hidden_layers

        return head_mask

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            Tuple[Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor], ...]
        ] = None,
        attention_mask: Optional[tt_lib.tensor.Tensor] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        inputs_embeds: Optional[tt_lib.tensor.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **deprecated_arguments,
    ) -> Union[
        Tuple[tt_lib.tensor.Tensor, ...], TtBaseModelOutputWithPastAndCrossAttentions
    ]:
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            _, batch_size, seq_length, _ = inputs_embeds.shape()
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if head_mask is not None:
            head_mask = tt_to_torch_tensor(head_mask)
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        inputs_embeds = torch_to_tt_tensor_rm(
            inputs_embeds, self.device, put_on_device=True
        )

        hidden_states = self.word_embeddings_layernorm(
            inputs_embeds,
            eps=self.config.layer_norm_epsilon,
            gamma=self.word_embeddings_layernorm_gamma,
            beta=self.word_embeddings_layernorm_beta,
        )

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None and past_key_values[0] is not None:
            past_key_values_length = past_key_values[0][0].shape()[-1]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if attention_mask is None:
            attention_mask = tt_lib.tensor.ones(
                ([1, 1, batch_size, seq_length_with_past])
            )

        alibi = self.build_alibi_tensor(
            attention_mask, self.num_heads, dtype=torch.float32, device=self.device
        )

        causal_mask = self._prepare_attn_mask(
            attention_mask,
            input_shape=(batch_size, seq_length),
            past_key_values_length=past_key_values_length,
        )

        causal_mask = causal_mask.type(torch.int32)
        causal_mask = torch_to_tt_tensor_rm(
            causal_mask, self.device, put_on_device=True
        )
        i = 0

        for block in self.h:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                layer_past=None,
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
                all_self_attentions = all_self_attentions + (
                    outputs[2 if use_cache else 1],
                )

            i = i + 1

        hidden_states = self.ln_f(
            hidden_states,
            eps=self.config.layer_norm_epsilon,
            gamma=self.ln_f_gamma,
            beta=self.ln_f_beta,
        )

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

        return TtBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

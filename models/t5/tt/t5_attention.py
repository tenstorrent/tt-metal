# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
from torch import nn
import tt_lib
from tt_lib import fallback_ops
from typing import Optional
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)
from models.helper_funcs import Linear as TtLinear


def t5_shape_tt(
    states: tt_lib.tensor.Tensor,
    batch_size: int,
    n_heads: int,
    key_value_proj_dim: int,
    device,
) -> tt_lib.tensor.Tensor:
    # Layout of states is Layout.TILE
    state = tt_to_torch_tensor(states)
    states.deallocate()
    states = torch_to_tt_tensor_rm(state, device, put_on_device=True)
    tt_out = tt_lib.tensor.reshape(states, batch_size, -1, n_heads, key_value_proj_dim)
    tt_out = tt_lib.tensor.transpose_hc(tt_out)
    return tt_out


def t5_unshape_tt(
    states: tt_lib.tensor.Tensor, batch_size: int, inner_dim: int, device
):
    state = tt_to_torch_tensor(states)
    states.deallocate()
    states = torch_to_tt_tensor_rm(state, device, put_on_device=True)
    states = tt_lib.tensor.transpose_hc(states)
    tt_out = tt_lib.tensor.reshape(states, 1, batch_size, -1, inner_dim)
    return tt_out


class TtT5Attention(nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
        has_relative_attention_bias=False,
    ):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.relative_attention_max_distance = config.relative_attention_max_distance
        self.d_model = config.d_model
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_heads
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.device = device
        self.out_mem_config_l1 = tt_lib.tensor.MemoryConfig(
            True, tt_lib.tensor.BufferType.L1
        )

        self.q_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.q.weight"], self.device, put_on_device=True
        )
        self.query = TtLinear(
            self.q_weights.shape()[-1],
            self.q_weights.shape()[-2],
            self.q_weights,
            output_mem_config=self.out_mem_config_l1,
        )

        self.k_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.k.weight"], self.device, put_on_device=True
        )
        self.key = TtLinear(
            self.k_weights.shape()[-1],
            self.k_weights.shape()[-2],
            self.k_weights,
            output_mem_config=self.out_mem_config_l1,
        )

        self.v_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.v.weight"], self.device, put_on_device=True
        )
        self.value = TtLinear(
            self.v_weights.shape()[-1],
            self.v_weights.shape()[-2],
            self.v_weights,
            output_mem_config=self.out_mem_config_l1,
        )

        self.o_weights = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.o.weight"], self.device, put_on_device=True
        )

        self.output = TtLinear(
            self.o_weights.shape()[-1],
            self.o_weights.shape()[-2],
            self.o_weights,
            output_mem_config=self.out_mem_config_l1,
        )

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(
                self.relative_attention_num_buckets, self.n_heads
            )
            self.relative_attention_bias.weight = nn.Parameter(
                state_dict[f"{base_address}.relative_attention_bias.weight"]
            )

        self.cached_position_bias = None
        self.cached_real_seq_length = None
        self.cached_key_length = None

        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(
        relative_position,
        bidirectional=True,
        num_buckets=32,
        max_distance=128,
        device=None,
    ):
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch_to_tt_tensor_rm(
                relative_position, device, put_on_device=True
            )
            relative_position = tt_lib.tensor.abs(relative_position)
            relative_position = (
                tt_to_torch_tensor(relative_position).squeeze(0).squeeze(0).long()
            )
        else:
            relative_position = -torch.min(
                relative_position, torch.zeros_like(relative_position)
            )

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(
            is_small, relative_position, relative_position_if_large
        )
        return relative_buckets

    def compute_bias_const(self, query_length, key_length):
        context_position = tt_lib.tensor.arange(0, query_length, 1).to(self.device)
        context_position = tt_lib.tensor.permute(context_position, 0, 1, 3, 2)
        memory_position = tt_lib.tensor.arange(0, key_length, 1).to(self.device)
        memory_position = (
            tt_to_torch_tensor(memory_position).squeeze(0).squeeze(0).long()
        )
        context_position = (
            tt_to_torch_tensor(context_position).squeeze(0).squeeze(0).long()
        )

        # cannot broadcast tensor of size [1, 1, 1, 32] and [1, 1, 32, 1]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(
            relative_position,
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
            device=self.device,
        )
        values = self.relative_attention_bias(relative_position_bucket)
        values = torch_to_tt_tensor_rm(values, self.device, put_on_device=True)
        value = tt_lib.tensor.permute(values, 0, 3, 1, 2)
        values = tt_to_torch_tensor(value)
        return values

    def forward(
        self,
        hidden_states: Optional[tt_lib.tensor.Tensor],
        mask: Optional[tt_lib.tensor.Tensor] = None,
        key_value_states: Optional[tt_lib.tensor.Tensor] = None,
        position_bias: Optional[tt_lib.tensor.Tensor] = None,
        past_key_value: Optional[tt_lib.tensor.Tensor] = None,
        layer_head_mask: Optional[tt_lib.tensor.Tensor] = None,
        query_length: Optional[int] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> tt_lib.tensor.Tensor:
        batch_size = hidden_states.shape()[1]
        seq_length = hidden_states.shape()[2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += (
                past_key_value[0].shape[2] if query_length is None else query_length
            )

        key_length = (
            real_seq_length if key_value_states is None else key_value_states.shape()[2]
        )

        def shape(states):
            return t5_shape_tt(
                states, batch_size, self.n_heads, self.key_value_proj_dim, self.device
            )

        def unshape(states):
            return t5_unshape_tt(states, batch_size, self.inner_dim, self.device)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            if key_value_states is None:
                hidden_states = shape(proj_layer(hidden_states.to(self.device)))

            elif past_key_value is None:
                hidden_states = shape(proj_layer(key_value_states.to(self.device)))

            if past_key_value is not None:
                if key_value_states is None:
                    hidden_states = tt_lib.tensor.concat(
                        [past_key_value, hidden_states], dim=2
                    )
                elif past_key_value.shape[2] != key_value_states.shape()[2]:
                    hidden_states = shape(proj_layer(key_value_states.to(self.device)))
                else:
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        layer = self.query(hidden_states.to(self.device))
        query_states = shape(layer)

        # get key/value states
        key_states = project(
            hidden_states,
            self.key,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )

        value_states = project(
            hidden_states,
            self.value,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        transposed_key_states = tt_lib.tensor.transpose(key_states)

        scores = tt_lib.tensor.bmm(
            query_states,
            transposed_key_states,
            output_mem_config=self.out_mem_config_l1,
        )

        if (
            position_bias is None
            and self.cached_real_seq_length == real_seq_length
            and self.cached_key_length == key_length
        ):
            position_bias = self.cached_position_bias

        elif position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = tt_lib.tensor.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                    output_mem_config=self.out_mem_config_l1,
                )

                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias_const(real_seq_length, key_length)
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.shape()[2] :, :]

            position_bias = fallback_ops.repeat(
                position_bias,
                ((batch_size, 1, 1, 1)),
            )
            position_bias = tt_to_torch_tensor(position_bias)

            if mask is not None:
                position_bias = torch_to_tt_tensor_rm(
                    position_bias, self.device, put_on_device=True
                )
                mask = torch_to_tt_tensor_rm(mask, self.device, put_on_device=True)

                if position_bias.shape()[-2] == mask.shape()[-2]:
                    position_bias = tt_lib.tensor.permute(
                        position_bias,
                        0,
                        3,
                        1,
                        2,
                        output_mem_config=self.out_mem_config_l1,
                    )
                    mask = tt_lib.tensor.permute(
                        mask, 0, 3, 1, 2, output_mem_config=self.out_mem_config_l1
                    )
                    position_bias = tt_lib.tensor.bcast(
                        position_bias,
                        mask,
                        tt_lib.tensor.BcastOpMath.ADD,
                        tt_lib.tensor.BcastOpDim.H,
                        output_mem_config=self.out_mem_config_l1,
                    )
                else:
                    position_bias = tt_lib.tensor.permute(
                        position_bias,
                        0,
                        3,
                        1,
                        2,
                        output_mem_config=self.out_mem_config_l1,
                    )

                    mask = tt_lib.tensor.permute(
                        mask,
                        0,
                        3,
                        1,
                        2,
                        output_mem_config=self.out_mem_config_l1,
                    )

                    position_bias = tt_lib.tensor.bcast(
                        position_bias,
                        mask,
                        tt_lib.tensor.BcastOpMath.ADD,
                        tt_lib.tensor.BcastOpDim.HW,
                        output_mem_config=self.out_mem_config_l1,
                    )

                position_bias = tt_lib.tensor.permute(
                    position_bias,
                    0,
                    2,
                    3,
                    1,
                    output_mem_config=self.out_mem_config_l1,
                )

            else:
                position_bias = torch_to_tt_tensor_rm(
                    position_bias, self.device, put_on_device=True
                )

            if self.pruned_heads:
                mask = torch.ones(position_bias.shape()[2])
                mask[list(self.pruned_heads)] = 0
                position_bias = position_bias[:, mask.bool()]

            self.cached_position_bias = position_bias
            self.cached_real_seq_length = real_seq_length
            self.cached_key_length = key_length

        scores = tt_lib.tensor.add(scores, position_bias)

        attn_weights = tt_lib.operations.primary.softmax_in_place(scores)

        if layer_head_mask is not None:
            attn_weights = tt_lib.tensor.mul(
                attn_weights, layer_head_mask, output_mem_config=self.out_mem_config_l1
            )

        attn_output = tt_lib.tensor.bmm(
            attn_weights, value_states, output_mem_config=self.out_mem_config_l1
        )

        attn_output = unshape(attn_output)
        attn_output = self.output(attn_output)

        present_key_value_state = (
            (key_states, value_states) if (self.is_decoder and use_cache) else None
        )
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs

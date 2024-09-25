# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import math
import torch
from torch import nn

import ttnn

from loguru import logger
from models.utility_functions import (
    torch2tt_tensor,
    tt2torch_tensor,
)


def t5_shape_tt(states, batch_size, n_heads, key_value_proj_dim, device):
    """projection"""
    fall_back_to_torch = True

    if fall_back_to_torch:
        states = tt2torch_tensor(states)
        states = torch.reshape(states, (batch_size, -1, n_heads, key_value_proj_dim))
        states = states.transpose(1, 2)
        tt_out = torch2tt_tensor(states, device)
    else:
        tt_out = ttnn.reshape_on_device(states, batch_size, -1, n_heads, key_value_proj_dim)
        tt_out = ttnn.transpose(tt_out, 1, -2)

    return tt_out


def t5_shape_pt(states, batch_size, n_heads, key_value_proj_dim):
    """
    projection
    batch_size eg. 32
    n_heads eg. 8
    key_value_proj_dim eg 64
    """
    pt_out = states.view(batch_size, -1, n_heads, key_value_proj_dim)
    return pt_out.transpose(1, 2)


def t5_unshape_pt(states, batch_size, inner_dim):
    return states.transpose(1, 2).contiguous().view(1, batch_size, -1, inner_dim)


def t5_unshape_tt(states, batch_size, inner_dim, device):
    # Leave as fallback due to perf
    fall_back_to_torch = True

    if fall_back_to_torch:
        states = tt2torch_tensor(states)
        states = t5_unshape_pt(states, batch_size, inner_dim)
        tt_out = torch2tt_tensor(states, device)
    else:
        states = ttnn.transpose(states, 1, -2)
        tt_out = ttnn.reshape_on_device(states, 1, batch_size, -1, inner_dim)

    return tt_out


# class T5Attention(nn.Module):
#     def __init__(self, config, hf_reference_module, has_relative_attention_bias=False):
#         super().__init__()
#         self.is_decoder = config["is_decoder"]
#         self.has_relative_attention_bias = has_relative_attention_bias
#         self.relative_attention_num_buckets = config["relative_attention_num_buckets"]
#         self.relative_attention_max_distance = config["relative_attention_max_distance"]
#         self.d_model = config["d_model"]
#         self.key_value_proj_dim = config["d_kv"]
#         self.n_heads = config["num_heads"]
#         self.dropout = config["dropout_rate"]
#         self.inner_dim = self.n_heads * self.key_value_proj_dim

#         self.q = hf_reference_module.q
#         self.k = hf_reference_module.k
#         self.v = hf_reference_module.v
#         self.o = hf_reference_module.o

#         if self.has_relative_attention_bias:
#             self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
#         self.pruned_heads = set()
#         self.gradient_checkpointing = False

#     def prune_heads(self, heads):
#         if len(heads) == 0:
#             return
#         heads, index = find_pruneable_heads_and_indices(
#             heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
#         )
#         # Prune linear layers
#         self.q = prune_linear_layer(self.q, index)
#         self.k = prune_linear_layer(self.k, index)
#         self.v = prune_linear_layer(self.v, index)
#         self.o = prune_linear_layer(self.o, index, dim=1)
#         # Update hyper params
#         self.n_heads = self.n_heads - len(heads)
#         self.inner_dim = self.key_value_proj_dim * self.n_heads
#         self.pruned_heads = self.pruned_heads.union(heads)

#     @staticmethod
#     def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
#         """
#         Adapted from Mesh Tensorflow:
#         https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
#         Translate relative position to a bucket number for relative attention. The relative position is defined as
#         memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
#         position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
#         small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
#         positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
#         This should allow for more graceful generalization to longer sequences than the model has been trained on
#         Args:
#             relative_position: an int32 Tensor
#             bidirectional: a boolean - whether the attention is bidirectional
#             num_buckets: an integer
#             max_distance: an integer
#         Returns:
#             a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
#         """
#         relative_buckets = 0
#         if bidirectional:
#             num_buckets //= 2
#             relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
#             relative_position = torch.abs(relative_position)
#         else:
#             relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
#         # now relative_position is in the range [0, inf)

#         # half of the buckets are for exact increments in positions
#         max_exact = num_buckets // 2
#         is_small = relative_position < max_exact

#         # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
#         relative_position_if_large = max_exact + (
#             torch.log(relative_position.float() / max_exact)
#             / math.log(max_distance / max_exact)
#             * (num_buckets - max_exact)
#         ).to(torch.long)
#         relative_position_if_large = torch.min(
#             relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
#         )

#         relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
#         return relative_buckets

#     def compute_bias(self, query_length, key_length, device=None):
#         """Compute binned relative position bias"""
#         if device is None:
#             device = self.relative_attention_bias.weight.device
#         context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
#         memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
#         relative_position = memory_position - context_position  # shape (query_length, key_length)
#         relative_position_bucket = self._relative_position_bucket(
#             relative_position,  # shape (query_length, key_length)
#             bidirectional=(not self.is_decoder),
#             num_buckets=self.relative_attention_num_buckets,
#             max_distance=self.relative_attention_max_distance,
#         )
#         values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
#         values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
#         return values

#     def forward(
#         self,
#         hidden_states,
#         mask=None,
#         key_value_states=None,
#         position_bias=None,
#         past_key_value=None,
#         layer_head_mask=None,
#         query_length=None,
#         use_cache=False,
#         output_attentions=False,
#     ):
#         """
#         Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
#         """
#         # Input is (batch_size, seq_length, dim)
#         # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
#         # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
#         batch_size, seq_length = hidden_states.shape[:2]

#         real_seq_length = seq_length

#         if past_key_value is not None:
#             assert (
#                 len(past_key_value) == 2
#             ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
#             real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

#         key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

#         def shape(states):
#             """projection"""
#             return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

#         def unshape(states):
#             """reshape"""
#             return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

#         def project(hidden_states, proj_layer, key_value_states, past_key_value):
#             """projects hidden states correctly to key/query states"""
#             if key_value_states is None:
#                 # self-attn
#                 # (batch_size, n_heads, seq_length, dim_per_head)
#                 hidden_states = shape(proj_layer(hidden_states))
#             elif past_key_value is None:
#                 # cross-attn
#                 # (batch_size, n_heads, seq_length, dim_per_head)
#                 hidden_states = shape(proj_layer(key_value_states))

#             if past_key_value is not None:
#                 if key_value_states is None:
#                     # self-attn
#                     # (batch_size, n_heads, key_length, dim_per_head)
#                     hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
#                 elif past_key_value.shape[2] != key_value_states.shape[1]:
#                     # checking that the `sequence_length` of the `past_key_value` is the same as
#                     # the provided `key_value_states` to support prefix tuning
#                     # cross-attn
#                     # (batch_size, n_heads, seq_length, dim_per_head)
#                     hidden_states = shape(proj_layer(key_value_states))
#                 else:
#                     # cross-attn
#                     hidden_states = past_key_value
#             return hidden_states

#         # get query states
#         query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

#         # get key/value states
#         key_states = project(
#             hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
#         )
#         value_states = project(
#             hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
#         )

#         # compute scores
#         scores = torch.matmul(
#             query_states, key_states.transpose(3, 2)
#         )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

#         if position_bias is None:
#             if not self.has_relative_attention_bias:
#                 position_bias = torch.zeros(
#                     (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
#                 )
#                 if self.gradient_checkpointing and self.training:
#                     position_bias.requires_grad = True
#             else:
#                 position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

#             # if key and values are already calculated
#             # we want only the last query position bias
#             if past_key_value is not None:
#                 position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

#             if mask is not None:
#                 position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

#         if self.pruned_heads:
#             mask = torch.ones(position_bias.shape[1])
#             mask[list(self.pruned_heads)] = 0
#             position_bias_masked = position_bias[:, mask.bool()]
#         else:
#             position_bias_masked = position_bias

#         scores += position_bias_masked

#         attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
#             scores
#         )  # (batch_size, n_heads, seq_length, key_length)

#         #attn_weights = nn.functional.dropout(
#         #    attn_weights, p=self.dropout, training=self.training
#         #)  # (batch_size, n_heads, seq_length, key_length)

#         # Mask heads if we want to
#         if layer_head_mask is not None:
#             attn_weights = attn_weights * layer_head_mask

#         attn_output = torch.matmul(attn_weights, value_states)
#         attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
#         attn_output = self.o(attn_output)

#         #return (attn_output, 0)

#         present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
#         outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

#         if output_attentions:
#             outputs = outputs + (attn_weights,)
#         return outputs


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
        self.is_decoder = config["is_decoder"]
        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config["relative_attention_num_buckets"]
        self.relative_attention_max_distance = config["relative_attention_max_distance"]
        self.d_model = config["d_model"]
        self.key_value_proj_dim = config["d_kv"]
        self.n_heads = config["num_heads"]
        self.dropout = config["dropout_rate"]
        self.inner_dim = self.n_heads * self.key_value_proj_dim
        self.device = device
        self.mem_config = ttnn.L1_MEMORY_CONFIG

        self.q_weights = torch2tt_tensor(state_dict[f"{base_address}.q.weight"], device)
        self.k_weights = torch2tt_tensor(state_dict[f"{base_address}.k.weight"], device)
        self.v_weights = torch2tt_tensor(state_dict[f"{base_address}.v.weight"], device)
        self.o_weights = torch2tt_tensor(state_dict[f"{base_address}.o.weight"], device)

        self.q_weights = ttnn.transpose(self.q_weights, -2, -1)
        self.k_weights = ttnn.transpose(self.k_weights, -2, -1)
        self.v_weights = ttnn.transpose(self.v_weights, -2, -1)
        self.o_weights = ttnn.transpose(self.o_weights, -2, -1)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            self.relative_attention_bias.weight = nn.Parameter(
                state_dict[f"{base_address}.relative_attention_bias.weight"]
            )

        self.cached_position_bias = None
        self.cached_real_seq_length = None
        self.cached_key_length = None

        self.pruned_heads = set()
        self.gradient_checkpointing = False

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on
        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer
        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_position_if_large = torch.min(
            relative_position_if_large,
            torch.full_like(relative_position_if_large, num_buckets - 1),
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias_const(self, query_length, key_length):
        """Compute binned relative position bias"""
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).

        Input is (batch_size, seq_length, dim) in tt (1, batch_size, seq_length, dim) or (batch_size, 1, seq_length, dim)
        Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        """
        batch_size = hidden_states.shape.with_tile_padding()[1]
        seq_length = hidden_states.shape.with_tile_padding()[2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape.with_tile_padding()[2]

        def shape(states):
            """projection"""
            # return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            return t5_shape_tt(states, batch_size, self.n_heads, self.key_value_proj_dim, self.device)

        def unshape(states):
            """reshape"""
            # return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            return t5_unshape_tt(states, batch_size, self.inner_dim, self.device)

        def project(hidden_states, proj_weights, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(ttnn.matmul(hidden_states, proj_weights, memory_config=self.mem_config))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(ttnn.matmul(key_value_states, proj_weights, memory_config=self.mem_config))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = tt2torch_tensor(hidden_states)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                    hidden_states = torch2tt_tensor(hidden_states, self.device)
                elif past_key_value.shape[2] != key_value_states.shape.with_tile_padding()[2]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(ttnn.matmul(key_value_states, proj_weights, memory_config=self.mem_config))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(
            ttnn.matmul(hidden_states, self.q_weights, memory_config=self.mem_config)
            # self.q(hidden_states)
        )  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states,
            self.k_weights,
            key_value_states,
            past_key_value[0] if past_key_value is not None else None,
        )
        value_states = project(
            hidden_states,
            self.v_weights,
            key_value_states,
            past_key_value[1] if past_key_value is not None else None,
        )

        # compute scores
        # scores = torch.matmul(query_states, key_states.transpose(3, 2))
        # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        transposed_key_states = ttnn.transpose(key_states, -2, -1)
        scores = ttnn.matmul(query_states, transposed_key_states, memory_config=self.mem_config)

        if (
            position_bias is None
            and self.cached_real_seq_length == real_seq_length
            and self.cached_key_length == key_length
        ):
            position_bias = self.cached_position_bias

        elif position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                )

                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias_const(real_seq_length, key_length)
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.shape.with_tile_padding()[2] :, :]

            # "Broadcast" position bias so it can be used in + operation
            position_bias = position_bias.repeat(batch_size, 1, 1, 1)

            if mask is not None:
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

            # Prunned heads!
            if self.pruned_heads:
                mask = torch.ones(position_bias.shape.with_tile_padding()[2])
                mask[list(self.pruned_heads)] = 0
                position_bias = position_bias[:, mask.bool()]

            # Transfer to tt device
            position_bias = torch2tt_tensor(position_bias, self.device)

            # Cache it
            self.cached_position_bias = position_bias
            self.cached_real_seq_length = real_seq_length
            self.cached_key_length = key_length

        # scores += position_bias_masked
        scores = ttnn.add(scores, position_bias, memory_config=self.mem_config)

        # attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = ttnn.softmax_in_place(scores)

        # Dropout is not used in inference
        # attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = ttnn.mul(attn_weights, layer_head_mask, self.mem_config)

        attn_output = ttnn.matmul(attn_weights, value_states, memory_config=self.mem_config)
        attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
        attn_output = ttnn.matmul(attn_output, self.o_weights, memory_config=self.mem_config)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs

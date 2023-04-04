from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

import math
import torch
import numpy as np
from torch import nn
from libs import tt_lib as ttm

from transformers import T5Model
from utility_functions import print_diff_argmax
from fused_ops.linear import Linear as TtLinear
from python_api_testing.sweep_tests.comparison_funcs import comp_allclose, comp_pcc

from loguru import logger
from python_api_testing.fused_ops.softmax import softmax as tt_softmax
from python_api_testing.models.t5.t5_utils import torch2tt_tensor, tt2torch_tensor, read_model_config, print_corr_coef


def t5_shape_tt(states, batch_size, n_heads, key_value_proj_dim, device):
    """projection"""
    fall_back_to_torch = True

    if fall_back_to_torch:
        states = tt2torch_tensor(states)
        states = torch.reshape(states, (batch_size, -1, n_heads, key_value_proj_dim))
        states = states.transpose(1, 2)
        tt_out = torch2tt_tensor(states, device)
    else:
        tt_out = ttm.tensor.reshape(states, batch_size, -1, n_heads, key_value_proj_dim)
        tt_out = ttm.tensor.transpose_hc(tt_out)

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


def test_t5_shape(device):

    batch_size = 32
    n_heads = 8
    key_value_proj_dim = 64 #

    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    pt_out = t5_shape_pt(test_input, batch_size, n_heads, key_value_proj_dim)
    tt_out = t5_shape_tt(torch2tt_tensor(test_input, device), batch_size, n_heads, key_value_proj_dim, device)
    tt_out = tt2torch_tensor(tt_out)

    # print(tt_out.shape)
    # print(pt_out.shape)
    assert(tt_out.shape == pt_out.shape)

    # print(pt_out[0, 0, 1:10, 1:10])
    # print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("t5_shape test Passed!")
    else:
        logger.warning("t5_shape test Failed!")


def t5_unshape_pt(states, batch_size, inner_dim):
    return states.transpose(1, 2).contiguous().view(1, batch_size, -1, inner_dim)


def t5_unshape_tt(states, batch_size, inner_dim, device):
    fall_back_to_torch = True

    if fall_back_to_torch:
        states = tt2torch_tensor(states)
        states = t5_unshape_pt(states, batch_size, inner_dim)
        tt_out = torch2tt_tensor(states, device)
    else:
        assert False

    return tt_out


def test_t5_unshape(device):
    torch.manual_seed(0)
    test_input = (torch.rand(32, 8, 128, 64) * 2) - 1

    batch_size = 32
    inner_dim = 512

    pt_out = t5_unshape_pt(test_input, batch_size, inner_dim)
    tt_out = t5_unshape_tt(torch2tt_tensor(test_input, device), batch_size, inner_dim, device)
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test unshape Passed!")
    else:
        logger.warning("Test unshape Failed!")


def test_transpose(device):
    torch.manual_seed(0)
    test_input = (torch.rand(1, 1, 2048, 512) * 2) - 1

    pt_out = test_input.transpose(3, 2)
    tt_out = ttm.tensor.transpose(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test traspose Passed!")
    else:
        logger.warning("Test traspose Failed!")


def test_matmul(device):
    torch.manual_seed(0)
    test_input1 = ((torch.rand(32, 8, 128, 64) * 2) - 1)
    test_input2 = ((torch.rand(32, 8, 64, 128) * 2) - 1)

    pt_out = torch.matmul(test_input1, test_input2)
    tt_out = ttm.tensor.bmm(torch2tt_tensor(test_input1, device), torch2tt_tensor(test_input2, device))
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test matmul Passed!")
    else:
        logger.warning("Test matmul Failed!")


def test_softmax(device):
    torch.manual_seed(0)
    test_input  = ((torch.rand(32, 8, 128, 128) * 2) - 1)

    pt_out = nn.functional.softmax(test_input.float(), dim=-1).type_as(test_input)

    tt_out = tt_softmax(torch2tt_tensor(test_input, device))
    tt_out = tt2torch_tensor(tt_out)

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.99)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    assert does_pass

    if does_pass:
        logger.info("Test softmax Passed!")
    else:
        logger.warning("Test softmax Failed!")


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

#         print(f"Linear layer dtype {self.q.weight.dtype}")

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
    def __init__(self, config, state_dict, base_address, device, has_relative_attention_bias=False):
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

        self.q_weights = torch2tt_tensor(state_dict[f"{base_address}.q.weight"], ttm.device.GetHost())
        self.k_weights = torch2tt_tensor(state_dict[f"{base_address}.k.weight"], ttm.device.GetHost())
        self.v_weights = torch2tt_tensor(state_dict[f"{base_address}.v.weight"], ttm.device.GetHost())
        self.o_weights = torch2tt_tensor(state_dict[f"{base_address}.o.weight"], ttm.device.GetHost())

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = TtLinear(self.d_model, self.inner_dim, weight=self.q_weights.data(), bias=None, device=device)
        self.k = TtLinear(self.d_model, self.inner_dim, weight=self.k_weights.data(), bias=None, device=device)
        self.v = TtLinear(self.d_model, self.inner_dim, weight=self.v_weights.data(), bias=None, device=device)
        self.o = TtLinear(self.d_model, self.inner_dim, weight=self.o_weights.data(), bias=None, device=device)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
            self.relative_attention_bias.weight = nn.Parameter(state_dict[f"{base_address}.relative_attention_bias.weight"])

        self.pruned_heads = set()
        self.gradient_checkpointing = False

    # def prune_heads(self, heads):
    #     if len(heads) == 0:
    #         return

    #     heads, index = find_pruneable_heads_and_indices(
    #         heads, self.n_heads, self.key_value_proj_dim, self.pruned_heads
    #     )

    #     # Prune linear layers
    #     self.q = prune_linear_layer(self.q, index)
    #     self.k = prune_linear_layer(self.k, index)
    #     self.v = prune_linear_layer(self.v, index)
    #     self.o = prune_linear_layer(self.o, index, dim=1)

    #     # Update hyper params
    #     self.n_heads = self.n_heads - len(heads)
    #     self.inner_dim = self.key_value_proj_dim * self.n_heads
    #     self.pruned_heads = self.pruned_heads.union(heads)

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
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
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
        """
        # Input is (batch_size, seq_length, dim) in tt (1, batch_size, seq_length, dim) or (batch_size, 1, seq_length, dim) ???
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size = hidden_states.shape()[1]
        seq_length = hidden_states.shape()[2]

        print(f"hidden states shape {hidden_states.shape()}")
        print(f"batch_size {batch_size}")
        print(f"seq_length {seq_length}")

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
            real_seq_length += past_key_value[0].shape()[3] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape()[2]

        def shape(states):
            """projection"""
            #return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            return t5_shape_tt(states, batch_size, self.n_heads, self.key_value_proj_dim, self.device)

        def unshape(states):
            """reshape"""
            #return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
            return t5_unshape_tt(states, batch_size, self.inner_dim, self.device)

        def project(hidden_states, proj_layer, key_value_states, past_key_value):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(hidden_states))
            elif past_key_value is None:
                # cross-attn
                # (batch_size, n_heads, seq_length, dim_per_head)
                hidden_states = shape(proj_layer(key_value_states))

            if past_key_value is not None:
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, key_length, dim_per_head)
                    hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                elif past_key_value.shape()[3] != key_value_states.shape()[2]:
                    # checking that the `sequence_length` of the `past_key_value` is the same as
                    # the provided `key_value_states` to support prefix tuning
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))
                else:
                    # cross-attn
                    hidden_states = past_key_value
            return hidden_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        key_states = project(
            hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
        )
        value_states = project(
            hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
        )

        # compute scores
        #scores = torch.matmul(
        #    query_states, key_states.transpose(3, 2)
        #)  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        transposed_key_states = ttm.tensor.transpose(key_states)

        # print(f"ttm.tensor.bmm query_states x transposed_key_states: {query_states.shape()} x {transposed_key_states.shape()}")
        scores = ttm.tensor.bmm(query_states, transposed_key_states)

        if position_bias is None:
            if not self.has_relative_attention_bias:

                position_bias = torch.zeros(
                    (1, self.n_heads, real_seq_length, key_length),
                )

                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

            # "Broadcast" position bias so it can be used in + operation
            position_bias = position_bias.repeat(batch_size, 1, 1, 1)

            if mask is not None:
                position_bias = position_bias + tt2torch_tensor(mask)  # (batch_size, n_heads, seq_length, key_length)

        # Prunned heads!
        if self.pruned_heads:
            mask = torch.ones(position_bias.shape()[2])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
            position_bias_masked = position_bias

        # Transfer to tt device
        position_bias_masked = torch2tt_tensor(position_bias_masked, self.device)

        # scores += position_bias_masked
        scores = ttm.tensor.add(scores, position_bias_masked)

        # attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = tt_softmax(scores, stable=False) # (batch_size, n_heads, seq_length, key_length)

        # Dropout is not used in inference
        # attn_weights = nn.functional.dropout(
        #    attn_weights, p=self.dropout, training=self.training
        #)  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            # attn_weights = attn_weights * layer_head_mask
            attn_weights = ttm.tensor.mul(attn_weights, layer_head_mask)

        # torch.matmul(attn_weights, value_states)
        attn_output = ttm.tensor.bmm(attn_weights, value_states)
        attn_output = unshape(attn_output)  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        # return (attn_output, 0)

        present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


def test_T5Attention_inference(device):
    hugging_face_reference_model = T5Model.from_pretrained("t5-small") #, torch_dtype=torch.float16)
    hugging_face_reference_model.eval()

    # Input is (batch_size, seq_length, dim)
    torch.manual_seed(0)
    test_input = ((torch.rand(32, 128, 512) * 2) - 1) #.to(torch.float16)

    # // https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/configuration_t5.py
    model_json_config = "tests/python_api_testing/models/t5/t5-small.json"
    config = read_model_config(model_json_config)
    block = 2
    has_relative_attention_bias = bool(block == 0)

    # Module to test
    if config["is_decoder"]:
        hf_reference_module = hugging_face_reference_model.decoder.block[block].layer[0].SelfAttention
        base_address = f"decoder.block.{block}.layer.0.SelfAttention"
    else:
        hf_reference_module = hugging_face_reference_model.encoder.block[block].layer[0].SelfAttention
        base_address = f"encoder.block.{block}.layer.0.SelfAttention"

    pytorch_model = hf_reference_module
    # pytorch_model = T5Attention(config, hf_reference_module)
    pt_out = pytorch_model(test_input)[0].unsqueeze(0)

    test_input = test_input.unsqueeze(0)
    tt_test_input = torch2tt_tensor(test_input, device)

    tt_model = TtT5Attention(config, hugging_face_reference_model.state_dict(), base_address, device, has_relative_attention_bias)
    tt_out = tt_model(tt_test_input)[0]
    tt_out = tt2torch_tensor(tt_out)

    print(pt_out[0, 0, 1:10, 1:10])
    print(tt_out[0, 0, 1:10, 1:10])

    print_diff_argmax(pt_out, tt_out)
    does_pass, pcc_message = comp_pcc(pt_out, tt_out, 0.98)

    print(comp_allclose(pt_out, tt_out))
    print(pcc_message)

    if does_pass:
        logger.info("test_T5Attention_inference Passed!")
    else:
        logger.warning("test_T5Attention_inference Failed!")


if __name__ == "__main__":
    device = ttm.device.CreateDevice(ttm.device.Arch.GRAYSKULL, 0)
    ttm.device.InitializeDevice(device)

    test_t5_shape(device)
    test_transpose(device)
    test_matmul(device)
    test_softmax(device)
    test_t5_unshape(device)
    test_T5Attention_inference(device)

    ttm.device.CloseDevice(device)

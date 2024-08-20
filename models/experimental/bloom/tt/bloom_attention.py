# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch.nn import functional as F

import ttnn
import models.experimental.bloom.bloom_utils as bloom_utils
import models.experimental.bloom.tt.bloom_merge_heads as bloom_merge_heads
from tt_lib.fused_ops.softmax import softmax as tt_softmax

import models.experimental.bloom.tt.baddbmm as baddbmm
from typing import Optional, Tuple, Union
from models.utility_functions import pad_by_zero


def split_heads(fused_qkv: torch.Tensor, num_heads, head_dim) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split the last dimension into (num_heads, head_dim) without making any copies, results share same memory
    storage as `fused_qkv`

    Args:
        fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length, num_heads * 3 * head_dim]

    Returns:
        query: [batch_size, seq_length, num_heads, head_dim] key: [batch_size, seq_length, num_heads, head_dim]
        value: [batch_size, seq_length, num_heads, head_dim]
    """
    if len(fused_qkv.shape) == 3:
        batch_size, seq_length, three_times_hidden_size = fused_qkv.shape
    else:
        _, batch_size, seq_length, three_times_hidden_size = fused_qkv.shape

    fused_qkv = fused_qkv.view(batch_size, seq_length, num_heads, 3, head_dim)
    return fused_qkv[..., 0, :], fused_qkv[..., 1, :], fused_qkv[..., 2, :]


def merge_heads(x: torch.Tensor, num_heads, head_dim) -> torch.Tensor:
    """
    Merge heads together over the last dimenstion

    Args:
        x: (`torch.tensor`, *required*): [batch_size * num_heads, seq_length, head_dim]

    Returns:
        torch.tensor: [batch_size, seq_length, num_heads * head_dim]
    """
    # What we want to achieve is:
    # batch_size * num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads * head_dim
    batch_size_and_num_heads, seq_length, _ = x.shape
    batch_size = batch_size_and_num_heads // num_heads

    # First view     to decompose the batch size
    # batch_size * num_heads, seq_length, head_dim -> batch_size, num_heads, seq_length, head_dim
    x = x.view(batch_size, num_heads, seq_length, head_dim)

    # batch_size, num_heads, seq_length, head_dim -> batch_size, seq_length, num_heads, head_dim
    x = x.permute(0, 2, 1, 3)

    # batch_size, seq_length, num_heads, head_dim -> batch_size, seq_length, num_heads * head_dim
    return x.reshape(batch_size, seq_length, num_heads * head_dim)


# class BloomAttention(torch.nn.Module):
#     def __init__(self, dict_name, num, bloom_reference_model, hidden_size, num_heads, hidden_dropout, beta):
#         super().__init__()

#         sd = bloom_reference_model.state_dict()

#         self.hidden_size = hidden_size
#         self.num_heads = num_heads
#         self.head_dim = self.hidden_size // self.num_heads
#         self.split_size = self.hidden_size
#         self.hidden_dropout = hidden_dropout

#         if self.head_dim * self.num_heads != self.hidden_size:
#             raise ValueError(
#                 f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
#                 f" {self.num_heads})."
#             )

#         # Layer-wise attention scaling
#         self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
#         self.beta = beta

#         weight_q = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.self_attention.query_key_value.weight", sd)
#         bias_q = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.self_attention.query_key_value.bias", sd)

#         weight_d = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.self_attention.dense.weight", sd)
#         bias_d = bloom_utils.pt_load_layer_weights(f"{dict_name}.{num}.self_attention.dense.bias", sd)

#         self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
#         self.dense = torch.nn.Linear(self.hidden_size, self.hidden_size)
#         self.attention_dropout = torch.nn.Dropout(self.hidden_dropout)

#         self.query_key_value.weight = weight_q
#         self.query_key_value.bias = bias_q

#         self.dense.weight = weight_d
#         self.dense.bias = bias_d

#         self.pretraining_tp = False
#         self.slow_but_exact = False

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         residual: torch.Tensor,
#         alibi: torch.Tensor,
#         attention_mask: torch.Tensor,
#         layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         use_cache: bool = False,
#         output_attentions: bool = False,
#     ):
#         fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]

#         # 3 x [batch_size, seq_length, num_heads, head_dim]
#         (query_layer, key_layer, value_layer) = split_heads(fused_qkv, self.num_heads, self.head_dim)

#         if use_cache is True:
#             present = (key_layer, value_layer)
#         else:
#             present = None

#         batch_size, q_length, _, _ = query_layer.shape

#         query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
#         key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
#         value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

#         if layer_past is not None:
#             past_key, past_value = layer_past
#             # concatenate along seq_length dimension:
#             #  - key: [batch_size * self.num_heads, head_dim, kv_length]
#             #  - value: [batch_size * self.num_heads, kv_length, head_dim]
#             key_layer = torch.cat((past_key, key_layer), dim=2)
#             value_layer = torch.cat((past_value, value_layer), dim=1)

#         _, _, kv_length = key_layer.shape

#         # [batch_size * num_heads, q_length, kv_length]
#         # we use `torch.Tensor.baddbmm` instead of `torch.baddbmm` as the latter isn't supported by TorchScript v1.11
#         matmul_result = alibi.baddbmm(
#             batch1=query_layer,
#             batch2=key_layer,
#             beta=self.beta,
#             alpha=self.inv_norm_factor,
#         )

#         # change view to [batch_size, num_heads, q_length, kv_length]
#         attention_scores = matmul_result.view(batch_size, self.num_heads, q_length, kv_length)

#         # cast attention scores to fp32, compute scaled softmax and cast back to initial dtype - [batch_size, num_heads, q_length, kv_length]
#         input_dtype = attention_scores.dtype

#         # `float16` has a minimum value of -65504.0, whereas `bfloat16` and `float32` have a minimum value of `-3.4e+38`
#         if input_dtype == torch.float16:
#             attention_scores = attention_scores.to(torch.float)

#         attn_weights = torch.masked_fill(attention_scores, attention_mask, torch.finfo(attention_scores.dtype).min)
#         attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(input_dtype)

#         # [batch_size, num_heads, q_length, kv_length]
#         # attention_probs = self.attention_dropout(attention_probs)

#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         # change view [batch_size x num_heads, q_length, kv_length]
#         attention_probs_reshaped = attention_probs.view(batch_size * self.num_heads, q_length, kv_length)

#         # matmul: [batch_size * num_heads, q_length, head_dim]
#         context_layer = torch.bmm(attention_probs_reshaped, value_layer)

#         # change view [batch_size, num_heads, q_length, head_dim]
#         context_layer = merge_heads(context_layer, self.num_heads, self.head_dim)

#         print('SHAPE-------')
#         print(context_layer.shape)

#         # aggregate results across tp ranks. See here: https://github.com/pytorch/pytorch/issues/76232
#         if self.pretraining_tp > 1 and self.slow_but_exact:
#             slices = self.hidden_size / self.pretraining_tp
#             output_tensor = torch.zeros_like(context_layer)
#             for i in range(self.pretraining_tp):
#                 output_tensor = output_tensor + F.linear(
#                     context_layer[:, :, int(i * slices) : int((i + 1) * slices)],
#                     self.dense.weight[:, int(i * slices) : int((i + 1) * slices)],
#                 )
#         else:
#             output_tensor = self.dense(context_layer)

#         output_tensor = dropout_add.dropout_add(output_tensor, residual, self.hidden_dropout, self.training)
#         return output_tensor


class TtBloomAttention(torch.nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.mem_config = ttnn.L1_MEMORY_CONFIG
        self.device = device
        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.hidden_dropout = config.hidden_dropout
        self.use_tt_softmax = False

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.weight_q = pad_by_zero(state_dict[f"{base_address}.query_key_value.weight"], device)[0]
        self.bias_q = pad_by_zero(state_dict[f"{base_address}.query_key_value.bias"], device)[0]

        self.weight_d = pad_by_zero(state_dict[f"{base_address}.dense.weight"], device)[0]
        self.bias_d = pad_by_zero(state_dict[f"{base_address}.dense.bias"], device)[0]

        # Transpose the weights
        self.weight_q = ttnn.transpose(self.weight_q, -2, -1)
        self.weight_d = ttnn.transpose(self.weight_d, -2, -1)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.alpha = None

        self.attention_dropout = torch.nn.Dropout(0.0)

    def get_alpha(self, q_length):
        if self.alpha is not None:
            if self.alpha.get_legacy_shape()[3] == q_length:
                return self.alpha

        alpha_beta_shape = [1, self.num_heads, q_length, q_length]
        self.alpha = bloom_utils.tt_const_tensor(self.inv_norm_factor, alpha_beta_shape, self.device)
        return self.alpha

    def forward(
        self,
        device,
        hidden_states,  # : torch.Tensor,
        residual,  #: torch.Tensor,
        alibi,  #: torch.Tensor,
        attention_mask,  #: torch.Tensor,
        layer_past=None,  #: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        head_mask=None,  #: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ):
        # fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        fused_qkv = bloom_utils.tt_matmul(hidden_states, self.weight_q, device)
        fused_qkv = ttnn.add(fused_qkv, self.bias_q, memory_config=self.mem_config)
        fused_qkv = bloom_utils.tt2torch_tensor(fused_qkv)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_heads(fused_qkv, self.num_heads, self.head_dim)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        batch_size, q_length, _, _ = query_layer.shape

        query_layer = query_layer.transpose(1, 2)
        query_layer = bloom_utils.torch2tt_tensor(query_layer, device)
        reshaped_query_layer = ttnn.reshape_on_device(
            query_layer, 1, batch_size * self.num_heads, q_length, self.head_dim
        )

        key_layer = key_layer.permute(0, 2, 3, 1)

        key_layer = bloom_utils.torch2tt_tensor(key_layer, device)
        reshaped_key_layer = ttnn.reshape_on_device(key_layer, 1, batch_size * self.num_heads, self.head_dim, q_length)

        value_layer = value_layer.transpose(1, 2)
        value_layer = bloom_utils.torch2tt_tensor(value_layer, device)
        reshaped_value_layer = ttnn.reshape_on_device(
            value_layer, 1, batch_size * self.num_heads, q_length, self.head_dim
        )

        _, _, _, kv_length = reshaped_key_layer.get_legacy_shape()

        matmul_result = baddbmm.tt_baddbmm(
            device=device,
            input=alibi,
            batch1=reshaped_query_layer,
            batch2=reshaped_key_layer,
            beta=self.beta,
            alpha=self.get_alpha(q_length),
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = ttnn.reshape_on_device(matmul_result, batch_size, self.num_heads, q_length, kv_length)
        attention_scores = bloom_utils.tt2torch_tensor(attention_scores)

        if self.use_tt_softmax:
            attn_weights = torch.masked_fill(attention_scores, attention_mask, -100.0)
            attn_weights = bloom_utils.torch2tt_tensor(attn_weights, device)
            attention_probs = tt_softmax(attn_weights, stable=False)
        else:
            attn_weights = torch.masked_fill(
                attention_scores,
                attention_mask,
                torch.finfo(attention_scores.dtype).min,
            )
            attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attention_scores.dtype)
            attention_probs = bloom_utils.torch2tt_tensor(attention_probs, device)

        if head_mask is not None:
            head_mask = bloom_utils.torch2tt_tensor(head_mask, device)
            attention_probs = ttnn.mul(attention_probs, head_mask)

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = ttnn.reshape_on_device(
            attention_probs, 1, batch_size * self.num_heads, q_length, kv_length
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = bloom_utils.tt_bmm(attention_probs_reshaped, reshaped_value_layer, device)
        context_layer = bloom_utils.tt2torch_tensor(context_layer)

        # merged_context_layer = bloom_attention_merge_heads.tt_merge_heads(pt_context_layer.squeeze(), self.num_heads, self.hidden_size, self.num_heads, device)
        context_layer = context_layer.squeeze(0)
        tt_context_layer = bloom_merge_heads.tt_merge_heads(
            context_layer, self.num_heads, self.hidden_size, self.num_heads, device
        )

        # output_tensor = self.dense(merged_context_layer)
        output_tensor = bloom_utils.tt_matmul(tt_context_layer, self.weight_d, device)
        output_tensor = ttnn.add(
            output_tensor,
            self.bias_d,
            memory_config=self.mem_config,
        )

        # Dropout is used in training only
        # output_tensor = F.dropout(output_tensor, p=self.hidden_dropout, training=False)
        output_tensor = ttnn.add(residual, output_tensor, memory_config=self.mem_config)

        outputs = (output_tensor, present)

        if output_attentions:
            outputs += (attention_probs,)

        return outputs

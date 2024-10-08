# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import math
from torch.nn import functional as F

import ttnn
import models.experimental.bloom_old.bloom_utils as bloom_utils
import models.experimental.bloom_old.tt.baddbmm as baddbmm
import models.experimental.bloom_old.tt.bloom_attention_merge_heads as bloom_attention_merge_heads

from fused_ops.linear import Linear as TtLinear
from fused_ops.softmax import softmax as TtSoftmax
from typing import Optional, Tuple, Union


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

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.hidden_dropout = config.hidden_dropout

        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.weight_q = bloom_utils.tt_load_layer_weights(f"{base_address}.query_key_value.weight", state_dict, device)
        self.bias_q = bloom_utils.tt_load_layer_weights(f"{base_address}.query_key_value.bias", state_dict, device)

        self.weight_d = bloom_utils.tt_load_layer_weights(f"{base_address}.dense.weight", state_dict, device)
        self.bias_d = bloom_utils.tt_load_layer_weights(f"{base_address}.dense.bias", state_dict, device)

        # Layer-wise attention scaling
        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        alpha_beta_shape = [1, self.num_heads, self.head_dim, self.head_dim]
        self.inv_norm_factor = bloom_utils.tt_const_tensor(self.inv_norm_factor, alpha_beta_shape, device)

        self.query_key_value = TtLinear(self.hidden_size, 3 * self.hidden_size, self.weight_q, self.bias_q, device)
        self.dense = TtLinear(self.hidden_size, self.hidden_size, self.weight_d, self.bias_d, device)
        self.attention_dropout = torch.nn.Dropout(0.0)

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
        fused_qkv = self.query_key_value(hidden_states)  # [batch_size, seq_length, 3 x hidden_size]
        fused_qkv = bloom_utils.tt2torch_tensor(fused_qkv)

        # 3 x [batch_size, seq_length, num_heads, head_dim]
        (query_layer, key_layer, value_layer) = split_heads(fused_qkv, self.num_heads, self.head_dim)

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        batch_size, q_length, _, _ = query_layer.shape

        # p_reshaped_query_layer = torch.Tensor(fused_qkv).reshape(1, batch_size, seq * self.num_heads,  q_length, self.head_dim)
        # query_layer = query_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)

        query_layer = query_layer.transpose(1, 2)
        query_layer = bloom_utils.torch2tt_tensor(query_layer, device)
        reshaped_query_layer = ttnn.reshape_on_device(
            query_layer, 1, batch_size * self.num_heads, q_length, self.head_dim
        )

        # key_layer = key_layer.permute(0, 2, 3, 1).reshape(batch_size * self.num_heads, self.head_dim, q_length)
        key_layer = key_layer.permute(0, 2, 3, 1)

        key_layer = bloom_utils.torch2tt_tensor(key_layer, device)
        reshaped_key_layer = ttnn.reshape_on_device(key_layer, 1, batch_size * self.num_heads, self.head_dim, q_length)

        # value_layer = value_layer.transpose(1, 2).reshape(batch_size * self.num_heads, q_length, self.head_dim)
        value_layer = value_layer.transpose(1, 2)
        value_layer = bloom_utils.torch2tt_tensor(value_layer, device)
        reshaped_value_layer = ttnn.reshape_on_device(
            value_layer, 1, batch_size * self.num_heads, q_length, self.head_dim
        )

        _, _, _, kv_length = reshaped_key_layer.shape.with_tile_padding()

        matmul_result = baddbmm.tt_baddbmm(
            device=device,
            input=alibi,
            batch1=reshaped_query_layer,
            batch2=reshaped_key_layer,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        # change view to [batch_size, num_heads, q_length, kv_length]
        attention_scores = ttnn.reshape_on_device(matmul_result, batch_size, self.num_heads, q_length, kv_length)
        attention_scores = bloom_utils.tt2torch_tensor(attention_scores)

        attn_weights = torch.masked_fill(
            attention_scores, attention_mask, -100000.0
        )  # torch.finfo(attention_scores.dtype).min)

        attention_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(attention_scores.dtype)
        attention_probs = bloom_utils.torch2tt_tensor(attention_probs, device)

        # attn_weights = bloom_utils.torch2tt_tensor(attn_weights, device)
        # attention_probs = TtSoftmax(attn_weights)

        if head_mask is not None:
            head_mask = bloom_utils.torch2tt_tensor(head_mask, device)
            attention_probs = ttnn.mul(attention_probs, head_mask)

        # change view [batch_size x num_heads, q_length, kv_length]
        attention_probs_reshaped = ttnn.reshape_on_device(
            attention_probs, 1, batch_size * self.num_heads, q_length, kv_length
        )

        # matmul: [batch_size * num_heads, q_length, head_dim]
        context_layer = ttnn.bmm(attention_probs_reshaped, reshaped_value_layer)
        context_layer = bloom_utils.tt2torch_tensor(context_layer)

        context_layer = context_layer.squeeze(0)
        context_layer = merge_heads(context_layer, self.num_heads, self.head_dim)
        merged_context_layer = bloom_utils.torch2tt_tensor(context_layer, device)

        # merged_context_layer = bloom_attention_merge_heads.tt_merge_heads(pt_context_layer.squeeze(), self.num_heads, self.hidden_size, self.num_heads, device)
        output_tensor = self.dense(merged_context_layer)

        # Dropout is used in training only
        # output_tensor = F.dropout(output_tensor, p=self.hidden_dropout, training=False)
        output_tensor = ttnn.add(residual, output_tensor)

        outputs = (output_tensor, present)

        if output_attentions:
            outputs += (attention_probs,)

        return outputs

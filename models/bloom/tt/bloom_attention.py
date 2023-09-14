"""
SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

SPDX-License-Identifier: Apache-2.0
"""

import tt_lib
from typing import Tuple, Optional

import torch
import torch.nn as nn
import math
import tt_lib.fallback_ops as fallback_ops

from models.helper_funcs import Linear as TtLinear
from models.bloom.tt.baddbmm import TtBaddbmm
from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)


def split_heads(
    fused_qkv: tt_lib.tensor.Tensor, num_heads, head_dim, device
) -> Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor, tt_lib.tensor.Tensor]:
    _, batch_size, seq_length, three_times_hidden_size = fused_qkv.shape()

    torch_fused_qkv = tt_to_torch_tensor(fused_qkv)
    torch_fused_qkv = torch_fused_qkv.view(
        batch_size, seq_length, num_heads, 3, head_dim
    )

    return (
        torch_to_tt_tensor_rm(torch_fused_qkv[..., 0, :], device, put_on_device=True),
        torch_to_tt_tensor_rm(torch_fused_qkv[..., 1, :], device, put_on_device=True),
        torch_to_tt_tensor_rm(torch_fused_qkv[..., 2, :], device, put_on_device=True),
    )


def merge_heads(
    x: tt_lib.tensor.Tensor,
    num_heads: int,
    hidden_size: int,
    num_attention_heads: int,
) -> tt_lib.tensor.Tensor:
    mem_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)
    head_dim = hidden_size // num_attention_heads

    _, batch_size_and_num_heads, seq_length, _ = x.shape()
    batch_size = batch_size_and_num_heads // num_heads

    x = tt_lib.tensor.reshape(
        x, batch_size, num_heads, seq_length, head_dim, mem_config
    )
    x = tt_lib.tensor.permute(x, 0, 2, 1, 3, mem_config)

    """ Using fallback_ops.reshape, because of the limitation on operands of reshape to be TILE Layout.
        Since the shape gets modified, we cannot pad and unpad to use tt_lib.reshape """
    return fallback_ops.reshape(x, 1, batch_size, seq_length, num_heads * head_dim)


class TtBloomAttention(nn.Module):
    def __init__(self, config, state_dict, base_address, device):
        super().__init__()

        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address
        self.config = config
        self.mem_config = tt_lib.tensor.MemoryConfig(True, tt_lib.tensor.BufferType.L1)

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.hidden_dropout = config.hidden_dropout
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(
                f"`hidden_size` must be divisible by num_heads (got `hidden_size`: {self.hidden_size} and `num_heads`:"
                f" {self.num_heads})."
            )

        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0
        self.alpha = None

        self.query_key_value_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.query_key_value.weight"], self.device
        )

        self.query_key_value_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.query_key_value.bias"], self.device
        )

        self.query_key_value = TtLinear(
            self.query_key_value_weight.shape()[-1],
            self.query_key_value_weight.shape()[-2],
            self.query_key_value_weight,
            self.query_key_value_bias,
            self.mem_config,
        )

        self.dense_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense.weight"], self.device
        )

        self.dense_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.dense.bias"], self.device
        )

        self.dense = TtLinear(
            self.dense_weight.shape()[-1],
            self.dense_weight.shape()[-2],
            self.dense_weight,
            self.dense_bias,
            self.mem_config,
        )

        self.baddbmm = TtBaddbmm(self.device)

    def get_alpha(self, q_length):
        if self.alpha is not None:
            if self.alpha.shape()[3] == q_length:
                return self.alpha

        alpha_beta_shape = [1, self.num_heads, q_length, q_length]
        self.alpha = tt_lib.tensor.full(
            alpha_beta_shape, self.inv_norm_factor, output_mem_config=self.mem_config
        )
        return self.alpha

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        residual: tt_lib.tensor.Tensor,
        alibi: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor,
        layer_past: Optional[Tuple[tt_lib.tensor.Tensor, tt_lib.tensor.Tensor]] = None,
        head_mask: Optional[tt_lib.tensor.Tensor] = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> Tuple[tt_lib.tensor.Tensor, ...]:
        fused_qkv = self.query_key_value(hidden_states)

        (query_layer, key_layer, value_layer) = split_heads(
            fused_qkv, self.num_heads, self.head_dim, self.device
        )

        if use_cache is True:
            present = (key_layer, value_layer)
        else:
            present = None

        batch_size, q_length, _, _ = query_layer.shape()

        query_layer = tt_lib.tensor.transpose(
            query_layer, 1, 2, output_mem_config=self.mem_config
        )

        reshaped_query_layer = tt_lib.tensor.reshape(
            query_layer,
            1,
            batch_size * self.num_heads,
            q_length,
            self.head_dim,
            self.mem_config,
        )

        key_layer = tt_lib.tensor.permute(key_layer, 0, 2, 3, 1, self.mem_config)

        reshaped_key_layer = tt_lib.tensor.reshape(
            key_layer,
            1,
            batch_size * self.num_heads,
            self.head_dim,
            q_length,
            self.mem_config,
        )

        value_layer = tt_lib.tensor.transpose(
            value_layer, 1, 2, output_mem_config=self.mem_config
        )
        reshaped_value_layer = tt_lib.tensor.reshape(
            value_layer,
            1,
            batch_size * self.num_heads,
            q_length,
            self.head_dim,
            self.mem_config,
        )

        _, _, _, kv_length = reshaped_key_layer.shape()

        matmul_result = self.baddbmm(
            input=alibi,
            batch1=reshaped_query_layer,
            batch2=reshaped_key_layer,
            beta=self.beta,
            alpha=self.get_alpha(q_length),
        )

        attention_scores = tt_lib.tensor.reshape(
            matmul_result,
            batch_size,
            self.num_heads,
            q_length,
            kv_length,
            self.mem_config,
        )
        attention_scores = tt_to_torch_tensor(attention_scores)

        attention_mask = tt_to_torch_tensor(attention_mask)
        attention_mask = attention_mask.type(torch.int64)

        attn_weights = torch.masked_fill(
            attention_scores,
            attention_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attn_weights = torch_to_tt_tensor_rm(
            attn_weights, self.device, put_on_device=True
        )

        # tt_lib.operations.primary.softmax_in_place afftects the PCC of the model so using fallback_ops.softmax
        attention_probs = fallback_ops.softmax(attn_weights, dim=-1)

        if head_mask is not None:
            attention_probs = tt_lib.tensor.mul(
                attention_probs, head_mask, output_mem_config=self.mem_config
            )

        attention_probs_reshaped = tt_lib.tensor.reshape(
            attention_probs,
            1,
            batch_size * self.num_heads,
            q_length,
            kv_length,
            self.mem_config,
        )

        context_layer = tt_lib.tensor.bmm(
            attention_probs_reshaped, reshaped_value_layer, self.mem_config
        )

        tt_context_layer = merge_heads(
            context_layer, self.num_heads, self.hidden_size, self.num_heads
        )

        output_tensor = self.dense(tt_context_layer)

        output_tensor = tt_lib.tensor.add(
            residual, output_tensor, output_mem_config=self.mem_config
        )

        outputs = (output_tensor, present)

        if output_attentions:
            outputs += (attention_probs,)

        return outputs

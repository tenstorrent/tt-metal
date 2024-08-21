# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, List
import torch
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
import ttnn
import tt_lib.fallback_ops as fallback_ops
from models.helper_funcs import Linear as TtLinear


class TtMultiHeadSelfAttention(nn.Module):
    def __init__(self, config, state_dict=None, base_address="", device=None):
        super().__init__()
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.device = device

        if self.dim % self.n_heads != 0:
            raise ValueError(f"self.n_heads: {self.n_heads} must divide self.dim: {self.dim} evenly")

        self.query_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.q_lin.weight"], self.device)
        self.query_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.q_lin.bias"], self.device)
        self.query_linear = TtLinear(
            self.query_weight.get_legacy_shape()[-1],
            self.query_weight.get_legacy_shape()[-2],
            self.query_weight,
            self.query_bias,
        )

        self.key_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.k_lin.weight"], self.device)
        self.key_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.k_lin.bias"], self.device)
        self.key_linear = TtLinear(
            self.key_weight.get_legacy_shape()[-1],
            self.key_weight.get_legacy_shape()[-2],
            self.key_weight,
            self.key_bias,
        )

        self.value_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.v_lin.weight"], self.device)
        self.value_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.v_lin.bias"], self.device)
        self.value_linear = TtLinear(
            self.value_weight.get_legacy_shape()[-1],
            self.value_weight.get_legacy_shape()[-2],
            self.value_weight,
            self.value_bias,
        )

        self.out_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.out_lin.weight"], self.device)
        self.out_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.attention.out_lin.bias"], self.device)
        self.out_linear = TtLinear(
            self.out_weight.get_legacy_shape()[-1],
            self.out_weight.get_legacy_shape()[-2],
            self.out_weight,
            self.out_bias,
        )

        self.attention_head_size = self.dim // self.n_heads

    def const_tensor(self, shape: List[int], value: int) -> ttnn.Tensor:
        return ttnn.full(shape, value)

    def get_min(self, tensor: ttnn.Tensor):
        tensor = tt_to_torch_tensor(tensor)
        return torch.finfo(ttnn.dtype).min

    def forward(
        self,
        query: ttnn.Tensor,
        key: ttnn.Tensor,
        value: ttnn.Tensor,
        mask: ttnn.Tensor,
        head_mask: Optional[ttnn.Tensor] = None,
        output_attention: bool = False,
    ) -> Tuple[ttnn.Tensor]:
        _, bs, q_length, dim = query.get_legacy_shape()
        k_length = key.get_legacy_shape()[-2]

        dim_per_head = self.dim // self.n_heads

        mask_reshape = (bs, 1, 1, k_length)

        def shape(x: ttnn.Tensor) -> ttnn.Tensor:
            x = fallback_ops.reshape(x, bs, -1, self.n_heads, dim_per_head)
            return ttnn.transpose(x, 1, -2)

        def unshape(x: ttnn.Tensor) -> ttnn.Tensor:
            x = ttnn.transpose(x, 1, -2)
            x = fallback_ops.reshape(x, 1, bs, -1, self.n_heads * dim_per_head)
            return x

        q = shape(self.query_linear(query))
        k = shape(self.key_linear(key))
        v = shape(self.value_linear(value))

        dim_per_head_tensor = self.const_tensor(q.get_legacy_shape(), dim_per_head)
        dim_per_head_tensor = ttnn.sqrt(dim_per_head_tensor)
        dim_per_head_tensor = ttnn.reciprocal(dim_per_head_tensor)

        q = ttnn.mul(q, dim_per_head_tensor)
        scores = ttnn.matmul(q, ttnn.transpose(k, -2, -1))
        score_value = self.get_min(scores)
        scores = tt_to_torch_tensor(scores)
        mask = tt_to_torch_tensor(mask)

        """dtype bool is not supported in TT """
        mask = (mask == 0).view(mask_reshape).expand_as(scores)

        scores = scores.masked_fill(mask, score_value)

        scores = torch_to_tt_tensor_rm(scores, self.device, put_on_device=False)

        weights = fallback_ops.softmax(scores, -1)

        if head_mask is not None:
            weights = ttnn.mul(weights, head_mask)

        context = ttnn.matmul(weights, v)
        context = unshape(context)
        context = self.out_linear(context)

        if output_attention:
            return (context, weights)

        return (context,)

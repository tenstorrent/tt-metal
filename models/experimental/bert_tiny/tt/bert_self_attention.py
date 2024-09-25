# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import ttnn
import math
import torch
from typing import Optional
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import tt_to_torch_tensor, torch_to_tt_tensor
from models.experimental.bert_tiny.bert_tiny_helper_funcs import Linear as TtLinear


def mha(
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    hidden_dim,
    num_heads,
    device,
    out_mem_config,
):
    Q_projection = TtLinear(
        query_weight,
        query_bias,
        device=device,
        output_mem_config=out_mem_config,
    )

    K_projection = TtLinear(
        key_weight,
        key_bias,
        device=device,
        output_mem_config=out_mem_config,
    )

    V_projection = TtLinear(
        value_weight,
        value_bias,
        device=device,
        output_mem_config=out_mem_config,
    )
    reciprocal_of_sqrt_hidden_dim_tensor = ttnn.to_device(
        ttnn.from_torch(
            torch.tensor([1 / math.sqrt(hidden_dim // num_heads)] + [0 for _ in range(32 * 32 - 1)]),
            dtype=ttnn.bfloat16,
        ),
        memory_config=out_mem_config,
        device=device,
    )
    reciprocal_of_sqrt_hidden_dim_tensor = ttnn.reshape(reciprocal_of_sqrt_hidden_dim_tensor, (1, 1, 32, 32))

    def make_attetntion_heads(x: ttnn.Tensor):
        if num_heads == 1:
            return x

        x = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)))
        x = torch_to_tt_tensor(x, device)

        # Reshape expects the expected output shape to be in TILE layout
        # Input shape : [1, 1, 128, 128]
        # Expected output shape: [1, 128, 2, 64]
        reshape_unt = fallback_ops.reshape(
            x,
            x.shape.with_tile_padding()[0],
            x.shape.with_tile_padding()[2],
            num_heads,
            x.shape.with_tile_padding()[3] // num_heads,
        )
        # Permute expects input to be in TILE layout
        # Input shape: [1, 128, 2, 64]
        transposed = ttnn.permute(reshape_unt, [0, 2, 1, 3])

        transposed = tt_to_torch_tensor(transposed)
        transposed = ttnn.from_torch(transposed, dtype=ttnn.bfloat16)
        transposed = ttnn.to_device(ttnn.to_layout(transposed, layout=ttnn.TILE_LAYOUT), device=device)

        return transposed

    def unmake_attention_heads(x: ttnn.Tensor):
        if num_heads == 1:
            return x
        else:
            ctx = ttnn.permute(x, (0, 2, 1, 3))
            ushape = ctx.shape
            reshaped = ttnn.reshape(ctx, (ushape[0], 1, ushape[1], ushape[2] * ushape[3]))
            return reshaped

    def multiply_by_sqrt_hidden_dim(x: ttnn.Tensor, reciprocal_of_sqrt_hidden_dim_tensor: ttnn.Tensor):
        x = ttnn.to_torch(ttnn.from_device(ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)))
        x = torch_to_tt_tensor(x, device)

        reciprocal_of_sqrt_hidden_dim_tensor = ttnn.to_torch(ttnn.from_device(reciprocal_of_sqrt_hidden_dim_tensor))
        reciprocal_of_sqrt_hidden_dim_tensor = torch_to_tt_tensor(reciprocal_of_sqrt_hidden_dim_tensor, device)

        return ttnn.multiply(
            x,
            reciprocal_of_sqrt_hidden_dim_tensor,
            memory_config=out_mem_config,
        )

    def mha_(activation: ttnn.Tensor, attention_mask: ttnn.Tensor):
        Q = Q_projection(activation)
        K = K_projection(activation)
        V = V_projection(activation)
        Q_heads = make_attetntion_heads(Q)
        K_heads = make_attetntion_heads(K)
        V_heads = make_attetntion_heads(V)

        K_T_heads = ttnn.permute(K_heads, (0, 1, -1, -2))
        qkt = ttnn.matmul(Q_heads, K_T_heads)

        (
            N,
            C,
            H,
            W,
        ) = qkt.shape

        attention_score_input = multiply_by_sqrt_hidden_dim(qkt, reciprocal_of_sqrt_hidden_dim_tensor)
        attention_score_input = tt_to_torch_tensor(attention_score_input)
        attention_score_input = ttnn.to_layout(
            ttnn.from_torch(attention_score_input, dtype=ttnn.bfloat16), layout=ttnn.TILE_LAYOUT
        )

        if attention_mask is not None:
            attention_score_input = ttnn.to_device(attention_score_input, device)
            attention_score_input = ttnn.add(attention_score_input, attention_mask)
        attention_scores = ttnn.reshape(ttnn.softmax(attention_score_input, -1), (N, C, H, W))
        weighted_activation = ttnn.matmul(attention_scores, V_heads)
        return unmake_attention_heads(
            weighted_activation
        )  # [N, num heads, seq len, hid size / num heads] -> [N, seq len, hid size]

    return mha_


class TtBertselfattention(nn.Module):
    def __init__(self, config, encoder_idx: int, state_dict=None, device=None, mem_config=None):
        super().__init__()
        self.config = config

        self.device = device

        self.output_mem_config = mem_config

        self.query_weight = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.weight"], dtype=ttnn.bfloat16
            ),
            device=self.device,
        )

        self.query_bias = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.query.bias"], dtype=ttnn.bfloat16
            ),
            memory_config=self.output_mem_config,
            device=self.device,
        )

        self.key_weight = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.weight"], dtype=ttnn.bfloat16
            ),
            memory_config=self.output_mem_config,
            device=self.device,
        )

        self.key_bias = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.key.bias"], dtype=ttnn.bfloat16
            ),
            memory_config=self.output_mem_config,
            device=self.device,
        )
        self.value_weight = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.weight"], dtype=ttnn.bfloat16
            ),
            memory_config=self.output_mem_config,
            device=self.device,
        )

        self.value_bias = ttnn.to_device(
            ttnn.from_torch(
                state_dict[f"bert.encoder.layer.{encoder_idx}.attention.self.value.bias"], dtype=ttnn.bfloat16
            ),
            memory_config=self.output_mem_config,
            device=self.device,
        )

        hidden_dim = self.query_weight.shape[-1]

        parameters = [
            self.query_weight,
            self.query_bias,
            self.key_weight,
            self.key_bias,
            self.value_weight,
            self.value_bias,
        ]

        self.mha = mha(*parameters, hidden_dim, config.num_attention_heads, device, self.output_mem_config)

    def forward(self, activation: Optional[ttnn.Tensor] = None, attention_mask: Optional[ttnn.Tensor] = None):
        result = self.mha(activation, attention_mask)
        return result

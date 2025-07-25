# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from torch import Tensor
from typing import Union, Callable, Optional
import torch.nn.functional as F
import math

import ttnn

# from models.experimental.uniad.tt.ttnn_mha import TtMultiheadAttention


class TtMultiheadAttention_nn:
    def __init__(
        self,
        params,
        device,
        embed_dims=256,
        num_heads=8,
        init_cfg=None,
        batch_first=False,
    ):
        super().__init__()
        self.params = params
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.device = device
        self.batch_first = batch_first
        self.attn_in_proj__weight = params.in_proj_weight  # Changed here
        self.attn_in_proj__weight = ttnn.to_layout(self.attn_in_proj__weight, layout=ttnn.TILE_LAYOUT)  # Changed here
        self.attn_in_proj__bias = params.in_proj_bias  # Changed here
        self.attn_in_proj__bias = ttnn.to_layout(self.attn_in_proj__bias, layout=ttnn.TILE_LAYOUT)  # Changed here
        self.attn_in_proj__weight_permute = self.attn_in_proj__weight  # Changed here
        self.attn_in_proj__bias_squeeze = ttnn.squeeze(self.attn_in_proj__bias, 0)
        self.attn_out_proj_weight = params.out_proj.weight
        self.attn_out_proj_weight = ttnn.to_layout(self.attn_out_proj_weight, layout=ttnn.TILE_LAYOUT)  # Changed here
        self.attn_out_proj_bias = params.out_proj.bias
        self.attn_out_proj_bias = ttnn.to_layout(self.attn_out_proj_bias, layout=ttnn.TILE_LAYOUT)  # Changed here

    def __call__(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        attn_mask=None,
        **kwargs,
    ):
        in_proj_bias = self.attn_in_proj__bias_squeeze

        in_proj_weight = self.attn_in_proj__weight_permute

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        q_weight = in_proj_weight[: self.embed_dims, :]  # Query weights
        k_weight = in_proj_weight[self.embed_dims : 2 * self.embed_dims, :]  # Key weights
        v_weight = in_proj_weight[2 * self.embed_dims :, :]  # Value weights

        q_bias = in_proj_bias[: self.embed_dims]  # Query biases
        k_bias = in_proj_bias[self.embed_dims : 2 * self.embed_dims]  # Key biases
        v_bias = in_proj_bias[2 * self.embed_dims :]  # Value biases

        q_batch_size, q_sequence_size, q_hidden_size = query.shape
        q_head_size = q_hidden_size // self.num_heads

        k_batch_size, k_sequence_size, k_hidden_size = key.shape
        k_head_size = k_hidden_size // self.num_heads

        v_batch_size, v_sequence_size, v_hidden_size = value.shape
        v_head_size = v_hidden_size // self.num_heads

        q_weight = ttnn.permute(q_weight, (1, 0))
        k_weight = ttnn.permute(k_weight, (1, 0))
        v_weight = ttnn.permute(v_weight, (1, 0))
        query = ttnn.linear(query, q_weight, bias=q_bias)

        key = ttnn.linear(key, k_weight, bias=k_bias)

        if value.get_layout() != ttnn.TILE_LAYOUT:
            value = ttnn.to_layout(value, ttnn.TILE_LAYOUT)
        value = ttnn.linear(value, v_weight, bias=v_bias)

        query = ttnn.reshape(query, (tgt_len, bsz * self.num_heads, q_head_size))
        query = ttnn.permute(query, (1, 0, 2))

        key = ttnn.reshape(key, (k_batch_size, bsz * self.num_heads, q_head_size))
        key = ttnn.permute(key, (1, 0, 2))

        value = ttnn.reshape(value, (v_batch_size, bsz * self.num_heads, q_head_size))
        value = ttnn.permute(value, (1, 0, 2))

        src_len = key.shape[1]

        B, Nt, E = query.shape
        q_scaled = query * math.sqrt(1.0 / float(E))
        key_transposed = ttnn.permute(key, (0, 2, 1))

        if attn_mask is not None:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)
            attn_output_weights = attn_output_weights + attn_mask
        else:
            attn_output_weights = ttnn.matmul(q_scaled, key_transposed)

        attn_output_weights = ttnn.softmax(attn_output_weights, dim=-1)

        attn_output = ttnn.matmul(attn_output_weights, value)

        attn_output = ttnn.permute(attn_output, (1, 0, 2))
        attn_output = ttnn.reshape(attn_output, (tgt_len * bsz, embed_dim))

        attn_output = ttnn.linear(attn_output, self.attn_out_proj_weight, bias=self.attn_out_proj_bias)
        attn_output = ttnn.reshape(attn_output, (tgt_len, bsz, attn_output.shape[1]))
        attn_output_weights = ttnn.reshape(attn_output_weights, (bsz, self.num_heads, tgt_len, src_len))
        attn_output_weights = ttnn.to_layout(attn_output_weights, ttnn.ROW_MAJOR_LAYOUT)
        attn_output_weights = ttnn.mean(attn_output_weights, dim=1)

        return attn_output


class TtTransformerDecoderLayer:
    __constants__ = ["norm_first"]

    def __init__(
        self,
        parameters,
        device,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        dtype=None,
    ) -> None:
        self.parameters = parameters
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = TtMultiheadAttention_nn(
            parameters.self_attn,
            device,
            d_model,
            nhead,
            None,
            batch_first=batch_first,
        )
        self.multihead_attn = TtMultiheadAttention_nn(
            parameters.multihead_attn,
            device,
            d_model,
            nhead,
            None,
            batch_first=batch_first,
        )
        # Implementation of Feedforward model
        self.linear1 = ttnn.linear
        self.linear2 = ttnn.linear

        self.norm_first = norm_first
        self.norm1 = ttnn.layer_norm
        self.norm2 = ttnn.layer_norm
        self.norm3 = ttnn.layer_norm

        # Legacy string support for activation function.
        self.activation = ttnn.relu

    def __call__(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self.self_attn(
                self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
            )
            # self._sa_block(
            #     self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal
            # )
            x = x + self.multihead_attn(
                self.norm2(x), memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
            )
            # self._mha_block(
            #     self.norm2(x),
            #     memory,
            #     memory_mask,
            #     memory_key_padding_mask,
            #     memory_is_causal,
            # )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(
                (x + self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)),
                weight=self.parameters.norm1.weight,
                bias=self.parameters.norm1.bias,
                # dtype=ttnn.bfloat16
            )
            print("x", x.shape, "memory", memory.shape)
            x = self.norm2(
                x
                + self.multihead_attn(
                    x, memory, memory, attn_mask=memory_mask, key_padding_mask=memory_key_padding_mask
                )
                # self._mha_block(
                #     x, memory, memory_mask, memory_key_padding_mask, memory_is_causal
                # )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(
        self,
        x: Tensor,
        mem: Tensor,
        attn_mask: Optional[Tensor],
        key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            mem,
            mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            need_weights=False,
        )[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.activation(self.linear1(x)))
        return x

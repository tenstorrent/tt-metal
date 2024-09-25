# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

import ttnn
from tt_lib import fallback_ops

from models.helper_funcs import Linear
from models.utility_functions import torch_to_tt_tensor_rm


class TtTrOCRAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(
        self,
        config,
        embed_dim: int,
        num_heads: int,
        kdim: int = None,
        vdim: int = None,
        is_decoder: bool = False,
        bias: bool = True,
        is_cross_attention: bool = False,
        device=None,
        state_dict=None,
        base_address=None,
        host=None,
    ):
        super().__init__()
        self.host = host
        self.device = device
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if not (self.head_dim * num_heads == self.embed_dim):
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {num_heads})."
            )
        self.scaling = math.sqrt(1 / self.head_dim)
        self.is_decoder = is_decoder

        self.k_proj_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.k_proj.weight"],
            self.device,
            put_on_device=False,
        )
        self.k_proj_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.k_proj.bias"], self.device, put_on_device=False
        )
        self.k_proj = Linear(self.kdim, self.embed_dim, self.k_proj_weight, self.k_proj_bias)

        self.v_proj_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.v_proj.weight"],
            self.device,
            put_on_device=False,
        )
        self.v_proj_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.v_proj.bias"], self.device, put_on_device=False
        )
        self.v_proj = Linear(self.vdim, self.embed_dim, self.v_proj_weight, self.v_proj_bias)

        self.q_proj_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.q_proj.weight"],
            self.device,
            put_on_device=False,
        )
        self.q_proj_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.q_proj.bias"], self.device, put_on_device=False
        )
        self.q_proj = Linear(self.embed_dim, self.embed_dim, self.q_proj_weight, self.q_proj_bias)

        self.out_proj_weight = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.out_proj.weight"],
            self.device,
            put_on_device=False,
        )
        self.out_proj_bias = torch_to_tt_tensor_rm(
            state_dict[f"{base_address}.out_proj.bias"],
            self.device,
            put_on_device=False,
        )
        self.out_proj = Linear(embed_dim, embed_dim, self.out_proj_weight, self.out_proj_bias)

    def _shape(self, tensor: ttnn.Tensor, seq_len: int, bsz: int):
        tensor = fallback_ops.reshape(tensor, bsz, seq_len, self.num_heads, self.head_dim)
        tensor = ttnn.transpose(tensor, 1, -2)
        return tensor

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        key_value_states: Optional[ttnn.Tensor] = None,
        past_key_value: Optional[Tuple[ttnn.Tensor]] = None,
        attention_mask: Optional[ttnn.Tensor] = None,
        layer_head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[ttnn.Tensor, Optional[ttnn.Tensor], Optional[Tuple[ttnn.Tensor]],]:
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        if attention_mask != None:
            attention_mask = torch_to_tt_tensor_rm(attention_mask, self.device, put_on_device=False)

        is_cross_attention = key_value_states is not None
        bsz, tgt_len, embed_dim = hidden_states.shape.with_tile_padding()[1:]

        # get query proj
        query_states = ttnn.multiply(self.q_proj(hidden_states), self.scaling)

        # get key, value proj
        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.k_proj(key_value_states), -1, bsz)
            value_states = self._shape(self.v_proj(key_value_states), -1, bsz)
        elif past_key_value is not None:
            # reuse k, v, self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        else:
            # self_attention
            key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
            value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_states, value_states)

        proj_shape = (1, bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)
        query_states = fallback_ops.reshape(query_states, *proj_shape)

        key_states = fallback_ops.reshape(key_states, *proj_shape)
        value_states = fallback_ops.reshape(value_states, *proj_shape)

        src_len = key_states.shape.with_tile_padding()[2]
        key_states = ttnn.transpose(key_states, -2, -1)
        attn_weights = ttnn.matmul(query_states, key_states)

        if attn_weights.shape.with_tile_padding() != [1, bsz * self.num_heads, tgt_len, src_len]:
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.shape.with_tile_padding()}"
            )

        if attention_mask is not None:
            if attention_mask.shape.with_tile_padding() != [bsz, 1, tgt_len, src_len]:
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.shape.with_tile_padding()}"
                )

            attn_weights = fallback_ops.reshape(attn_weights, bsz, self.num_heads, tgt_len, src_len)

            if attention_mask == None:
                attn_weights = ttnn.add(attn_weights, attention_mask)

            attn_weights = fallback_ops.reshape(attn_weights, 1, bsz * self.num_heads, tgt_len, src_len)

        attn_weights = fallback_ops.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            if layer_head_mask.size() != (self.num_heads,):
                raise ValueError(
                    f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                    f" {layer_head_mask.size()}"
                )
            attn_weights = fallback_ops.reshape(layer_head_mask, 1, -1, 1, 1) * fallback_ops.reshape(
                attn_weights, bsz, self.num_heads, tgt_len, src_len
            )
            attn_weights = fallback_ops.reshape(attn_weights, bsz * self.num_heads, tgt_len, src_len)

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = fallback_ops.reshape(attn_weights, bsz, self.num_heads, tgt_len, src_len)
            attn_weights = fallback_ops.reshape(attn_weights_reshaped, bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_output = ttnn.matmul(attn_weights, value_states)

        if attn_output.shape.with_tile_padding() != [1, bsz * self.num_heads, tgt_len, self.head_dim]:
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.shape.with_tile_padding()}"
            )

        attn_output = fallback_ops.reshape(attn_output, bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = ttnn.transpose(attn_output, 1, -2)
        attn_output = fallback_ops.reshape(attn_output, 1, bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped, past_key_value

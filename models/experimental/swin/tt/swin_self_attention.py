# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import collections.abc

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)
from models.experimental.swin.swin_helper_funcs import linear as TtLinear
from models.experimental.swin.swin_utils import meshgrid
import ttnn
from tt_lib.fallback_ops import fallback_ops


class TtSwinSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        dim,
        num_heads,
        window_size,
        state_dict,
        base_address,
        device,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({dim}) is not a multiple of the number of attention heads ({num_heads})"
            )
        self.device = device
        self.num_attention_heads = num_heads
        self.attention_head_size = int(dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.window_size = (
            window_size if isinstance(window_size, collections.abc.Iterable) else (window_size, window_size)
        )

        self.relative_position_bias_table = torch.zeros(
            (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), num_heads
        )

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(meshgrid([coords_h, coords_w], indexing="ij"))

        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)
        self.query_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.query.weight"], self.device)
        self.query_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.query.bias"], self.device)

        self.key_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.key.weight"], self.device)
        self.key_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.key.bias"], self.device)

        self.value_weight = torch_to_tt_tensor_rm(state_dict[f"{base_address}.value.weight"], self.device)
        self.value_bias = torch_to_tt_tensor_rm(state_dict[f"{base_address}.value.bias"], self.device)

    def const_tensor(self, shape, value):
        return ttnn.full(shape, value)

    def transpose_for_scores(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # x must be 4d originaly
        # 1 is appended to the beggining
        # so create tensor shape by ommiting the first dimension
        new_x_shape = list(x.shape.with_tile_padding())[1:-1] + [
            self.num_attention_heads,
            self.attention_head_size,
        ]
        x = fallback_ops.reshape(x, *new_x_shape)
        x = ttnn.permute(x, (0, 2, 1, 3))
        return x

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        attention_mask: Optional[ttnn.Tensor] = None,
        head_mask: Optional[ttnn.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[ttnn.Tensor]:
        _, batch_size, dim, num_channels = hidden_states.shape.with_tile_padding()
        mixed_query_layer = TtLinear(hidden_states, self.query_weight, self.query_bias)

        key_layer = self.transpose_for_scores(TtLinear(hidden_states, self.key_weight, self.key_bias))
        value_layer = self.transpose_for_scores(TtLinear(hidden_states, self.value_weight, self.value_bias))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer_transposed = ttnn.transpose(key_layer, -2, -1)

        attention_scores = ttnn.matmul(query_layer, key_layer_transposed)

        attention_head_size_tt = self.const_tensor(attention_scores.shape.with_tile_padding(), self.attention_head_size)
        attention_head_size_tt = ttnn.sqrt(attention_head_size_tt)
        attention_head_size_tt = ttnn.reciprocal(attention_head_size_tt)

        attention_scores = ttnn.mul(attention_scores, attention_head_size_tt)
        """
        The index value must be long or byte or bool, hence using pytorch tensor
        """
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = torch_to_tt_tensor_rm(relative_position_bias, self.device, put_on_device=False)
        relative_position_bias = fallback_ops.reshape(
            relative_position_bias,
            -1,
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            1,
        )
        attention_scores = ttnn.permute(attention_scores, (1, 2, 3, 0))
        attention_scores = ttnn.add(
            attention_scores,
            relative_position_bias,
        )

        attention_scores = ttnn.permute(attention_scores, (3, 0, 1, 2))

        attention_scores = tt_to_torch_tensor(attention_scores)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in SwinModel forward() function)
            mask_shape = attention_mask.shape.with_tile_padding()[1]
            """
            Converting attention_scores to 5D, Hence pytorch tensor
            """
            attention_scores = attention_scores.view(
                batch_size // mask_shape,
                mask_shape,
                self.num_attention_heads,
                dim,
                dim,
            )
            attention_scores = attention_scores + tt_to_torch_tensor(attention_mask).unsqueeze(2)
            """
            attention score is 5 D tensor
            """
            attention_scores = torch_to_tt_tensor_rm(
                attention_scores.view(-1, self.num_attention_heads, dim, dim), self.device, put_on_device=False
            )

        # Normalize the attention scores to probabilities.
        attention_probs = fallback_ops.softmax(attention_scores, dim=-1)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = ttnn.matmul(attention_probs, value_layer)
        context_layer = ttnn.permute(context_layer, (0, 2, 1, 3))

        new_context_layer_shape = tuple(context_layer.shape.with_tile_padding())[:-2] + (self.all_head_size,)
        context_layer = fallback_ops.reshape(
            context_layer,
            1,
            new_context_layer_shape[0],
            new_context_layer_shape[1],
            new_context_layer_shape[2],
        )
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs

from pathlib import Path
import sys
f = f"{Path(__file__).parent}"
sys.path.append(f"{f}")
sys.path.append(f"{f}/../")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")
sys.path.append(f"{f}/../../../../..")

import collections.abc
import math
from dataclasses import dataclass
from typing import Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from activations import ACT2FN
from deit_config import DeiTConfig
from helper_funcs import make_linear

import tt_lib
from tt_lib.fallback_ops import fallback_ops



class TtDeiTSelfAttention(nn.Module):
    def __init__(self, config: DeiTConfig(), host, device, state_dict=None, base_address="") -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        query_weight = state_dict[f"{base_address}.query.weight"]
        query_bias = state_dict[f"{base_address}.query.bias"]
        self.query = make_linear(config.hidden_size, self.all_head_size, query_weight, query_bias, device)

        key_weight = state_dict[f"{base_address}.key.weight"]
        key_bias = state_dict[f"{base_address}.key.bias"]
        self.key = make_linear(config.hidden_size, self.all_head_size, key_weight, key_bias, device)

        value_weight = state_dict[f"{base_address}.value.weight"]
        value_bias = state_dict[f"{base_address}.value.bias"]
        self.value  = make_linear(config.hidden_size, self.all_head_size, value_weight, value_bias, device)

    def transpose_for_scores(self, x):
        new_x_shape = x.shape()[-3:-1] +  [self.num_attention_heads, self.attention_head_size]
        x = fallback_ops.reshape(x, new_x_shape[0], new_x_shape[2], new_x_shape[1], new_x_shape[3])
        return x

    def forward(self,hidden_states):

        key = self.key(hidden_states)
        value = self.value(hidden_states)
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        key_layer_transposed = tt_lib.tensor.transpose(key_layer)
        print('q layer shape:', query_layer.shape())
        print('k layer T shape:', key_layer_transposed.shape())

        attention_scores = tt_lib.tensor.bmm(query_layer, key_layer_transposed)
        attention_head_size_sqrt_rec = 1 / math.sqrt(self.attention_head_size)
        tensor_size = attention_scores.shape()
        print('shape shape', tensor_size)

        attention_head_tensor = tt_lib.tensor.Tensor(tensor_size[1]*tensor_size[2]*tensor_size[3]*[attention_head_size_sqrt_rec], tensor_size, tt_lib.tensor.DataType.BFLOAT16, tt_lib.tensor.Layout.ROW_MAJOR)

        attention_scores = tt_lib.tensor.mul(attention_scores , attention_head_tensor)

        # Normalize the attention scores to probabilities.
        attention_probs = fallback_ops.softmax(attention_scores, dim=-1)

        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        print('probs shape:', attention_probs.shape())
        print('value_layer shape:', value_layer.shape())

        context_layer = tt_lib.tensor.bmm(attention_probs, value_layer)
        context_layer_shape = context_layer.shape()
        context_layer = fallback_ops.reshape(context_layer, context_layer_shape[0], context_layer_shape[2], context_layer_shape[1], context_layer_shape[3])

        new_context_layer_shape = [1] + context_layer.shape()[:-2] + [self.all_head_size]
        context_layer = fallback_ops.reshape(context_layer, new_context_layer_shape[0], new_context_layer_shape[1], new_context_layer_shape[2], new_context_layer_shape[3])
        # context_layer = context_layer.view(new_context_layer_shape)
        print('context layer shape', context_layer.shape())

        outputs = context_layer

        return outputs

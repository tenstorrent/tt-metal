from typing import List
import torch
import torch.nn as nn

from models.utility_functions import (
    tt_to_torch_tensor,
    torch_to_tt_tensor_rm,
)

import tt_lib
import tt_lib.fallback_ops as fallback_ops


class TtSqueezeBertSelfAttention(nn.Module):
    def __init__(
        self,
        config,
        cin: int,
        q_groups: int = 1,
        k_groups: int = 1,
        v_groups: int = 1,
        base_address="",
        state_dict=None,
        device=None,
    ) -> None:
        super().__init__()
        self.config = config
        self.cin = cin
        self.q_groups = q_groups
        self.k_groups = k_groups
        self.v_groups = v_groups
        self.base_address = base_address
        self.state_dict = state_dict
        self.device = device

        if self.cin % self.config.num_attention_heads != 0:
            raise ValueError(
                f"cin ({cin}) is not a multiple of the number of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = self.config.num_attention_heads
        self.attention_head_size = int(cin / self.config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Conv1d(
            in_channels=cin,
            out_channels=cin,
            kernel_size=1,
            groups=q_groups,
        )
        self.query.weight = nn.Parameter(
            state_dict[f"{self.base_address}.query.weight"].to(dtype=torch.bfloat16)
        )
        self.query.bias = nn.Parameter(
            state_dict[f"{self.base_address}.query.bias"].to(dtype=torch.bfloat16)
        )

        self.key = nn.Conv1d(
            in_channels=cin,
            out_channels=cin,
            kernel_size=1,
            groups=k_groups,
        )
        self.key.weight = nn.Parameter(
            state_dict[f"{self.base_address}.key.weight"].to(dtype=torch.bfloat16)
        )
        self.key.bias = nn.Parameter(
            state_dict[f"{self.base_address}.key.bias"].to(dtype=torch.bfloat16)
        )

        self.value = nn.Conv1d(
            in_channels=cin,
            out_channels=cin,
            kernel_size=1,
            groups=v_groups,
        )

        self.value.weight = nn.Parameter(
            state_dict[f"{self.base_address}.value.weight"].to(dtype=torch.bfloat16)
        )
        self.value.bias = nn.Parameter(
            state_dict[f"{self.base_address}.value.bias"].to(dtype=torch.bfloat16)
        )

    def const_tensor(self, shape: List[int], value: int):
        return fallback_ops.full(shape, value)

    def transpose_for_scores(self, input: tt_lib.tensor.Tensor, permute_tensor: bool):
        new_shape = (
            input.shape()[0],
            self.num_attention_heads,
            self.attention_head_size,
            input.shape()[-1],
        )

        input = fallback_ops.reshape(input, *new_shape)
        if permute_tensor:
            return tt_lib.tensor.transpose(input)
        return input

    def transpose_output(self, input: tt_lib.tensor.Tensor):
        input = tt_lib.tensor.transpose(input)
        new_shape = (1, input.shape()[0], self.all_head_size, input.shape()[-1])
        return fallback_ops.reshape(input, *new_shape)

    def forward(
        self,
        hidden_states: tt_lib.tensor.Tensor,
        attention_mask: tt_lib.tensor.Tensor,
        output_attentions: bool,
    ):
        hidden_states = tt_to_torch_tensor(hidden_states).squeeze(0)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        mixed_query_layer = torch_to_tt_tensor_rm(
            mixed_query_layer, self.device, put_on_device=False
        )
        mixed_key_layer = torch_to_tt_tensor_rm(
            mixed_key_layer, self.device, put_on_device=False
        )
        mixed_value_layer = torch_to_tt_tensor_rm(
            mixed_value_layer, self.device, put_on_device=False
        )

        query_layer = self.transpose_for_scores(mixed_query_layer, True)
        key_layer = self.transpose_for_scores(mixed_key_layer, False)
        value_layer = self.transpose_for_scores(mixed_value_layer, True)

        attention_score = tt_lib.tensor.bmm(query_layer, key_layer)

        attention_head_size_tt = self.const_tensor(
            attention_score.shape(), self.attention_head_size
        )
        attention_head_size_tt = tt_lib.tensor.sqrt(attention_head_size_tt)
        attention_head_size_tt = tt_lib.tensor.recip(attention_head_size_tt)

        attention_score = tt_lib.tensor.mul(attention_score, attention_head_size_tt)

        attention_score = tt_lib.tensor.transpose(attention_score)
        attention_mask = tt_lib.tensor.transpose(attention_mask)

        attention_score = tt_lib.tensor.transpose(attention_score, 1, 2)
        attention_mask = tt_lib.tensor.transpose(attention_mask, 1, 2)

        attention_score = tt_lib.tensor.bcast(
            attention_score,
            attention_mask,
            tt_lib.tensor.BcastOpMath.ADD,
            tt_lib.tensor.BcastOpDim.HW,
        )
        attention_score = tt_lib.tensor.transpose(attention_score, 1, 2)
        attention_score = tt_lib.tensor.transpose(attention_score)

        attention_probs = fallback_ops.softmax(attention_score, dim=-1)

        context_layer = tt_lib.tensor.bmm(attention_probs, value_layer)

        context_layer = self.transpose_output(context_layer)

        result = {"context_layer": context_layer}
        if output_attentions:
            result["attention_score"] = attention_score
        return result

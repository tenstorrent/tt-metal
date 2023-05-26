import math
from pathlib import Path
import sys
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from loguru import logger

f = f"{Path(__file__).parent}"
sys.path.append(f"{f}/..")
sys.path.append(f"{f}/../..")
sys.path.append(f"{f}/../../..")
sys.path.append(f"{f}/../../../..")

from python_api_testing.models.roberta.roberta_common import (
    torch2tt_tensor,
    tt2torch_tensor,
)
import tt_lib
from tt_lib.fallback_ops import fallback_ops


# Copied from transformers.models.bert.modeling_bert.BertOutput
class TtRobertaOutput(nn.Module):
    def __init__(
        self,
        config,
        state_dict,
        base_address,
        device,
    ):
        super().__init__()
        self.device = device

        self.dense_weight = torch2tt_tensor(
            state_dict[f"{base_address}.dense.weight"], self.device
        )
        self.dense_bias = torch2tt_tensor(
            state_dict[f"{base_address}.dense.bias"], self.device
        )

        gamma = torch2tt_tensor(
            state_dict[f"{base_address}.LayerNorm.weight"], self.device
        )
        beta = torch2tt_tensor(
            state_dict[f"{base_address}.LayerNorm.bias"], self.device
        )

        self.LayerNorm = fallback_ops.LayerNorm(
            gamma, beta, eps=config.layer_norm_eps, normalized_shape=config.hidden_size
        )

        # TODO: Add dropout when supported
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight)
        x = tt_lib.tensor.matmul(x, weight)
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
        return x

    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.linear(hidden_states, self.dense_weight, self.dense_bias)
        # TODO: Add dropout when supported
        # hidden_states = self.dropout(hidden_states)
        hidden_states = tt_lib.tensor.add(hidden_states, input_tensor)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

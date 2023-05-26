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

from transformers import RobertaModel


# Copied from transformers.models.bert.modeling_bert.BertIntermediate
class TtRobertaIntermediate(nn.Module):
    def __init__(
        self, config, state_dict, base_address, device, fall_back_to_torch_gelu=True
    ):
        super().__init__()
        self.device = device

        self.fall_back_to_torch_gelu = fall_back_to_torch_gelu

        self.dense_weight = torch2tt_tensor(
            state_dict[f"{base_address}.dense.weight"], self.device
        )
        self.dense_bias = torch2tt_tensor(
            state_dict[f"{base_address}.dense.bias"], self.device
        )

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight)
        x = tt_lib.tensor.matmul(x, weight)
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
        return x

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(hidden_states, self.dense_weight, self.dense_bias)
        if self.fall_back_to_torch_gelu:
            torch_hidden_states = tt2torch_tensor(hidden_states)
            torch_hidden_states = torch.nn.functional.gelu(torch_hidden_states)
            hidden_states = torch2tt_tensor(torch_hidden_states, self.device)
        else:
            hidden_states = tt_lib.tensor.gelu(hidden_states)
        return hidden_states

import math
from pathlib import Path
import sys
from typing import Optional, Tuple, Union, List
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
from python_api_testing.models.roberta.roberta_model import TtRobertaModel

import tt_lib
from tt_lib.fallback_ops import fallback_ops


class TtRobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, state_dict, base_address, device):
        super().__init__()
        self.device = device

        self.dense_weight = torch2tt_tensor(
            state_dict[f"{base_address}.dense.weight"], self.device
        )
        self.dense_bias = torch2tt_tensor(
            state_dict[f"{base_address}.dense.bias"], self.device
        )

        # TODO: Add when supporting training
        # classifier_dropout = (
        #     config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        # )
        # self.dropout = nn.Dropout(classifier_dropout)

        self.out_proj_weight = torch2tt_tensor(
            state_dict[f"{base_address}.out_proj.weight"], self.device
        )
        self.out_proj_bias = torch2tt_tensor(
            state_dict[f"{base_address}.out_proj.bias"], self.device
        )

    def linear(self, x, weight, bias):
        weight = tt_lib.tensor.transpose(weight)
        x = tt_lib.tensor.matmul(x, weight)
        x = tt_lib.tensor.bcast(
            x, bias, tt_lib.tensor.BcastOpMath.ADD, tt_lib.tensor.BcastOpDim.H
        )
        return x

    def forward(self, features, **kwargs):
        torch_features = tt2torch_tensor(features).squeeze(0)
        torch_x = torch_features[:, 0, :]  # take <s> token (equiv. to [CLS])

        x = torch2tt_tensor(torch_x, self.device)
        # x = self.dropout(x)
        x = self.linear(x, self.dense_weight, self.dense_bias)
        x = tt_lib.tensor.tanh(x)
        # x = torch.tanh(x)

        # x = self.dropout(x)
        x = self.linear(x, self.out_proj_weight, self.out_proj_bias)

        return x

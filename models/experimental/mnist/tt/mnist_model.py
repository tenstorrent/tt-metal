# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import torch
import ttnn

from tt_lib.fused_ops.softmax import softmax
from tt_lib.fallback_ops import fallback_ops
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from models.experimental.mnist.reference.mnist import MnistModel


class TtMnistModel(torch.nn.Module):
    def __init__(self, device, state_dict):
        super().__init__()
        self.device = device

        # Extract params from state dict
        self.fc1_weight = torch2tt_tensor(state_dict["fc1.weight"], device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.fc1_bias = torch2tt_tensor(state_dict["fc1.bias"], device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        self.fc2_weight = torch2tt_tensor(state_dict["fc2.weight"], device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.fc2_bias = torch2tt_tensor(state_dict["fc2.bias"], device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        self.fc3_weight = torch2tt_tensor(state_dict["fc3.weight"], device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)
        self.fc3_bias = torch2tt_tensor(state_dict["fc3.bias"], device, tt_layout=ttnn.ROW_MAJOR_LAYOUT)

        self.fc1_weight = ttnn.transpose(self.fc1_weight, -2, -1)
        self.fc2_weight = ttnn.transpose(self.fc2_weight, -2, -1)
        self.fc3_weight = ttnn.transpose(self.fc3_weight, -2, -1)

    def forward(self, x):
        # ttnn.reshape_on_device throws an assertion RuntimeError: TT_ASSERT @ tt_eager/tt_dnn/op_library/reshape/reshape_op.cpp:295: input_tensor_a.shape.with_tile_padding()[3] % TILE_WIDTH == 0 && W % TILE_WIDTH == 0 info:
        # Operand/target width must be a multiple of 32. So using fallback_ops.reshape.
        x = fallback_ops.reshape(x, x.shape.with_tile_padding()[0], 1, 1, 784)

        x = ttnn.matmul(x, self.fc1_weight)
        x = ttnn.add(x, self.fc1_bias)
        x = ttnn.relu(x)

        x = ttnn.matmul(x, self.fc2_weight)
        x = ttnn.add(x, self.fc2_bias)
        x = ttnn.relu(x)

        x = ttnn.matmul(x, self.fc3_weight)
        x = ttnn.add(x, self.fc3_bias)
        x = ttnn.relu(x)

        x = softmax(x)
        # x = fallback_ops.softmax(x, -1)
        return x


def _mnist_model(device, state_dict) -> TtMnistModel:
    return TtMnistModel(device, state_dict)


def mnist_model(device, model_location_generator) -> TtMnistModel:
    # Trained to 68% accuracy in modelzoo
    state_dict = torch.load(model_location_generator("mnist_model.pt", model_subdir="mnist"))
    tt_model = _mnist_model(device, state_dict)
    pt_model = MnistModel(state_dict)

    return tt_model, pt_model

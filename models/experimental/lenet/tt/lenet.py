# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn


from models.experimental.lenet.lenet_utils import load_torch_lenet
from models.utility_functions import torch2tt_tensor, tt2torch_tensor
from tt_lib.fallback_ops import fallback_ops


class TtLeNet5(nn.Module):
    def __init__(self, num_classes, device, state_dict):
        super().__init__()
        self.device = device

        conv1_weight = state_dict["layer1.0.weight"]
        conv1_bias = state_dict["layer1.0.bias"]
        self.conv1 = fallback_ops.Conv2d(conv1_weight, conv1_bias, 1, 6, kernel_size=5, stride=1, padding=0)

        batch_norm1_weight = state_dict["layer1.1.weight"]
        batch_norm1_bias = state_dict["layer1.1.bias"]
        batch_norm1_running_mean = state_dict["layer1.1.running_mean"]
        batch_norm1_running_var = state_dict["layer1.1.running_var"]
        batch_norm1_num_batches_tracked = state_dict["layer1.1.num_batches_tracked"]
        self.batch_norm1 = fallback_ops.BatchNorm2d(
            batch_norm1_weight,
            batch_norm1_bias,
            batch_norm1_running_mean,
            batch_norm1_running_var,
            batch_norm1_num_batches_tracked,
            6,
            eps=0.00001,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

        self.relu1 = ttnn.relu

        self.maxp1 = fallback_ops.MaxPool2d(kernel_size=2, stride=2)

        conv2_weight = state_dict["layer2.0.weight"]
        conv2_bias = state_dict["layer2.0.bias"]
        self.conv2 = fallback_ops.Conv2d(conv2_weight, conv2_bias, 6, 16, kernel_size=5, stride=1, padding=0)

        batch_norm2_weight = state_dict["layer2.1.weight"]
        batch_norm2_bias = state_dict["layer2.1.bias"]
        batch_norm2_running_mean = state_dict["layer2.1.running_mean"]
        batch_norm2_running_var = state_dict["layer2.1.running_var"]
        batch_norm2_num_batches_tracked = state_dict["layer2.1.num_batches_tracked"]
        self.batch_norm2 = fallback_ops.BatchNorm2d(
            batch_norm2_weight,
            batch_norm2_bias,
            batch_norm2_running_mean,
            batch_norm2_running_var,
            batch_norm2_num_batches_tracked,
            16,
            eps=0.00001,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
        )

        self.relu2 = ttnn.relu

        self.maxp2 = fallback_ops.MaxPool2d(kernel_size=2, stride=2)

        fc_weights = state_dict[f"fc.weight"]
        self.fc_weights = torch2tt_tensor(
            fc_weights.reshape(list((1, 1) + fc_weights.shape)),
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        fc_bias = state_dict[f"fc.bias"]
        self.fc_bias = torch2tt_tensor(
            fc_bias.reshape(list((1, 1, 1) + fc_bias.shape)),
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        fc1_weights = state_dict[f"fc1.weight"]
        self.fc1_weights = torch2tt_tensor(
            fc1_weights.reshape(list((1, 1) + fc1_weights.shape)),
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        fc1_bias = state_dict[f"fc1.bias"]
        self.fc1_bias = torch2tt_tensor(
            fc1_bias.reshape(list((1, 1, 1) + fc1_bias.shape)),
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )

        fc2_weights = state_dict[f"fc2.weight"]
        self.fc2_weights = torch2tt_tensor(
            fc2_weights.reshape(list((1, 1) + fc2_weights.shape)),
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )
        fc2_bias = state_dict[f"fc2.bias"]
        self.fc2_bias = torch2tt_tensor(
            fc2_bias.reshape(list((1, 1, 1) + fc2_bias.shape)),
            self.device,
            ttnn.ROW_MAJOR_LAYOUT,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        out = self.conv1(x)  # HOST (fallback)

        out = self.batch_norm1(out)  # HOST (fallback)

        out = self.relu1(out)  # DEVICE

        out = self.maxp1(out)  # HOST (fallback)

        out = self.conv2(out)  # HOST (fallback)

        out = self.batch_norm2(out)  # HOST (fallback)

        out = self.relu2(out)  # DEVICE

        out = self.maxp2(out)  # HOST (fallback)

        # using fallback since last dimension of tensor is not divisible by 2
        out_shape = out.shape.with_tile_padding()
        out = fallback_ops.reshape(
            out, out_shape[0], 1, 1, out_shape[1] * out_shape[2] * out_shape[3]
        )  # HOST (fallback)

        # fc
        weight_T = ttnn.transpose(self.fc_weights, -2, -1)
        output = ttnn.matmul(out, weight_T)
        out = ttnn.add(output, self.fc_bias)
        # relu 2
        out = self.relu2(out)

        # fc1
        weight_T = ttnn.transpose(self.fc1_weights, -2, -1)
        output = ttnn.matmul(out, weight_T)
        out = ttnn.add(
            output,
            self.fc1_bias,
        )

        # relu 2
        out = self.relu2(out)

        # fc2
        weight_T = ttnn.transpose(self.fc2_weights, -2, -1)
        output = ttnn.matmul(out, weight_T)
        out = ttnn.add(output, self.fc2_bias)

        return out


def _lenet5(num_classes, device, state_dict) -> TtLeNet5:
    return TtLeNet5(num_classes, device, state_dict)


def lenet5(num_classes, device, model_location_generator) -> TtLeNet5:
    pt_model_path = model_location_generator("model.pt", model_subdir="LeNet")
    _, state_dict = load_torch_lenet(pt_model_path, num_classes)
    model = _lenet5(num_classes, device, state_dict)
    return model

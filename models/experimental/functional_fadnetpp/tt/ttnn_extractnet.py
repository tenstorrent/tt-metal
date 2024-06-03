# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


import torch
import ttnn
from tt_lib.fallback_ops import fallback_ops
from models.experimental.functional_fadnetpp.tt.ttnn_resblock import TtResBlock
from models.utility_functions import (
    torch_to_tt_tensor_rm,
)
import tt_lib


class ttExtractNet:
    def output_preprocessing(self, output_tensor, height, width, device):
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = torch.permute(output_tensor, (0, 3, 1, 2))
        output_tensor = torch.reshape(
            output_tensor,
            [
                output_tensor.shape[0],
                output_tensor.shape[1],
                height,
                width,
            ],
        )
        output_tensor = torch_to_tt_tensor_rm(output_tensor, device, put_on_device=True)
        return output_tensor

    def input_preprocessing(self, input_tensor, device):
        input_tensor = ttnn.to_torch(input_tensor)
        input_tensor = torch.permute(input_tensor, (0, 2, 3, 1))
        input_tensor = torch.reshape(
            input_tensor,
            (input_tensor.shape[0], 1, input_tensor.shape[1] * input_tensor.shape[2], input_tensor.shape[3]),
        )
        input_tensor = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
        return input_tensor

    def __init__(self, parameters, resBlock=True) -> None:
        self.resBlock = resBlock
        self.conv1a = fallback_ops.Conv2d(
            parameters.conv1a["weight"], parameters.conv1a["bias"], 3, 32, 7, 2, padding=3, bias=True
        )
        self.conv1b = fallback_ops.Conv2d(
            parameters.conv1b["weight"], parameters.conv1b["bias"], 3, 32, 7, 2, padding=3, bias=True
        )
        if self.resBlock:
            self.conv2 = TtResBlock(parameters.conv2, 32, 64, stride=2)
            self.conv3 = TtResBlock(parameters.conv3, 64, 128, stride=2)
        else:
            self.conv2 = parameters.conv2
            self.conv3 = parameters.conv3

    def __call__(self, device, input_tensor1, input_tensor2):
        img_left = input_tensor1
        img_left = self.output_preprocessing(img_left, 960, 576, device)
        conv1_l = self.conv1a(img_left)
        conv1_l = self.input_preprocessing(conv1_l, device)
        conv1_l = conv1_l.to(device)
        conv1_l = ttnn.relu(conv1_l, memory_config=ttnn.L1_MEMORY_CONFIG)

        img_right = input_tensor2
        img_right = self.output_preprocessing(img_right, 960, 576, device)
        conv1_r = self.conv1b(img_right)
        conv1_r = self.input_preprocessing(conv1_r, device)
        conv1_r = conv1_r.to(device)
        conv1_r = ttnn.relu(conv1_r, memory_config=ttnn.L1_MEMORY_CONFIG)

        conv2_l = self.conv2(device, conv1_l)

        conv3_l = self.conv3(device, conv2_l)

        conv2_r = self.conv2(device, conv1_r)
        conv3_r = self.conv3(device, conv2_r)

        return (
            ttnn.from_device(conv1_l),
            ttnn.from_device(conv2_l),
            ttnn.from_device(conv3_l),
            ttnn.from_device(conv3_r),
        )

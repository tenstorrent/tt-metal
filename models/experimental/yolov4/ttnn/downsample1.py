# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn.common import Conv, fold_bn_to_conv_weights_bias_torch
import torch.nn.functional as F


class Down1:
    def __init__(self, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(
            torch_model,
            "down1.conv1",
            [1, 640, 640, 3],
            (1, 1, 1, 1),
            act_block_h=240,
            auto_shard=True,
        )

        # self.conv2_weights, self.conv2_bias = fold_bn_to_conv_weights_bias_torch(torch_model, "down1.conv2")

        self.conv2 = Conv(
            torch_model,
            "down1.conv2",
            [1, 640, 640, 32],
            (2, 2, 1, 1),
            auto_shard=True,
            # act_block_h=512,
            # enable_split_reader=True,
            # enable_act_double_buffer=True,
        )
        self.conv3 = Conv(
            torch_model,
            "down1.conv3",
            [1, 320, 320, 64],
            (1, 1, 0, 0),
            deallocate=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
            auto_shard=True,
        )
        self.conv4 = Conv(
            torch_model,
            "down1.conv4",
            [1, 320, 320, 64],
            (1, 1, 0, 0),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv5 = Conv(
            torch_model,
            "down1.conv5",
            [1, 320, 320, 64],
            (1, 1, 0, 0),
            deallocate=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
            auto_shard=True,
        )
        self.conv6 = Conv(
            torch_model,
            "down1.conv6",
            [1, 320, 320, 32],
            (1, 1, 1, 1),
            act_block_h=240,
            auto_shard=True,
        )
        self.conv7 = Conv(
            torch_model,
            "down1.conv7",
            [1, 320, 320, 64],
            (1, 1, 0, 0),
            enable_split_reader=True,
            enable_act_double_buffer=True,
            auto_shard=True,
        )
        self.conv8 = Conv(
            torch_model,
            "down1.conv8",
            [1, 320, 320, 128],
            (1, 1, 0, 0),
            # auto_shard=True,
        )

    def __call__(self, device, input_tensor):
        output_tensor = self.conv1(device, input_tensor)
        output_tensor = ttnn.mish(output_tensor)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_split = self.conv2(device, output_tensor)
        ttnn.deallocate(output_tensor)
        output_tensor_split = ttnn.mish(output_tensor_split)

        output_tensor_left = self.conv3(device, output_tensor_split)
        output_tensor_left = ttnn.mish(output_tensor_left)

        output_tensor_split_2 = self.conv4(device, output_tensor_split)
        ttnn.deallocate(output_tensor_split)

        output_tensor_split_2 = ttnn.mish(output_tensor_split_2)
        output_tensor = self.conv5(device, output_tensor_split_2)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.conv6(device, output_tensor)

        output_tensor = ttnn.mish(output_tensor)

        output_tensor = output_tensor_split_2 + output_tensor

        ttnn.deallocate(output_tensor_split_2)
        if output_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)

        output_tensor = self.conv7(device, output_tensor)

        output_tensor = ttnn.mish(output_tensor)

        if output_tensor.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        if output_tensor_left.get_layout() != ttnn.ROW_MAJOR_LAYOUT:
            output_tensor_left = ttnn.to_layout(output_tensor_left, ttnn.ROW_MAJOR_LAYOUT)

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if output_tensor_left.is_sharded():
            output_tensor_left = ttnn.sharded_to_interleaved(output_tensor_left, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat([output_tensor, output_tensor_left], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(output_tensor_left)

        output_tensor = self.conv8(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        return output_tensor

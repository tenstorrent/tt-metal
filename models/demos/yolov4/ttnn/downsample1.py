# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.demos.yolov4.ttnn.common import Conv
from tests.ttnn.ttnn_utility_fuction import get_shard_grid_from_num_cores
from tests.ttnn.utils_for_testing import assert_with_pcc, check_with_pcc_without_tensor_printout


class Down1:
    def __init__(self, device, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(
            device,
            torch_model,
            "down1.conv1",
            [1, 320, 320, 3],
            (1, 1, 1, 1),
            act_block_h=128,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv2 = Conv(
            device,
            torch_model,
            "down1.conv2",
            [1, 320, 320, 32],
            (2, 2, 1, 1),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv3 = Conv(
            device,
            torch_model,
            "down1.conv3",
            [1, 160, 160, 64],
            (1, 1, 0, 0),
            deallocate=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv4 = Conv(
            device,
            torch_model,
            "down1.conv4",
            [1, 160, 160, 64],
            (1, 1, 0, 0),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv5 = Conv(
            device,
            torch_model,
            "down1.conv5",
            [1, 160, 160, 64],
            (1, 1, 0, 0),
            deallocate=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv6 = Conv(
            device,
            torch_model,
            "down1.conv6",
            [1, 160, 160, 32],
            (1, 1, 1, 1),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv7 = Conv(
            device,
            torch_model,
            "down1.conv7",
            [1, 160, 160, 64],
            (1, 1, 0, 0),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv8 = Conv(
            device,
            torch_model,
            "down1.conv8",
            [1, 160, 160, 128],
            (1, 1, 0, 0),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]

    def __call__(self, input_tensor):
        output_tensor = self.conv1(input_tensor)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor_split = self.conv2(output_tensor)
        output_tensor_split = ttnn.mish(output_tensor_split)

        output_tensor_left = self.conv3(output_tensor_split)
        output_tensor_left = ttnn.mish(output_tensor_left)

        output_tensor_split_2 = self.conv4(output_tensor_split)
        output_tensor_split_2 = ttnn.mish(output_tensor_split_2)
        output_tensor = self.conv5(output_tensor_split_2)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.conv6(output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = output_tensor_split_2 + output_tensor

        ttnn.deallocate(output_tensor_split_2)
        output_tensor = self.conv7(output_tensor)
        output_tensor = ttnn.mish(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_left = ttnn.to_layout(output_tensor_left, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_sharded_memory_config = ttnn.create_sharded_memory_config(
            [output_tensor.memory_config().shard_spec.shape[0], 2 * output_tensor.memory_config().shard_spec.shape[1]],
            core_grid=output_tensor_left.memory_config().shard_spec.grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            use_height_and_width_as_shard_shape=True,
        )
        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_left], dim=3, memory_config=output_sharded_memory_config
        )
        ttnn.deallocate(output_tensor_left)

        output_tensor = self.conv8(output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        return output_tensor

    def __str__(self) -> str:
        this_str = ""
        index = 1
        for conv in self.convs:
            this_str += str(index) + " " + str(conv)
            this_str += " \n"
            index += 1
        return this_str

# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn.common import Conv


class Down2:
    def __init__(self, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(
            torch_model,
            "down2.conv1",
            [1, 320, 320, 64],
            (2, 2, 1, 1),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv2 = Conv(
            torch_model,
            "down2.conv2",
            [1, 160, 160, 128],
            (1, 1, 0, 0),
            deallocate=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv3 = Conv(
            torch_model,
            "down2.conv3",
            [1, 160, 160, 128],
            (1, 1, 0, 0),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.conv4 = Conv(
            torch_model,
            "down2.conv4",
            [1, 160, 160, 64],
            (1, 1, 0, 0),
            deallocate=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )

        self.res1_conv1 = Conv(
            torch_model,
            "down2.resblock.module_list.0.0",
            [1, 160, 160, 64],
            (1, 1, 0, 0),
            deallocate=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.res1_conv2 = Conv(
            torch_model,
            "down2.resblock.module_list.0.1",
            [1, 160, 160, 64],
            (1, 1, 1, 1),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.res2_conv1 = Conv(
            torch_model,
            "down2.resblock.module_list.1.0",
            [1, 160, 160, 64],
            (1, 1, 0, 0),
            deallocate=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )
        self.res2_conv2 = Conv(
            torch_model,
            "down2.resblock.module_list.1.1",
            [1, 160, 160, 64],
            (1, 1, 1, 1),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )

        self.conv5 = Conv(
            torch_model,
            "down2.conv5",
            [1, 160, 160, 128],
            (1, 1, 0, 0),
            enable_split_reader=True,
            enable_act_double_buffer=True,
        )

    def __call__(self, device, input_tensor):
        output_tensor_split = self.conv1(device, input_tensor)
        output_tensor_split = ttnn.mish(output_tensor_split)
        output_tensor_left = self.conv2(device, output_tensor_split)
        output_tensor_left = ttnn.mish(output_tensor_left)

        res1_split = self.conv3(device, output_tensor_split)
        res1_split = ttnn.mish(res1_split)

        output_tensor = self.res1_conv1(device, res1_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res1_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res2_split = res1_split + output_tensor
        ttnn.deallocate(res1_split)

        output_tensor = self.res2_conv1(device, res2_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res2_conv2(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = res2_split + output_tensor

        ttnn.deallocate(res2_split)

        output_tensor = self.conv4(device, output_tensor)
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

        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = self.conv5(device, output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        return output_tensor

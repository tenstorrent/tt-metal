# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov4.tt.common import Conv


def sharded_concat(input_tensors, num_cores=64, dim=3):  # expected input tensors to be in fp16, RM, same (h*w)
    shard_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))})
    in_shard_width = input_tensors[0].shape[-1]
    shard_height = ((input_tensors[0].shape[1] * input_tensors[0].shape[2]) + num_cores - 1) // num_cores
    input_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, in_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    out_shard_width = 0
    for i in range(len(input_tensors)):
        out_shard_width += input_tensors[i].shape[-1]
        input_tensors[i] = ttnn.to_memory_config(input_tensors[i], input_sharded_memory_config)
    output_sharded_memory_config = ttnn.create_sharded_memory_config(
        (shard_height, out_shard_width),
        core_grid=shard_grid,
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )
    output = ttnn.concat(input_tensors, dim, memory_config=output_sharded_memory_config)

    return output


class Down1:
    def __init__(self, device, parameters, conv_args) -> None:
        self.parameters = parameters
        self.conv1 = Conv(
            device,
            conv_args.c1,
            parameters.c1,
        )

        self.conv2 = Conv(
            device,
            conv_args.c2,
            parameters.c2,
        )
        self.conv3 = Conv(
            device,
            conv_args.c3,
            parameters.c3,
        )
        self.conv4 = Conv(
            device,
            conv_args.c4,
            parameters.c4,
        )
        self.conv5 = Conv(
            device,
            conv_args.c5,
            parameters.c5,
        )
        self.conv6 = Conv(
            device,
            conv_args.c6,
            parameters.c6,
        )
        self.conv7 = Conv(
            device,
            conv_args.c7,
            parameters.c7,
        )
        self.conv8 = Conv(
            device,
            conv_args.c8,
            parameters.c8,
        )
        self.convs = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]

    def __call__(self, input_tensor):
        output_tensor = self.conv1(input_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)

        if self.parameters.resolution[0] != 320:
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor_split = self.conv2(output_tensor)[0]
        ttnn.deallocate(output_tensor)
        output_tensor_split = ttnn.mish(output_tensor_split)

        output_tensor_left = self.conv3(output_tensor_split)[0]
        output_tensor_left = ttnn.mish(output_tensor_left)

        output_tensor_split_2 = self.conv4(output_tensor_split)[0]
        ttnn.deallocate(output_tensor_split)
        output_tensor_split_2 = ttnn.mish(output_tensor_split_2)
        output_tensor = self.conv5(output_tensor_split_2)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.conv6(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = output_tensor_split_2 + output_tensor

        ttnn.deallocate(output_tensor_split_2)
        if self.parameters.resolution[0] != 320:
            output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = self.conv7(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_left = ttnn.to_layout(output_tensor_left, layout=ttnn.ROW_MAJOR_LAYOUT)

        if self.parameters.resolution[0] == 320:
            output_sharded_memory_config = ttnn.create_sharded_memory_config(
                [
                    output_tensor.memory_config().shard_spec.shape[0],
                    2 * output_tensor.memory_config().shard_spec.shape[1],
                ],
                core_grid=output_tensor_left.memory_config().shard_spec.grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
            output_tensor = ttnn.concat(
                [output_tensor, output_tensor_left], dim=3, memory_config=output_sharded_memory_config
            )
        else:
            output_tensor = sharded_concat([output_tensor, output_tensor_left])

        ttnn.deallocate(output_tensor_left)

        output_tensor = self.conv8(output_tensor)[0]
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

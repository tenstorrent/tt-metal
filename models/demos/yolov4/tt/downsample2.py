# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov4.common import create_conv2d_config
from models.tt_cnn.tt.builder import TtConv2d


class Down2:
    def __init__(self, device, parameters, conv_args) -> None:
        self.parameters = parameters
        self.conv1 = TtConv2d(
            create_conv2d_config(conv_args.c1, parameters.c1.weight, parameters.c1.bias),
            device=device,
        )
        self.conv2 = TtConv2d(
            create_conv2d_config(conv_args.c2, parameters.c2.weight, parameters.c2.bias),
            device=device,
        )
        self.conv3 = TtConv2d(
            create_conv2d_config(conv_args.c3, parameters.c3.weight, parameters.c3.bias),
            device=device,
        )
        self.conv4 = TtConv2d(
            create_conv2d_config(conv_args.c4, parameters.c4.weight, parameters.c4.bias),
            device=device,
        )

        self.res1_conv1 = TtConv2d(
            create_conv2d_config(conv_args.res["0"], parameters.res["0"]["0"].weight, parameters.res["0"]["0"].bias),
            device=device,
        )
        self.res1_conv2 = TtConv2d(
            create_conv2d_config(conv_args.res["3"], parameters.res["0"]["3"].weight, parameters.res["0"]["3"].bias),
            device=device,
        )
        self.res2_conv1 = TtConv2d(
            create_conv2d_config(conv_args.res[0], parameters.res["1"]["0"].weight, parameters.res["1"]["0"].bias),
            device=device,
        )
        self.res2_conv2 = TtConv2d(
            create_conv2d_config(conv_args.res[3], parameters.res["1"]["3"].weight, parameters.res["1"]["3"].bias),
            device=device,
        )

        self.conv5 = TtConv2d(
            create_conv2d_config(conv_args.c5, parameters.c5.weight, parameters.c5.bias),
            device=device,
        )

    def __call__(self, input_tensor):
        output_tensor_split = self.conv1(input_tensor)
        output_tensor_split = ttnn.hardmish(output_tensor_split)
        output_tensor_left = self.conv2(output_tensor_split)
        output_tensor_left = ttnn.hardmish(output_tensor_left)

        res1_split = self.conv3(output_tensor_split)
        ttnn.deallocate(output_tensor_split)
        res1_split = ttnn.hardmish(res1_split)

        output_tensor = self.res1_conv1(res1_split)
        output_tensor = ttnn.hardmish(output_tensor)
        output_tensor = self.res1_conv2(output_tensor)
        output_tensor = ttnn.hardmish(output_tensor)
        res2_split = res1_split + output_tensor
        ttnn.deallocate(res1_split)

        output_tensor = self.res2_conv1(res2_split)
        output_tensor = ttnn.hardmish(output_tensor)
        output_tensor = self.res2_conv2(output_tensor)
        output_tensor = ttnn.hardmish(output_tensor)
        output_tensor = res2_split + output_tensor

        ttnn.deallocate(res2_split)

        output_tensor = self.conv4(output_tensor)
        output_tensor = ttnn.hardmish(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_left = ttnn.to_layout(output_tensor_left, layout=ttnn.ROW_MAJOR_LAYOUT)

        if self.parameters.resolution[0] == 320:
            output_sharded_memory_config = ttnn.create_sharded_memory_config(
                [128, 128],
                core_grid=output_tensor_left.memory_config().shard_spec.grid,
                strategy=ttnn.ShardStrategy.HEIGHT,
                use_height_and_width_as_shard_shape=True,
            )
        else:
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
        ttnn.deallocate(output_tensor_left)

        output_tensor = self.conv5(output_tensor)
        output_tensor = ttnn.hardmish(output_tensor)
        return output_tensor

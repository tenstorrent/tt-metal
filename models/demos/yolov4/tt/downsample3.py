# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov4.tt.common import Conv


class Down3:
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

        self.res1_conv1 = Conv(
            device,
            conv_args.res["0"],
            parameters.res["0"]["0"],
        )
        self.res1_conv2 = Conv(
            device,
            conv_args.res["3"],
            parameters.res["0"]["3"],
        )
        self.res2_conv1 = Conv(
            device,
            conv_args.res["0"],
            parameters.res["1"]["0"],
        )
        self.res2_conv2 = Conv(
            device,
            conv_args.res["3"],
            parameters.res["1"]["3"],
        )
        self.res3_conv1 = Conv(
            device,
            conv_args.res["0"],
            parameters.res["2"]["0"],
        )
        self.res3_conv2 = Conv(
            device,
            conv_args.res["3"],
            parameters.res["2"]["3"],
        )
        self.res4_conv1 = Conv(
            device,
            conv_args.res["0"],
            parameters.res["3"]["0"],
        )
        self.res4_conv2 = Conv(
            device,
            conv_args.res["3"],
            parameters.res["3"]["3"],
        )
        self.res5_conv1 = Conv(
            device,
            conv_args.res["0"],
            parameters.res["4"]["0"],
        )
        self.res5_conv2 = Conv(
            device,
            conv_args.res["3"],
            parameters.res["4"]["3"],
        )
        self.res6_conv1 = Conv(
            device,
            conv_args.res["0"],
            parameters.res["5"]["0"],
        )
        self.res6_conv2 = Conv(
            device,
            conv_args.res["3"],
            parameters.res["5"]["3"],
        )
        self.res7_conv1 = Conv(
            device,
            conv_args.res["0"],
            parameters.res["6"]["0"],
        )
        self.res7_conv2 = Conv(
            device,
            conv_args.res["3"],
            parameters.res["6"]["3"],
        )
        self.res8_conv1 = Conv(
            device,
            conv_args.res["0"],
            parameters.res["7"]["0"],
        )
        self.res8_conv2 = Conv(
            device,
            conv_args.res["3"],
            parameters.res["7"]["3"],
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

    def __call__(self, input_tensor):
        output_tensor_split = self.conv1(input_tensor)[0]
        output_tensor_split = ttnn.mish(output_tensor_split)
        output_tensor_left = self.conv2(output_tensor_split)[0]
        output_tensor_left = ttnn.mish(output_tensor_left)

        res1_split = self.conv3(output_tensor_split)[0]
        ttnn.deallocate(output_tensor_split)
        res1_split = ttnn.mish(res1_split)

        output_tensor = self.res1_conv1(res1_split)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res1_conv2(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        res2_split = res1_split + output_tensor
        ttnn.deallocate(res1_split)

        output_tensor = self.res2_conv1(res2_split)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res2_conv2(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        res3_split = res2_split + output_tensor

        ttnn.deallocate(res2_split)

        output_tensor = self.res3_conv1(res3_split)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res3_conv2(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        res4_split = res3_split + output_tensor

        ttnn.deallocate(res3_split)

        output_tensor = self.res4_conv1(res4_split)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res4_conv2(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        res5_split = res4_split + output_tensor

        ttnn.deallocate(res4_split)

        output_tensor = self.res5_conv1(res5_split)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res5_conv2(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        res6_split = res5_split + output_tensor

        ttnn.deallocate(res5_split)

        output_tensor = self.res6_conv1(res6_split)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res6_conv2(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        res7_split = res6_split + output_tensor

        ttnn.deallocate(res6_split)

        output_tensor = self.res7_conv1(res7_split)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res7_conv2(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        res8_split = res7_split + output_tensor

        ttnn.deallocate(res7_split)

        output_tensor = self.res8_conv1(res8_split)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res8_conv2(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = res8_split + output_tensor

        ttnn.deallocate(res8_split)

        output_tensor = self.conv4(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)

        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_left = ttnn.to_layout(output_tensor_left, layout=ttnn.ROW_MAJOR_LAYOUT)
        if self.parameters.resolution[0] == 320:
            output_sharded_memory_config = ttnn.create_sharded_memory_config(
                [32, 256],
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

        output_tensor = self.conv5(output_tensor)[0]
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

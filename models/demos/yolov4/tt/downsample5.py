# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov4.tt.common import Conv


class Down5:
    def __init__(self, device, parameters, conv_args) -> None:
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
        output_tensor = res4_split + output_tensor

        ttnn.deallocate(res4_split)

        output_tensor = self.conv4(output_tensor)[0]
        output_tensor = ttnn.mish(output_tensor)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_left = ttnn.sharded_to_interleaved(output_tensor_left, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_left], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
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

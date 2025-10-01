# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov4.common import create_conv2d_config
from models.tt_cnn.tt.builder import TtConv2d


class Down5:
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

        self.res1_conv1 = TtConv2d(
            create_conv2d_config(conv_args.res["0"], parameters.res["0"]["0"].weight, parameters.res["0"]["0"].bias),
            device=device,
        )
        self.res1_conv2 = TtConv2d(
            create_conv2d_config(conv_args.res["3"], parameters.res["0"]["3"].weight, parameters.res["0"]["3"].bias),
            device=device,
        )
        self.res2_conv1 = TtConv2d(
            create_conv2d_config(conv_args.res["0"], parameters.res["1"]["0"].weight, parameters.res["1"]["0"].bias),
            device=device,
        )
        self.res2_conv2 = TtConv2d(
            create_conv2d_config(conv_args.res["3"], parameters.res["1"]["3"].weight, parameters.res["1"]["3"].bias),
            device=device,
        )
        self.res3_conv1 = TtConv2d(
            create_conv2d_config(conv_args.res["0"], parameters.res["2"]["0"].weight, parameters.res["2"]["0"].bias),
            device=device,
        )
        self.res3_conv2 = TtConv2d(
            create_conv2d_config(conv_args.res["3"], parameters.res["2"]["3"].weight, parameters.res["2"]["3"].bias),
            device=device,
        )
        self.res4_conv1 = TtConv2d(
            create_conv2d_config(conv_args.res["0"], parameters.res["3"]["0"].weight, parameters.res["3"]["0"].bias),
            device=device,
        )
        self.res4_conv2 = TtConv2d(
            create_conv2d_config(conv_args.res["3"], parameters.res["3"]["3"].weight, parameters.res["3"]["3"].bias),
            device=device,
        )

        self.conv4 = TtConv2d(
            create_conv2d_config(conv_args.c4, parameters.c4.weight, parameters.c4.bias),
            device=device,
        )

        self.conv5 = TtConv2d(
            create_conv2d_config(conv_args.c5, parameters.c5.weight, parameters.c5.bias),
            device=device,
        )

    def __call__(self, input_tensor):
        output_tensor_split = self.conv1(input_tensor)
        output_tensor_split = ttnn.mish(output_tensor_split)
        output_tensor_left = self.conv2(output_tensor_split)
        output_tensor_left = ttnn.mish(output_tensor_left)

        res1_split = self.conv3(output_tensor_split)
        res1_split = ttnn.mish(res1_split)

        output_tensor = self.res1_conv1(res1_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res1_conv2(output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res2_split = res1_split + output_tensor
        ttnn.deallocate(res1_split)

        output_tensor = self.res2_conv1(res2_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res2_conv2(output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res3_split = res2_split + output_tensor

        ttnn.deallocate(res2_split)

        output_tensor = self.res3_conv1(res3_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res3_conv2(output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        res4_split = res3_split + output_tensor

        ttnn.deallocate(res3_split)

        output_tensor = self.res4_conv1(res4_split)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = self.res4_conv2(output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        output_tensor = res4_split + output_tensor

        ttnn.deallocate(res4_split)

        output_tensor = self.conv4(output_tensor)
        output_tensor = ttnn.mish(output_tensor)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_left = ttnn.sharded_to_interleaved(output_tensor_left, ttnn.L1_MEMORY_CONFIG)
        # if self.parameters.resolution[0] == 640:
        #     output_tensor = ttnn.add(output_tensor, 0.0, dtype=ttnn.bfloat8_b)
        #     output_tensor_left = ttnn.add(output_tensor_left, 0.0, dtype=ttnn.bfloat8_b)
        output_tensor = ttnn.concat([output_tensor, output_tensor_left], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(output_tensor_left)

        output_tensor = self.conv5(output_tensor)
        output_tensor = ttnn.mish(output_tensor)
        return output_tensor

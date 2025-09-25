# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov4.tt.common import Conv


class TtHead:
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
        self.conv9 = Conv(
            device,
            conv_args.c9,
            parameters.c9,
        )
        self.conv10 = Conv(
            device,
            conv_args.c10,
            parameters.c10,
        )
        self.conv11 = Conv(
            device,
            conv_args.c11,
            parameters.c11,
        )

        self.conv12 = Conv(
            device,
            conv_args.c12,
            parameters.c12,
        )
        self.conv13 = Conv(
            device,
            conv_args.c13,
            parameters.c13,
        )
        self.conv14 = Conv(
            device,
            conv_args.c14,
            parameters.c14,
        )
        self.conv15 = Conv(
            device,
            conv_args.c15,
            parameters.c15,
        )
        self.conv16 = Conv(
            device,
            conv_args.c16,
            parameters.c16,
        )
        self.conv17 = Conv(
            device,
            conv_args.c17,
            parameters.c17,
        )
        self.conv18 = Conv(
            device,
            conv_args.c18,
            parameters.c18,
        )

    def __call__(self, input_tensor):
        output_tensor = self.conv1(input_tensor[0])[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor_left_1 = self.conv2(output_tensor)[0]

        output_tensor = self.conv3(input_tensor[0])[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)
        outfrom_Neck1 = input_tensor[2]

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if (
            outfrom_Neck1.memory_config().is_sharded()
        ):  # This is used because test of head sub_module passes interleaved tensor
            outfrom_Neck1 = ttnn.sharded_to_interleaved(outfrom_Neck1, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat([output_tensor, outfrom_Neck1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv4(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv5(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv6(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv7(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv8(output_tensor)[0]
        output_tensor_split = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv9(output_tensor_split)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor_left_2 = self.conv10(output_tensor)[0]

        output_tensor = self.conv11(output_tensor_split)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        outfromNeck2 = input_tensor[1]
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if (
            outfromNeck2.memory_config().is_sharded()
        ):  # This is used because test of head sub_module passes interleaved tensor
            outfromNeck2 = ttnn.sharded_to_interleaved(outfromNeck2, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, outfromNeck2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv12(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv13(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv14(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv15(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv16(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv17(output_tensor)[0]
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor_left_3 = self.conv18(output_tensor)[0]

        return output_tensor_left_1, output_tensor_left_2, output_tensor_left_3

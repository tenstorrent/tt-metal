# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov4.common import create_conv2d_config
from models.tt_cnn.tt.builder import TtConv2d


class TtHead:
    def __init__(self, device, parameters, conv_args) -> None:
        self.parameters = parameters

        self.conv1 = TtConv2d(
            create_conv2d_config(
                conv_args.c1,
                parameters.c1.weight,
                parameters.c1.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )

        self.conv2 = TtConv2d(
            create_conv2d_config(conv_args.c2, parameters.c2.weight, parameters.c2.bias),
            device=device,
        )
        self.conv3 = TtConv2d(
            create_conv2d_config(
                conv_args.c3,
                parameters.c3.weight,
                parameters.c3.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv4 = TtConv2d(
            create_conv2d_config(
                conv_args.c4,
                parameters.c4.weight,
                parameters.c4.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv5 = TtConv2d(
            create_conv2d_config(
                conv_args.c5,
                parameters.c5.weight,
                parameters.c5.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv6 = TtConv2d(
            create_conv2d_config(
                conv_args.c6,
                parameters.c6.weight,
                parameters.c6.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv7 = TtConv2d(
            create_conv2d_config(
                conv_args.c7,
                parameters.c7.weight,
                parameters.c7.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv8 = TtConv2d(
            create_conv2d_config(
                conv_args.c8,
                parameters.c8.weight,
                parameters.c8.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv9 = TtConv2d(
            create_conv2d_config(
                conv_args.c9,
                parameters.c9.weight,
                parameters.c9.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv10 = TtConv2d(
            create_conv2d_config(conv_args.c10, parameters.c10.weight, parameters.c10.bias),
            device=device,
        )
        self.conv11 = TtConv2d(
            create_conv2d_config(
                conv_args.c11,
                parameters.c11.weight,
                parameters.c11.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )

        self.conv12 = TtConv2d(
            create_conv2d_config(
                conv_args.c12,
                parameters.c12.weight,
                parameters.c12.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv13 = TtConv2d(
            create_conv2d_config(
                conv_args.c13,
                parameters.c13.weight,
                parameters.c13.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv14 = TtConv2d(
            create_conv2d_config(
                conv_args.c14,
                parameters.c14.weight,
                parameters.c14.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv15 = TtConv2d(
            create_conv2d_config(
                conv_args.c15,
                parameters.c15.weight,
                parameters.c15.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv16 = TtConv2d(
            create_conv2d_config(
                conv_args.c16,
                parameters.c16.weight,
                parameters.c16.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv17 = TtConv2d(
            create_conv2d_config(
                conv_args.c17,
                parameters.c17.weight,
                parameters.c17.bias,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.LEAKY_RELU, 0.1),
            ),
            device=device,
        )
        self.conv18 = TtConv2d(
            create_conv2d_config(conv_args.c18, parameters.c18.weight, parameters.c18.bias),
            device=device,
        )

    def __call__(self, input_tensor):
        output_tensor = self.conv1(input_tensor[0])

        output_tensor_left_1 = self.conv2(output_tensor)

        output_tensor = self.conv3(input_tensor[0])
        outfrom_Neck1 = input_tensor[2]

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if (
            outfrom_Neck1.memory_config().is_sharded()
        ):  # This is used because test of head sub_module passes interleaved tensor
            outfrom_Neck1 = ttnn.sharded_to_interleaved(outfrom_Neck1, ttnn.L1_MEMORY_CONFIG)

        if self.parameters.resolution[0] == 320:
            output_tensor = ttnn.add(output_tensor, 0.0, dtype=ttnn.bfloat8_b)
            outfrom_Neck1 = ttnn.add(outfrom_Neck1, 0.0, dtype=ttnn.bfloat8_b)
        output_tensor = ttnn.concat([output_tensor, outfrom_Neck1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv4(output_tensor)

        output_tensor = self.conv5(output_tensor)

        output_tensor = self.conv6(output_tensor)

        output_tensor = self.conv7(output_tensor)

        output_tensor_split = self.conv8(output_tensor)

        output_tensor = self.conv9(output_tensor_split)

        output_tensor_left_2 = self.conv10(output_tensor)

        output_tensor = self.conv11(output_tensor_split)

        outfromNeck2 = input_tensor[1]
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if (
            outfromNeck2.memory_config().is_sharded()
        ):  # This is used because test of head sub_module passes interleaved tensor
            outfromNeck2 = ttnn.sharded_to_interleaved(outfromNeck2, ttnn.L1_MEMORY_CONFIG)
        if self.parameters.resolution[0] == 320:
            output_tensor = ttnn.add(output_tensor, 0.0, dtype=ttnn.bfloat8_b)
            outfromNeck2 = ttnn.add(outfromNeck2, 0.0, dtype=ttnn.bfloat8_b)
        if self.parameters.resolution[0] == 640:
            output_tensor = ttnn.add(output_tensor, 0.0, dtype=ttnn.bfloat8_b)
        output_tensor = ttnn.concat([output_tensor, outfromNeck2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv12(output_tensor)

        output_tensor = self.conv13(output_tensor)

        output_tensor = self.conv14(output_tensor)

        output_tensor = self.conv15(output_tensor)

        output_tensor = self.conv16(output_tensor)

        output_tensor = self.conv17(output_tensor)

        output_tensor_left_3 = self.conv18(output_tensor)

        return output_tensor_left_1, output_tensor_left_2, output_tensor_left_3

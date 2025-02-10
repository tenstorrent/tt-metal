# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.demos.yolov4.ttnn.common import Conv


class TtHead:
    def __init__(self, device, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(
            device, torch_model, "head.conv1", [1, 40, 40, 128], (1, 1, 1, 1), reshard=True, deallocate=False
        )
        self.conv2 = Conv(device, torch_model, "head.conv2", [1, 40, 40, 256], (1, 1, 0, 0), fused_op=False)
        self.conv3 = Conv(
            device,
            torch_model,
            "head.conv3",
            [1, 40, 40, 128],
            (2, 2, 1, 1),
            reshard=True,
            deallocate=False,
            height_sharding=False,
        )
        self.conv4 = Conv(
            device,
            torch_model,
            "head.conv4",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv5 = Conv(
            device,
            torch_model,
            "head.conv5",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.conv6 = Conv(
            device,
            torch_model,
            "head.conv6",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv7 = Conv(
            device,
            torch_model,
            "head.conv7",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            height_sharding=False,
        )
        self.conv8 = Conv(
            device,
            torch_model,
            "head.conv8",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv9 = Conv(
            device,
            torch_model,
            "head.conv9",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            deallocate=False,
            height_sharding=False,
        )
        self.conv10 = Conv(
            device,
            torch_model,
            "head.conv10",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            fused_op=False,
        )
        self.conv11 = Conv(
            device,
            torch_model,
            "head.conv11",
            [1, 20, 20, 256],
            (2, 2, 1, 1),
            reshard=True,
            height_sharding=False,
        )
        self.conv12 = Conv(
            device,
            torch_model,
            "head.conv12",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv13 = Conv(
            device,
            torch_model,
            "head.conv13",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            width_sharding=True,
        )
        self.conv14 = Conv(
            device,
            torch_model,
            "head.conv14",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv15 = Conv(
            device,
            torch_model,
            "head.conv15",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            width_sharding=True,
        )
        self.conv16 = Conv(
            device,
            torch_model,
            "head.conv16",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
        )
        self.conv17 = Conv(
            device,
            torch_model,
            "head.conv17",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            width_sharding=True,
        )
        self.conv18 = Conv(
            device,
            torch_model,
            "head.conv18",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            fused_op=False,
            height_sharding=False,
        )

    def __call__(self, input_tensor):
        output_tensor = self.conv1(input_tensor[0])
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor_left_1 = self.conv2(output_tensor)

        output_tensor = self.conv3(input_tensor[0])
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)
        outfrom_Neck1 = input_tensor[2]

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if (
            outfrom_Neck1.memory_config().is_sharded()
        ):  # This is used because test of head sub_module passes interleaved tensor
            outfrom_Neck1 = ttnn.sharded_to_interleaved(outfrom_Neck1, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat([output_tensor, outfrom_Neck1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv4(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv5(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv6(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv7(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv8(output_tensor)
        output_tensor_split = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv9(output_tensor_split)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor_left_2 = self.conv10(output_tensor)

        output_tensor = self.conv11(output_tensor_split)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        outfromNeck2 = input_tensor[1]
        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        if (
            outfromNeck2.memory_config().is_sharded()
        ):  # This is used because test of head sub_module passes interleaved tensor
            outfromNeck2 = ttnn.sharded_to_interleaved(outfromNeck2, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, outfromNeck2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.conv12(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv13(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv14(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv15(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv16(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor = self.conv17(output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, negative_slope=0.1)

        output_tensor_left_3 = self.conv18(output_tensor)

        return output_tensor_left_1, output_tensor_left_2, output_tensor_left_3

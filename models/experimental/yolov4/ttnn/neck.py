# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.yolov4.ttnn.common import Conv
from tt_lib.fallback_ops import fallback_ops


class TtNeck:
    def __init__(self, model) -> None:
        if type(model) is str:
            torch_model = torch.load(model)
        else:
            torch_model = model.torch_model
        self.torch_model = torch_model
        self.conv1 = Conv(
            torch_model,
            "neek.conv1",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )
        self.conv2 = Conv(
            torch_model,
            "neek.conv2",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            height_sharding=False,
            reshard=True,
        )
        self.conv3 = Conv(
            torch_model,
            "neek.conv3",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )

        self.p1 = fallback_ops.MaxPool2d(kernel_size=5, stride=1, padding=2, dilation=1, ceil_mode=False)
        #  ttnn.MaxPool2d(
        #     kernel_size=(5, 5),
        #     stride=(1, 1),
        #     padding=(2, 2),
        #     dilation=(1, 1),
        #     dtype=ttnn.bfloat16,
        #     device=self.device,
        #     batch_size=self.batch_size,
        #     input_height=10,
        #     input_width=10,
        #     reader_patterns_cache=self.max_pool_reader_patterns_cache,
        #     deallocate_activation=True,
        #     parallel_config_override={},
        #     channels=512
        # )

        self.p2 = fallback_ops.MaxPool2d(kernel_size=9, stride=1, padding=4, dilation=1, ceil_mode=False)
        #  ttnn.MaxPool2d(
        #     kernel_size=(9, 9),
        #     stride=(1, 1),
        #     padding=(4, 4),
        #     dilation=(1, 1),
        #     dtype=ttnn.bfloat16,
        #     device=self.device,
        #     batch_size=self.batch_size,
        #     input_height=10,
        #     input_width=10,
        #     reader_patterns_cache=self.max_pool_reader_patterns_cache,
        #     deallocate_activation=True,
        #     parallel_config_override={},
        #     channels=512
        # )

        self.p3 = fallback_ops.MaxPool2d(kernel_size=13, stride=1, padding=6, dilation=1, ceil_mode=False)
        #  ttnn.MaxPool2d(
        #     kernel_size=(13, 13),
        #     stride=(1, 1),
        #     padding=(6, 6),
        #     dilation=(1, 1),
        #     dtype=ttnn.bfloat16,
        #     device=self.device,
        #     batch_size=self.batch_size,
        #     input_height=10,
        #     input_width=10,
        #     reader_patterns_cache=self.max_pool_reader_patterns_cache,
        #     deallocate_activation=True,
        #     parallel_config_override={},
        #     channels=512
        # )

        self.conv4 = Conv(
            torch_model,
            "neek.conv4",
            [1, 10, 10, 2048],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )
        self.conv5 = Conv(
            torch_model,
            "neek.conv5",
            [1, 10, 10, 512],
            (1, 1, 1, 1),
            height_sharding=False,
            reshard=True,
        )
        self.conv6 = Conv(
            torch_model,
            "neek.conv6",
            [1, 10, 10, 1024],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )
        self.conv7 = Conv(
            torch_model,
            "neek.conv7",
            [1, 10, 10, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
            deallocate=False,
        )
        self.conv7_2 = Conv(
            torch_model,
            "neek.conv8",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )
        self.conv7_3 = Conv(
            torch_model,
            "neek.conv9",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )
        self.conv8 = Conv(
            torch_model,
            "neek.conv10",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            reshard=True,
        )
        self.conv7_4 = Conv(
            torch_model,
            "neek.conv11",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )
        self.conv8_2 = Conv(
            torch_model,
            "neek.conv12",
            [1, 20, 20, 256],
            (1, 1, 1, 1),
            reshard=True,
        )
        self.conv7_5 = Conv(
            torch_model,
            "neek.conv13",
            [1, 20, 20, 512],
            (1, 1, 0, 0),
            height_sharding=False,
            reshard=True,
        )

        self.conv9 = Conv(
            torch_model,
            "neek.conv14",
            [1, 20, 20, 256],
            (1, 1, 0, 0),
            reshard=True,
            deallocate=False,
        )
        self.conv9_2 = Conv(
            torch_model,
            "neek.conv15",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
            reshard=True,
        )
        self.conv9_3 = Conv(
            torch_model,
            "neek.conv16",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
            reshard=True,
        )
        self.conv10 = Conv(
            torch_model,
            "neek.conv17",
            [1, 40, 40, 128],
            (1, 1, 1, 1),
            reshard=True,
        )

        self.conv9_4 = Conv(
            torch_model,
            "neek.conv18",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
            reshard=True,
        )
        self.conv10_2 = Conv(
            torch_model,
            "neek.conv19",
            [1, 40, 40, 128],
            (1, 1, 1, 1),
            reshard=True,
        )
        self.conv9_5 = Conv(
            torch_model,
            "neek.conv20",
            [1, 40, 40, 256],
            (1, 1, 0, 0),
            reshard=True,
        )

    def __call__(self, device, input_tensor):
        output_tensor = self.conv1(device, input_tensor[0])
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv2(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv3(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor_conv3 = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        # Once issue #7746 is resolved we will use ttnn.MaxPool instead of fallback.MaxPool
        output_tensor_conv3 = ttnn.to_layout(output_tensor_conv3, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_conv3 = ttnn.reshape(
            output_tensor_conv3, (1, 10, 10, 512)
        )  # hard coded the shape as in future we will be using ttnn.MaxPool
        output_tensor_conv3 = ttnn.permute(output_tensor_conv3, (0, 3, 1, 2))

        pool_1 = self.p1(output_tensor_conv3)
        pool_2 = self.p2(output_tensor_conv3)
        pool_3 = self.p3(output_tensor_conv3)

        pool_1 = ttnn.permute(pool_1, (0, 2, 3, 1))
        pool_1 = ttnn.reshape(pool_1, (1, 1, pool_1.shape[1] * pool_1.shape[2], pool_1.shape[3]))
        pool_2 = ttnn.permute(pool_2, (0, 2, 3, 1))
        pool_2 = ttnn.reshape(pool_2, (1, 1, pool_2.shape[1] * pool_2.shape[2], pool_2.shape[3]))
        pool_3 = ttnn.permute(pool_3, (0, 2, 3, 1))
        pool_3 = ttnn.reshape(pool_3, (1, 1, pool_3.shape[1] * pool_3.shape[2], pool_3.shape[3]))
        pool_1 = ttnn.to_layout(pool_1, layout=ttnn.TILE_LAYOUT)
        pool_2 = ttnn.to_layout(pool_2, layout=ttnn.TILE_LAYOUT)
        pool_3 = ttnn.to_layout(pool_3, layout=ttnn.TILE_LAYOUT)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([pool_3, pool_2, pool_1, output_tensor], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(pool_3)
        ttnn.deallocate(pool_2)
        ttnn.deallocate(pool_1)

        output_tensor = self.conv4(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv5(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv6(device, output_tensor)
        output_tensor_left_1 = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv7(device, output_tensor_left_1)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_upsample_1 = ttnn.upsample(output_tensor, (1, 4, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor_upsample_1 = ttnn.to_layout(output_tensor_upsample_1, layout=ttnn.TILE_LAYOUT)

        outDowSample5 = input_tensor[1]
        output_tensor = self.conv7_2(device, outDowSample5)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_upsample_1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(output_tensor_upsample_1)

        output_tensor = self.conv7_3(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv8(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv7_4(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv8_2(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv7_5(device, output_tensor)
        output_tensor_left_2 = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv9(device, output_tensor_left_2)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, ttnn.ROW_MAJOR_LAYOUT)
        output_tensor_upsample_2 = ttnn.upsample(output_tensor, (1, 4, 1), memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor_upsample_2 = ttnn.to_layout(output_tensor_upsample_2, ttnn.TILE_LAYOUT)

        outDowSample3 = input_tensor[2]

        output_tensor = self.conv9_2(device, outDowSample3)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat(
            [output_tensor, output_tensor_upsample_2], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        ttnn.deallocate(output_tensor_upsample_2)

        output_tensor = self.conv9_3(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv10(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv9_4(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv10_2(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        output_tensor = self.conv9_5(device, output_tensor)
        output_tensor = ttnn.leaky_relu(output_tensor, slope=0.1)

        return output_tensor, output_tensor_left_1, output_tensor_left_2

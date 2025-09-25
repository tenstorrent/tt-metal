# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0


import ttnn

from models.experimental.yolo_common.yolo_utils import concat

from models.experimental.yolov5x.tt.c3 import TtnnC3
from models.experimental.yolov5x.tt.sppf import TtnnSPPF
from models.experimental.yolov5x.tt.detect import TtnnDetect
from models.experimental.yolov5x.tt.common import TtYOLOv5xConv2D, interleaved_to_sharded


class Yolov5x:
    def __init__(self, device, parameters, conv_pt):
        self.device = device

        self.conv1 = TtYOLOv5xConv2D(
            device,
            parameters.conv_args[0].conv,
            conv_pt.model[0].conv,
            config_override={"act_block_h": 64},
            activation="silu",
            deallocate_activation=True,
        )
        self.conv2 = TtYOLOv5xConv2D(
            device, parameters.conv_args[1].conv, conv_pt.model[1].conv, deallocate_activation=True, activation="silu"
        )
        self.c3_1 = TtnnC3(
            shortcut=True, n=4, device=self.device, parameters=parameters.conv_args[2], conv_pt=conv_pt.model[2]
        )
        self.conv3 = TtYOLOv5xConv2D(
            device, parameters.conv_args[3].conv, conv_pt.model[3].conv, deallocate_activation=False, activation="silu"
        )
        self.c3_2 = TtnnC3(
            shortcut=True, n=8, device=self.device, parameters=parameters.conv_args[4], conv_pt=conv_pt.model[4]
        )
        self.conv4 = TtYOLOv5xConv2D(
            device, parameters.conv_args[5].conv, conv_pt.model[5].conv, deallocate_activation=False, activation="silu"
        )
        self.c3_3 = TtnnC3(
            shortcut=True, n=12, device=self.device, parameters=parameters.conv_args[6], conv_pt=conv_pt.model[6]
        )
        self.conv5 = TtYOLOv5xConv2D(
            device,
            parameters.conv_args[7].conv,
            conv_pt.model[7].conv,
            deallocate_activation=False,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )
        self.c3_4 = TtnnC3(
            shortcut=True,
            n=4,
            device=self.device,
            parameters=parameters.conv_args[8],
            conv_pt=conv_pt.model[8],
            use_block_shard=True,
        )
        self.sppf = TtnnSPPF(device, parameters.conv_args[9], conv_pt.model[9])
        self.conv6 = TtYOLOv5xConv2D(
            device,
            parameters.conv_args[10].conv,
            conv_pt.model[10].conv,
            deallocate_activation=False,
            activation="silu",
        )

        self.c3_5 = TtnnC3(
            shortcut=False, n=4, device=self.device, parameters=parameters.conv_args[13], conv_pt=conv_pt.model[13]
        )
        self.conv7 = TtYOLOv5xConv2D(
            device,
            parameters.conv_args[14].conv,
            conv_pt.model[14].conv,
            deallocate_activation=False,
            activation="silu",
        )

        self.c3_6 = TtnnC3(
            shortcut=False, n=4, device=self.device, parameters=parameters.conv_args[17], conv_pt=conv_pt.model[17]
        )
        self.conv8 = TtYOLOv5xConv2D(
            device,
            parameters.conv_args[18].conv,
            conv_pt.model[18].conv,
            deallocate_activation=False,
            activation="silu",
        )

        self.c3_7 = TtnnC3(
            shortcut=False, n=4, device=self.device, parameters=parameters.conv_args[20], conv_pt=conv_pt.model[20]
        )
        self.conv9 = TtYOLOv5xConv2D(
            device,
            parameters.conv_args[21].conv,
            conv_pt.model[21].conv,
            deallocate_activation=False,
            activation="silu",
            use_1d_systolic_array=False,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        )

        self.c3_8 = TtnnC3(
            shortcut=False,
            n=4,
            device=self.device,
            parameters=parameters.conv_args[23],
            conv_pt=conv_pt.model[23],
            use_block_shard=True,
        )
        self.detect = TtnnDetect(device, parameters.model_args.model[24], conv_pt.model[24])

    def __call__(self, x):
        N, C, H, W = x.shape
        min_channels = 16
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(x, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = x
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))  # NCHW -> NHWC
        ttnn.deallocate(nchw)
        ttnn.deallocate(x)
        nhwc = ttnn.reallocate(nhwc)
        x = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])
        ttnn.deallocate(nhwc)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        x = self.conv3(x)
        x = self.c3_2(x)
        x4 = x

        x = self.conv4(x)
        x = self.c3_3(x)
        x6 = x

        x = self.conv5(x)
        x = self.c3_4(x)
        x = self.sppf(x)
        x = self.conv6(x)
        x10 = x

        x = interleaved_to_sharded(x)
        x = ttnn.upsample(x, scale_factor=2)

        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

        x = concat(-1, True, x, x6)
        ttnn.deallocate(x6)

        x = self.c3_5(x)
        x = self.conv7(x)
        x14 = x

        x = interleaved_to_sharded(x)
        x = ttnn.upsample(x, scale_factor=2)

        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))

        x = concat(-1, True, x, x4)
        ttnn.deallocate(x4)

        x = self.c3_6(x)
        x17 = x
        x = self.conv8(x)

        x = concat(-1, True, x, x14)
        ttnn.deallocate(x14)

        x = self.c3_7(x)
        x20 = x
        x = self.conv9(x)

        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        x = concat(-1, True, x, x10)
        ttnn.deallocate(x10)

        x = self.c3_8(x)
        x23 = x
        x = self.detect(x17, x20, x23)

        ttnn.deallocate(x17)
        ttnn.deallocate(x20)
        ttnn.deallocate(x23)

        return x

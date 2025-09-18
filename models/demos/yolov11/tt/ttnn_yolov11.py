# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import ttnn
from models.demos.yolov11.tt.common import TtnnConv, deallocate_tensors, sharded_concat, sharded_concat_2
from models.demos.yolov11.tt.ttnn_yolov11_c2psa import TtnnC2PSA
from models.demos.yolov11.tt.ttnn_yolov11_c3k2 import TtnnC3k2
from models.demos.yolov11.tt.ttnn_yolov11_detect import TtnnDetect
from models.demos.yolov11.tt.ttnn_yolov11_sppf import TtnnSPPF
from models.experimental.yolo_common.yolo_utils import determine_num_cores, get_core_grid_from_num_cores


class TtnnYoloV11:
    def __init__(self, device, parameters):
        self.device = device
        self.conv1 = TtnnConv(device, parameters.conv_args[0], parameters.model[0], deallocate_activation=True)
        self.conv2 = TtnnConv(device, parameters.conv_args[1], parameters.model[1])
        self.c3k2_1 = TtnnC3k2(device, parameters.conv_args[2], parameters.model[2], is_bk_enabled=True)
        self.conv3 = TtnnConv(device, parameters.conv_args[3], parameters.model[3])
        self.c3k2_2 = TtnnC3k2(device, parameters.conv_args[4], parameters.model[4], is_bk_enabled=True)
        self.conv5 = TtnnConv(device, parameters.conv_args[5], parameters.model[5])
        self.c3k2_3 = TtnnC3k2(device, parameters.conv_args[6], parameters.model[6], is_bk_enabled=False)
        self.conv6 = TtnnConv(device, parameters.conv_args[7], parameters.model[7])
        self.c3k2_4 = TtnnC3k2(device, parameters.conv_args[8], parameters.model[8], is_bk_enabled=False)
        self.sppf = TtnnSPPF(device, parameters.conv_args[9], parameters.model[9])
        self.c2psa = TtnnC2PSA(device, parameters.conv_args[10], parameters.model[10])
        self.c3k2_5 = TtnnC3k2(device, parameters.conv_args[13], parameters.model[13], is_bk_enabled=True, reshard=True)
        self.c3k2_6 = TtnnC3k2(device, parameters.conv_args[16], parameters.model[16], is_bk_enabled=True, reshard=True)
        self.conv7 = TtnnConv(device, parameters.conv_args[17], parameters.model[17])
        self.c3k2_7 = TtnnC3k2(device, parameters.conv_args[19], parameters.model[19], is_bk_enabled=True, reshard=True)
        self.conv8 = TtnnConv(device, parameters.conv_args[20], parameters.model[20])
        self.c3k2_8 = TtnnC3k2(
            device, parameters.conv_args[22], parameters.model[22], is_bk_enabled=False, reshard=True
        )
        self.detect = TtnnDetect(device, parameters.model_args.model[23], parameters.model[23])

    def __call__(self, input, min_channels=8):
        n, c, h, w = input.shape
        channel_padding_needed = min_channels - c
        x = ttnn.pad(input, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        ttnn.deallocate(input)
        x = ttnn.permute(x, (0, 2, 3, 1))
        x = ttnn.reshape(x, (1, 1, n * h * w, min_channels))
        x = self.conv1(self.device, x)
        x = self.conv2(self.device, x)
        x = self.c3k2_1(self.device, x)
        x = self.conv3(self.device, x)
        x = self.c3k2_2(self.device, x)
        x4 = x
        x = self.conv5(self.device, x)
        x = self.c3k2_3(self.device, x)
        x6 = x
        x = self.conv6(self.device, x)
        x = self.c3k2_4(self.device, x)
        x = self.sppf(self.device, x)
        x = self.c2psa(self.device, x)
        x10 = x
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)
        x = ttnn.upsample(x, scale_factor=2, memory_config=x.memory_config())
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        x = sharded_concat_2(x, x6)
        ttnn.deallocate(x6)
        x = self.c3k2_5(self.device, x)
        x13 = x
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (x.shape[0], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])), x.shape[3]))
        nhw = x.shape[0] * x.shape[1] * x.shape[2]
        num_cores = determine_num_cores(nhw, x.shape[2])
        core_grid = get_core_grid_from_num_cores(num_cores)
        shardspec = ttnn.create_sharded_memory_config_(
            x.shape, core_grid, ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR
        )
        if x.is_sharded():
            x = ttnn.reshard(x, shardspec)
        else:
            x = ttnn.interleaved_to_sharded(x, shardspec)
        x = ttnn.upsample(x, scale_factor=2, memory_config=x.memory_config())
        x = ttnn.reshape(x, (1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3]))
        x4 = ttnn.to_layout(x4, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = sharded_concat([x, x4], to_interleaved=False)
        ttnn.deallocate(x4)
        x = self.c3k2_6(self.device, x)
        x16 = x
        x = self.conv7(self.device, x)
        x = sharded_concat_2(x, x13)
        ttnn.deallocate(x13)
        x = self.c3k2_7(self.device, x)
        x19 = x
        x = self.conv8(self.device, x)
        x = sharded_concat_2(x, x10)
        ttnn.deallocate(x10)
        x = self.c3k2_8(self.device, x)
        x22 = x
        x = self.detect(self.device, x16, x19, x22)
        deallocate_tensors(x16, x19, x22)

        return x

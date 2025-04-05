# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_yolov10.tt.scdown import TtnnSCDown
from models.experimental.functional_yolov10.tt.sppf import TtnnSPPF
from models.experimental.functional_yolov10.tt.psa import TtnnPSA
from models.experimental.functional_yolov10.tt.c2f import TtnnC2f
from models.experimental.functional_yolov10.tt.c2fcib import TtnnC2fCIB
from models.experimental.functional_yolov10.tt.v10detect import TtnnV10Detect
from models.experimental.functional_yolov10.tt.common import interleaved_to_sharded, Conv


class TtnnYolov10:
    def __init__(self, device, parameters, conv_pt):
        self.device = device
        self.conv1 = Conv(device, parameters.conv_args[0], conv_pt.model[0], config_override={"act_block_h": 64})
        self.conv2 = Conv(device, parameters.conv_args[1], conv_pt.model[1], auto_shard=True)
        self.c2f_1 = TtnnC2f(
            shortcut=True, n=3, device=self.device, parameters=parameters.conv_args[2], conv_pt=conv_pt.model[2]
        )
        self.conv3 = Conv(device, parameters.conv_args[3], conv_pt.model[3], auto_shard=True)
        self.c2f_2 = TtnnC2f(
            shortcut=True, n=6, device=self.device, parameters=parameters.conv_args[4], conv_pt=conv_pt.model[4]
        )
        self.scdown_1 = TtnnSCDown(device=device, parameters=parameters.conv_args[5], conv_pt=conv_pt.model[5])
        self.c2fcib_1 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[6], n=6, conv_pt=conv_pt.model[6])
        self.scdown_2 = TtnnSCDown(device=device, parameters=parameters.conv_args[7], conv_pt=conv_pt.model[7])
        self.c2fcib_2 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[8], conv_pt=conv_pt.model[8])
        self.sppf = TtnnSPPF(device=device, parameters=parameters.conv_args[9], conv_pt=conv_pt.model[9])
        self.psa_1 = TtnnPSA(device=device, parameters=parameters.conv_args[10], conv_pt=conv_pt.model[10])
        self.c2fcib_3 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[13], conv_pt=conv_pt.model[13])
        self.c2f_3 = TtnnC2f(
            shortcut=False, n=3, device=self.device, parameters=parameters.conv_args[16], conv_pt=conv_pt.model[16]
        )
        self.conv4 = Conv(device, parameters.conv_args[17], conv_pt.model[17], auto_shard=True)
        self.c2fcib_4 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[19], conv_pt=conv_pt.model[19])
        self.scdown_3 = TtnnSCDown(device=device, parameters=parameters.conv_args[20], conv_pt=conv_pt.model[20])
        self.c2fcib_5 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[22], conv_pt=conv_pt.model[22])
        self.detect = TtnnV10Detect(
            device=device, parameters=parameters.model_args.model[23], conv_pt=conv_pt.model[23]
        )

    def __call__(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.c2f_1(x1)
        x3 = self.conv3(x2)
        x4 = self.c2f_2(x3)
        x5 = self.scdown_1(x4)
        x6 = self.c2fcib_1(x5)
        x7 = self.scdown_2(x6)
        x8 = self.c2fcib_2(x7)
        x9 = self.sppf(x8)
        x10 = self.psa_1(x9)
        ttnn.deallocate(x)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)
        ttnn.deallocate(x5)
        ttnn.deallocate(x7)
        ttnn.deallocate(x8)
        ttnn.deallocate(x9)

        x10 = interleaved_to_sharded(x10)
        x11 = ttnn.upsample(x10, scale_factor=2)

        if x11.is_sharded():
            x11 = ttnn.sharded_to_interleaved(x11, memory_config=ttnn.L1_MEMORY_CONFIG)
        x11 = ttnn.reshape(x11, (1, 1, x11.shape[0] * x11.shape[1] * x11.shape[2], x11.shape[3]))
        x11 = ttnn.to_layout(x11, layout=ttnn.ROW_MAJOR_LAYOUT)
        x12 = ttnn.concat((x11, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x6)
        ttnn.deallocate(x11)
        x13 = self.c2fcib_3(x12)
        ttnn.deallocate(x12)

        x13 = interleaved_to_sharded(x13)
        x14 = ttnn.upsample(x13, scale_factor=2)

        if x14.is_sharded():
            x14 = ttnn.sharded_to_interleaved(x14, memory_config=ttnn.L1_MEMORY_CONFIG)
        x14 = ttnn.reshape(x14, (1, 1, x14.shape[0] * x14.shape[1] * x14.shape[2], x14.shape[3]))
        x14 = ttnn.to_layout(x14, layout=ttnn.ROW_MAJOR_LAYOUT)
        x15 = ttnn.concat((x14, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x14)
        ttnn.deallocate(x4)

        x16 = self.c2f_3(x15)
        ttnn.deallocate(x15)
        x17 = self.conv4(x16)

        if x17.is_sharded():
            x17 = ttnn.sharded_to_interleaved(x17, memory_config=ttnn.L1_MEMORY_CONFIG)
        x17 = ttnn.reshape(x17, (1, 1, x17.shape[0] * x17.shape[1] * x17.shape[2], x17.shape[3]))
        x17 = ttnn.to_layout(x17, layout=ttnn.ROW_MAJOR_LAYOUT)
        x13 = ttnn.reshape(x13, ((1, 1, x13.shape[0] * x13.shape[1] * x13.shape[2], x13.shape[3])))
        x13 = ttnn.sharded_to_interleaved(x13, memory_config=ttnn.L1_MEMORY_CONFIG)
        x18 = ttnn.concat((x17, x13), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(x4)
        ttnn.deallocate(x17)
        ttnn.deallocate(x13)

        x19 = self.c2fcib_4(x18)
        x20 = self.scdown_3(x19)
        ttnn.deallocate(x18)

        if x20.is_sharded():
            x20 = ttnn.sharded_to_interleaved(x20, memory_config=ttnn.L1_MEMORY_CONFIG)
        x20 = ttnn.reshape(x20, (1, 1, x20.shape[0] * x20.shape[1] * x20.shape[2], x20.shape[3]))
        x20 = ttnn.to_layout(x20, layout=ttnn.ROW_MAJOR_LAYOUT)
        x10 = ttnn.reshape(x10, ((1, 1, x10.shape[0] * x10.shape[1] * x10.shape[2], x10.shape[3])))
        x10 = ttnn.sharded_to_interleaved(x10, memory_config=ttnn.L1_MEMORY_CONFIG)

        x21 = ttnn.concat((x20, x10), -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        x22 = self.c2fcib_5(x21)
        x23 = self.detect(x16, x19, x22)
        return x23

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.yolov10.tt.scdown import TtnnSCDown
from models.experimental.yolov10.tt.sppf import TtnnSPPF
from models.experimental.yolov10.tt.psa import TtnnPSA
from models.experimental.yolov10.tt.c2f import TtnnC2f
from models.experimental.yolov10.tt.c2fcib import TtnnC2fCIB
from models.experimental.yolov10.tt.v10detect import TtnnV10Detect
from models.experimental.yolov10.tt.common import interleaved_to_sharded, Conv
from models.experimental.yolo_common.yolo_utils import concat


class TtnnYolov10:
    def __init__(self, device, parameters, conv_pt):
        self.device = device
        self.conv1 = Conv(device, parameters.conv_args[0], conv_pt.model[0], deallocate_activation=True)
        self.conv2 = Conv(device, parameters.conv_args[1], conv_pt.model[1], deallocate_activation=True)
        self.c2f_1 = TtnnC2f(
            shortcut=True, n=3, device=self.device, parameters=parameters.conv_args[2], conv_pt=conv_pt.model[2]
        )
        self.conv3 = Conv(device, parameters.conv_args[3], conv_pt.model[3], deallocate_activation=True)
        self.c2f_2 = TtnnC2f(
            shortcut=True, n=6, device=self.device, parameters=parameters.conv_args[4], conv_pt=conv_pt.model[4]
        )
        self.scdown_1 = TtnnSCDown(
            device=device, parameters=parameters.conv_args[5], conv_pt=conv_pt.model[5], auto_shard=True
        )
        self.c2fcib_1 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[6], n=6, conv_pt=conv_pt.model[6])
        self.scdown_2 = TtnnSCDown(device=device, parameters=parameters.conv_args[7], conv_pt=conv_pt.model[7])
        self.c2fcib_2 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[8], conv_pt=conv_pt.model[8])
        self.sppf = TtnnSPPF(device=device, parameters=parameters.conv_args[9], conv_pt=conv_pt.model[9])
        self.psa_1 = TtnnPSA(device=device, parameters=parameters.conv_args[10], conv_pt=conv_pt.model[10])
        self.c2fcib_3 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[13], conv_pt=conv_pt.model[13])
        self.c2f_3 = TtnnC2f(
            shortcut=False, n=3, device=self.device, parameters=parameters.conv_args[16], conv_pt=conv_pt.model[16]
        )
        self.conv4 = Conv(device, parameters.conv_args[17], conv_pt.model[17])
        self.c2fcib_4 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[19], conv_pt=conv_pt.model[19])
        self.scdown_3 = TtnnSCDown(device=device, parameters=parameters.conv_args[20], conv_pt=conv_pt.model[20])
        self.c2fcib_5 = TtnnC2fCIB(device=device, parameters=parameters.conv_args[22], conv_pt=conv_pt.model[22])
        self.detect = TtnnV10Detect(
            device=device, parameters=parameters.model_args.model[23], conv_pt=conv_pt.model[23]
        )

    def __call__(self, input_tensor):
        conv1_out = self.conv1(input_tensor)
        conv2_out = self.conv2(conv1_out)
        ttnn.deallocate(conv1_out)

        features1 = self.c2f_1(conv2_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(conv2_out)

        conv3_out = self.conv3(features1)
        ttnn.deallocate(features1)

        features2 = self.c2f_2(conv3_out)
        ttnn.deallocate(conv3_out)

        down1 = self.scdown_1(features2)
        branch1 = self.c2fcib_1(down1)
        ttnn.deallocate(down1)

        down2 = self.scdown_2(branch1)
        branch2 = self.c2fcib_2(down2)
        ttnn.deallocate(down2)

        sppf_out = self.sppf(branch2)
        ttnn.deallocate(branch2)

        attention_out = self.psa_1(sppf_out)
        ttnn.deallocate(sppf_out)

        attention_out = interleaved_to_sharded(attention_out)
        up1 = ttnn.upsample(attention_out, scale_factor=2)

        if up1.is_sharded():
            up1 = ttnn.sharded_to_interleaved(up1, memory_config=ttnn.L1_MEMORY_CONFIG)
        up1 = ttnn.reshape(up1, (1, 1, up1.shape[0] * up1.shape[1] * up1.shape[2], up1.shape[3]))

        fused_1 = concat(-1, True, up1, branch1)
        ttnn.deallocate(up1)
        ttnn.deallocate(branch1)
        branch3 = self.c2fcib_3(fused_1)

        ttnn.deallocate(fused_1)
        if branch3.is_sharded():
            branch3 = ttnn.sharded_to_interleaved(branch3, ttnn.L1_MEMORY_CONFIG)
        branch3 = interleaved_to_sharded(branch3)  # check
        up2 = ttnn.upsample(branch3, scale_factor=2)

        if up2.is_sharded():
            up2 = ttnn.sharded_to_interleaved(up2, memory_config=ttnn.L1_MEMORY_CONFIG)
        up2 = ttnn.reshape(up2, (1, 1, up2.shape[0] * up2.shape[1] * up2.shape[2], up2.shape[3]))
        if up2.get_layout() == ttnn.TILE_LAYOUT:
            up2 = ttnn.to_layout(up2, layout=ttnn.ROW_MAJOR_LAYOUT)

        fused_2 = concat(-1, False, up2, features2)
        ttnn.deallocate(up2)
        ttnn.deallocate(features2)

        features3 = self.c2f_3(fused_2)
        ttnn.deallocate(fused_2)
        conv4_out = self.conv4(features3)

        if conv4_out.is_sharded():
            conv4_out = ttnn.sharded_to_interleaved(conv4_out, memory_config=ttnn.L1_MEMORY_CONFIG)
        conv4_out = ttnn.reshape(
            conv4_out, (1, 1, conv4_out.shape[0] * conv4_out.shape[1] * conv4_out.shape[2], conv4_out.shape[3])
        )
        if conv4_out.get_layout() == ttnn.TILE_LAYOUT:
            conv4_out = ttnn.to_layout(conv4_out, layout=ttnn.ROW_MAJOR_LAYOUT)
        branch3 = ttnn.reshape(
            branch3, ((1, 1, branch3.shape[0] * branch3.shape[1] * branch3.shape[2], branch3.shape[3]))
        )
        if branch3.is_sharded():
            branch3 = ttnn.sharded_to_interleaved(branch3, memory_config=ttnn.L1_MEMORY_CONFIG)

        fused3 = concat(-1, False, conv4_out, branch3)

        ttnn.deallocate(branch3)
        ttnn.deallocate(conv4_out)

        branch4 = self.c2fcib_4(fused3)
        down3 = self.scdown_3(branch4)

        ttnn.deallocate(fused3)

        if down3.is_sharded():
            down3 = ttnn.sharded_to_interleaved(down3, memory_config=ttnn.L1_MEMORY_CONFIG)
        down3 = ttnn.reshape(down3, (1, 1, down3.shape[0] * down3.shape[1] * down3.shape[2], down3.shape[3]))
        if down3.get_layout() == ttnn.TILE_LAYOUT:
            down3 = ttnn.to_layout(down3, layout=ttnn.ROW_MAJOR_LAYOUT)
        attention_out = ttnn.reshape(
            attention_out,
            ((1, 1, attention_out.shape[0] * attention_out.shape[1] * attention_out.shape[2], attention_out.shape[3])),
        )
        if attention_out.is_sharded():
            attention_out = ttnn.sharded_to_interleaved(attention_out, memory_config=ttnn.L1_MEMORY_CONFIG)

        fused4 = concat(-1, False, down3, attention_out)

        branch5 = self.c2fcib_5(fused4)

        output = self.detect(features3, branch4, branch5)

        return output

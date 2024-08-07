# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0
import ttnn
import tt_lib
from typing import List
from models.experimental.functional_yolox_m.tt.ttnn_cspdarknet import TtCSPDarknet
from models.experimental.functional_yolox_m.tt.ttnn_bottleneck_block import TtBottleneckBlock


class TtYOLOPAFPN:
    def __init__(
        self,
        device,
        parameters,
    ) -> None:
        self.backbone = TtCSPDarknet(device, parameters["backbone"])
        self.c1 = parameters.c1
        self.c2 = parameters.c2
        self.c3 = parameters.c3
        self.c4 = parameters.c4

        self.bblock1 = TtBottleneckBlock(parameters.bblock1, 2, False)

        self.c5 = parameters.c5
        self.c6 = parameters.c6
        self.c7 = parameters.c7
        self.c8 = parameters.c8

        self.bblock2 = TtBottleneckBlock(parameters.bblock2, 2, False)

        self.c9 = parameters.c9
        self.c10 = parameters.c10
        self.c11 = parameters.c11
        self.c12 = parameters.c12

        self.bblock3 = TtBottleneckBlock(parameters.bblock3, 2, False)

        self.c13 = parameters.c13
        self.c14 = parameters.c14
        self.c15 = parameters.c15
        self.c16 = parameters.c16

        self.bblock4 = TtBottleneckBlock(parameters.bblock4, 2, False)
        self.in_features = ("dark3", "dark4", "dark5")

    def __call__(self, device, input_tensor: List[ttnn.Tensor]):
        out_features = self.backbone(device, input_tensor)
        features = [out_features[f] for f in self.in_features]
        # features= input_tensor

        features[2] = features[2].to(device, self.c1.conv.input_sharded_memory_config)
        output_tensor = self.c1(features[2])
        output_tensor = ttnn.silu(output_tensor)
        fpn_out0 = output_tensor

        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        output_tensor = output_tensor.reshape(1, 20, 20, 384)
        output_tensor = ttnn.upsample(output_tensor, (2, 2, 1), memory_config=output_tensor.memory_config())
        output_tensor = output_tensor.reshape(1, 1, 1600, 384)
        features_1 = ttnn.to_torch(features[1])
        features_1 = ttnn.from_torch(
            features_1,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )

        output_tensor = ttnn.concat([output_tensor, features_1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        f_out0 = output_tensor

        # C3_p4
        output_tensor = output_tensor.to(device, self.c2.conv.input_sharded_memory_config)
        output_tensor = self.c2(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor_c2 = output_tensor
        f_out0 = f_out0.to(device, self.c3.conv.input_sharded_memory_config)
        output_tensor = self.c3(f_out0)
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c3 = output_tensor

        output_tensor = self.bblock1(device, output_tensor_c2)
        output_tensor = output_tensor.to(device, self.c3.conv.output_sharded_memory_config)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_c3 = tt_lib.tensor.sharded_to_interleaved(output_tensor_c3, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_c3], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        output_tensor = self.c4(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        # reduce_conv1
        output_tensor = self.c5(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        fpn_out1 = output_tensor
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(output_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
        output_tensor = output_tensor.reshape(1, 40, 40, 192)
        output_tensor = ttnn.upsample(output_tensor, (2, 2, 1), memory_config=output_tensor.memory_config())
        output_tensor = output_tensor.reshape(1, 1, 6400, 192)

        features_0 = ttnn.to_torch(features[0])
        features_0 = ttnn.from_torch(
            features_0,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            dtype=ttnn.bfloat16,
        )
        output_tensor = ttnn.concat([output_tensor, features_0], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        f_out1 = output_tensor

        # C3_p3
        output_tensor = output_tensor.to(device, self.c6.conv.input_sharded_memory_config)
        output_tensor = self.c6(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c6 = output_tensor

        f_out1 = f_out1.to(device, self.c7.conv.input_sharded_memory_config)
        output_tensor = self.c7(f_out1)
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c7 = output_tensor
        output_tensor_c6 = output_tensor_c6.to(device, self.c6.conv.output_sharded_memory_config)
        output_tensor = self.bblock2(device, output_tensor_c6)

        output_tensor = output_tensor.to(device, self.c7.conv.output_sharded_memory_config)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_c7 = tt_lib.tensor.sharded_to_interleaved(output_tensor_c7, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_c7], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c8(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        pan_out2 = output_tensor
        pan_out2 = pan_out2.to(device)

        # bu_conv2
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c9.conv.input_sharded_memory_config)
        output_tensor = self.c9(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        fpn_out1 = fpn_out1.to(device, self.c9.conv.output_sharded_memory_config)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        fpn_out1 = tt_lib.tensor.sharded_to_interleaved(fpn_out1, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, fpn_out1], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)
        p_out1 = output_tensor
        p_out1 = p_out1.to(device, self.c11.conv.output_sharded_memory_config)

        # C3_n3
        output_tensor = self.c10(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor_c10 = output_tensor
        output_tensor = self.c11(p_out1)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor_c11 = output_tensor

        output_tensor = self.bblock3(device, output_tensor_c10)

        output_tensor = output_tensor.to(device, self.c11.conv.output_sharded_memory_config)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_c11 = tt_lib.tensor.sharded_to_interleaved(output_tensor_c11, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_c11], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c12(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        pan_out1 = output_tensor
        pan_out1 = pan_out1.to(device)
        # bu_conv1
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(output_tensor, self.c13.conv.input_sharded_memory_config)
        # ------
        output_tensor = self.c13(output_tensor)
        output_tensor = ttnn.silu(output_tensor)

        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        fpn_out0 = ttnn.to_torch(fpn_out0)
        fpn_out0 = ttnn.from_torch(
            fpn_out0,
            device=device,
            memory_config=self.c13.conv.output_sharded_memory_config,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )
        output_tensor = ttnn.to_torch(output_tensor)
        output_tensor = ttnn.from_torch(
            output_tensor,
            device=device,
            memory_config=self.c13.conv.output_sharded_memory_config,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
        )
        fpn_out0 = tt_lib.tensor.sharded_to_interleaved(fpn_out0, ttnn.L1_MEMORY_CONFIG)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)

        output_tensor = ttnn.concat([output_tensor, fpn_out0], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        p_out0 = output_tensor

        # C3_n4
        output_tensor = ttnn.to_layout(output_tensor, ttnn.TILE_LAYOUT)
        output_tensor = output_tensor.to(device, self.c14.conv.input_sharded_memory_config)
        output_tensor = self.c14(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c14 = output_tensor
        p_out0 = ttnn.to_layout(p_out0, ttnn.TILE_LAYOUT)
        p_out0 = p_out0.to(device, self.c7.conv.input_sharded_memory_config)
        output_tensor = self.c15(p_out0)
        output_tensor = ttnn.silu(output_tensor)
        output_tensor_c15 = output_tensor
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = tt_lib.tensor.interleaved_to_sharded(
            output_tensor, self.bblock4.module_list[0][0].conv.input_sharded_memory_config
        )
        output_tensor = self.bblock4(device, output_tensor_c14)

        output_tensor = output_tensor.to(device, self.c15.conv.output_sharded_memory_config)
        output_tensor = tt_lib.tensor.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor_c15 = tt_lib.tensor.sharded_to_interleaved(output_tensor_c15, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.concat([output_tensor, output_tensor_c15], dim=3, memory_config=ttnn.L1_MEMORY_CONFIG)

        output_tensor = self.c16(output_tensor)
        output_tensor = ttnn.silu(output_tensor)
        pan_out0 = output_tensor
        pan_out0 = pan_out0.to(device)

        outputs = (ttnn.from_device(pan_out2), ttnn.from_device(pan_out1), ttnn.from_device(pan_out0))
        return outputs

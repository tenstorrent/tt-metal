# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.functional_mobilenetv2.tt.common import MobileNetV2Conv2D, InvertedResidual


class MobileNetV2:
    def __init__(self, model_params, device, batchsize) -> None:
        self.device = device
        self.model_parameters = model_params
        self.batchsize = batchsize

        self.conv1 = MobileNetV2Conv2D(
            [3, 2, 1, 32],
            (model_params["fused_conv_0_weight"], model_params["fused_conv_0_bias"]),
            device,
            batchsize,
            use_shallow_covariant=True,
            deallocate_activation=True,
        )
        self.conv2 = MobileNetV2Conv2D(
            [3, 1, 1, 32],
            (model_params["fused_conv_1_weight"], model_params["fused_conv_1_bias"]),
            device,
            batchsize,
            groups=32,
        )
        self.conv3 = MobileNetV2Conv2D(
            [1, 1, 0, 16], (model_params["conv_0_weight"], model_params["conv_0_bias"]), device, batchsize
        )

        self.block1 = InvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=2,
            in_channels=16,
            out_channels=24,
            id=1,
            block_shard=False,
        )
        self.block2 = InvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=24,
            out_channels=24,
            id=2,
            block_shard=False,
        )
        self.block3 = InvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=2,
            in_channels=24,
            out_channels=32,
            id=3,
            block_shard=False,
        )
        self.block4 = InvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=32,
            out_channels=32,
            id=4,
            block_shard=False,
        )
        self.block5 = InvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=32,
            out_channels=32,
            id=5,
            block_shard=False,
        )
        self.block6 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=2, in_channels=32, out_channels=64, id=6
        )
        self.block7 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=64, out_channels=64, id=7
        )
        self.block8 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=64, out_channels=64, id=8
        )
        self.block9 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=64, out_channels=64, id=9
        )
        self.block10 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=64, out_channels=96, id=10
        )
        self.block11 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=96, out_channels=96, id=11
        )
        self.block12 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=96, out_channels=96, id=12
        )
        self.block13 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=2, in_channels=96, out_channels=160, id=13
        )
        self.block14 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=160, out_channels=160, id=14
        )
        self.block15 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=160, out_channels=160, id=15
        )
        self.block16 = InvertedResidual(
            model_params, device, batchsize, expand_ratio=6, stride=1, in_channels=160, out_channels=320, id=16
        )

        self.conv4 = MobileNetV2Conv2D(
            [1, 1, 0, 1280],
            (model_params["fused_conv_34_weight"], model_params["fused_conv_34_bias"]),
            device,
            batchsize,
        )
        self.l1_weight = model_params["classifier_1_weight"]
        self.l1_bias = model_params["classifier_1_bias"]

    def __call__(
        self,
        x,
    ):
        output_tensor, h, w = self.conv1(x)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor, h, w = self.conv2(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        output_tensor, h, w = self.conv3(output_tensor)
        output_tensor = self.block1(output_tensor)
        output_tensor = self.block2(output_tensor)
        output_tensor = self.block3(output_tensor)
        output_tensor = self.block4(output_tensor)
        output_tensor = self.block5(output_tensor)
        output_tensor = self.block6(output_tensor)
        output_tensor = self.block7(output_tensor)
        output_tensor = self.block8(output_tensor)
        output_tensor = self.block9(output_tensor)
        output_tensor = self.block10(output_tensor)
        output_tensor = self.block11(output_tensor)
        output_tensor = self.block12(output_tensor)
        output_tensor = self.block13(output_tensor)
        output_tensor = self.block14(output_tensor)
        output_tensor = self.block15(output_tensor)
        output_tensor = self.block16(output_tensor)

        output_tensor, h, w = self.conv4(output_tensor)
        output_tensor = ttnn.relu6(output_tensor)
        if output_tensor.is_sharded():
            output_tensor = ttnn.sharded_to_interleaved(output_tensor, ttnn.L1_MEMORY_CONFIG)
        output_tensor = ttnn.to_layout(output_tensor, layout=ttnn.ROW_MAJOR_LAYOUT)
        output_tensor = ttnn.reshape(output_tensor, (self.batchsize, h, w, output_tensor.shape[3]))

        output_tensor = ttnn.global_avg_pool2d(output_tensor)

        output_tensor = ttnn.reshape(output_tensor, (self.batchsize, -1))

        output_tensor = ttnn.linear(output_tensor, self.l1_weight, bias=self.l1_bias)

        return output_tensor

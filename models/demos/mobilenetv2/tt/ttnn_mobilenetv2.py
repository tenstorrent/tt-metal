# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.mobilenetv2.tt.common import TtInvertedResidual, TtMobileNetV2Conv2D


class TtMobileNetV2:
    def __init__(self, model_params, device, batchsize) -> None:
        self.device = device
        self.model_parameters = model_params
        self.batchsize = batchsize

        self.conv1 = TtMobileNetV2Conv2D(
            [3, 2, 1, 32],
            (model_params["fused_conv_0_weight"], model_params["fused_conv_0_bias"]),
            device,
            batchsize,
            deallocate_activation=True,
            # enable_split_reader = True
            reshard_if_not_optimal=False,
        )
        self.conv2 = TtMobileNetV2Conv2D(
            [3, 1, 1, 32],
            (model_params["fused_conv_1_weight"], model_params["fused_conv_1_bias"]),
            device,
            batchsize,
            groups=32,
        )
        self.conv3 = TtMobileNetV2Conv2D(
            [1, 1, 0, 16], (model_params["conv_0_weight"], model_params["conv_0_bias"]), device, batchsize
        )

        self.block1 = TtInvertedResidual(
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
        self.block2 = TtInvertedResidual(
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
        self.block3 = TtInvertedResidual(
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
        self.block4 = TtInvertedResidual(
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
        self.block5 = TtInvertedResidual(
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
        self.block6 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=2,
            in_channels=32,
            out_channels=64,
            id=6,
            block_shard=True,
        )
        self.block7 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=64,
            out_channels=64,
            id=7,
            block_shard=True,
        )
        self.block8 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=64,
            out_channels=64,
            id=8,
            block_shard=True,
        )
        self.block9 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=64,
            out_channels=64,
            id=9,
            block_shard=True,
        )
        self.block10 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=64,
            out_channels=96,
            id=10,
            block_shard=True,
        )
        self.block11 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=96,
            out_channels=96,
            id=11,
            block_shard=True,
        )
        self.block12 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=96,
            out_channels=96,
            id=12,
            block_shard=True,
        )
        self.block13 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=2,
            in_channels=96,
            out_channels=160,
            id=13,
            block_shard=True,
        )
        self.block14 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=160,
            out_channels=160,
            id=14,
            block_shard=True,
        )
        self.block15 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=160,
            out_channels=160,
            id=15,
            block_shard=True,
        )
        self.block16 = TtInvertedResidual(
            model_params,
            device,
            batchsize,
            expand_ratio=6,
            stride=1,
            in_channels=160,
            out_channels=320,
            id=16,
            block_shard=True,
        )

        self.conv4 = TtMobileNetV2Conv2D(
            [1, 1, 0, 1280],
            (model_params["fused_conv_34_weight"], model_params["fused_conv_34_bias"]),
            device,
            batchsize,
            width_shard=True,
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

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(8, 8),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )
        grid_size = (8, 8)
        shard_grid = ttnn.CoreRangeSet(
            {
                ttnn.CoreRange(
                    ttnn.CoreCoord(0, 0),
                    ttnn.CoreCoord(grid_size[0] - 1, grid_size[1] - 1),
                )
            }
        )
        shard_shape = [32, 32]
        shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
        width_sharded_mem_config = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec
        )
        output_tensor = ttnn.to_memory_config(output_tensor, width_sharded_mem_config)
        output_tensor = ttnn.linear(
            output_tensor,
            self.l1_weight,
            bias=self.l1_bias,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=matmul_config,
            compute_kernel_config=compute_config,
        )

        return output_tensor

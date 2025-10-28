# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.utility_functions import nearest_32
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
            enable_act_double_buffer=True,
            reshard_if_not_optimal=False,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        )
        self.conv2 = TtMobileNetV2Conv2D(
            [3, 1, 1, 32],
            (model_params["fused_conv_1_weight"], model_params["fused_conv_1_bias"]),
            device,
            batchsize,
            groups=32,
            enable_act_double_buffer=True,
            deallocate_activation=True,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        )
        self.conv3 = TtMobileNetV2Conv2D(
            [1, 1, 0, 16],
            (model_params["conv_0_weight"], model_params["conv_0_bias"]),
            device,
            batchsize,
            enable_act_double_buffer=True,
            deallocate_activation=True,
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
            deallocate_activation=True,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU6),
        )
        self.l1_weight = model_params["classifier_1_weight"]
        self.l1_bias = model_params["classifier_1_bias"]

    def __call__(
        self,
        x,
    ):
        output_tensor, h, w = self.conv1(x)
        output_tensor, h, w = self.conv2(output_tensor)
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

        num_cores = output_tensor.shape[3] // 32
        grid = ttnn.num_cores_to_corerangeset(num_cores, self.device.compute_with_storage_grid_size())
        width_mem_config = ttnn.create_sharded_memory_config_(
            [nearest_32(output_tensor.shape[2]), output_tensor.shape[3] // num_cores],
            grid,
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.ShardOrientation.ROW_MAJOR,
            tile_layout=True,
            use_height_and_width_as_shard_shape=True,
        )
        output_tensor = ttnn.to_memory_config(output_tensor, width_mem_config)
        output_tensor = ttnn.avg_pool2d(
            input_tensor=output_tensor,
            batch_size=self.batchsize,
            input_h=h,
            input_w=w,
            channels=output_tensor.shape[3],
            kernel_size=[h, w],
            stride=[1, 1],
            padding=[0, 0, 0, 0],
            output_layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=ttnn.init_device_compute_kernel_config(
                self.device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
            ),
        )

        compute_config = ttnn.init_device_compute_kernel_config(
            self.device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        grid_size = grid.bounding_box().grid_size()
        matmul_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(grid_size.x, grid_size.y),
            in0_block_w=1,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=1,
            fuse_batch=True,
            fused_activation=None,
            mcast_in0=True,
        )

        output_tensor = ttnn.linear(
            output_tensor,
            self.l1_weight,
            bias=self.l1_bias,
            program_config=matmul_config,
            compute_kernel_config=compute_config,
        )
        output_tensor = ttnn.squeeze(output_tensor)

        return output_tensor

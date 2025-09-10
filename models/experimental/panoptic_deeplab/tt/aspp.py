# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from loguru import logger
from models.experimental.panoptic_deeplab.tt.common import TTConv2D, TTUpsample


class PanopticDeeplabASPP:
    def __init__(self, parameters, model_config, layer_optimisations=None) -> None:
        self.model_config = model_config

        # ASPP_0_Conv
        self.ASPP_0_Conv = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.ASPP_0_Conv,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=32,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            reshard_if_not_optimal=True,
        )

        # ASPP_1_Depthwise
        self.ASPP_1_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=6,
            dilation=6,
            groups=2048,
            parameters=parameters.ASPP_1_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=64,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            reallocate_halo_output=True,
            enable_split_reader=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
        )

        # ASPP_1_pointwise
        self.ASPP_1_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.ASPP_1_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reshard_if_not_optimal=True,
        )

        # ASPP_2_Depthwise
        self.ASPP_2_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=12,
            dilation=12,
            groups=2048,
            parameters=parameters.ASPP_2_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=1024,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            reallocate_halo_output=False,
            enable_split_reader=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
        )

        # ASPP_2_pointwise
        self.ASPP_2_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.ASPP_2_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reshard_if_not_optimal=True,
        )

        # ASPP_3_Depthwise
        self.ASPP_3_Depthwise = TTConv2D(
            kernel_size=3,
            stride=1,
            padding=18,
            dilation=18,
            groups=2048,
            parameters=parameters.ASPP_3_Depthwise,
            kernel_fidelity=model_config,
            activation="relu",
            act_block_h=512,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            # deallocate_activation=True,
            enable_split_reader=True,
            reallocate_halo_output=True,
            enable_act_double_buffer=True,
            enable_weights_double_buffer=True,
            reshard_if_not_optimal=True,
        )

        # ASPP_3_ pointwise
        self.ASPP_3_pointwise = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.ASPP_3_pointwise,
            kernel_fidelity=model_config,
            activation="relu",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            deallocate_activation=True,
            reshard_if_not_optimal=True,
        )
        # ASPP_4_Conv_1
        self.ASPP_4_Conv_1 = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.ASPP_4_Conv_1,
            kernel_fidelity=model_config,
            activation="relu",
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            shard_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            deallocate_activation=True,
            reallocate_halo_output=True,
            reshard_if_not_optimal=True,
        )

        # ASPP4_upsample
        self.ASPP4_upsample = TTUpsample(
            scale_factor=(32, 64),
            mode="bilinear",
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
        )

        # ASPP_project
        self.ASPP_project = TTConv2D(
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            parameters=parameters.ASPP_project,
            kernel_fidelity=model_config,
            activation="relu",
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=True,
            reshard_if_not_optimal=True,
        )

    def __call__(
        self,
        x,
        device,
    ):
        # ASPP branch
        logger.debug("Running ASPP_0_Conv")
        aspp0, shape = self.ASPP_0_Conv(device, x, (1, 32, 64, 2048))

        logger.debug("Running ASPP_1_Depthwise")
        aspp1_dw, shape = self.ASPP_1_Depthwise(device, x, (1, 32, 64, 2048))

        logger.debug("Running ASPP_1_pointwise")
        aspp1, shape = self.ASPP_1_pointwise(device, aspp1_dw, shape)

        logger.debug("Running ASPP_2_Depthwise")
        aspp2_dw, shape = self.ASPP_2_Depthwise(device, x, (1, 32, 64, 2048))

        logger.debug("Running ASPP_2_pointwise")
        aspp2, shape = self.ASPP_2_pointwise(device, aspp2_dw, shape)

        logger.debug("Running ASPP_3_Depthwise")
        aspp3_dw, shape = self.ASPP_3_Depthwise(device, x, (1, 32, 64, 2048))

        logger.debug("Running ASPP_3_pointwise")
        aspp3, shape = self.ASPP_3_pointwise(device, aspp3_dw, shape)

        logger.debug("Running ASPP_4_avg_pool")
        x = ttnn.reshape(x, [1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[-1]])

        aspp4 = ttnn.avg_pool2d(
            input_tensor=x,
            batch_size=1,
            input_h=32,
            input_w=64,
            channels=2048,
            kernel_size=(32, 64),
            stride=(1, 1),
            padding=(0, 0),
            applied_shard_scheme=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            in_place_halo=True,
            deallocate_input=True,
            reallocate_halo_output=True,
        )

        ttnn.deallocate(x, force=True)

        logger.debug("Running ASPP_4_Conv_1")
        shape = (1, 1, 1, 2048)
        aspp4_conv, shape = self.ASPP_4_Conv_1(device, aspp4, shape)
        ttnn.deallocate(aspp4, force=True)

        logger.debug("Running ASPP_4_upsample")
        aspp4_conv_upsample = self.ASPP4_upsample(
            device, aspp4_conv, [1, 1, 1, 256], reshape_output=True, dtype=ttnn.bfloat8_b
        )

        ttnn.deallocate(aspp4_conv, force=True)

        logger.debug("Running ASPP_concat")
        aspp_concat = ttnn.concat(
            [aspp0, aspp1, aspp2, aspp3, aspp4_conv_upsample],
            dim=3,
        )

        logger.debug("Running ASPP_project")
        shape = (1, 32, 64, 1280)
        output, shape = self.ASPP_project(device, aspp_concat, shape)

        logger.debug("finished with ttnn imp")

        return output

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List, Optional
from dataclasses import dataclass

from models.experimental.retinanet.tt.utils import (
    _create_conv_config_from_params,
    override_conv_config,
    Conv2dNormActivation,
)
from models.tt_cnn.tt.builder import (
    TtConv2d,
    HeightShardedStrategyConfiguration,
    AutoShardedStrategyConfiguration,
)


@dataclass
class RetinaNetHeadOptimizer:
    fpn0_conv_blocks: dict
    fpn1_conv_blocks: dict
    fpn2_conv_blocks: dict
    fpn3_conv_blocks: dict
    fpn4_conv_blocks: dict

    fpn0_final_conv: dict
    fpn1_final_conv: dict
    fpn2_final_conv: dict
    fpn3_final_conv: dict
    fpn4_final_conv: dict


retinanet_head_optimizations = {
    "optimized": RetinaNetHeadOptimizer(
        fpn0_conv_blocks={"sharding_strategy": AutoShardedStrategyConfiguration()},
        fpn0_final_conv={"sharding_strategy": AutoShardedStrategyConfiguration()},
        fpn1_conv_blocks={"sharding_strategy": AutoShardedStrategyConfiguration()},
        fpn1_final_conv={"sharding_strategy": AutoShardedStrategyConfiguration()},
        fpn2_conv_blocks={"sharding_strategy": AutoShardedStrategyConfiguration()},
        fpn2_final_conv={"sharding_strategy": AutoShardedStrategyConfiguration()},
        fpn3_conv_blocks={"sharding_strategy": AutoShardedStrategyConfiguration()},
        fpn3_final_conv={"sharding_strategy": AutoShardedStrategyConfiguration()},
        fpn4_conv_blocks={"sharding_strategy": HeightShardedStrategyConfiguration(act_block_h_override=32)},
        fpn4_final_conv={"sharding_strategy": HeightShardedStrategyConfiguration(act_block_h_override=32)},
    ),
}


class TtnnRetinaNetRegressionHead:
    def __init__(
        self,
        parameters: dict,
        device: ttnn.Device,
        in_channels: int = 256,
        num_anchors: int = 9,
        batch_size: int = 1,
        input_shapes: List[tuple] = None,
        model_config: dict = None,
        optimization_profile: str = "optimized",
    ):
        self.device = device
        self.in_channels = in_channels
        self.num_anchors = num_anchors
        self.batch_size = batch_size
        self.input_shapes = input_shapes if input_shapes is not None else [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]
        self.model_config = model_config
        self.optimization_profile = optimization_profile

        self.parameters = parameters
        self.opt_config = retinanet_head_optimizations[optimization_profile]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=model_config.get("MATH_FIDELITY", ttnn.MathFidelity.HiFi4),
            math_approx_mode=model_config.get("MATH_APPROX_MODE", False),
        )

        self.grid_size = ttnn.CoreGrid(y=8, x=8)
        input_mask_tensor = ttnn.create_group_norm_input_mask(in_channels, 32, self.grid_size.y)
        self.input_mask_tensor = input_mask_tensor.to(device, ttnn.DRAM_MEMORY_CONFIG)

        self.fpn_optimizer_configs = {
            0: (self.opt_config.fpn0_conv_blocks, self.opt_config.fpn0_final_conv),
            1: (self.opt_config.fpn1_conv_blocks, self.opt_config.fpn1_final_conv),
            2: (self.opt_config.fpn2_conv_blocks, self.opt_config.fpn2_final_conv),
            3: (self.opt_config.fpn3_conv_blocks, self.opt_config.fpn3_final_conv),
            4: (self.opt_config.fpn4_conv_blocks, self.opt_config.fpn4_final_conv),
        }

        self.conv_blocks_by_fpn = {}
        self.final_convs_by_fpn = {}

        for fpn_idx in range(5):
            conv_opt, final_opt = self.fpn_optimizer_configs[fpn_idx]
            H, W = self.input_shapes[fpn_idx]

            conv_blocks = []
            for conv_idx in range(4):
                conv_params = self.parameters["conv"].get(str(conv_idx), {}).get("0", None)
                if conv_params is None:
                    conv_params = self.parameters["conv"][conv_idx]

                conv_block = Conv2dNormActivation(
                    parameters=conv_params,
                    device=self.device,
                    in_channels=self.in_channels,
                    out_channels=self.in_channels,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                    num_groups=32,
                    grid_size=self.grid_size,
                    input_mask=self.input_mask_tensor,
                    model_config=self.model_config,
                    compute_config=self.compute_config,
                    conv_config=conv_opt,
                    batch_size=self.batch_size,
                    input_height=H,
                    input_width=W,
                )
                conv_blocks.append(conv_block)

            self.conv_blocks_by_fpn[fpn_idx] = conv_blocks

            conv_final_config = _create_conv_config_from_params(
                input_height=H,
                input_width=W,
                in_channels=self.in_channels,
                out_channels=self.num_anchors * 4,
                kernel_size=(3, 3),
                batch_size=self.batch_size,
                parameters=self.parameters["bbox_reg"],
                stride=(1, 1),
                padding=(1, 1),
            )

            if final_opt:
                conv_final_config = override_conv_config(conv_final_config, final_opt)

            self.final_convs_by_fpn[fpn_idx] = TtConv2d(conv_final_config, self.device)

    def forward(
        self,
        feature_maps: List[ttnn.Tensor],
        batch_size: Optional[int] = None,
        input_shapes: Optional[List[tuple]] = None,
    ) -> ttnn.Tensor:
        current_batch_size = batch_size if batch_size is not None else self.batch_size
        current_input_shapes = input_shapes if input_shapes is not None else self.input_shapes

        if current_input_shapes is None:
            current_input_shapes = [(fm.shape[1], fm.shape[2]) for fm in feature_maps]

        all_bbox_regression = []

        for fpn_idx, (feature_map, (H, W)) in enumerate(zip(feature_maps, current_input_shapes)):
            conv_blocks = self.conv_blocks_by_fpn[fpn_idx]
            conv_final = self.final_convs_by_fpn[fpn_idx]

            x = feature_map
            for conv_block in conv_blocks:
                x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                x = conv_block(x)

            bbox_regression, [H_out, W_out] = conv_final(x, return_output_dim=True)

            N, _, _, _ = bbox_regression.shape
            bbox_regression = ttnn.to_memory_config(bbox_regression, ttnn.DRAM_MEMORY_CONFIG)
            bbox_regression = ttnn.sharded_to_interleaved(bbox_regression, ttnn.DRAM_MEMORY_CONFIG)
            bbox_regression = ttnn.reshape(bbox_regression, (N, H_out, W_out, self.num_anchors, 4))
            bbox_regression = ttnn.reshape(bbox_regression, (N, H_out * W_out * self.num_anchors, 4))

            all_bbox_regression.append(bbox_regression)

        output = ttnn.concat(all_bbox_regression, dim=1)
        return output

    def __call__(self, feature_maps: List[ttnn.Tensor], **kwargs) -> ttnn.Tensor:
        return self.forward(feature_maps, **kwargs)

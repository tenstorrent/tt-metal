# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import ttnn
from dataclasses import dataclass
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    AutoShardedStrategyConfiguration,
    BlockShardedStrategyConfiguration,
    HeightShardedStrategyConfiguration,
    L1FullSliceStrategyConfiguration,
)
from models.experimental.SSD512.tt.utils import Conv2dOperation, Maxpool2DOperation, override_conv_config


@dataclass
class VGGBackboneOptimizationConfig:
    conv1: dict
    conv2: dict
    conv3: dict
    conv4: dict
    conv5: dict
    conv8: dict
    conv9: dict
    conv10: dict
    conv11: dict
    conv12: dict
    conv13: dict
    conv14: dict
    conv15: dict
    conv16: dict
    conv17: dict
    conv18: dict
    conv19: dict


vgg_backbone_optimizations = VGGBackboneOptimizationConfig(
    conv1={
        "sharding_strategy": HeightShardedStrategyConfiguration(act_block_h_override=15 * 32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv2={
        "sharding_strategy": HeightShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv3={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv4={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv5={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv8={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv9={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv10={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv11={
        "sharding_strategy": AutoShardedStrategyConfiguration(),
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv12={
        "sharding_strategy": BlockShardedStrategyConfiguration(reshard_if_not_optimal=True, act_block_h_override=32),
        "enable_act_double_buffer": False,
        "enable_weights_double_buffer": False,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv13={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv14={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv15={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv16={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv17={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv18={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
    conv19={
        "sharding_strategy": BlockShardedStrategyConfiguration(act_block_h_override=32),
        "slice_strategy": L1FullSliceStrategyConfiguration(),
        "enable_act_double_buffer": True,
        "enable_weights_double_buffer": True,
        "deallocate_activation": True,
        "reallocate_halo_output": True,
    },
)


# VGG backbone network with per-layer optimization configurations for SSD512
class TtVGGBackbone:
    def __init__(self, config_layers, device, batch_size: int):
        self.batch_size = batch_size
        self.device = device

        layers = []

        # Build layers with per-conv optimization overrides
        for i, conv_config in enumerate(config_layers):
            if isinstance(conv_config, Conv2dConfiguration):
                optimization_key = f"conv{i+1}"
                override_dict = getattr(vgg_backbone_optimizations, optimization_key, {})
                updated_config = override_conv_config(conv_config, override_dict)

                layers.append(
                    Conv2dOperation(
                        device=device,
                        conv_config=updated_config,
                        activation_layer=ttnn.relu,
                    )
                )
            elif isinstance(conv_config, MaxPool2dConfiguration):
                layers.append(
                    Maxpool2DOperation(
                        device=device,
                        conv_config=conv_config,
                    )
                )
            else:
                raise ValueError(f"Unsupported layer configuration found: {type(conv_config)}")

        self.block = layers

    # Forward pass through VGG backbone
    def __call__(self, device, input, return_residual_sources=False):
        residual_sources = []
        for i, layer in enumerate(self.block):
            if i == 0:
                result = layer(device, input)
            else:
                result = layer(device, result)

            # Extract feature map at layer 12 for SSD512 multi-scale detection
            if i == 12:
                residual_sources.append(result)

        if return_residual_sources:
            return result, residual_sources
        return result

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.tt_cnn.tt.builder import BlockShardedStrategyConfiguration
from models.experimental.efficientdetd0.tt.utils import TtSeparableConvBlock


class TtClassifier:
    def __init__(
        self,
        device,
        parameters,
        module_args,
        num_classes,
        num_layers,
        pyramid_levels=5,
    ):
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = [
            [
                TtSeparableConvBlock(
                    device=device,
                    parameters=parameters.conv_list[j][i],
                    module_args=module_args.conv_list[j][i],
                    sharding_strategy=BlockShardedStrategyConfiguration(reshard_if_not_optimal=True),
                    deallocate_activation=True,
                    dtype=ttnn.bfloat8_b,
                )
                for i in range(num_layers)
            ]
            for j in range(pyramid_levels)
        ]

        self.header_list = [
            TtSeparableConvBlock(
                device=device,
                parameters=parameters.header_list[j],
                module_args=module_args.header[j],
                sharding_strategy=BlockShardedStrategyConfiguration(reshard_if_not_optimal=True),
                deallocate_activation=True,
                dtype=ttnn.bfloat8_b,
            )
            for j in range(pyramid_levels)
        ]

    def __call__(self, inputs):
        feats = []
        for feat, conv_list, header in zip(inputs, self.conv_list, self.header_list):
            for conv in conv_list:
                feat = conv(feat)
                feat = feat * ttnn.sigmoid_accurate(feat)
            feat = header(feat)
            feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG, dtype=ttnn.float32)
            feat = ttnn.reshape(feat, (feat.shape[0], -1, self.num_classes))
            feats.append(feat)
        concated_feats = ttnn.concat(feats, dim=1)
        for t in feats:
            ttnn.deallocate(t)
        return ttnn.sigmoid_accurate(concated_feats)

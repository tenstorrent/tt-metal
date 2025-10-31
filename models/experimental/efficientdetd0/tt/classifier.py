# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.efficientdetd0.tt.utils import SeparableConvBlock, BatchNorm2d


class Classifier:
    def __init__(
        self,
        device,
        parameters,
        conv_params,
        num_anchors,
        num_classes,
        num_layers,
        pyramid_levels=5,
    ):
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.conv_list = [
            SeparableConvBlock(
                device=device,
                parameters=parameters.conv_list[i],
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                conv_params=conv_params[i],
                batch=1,
                deallocate_activation=True,
            )
            for i in range(num_layers)
        ]
        self.bn_list = [
            [BatchNorm2d(device=device, parameters=parameters.bn_list[j][i]) for i in range(num_layers)]
            for j in range(pyramid_levels)
        ]
        self.header = SeparableConvBlock(
            device=device,
            parameters=parameters.header,
            shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
            conv_params=conv_params.header,
            batch=1,
            deallocate_activation=True,
        )

    def __call__(self, inputs):
        feats = []
        for feat, bn_list in zip(inputs, self.bn_list):
            for bn, conv in zip(bn_list, self.conv_list):
                feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)
                print(f"TTNN PreConv {feat.shape}")
                feat = conv(feat)
                print(f"TTNN PostConv {feat.shape}")
                feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)
                feat = bn(feat)
                feat = feat * ttnn.sigmoid_accurate(feat, True)
            feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)
            feat = self.header(feat)

            feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)
            feat = ttnn.permute(feat, (0, 3, 1, 2))
            feat = ttnn.permute(feat, (0, 2, 3, 1))
            # feat = ttnn.permute(feat, (0, 1, 2, 3))
            # feat = ttnn.reshape(feat, (feat.shape[0], feat.shape[1], feat.shape[2], self.num_anchors, self.num_classes))
            feat = ttnn.reshape(feat, (feat.shape[1], feat.shape[2], self.num_anchors, self.num_classes))
            feat = ttnn.reshape(feat, (feat.shape[0], -1, self.num_classes))
            feats.append(feat)

        feats = ttnn.concat(feats, dim=1)
        feats = ttnn.sigmoid_accurate(feats, True)

        return feats

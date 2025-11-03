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
            [
                SeparableConvBlock(
                    device=device,
                    parameters=parameters.conv_list[i],
                    shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                    conv_params=conv_params.conv_list[j][i],
                    batch=1,
                    deallocate_activation=True,
                )
                for i in range(num_layers)
            ]
            for j in range(pyramid_levels)
        ]

        self.bn_list = [
            [BatchNorm2d(device=device, parameters=parameters.bn_list[j][i]) for i in range(num_layers)]
            for j in range(pyramid_levels)
        ]
        self.header_list = [
            SeparableConvBlock(
                device=device,
                parameters=parameters.header,
                shard_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                conv_params=conv_params.header_list[j],
                batch=1,
                deallocate_activation=True,
            )
            for j in range(pyramid_levels)
        ]

    def __call__(self, inputs):
        feats = []
        for feat, conv_list, bn_list, header in zip(inputs, self.conv_list, self.bn_list, self.header_list):
            conv_in_shape = feat.shape
            for bn, conv in zip(bn_list, conv_list):
                feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)
                feat = conv(feat)
                feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)
                feat = ttnn.to_layout(feat, ttnn.TILE_LAYOUT)
                feat = ttnn.reshape(feat, conv_in_shape)
                feat = ttnn.permute(feat, (0, 3, 1, 2))
                # feat = bn(feat)
                feat = ttnn.permute(feat, (0, 2, 3, 1))
                feat = feat * ttnn.sigmoid_accurate(feat, True)
            feat = header(feat)
            feat = ttnn.to_memory_config(feat, ttnn.DRAM_MEMORY_CONFIG)

            feat = ttnn.reshape(feat, (feat.shape[1], feat.shape[2], self.num_anchors, self.num_classes))
            feat = ttnn.reshape(feat, (feat.shape[0], -1, self.num_classes))
            feats.append(feat)

        feats = ttnn.concat(feats, dim=1)
        # feats = ttnn.sigmoid_accurate(feats)

        return feats

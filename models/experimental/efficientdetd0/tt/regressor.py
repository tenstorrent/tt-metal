# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.efficientdetd0.tt.utils import SeparableConvBlock


class Regressor:
    def __init__(
        self,
        device,
        parameters,
        conv_params,
        num_layers,
    ):
        self.num_layers = num_layers

        self.conv_list = [
            SeparableConvBlock(
                device=device,
                parameters=parameters.conv_list[i],
                shard_layout=None,
                conv_params=conv_params.conv_list[i],
                batch=1,
                deallocate_activation=True,
            )
            for i in range(num_layers)
        ]
        self.header = SeparableConvBlock(
            device=device,
            parameters=parameters.header,
            shard_layout=None,
            conv_params=conv_params.header,
            batch=1,
            deallocate_activation=True,
        )

    def __call__(self, inputs):
        feats = []
        for feat in inputs:
            for conv in self.conv_list:
                feat = conv(feat)
                feat = feat * ttnn.sigmoid_accurate(feat, True)
            feat = self.header(feat)

            feat = ttnn.permute(feat, (0, 2, 3, 1))
            feat = ttnn.reshape(feat, (feat.shape[0], -1, 4))

            feats.append(feat)

        feats = ttnn.concat(feats, dim=1)

        return feats

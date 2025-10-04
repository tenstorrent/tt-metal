# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from typing import List
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck


class Ttstages:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        # layer_optimisations=neck_optimisations,
    ) -> None:
        # self.inplanes = 32

        # self.layer = self._make_layer(
        #     parameters=parameters.image_encoder.features.layer1,
        #     planes=72,
        #     blocks=2,
        #     stride=stride,
        #     groups=3,
        #     model_config=model_config,
        # )

        print(parameters)
        self.layer = TTRegNetBottleneck(
            parameters=parameters.layer1.b1,
            # parameters=parameters.image_encoder.features.layer1.b1,
            model_config=model_config,
            stride=stride,
            downsample=True,
            groups=3,
        )

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config=None,
    ) -> List[TTRegNetBottleneck]:
        layers = []
        self.inplanes = 32

        # First block (may have downsample)
        downsample = stride != 1 or self.inplanes != planes
        print(f"downsampleeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee ..  .......{downsample}")
        layers.append(
            TTRegNetBottleneck(
                parameters=parameters["b1"],
                model_config=model_config,
                stride=stride,
                downsample=downsample,
                groups=groups,
            )
        )
        self.inplanes = planes

        # Remaining blocks
        for block_num in range(1, blocks):
            block_name = f"b{block_num + 1}"
            layers.append(
                TTRegNetBottleneck(
                    parameters=parameters[block_name],
                    model_config=model_config,
                    stride=1,
                    downsample=False,
                    groups=groups,
                )
            )

        return layers

    def __call__(self, x, device):
        # Process image input
        # for block in self.layer:
        #     x = block(x, device)
        #     return x
        x = self.layer(x, device)

        return x

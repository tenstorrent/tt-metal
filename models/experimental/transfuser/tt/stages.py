# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List
from models.experimental.transfuser.tt.bottleneck import TTRegNetBottleneck


shard_dict = {
    # stage_name : shard_layout
    "layer1": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "layer2": ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
    "layer3": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
    "layer4": ttnn.TensorMemoryLayout.WIDTH_SHARDED,
}


class Ttstages:
    def __init__(
        self,
        parameters,
        stride,
        model_config,
        stage_name,
        # layer_optimisations=neck_optimisations,
    ) -> None:
        self.inplanes = 32
        self.layer = self._make_layer(
            parameters=parameters,
            planes=72,
            blocks=len(parameters.keys()),
            stride=stride,
            groups=3,
            model_config=model_config,
            stage_name=stage_name,
        )

    def _make_layer(
        self,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        groups: int = 1,
        model_config=None,
        stage_name=None,
    ) -> List[TTRegNetBottleneck]:
        layers = []
        self.inplanes = 32

        shard_layout = shard_dict[stage_name]

        # First block (may have downsample)
        downsample = stride != 1 or self.inplanes != planes
        layers.append(
            TTRegNetBottleneck(
                parameters=parameters["b1"],
                model_config=model_config,
                stride=stride,
                downsample=downsample,
                groups=groups,
                shard_layout=shard_layout,
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
                    shard_layout=shard_layout,
                )
            )

        return layers

    def __call__(self, x, device):
        # Process image input
        for block in self.layer:
            x = block(x, device)

        return x

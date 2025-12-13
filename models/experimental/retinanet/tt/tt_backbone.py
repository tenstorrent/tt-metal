# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
from typing import List, Optional
from models.experimental.retinanet.tt.tt_bottleneck import TTBottleneck, get_bottleneck_optimisation
from models.experimental.retinanet.tt.tt_stem import resnet50Stem, neck_optimisations
from models.experimental.retinanet.tt.tt_fpn import resnet50Fpn, fpn_optimisations

from loguru import logger


class TTBackbone:
    def __init__(self, parameters, model_config, name="backbone"):
        layers = [3, 4, 6, 3]
        self.inplanes = 64
        parameters_fpn = parameters.fpn
        parameters_body = parameters.body

        # stem
        class StemParams:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)

        stem_layers = ["conv1", "bn1", "relu", "maxpool"]
        parameters_body["stem"] = StemParams({k: parameters_body.pop(k) for k in stem_layers if k in parameters_body})

        self.stem = resnet50Stem(
            parameters_body.stem,
            stride=1,
            model_config=model_config,
            layer_optimisations=neck_optimisations,
        )
        # Four bottleneck stages (layer1, layer2, layer3, layer4)

        self.layer1 = self._make_layer(
            name=f"{name}.layer1",
            parameters=parameters_body.layer1,
            planes=64,
            blocks=layers[0],
            stride=1,
            dilate_config=None,
            model_config=model_config,
            layer_optimisations=get_bottleneck_optimisation("layer1"),
        )
        self.layer2 = self._make_layer(
            name=f"{name}.layer2",
            parameters=parameters_body.layer2,
            planes=128,
            blocks=layers[1],
            stride=2,
            dilate_config=None,
            model_config=model_config,
            layer_optimisations=get_bottleneck_optimisation("layer2"),
        )
        self.layer3 = self._make_layer(
            name=f"{name}.layer3",
            parameters=parameters_body.layer3,
            planes=256,
            blocks=layers[2],
            stride=2,
            dilate_config=None,
            model_config=model_config,
            layer_optimisations=get_bottleneck_optimisation("layer3"),
        )
        self.layer4 = self._make_layer(
            name=f"{name}.layer4",
            parameters=parameters_body.layer4,
            planes=512,
            blocks=layers[3],
            stride=2,
            dilate_config=None,
            model_config=model_config,
            layer_optimisations=get_bottleneck_optimisation("layer4"),
        )
        self.fpn = resnet50Fpn(
            parameters=parameters_fpn,
            model_config=model_config,
            layer_optimisations=fpn_optimisations,
        )

    def _make_layer(
        self,
        name: str,
        parameters,
        planes: int,
        blocks: int,
        stride: int,
        dilate_config: Optional[List[int]] = None,
        model_config=None,
        layer_optimisations=get_bottleneck_optimisation("default"),
    ) -> List[TTBottleneck]:
        if dilate_config is None:
            dilate_config = [1] * blocks
        layers = []
        layers.append(
            TTBottleneck(
                parameters=getattr(parameters, "0", None),
                downsample=stride != 1 or self.inplanes != planes * TTBottleneck.expansion,
                stride=stride,
                model_config=model_config,
                dilation=dilate_config[0],
                name=f"{name}.0",
                layer_optimisations=layer_optimisations,
            )
        )
        self.inplanes = planes * TTBottleneck.expansion
        for block_num in range(1, blocks):
            layers.append(
                TTBottleneck(
                    parameters=getattr(parameters, f"{block_num}", None),
                    downsample=False,
                    stride=1,
                    model_config=model_config,
                    dilation=dilate_config[block_num],
                    name=f"{name}.{block_num}",
                    layer_optimisations=layer_optimisations,
                )
            )
        return layers

    def __call__(self, x, device):
        x = self.stem(x, device)
        shape = x.shape

        for i, block in enumerate(self.layer1):
            x, shape = block(x, device, shape)

        for i, block in enumerate(self.layer2):
            x, shape = block(x, device, shape)

        c3 = ttnn.clone(x)
        c3 = ttnn.to_memory_config(c3, ttnn.DRAM_MEMORY_CONFIG)

        for i, block in enumerate(self.layer3):
            x, shape = block(x, device, shape)

        c4 = ttnn.clone(x)
        c4 = ttnn.to_memory_config(c4, ttnn.DRAM_MEMORY_CONFIG)

        for i, block in enumerate(self.layer4):
            x, shape = block(x, device, shape)

        c5 = ttnn.clone(x)
        c5 = ttnn.to_memory_config(c5, ttnn.DRAM_MEMORY_CONFIG)
        out = {"c3": c3, "c4": c4, "c5": c5}
        out = self.fpn(out, device)
        logger.debug("✅✅✅ FPN Complete ✅✅✅")

        return out

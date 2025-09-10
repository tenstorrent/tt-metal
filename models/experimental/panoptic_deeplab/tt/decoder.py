# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

from models.experimental.panoptic_deeplab.tt.aspp import PanopticDeeplabASPP as TTASPP
from models.experimental.panoptic_deeplab.tt.head import TTHead
from models.experimental.panoptic_deeplab.tt.res_block import TTRes
from models.experimental.panoptic_deeplab.tt.res_block import res_layer_optimisations
from models.experimental.panoptic_deeplab.tt.head import head_layer_optimisations
from dataclasses import dataclass
import ttnn


@dataclass
class DecoderOptimizer:
    res_layer_optimisations: dict
    head_layer_optimisations: dict
    shape: tuple


decoder_layer_optimisations = {
    "default": DecoderOptimizer(
        res_layer_optimisations=res_layer_optimisations["default"],
        head_layer_optimisations=head_layer_optimisations["default"],
        shape=(0, 0, 0, 0),
    ),
    "Semantics_head": DecoderOptimizer(
        res_layer_optimisations={
            "res3": res_layer_optimisations["semantics_Res3"],
            "res2": res_layer_optimisations["semantics_Res2"],
        },
        head_layer_optimisations={
            "head_1": head_layer_optimisations["semantic_head"],
        },
        shape=(1, 128, 256, 256),
    ),
    "instance_head": DecoderOptimizer(
        res_layer_optimisations={
            "res3": res_layer_optimisations["instance_Res3"],
            "res2": res_layer_optimisations["instance_Res2"],
        },
        head_layer_optimisations={
            "head_1": head_layer_optimisations["instance_offset_head"],
            "head_2": head_layer_optimisations["instance_center_head"],
        },
        shape=(1, 128, 256, 128),
    ),
}


class TTDecoder:
    def __init__(
        self, parameters, model_config, layer_optimisations=decoder_layer_optimisations["default"], name="default"
    ) -> None:
        super().__init__()
        self.shape = layer_optimisations.shape
        self.aspp = TTASPP(parameters.aspp, model_config, layer_optimisations=None)
        self.res3 = TTRes(
            parameters.res3,
            model_config,
            layer_optimisations=layer_optimisations.res_layer_optimisations["res3"],
        )
        self.res2 = TTRes(
            parameters.res2,
            model_config,
            layer_optimisations=layer_optimisations.res_layer_optimisations["res2"],
        )
        self.head = TTHead(
            parameters.head_1,
            model_config,
            layer_optimisations=layer_optimisations.head_layer_optimisations["head_1"],
        )
        self.name = name

        if self.shape[-1] == 128:
            self.head_2 = TTHead(
                parameters.head_2,
                model_config,
                layer_optimisations=layer_optimisations.head_layer_optimisations["head_2"],
            )
        if self.name == "Semantics_head":
            self.res3_upsample_channels = 256
            self.res2_upsample_channels = 256
        else:
            self.res3_upsample_channels = 256
            self.res2_upsample_channels = 128

    def __call__(self, x, res3, res2, upsample_channels, device):
        y = self.aspp(x, device)
        y = self.res3(y, res3, self.res3_upsample_channels, device)
        y = self.res2(y, res2, self.res2_upsample_channels, device)

        if self.name == "instance_head":
            activation_copy = ttnn.clone(y)
        out = self.head(y, device)

        if self.name == "instance_head":
            y = self.head_2(activation_copy, device)
            return out, y

        return out, None

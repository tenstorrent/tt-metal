# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ttnn.model_preprocessing import preprocess_model_parameters

from models.experimental.liquid.tt.vision_encoder import siglip2_encoder
from models.experimental.liquid.tt.lm_backbone import LFMBackbone


class LiquidVL:
    def __init__(self, device, parameters, model_args):
        self.device = device
        self.parameters = parameters
        self.model_args = model_args
        self.backbone = LFMBackbone(device, model_args, parameters.lm_backbone)

    def encode_image(self, pixel_values):
        _, visual_features = siglip2_encoder(pixel_values, self.parameters.vision_encoder)
        visual_features = ttnn.to_layout(visual_features, layout=ttnn.TILE_LAYOUT)
        projected = ttnn.linear(
            visual_features,
            self.parameters.projector.weight,
            bias=self.parameters.projector.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        return projected

    def generate(self, input_ids, image_features=None, max_new_tokens=64):
        if image_features is not None:
            combined = ttnn.concat([image_features, input_ids], dim=1)
        else:
            combined = input_ids
        return self.backbone(combined, mode="prefill")

    @staticmethod
    def preprocess_weights(vit_params, projector_weight, projector_bias, lm_params, device):
        params = {}
        params["vision_encoder"] = vit_params
        params["projector"] = {"weight": projector_weight, "bias": projector_bias}
        params["lm_backbone"] = lm_params
        return params

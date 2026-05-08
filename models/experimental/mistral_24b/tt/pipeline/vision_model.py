# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
This is the end-to-end architecture of the Mistral-24B vision model.

It brings together all components related to visual and MultiModalProjector together.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.mistral_24b.tt.pipeline.mistral_vision_tower import MistralVisionTower
from models.experimental.mistral_24b.tt.vision_mmp import TTMistral3MultiModalProjector

try:
    from tracy import signpost
except ImportError:

    def signpost(*args, **kwargs):
        pass


class TtMistralVisionTransformer(LightweightModule):
    def __init__(self, mesh_device, tt_ccl, state_dict, state_dict_prefix, dtype, model_args):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl

        self.vision_tower = MistralVisionTower(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            state_dict=state_dict,
            state_dict_prefix=state_dict_prefix,
            dtype=dtype,
            configuration=model_args,
        )

        self.mmp = TTMistral3MultiModalProjector(
            mesh_device=mesh_device,
            args=model_args,
            state_dict=state_dict,
            state_dict_prefix="multi_modal_projector.",
            dtype=dtype,
            eps=1e-05,  # layer_norm_eps
        )

    def forward(self, input_tensor, image_sizes):
        """
        input_tensor shape: (B, C, H, W)
        image_sizes: List of tuples [(height, width), ...] for each image in the batch
        """
        if image_sizes is None or len(image_sizes) == 0:
            raise ValueError("image_sizes must be provided and non-empty")

        signpost("Mistral24B::VisionModel::Start", f"image_sizes={image_sizes}")
        signpost("Mistral24B::VisionTower::Start")
        x = self.vision_tower(input_tensor, image_sizes=image_sizes)
        signpost("Mistral24B::VisionTower::End")
        signpost("Mistral24B::VisionModel::Squeeze::Start")
        x = ttnn.squeeze(ttnn.squeeze(x, 0), 0)
        signpost("Mistral24B::VisionModel::Squeeze::End")
        signpost("Mistral24B::MultimodalProjector::Start")
        x = self.mmp(x, image_sizes)
        signpost("Mistral24B::MultimodalProjector::End")
        signpost("Mistral24B::VisionModel::End")
        return x

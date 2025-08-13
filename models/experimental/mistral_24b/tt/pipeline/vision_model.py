"""
This is the end-to-end architecture of the Mistral-24B vision model.

It brings together all components related to visual and MultiModalProjector together.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule

from models.experimental.mistral_24b.tt.pipeline.mistral_vision_tower import MistralVisionTower
from models.experimental.mistral_24b.tt.vision_mmp import TTMistral3MultiModalProjector


class TtMistralVisionTransformer(LightweightModule):
    def __init__(self, mesh_device, state_dict, state_dict_prefix, dtype, model_args):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.vision_tower = MistralVisionTower(
            mesh_device=mesh_device,
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

    def forward(self, input_tensor, image_sizes=None):
        """
        input_tensor shape: (B, C, H, W)
        """

        x = self.vision_tower(input_tensor, image_sizes=image_sizes)
        x = ttnn.squeeze(ttnn.squeeze(x, 0), 0)
        x = self.mmp(x, image_sizes)
        return x

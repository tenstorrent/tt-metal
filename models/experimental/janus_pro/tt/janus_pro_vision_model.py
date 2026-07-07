"""
Vision tower for Janus-Pro-7B: the SigLIP vision encoder followed by the
vision-to-text aligner. Mirrors the Gemma vision transformer wrapper
(TtGemmaTransformerVision) but is Janus-owned.

HF reference: JanusModel.get_image_features, which runs
    aligner(vision_model(pixel_values).last_hidden_state)
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

from models.common.lightweightmodule import LightweightModule
from models.experimental.janus_pro.tt.janus_pro_vision_aligner import TtJanusProVisionAligner
from models.experimental.janus_pro.tt.janus_pro_vision_block import TtJanusProVisionModel


class TtJanusProTransformerVision(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        tt_ccl,
        state_dict_prefix,  # "model."
        dtype,
        configuration,
        weight_cache_path=None,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.configuration = configuration

        self.vision_model = TtJanusProVisionModel(
            mesh_device,
            state_dict=state_dict,
            tt_ccl=tt_ccl,
            state_dict_prefix=f"{state_dict_prefix}vision_model.",
            dtype=dtype,
            configuration=configuration,
        )

        self.aligner = TtJanusProVisionAligner(
            mesh_device=mesh_device,
            args=configuration,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}aligner.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=dtype,
        )

    def forward(self, images):
        # images: torch tensor (B, 3, 384, 384) — the vision encoder unfolds it in conv1.
        x = self.vision_model(images)
        x = self.aligner(x)
        return x

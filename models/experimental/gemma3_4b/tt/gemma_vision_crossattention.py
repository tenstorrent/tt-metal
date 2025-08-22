"""
This is the Vision Transformer Block for Gemma-3-4b-it.
This involves vision followed by MultiModalProjector processing
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


from models.common.lightweightmodule import LightweightModule
from models.experimental.gemma3_4b.tt.gemma_vision_model import TtSiglipGemmaVisionModel
from models.experimental.gemma3_4b.tt.mmp import TtGemma3MultiModalProjector


class TtGemmaTransformerVision(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        state_dict,
        state_dict_prefix,
        dtype,
        configuration,
        weight_cache_path=None,
        return_intermediate=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.model_config = configuration.get_model_config()

        self.dim = configuration.dim
        self.vision_dim = configuration.vision_dim
        self.image_res = configuration.image_size
        self.patch_size = configuration.vision_patch_size
        self.configuration = configuration

        self.vision_encoder = TtSiglipGemmaVisionModel(
            mesh_device,
            self.tt_ccl,
            state_dict,
            state_dict_prefix,
            dtype=dtype,
            configuration=configuration,
            weight_cache_path=configuration.weight_cache_path(dtype),
            return_intermediate=return_intermediate,
        )

        self.mmp = TtGemma3MultiModalProjector(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="multi_modal_projector",
            image_size=configuration.image_size,
            patch_size=configuration.vision_patch_size,
            hidden_size=configuration.vision_hidden_dim,
            mm_tokens_per_image=configuration.mm_tokens_per_image,
            weight_cache_path=configuration.weight_cache_path(dtype),
            layer_norm_eps=1e-06,  # layer_norm_eps
            dtype=dtype,
            configuration=configuration,
        )

    def forward(self, images):
        vision_tokens = self.vision_encoder(images)[0, :, :, :]

        vision_tokens = self.mmp(vision_tokens)
        return vision_tokens

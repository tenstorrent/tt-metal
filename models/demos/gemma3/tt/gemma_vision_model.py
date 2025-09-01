"""
This is the Vision Transformer Block for Gemma-3-4b-it.
This involves vision followed by MultiModalProjector processing
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0


from models.common.lightweightmodule import LightweightModule
from models.demos.gemma3.tt.gemma_vision_block import TtSiglipGemmaVisionModel
from models.demos.gemma3.tt.multi_modal_projector import TtGemma3MultiModalProjector


class TtGemmaTransformerVision(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        tt_ccl,
        state_dict_prefix,
        dtype,
        configuration,
        weight_cache_path=None,
        return_intermediate=None,
    ):
        super().__init__()

        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.model_config = configuration.get_model_config()

        self.dim = configuration.dim
        self.vision_dim = configuration.vision_dim
        self.image_res = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size
        self.configuration = configuration

        self.vision_encoder = TtSiglipGemmaVisionModel(
            mesh_device,
            state_dict=state_dict,
            tt_ccl=tt_ccl,
            state_dict_prefix=f"{state_dict_prefix}",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=dtype,
            configuration=configuration,
            return_intermediate=return_intermediate,
        )

        self.mmp = TtGemma3MultiModalProjector(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix="model.multi_modal_projector",
            image_size=configuration.vision_chunk_size,
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

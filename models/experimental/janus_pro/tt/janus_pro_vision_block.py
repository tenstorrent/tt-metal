"""
This is the Vision Tower Model for Janus-Pro-7B.
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.common.lightweightmodule import LightweightModule
from models.experimental.janus_pro.tt.janus_pro_image_transformer import TtJanusProImageTransformer
from models.experimental.janus_pro.tt.janus_pro_vision_embedding import TtJanusProVisionEmbeddings
from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm


class TtJanusProVisionModel(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        tt_ccl,
        state_dict_prefix,
        dtype,
        configuration,
        weight_cache_path=None,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device

        self.image_size = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size

        self.width = configuration.vision_dim
        self.layers = configuration.vision_n_layers
        self.heads = configuration.vision_attn_n_heads
        self.mlp_ratio = configuration.vision_mlp_ratio
        self.act_layer = configuration.vision_act_layer
        self.in_channels = configuration.vision_in_channels

        self.embeddings = TtJanusProVisionEmbeddings(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}embeddings.",
            dtype=dtype,
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_channels=self.in_channels,
            hidden_dim=self.width,
            bias=True,
        )

        # transformer
        self.encoder = TtJanusProImageTransformer(
            mesh_device=mesh_device,
            state_dict=state_dict,
            tt_ccl=tt_ccl,
            state_dict_prefix=f"{state_dict_prefix}encoder.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            dtype=dtype,
            configuration=configuration,
            layers=self.layers,
            block_key="layers",
        )

        self.prepare_residual_tensor_prefill = configuration.prepare_residual_tensor_prefill

        self.ln_post = TtLayerNorm(
            device=mesh_device,
            dim=self.width,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}ln_post.",
            weight_cache_path=configuration.weight_cache_path(dtype),
            weight_dtype=dtype,
            eps=configuration.norm_eps,
        )

    def forward(self, images):
        assert isinstance(
            images, torch.Tensor
        ), "VisionEncoder input must be a torch tensor because of unfold in self.conv1"

        x = self.embeddings(images)

        # SigLIP vision uses full bidirectional attention; there is no attention mask
        # (an all-zeros additive mask would be a no-op), so SDPA runs without one.
        x = self.encoder(x, mask=None)

        x = self.ln_post(x)

        return x

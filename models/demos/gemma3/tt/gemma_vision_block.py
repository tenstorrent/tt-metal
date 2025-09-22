"""
This is the Vision Tower Model for Gemma-3-4b-it.
"""

# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma3.tt.gemma_image_transformer import TtGemmaImageTransformer
from models.demos.gemma3.tt.siglip_vision_embedding import TtSiglipVisionEmbeddings
from models.tt_transformers.tt.multimodal.llama_layernorm import TtLayerNorm


class TtSiglipGemmaVisionModel(LightweightModule):
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

        self.image_size = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size

        self.width = configuration.vision_dim
        self.layers = configuration.vision_n_layers
        self.heads = configuration.vision_attn_n_heads
        self.mlp_ratio = configuration.vision_mlp_ratio
        self.act_layer = configuration.vision_act_layer
        self.in_channels = configuration.vision_in_channels
        self.n_global_layers = configuration.vision_n_global_layers
        self.return_intermediate = return_intermediate

        self.embeddings = TtSiglipVisionEmbeddings(
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
        self.encoder = TtGemmaImageTransformer(
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

        bsz, in_channel, h, w = images.shape

        x = self.embeddings(images)
        attention_mask = torch.zeros(bsz, 1, x.shape[1], x.shape[1])

        tt_mask = ttnn.from_torch(
            attention_mask,
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        x = self.encoder(
            x,
            mask=tt_mask,
        )

        x = self.ln_post(x)

        return x

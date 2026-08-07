"""
The SigLIP vision encoder of Janus-Pro-7B: embeddings, the block stack, then a final layer norm.
The aligner that follows it lives in `janus_pro_vision_model.py`.

HF reference: `vision_model` (`ModelArgs.reference_vision_model`).
"""

# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn

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

        self.image_size = configuration.vision_chunk_size
        self.patch_size = configuration.vision_patch_size

        self.width = configuration.vision_dim
        self.layers = configuration.vision_n_layers

        self.embeddings = TtJanusProVisionEmbeddings(
            mesh_device=mesh_device,
            state_dict=state_dict,
            state_dict_prefix=f"{state_dict_prefix}embeddings.",
            dtype=dtype,
            image_size=self.image_size,
            patch_size=self.patch_size,
            hidden_dim=self.width,
            configuration=configuration,
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
        )

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

        return self._encode(self.embeddings(images))

    def _encode(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # SigLIP vision uses full bidirectional attention; there is no attention mask
        # (an all-zeros additive mask would be a no-op), so SDPA runs without one.
        x = self.encoder(x, mask=None)
        return self.ln_post(x)

    def forward_device(self, patches: ttnn.Tensor) -> ttnn.Tensor:
        """Same as :meth:`forward` from patches already on device; traceable end to end."""
        return self._encode(self.embeddings.forward_device(patches))

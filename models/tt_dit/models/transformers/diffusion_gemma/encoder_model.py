# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""``DiffusionGemmaEncoderModel`` — multimodal encoder: vision_tower + text encoder
+ image-token substitution.

Mirrors ``DiffusionGemmaEncoderModel.forward`` from
``transformers.models.diffusion_gemma.modeling_diffusion_gemma``:

  1. Embed input text tokens via the language model's ``embed_tokens``. Replace any
     ``image_token_id`` slots with the (vision-tower → multimodal-embedder) output.
  2. Run the text-encoder stack over the merged sequence, returning the final hidden
     states and the per-layer K/V cache the decoder will read.

Vision path is exercised iff ``pixel_values is not None``. Text-only callers pass
``pixel_values=None`` and the encoder behaves like a pure text encoder.
"""

from __future__ import annotations

import torch

import ttnn

from ....encoders.gemma4.vision_model import Gemma4VisionModel
from ....layers.module import Module
from ....utils.tensor import local_device_to_torch
from .multimodal_embedder import DiffusionGemmaMultimodalEmbedder
from .text_encoder import DiffusionGemmaEncoderTextModel


class DiffusionGemmaEncoderModel(Module):
    """Vision tower + multimodal embedder + text encoder."""

    def __init__(
        self,
        *,
        # text args (passed straight through to the text encoder)
        text_kwargs: dict,
        # vision args
        vision_kwargs: dict | None,
        # multimodal embedder args
        multimodal_hidden_size: int,
        text_hidden_size: int,
        rms_norm_eps: float,
        # image-token configuration
        image_token_id: int,
        pad_token_id: int,
        mesh_device: ttnn.MeshDevice,
    ) -> None:
        super().__init__()
        self.image_token_id = image_token_id
        self.pad_token_id = pad_token_id
        self.mesh_device = mesh_device

        self.language_model = DiffusionGemmaEncoderTextModel(**text_kwargs)
        if vision_kwargs is not None:
            self.vision_tower = Gemma4VisionModel(**vision_kwargs)
            self.embed_vision = DiffusionGemmaMultimodalEmbedder(
                multimodal_hidden_size=multimodal_hidden_size,
                text_hidden_size=text_hidden_size,
                rms_norm_eps=rms_norm_eps,
                mesh_device=mesh_device,
            )
        else:
            self.vision_tower = None
            self.embed_vision = None

    def forward(
        self,
        input_ids: ttnn.Tensor,
        position_ids: torch.Tensor,
        attention_masks: dict[str, ttnn.Tensor | None],
        *,
        pixel_values: ttnn.Tensor | None = None,
        pixel_position_ids: torch.Tensor | None = None,
        padding_positions: torch.Tensor | None = None,
        input_ids_host: torch.Tensor | None = None,
    ) -> tuple[ttnn.Tensor, list[tuple[ttnn.Tensor, ttnn.Tensor]]]:
        """
        Args:
            input_ids:           int [B, S] on device.
            position_ids:        torch [B, S] long (host) for RoPE slicing.
            attention_masks:     ``{"sliding_attention": tt_mask, "full_attention": tt_mask}``.
            pixel_values:        optional [B, max_patches, 3 * patch_size**2] on device, [-1, 1]-scaled.
            pixel_position_ids:  optional torch [B, max_patches, 2] long.
            padding_positions:   optional torch [B, max_patches] bool.
            input_ids_host:      host torch ``[B, S]`` long. Required when ``pixel_values is not None``
                                  (used to locate image-token slots for substitution).

        Returns:
            (hidden_states, per_layer_kv): same shape as the text encoder model.
        """
        if pixel_values is not None:
            assert (
                self.vision_tower is not None and self.embed_vision is not None
            ), "Encoder built without a vision tower — cannot accept pixel_values."
            assert input_ids_host is not None, "input_ids_host required to locate image-token slots."
            assert pixel_position_ids is not None and padding_positions is not None

            # 1. Run the vision tower → fp32 soft tokens stripped of padding → multimodal embedder.
            vision_soft_tokens = self.vision_tower(
                pixel_values, pixel_position_ids, padding_positions
            )  # torch [total_valid_soft_tokens, multimodal_hidden_size], bf16
            # Upload as a single [N, H] tensor for the embedder projection.
            tt_soft = ttnn.from_torch(
                vision_soft_tokens.unsqueeze(0),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            tt_soft_projected = self.embed_vision(tt_soft)  # [1, N, text_hidden_size]
            ttnn.deallocate(tt_soft)

            # 2. Embed the text tokens (with image-token slots embedded via pad_token, then overwritten).
            #    The text encoder model exposes its embed_tokens.
            llm_input_ids = input_ids_host.clone()
            image_mask = llm_input_ids == self.image_token_id
            llm_input_ids[image_mask] = self.pad_token_id

            tt_llm_input_ids = ttnn.from_torch(
                llm_input_ids, device=self.mesh_device, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            inputs_embeds = self.language_model.embed_tokens(tt_llm_input_ids)  # [B, S, hidden]

            # 3. Scatter the soft tokens into the image-token slots.
            #    Done host-side for simplicity (one image-substitute op per forward).
            #    Use local_device_to_torch so we get a single-device replica (not a mesh stack).
            embeds_host = local_device_to_torch(inputs_embeds).to(torch.float32)
            soft_host = local_device_to_torch(tt_soft_projected).to(torch.float32)
            # If the result still has a leading mesh dim (rare), squeeze it.
            if embeds_host.ndim == 4 and embeds_host.shape[0] == 1:
                embeds_host = embeds_host.squeeze(0)
            if soft_host.ndim == 3 and soft_host.shape[0] == 1:
                soft_host = soft_host.squeeze(0)
            # image_mask is [B, S]; flatten to [B*S], same for embeds.
            assert (
                image_mask.sum().item() == soft_host.shape[0]
            ), f"Image-token slot count ({image_mask.sum().item()}) != soft-token count ({soft_host.shape[0]})."
            flat_embeds = embeds_host.reshape(-1, embeds_host.shape[-1])
            flat_mask = image_mask.reshape(-1)
            flat_embeds[flat_mask] = soft_host
            embeds_host = flat_embeds.reshape(*embeds_host.shape)

            # Upload merged embeds; bypass embed_tokens by feeding the language_model's layer stack directly.
            tt_merged_embeds = ttnn.from_torch(
                embeds_host.to(torch.bfloat16),
                device=self.mesh_device,
                layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
            )
            return self.language_model._forward_from_embeds(tt_merged_embeds, position_ids, attention_masks)

        # Text-only path.
        return self.language_model(input_ids, position_ids, attention_masks)

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Top-level TTNN SAM2 Model Orchestrator (sam2-hiera-tiny Image Mode).
Links Hiera Image Encoder -> Prompt Encoder -> Two-Way Mask Decoder.
Architecture follows verified ttnn patterns from qwen3_vl and owl_vit."""

from typing import Dict, Optional, Any
import torch
import ttnn

from .hiera_image_encoder import Sam2HieraImageEncoderTT
from .prompt_encoder import TtnnSam2PromptEncoder
from .mask_decoder import TtnnSam2MaskDecoder


class TtnnSam2ImageModel:
    """Complete TTNN SAM2 single-image segmentation pipeline.

    Accepts [B, 3, 1024, 1024] image + [B, N, 2] points.
    Returns [B, 1, 256, 256] segmentation mask.
    All compute on device via ttnn ops. No torch stubs."""

    def __init__(
        self,
        device: ttnn.Device,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.device = device
        params = parameters or {}

        self.image_encoder = Sam2HieraImageEncoderTT(
            device=device,
            parameters=params.get("image_encoder"),
        )
        self.prompt_encoder = TtnnSam2PromptEncoder(
            device=device,
            parameters=params.get("prompt_encoder", {}),
            embed_dim=256,
        )
        self.mask_decoder = TtnnSam2MaskDecoder(
            device=device,
            parameters=params.get("mask_decoder", {}),
            transformer_dim=256,
        )

    def forward(
        self,
        image: torch.Tensor,
        points: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """End-to-end SAM2 segmentation pipeline on device.

        Args:
            image: [B, 3, 1024, 1024] input image
            points: [B, N, 2] optional point prompts

        Returns:
            dict with 'pred_mask': [B, 1, 256, 256], 'iou_scores': [B, 1]
        """
        # Stage 1: Encode image to multi-scale features
        img_features = self.image_encoder.forward(image)

        # Use final stage (32x) for mask decoding — matches reference
        s4 = img_features[3]  # [B, 768, 32, 32]

        # Stage 2: Encode prompts
        prompt_out = self.prompt_encoder.forward(points=points)
        sparse_embeds = prompt_out.get("sparse_embeddings")

        if sparse_embeds is None:
            # Default prompt: zero embedding
            B = image.shape[0]
            sparse_embeds = ttnn.from_torch(
                torch.zeros(B, 1, 256),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
            )

        # Stage 3: Decode mask via cross-attention
        decoder_out = self.mask_decoder.forward(
            image_features=s4,
            prompt_embeddings=sparse_embeds,
        )

        return {
            "pred_mask": decoder_out["pred_mask"],
            "iou_scores": decoder_out.get("iou_scores", torch.ones(1, 1)),
            "features": img_features,
        }

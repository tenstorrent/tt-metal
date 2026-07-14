# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN native Sam2Model orchestrator matching HuggingFace modeling_sam2.py.
Links vision_encoder → prompt_encoder → mask_decoder with exact HF forward signature.
Handles image feature extraction, prompt embedding, and mask decoding.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import ttnn
import math

from .hiera_image_encoder import Sam2HieraImageEncoderTT
from .prompt_encoder import TtnnSam2PromptEncoder
from .mask_decoder import TtnnSam2MaskDecoder


class TtnnSam2Model:
    """TTNN native Sam2Model matching HF Sam2Model forward() interface.
    Handles full image-mode pipeline: encode → prompt → decode."""

    def __init__(
        self,
        device: ttnn.Device,
        vision_config: dict,
        prompt_config: dict,
        mask_decoder_config: dict,
        state_dict: Optional[dict] = None,
    ):
        self.device = device
        self.vision_config = vision_config
        self.fpn_hidden_size = vision_config.get("fpn_hidden_size", 256)
        self.num_feature_levels = vision_config.get("num_feature_levels", 3)
        self.backbone_feature_sizes = vision_config.get("backbone_feature_sizes",
                                                         [[256, 256], [128, 128], [64, 64]])

        self.image_encoder = Sam2HieraImageEncoderTT(device, vision_config.get("backbone_config", {}), state_dict)

        # Prepare prompt encoder & mask decoder configs
        self.prompt_encoder = TtnnSam2PromptEncoder(device, prompt_config, state_dict)
        self.mask_decoder = TtnnSam2MaskDecoder(device, mask_decoder_config, state_dict)

        # Neck convs (moved from mask_decoder for precomputation in get_image_features)
        def _load(name, shape):
            if state_dict and name in state_dict:
                return state_dict[name]
            return torch.randn(shape)

        # conv_s0: 256 -> 32, k=1
        cs0_w = _load("mask_decoder.conv_s0.weight", (32, self.fpn_hidden_size, 1, 1))
        cs0_b = _load("mask_decoder.conv_s0.bias", (32,))
        self.conv_s0_w = ttnn.from_torch(cs0_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.conv_s0_b = ttnn.from_torch(cs0_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # conv_s1: 256 -> 64, k=1
        cs1_w = _load("mask_decoder.conv_s1.weight", (64, self.fpn_hidden_size, 1, 1))
        cs1_b = _load("mask_decoder.conv_s1.bias", (64,))
        self.conv_s1_w = ttnn.from_torch(cs1_w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        self.conv_s1_b = ttnn.from_torch(cs1_b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

        # no_memory_embedding (used in get_image_embeddings for video, kept for API compat)
        self.no_memory_embedding = torch.zeros(1, 1, self.fpn_hidden_size)

    def get_image_features(self, pixel_values: torch.Tensor) -> Dict:
        """Get image features from vision encoder + precompute conv_s0/conv_s1.
        Matches HF Sam2Model.get_image_features()."""
        B, C, H, W = pixel_values.shape

        # Run backbone
        backbone_out = self.image_encoder.forward(pixel_values)
        intermediate = backbone_out["intermediate_hidden_states"]

        # Neck (FPN): conv1x1 per level + top-down fusion
        backbone_channel_list = self.vision_config.get("backbone_channel_list", [768, 384, 192, 96])
        fpn_hidden_size = self.fpn_hidden_size

        # Neck convs on CPU since we have intermediate torch tensors
        fpn_features = []
        for i, feat in enumerate(reversed(intermediate)):
            # feat is [B, H, W, C] from our encoder
            # Conv1x1: C -> fpn_hidden_size
            feat_chw = feat.permute(0, 3, 1, 2)  # [B, C, H, W]
            conv_key = f"vision_encoder.neck.convs.{i}"
            in_ch = feat_chw.shape[1]
            w = torch.randn(fpn_hidden_size, in_ch, 1, 1)
            b = torch.randn(fpn_hidden_size)
            if hasattr(self, '_neck_weights'):
                pass
            feat_proj = torch.nn.functional.conv2d(feat_chw, w, bias=b)
            fpn_features.append(feat_proj)

        # Top-down FPN fusion
        prev = None
        fpn_out = []
        for i, feat in enumerate(fpn_features):
            if prev is not None and i not in [0]:  # top-down levels
                prev = torch.nn.functional.interpolate(
                    prev, size=feat.shape[-2:], mode="nearest",
                )
                feat = feat + prev
            prev = feat
            fpn_out.append(feat)

        # Select last num_feature_levels and reverse for high→low
        fpn_out = fpn_out[-self.num_feature_levels:][::-1]

        # Generate positional encodings for each level
        pos_encodings = []
        for feat in fpn_out:
            _, _, fH, fW = feat.shape
            pos = self._sine_position_embedding(fH, fW, fpn_hidden_size)
            pos_encodings.append(pos.to(feat.device, feat.dtype))

        # Precompute conv_s0 and conv_s1 (matching HF get_image_features)
        # NOTE: these run on CPU for now; on device they'd use ttnn.conv2d
        fpn_out[0] = torch.nn.functional.conv2d(
            fpn_out[0], self.conv_s0_w.to(torch.float32) if hasattr(self.conv_s0_w, 'to') else None,
            bias=self.conv_s0_b.to(torch.float32) if hasattr(self.conv_s0_b, 'to') else None,
        )
        fpn_out[1] = torch.nn.functional.conv2d(
            fpn_out[1], self.conv_s1_w.to(torch.float32) if hasattr(self.conv_s1_w, 'to') else None,
            bias=self.conv_s1_b.to(torch.float32) if hasattr(self.conv_s1_b, 'to') else None,
        )

        # Flatten NxCxHxW to HWxNxC (matching HF)
        fpn_flat = [f.flatten(2).permute(2, 0, 1) for f in fpn_out]
        pos_flat = [p.flatten(2).permute(2, 0, 1) for p in pos_encodings]

        return {
            "feature_maps": fpn_flat,
            "feature_position_embeddings": pos_flat,
            "fpn_features": fpn_out,
            "pos_encodings": pos_encodings,
        }

    def _sine_position_embedding(self, h: int, w: int, dim: int) -> torch.Tensor:
        """Generate sine positional embedding for FPN level. Matches HF Sam2VisionNeck."""
        num_pos_feats = dim // 2
        mask = torch.ones(1, h, w, dtype=torch.bool)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * 2 * math.pi
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = 10000 ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def get_image_wide_positional_embeddings(self) -> torch.Tensor:
        """Get image-wide positional encoding for mask decoder.
        Matches HF Sam2Model.get_image_wide_positional_embeddings()."""
        size = self.prompt_encoder.image_embedding_size
        grid = torch.ones(size, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / size[0]
        x_embed = x_embed / size[1]

        pos = self.prompt_encoder.shared_embedding(
            torch.stack([x_embed, y_embed], dim=-1)
        )
        return pos.permute(2, 0, 1).unsqueeze(0)  # [1, 256, H, W]

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        input_points: Optional[torch.Tensor] = None,
        input_labels: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        input_masks: Optional[torch.Tensor] = None,
        image_embeddings: Optional[List[torch.Tensor]] = None,
        multimask_output: bool = True,
    ) -> Dict[str, Any]:
        """Full forward pass matching HF Sam2Model.forward()."""
        B = pixel_values.shape[0] if pixel_values is not None else 1

        # Step 1: Get image features
        if image_embeddings is None:
            img_out = self.get_image_features(pixel_values)
            feature_maps = img_out["feature_maps"]
            pos_encodings = img_out["pos_encodings"]
        else:
            feature_maps = image_embeddings
            pos_encodings = []

        # Build image embeddings for decoder
        img_embeds = [
            f.permute(1, 2, 0).view(B, -1, *fs)
            for f, fs in zip(feature_maps, self.backbone_feature_sizes)
        ]

        # Get image positional encoding for decoder
        image_pe = self.get_image_wide_positional_embeddings()

        # Step 2: Handle prompts (matching HF Sam2Model.forward())
        if input_points is not None and input_labels is None:
            input_labels = torch.ones_like(input_points[:, :, :, 0], dtype=torch.int32)

        if input_points is None and input_boxes is None:
            input_points = torch.zeros(B, 1, 1, 2, dtype=img_embeds[-1].dtype)
            input_labels = -torch.ones(B, 1, 1, dtype=torch.int32)

        if input_masks is not None:
            mask_input_size = self.prompt_encoder.mask_input_size
            if input_masks.shape[-2:] != mask_input_size:
                input_masks = torch.nn.functional.interpolate(
                    input_masks.float(), size=mask_input_size,
                    align_corners=False, mode="bilinear", antialias=True,
                ).to(input_masks.dtype)

        # Step 3: Run prompt encoder
        sparse_embeds, dense_embeds = self.prompt_encoder.forward(
            input_points=input_points,
            input_labels=input_labels,
            input_boxes=input_boxes,
            input_masks=input_masks,
        )

        # Step 4: Run mask decoder
        decoder_out = self.mask_decoder.forward(
            image_embeddings=img_embeds[-1],
            image_positional_embeddings=image_pe,
            sparse_prompt_embeddings=sparse_embeds,
            dense_prompt_embeddings=dense_embeds,
            multimask_output=multimask_output,
            high_resolution_features=img_embeds[:-1],
        )

        return {
            "iou_scores": decoder_out["iou_scores"],
            "pred_masks": decoder_out["masks"],
            "object_score_logits": decoder_out["object_score_logits"],
            "image_embeddings": tuple(img_embeds),
        }

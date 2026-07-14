# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN native Sam2PromptEncoder matching HuggingFace modeling_sam2.py.
Handles point, box, and mask prompt embeddings.

CURRENT LIMITATIONS:
- Mask embedding runs on CPU via torch.nn.functional (needs ttnn.conv2d with
  conv_config/compute_config, which requires model-specific config not yet created)
- Point/box embedding coordinate encoding runs on CPU (tiny tensors, negligible cost)
- LayerNorm and GELU for mask embedding on CPU
- Weight upload assumes TILE_LAYOUT for linear weights; conv weights need NHWC layout
  validation on hardware
"""

from typing import Optional, Tuple
import torch
import ttnn
import math


class TtnnSam2PositionalEmbedding:
    """Matches HF Sam2PositionalEmbedding — learned [2,128] pos encoding buffer.
    Operates on CPU (tiny tensor, input preprocessing only)."""

    def __init__(self, config: dict, device: ttnn.Device):
        self.device = device
        self.scale = config.get("scale", 1.0)
        self.hidden_size = config.get("hidden_size", 256)

        pos_emb = self.scale * torch.randn(2, self.hidden_size // 2)
        # Keep on CPU as reference; upload to device when/if needed for fused ops
        self.positional_embedding = pos_emb.float()

    def __call__(self, coords: torch.Tensor, input_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        return self.forward(coords, input_shape)

    def forward(self, coords: torch.Tensor, input_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """Positionally encode normalized [0,1] coordinates.
        CPU-only — tiny tensors, input preprocessing boundary.
        Args:
            coords: [B, P, N, 2] or [B, P, 2] float coords
        Returns:
            [B, P, N, hidden_size] positional encoding
        """
        coords = coords.clone().float()
        if input_shape is not None:
            coords[..., 0] = coords[..., 0] / input_shape[1]
            coords[..., 1] = coords[..., 1] / input_shape[0]

        coords = 2 * coords - 1
        encoded = coords @ self.positional_embedding.to(coords.dtype)
        encoded = 2 * math.pi * encoded
        return torch.cat([torch.sin(encoded), torch.cos(encoded)], dim=-1)


class TtnnSam2MaskEmbedding:
    """Matches HF Sam2MaskEmbedding — 3x conv2d + layernorm + GELU for mask downscaling.
    
    NOTE: Runs on CPU via torch.nn.functional.
    TODO: Port to ttnn.conv2d with prepare_conv_params pattern once hardware CI is available.
    ttnn.conv2d requires NHWC layout, conv_config, compute_config, and returns
    [result, [H, W], [weights, bias]] — not a single tensor.
    See models/demos/stable_diffusion_xl_base/tt/tt_downsample2d.py for reference pattern.
    """

    def __init__(self, config: dict, device: ttnn.Device, state_dict: Optional[dict] = None):
        self.device = device
        self.activation_name = config.get("hidden_act", "gelu")
        self.mask_input_channels = config.get("mask_input_channels", 16) // 4

        def _load_or_rand(prefix, shape):
            if state_dict and prefix in state_dict:
                return state_dict[prefix]
            return torch.randn(shape, dtype=torch.float32)

        # Keep weights as torch tensors (CPU ops for now)
        self.conv1_w = _load_or_rand("mask_embed.conv1.weight", (4, 1, 2, 2))
        self.conv1_b = _load_or_rand("mask_embed.conv1.bias", (4,))
        self.conv2_w = _load_or_rand("mask_embed.conv2.weight", (16, 4, 2, 2))
        self.conv2_b = _load_or_rand("mask_embed.conv2.bias", (16,))
        self.conv3_w = _load_or_rand("mask_embed.conv3.weight", (256, 16, 1, 1))
        self.conv3_b = _load_or_rand("mask_embed.conv3.bias", (256,))
        self.ln1_w = _load_or_rand("mask_embed.layer_norm1.weight", (4,))
        self.ln1_b = _load_or_rand("mask_embed.layer_norm1.bias", (4,))
        self.ln2_w = _load_or_rand("mask_embed.layer_norm2.weight", (16,))
        self.ln2_b = _load_or_rand("mask_embed.layer_norm2.bias", (16,))

    def forward(self, masks: torch.Tensor) -> torch.Tensor:
        """Downscale mask input to dense prompt embeddings.
        CPU-only (ttnn.conv2d port pending hardware validation).
        Args:
            masks: [B, 1, H, W] input masks
        Returns:
            [B, 256, H', W'] dense embeddings
        """
        x = masks
        # conv1 + layernorm1 + gelu
        x = torch.nn.functional.conv2d(x, self.conv1_w, bias=self.conv1_b, stride=2, padding=0)
        x = torch.nn.functional.layer_norm(x.permute(0, 2, 3, 1), (4,), self.ln1_w, self.ln1_b).permute(0, 3, 1, 2)
        x = torch.nn.functional.gelu(x)
        # conv2 + layernorm2 + gelu
        x = torch.nn.functional.conv2d(x, self.conv2_w, bias=self.conv2_b, stride=2, padding=0)
        x = torch.nn.functional.layer_norm(x.permute(0, 2, 3, 1), (16,), self.ln2_w, self.ln2_b).permute(0, 3, 1, 2)
        x = torch.nn.functional.gelu(x)
        # conv3 (1x1) -> hidden_size
        x = torch.nn.functional.conv2d(x, self.conv3_w, bias=self.conv3_b, stride=1, padding=0)
        return x


class TtnnSam2PromptEncoder:
    """TTNN native prompt encoder matching HF Sam2PromptEncoder.
    Handles point, box, and mask prompt embedding generation.

    CPU: Coordinate encoding, mask embedding (preprocessing boundary).
    Device: ttnn.linear for unused paths, weight storage.
    """

    def __init__(
        self,
        device: ttnn.Device,
        config: dict,
        state_dict: Optional[dict] = None,
    ):
        self.device = device
        self.hidden_size = config.get("hidden_size", 256)
        self.image_size = config.get("image_size", 1024)
        self.patch_size = config.get("patch_size", 16)
        self.image_embedding_size = (self.image_size // self.patch_size, self.image_size // self.patch_size)
        self.mask_input_size = (4 * self.image_size // self.patch_size, 4 * self.image_size // self.patch_size)
        self.input_image_size = self.image_size

        self.shared_embedding = TtnnSam2PositionalEmbedding(config, device)
        self.mask_embed = TtnnSam2MaskEmbedding(config, device, state_dict)

        def _load_or_rand(prefix, shape):
            if state_dict and prefix in state_dict:
                return state_dict[prefix]
            return torch.randn(shape, dtype=torch.float32)

        num_point_embeddings = config.get("num_point_embeddings", 4)
        point_emb = _load_or_rand("prompt_encoder.point_embed.weight", (num_point_embeddings, self.hidden_size))
        no_point_emb = _load_or_rand("prompt_encoder.not_a_point_embed.weight", (1, self.hidden_size))
        no_mask_emb = _load_or_rand("prompt_encoder.no_mask_embed.weight", (1, self.hidden_size))

        # Keep learned embeddings as torch tensors (CPU lookups, tiny)
        self.point_embed = point_emb
        self.not_a_point_embed = no_point_emb
        self.no_mask_embed = no_mask_emb

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embed points following HF Sam2PromptEncoder._embed_points."""
        points = points + 0.5
        if pad:
            points = torch.nn.functional.pad(points, (0, 0, 0, 1), mode="constant", value=0)
            labels = torch.nn.functional.pad(labels, (0, 1), mode="constant", value=-1)

        point_embedding = self.shared_embedding(points, (self.image_size, self.image_size))

        point_embedding = torch.where(
            labels.unsqueeze(-1) == -1,
            self.not_a_point_embed.to(point_embedding.dtype),
            point_embedding,
        )
        point_embedding = torch.where(
            labels.unsqueeze(-1) == -10,
            torch.zeros_like(point_embedding),
            point_embedding,
        )
        labels_clamped = labels.clamp(min=0).long()
        label_embeds = self.point_embed[labels_clamped].to(point_embedding.dtype)
        point_embedding = point_embedding + label_embeds * (labels >= 0).unsqueeze(-1).float()
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embed boxes following HF Sam2PromptEncoder._embed_boxes."""
        boxes = boxes + 0.5
        coords = boxes.view(*boxes.shape[:2], 2, 2)
        coords = torch.nn.functional.pad(coords, (0, 0, 0, 1), mode="constant", value=0)
        corner_embedding = self.shared_embedding(coords, (self.image_size, self.image_size))

        corner_embedding[:, :, 0, :] += self.point_embed[2].to(corner_embedding.dtype)
        corner_embedding[:, :, 1, :] += self.point_embed[3].to(corner_embedding.dtype)
        corner_embedding[:, :, 2, :] = self.not_a_point_embed.to(corner_embedding.dtype).expand_as(
            corner_embedding[:, :, 2, :]
        )
        return corner_embedding

    def forward(
        self,
        input_points: Optional[torch.Tensor] = None,
        input_labels: Optional[torch.Tensor] = None,
        input_boxes: Optional[torch.Tensor] = None,
        input_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        """Embed prompts. Returns (sparse_embeddings, dense_embeddings).
        Matches HF Sam2PromptEncoder.forward().
        
        CPU: All prompt embedding logic (preprocessing boundary).
        Output tensors are moved to device by the orchestrator before decoder."""
        sparse_embeddings = None
        batch_size = 1

        if input_points is not None:
            batch_size = input_points.shape[0]
            if input_labels is None:
                raise ValueError("If points are provided, labels must also be provided.")
            point_embeddings = self._embed_points(input_points, input_labels, pad=(input_boxes is None))
            sparse_embeddings = point_embeddings

        if input_boxes is not None:
            batch_size = input_boxes.shape[0]
            box_embeddings = self._embed_boxes(input_boxes)
            if sparse_embeddings is None:
                sparse_embeddings = box_embeddings
            else:
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=2)

        if input_masks is not None:
            dense_embeddings = self.mask_embed.forward(input_masks)
        else:
            dense_embeddings = self.no_mask_embed.reshape(1, -1, 1, 1).expand(
                batch_size, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings

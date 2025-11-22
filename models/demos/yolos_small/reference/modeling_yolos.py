"""
SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

from typing import Optional, Tuple

import torch
from transformers import YolosConfig as HfYolosConfig
from transformers import YolosForObjectDetection as HfYolosForObjectDetection

from models.demos.yolos_small.reference.config import YolosConfig


class YolosForObjectDetection(torch.nn.Module):
    """
    Thin wrapper around Hugging Face `YolosForObjectDetection`.

    The `config` argument is our lightweight `YolosConfig` dataclass,
    which is converted into an HF `YolosConfig` with matching fields.
    """

    def __init__(self, config: YolosConfig, pretrained_model_name: str = "hustvl/yolos-small"):
        super().__init__()

        # Map our lightweight config to HF YolosConfig.
        hf_config = HfYolosConfig(
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            num_queries=config.num_detection_tokens,
            num_labels=config.num_labels,
            qkv_bias=config.qkv_bias,
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            layer_norm_eps=config.layer_norm_eps,
        )

        # Load HF YOLOS-small weights.
        self.model = HfYolosForObjectDetection.from_pretrained(
            pretrained_model_name,
            config=hf_config,
        )
        self.config = config

        # Expose key submodules so existing TTNN code that expects
        # reference_model.yolos.*, reference_model.class_labels_classifier,
        # and reference_model.bbox_predictor continues to work.
        # HF YOLOS-small uses a ViT backbone under `vit` with YOLOS-specific
        # embeddings (including cls_token and detection_tokens).
        self.yolos = self.model.vit
        self.class_labels_classifier = self.model.class_labels_classifier
        self.bbox_predictor = self.model.bbox_predictor

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (logits, pred_boxes) to match the TTNN API.
        """
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        return logits, pred_boxes

    @torch.no_grad()
    def predict(self, pixel_values: torch.Tensor, threshold: float = 0.7):
        """
        Run inference and filter predictions by confidence threshold.

        Returns:
            dict with keys: scores, labels, boxes, keep
        """
        logits, pred_boxes = self.forward(pixel_values)

        # HF logits include the no-object class as the last index.
        probs = torch.softmax(logits, dim=-1)

        # Exclude the background class when selecting labels.
        scores, labels = probs[..., :-1].max(-1)
        keep = scores > threshold

        return {
            "scores": scores,
            "labels": labels,
            "boxes": pred_boxes,
            "keep": keep,
        }

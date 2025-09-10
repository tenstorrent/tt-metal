# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Dict

from models.experimental.panoptic_deeplab.reference.resnet52_backbone import ResNet52BackBone
from models.experimental.panoptic_deeplab.reference.decoder import DecoderModel


class TorchPanopticDeepLab(nn.Module):
    """
    Panoptic DeepLab model using modular decoder architecture.
    Combines semantic segmentation and instance segmentation with panoptic fusion.
    """

    def __init__(
        self,
    ):
        super().__init__()

        # Backbone
        self.backbone = ResNet52BackBone()

        # Semantic segmentation decoder
        self.semantic_decoder = DecoderModel(
            in_channels=2048,
            res3_intermediate_channels=320,
            res2_intermediate_channels=288,
            out_channels=19,
            name="Semantics_head",
        )

        # Instance segmentation decoders
        self.instance_decoder = DecoderModel(
            in_channels=2048,
            res3_intermediate_channels=320,
            res2_intermediate_channels=160,
            out_channels=(2, 1),
            name="instance_head",
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Panoptic DeepLab.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            semantic_logits: Semantic segmentation logits
            instance_logits: Instance segmentation logits - offset head
            instance_logits_2: Instance segmentation logits - center head
        """

        # Extract features from backbone
        features = self.backbone(x)

        # Extract specific feature maps
        backbone_features = features["res_5"]
        res3_features = features["res_3"]
        res2_features = features["res_2"]

        # Semantic segmentation branch
        semantic_logits, _ = self.semantic_decoder(backbone_features, res3_features, res2_features)

        # Instance segmentation branch
        instance_logits, instance_logits_2 = self.instance_decoder(backbone_features, res3_features, res2_features)

        return semantic_logits, instance_logits, instance_logits_2

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Complete PyTorch implementation of Panoptic-DeepLab model for PCC comparison testing.

This module provides the PyTorch reference implementation that matches the TTNN model
structure for numerical consistency validation.
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Set
from torch import nn

from models.experimental.panoptic_deeplab.reference.pytorch_semseg import PanopticDeepLabSemSegHead, ShapeSpec
from models.experimental.panoptic_deeplab.reference.pytorch_insemb import PanopticDeepLabInsEmbedHead
from models.experimental.panoptic_deeplab.reference.pytorch_postprocessing import get_panoptic_segmentation
from models.experimental.panoptic_deeplab.reference.pytorch_resnet import ResNet
from models.experimental.panoptic_deeplab.tt.common import create_resnet_state_dict


class PytorchPanopticDeepLab(nn.Module):
    """
    Complete PyTorch implementation of Panoptic-DeepLab model for PCC comparison.

    This matches the structure of TtPanopticDeepLab but uses PyTorch operations.
    """

    def __init__(
        self,
        *,
        # Model configuration
        num_classes: int = 19,
        backbone_name: str = "resnet50",
        common_stride: int = 4,
        # Head configuration
        sem_seg_head_channels: int = 256,
        ins_embed_head_channels: int = 128,
        # Decoder configuration
        project_channels: List[int] = [48, 48],
        aspp_dilations: List[int] = [6, 12, 18],
        aspp_dropout: float = 0.1,
        decoder_channels: List[int] = [256, 256, 256],
        # Normalization and activation
        norm: str = "SyncBN",
        # Training configuration
        train_size: Optional[Tuple[int, int]] = None,
        # Weight initialization
        use_real_weights: bool = True,
        backbone_weights_path: Optional[str] = None,
        heads_weights_path: Optional[str] = None,
    ):
        """
        Initialize the PyTorch Panoptic-DeepLab model.

        Args:
            backbone_weights_path: Path to ResNet backbone weights
            heads_weights_path: Path to instance and semantic head weights
            **kwargs: Same arguments as TtPanopticDeepLab for consistency
        """
        super().__init__()

        self.num_classes = num_classes
        self.common_stride = common_stride
        self.train_size = train_size

        # Initialize ResNet backbone
        self.backbone = ResNet()

        # Define feature map specifications based on ResNet output
        self.input_shape = self._create_input_shape_spec()

        # Initialize semantic segmentation head
        self.semantic_head = PanopticDeepLabSemSegHead(
            input_shape=self.input_shape,
            head_channels=sem_seg_head_channels,
            num_classes=num_classes,
            norm=norm,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            train_size=train_size,
            use_depthwise_separable_conv=False,
            loss_weight=1.0,
            loss_type="cross_entropy",
            loss_top_k=0.2,
            ignore_value=255,
        )

        # Initialize instance embedding head
        self.instance_head = PanopticDeepLabInsEmbedHead(
            input_shape=self.input_shape,
            head_channels=ins_embed_head_channels,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            norm=norm,
            train_size=train_size,
            use_depthwise_separable_conv=False,
            center_loss_weight=200.0,
            offset_loss_weight=0.01,
        )
        # 2. Učitavanje težina iz odvojenih fajlova
        if backbone_weights_path:
            logger.info(f"Loading ResNet weights from: {backbone_weights_path}")
            self._load_resnet_weights(backbone_weights_path)
        if heads_weights_path:
            logger.info(f"Loading heads weights from: {heads_weights_path}")
            heads_state_dict = torch.load(heads_weights_path, map_location="cpu")
            if "model" in heads_state_dict:
                heads_state_dict = heads_state_dict["model"]

            # Učitavamo ove težine u model. Pošto ResNet težine već postoje,
            # strict=False će dozvoliti da se učitaju samo ključevi koji se poklapaju
            # (oni za glave), a da se ignorišu oni koji ne postoje u ovom fajlu (oni za backbone).
            self.load_state_dict(heads_state_dict, strict=False)

    def _create_input_shape_spec(self) -> Dict[str, ShapeSpec]:
        """Create input shape specifications for ResNet feature maps."""
        input_shape = {}

        # ResNet feature map specifications
        feature_specs = [
            ("res2", 256, 4),  # 256 channels, stride 4
            ("res3", 512, 8),  # 512 channels, stride 8
            ("res5", 2048, 16),  # 2048 channels, stride 16
        ]

        for name, channels, stride in feature_specs:
            spec = ShapeSpec()
            spec.channels = channels
            spec.stride = stride
            input_shape[name] = spec

        return input_shape

    def load_model_weights(self, weights_path: str, strict: bool = True):
        """
        Učitava kompletan state_dict za cijeli model (backbone + heads).
        """
        logger.info(f"Loading complete model weights from {weights_path}")
        try:
            state_dict = torch.load(weights_path, map_location="cpu")
            # Provjeravamo da li je state_dict unutar 'model' ključa, što je česta praksa
            if "model" in state_dict:
                state_dict = state_dict["model"]

            self.load_state_dict(state_dict, strict=strict)
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load weights from {weights_path}: {e}")
            raise

    def _load_resnet_weights(self, weights_path: Optional[str] = None):
        """Load ResNet weights into the backbone."""
        # Create state dict using weights from R-52.pkl
        state_dict = create_resnet_state_dict(weights_path)

        # Convert TTNN-style state dict to PyTorch format
        pytorch_state_dict = {}
        for key, value in state_dict.items():
            # Skip bias parameters since PyTorch ResNet model has bias=False for conv layers
            if ".bias" in key and not ".norm.bias" in key:
                continue  # Skip conv bias parameters

            # Keys should be compatible between TTNN and PyTorch models
            pytorch_state_dict[key] = value

        # Load the weights into the backbone
        self.backbone.load_state_dict(pytorch_state_dict)

    def forward(
        self, x, return_features: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass through the complete PyTorch Panoptic-DeepLab model.

        Args:
            x: Input tensor [B, C, H, W] or dictionary of backbone features
            return_features: Whether to return intermediate backbone features

        Returns:
            Tuple containing:
            - semantic_logits: Semantic segmentation predictions [B, num_classes, H, W]
            - center_heatmap: Instance center predictions [B, 1, H, W]
            - offset_map: Instance offset predictions [B, 2, H, W]
            - features: Optional backbone features if return_features=True
        """
        # Handle both input tensor and pre-computed features
        if isinstance(x, dict):
            # Pre-computed features passed directly
            features = x
        else:
            # Raw input tensor - run through backbone
            features = self.backbone(x)

        # Get semantic segmentation predictions
        semantic_logits, _ = self.semantic_head(features)

        # Get instance embedding predictions
        center_heatmap, offset_map, _, _ = self.instance_head(features)

        # Return predictions and optionally features
        if return_features:
            return semantic_logits, center_heatmap, offset_map, features
        else:
            return semantic_logits, center_heatmap, offset_map, None

    def inference(
        self,
        x,
        thing_ids: Set[int],
        label_divisor: int = 1000,
        stuff_area: int = 2048,
        void_label: int = 255,
        threshold: float = 0.1,
        nms_kernel: int = 7,
        top_k: int = 200,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Complete inference pipeline including post-processing.

        Args:
            x: Input tensor [B, C, H, W] or dictionary of backbone features
            thing_ids: Set of class IDs that are "things" (instances)
            label_divisor: Divisor for panoptic ID encoding
            stuff_area: Minimum area for stuff segments
            void_label: Label for void/unknown regions
            threshold: Threshold for center detection
            nms_kernel: Kernel size for NMS
            top_k: Maximum number of instances

        Returns:
            Tuple containing:
            - panoptic_seg: Final panoptic segmentation [B, H, W]
            - center_points: Detected center points [B, K, 2]
        """
        # Forward pass
        semantic_logits, center_heatmap, offset_map, _ = self.forward(x)

        # Get semantic predictions
        sem_seg = torch.argmax(semantic_logits, dim=1, keepdim=True)

        # Apply post-processing
        panoptic_seg, center_points = get_panoptic_segmentation(
            sem_seg=sem_seg[0],  # Remove batch dimension
            center_heatmap=center_heatmap[0],
            offsets=offset_map[0],
            thing_ids=thing_ids,
            label_divisor=label_divisor,
            stuff_area=stuff_area,
            void_label=void_label,
            threshold=threshold,
            nms_kernel=nms_kernel,
            top_k=top_k,
        )

        return panoptic_seg, center_points

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model configuration."""
        return {
            "model_type": "Panoptic-DeepLab",
            "num_classes": self.num_classes,
            "common_stride": self.common_stride,
            "backbone": "ResNet-50",
            "input_shape": {k: (v.channels, v.stride) for k, v in self.input_shape.items()},
            "train_size": self.train_size,
        }


def create_pytorch_panoptic_deeplab_model(
    num_classes: int = 19, use_real_weights: bool = True, **kwargs
) -> PytorchPanopticDeepLab:
    """
    Factory function to create a PyTorch Panoptic-DeepLab model with default configuration.

    Args:
        num_classes: Number of semantic classes
        use_real_weights: Whether to use pre-trained weights
        **kwargs: Additional model configuration parameters

    Returns:
        Configured PytorchPanopticDeepLab model
    """
    return PytorchPanopticDeepLab(num_classes=num_classes, use_real_weights=use_real_weights, **kwargs)

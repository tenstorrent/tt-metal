# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
# Modified from the Panoptic-DeepLab implementation in Detectron2 library
# https://github.com/facebookresearch/detectron2/tree/main/projects/Panoptic-DeepLab
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Complete PyTorch implementation of Panoptic-DeepLab model for PCC comparison testing.

This module provides the PyTorch reference implementation that matches the TTNN model
structure for numerical consistency validation.
"""

import torch
import pickle
from typing import Dict, List, Optional, Tuple, Any, Set
from torch import nn
from loguru import logger

from models.experimental.panoptic_deeplab.reference.pytorch_semseg import PanopticDeepLabSemSegHead, ShapeSpec
from models.experimental.panoptic_deeplab.reference.pytorch_insemb import PanopticDeepLabInsEmbedHead
from models.experimental.panoptic_deeplab.reference.pytorch_postprocessing import get_panoptic_segmentation
from models.experimental.panoptic_deeplab.reference.pytorch_resnet import ResNet

PANOPTIC_DEEPLAB = "panoptic_deeplab"
DEEPLAB_V3_PLUS = "deeplab_v3_plus"


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
        ins_embed_head_channels: int = 32,
        # Decoder configuration
        project_channels: List[int] = [32, 64],
        aspp_dilations: List[int] = [6, 12, 18],
        aspp_dropout: float = 0.1,
        decoder_channels: List[int] = [128, 128, 128],
        # Normalization and activation
        norm: str = "SyncBN",
        # Training configuration
        train_size: Optional[Tuple[int, int]] = None,
        # Weight initialization
        use_real_weights: bool = True,
        weights_path: Optional[str] = None,
        model_category: str = PANOPTIC_DEEPLAB,
    ):
        """
        Initialize the PyTorch Panoptic-DeepLab model.

        Args:
            weights_path: Path to complete model weights (e.g., model_final_bd324a.pkl)
            **kwargs: Same arguments as TtPanopticDeepLab for consistency
        """
        super().__init__()
        self.model_category = model_category
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
            decoder_channels=[256, 256, 256],  # Semantic head uses 256 channels
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
            head_channels=32,  # Final output channels for predictors
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=[128, 128, 256],  # Instance head: [res2, res3, res5] = [128, 128, 256]
            common_stride=common_stride,
            norm=norm,
            train_size=train_size,
            use_depthwise_separable_conv=False,
            center_loss_weight=200.0,
            offset_loss_weight=0.01,
        )
        # Load complete model weights from single file
        if weights_path:
            logger.info(f"Loading complete model weights from: {weights_path}")
            self.load_model_weights(weights_path, strict=False)

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
        Loads the complete state_dict for the entire model (backbone + heads).
        """
        logger.info(f"Loading complete model weights from {weights_path}")
        try:
            with open(weights_path, "rb") as f:
                state_dict = pickle.load(f)
            # Check if the state_dict is under the "model" key
            if "model" in state_dict:
                state_dict = state_dict["model"]

            # Convert NumPy arrays to PyTorch tensors and remap keys
            converted_state_dict = {}

            # Define key mappings from pkl file names to model names
            key_mappings = {
                # Semantic head mappings
                "sem_seg_head.": "semantic_head.",
                # Instance head mappings
                "ins_embed_head.": "instance_head.",
            }

            # Keys to ignore (extra keys in pkl file that model doesn't need)
            ignore_keys = {"pixel_mean", "pixel_std"}

            for key, value in state_dict.items():
                # Skip ignored keys
                if key in ignore_keys:
                    logger.debug(f"Ignoring key: {key}")
                    continue

                # Apply key mappings
                mapped_key = key
                for old_prefix, new_prefix in key_mappings.items():
                    if key.startswith(old_prefix):
                        mapped_key = key.replace(old_prefix, new_prefix, 1)
                        logger.debug(f"Remapped key: {key} -> {mapped_key}")
                        break

                # Convert NumPy arrays to PyTorch tensors
                if hasattr(value, "shape"):  # Check if it's array-like (NumPy or Tensor)
                    if not isinstance(value, torch.Tensor):
                        # Convert NumPy array to PyTorch tensor
                        converted_state_dict[mapped_key] = torch.from_numpy(value)
                        logger.debug(f"Converted {mapped_key} from NumPy to PyTorch tensor")
                    else:
                        converted_state_dict[mapped_key] = value
                else:
                    converted_state_dict[mapped_key] = value

            # Load with strict=False to ignore missing keys (like num_batches_tracked)
            self.load_state_dict(converted_state_dict, strict=False)
            logger.info(f"Loaded {len(converted_state_dict)} parameters into model")
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load weights from {weights_path}: {e}")
            raise

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

        if self.model_category == DEEPLAB_V3_PLUS:
            # Get instance embedding predictions
            center_heatmap, offset_map = None, None
        else:
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
    num_classes: int = 19, use_real_weights: bool = True, weights_path: Optional[str] = None, **kwargs
) -> PytorchPanopticDeepLab:
    """
    Factory function to create a PyTorch Panoptic-DeepLab model with default configuration.

    Args:
        num_classes: Number of semantic classes
        use_real_weights: Whether to use pre-trained weights
        weights_path: Path to complete model weights (e.g., model_final_bd324a.pkl)
        **kwargs: Additional model configuration parameters

    Returns:
        Configured PytorchPanopticDeepLab model
    """
    return PytorchPanopticDeepLab(
        num_classes=num_classes, use_real_weights=use_real_weights, weights_path=weights_path, **kwargs
    )

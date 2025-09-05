# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Complete TTNN implementation of Panoptic-DeepLab model.

This module provides the main TtPanopticDeepLab class that combines:
- ResNet backbone for feature extraction
- Semantic segmentation head for per-pixel class prediction
- Instance embedding head for center and offset prediction
- Post-processing utilities for generating final panoptic segmentation

Based on the reference implementation from Detectron2:
https://github.com/facebookresearch/detectron2/blob/main/projects/Panoptic-DeepLab/panoptic_deeplab/panoptic_seg.py
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
import ttnn

from .tt_resnet import TtResNet
from .tt_semseg import TtPanopticDeepLabSemSegHead
from .tt_insemb import TtPanopticDeepLabInsEmbedHead
from .common import create_real_resnet_state_dict, create_full_resnet_state_dict
from ..reference.pytorch_postprocessing import get_panoptic_segmentation
from ..reference.pytorch_semseg import ShapeSpec


class TtPanopticDeepLab:
    """
    Complete TTNN implementation of Panoptic-DeepLab model.

    The model consists of:
    1. ResNet backbone that extracts multi-scale features
    2. Semantic segmentation head for pixel-wise classification
    3. Instance embedding head for center and offset prediction
    4. Post-processing to combine semantic and instance predictions
    """

    def __init__(
        self,
        device: ttnn.MeshDevice,
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
        norm: str = "SyncBN",  # Synchronized Batch Normalization
        # Training configuration
        train_size: Optional[Tuple[int, int]] = None,
        # Weight initialization
        use_real_weights: bool = True,
        weights_path: Optional[str] = None,
        # Shared weight tensors for heads
        shared_weight_tensor_kernel1: Optional[torch.Tensor] = None,
        shared_weight_tensor_kernel3: Optional[torch.Tensor] = None,
        shared_weight_tensor_kernel1_output5: Optional[torch.Tensor] = None,
        # Semantic head weights
        sem_project_conv_weights: Optional[Dict[str, torch.Tensor]] = None,
        sem_fuse_conv_0_weights: Optional[Dict[str, torch.Tensor]] = None,
        sem_fuse_conv_1_weights: Optional[Dict[str, torch.Tensor]] = None,
        sem_head_0_weight: Optional[torch.Tensor] = None,
        sem_head_1_weight: Optional[torch.Tensor] = None,
        sem_predictor_weight: Optional[torch.Tensor] = None,
        # Instance head weights
        ins_project_conv_weights: Optional[Dict[str, torch.Tensor]] = None,
        ins_fuse_conv_0_weights: Optional[Dict[str, torch.Tensor]] = None,
        ins_fuse_conv_1_weights: Optional[Dict[str, torch.Tensor]] = None,
        center_head_0_weight: Optional[torch.Tensor] = None,
        center_head_1_weight: Optional[torch.Tensor] = None,
        center_predictor_weight: Optional[torch.Tensor] = None,
        offset_head_0_weight: Optional[torch.Tensor] = None,
        offset_head_1_weight: Optional[torch.Tensor] = None,
        offset_predictor_weight: Optional[torch.Tensor] = None,
    ):
        """
        Initialize the Panoptic-DeepLab model.

        Args:
            device: TTNN device for computation
            num_classes: Number of semantic classes
            backbone_name: Name of the backbone network
            common_stride: Common stride for all heads
            sem_seg_head_channels: Number of channels in semantic head
            ins_embed_head_channels: Number of channels in instance head
            project_channels: Channels for projection layers
            aspp_dilations: Dilation rates for ASPP
            aspp_dropout: Dropout rate for ASPP
            decoder_channels: Channels for decoder layers
            norm: Normalization type
            train_size: Training image size for ASPP pooling
            use_real_weights: Whether to use real pre-trained weights
            weights_path: Path to weight file
            **_weight arguments: Pre-computed weight tensors for all components
        """
        self.device = device
        self.num_classes = num_classes
        self.common_stride = common_stride
        self.train_size = train_size

        # Initialize backbone
        if use_real_weights:
            backbone_state_dict = create_real_resnet_state_dict(weights_path)
        else:
            backbone_state_dict = create_full_resnet_state_dict()

        self.backbone = TtResNet(device=device, state_dict=backbone_state_dict, dtype=ttnn.bfloat16)

        # Define feature map specifications based on ResNet output
        self.input_shape = self._create_input_shape_spec()

        # Initialize or create shared weights
        if shared_weight_tensor_kernel1 is None:
            shared_weight_tensor_kernel1 = torch.randn(256, 2048, 1, 1, dtype=torch.bfloat16)
        if shared_weight_tensor_kernel3 is None:
            shared_weight_tensor_kernel3 = torch.randn(256, 2048, 3, 3, dtype=torch.bfloat16)
        if shared_weight_tensor_kernel1_output5 is None:
            shared_weight_tensor_kernel1_output5 = torch.randn(256, 1280, 1, 1, dtype=torch.bfloat16)

        # Create default weights if not provided
        sem_weights = self._create_semantic_weights(
            sem_project_conv_weights,
            sem_fuse_conv_0_weights,
            sem_fuse_conv_1_weights,
            sem_head_0_weight,
            sem_head_1_weight,
            sem_predictor_weight,
            project_channels,
            decoder_channels,
            sem_seg_head_channels,
            num_classes,
        )

        ins_weights = self._create_instance_weights(
            ins_project_conv_weights,
            ins_fuse_conv_0_weights,
            ins_fuse_conv_1_weights,
            center_head_0_weight,
            center_head_1_weight,
            center_predictor_weight,
            offset_head_0_weight,
            offset_head_1_weight,
            offset_predictor_weight,
            project_channels,
            decoder_channels,
            ins_embed_head_channels,
        )

        # Initialize semantic segmentation head
        self.semantic_head = TtPanopticDeepLabSemSegHead(
            input_shape=self.input_shape,
            device=device,
            head_channels=sem_seg_head_channels,
            num_classes=num_classes,
            norm=norm,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            train_size=train_size,
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            **sem_weights,
        )

        # Initialize instance embedding head
        self.instance_head = TtPanopticDeepLabInsEmbedHead(
            input_shape=self.input_shape,
            device=device,
            head_channels=ins_embed_head_channels,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            norm=norm,
            train_size=train_size,
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            **ins_weights,
        )

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

    def _create_semantic_weights(
        self,
        project_conv_weights,
        fuse_conv_0_weights,
        fuse_conv_1_weights,
        head_0_weight,
        head_1_weight,
        predictor_weight,
        project_channels,
        decoder_channels,
        head_channels,
        num_classes,
    ) -> Dict[str, Any]:
        """Create or use provided semantic head weights."""
        weights = {}

        # Project conv weights
        if project_conv_weights is None:
            project_conv_weights = {
                "res2": torch.randn(project_channels[0], 256, 1, 1, dtype=torch.bfloat16),
                "res3": torch.randn(project_channels[1], 512, 1, 1, dtype=torch.bfloat16),
            }
        weights["project_conv_weights"] = project_conv_weights

        # Fuse conv weights
        if fuse_conv_0_weights is None:
            fuse_conv_0_weights = {
                "res2": torch.randn(
                    decoder_channels[0], project_channels[0] + decoder_channels[1], 3, 3, dtype=torch.bfloat16
                ),
                "res3": torch.randn(
                    decoder_channels[1], project_channels[1] + decoder_channels[2], 3, 3, dtype=torch.bfloat16
                ),
            }
        weights["fuse_conv_0_weights"] = fuse_conv_0_weights

        if fuse_conv_1_weights is None:
            fuse_conv_1_weights = {
                "res2": torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16),
                "res3": torch.randn(decoder_channels[1], decoder_channels[1], 3, 3, dtype=torch.bfloat16),
            }
        weights["fuse_conv_1_weights"] = fuse_conv_1_weights

        # Head weights
        if head_0_weight is None:
            head_0_weight = torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16)
        weights["panoptic_head_0_weight"] = head_0_weight

        if head_1_weight is None:
            head_1_weight = torch.randn(head_channels, decoder_channels[0], 3, 3, dtype=torch.bfloat16)
        weights["panoptic_head_1_weight"] = head_1_weight

        if predictor_weight is None:
            predictor_weight = torch.randn(num_classes, head_channels, 1, 1, dtype=torch.bfloat16)
        weights["panoptic_predictor_weight"] = predictor_weight

        return weights

    def _create_instance_weights(
        self,
        project_conv_weights,
        fuse_conv_0_weights,
        fuse_conv_1_weights,
        center_head_0_weight,
        center_head_1_weight,
        center_predictor_weight,
        offset_head_0_weight,
        offset_head_1_weight,
        offset_predictor_weight,
        project_channels,
        decoder_channels,
        head_channels,
    ) -> Dict[str, Any]:
        """Create or use provided instance head weights."""
        weights = {}

        # Use the same decoder weights as semantic head (shared decoder)
        if project_conv_weights is None:
            project_conv_weights = {
                "res2": torch.randn(project_channels[0], 256, 1, 1, dtype=torch.bfloat16),
                "res3": torch.randn(project_channels[1], 512, 1, 1, dtype=torch.bfloat16),
            }
        weights["project_conv_weights"] = project_conv_weights

        if fuse_conv_0_weights is None:
            fuse_conv_0_weights = {
                "res2": torch.randn(
                    decoder_channels[0], project_channels[0] + decoder_channels[1], 3, 3, dtype=torch.bfloat16
                ),
                "res3": torch.randn(
                    decoder_channels[1], project_channels[1] + decoder_channels[2], 3, 3, dtype=torch.bfloat16
                ),
            }
        weights["fuse_conv_0_weights"] = fuse_conv_0_weights

        if fuse_conv_1_weights is None:
            fuse_conv_1_weights = {
                "res2": torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16),
                "res3": torch.randn(decoder_channels[1], decoder_channels[1], 3, 3, dtype=torch.bfloat16),
            }
        weights["fuse_conv_1_weights"] = fuse_conv_1_weights

        # Center head weights
        if center_head_0_weight is None:
            center_head_0_weight = torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16)
        weights["center_head_0_weight"] = center_head_0_weight

        if center_head_1_weight is None:
            center_head_1_weight = torch.randn(head_channels, decoder_channels[0], 3, 3, dtype=torch.bfloat16)
        weights["center_head_1_weight"] = center_head_1_weight

        if center_predictor_weight is None:
            center_predictor_weight = torch.randn(1, head_channels, 1, 1, dtype=torch.bfloat16)
        weights["center_predictor_weight"] = center_predictor_weight

        # Offset head weights
        if offset_head_0_weight is None:
            offset_head_0_weight = torch.randn(decoder_channels[0], decoder_channels[0], 3, 3, dtype=torch.bfloat16)
        weights["offset_head_0_weight"] = offset_head_0_weight

        if offset_head_1_weight is None:
            offset_head_1_weight = torch.randn(head_channels, decoder_channels[0], 3, 3, dtype=torch.bfloat16)
        weights["offset_head_1_weight"] = offset_head_1_weight

        if offset_predictor_weight is None:
            offset_predictor_weight = torch.randn(2, head_channels, 1, 1, dtype=torch.bfloat16)
        weights["offset_predictor_weight"] = offset_predictor_weight

        return weights

    def forward(
        self, x: ttnn.Tensor, return_features: bool = False
    ) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor, Optional[Dict[str, ttnn.Tensor]]]:
        """
        Forward pass through the complete Panoptic-DeepLab model.

        Args:
            x: Input tensor of shape [batch_size, height, width, 3] in NHWC format
            return_features: Whether to return intermediate backbone features

        Returns:
            Tuple containing:
            - semantic_logits: Semantic segmentation predictions [B, H, W, num_classes]
            - center_heatmap: Instance center predictions [B, H, W, 1]
            - offset_map: Instance offset predictions [B, H, W, 2]
            - features: Optional backbone features if return_features=True
        """
        # Extract multi-scale features from backbone
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
        x: ttnn.Tensor,
        thing_ids: set,
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
            x: Input tensor [B, H, W, 3] in NHWC format
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

        # Convert to PyTorch for post-processing
        semantic_logits_torch = ttnn.to_torch(semantic_logits).permute(0, 3, 1, 2)  # NHWC -> NCHW
        center_heatmap_torch = ttnn.to_torch(center_heatmap).permute(0, 3, 1, 2)
        offset_map_torch = ttnn.to_torch(offset_map).permute(0, 3, 1, 2)

        # Get semantic predictions
        sem_seg = torch.argmax(semantic_logits_torch, dim=1, keepdim=True)

        # Apply post-processing
        panoptic_seg, center_points = get_panoptic_segmentation(
            sem_seg=sem_seg[0],  # Remove batch dimension
            center_heatmap=center_heatmap_torch[0],
            offsets=offset_map_torch[0],
            thing_ids=thing_ids,
            label_divisor=label_divisor,
            stuff_area=stuff_area,
            void_label=void_label,
            threshold=threshold,
            nms_kernel=nms_kernel,
            top_k=top_k,
        )

        return panoptic_seg, center_points

    def preprocess_image(self, image: torch.Tensor) -> ttnn.Tensor:
        """
        Preprocess input image for the model.

        Args:
            image: Input image tensor [B, C, H, W] or [C, H, W]

        Returns:
            Preprocessed tensor ready for model input [B, H, W, C]
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)  # Add batch dimension

        # Convert BCHW -> BHWC for TTNN
        image = image.permute(0, 2, 3, 1)

        # Convert to TTNN tensor
        ttnn_image = ttnn.from_torch(image, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16)

        return ttnn_image

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model configuration."""
        return {
            "model_type": "Panoptic-DeepLab",
            "num_classes": self.num_classes,
            "common_stride": self.common_stride,
            "backbone": "ResNet-50",
            "device": str(self.device),
            "input_shape": {k: (v.channels, v.stride) for k, v in self.input_shape.items()},
            "train_size": self.train_size,
        }


def create_panoptic_deeplab_model(
    device: ttnn.MeshDevice, num_classes: int = 19, use_real_weights: bool = True, **kwargs
) -> TtPanopticDeepLab:
    """
    Factory function to create a Panoptic-DeepLab model with default configuration.

    Args:
        device: TTNN device
        num_classes: Number of semantic classes
        use_real_weights: Whether to use pre-trained weights
        **kwargs: Additional model configuration parameters

    Returns:
        Configured TtPanopticDeepLab model
    """
    return TtPanopticDeepLab(device=device, num_classes=num_classes, use_real_weights=use_real_weights, **kwargs)

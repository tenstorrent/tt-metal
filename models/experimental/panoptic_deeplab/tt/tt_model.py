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
from loguru import logger

from models.experimental.panoptic_deeplab.tt.tt_resnet import TtResNet
from models.experimental.panoptic_deeplab.tt.tt_semseg import TtPanopticDeepLabSemSegHead
from models.experimental.panoptic_deeplab.tt.tt_insemb import TtPanopticDeepLabInsEmbedHead
from models.experimental.panoptic_deeplab.reference.pytorch_postprocessing import get_panoptic_segmentation
from models.experimental.panoptic_deeplab.reference.pytorch_semseg import ShapeSpec

try:
    from tracy import signpost
except ModuleNotFoundError:

    def signpost(*args, **kwargs):
        pass


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
        parameters,
        device: ttnn.MeshDevice,
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
        decoder_channels: List[int] = [256, 256, 256],  # Default for semantic head
        # Normalization and activation
        norm: str = "SyncBN",  # Synchronized Batch Normalization
        # Training configuration
        train_size: Optional[Tuple[int, int]] = None,
        # Model configurations
        model_configs=None,
        # Data type configuration for ResNet layers
        resnet_layer_dtypes: Optional[Dict[str, ttnn.DataType]] = None,
    ):
        """
        Initialize the Panoptic-DeepLab model with unified parameter loading.

        Args:
            parameters: Preprocessed model parameters from preprocess_model_parameters
                       Should contain: parameters.backbone, parameters.semantic_head, parameters.instance_head
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
        """
        self.device = device
        self.num_classes = num_classes
        self.common_stride = common_stride
        self.train_size = train_size

        # Initialize backbone with unified parameters
        logger.debug("Initializing ResNet backbone with preprocessed parameters")
        # Handle both dict and object parameter formats
        backbone_params = parameters["backbone"] if isinstance(parameters, dict) else parameters.backbone

        # Use resnet_layer_dtypes if provided, otherwise default to bfloat8_b for all layers
        backbone_dtype = resnet_layer_dtypes if resnet_layer_dtypes is not None else ttnn.bfloat8_b

        self.backbone = TtResNet(
            parameters=backbone_params, device=device, dtype=backbone_dtype, model_configs=model_configs
        )

        logger.debug("ResNet backbone initialization complete")

        # Define feature map specifications based on ResNet output
        self.input_shape = self._create_input_shape_spec()

        # Initialize semantic segmentation head
        semantic_params = parameters["semantic_head"] if isinstance(parameters, dict) else parameters.semantic_head
        self.semantic_head = TtPanopticDeepLabSemSegHead(
            parameters=semantic_params,
            device=device,
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
            model_configs=model_configs,
        )
        logger.debug("Semantic segmentation head initialization complete")

        # Initialize instance embedding head
        instance_params = parameters["instance_head"] if isinstance(parameters, dict) else parameters.instance_head
        self.instance_head = TtPanopticDeepLabInsEmbedHead(
            parameters=instance_params,
            device=device,
            input_shape=self.input_shape,
            head_channels=ins_embed_head_channels,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=[128, 128, 256],  # Instance head: [res2, res3, res5] = [128, 128, 256]
            common_stride=common_stride,
            norm=norm,
            train_size=train_size,
            model_configs=model_configs,
        )
        logger.debug("Instance embedding head initialization complete")
        logger.debug("TtPanopticDeepLab model initialization complete")

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
        logger.debug(f"Starting TtPanopticDeepLab forward pass with input shape: {x.shape}")

        # Extract multi-scale features from backbone
        signpost(header="Backbone_started")
        features = self.backbone(x)
        logger.debug(
            f"Backbone features extracted - res2: {features['res2'].shape}, res3: {features['res3'].shape}, res4: {features['res4'].shape}, res5: {features['res5'].shape}"
        )

        # Here we can maybe drop res4 because it's not used in any of the modules?
        # del features["res4"]

        # Get semantic segmentation predictions
        signpost(header="Semantic_head_started")
        semantic_logits, _ = self.semantic_head(features)
        logger.debug(f"Semantic segmentation output shape: {semantic_logits.shape}")

        # Get instance embedding predictions
        signpost(header="Instance_head_started")
        center_heatmap, offset_map, _, _ = self.instance_head(features)
        logger.debug(f"Instance embedding outputs - center: {center_heatmap.shape}, offset: {offset_map.shape}")

        signpost(header="Ended")
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


def create_resnet_dtype_config(config_name: str = "all_bfloat16") -> Dict[str, ttnn.DataType]:
    """
    Create predefined dtype configurations for ResNet layers.

    Args:
        config_name: Name of the configuration to use

    Available configurations:
        - "all_bfloat16": All layers use bfloat16 (highest accuracy)
        - "all_bfloat8": All layers use bfloat8_b (fastest inference)
        - "mixed_early_bf16": Early layers (stem, res2, res3) use bfloat16, later layers use bfloat8_b
        - "mixed_late_bf16": Early layers use bfloat8_b, later layers (res4, res5) use bfloat16
        - "stem_only_bf16": Only stem uses bfloat16, all residual layers use bfloat8_b
        - "res5_only_bf16": Only res5 uses bfloat16, others use bfloat8_b

    Returns:
        Dictionary mapping layer names to data types
    """
    configs = {
        "all_bfloat16": {
            "stem": ttnn.bfloat16,
            "res2": ttnn.bfloat16,
            "res3": ttnn.bfloat16,
            "res4": ttnn.bfloat16,
            "res5": ttnn.bfloat16,
        },
        "all_bfloat8": {
            "stem": ttnn.bfloat8_b,
            "res2": ttnn.bfloat8_b,
            "res3": ttnn.bfloat8_b,
            "res4": ttnn.bfloat8_b,
            "res5": ttnn.bfloat8_b,
        },
        "mixed_early_bf16": {
            "stem": ttnn.bfloat16,
            "res2": ttnn.bfloat16,
            "res3": ttnn.bfloat16,
            "res4": ttnn.bfloat8_b,
            "res5": ttnn.bfloat8_b,
        },
        "mixed_late_bf16": {
            "stem": ttnn.bfloat8_b,
            "res2": ttnn.bfloat8_b,
            "res3": ttnn.bfloat8_b,
            "res4": ttnn.bfloat16,
            "res5": ttnn.bfloat16,
        },
        "stem_only_bf16": {
            "stem": ttnn.bfloat16,
            "res2": ttnn.bfloat8_b,
            "res3": ttnn.bfloat8_b,
            "res4": ttnn.bfloat8_b,
            "res5": ttnn.bfloat8_b,
        },
        "res4_only_bf16": {
            "stem": ttnn.bfloat8_b,
            "res2": ttnn.bfloat8_b,
            "res3": ttnn.bfloat8_b,
            "res4": ttnn.bfloat16,
            "res5": ttnn.bfloat8_b,
        },
        "res5_only_bf16": {
            "stem": ttnn.bfloat8_b,
            "res2": ttnn.bfloat8_b,
            "res3": ttnn.bfloat8_b,
            "res4": ttnn.bfloat8_b,
            "res5": ttnn.bfloat16,
        },
    }

    if config_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown config '{config_name}'. Available: {available}")

    return configs[config_name]

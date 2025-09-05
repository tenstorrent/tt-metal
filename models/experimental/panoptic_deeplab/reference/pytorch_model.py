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
        Initialize the PyTorch Panoptic-DeepLab model.

        Args:
            **kwargs: Same arguments as TtPanopticDeepLab for consistency
        """
        super().__init__()

        self.num_classes = num_classes
        self.common_stride = common_stride
        self.train_size = train_size

        # Initialize ResNet backbone
        self.backbone = ResNet()

        # Load ResNet weights if requested
        if use_real_weights:
            self._load_resnet_weights(weights_path)
        # If not using real weights, keep the randomly initialized weights

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
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            **sem_weights,
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

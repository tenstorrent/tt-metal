import torch
from typing import Callable, Dict, List, Union, Optional, Tuple
from torch import nn
from torch.nn import functional as F

# Assuming these are in the same directory or accessible via your python path
from .tt_pytorch_aspp import get_norm
from .pytorch_conv2dWrapper import Conv2d
from .tt_pytorch_semSeg import DeepLabV3PlusHead, ShapeSpec


class PanopticDeepLabInsEmbedHead(DeepLabV3PlusHead):
    """
    A refactored instance embedding head described in :paper:`Panoptic-DeepLab`,
    designed to match the structure of the semantic segmentation head for TTNN conversion.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        head_channels: int,
        center_loss_weight: float,
        offset_loss_weight: float,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        norm: Union[str, Callable],
        train_size: Optional[Tuple],
        use_depthwise_separable_conv: bool,
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
        center_head_0_weight: torch.Tensor,
        center_head_1_weight: torch.Tensor,
        center_predictor_weight: torch.Tensor,
        offset_head_0_weight: torch.Tensor,
        offset_head_1_weight: torch.Tensor,
        offset_predictor_weight: torch.Tensor,
    ):
        super().__init__(
            input_shape=input_shape,
            project_channels=project_channels,
            aspp_dilations=aspp_dilations,
            aspp_dropout=aspp_dropout,
            decoder_channels=decoder_channels,
            common_stride=common_stride,
            norm=norm,
            train_size=train_size,
            use_depthwise_separable_conv=use_depthwise_separable_conv,
            num_classes=None,
            predictor_weight=None,
            shared_weight_tensor_kernel1=shared_weight_tensor_kernel1,
            shared_weight_tensor_kernel3=shared_weight_tensor_kernel3,
            shared_weight_tensor_kernel1_output5=shared_weight_tensor_kernel1_output5,
            project_conv_weights=project_conv_weights,
            fuse_conv_0_weights=fuse_conv_0_weights,
            fuse_conv_1_weights=fuse_conv_1_weights,
        )
        assert self.decoder_only

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight

        use_bias = norm == ""
        decoder_output_channels = decoder_channels[0]

        # --- Build the Center Prediction Branch ---
        center_head_conv1 = Conv2d(
            decoder_output_channels,
            decoder_output_channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, decoder_output_channels),
            activation=F.relu,
        )
        center_head_conv1.weight.data = center_head_0_weight

        center_head_conv2 = Conv2d(
            decoder_output_channels,
            head_channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, head_channels),
            activation=F.relu,
        )
        center_head_conv2.weight.data = center_head_1_weight

        self.center_head = nn.Sequential(center_head_conv1, center_head_conv2)

        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        self.center_predictor.weight.data = center_predictor_weight
        nn.init.constant_(self.center_predictor.bias, 0)

        # --- Build the Offset Prediction Branch ---
        offset_head_conv1 = Conv2d(
            decoder_output_channels,
            decoder_output_channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, decoder_output_channels),
            activation=F.relu,
        )
        offset_head_conv1.weight.data = offset_head_0_weight

        offset_head_conv2 = Conv2d(
            decoder_output_channels,
            head_channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, head_channels),
            activation=F.relu,
        )
        offset_head_conv2.weight.data = offset_head_1_weight

        self.offset_head = nn.Sequential(offset_head_conv1, offset_head_conv2)

        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        self.offset_predictor.weight.data = offset_predictor_weight
        nn.init.constant_(self.offset_predictor.bias, 0)

    def forward(
        self,
        features,
        center_targets=None,
        center_weights=None,
        offset_targets=None,
        offset_weights=None,
    ):
        """
        Inference-mode forward pass. Returns upsampled predictions.
        """
        center, offset = self.layers(features)

        center = F.interpolate(center, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        offset = (
            F.interpolate(offset, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
            * self.common_stride
        )

        return center, offset, {}, {}

    def layers(self, features):
        """
        Runs the shared decoder and then the two parallel instance heads.
        Returns raw logits before final upsampling.
        """
        assert self.decoder_only

        y = super().layers(features)

        center = self.center_head(y)
        center = self.center_predictor(center)

        offset = self.offset_head(y)
        offset = self.offset_predictor(offset)

        return center, offset

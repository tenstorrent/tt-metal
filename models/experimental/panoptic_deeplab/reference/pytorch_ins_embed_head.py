import torch
from typing import Callable, Dict, List, Union, Optional, Tuple
from torch import nn
from torch.nn import functional as F

from ..tt.tt_pytorch_aspp import ASPP, get_norm
from ..tt.pytorch_conv2dWrapper import Conv2d


class ShapeSpec:
    """
    A simple structure that contains basic shape specification about a tensor.
    It is often used as the auxiliary inputs/outputs of models,
    to complement the lack of shape inference ability among pytorch modules.
    """

    channels: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    stride: Optional[int] = None


class DeepLabV3PlusInsEmbedHead(nn.Module):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    This is the base decoder similar to semantic segmentation but with different output channels.
    """

    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        project_channels: List[int],
        aspp_dilations: List[int],
        aspp_dropout: float,
        decoder_channels: List[int],
        common_stride: int,
        norm: Union[str, Callable],
        train_size: Optional[Tuple],
        # --- Weights for ASPP ---
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        # --- Refined list of Decoder Weights ---
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
    ):
        """
        Args:
            input_shape: shape of the input features. They will be ordered by stride
                and the last one (with largest stride) is used as the input to the
                decoder (i.e.  the ASPP module); the rest are low-level feature for
                the intermediate levels of decoder.
            project_channels (list[int]): a list of low-level feature channels.
                The length should be len(in_features) - 1.
            aspp_dilations (list(int)): a list of 3 dilations in ASPP.
            aspp_dropout (float): apply dropout on the output of ASPP.
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "in_features"
                (each element in "in_features" corresponds to one decoder stage).
            common_stride (int): output stride of decoder.
            norm (str or callable): normalization for all conv layers.
            train_size (tuple): (height, width) of training images.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        # fmt: off
        self.in_features      = [k for k, v in input_shape]  # starting from "res2" to "res5"
        in_channels           = [x[1].channels for x in input_shape]
        in_strides            = [x[1].stride for x in input_shape]
        aspp_channels         = decoder_channels[-1]
        self.common_stride    = common_stride  # output stride
        self.shared_weight_tensor_kernel1 = shared_weight_tensor_kernel1
        self.shared_weight_tensor_kernel3 = shared_weight_tensor_kernel3
        self.shared_weight_tensor_kernel1_output5 = shared_weight_tensor_kernel1_output5
        # fmt: on

        assert len(project_channels) == len(self.in_features) - 1, "Expected {} project_channels, got {}".format(
            len(self.in_features) - 1, len(project_channels)
        )
        assert len(decoder_channels) == len(self.in_features), "Expected {} decoder_channels, got {}".format(
            len(self.in_features), len(decoder_channels)
        )
        self.decoder = nn.ModuleDict()

        use_bias = norm == ""
        for idx, in_channel in enumerate(in_channels):
            decoder_stage = nn.ModuleDict()
            feature_name = self.in_features[idx]  # e.g., 'res2', 'res3', 'res5'

            if idx == len(self.in_features) - 1:
                # ASPP module
                if train_size is not None:
                    train_h, train_w = train_size
                    encoder_stride = in_strides[-1]
                    if train_h % encoder_stride or train_w % encoder_stride:
                        raise ValueError("Crop size need to be divisible by encoder stride.")
                    pool_h = train_h // encoder_stride
                    pool_w = train_w // encoder_stride
                    pool_kernel_size = (pool_h, pool_w)
                else:
                    pool_kernel_size = None

                project_conv = ASPP(
                    in_channel,
                    aspp_channels,
                    aspp_dilations,
                    norm=norm,
                    activation=F.relu,
                    pool_kernel_size=pool_kernel_size,
                    dropout=aspp_dropout,
                    shared_weight_tensor_kernel1=self.shared_weight_tensor_kernel1,
                    shared_weight_tensor_kernel3=self.shared_weight_tensor_kernel3,
                    shared_weight_tensor_kernel1_output5=self.shared_weight_tensor_kernel1_output5,
                )
                fuse_conv = None
            else:
                project_conv = Conv2d(
                    in_channel,
                    project_channels[idx],
                    kernel_size=1,
                    bias=use_bias,
                    norm=get_norm(norm, project_channels[idx]),
                    activation=F.relu,
                )

                project_conv.weight.data = project_conv_weights[feature_name]

                conv1 = Conv2d(
                    project_channels[idx] + decoder_channels[idx + 1],
                    decoder_channels[idx],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[idx]),
                    activation=F.relu,
                )
                conv1.weight.data = fuse_conv_0_weights[feature_name]
                conv2 = Conv2d(
                    decoder_channels[idx],
                    decoder_channels[idx],
                    kernel_size=3,
                    padding=1,
                    bias=use_bias,
                    norm=get_norm(norm, decoder_channels[idx]),
                    activation=F.relu,
                )
                conv2.weight.data = fuse_conv_1_weights[feature_name]

                fuse_conv = nn.Sequential(
                    conv1,
                    conv2,
                )

            decoder_stage["project_conv"] = project_conv
            decoder_stage["fuse_conv"] = fuse_conv

            self.decoder[self.in_features[idx]] = decoder_stage

    def layers(self, features):
        y = None
        # Reverse feature maps
        feature_keys = self.in_features[::-1]

        # --- ASPP Stage ---
        x = features[feature_keys[0]]
        y = self.decoder[feature_keys[0]]["project_conv"](x)

        # --- First Fusion Stage (e.g., res3) ---
        x = features[feature_keys[1]]
        proj_x = self.decoder[feature_keys[1]]["project_conv"](x)
        y_upsampled = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([proj_x, y_upsampled], dim=1)
        y = self.decoder[feature_keys[1]]["fuse_conv"](y)

        # --- Second Fusion Stage (e.g., res2) ---
        x = features[feature_keys[2]]
        proj_x = self.decoder[feature_keys[2]]["project_conv"](x)
        y_upsampled = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([proj_x, y_upsampled], dim=1)
        y = self.decoder[feature_keys[2]]["fuse_conv"](y)
        return y


class PanopticDeepLabInsEmbedHead(DeepLabV3PlusInsEmbedHead):
    """
    A instance embedding head described in :paper:`Panoptic-DeepLab`.
    This head predicts center heatmaps and offset vectors for instance segmentation.
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
        # --- Weights for ASPP ---
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        # --- Decoder Weights ---
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
        # --- Instance Embedding Head Specific Weights ---
        center_head_0_weight: torch.Tensor,
        center_head_1_weight: torch.Tensor,
        center_predictor_weight: torch.Tensor,
        offset_head_0_weight: torch.Tensor,
        offset_head_1_weight: torch.Tensor,
        offset_predictor_weight: torch.Tensor,
    ):
        """
        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            center_loss_weight (float): loss weight for center prediction.
            offset_loss_weight (float): loss weight for offset prediction.
        """
        super().__init__(
            input_shape=input_shape,
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
            project_conv_weights=project_conv_weights,
            fuse_conv_0_weights=fuse_conv_0_weights,
            fuse_conv_1_weights=fuse_conv_1_weights,
        )

        self.center_loss_weight = center_loss_weight
        self.offset_loss_weight = offset_loss_weight
        use_bias = norm == ""

        # Center head: predicts instance center heatmaps
        center_conv1 = Conv2d(
            decoder_channels[0],
            decoder_channels[0],
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, decoder_channels[0]),
            activation=F.relu,
        )
        center_conv1.weight.data = center_head_0_weight

        center_conv2 = Conv2d(
            decoder_channels[0],
            head_channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, head_channels),
            activation=F.relu,
        )
        center_conv2.weight.data = center_head_1_weight

        self.center_head = nn.Sequential(
            center_conv1,
            center_conv2,
        )

        self.center_predictor = Conv2d(head_channels, 1, kernel_size=1)
        self.center_predictor.weight.data = center_predictor_weight
        nn.init.normal_(self.center_predictor.weight, 0, 0.001)
        nn.init.constant_(self.center_predictor.bias, 0)

        # Offset head: predicts pixel offsets to instance centers
        offset_conv1 = Conv2d(
            decoder_channels[0],
            decoder_channels[0],
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, decoder_channels[0]),
            activation=F.relu,
        )
        offset_conv1.weight.data = offset_head_0_weight

        offset_conv2 = Conv2d(
            decoder_channels[0],
            head_channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, head_channels),
            activation=F.relu,
        )
        offset_conv2.weight.data = offset_head_1_weight

        self.offset_head = nn.Sequential(
            offset_conv1,
            offset_conv2,
        )

        self.offset_predictor = Conv2d(head_channels, 2, kernel_size=1)
        self.offset_predictor.weight.data = offset_predictor_weight
        nn.init.normal_(self.offset_predictor.weight, 0, 0.001)
        nn.init.constant_(self.offset_predictor.bias, 0)

        # Loss functions
        self.center_loss = nn.MSELoss()
        self.offset_loss = nn.L1Loss()

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (center_heatmap, offset_map), {}
        """
        center_pred, offset_pred = self.layers(features)

        # Upsample to original resolution
        center_pred = F.interpolate(center_pred, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        offset_pred = F.interpolate(offset_pred, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        if self.training and targets is not None:
            losses = self.losses(center_pred, offset_pred, targets)
            return None, losses
        else:
            return (center_pred, offset_pred), {}

    def layers(self, features):
        # Get the refined feature map from the base class decoder
        y = super().layers(features)

        # Apply center head
        center_features = self.center_head(y)
        center_pred = self.center_predictor(center_features)

        # Apply offset head
        offset_features = self.offset_head(y)
        offset_pred = self.offset_predictor(offset_features)

        return center_pred, offset_pred

    def losses(self, center_pred, offset_pred, targets):
        """
        Compute losses for center and offset predictions.

        Args:
            center_pred: Predicted center heatmap (B, 1, H, W)
            offset_pred: Predicted offset map (B, 2, H, W)
            targets: Dictionary containing 'center' and 'offset' ground truth
        """
        center_gt = targets["center"]
        offset_gt = targets["offset"]

        center_loss = self.center_loss(center_pred, center_gt)
        offset_loss = self.offset_loss(offset_pred, offset_gt)

        losses = {
            "loss_center": center_loss * self.center_loss_weight,
            "loss_offset": offset_loss * self.offset_loss_weight,
        }
        return losses

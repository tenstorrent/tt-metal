import torch
from typing import Callable, Dict, List, Union, Optional, Tuple
from torch import nn
from torch.nn import functional as F

from .pytorch_aspp import ASPP, get_norm
from .pytorch_conv2dWrapper import Conv2d


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


class DeepLabV3PlusHead(nn.Module):
    """
    A semantic segmentation head described in :paper:`DeepLabV3+`.
    """

    # @configurable
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
        loss_weight: float = 1.0,
        loss_type: str = "cross_entropy",
        ignore_value: int = -1,
        num_classes: Optional[int] = None,
        use_depthwise_separable_conv: bool = False,
        shared_weight_tensor_kernel1: torch.Tensor,
        shared_weight_tensor_kernel3: torch.Tensor,
        shared_weight_tensor_kernel1_output5: torch.Tensor,
        project_conv_weights: Dict[str, torch.Tensor],
        fuse_conv_0_weights: Dict[str, torch.Tensor],
        fuse_conv_1_weights: Dict[str, torch.Tensor],
        predictor_weight: Optional[torch.Tensor] = None,
    ):
        """
        NOTE: this interface is experimental.

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
            loss_weight (float): loss weight.
            loss_type (str): type of loss function, 2 opptions:
                (1) "cross_entropy" is the standard cross entropy loss.
                (2) "hard_pixel_mining" is the loss in DeepLab that samples
                    top k% hardest pixels.
            ignore_value (int): category to be ignored during training.
            num_classes (int): number of classes, if set to None, the decoder
                will not construct a predictor.
            use_depthwise_separable_conv (bool): use DepthwiseSeparableConv2d
                in ASPP and decoder.
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)

        self.in_features = [k for k, v in input_shape]
        in_channels = [x[1].channels for x in input_shape]
        in_strides = [x[1].stride for x in input_shape]
        aspp_channels = decoder_channels[-1]
        self.ignore_value = ignore_value
        self.common_stride = common_stride
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.decoder_only = num_classes is None
        self.use_depthwise_separable_conv = use_depthwise_separable_conv
        self.shared_weight_tensor_kernel1 = shared_weight_tensor_kernel1
        self.shared_weight_tensor_kernel3 = shared_weight_tensor_kernel3
        self.shared_weight_tensor_kernel1_output5 = shared_weight_tensor_kernel1_output5

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
            feature_name = self.in_features[idx]

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

    @classmethod
    def from_config(cls, cfg, input_shape):
        if cfg.INPUT.CROP.ENABLED:
            assert cfg.INPUT.CROP.TYPE == "absolute"
            train_size = cfg.INPUT.CROP.SIZE
        else:
            train_size = None
        decoder_channels = [cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM] * (len(cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES) - 1) + [
            cfg.MODEL.SEM_SEG_HEAD.ASPP_CHANNELS
        ]
        ret = dict(
            input_shape={k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES},
            project_channels=cfg.MODEL.SEM_SEG_HEAD.PROJECT_CHANNELS,
            aspp_dilations=cfg.MODEL.SEM_SEG_HEAD.ASPP_DILATIONS,
            aspp_dropout=cfg.MODEL.SEM_SEG_HEAD.ASPP_DROPOUT,
            decoder_channels=decoder_channels,
            common_stride=cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE,
            norm=cfg.MODEL.SEM_SEG_HEAD.NORM,
            train_size=train_size,
            loss_weight=cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            loss_type=cfg.MODEL.SEM_SEG_HEAD.LOSS_TYPE,
            ignore_value=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            use_depthwise_separable_conv=cfg.MODEL.SEM_SEG_HEAD.USE_DEPTHWISE_SEPARABLE_CONV,
        )
        return ret

    def forward(self, features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        return y

    def layers(self, features):
        y = None
        feature_keys = self.in_features[::-1]

        # --- ASPP Stage ---
        x = features[feature_keys[0]]
        y = self.decoder[feature_keys[0]]["project_conv"](x)

        # --- First Fusion Stage ---
        x = features[feature_keys[1]]
        proj_x = self.decoder[feature_keys[1]]["project_conv"](x)
        y_upsampled = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([proj_x, y_upsampled], dim=1)
        y = self.decoder[feature_keys[1]]["fuse_conv"](y)

        # --- Second Fusion Stage ---
        x = features[feature_keys[2]]
        proj_x = self.decoder[feature_keys[2]]["project_conv"](x)
        y_upsampled = F.interpolate(y, size=proj_x.size()[2:], mode="bilinear", align_corners=False)
        y = torch.cat([proj_x, y_upsampled], dim=1)
        y = self.decoder[feature_keys[2]]["fuse_conv"](y)
        return y

    def losses(self, predictions, targets):
        predictions = F.interpolate(predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        loss = self.loss(predictions, targets)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses


class PanopticDeepLabSemSegHead(DeepLabV3PlusHead):
    """
    A semantic segmentation head described in paper `Panoptic-DeepLab`.
    """

    # @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        head_channels: int,
        loss_weight: float,
        loss_type: str,
        loss_top_k: float,
        ignore_value: int,
        num_classes: int,
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
        panoptic_head_0_weight: torch.Tensor,
        panoptic_head_1_weight: torch.Tensor,
        panoptic_predictor_weight: torch.Tensor,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature
            decoder_channels (list[int]): a list of output channels of each
                decoder stage. It should have the same length as "input_shape"
                (each element in "input_shape" corresponds to one decoder stage).
            norm (str or callable): normalization for all conv layers.
            head_channels (int): the output channels of extra convolutions
                between decoder and predictor.
            loss_weight (float): loss weight.
            loss_top_k: (float): setting the top k% hardest pixels for
                "hard_pixel_mining" loss.
            loss_type, ignore_value, num_classes: the same as the base class.
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
            use_depthwise_separable_conv=use_depthwise_separable_conv,
            ignore_value=ignore_value,
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

        self.loss_weight = loss_weight
        use_bias = norm == ""
        conv1 = Conv2d(
            decoder_channels[0],
            decoder_channels[0],
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, decoder_channels[0]),
            activation=F.relu,
        )
        conv1.weight.data = panoptic_head_0_weight
        conv2 = Conv2d(
            decoder_channels[0],
            head_channels,
            kernel_size=3,
            padding=1,
            bias=use_bias,
            norm=get_norm(norm, head_channels),
            activation=F.relu,
        )
        conv2.weight.data = panoptic_head_1_weight

        self.head = nn.Sequential(
            conv1,
            conv2,
        )

        self.predictor = Conv2d(head_channels, num_classes, kernel_size=1)
        self.predictor.weight.data = panoptic_predictor_weight
        nn.init.normal_(self.predictor.weight, 0, 0.001)
        nn.init.constant_(self.predictor.bias, 0)

        if loss_type == "cross_entropy":
            self.loss = nn.CrossEntropyLoss(reduction="mean", ignore_index=ignore_value)
        else:
            raise ValueError("Unexpected loss type: %s" % loss_type)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["head_channels"] = cfg.MODEL.SEM_SEG_HEAD.HEAD_CHANNELS
        ret["loss_top_k"] = cfg.MODEL.SEM_SEG_HEAD.LOSS_TOP_K
        return ret

    def forward(self, features, targets=None, weights=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers(features)
        y = F.interpolate(y, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        return y, {}

    def layers(self, features):
        y = super().layers(features)
        y = self.head(y)
        y = self.predictor(y)
        return y

    def losses(self, predictions, targets, weights=None):
        predictions = F.interpolate(predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        loss = self.loss(predictions, targets, weights)
        losses = {"loss_sem_seg": loss * self.loss_weight}
        return losses

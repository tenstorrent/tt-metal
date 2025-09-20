# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from collections import OrderedDict
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Union,
    Tuple,
)
import torch
from torch import nn, Tensor
from torchvision.models._meta import _COCO_CATEGORIES
from models.experimental.ssd.reference.torch_utils import (
    Weights,
    WeightsEnum,
    DefaultBoxGenerator,
    _ovewrite_value_param,
    _log_api_usage_once,
    retrieve_out_channels,
    ObjectDetection,
    Conv2dNormActivation,
)
from models.experimental.ssd.reference.backbone import mobilenet_v3_large, MobileNet_V3_Large_Weights, MobileNetV3
from models.experimental.ssd.reference.ssd import SSD, SSDScoringHead


class SSDLite320_MobileNet_V3_Large_Weights(WeightsEnum):
    COCO_V1 = Weights(
        url="https://download.pytorch.org/models/ssdlite320_mobilenet_v3_large_coco-a79551df.pth",
        transforms=ObjectDetection,
        meta={
            "num_params": 3440060,
            "categories": _COCO_CATEGORIES,
            "min_size": (1, 1),
            "recipe": "https://github.com/pytorch/vision/tree/main/references/detection#ssdlite320-mobilenetv3-large",
            "_metrics": {
                "COCO-val2017": {
                    "box_map": 21.3,
                }
            },
            "_ops": 0.583,
            "_file_size": 13.418,
            "_docs": """These weights were produced by following a similar training recipe as on the paper.""",
        },
    )
    DEFAULT = COCO_V1


class SSDLiteClassificationHead(SSDScoringHead):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_classes: int,
        norm_layer: Callable[..., nn.Module],
    ):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(_prediction_block(channels, num_classes * anchors, 3, norm_layer))
        _normal_init(cls_logits)
        super().__init__(cls_logits, num_classes)


class SSDLiteRegressionHead(SSDScoringHead):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        norm_layer: Callable[..., nn.Module],
    ):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(_prediction_block(channels, 4 * anchors, 3, norm_layer))
        _normal_init(bbox_reg)
        super().__init__(bbox_reg, 4)


class SSDLiteHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_anchors: List[int],
        num_classes: int,
        norm_layer: Callable[..., nn.Module],
    ):
        super().__init__()
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer)

    def forward(self, x: List[Tensor]) -> Dict[str, Tensor]:
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }


class SSDLiteFeatureExtractorMobileNet(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        c4_pos: int,
        norm_layer: Callable[..., nn.Module],
        width_mult: float = 1.0,
        min_depth: int = 16,
    ):
        super().__init__()
        _log_api_usage_once(self)

        if backbone[c4_pos].use_res_connect:
            raise ValueError("backbone[c4_pos].use_res_connect should be False")

        self.features = nn.Sequential(
            # As described in section 6.3 of MobileNetV3 paper
            nn.Sequential(*backbone[:c4_pos], backbone[c4_pos].block[0]),  # from start until C4 expansion layer
            nn.Sequential(backbone[c4_pos].block[1:], *backbone[c4_pos + 1 :]),  # from C4 depthwise until end
        )

        get_depth = lambda d: max(min_depth, int(d * width_mult))  # noqa: E731
        extra = nn.ModuleList(
            [
                _extra_block(backbone[-1].out_channels, get_depth(512), norm_layer),
                _extra_block(get_depth(512), get_depth(256), norm_layer),
                _extra_block(get_depth(256), get_depth(256), norm_layer),
                _extra_block(get_depth(256), get_depth(128), norm_layer),
            ]
        )
        _normal_init(extra)

        self.extra = extra

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Get feature maps from backbone and extra. Can't be refactored due to JIT limitations.
        output = []
        for block in self.features:
            x = block(x)
            output.append(x)

        for block in self.extra:
            x = block(x)
            output.append(x)

        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def _mobilenet_extractor(
    backbone: Union[None, MobileNetV3],
    trainable_layers: int,
    norm_layer: Callable[..., nn.Module],
):
    backbone = backbone.features
    # Gather the indices of blocks which are strided. These are the locations of C1, ..., Cn-1 blocks.
    # The first and last blocks are always included because they are the C0 (conv1) and Cn.
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    num_stages = len(stage_indices)

    # find the index of the layer from which we won't freeze
    if not 0 <= trainable_layers <= num_stages:
        raise ValueError("trainable_layers should be in the range [0, {num_stages}], instead got {trainable_layers}")
    freeze_before = len(backbone) if trainable_layers == 0 else stage_indices[num_stages - trainable_layers]

    for b in backbone[:freeze_before]:
        for parameter in b.parameters():
            parameter.requires_grad_(False)

    return SSDLiteFeatureExtractorMobileNet(backbone, stage_indices[-2], norm_layer)


def _prediction_block(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    norm_layer: Callable[..., nn.Module],
) -> nn.Sequential:
    return nn.Sequential(
        # 3x3 depthwise with stride 1 and padding 1
        Conv2dNormActivation(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            norm_layer=norm_layer,
            activation_layer=nn.ReLU6,
        ),
        # 1x1 projetion to output channels
        nn.Conv2d(in_channels, out_channels, 1),
    )


def _extra_block(in_channels: int, out_channels: int, norm_layer: Callable[..., nn.Module]) -> nn.Sequential:
    activation = nn.ReLU6
    intermediate_channels = out_channels // 2
    return nn.Sequential(
        # 1x1 projection to half output channels
        Conv2dNormActivation(
            in_channels,
            intermediate_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation,
        ),
        # 3x3 depthwise with stride 2 and padding 1
        Conv2dNormActivation(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=2,
            groups=intermediate_channels,
            norm_layer=norm_layer,
            activation_layer=activation,
        ),
        # 1x1 projetion to output channels
        Conv2dNormActivation(
            intermediate_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation,
        ),
    )


def _normal_init(conv: nn.Module):
    for layer in conv.modules():
        if isinstance(layer, nn.Conv2d):
            torch.nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                torch.nn.init.constant_(layer.bias, 0.0)


def ssdlite320_mobilenet_v3_large(
    *,
    weights: Optional[SSDLite320_MobileNet_V3_Large_Weights] = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1,
    progress: bool = True,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[MobileNet_V3_Large_Weights] = MobileNet_V3_Large_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any,
) -> SSD:
    weights = SSDLite320_MobileNet_V3_Large_Weights.verify(weights)
    weights_backbone = MobileNet_V3_Large_Weights.verify(weights_backbone)

    if weights is not None:
        weights_backbone = None
        num_classes = _ovewrite_value_param("num_classes", num_classes, len(weights.meta["categories"]))
    elif num_classes is None:
        num_classes = 91

    trainable_backbone_layers = 0

    # Enable reduced tail if no pretrained backbone is selected. See Table 6 of MobileNetV3 paper.
    reduce_tail = weights_backbone is None

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    backbone = mobilenet_v3_large(
        weights=weights_backbone,
        progress=progress,
        norm_layer=norm_layer,
        reduced_tail=reduce_tail,
        **kwargs,
    )
    if weights_backbone is None:
        # Change the default initialization scheme if not pretrained
        _normal_init(backbone)
    backbone = _mobilenet_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer,
    )

    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    if len(out_channels) != len(anchor_generator.aspect_ratios):
        raise ValueError(
            f"The length of the output channels from the backbone {len(out_channels)} do not match the length of the anchor generator aspect ratios {len(anchor_generator.aspect_ratios)}"
        )

    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        # Rescale the input in a way compatible to the backbone:
        # The following mean/std rescale the data from [0, 1] to [-1, 1]
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs: Any = {**defaults, **kwargs}
    model = SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=SSDLiteHead(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))

    return model


def retrieve_out_channels(model: nn.Module, size: Tuple[int, int]) -> List[int]:
    in_training = model.training
    model.eval()

    with torch.no_grad():
        # Use dummy data to retrieve the feature map sizes to avoid hard-coding their values
        device = next(model.parameters()).device
        tmp_img = torch.zeros((1, 3, size[1], size[0]), device=device)
        features = model(tmp_img)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        out_channels = [x.size(1) for x in features.values()]

    if in_training:
        model.train()

    return out_channels

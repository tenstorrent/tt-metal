# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from torch import nn
import ttnn
import tt_lib.fallback_ops as fallback_ops
from typing import List, Optional, Tuple, Dict, OrderedDict

from models.experimental.ssd.tt.ssd_backbone import (
    TtSSDLiteFeatureExtractorMobileNet,
)
from models.experimental.ssd.tt.ssd_box_generator import (
    TtDefaultBoxGenerator,
)
from models.experimental.ssd.tt.ssd_lite_head import TtSSDLiteHead

from models.utility_functions import torch_to_tt_tensor_rm, tt_to_torch_tensor
from models.experimental.ssd.tt.ssd_backbone import TtSSDLiteFeatureExtractorMobileNet
from models.experimental.ssd.tt.ssd_box_generator import TtDefaultBoxGenerator
from torchvision.ops.boxes import batched_nms, clip_boxes_to_image
from torchvision.models.detection._utils import BoxCoder, Matcher, SSDMatcher
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class TtSSD(nn.Module):
    __annotations__ = {
        "box_coder": BoxCoder,
        "proposal_matcher": Matcher,
    }

    def __init__(
        self,
        config,
        size: Tuple[int, int],
        num_classes: int,
        image_mean: Optional[List[float]] = None,
        image_std: Optional[List[float]] = None,
        score_thresh: float = 0.01,
        nms_thresh: float = 0.45,
        detections_per_image: int = 200,
        iou_thresh: float = 0.5,
        topk_candidates: int = 400,
        positive_fraction: float = 0.25,
        state_dict=None,
        base_address="",
        device=None,
    ):
        super().__init__()
        self.device = device
        self.state_dict = state_dict
        self.base_address = base_address
        self.box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.proposal_matcher = SSDMatcher(iou_thresh)
        if image_mean is None:
            image_mean = [0.485, 0.456, 0.406]
        if image_std is None:
            image_std = [0.229, 0.224, 0.225]

        self.Ttbackbone = TtSSDLiteFeatureExtractorMobileNet(
            config,
            state_dict=self.state_dict,
            base_address=f"backbone",
            device=device,
        )

        self.anchor_generator = TtDefaultBoxGenerator(
            [[2, 3] for _ in range(6)],
            min_ratio=0.2,
            max_ratio=0.95,
            device=self.device,
        )

        self.Ttssdhead = TtSSDLiteHead(
            config,
            in_channels=self.get_in_channels(self.Ttbackbone),
            num_anchors=self.anchor_generator.num_anchors_per_location(),
            num_classes=num_classes,
            num_columns=4,
            state_dict=self.state_dict,
            base_address=self.base_address,
            device=self.device,
        )
        self.transform = GeneralizedRCNNTransform(
            min(size),
            max(size),
            image_mean,
            image_std,
            size_divisible=1,
            fixed_size=size,
        )

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.detections_per_image = detections_per_image
        self.topk_candidates = topk_candidates
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

        # used only on torchscript mode
        self._has_warned = False

    def get_in_channels(self, backbone: TtSSDLiteFeatureExtractorMobileNet):
        size = (320, 320)
        temporary_image = ttnn.ones([1, 3, size[1], size[0]], device=self.device)
        backbone.eval()
        features = backbone(temporary_image)
        out_channels = [tensor.shape.with_tile_padding()[1] for i, tensor in features.items()]
        return out_channels

    def postprocess_detections(
        self,
        head_outputs: Dict[str, ttnn.Tensor],
        image_anchors: List[ttnn.Tensor],
        image_shapes: Tuple[int, int],
    ) -> List[Dict[str, ttnn.Tensor]]:
        bbox_regression = head_outputs["bbox_regression"]
        pred_scores = fallback_ops.softmax(head_outputs["cls_logits"], dim=-1)

        num_classes = pred_scores.shape.with_tile_padding()[-1]

        detections: List[Dict[str, ttnn.Tensor]] = []

        boxes = tt_to_torch_tensor(bbox_regression).to(torch.float).squeeze(0).squeeze(0)
        anchors = tt_to_torch_tensor(image_anchors[0]).squeeze(0).squeeze(0)
        scores = tt_to_torch_tensor(pred_scores).to(torch.float).squeeze(0).squeeze(0)

        boxes = self.box_coder.decode_single(boxes, anchors)

        boxes = clip_boxes_to_image(boxes, image_shapes)

        image_boxes = []
        image_scores = []
        image_labels = []
        for label in range(1, num_classes):
            score = scores[:, label]

            keep_idxs = score > self.score_thresh
            score = score[keep_idxs]
            box = boxes[keep_idxs]

            num_topk = min(score.size(0), self.topk_candidates)
            score, idxs = score.topk(num_topk)
            box = box[idxs]

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64))

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)

        # non-maximum suppression

        keep = batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = keep[: self.detections_per_image]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
            }
        )
        return detections

    def forward(
        self,
        image: ttnn.Tensor,
        targets: Optional[List[Dict[str, ttnn.Tensor]]] = None,
    ) -> List[Dict[str, ttnn.Tensor]]:
        original_image_sizes: List[tuple[int, int]] = []

        val = image.shape.with_tile_padding()[-2:]
        original_image_sizes.append((val[0], val[1]))

        image = tt_to_torch_tensor(image)
        image, targets = self.transform(image, targets)
        image_shape = image.image_sizes.copy()
        image = torch_to_tt_tensor_rm(image.tensors, self.device)
        features = self.Ttbackbone(image)

        if isinstance(features, ttnn.Tensor):
            features = OrderedDict([("0", features)])

        features = list(features.values())
        head_outputs = self.Ttssdhead(features)
        anchors = self.anchor_generator(image, features)
        detections: List[Dict[str, ttnn.Tensor]] = []

        detections = self.postprocess_detections(head_outputs, anchors, image_shape[0])
        detections = self.transform.postprocess(detections, image_shape, original_image_sizes)
        return detections

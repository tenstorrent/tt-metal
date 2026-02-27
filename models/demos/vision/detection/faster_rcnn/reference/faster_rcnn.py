# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
PyTorch reference implementation wrapper for Faster-RCNN with ResNet-50-FPN backbone.
Uses torchvision's pretrained model as the ground truth reference for PCC validation.
"""

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.ops import misc as misc_nn_ops


class FasterRCNNReference:
    """Wrapper around torchvision's Faster-RCNN for reference inference."""

    def __init__(self, pretrained=True):
        if pretrained:
            self.model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
        else:
            self.model = fasterrcnn_resnet50_fpn(weights=None, num_classes=91)
        self.model.eval()

    def get_backbone(self):
        return self.model.backbone

    def get_rpn(self):
        return self.model.rpn

    def get_roi_heads(self):
        return self.model.roi_heads

    @torch.no_grad()
    def run_backbone(self, images):
        """Run only the backbone+FPN, returning multi-scale feature maps."""
        return self.model.backbone(images)

    @torch.no_grad()
    def run_full_model(self, images):
        """Run full Faster-RCNN inference, returning detections."""
        return self.model(images)

    @torch.no_grad()
    def run_backbone_body(self, images):
        """Run only the ResNet body (without FPN), returning intermediate features."""
        backbone = self.model.backbone
        body = backbone.body
        features = {}
        x = images
        for name, module in body.named_children():
            x = module(x)
            if name in backbone.body.return_layers:
                features[backbone.body.return_layers[name]] = x
        return features

    @torch.no_grad()
    def run_rpn_head(self, features):
        """Run the RPN head on feature maps, returning objectness and bbox regression."""
        rpn = self.model.rpn
        head = rpn.head
        objectness = []
        pred_bbox_deltas = []
        for feature in features.values():
            t = head.conv(feature)
            objectness.append(head.cls_logits(t))
            pred_bbox_deltas.append(head.bbox_pred(t))
        return objectness, pred_bbox_deltas


def get_resnet50_backbone_state_dict(model):
    """Extract ResNet-50 backbone weights from the full Faster-RCNN model."""
    state_dict = {}
    for name, param in model.backbone.body.named_parameters():
        state_dict[name] = param.data
    for name, buffer in model.backbone.body.named_buffers():
        state_dict[name] = buffer.data
    return state_dict


def get_fpn_state_dict(model):
    """Extract FPN weights from the full Faster-RCNN model."""
    state_dict = {}
    for name, param in model.backbone.fpn.named_parameters():
        state_dict[name] = param.data
    for name, buffer in model.backbone.fpn.named_buffers():
        state_dict[name] = buffer.data
    return state_dict


def get_rpn_state_dict(model):
    """Extract RPN weights from the full Faster-RCNN model."""
    state_dict = {}
    for name, param in model.rpn.named_parameters():
        state_dict[name] = param.data
    return state_dict


def get_roi_heads_state_dict(model):
    """Extract ROI head weights from the full Faster-RCNN model."""
    state_dict = {}
    for name, param in model.roi_heads.named_parameters():
        state_dict[name] = param.data
    return state_dict

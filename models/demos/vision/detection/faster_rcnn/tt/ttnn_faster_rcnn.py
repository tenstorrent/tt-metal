# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Full Faster-RCNN model implementation using TTNN APIs.

Architecture:
    - ResNet-50 backbone (TTNN): extracts multi-scale features
    - FPN (TTNN): produces pyramid feature maps (P2-P5 + pool)
    - RPN head convolutions (TTNN): classification + box regression per level
    - RPN proposal generation (CPU): anchor generation, NMS
    - ROI Align (CPU): extract features for each proposal
    - Box Head FC layers (CPU): classification + box regression
    - Post-processing (CPU): NMS, score filtering

The backbone, FPN, and RPN convolutions run on TT hardware for maximum throughput.
Dynamic operations (NMS, ROI Align, anchor generation) run on CPU.
"""

import math
from collections import OrderedDict

import torch

import ttnn

from models.demos.vision.detection.faster_rcnn.tt.ttnn_resnet50_backbone import TtResNet50Backbone, TtConv2D
from models.demos.vision.detection.faster_rcnn.tt.ttnn_fpn import TtFPN

FEATURE_LEVEL_KEYS = ["0", "1", "2", "3", "pool"]
FPN_CHANNELS = 256


class TtRPNHead:
    """RPN Head in TTNN.

    Applies a 3x3 conv + ReLU shared across all FPN levels,
    then 1x1 convs for objectness classification and box regression.
    The same weight tensors are reused for each level.
    """

    def __init__(self, parameters, device, batch_size=1, num_anchors=3):
        self.device = device
        self.batch_size = batch_size
        self.num_anchors = num_anchors

        self.conv_weight = parameters["rpn.conv.0.0.weight"]
        self.conv_bias = parameters["rpn.conv.0.0.bias"]
        self.cls_weight = parameters["rpn.cls_logits.weight"]
        self.cls_bias = parameters["rpn.cls_logits.bias"]
        self.bbox_weight = parameters["rpn.bbox_pred.weight"]
        self.bbox_bias = parameters["rpn.bbox_pred.bias"]

        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    def _run_conv_on_feature(
        self, feature, in_channels, out_channels, weight, bias, kernel_size, padding, activation=None
    ):
        """Run a single convolution on a feature tensor."""
        spatial = feature.shape[2] // self.batch_size
        h = int(math.sqrt(spatial))
        w = h

        conv_config = ttnn.Conv2dConfig(
            weights_dtype=ttnn.bfloat8_b,
            activation=activation,
            shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            deallocate_activation=False,
            enable_act_double_buffer=True,
            output_layout=ttnn.TILE_LAYOUT,
            reallocate_halo_output=False,
            reshard_if_not_optimal=True,
            enable_weights_double_buffer=True,
        )

        [out, [h_out, w_out], [weight, bias]] = ttnn.conv2d(
            input_tensor=feature,
            weight_tensor=weight,
            in_channels=in_channels,
            out_channels=out_channels,
            device=self.device,
            bias_tensor=bias,
            kernel_size=(kernel_size, kernel_size),
            stride=(1, 1),
            padding=(padding, padding),
            batch_size=self.batch_size,
            input_height=h,
            input_width=w,
            conv_config=conv_config,
            compute_config=self.compute_config,
            return_weights_and_bias=True,
            return_output_dim=True,
            dtype=ttnn.bfloat16,
        )
        return out, weight, bias

    def __call__(self, features):
        """Run RPN head on each FPN feature level.

        Args:
            features: dict with keys "0"-"3" and "pool"

        Returns:
            objectness: list of TTNN tensors per level
            pred_bbox_deltas: list of TTNN tensors per level
        """
        objectness = []
        pred_bbox_deltas = []

        for key in FEATURE_LEVEL_KEYS:
            if key not in features:
                continue
            feat = features[key]

            t, self.conv_weight, self.conv_bias = self._run_conv_on_feature(
                feat,
                FPN_CHANNELS,
                FPN_CHANNELS,
                self.conv_weight,
                self.conv_bias,
                kernel_size=3,
                padding=1,
                activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU),
            )

            cls, self.cls_weight, self.cls_bias = self._run_conv_on_feature(
                t,
                FPN_CHANNELS,
                self.num_anchors,
                self.cls_weight,
                self.cls_bias,
                kernel_size=1,
                padding=0,
            )

            bbox, self.bbox_weight, self.bbox_bias = self._run_conv_on_feature(
                t,
                FPN_CHANNELS,
                self.num_anchors * 4,
                self.bbox_weight,
                self.bbox_bias,
                kernel_size=1,
                padding=0,
            )

            ttnn.deallocate(t)
            objectness.append(cls)
            pred_bbox_deltas.append(bbox)

        return objectness, pred_bbox_deltas


def ttnn_to_torch_feature(ttnn_tensor, batch_size, channels):
    """Convert TTNN feature tensor [1,1,NHW,C] to PyTorch [N,C,H,W]."""
    torch_tensor = ttnn.to_torch(ttnn.from_device(ttnn_tensor))
    nhw = torch_tensor.shape[2]
    c = torch_tensor.shape[3]
    spatial = nhw // batch_size
    h = int(math.sqrt(spatial))
    w = spatial // h
    return torch_tensor.reshape(batch_size, h, w, c).permute(0, 3, 1, 2).contiguous().float()


class TtFasterRCNN:
    """Full Faster-RCNN model with TTNN backbone, FPN, and RPN convolutions.

    Backbone + FPN + RPN conv layers run on TT hardware.
    Proposal generation, ROI Align, and box head run on CPU.
    """

    def __init__(self, parameters, device, torch_model, batch_size=1, input_height=320, input_width=320):
        self.device = device
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width

        self.backbone = TtResNet50Backbone(parameters, device, batch_size)
        self.fpn = TtFPN(parameters, device, batch_size)
        self.rpn_head = TtRPNHead(parameters, device, batch_size)

        self.torch_rpn = torch_model.rpn
        self.torch_roi_heads = torch_model.roi_heads
        self.torch_transform = torch_model.transform

    def _run_rpn_proposals(self, features_torch, image_sizes, objectness_torch, pred_bbox_deltas_torch):
        """Generate proposals using PyTorch RPN infrastructure (CPU)."""
        rpn = self.torch_rpn

        feature_list = list(features_torch.values())
        anchors = rpn.anchor_generator(None, feature_list)

        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness_torch]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]

        objectness_flat, pred_bbox_deltas_flat = rpn.concat_box_prediction_layers(
            objectness_torch, pred_bbox_deltas_torch
        )

        proposals = rpn.box_coder.decode(pred_bbox_deltas_flat.detach(), anchors)
        proposals = proposals.view(len(anchors), -1, 4)
        objectness_flat = objectness_flat.view(len(anchors), -1)

        boxes, scores = rpn.filter_proposals(proposals, objectness_flat, image_sizes, num_anchors_per_level)
        return boxes, scores

    def _run_roi_heads(self, features_torch, proposals, image_sizes):
        """Run ROI heads on CPU for classification and box regression."""
        roi_heads = self.torch_roi_heads

        box_features = roi_heads.box_roi_pool(features_torch, proposals, image_sizes)
        box_features = roi_heads.box_head(box_features)
        class_logits, box_regression = roi_heads.box_predictor(box_features)

        boxes, scores, labels = roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_sizes)

        result = []
        for i in range(len(boxes)):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        return result

    def __call__(self, images_tensor):
        """Run full Faster-RCNN inference.

        Args:
            images_tensor: NCHW PyTorch tensor (unnormalized, [0, 1] range)

        Returns:
            List of dicts with 'boxes', 'labels', 'scores' per image
        """
        with torch.no_grad():
            image_list, _ = self.torch_transform(images_tensor)
            transformed_images = image_list.tensors
            image_sizes = image_list.image_sizes

            nhwc = transformed_images.permute(0, 2, 3, 1).contiguous()
            nhwc = torch.nn.functional.pad(nhwc, (0, 16 - nhwc.shape[-1]), value=0)
            ttnn_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn_input = ttnn.reshape(
                ttnn_input,
                (1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]),
            )

            backbone_features = self.backbone(ttnn_input)
            fpn_features = self.fpn(backbone_features)
            objectness_ttnn, pred_bbox_deltas_ttnn = self.rpn_head(fpn_features)

            features_torch = OrderedDict()
            for key in FEATURE_LEVEL_KEYS:
                if key in fpn_features:
                    features_torch[key] = ttnn_to_torch_feature(fpn_features[key], self.batch_size, FPN_CHANNELS)

            objectness_torch = []
            pred_bbox_deltas_torch = []
            for obj_ttnn, bbox_ttnn in zip(objectness_ttnn, pred_bbox_deltas_ttnn):
                obj_torch = ttnn_to_torch_feature(obj_ttnn, self.batch_size, self.rpn_head.num_anchors)
                objectness_torch.append(obj_torch)

                bbox_torch = ttnn_to_torch_feature(bbox_ttnn, self.batch_size, self.rpn_head.num_anchors * 4)
                pred_bbox_deltas_torch.append(bbox_torch)

            proposals, _ = self._run_rpn_proposals(
                features_torch, image_sizes, objectness_torch, pred_bbox_deltas_torch
            )

            detections = self._run_roi_heads(features_torch, proposals, image_sizes)

            original_sizes = [(self.input_height, self.input_width)] * self.batch_size
            detections = self.torch_transform.postprocess(detections, image_sizes, original_sizes)

            for key in fpn_features:
                ttnn.deallocate(fpn_features[key])
            for key in backbone_features:
                ttnn.deallocate(backbone_features[key])
            for t in objectness_ttnn + pred_bbox_deltas_ttnn:
                ttnn.deallocate(t)

        return detections


class TtFasterRCNNBackboneOnly:
    """Runs only the backbone+FPN on TTNN for PCC validation."""

    def __init__(self, parameters, device, batch_size=1):
        self.device = device
        self.batch_size = batch_size
        self.backbone = TtResNet50Backbone(parameters, device, batch_size)
        self.fpn = TtFPN(parameters, device, batch_size)

    def __call__(self, ttnn_input):
        backbone_features = self.backbone(ttnn_input)
        fpn_features = self.fpn(backbone_features)
        return fpn_features

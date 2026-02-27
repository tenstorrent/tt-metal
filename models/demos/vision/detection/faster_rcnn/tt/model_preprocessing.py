# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
Weight preprocessing for Faster-RCNN TTNN implementation.
Handles BatchNorm folding into Conv2d and conversion to TTNN tensors.

Explicitly traverses the known torchvision Faster-RCNN model hierarchy:
  - backbone.body (ResNet-50 with FrozenBatchNorm2d)
  - backbone.fpn (Feature Pyramid Network)
  - rpn.head (Region Proposal Network head)
  - roi_heads (box_head + box_predictor)
"""

import torch
import torch.nn as nn

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers


def fold_batch_norm_into_conv(conv, bn):
    """Fold BatchNorm (or FrozenBatchNorm2d) into Conv2d weights and bias."""
    weight = conv.weight.data.clone()
    bias = conv.bias.data.clone() if conv.bias is not None else torch.zeros(conv.out_channels)

    running_mean = bn.running_mean.data
    running_var = bn.running_var.data
    eps = bn.eps if hasattr(bn, "eps") else 1e-5
    bn_weight = bn.weight.data if bn.weight is not None else torch.ones_like(running_mean)
    bn_bias = bn.bias.data if bn.bias is not None else torch.zeros_like(running_mean)

    inv_std = bn_weight / torch.sqrt(running_var + eps)
    weight = weight * inv_std[:, None, None, None]
    bias = (bias - running_mean) * inv_std + bn_bias

    return weight, bias


def convert_weight_bias_to_ttnn(weight, bias, mesh_mapper=None):
    """Convert PyTorch weight and bias tensors to TTNN format."""
    weight_ttnn = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    bias_reshaped = torch.reshape(bias, (1, 1, 1, -1))
    bias_ttnn = ttnn.from_torch(bias_reshaped, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    return weight_ttnn, bias_ttnn


def convert_linear_weight_bias_to_ttnn(weight, bias, mesh_mapper=None):
    """Convert linear layer weights to TTNN format (transposed for matmul)."""
    weight_t = weight.T.contiguous()
    weight_ttnn = ttnn.from_torch(weight_t, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    if bias is not None:
        bias_ttnn = ttnn.from_torch(bias.reshape(1, -1), dtype=ttnn.float32, mesh_mapper=mesh_mapper)
    else:
        bias_ttnn = None
    return weight_ttnn, bias_ttnn


def preprocess_resnet50_backbone(body, mesh_mapper=None):
    """Extract and fold all Conv2d+BN pairs from the ResNet-50 backbone body.

    Explicitly traverses the known structure:
      - body.conv1 + body.bn1
      - body.layer{1-4}.{block_idx}.conv{1-3} + bn{1-3}
      - body.layer{1-4}.{block_idx}.downsample.{0,1} (if exists)
    """
    parameters = {}

    weight, bias = fold_batch_norm_into_conv(body.conv1, body.bn1)
    w, b = convert_weight_bias_to_ttnn(weight, bias, mesh_mapper)
    parameters["backbone.conv1.weight"] = w
    parameters["backbone.conv1.bias"] = b

    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(body, layer_name)
        for block_idx, block in enumerate(layer):
            prefix = f"backbone.{layer_name}.{block_idx}"

            for conv_idx in [1, 2, 3]:
                conv = getattr(block, f"conv{conv_idx}")
                bn = getattr(block, f"bn{conv_idx}")
                weight, bias = fold_batch_norm_into_conv(conv, bn)
                w, b = convert_weight_bias_to_ttnn(weight, bias, mesh_mapper)
                parameters[f"{prefix}.conv{conv_idx}.weight"] = w
                parameters[f"{prefix}.conv{conv_idx}.bias"] = b

            if block.downsample is not None:
                ds_conv = block.downsample[0]
                ds_bn = block.downsample[1]
                weight, bias = fold_batch_norm_into_conv(ds_conv, ds_bn)
                w, b = convert_weight_bias_to_ttnn(weight, bias, mesh_mapper)
                parameters[f"{prefix}.downsample.0.weight"] = w
                parameters[f"{prefix}.downsample.0.bias"] = b

    return parameters


def preprocess_fpn(fpn, mesh_mapper=None):
    """Extract FPN inner_blocks and layer_blocks weights (all plain Conv2d)."""
    parameters = {}

    for idx, conv in enumerate(fpn.inner_blocks):
        weight = conv.weight.data.clone()
        bias = conv.bias.data.clone() if conv.bias is not None else torch.zeros(conv.out_channels)
        w, b = convert_weight_bias_to_ttnn(weight, bias, mesh_mapper)
        parameters[f"fpn.inner_blocks.{idx}.weight"] = w
        parameters[f"fpn.inner_blocks.{idx}.bias"] = b

    for idx, conv in enumerate(fpn.layer_blocks):
        weight = conv.weight.data.clone()
        bias = conv.bias.data.clone() if conv.bias is not None else torch.zeros(conv.out_channels)
        w, b = convert_weight_bias_to_ttnn(weight, bias, mesh_mapper)
        parameters[f"fpn.layer_blocks.{idx}.weight"] = w
        parameters[f"fpn.layer_blocks.{idx}.bias"] = b

    return parameters


def preprocess_rpn(rpn, mesh_mapper=None):
    """Extract RPN head weights.

    Structure: head.conv = Sequential(Conv2dNormActivation(Conv2d, ReLU))
               head.cls_logits = Conv2d
               head.bbox_pred = Conv2d
    """
    parameters = {}
    head = rpn.head

    for idx, conv_block in enumerate(head.conv):
        conv_layer = conv_block[0]
        weight = conv_layer.weight.data.clone()
        bias = conv_layer.bias.data.clone() if conv_layer.bias is not None else torch.zeros(conv_layer.out_channels)
        w, b = convert_weight_bias_to_ttnn(weight, bias, mesh_mapper)
        parameters[f"rpn.conv.{idx}.0.weight"] = w
        parameters[f"rpn.conv.{idx}.0.bias"] = b

    for name, conv in [("cls_logits", head.cls_logits), ("bbox_pred", head.bbox_pred)]:
        weight = conv.weight.data.clone()
        bias = conv.bias.data.clone()
        w, b = convert_weight_bias_to_ttnn(weight, bias, mesh_mapper)
        parameters[f"rpn.{name}.weight"] = w
        parameters[f"rpn.{name}.bias"] = b

    return parameters


def preprocess_roi_heads(roi_heads, device, mesh_mapper=None):
    """Extract ROI head FC layer weights and move to device."""
    parameters = {}

    box_head = roi_heads.box_head
    for name, module in box_head.named_modules():
        if isinstance(module, nn.Linear):
            w, b = convert_linear_weight_bias_to_ttnn(module.weight.data, module.bias.data, mesh_mapper)
            w = ttnn.to_device(w, device)
            b = ttnn.to_device(b, device)
            parameters[f"roi_heads.box_head.{name}.weight"] = w
            parameters[f"roi_heads.box_head.{name}.bias"] = b

    box_predictor = roi_heads.box_predictor
    for attr_name in ["cls_score", "bbox_pred"]:
        linear = getattr(box_predictor, attr_name)
        w, b = convert_linear_weight_bias_to_ttnn(linear.weight.data, linear.bias.data, mesh_mapper)
        w = ttnn.to_device(w, device)
        b = ttnn.to_device(b, device)
        parameters[f"roi_heads.box_predictor.{attr_name}.weight"] = w
        parameters[f"roi_heads.box_predictor.{attr_name}.bias"] = b

    return parameters


def create_faster_rcnn_model_parameters(torch_model, device):
    """Create all TTNN model parameters from a torchvision Faster-RCNN model."""
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = {}
    parameters.update(preprocess_resnet50_backbone(torch_model.backbone.body, weights_mesh_mapper))
    parameters.update(preprocess_fpn(torch_model.backbone.fpn, weights_mesh_mapper))
    parameters.update(preprocess_rpn(torch_model.rpn, weights_mesh_mapper))
    parameters.update(preprocess_roi_heads(torch_model.roi_heads, device, weights_mesh_mapper))

    return parameters


def create_faster_rcnn_input_tensors(batch=1, input_height=320, input_width=320, mesh_mapper=None):
    """Create input tensors for Faster-RCNN inference.

    Returns:
        torch_input: NCHW tensor for PyTorch reference [N, 3, H, W]
        ttnn_input: Flattened [1, 1, N*H*W, 16] tensor for TTNN (padded to 16 channels)
    """
    torch_input = torch.randn(batch, 3, input_height, input_width)

    nhwc = torch.permute(torch_input, (0, 2, 3, 1))
    nhwc = torch.nn.functional.pad(nhwc, (0, 16 - nhwc.shape[-1]), value=0)
    ttnn_input = ttnn.from_torch(nhwc, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, mesh_mapper=mesh_mapper)
    ttnn_input = ttnn.reshape(
        ttnn_input,
        (1, 1, ttnn_input.shape[0] * ttnn_input.shape[1] * ttnn_input.shape[2], ttnn_input.shape[3]),
    )

    return torch_input, ttnn_input

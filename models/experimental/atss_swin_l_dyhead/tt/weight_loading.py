# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Weight loading for ATSS detection model (FPN + DyHead + ATSS Head).

The Swin-L backbone weights are loaded by the swin_l module's own
model_preprocessing.py. This module handles the detection-specific components.

mmdet checkpoint key structure:
  FPN:
    neck.0.lateral_convs.{i}.conv.{weight,bias}
    neck.0.fpn_convs.{i}.conv.{weight,bias}

  DyHead:
    neck.1.dyhead_blocks.{b}.spatial_conv_offset.{weight,bias}
    neck.1.dyhead_blocks.{b}.spatial_conv_mid.conv.{weight,bias}
    neck.1.dyhead_blocks.{b}.spatial_conv_mid.norm.{weight,bias}
    neck.1.dyhead_blocks.{b}.spatial_conv_high.conv.{weight,bias}
    neck.1.dyhead_blocks.{b}.spatial_conv_high.norm.{weight,bias}
    neck.1.dyhead_blocks.{b}.spatial_conv_low.conv.{weight,bias}
    neck.1.dyhead_blocks.{b}.spatial_conv_low.norm.{weight,bias}
    neck.1.dyhead_blocks.{b}.scale_attn_module.1.{weight,bias}
    neck.1.dyhead_blocks.{b}.task_attn_module.conv1.conv.{weight,bias}
    neck.1.dyhead_blocks.{b}.task_attn_module.conv2.conv.{weight,bias}

  ATSS Head:
    bbox_head.atss_cls.{weight,bias}
    bbox_head.atss_reg.{weight,bias}
    bbox_head.atss_centerness.{weight,bias}
    bbox_head.scales.{i}.scale
"""

from typing import Dict

import torch
import ttnn


def _get(sd: dict, key: str) -> torch.Tensor:
    if key not in sd:
        raise KeyError(f"Missing key in checkpoint: {key}")
    return sd[key]


def _to_ttnn_conv_weight(weight: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Conv2d weight [out_ch, in_ch, kH, kW] → ttnn tensor."""
    return ttnn.from_torch(weight, dtype=dtype)


def _to_ttnn_conv_bias(bias: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Conv2d bias [out_ch] → ttnn tensor [1, 1, 1, out_ch]."""
    return ttnn.from_torch(bias.reshape(1, 1, 1, -1), dtype=dtype)


def _to_ttnn_scale(scale_val: torch.Tensor, dtype=ttnn.bfloat16) -> ttnn.Tensor:
    """Scalar scale → ttnn tensor [1, 1] for broadcasting."""
    return ttnn.from_torch(scale_val.reshape(1, 1), dtype=dtype, layout=ttnn.TILE_LAYOUT)


def load_fpn_weights(
    checkpoint_path: str,
    device: ttnn.Device,
    in_channels=(384, 768, 1536),
    out_channels=256,
    num_outs=5,
) -> dict:
    """
    Load FPN weights from mmdet checkpoint.
    Returns dict matching TtFPN expected structure.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    params: Dict = {"lateral_convs": {}, "fpn_convs": {}}
    num_ins = len(in_channels)

    for i in range(num_ins):
        params["lateral_convs"][i] = {
            "weight": _to_ttnn_conv_weight(_get(sd, f"neck.0.lateral_convs.{i}.conv.weight")),
            "bias": _to_ttnn_conv_bias(_get(sd, f"neck.0.lateral_convs.{i}.conv.bias")),
        }

    for i in range(num_ins):
        params["fpn_convs"][i] = {
            "weight": _to_ttnn_conv_weight(_get(sd, f"neck.0.fpn_convs.{i}.conv.weight")),
            "bias": _to_ttnn_conv_bias(_get(sd, f"neck.0.fpn_convs.{i}.conv.bias")),
        }

    num_extra = num_outs - num_ins
    for i in range(num_extra):
        params["fpn_convs"][num_ins + i] = {
            "weight": _to_ttnn_conv_weight(_get(sd, f"neck.0.fpn_convs.{num_ins + i}.conv.weight")),
            "bias": _to_ttnn_conv_bias(_get(sd, f"neck.0.fpn_convs.{num_ins + i}.conv.bias")),
        }

    params = _to_device_recursive(params, device)
    return params


def load_atss_head_weights(
    checkpoint_path: str,
    device: ttnn.Device,
    num_classes=80,
    num_anchors=1,
    num_levels=5,
) -> dict:
    """
    Load ATSS Head weights from mmdet checkpoint.
    Returns dict matching TtATSSHead expected structure.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    params: Dict = {}

    params["atss_cls"] = {
        "weight": _to_ttnn_conv_weight(_get(sd, "bbox_head.atss_cls.weight")),
        "bias": _to_ttnn_conv_bias(_get(sd, "bbox_head.atss_cls.bias")),
    }
    params["atss_reg"] = {
        "weight": _to_ttnn_conv_weight(_get(sd, "bbox_head.atss_reg.weight")),
        "bias": _to_ttnn_conv_bias(_get(sd, "bbox_head.atss_reg.bias")),
    }
    params["atss_centerness"] = {
        "weight": _to_ttnn_conv_weight(_get(sd, "bbox_head.atss_centerness.weight")),
        "bias": _to_ttnn_conv_bias(_get(sd, "bbox_head.atss_centerness.bias")),
    }

    params["scales"] = {}
    for i in range(num_levels):
        params["scales"][i] = _to_ttnn_scale(_get(sd, f"bbox_head.scales.{i}.scale"))

    params = _to_device_recursive(params, device)
    return params


def load_dyhead_weights(
    checkpoint_path: str,
    device: ttnn.Device,
    num_blocks=6,
    in_channels=256,
    out_channels=256,
) -> dict:
    """
    Load DyHead weights from mmdet checkpoint.

    DyHead uses DCNv2 which cannot run natively on TTNN, so the DyHead
    forward pass will fall back to PyTorch. This function loads the weights
    as PyTorch tensors (not TTNN tensors) into the reference DyHead model.

    Returns the raw state_dict subset for DyHead.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    dyhead_sd = {}
    prefix = "neck.1."
    for k, v in sd.items():
        if k.startswith(prefix):
            new_k = k[len(prefix) :]
            # mmdet ConvModule wraps conv as .conv.{weight,bias}
            # our standalone DyReLU uses nn.Sequential: .0.{weight,bias}
            new_k = new_k.replace("task_attn_module.conv1.conv.", "task_attn_module.conv1.0.")
            new_k = new_k.replace("task_attn_module.conv2.conv.", "task_attn_module.conv2.0.")
            dyhead_sd[new_k] = v

    return dyhead_sd


def _to_device_recursive(d, device):
    """Recursively move all ttnn.Tensor values in a nested dict to device."""
    if isinstance(d, ttnn.Tensor):
        return ttnn.to_device(d, device)
    if isinstance(d, dict):
        return {k: _to_device_recursive(v, device) for k, v in d.items()}
    return d

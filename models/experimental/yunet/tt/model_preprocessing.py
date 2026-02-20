# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Weight preprocessing and input tensor creation for YUNet."""

import torch
import ttnn

from models.experimental.yunet.common import YUNET_INPUT_SIZE


def create_yunet_input_tensor(image_tensor: torch.Tensor, device) -> ttnn.Tensor:
    """
    Create TTNN input tensor from image.

    Args:
        image_tensor: Torch tensor in NHWC format, bfloat16, [0-255] range
        device: TTNN device

    Returns:
        TTNN tensor ready for inference
    """
    return ttnn.from_torch(image_tensor, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)


def preprocess_image(image, target_size=YUNET_INPUT_SIZE):
    """
    Preprocess image for YUNet inference.

    Args:
        image: BGR image from cv2.imread
        target_size: Target size (default 640)

    Returns:
        torch.Tensor in NHWC format, bfloat16
    """
    import cv2

    # Resize
    image_resized = cv2.resize(image, (target_size, target_size))
    # BGR -> RGB
    image_rgb = image_resized[:, :, ::-1]
    # To tensor [H, W, C] -> [1, H, W, C]
    tensor = torch.from_numpy(image_rgb.copy()).float()
    tensor = tensor.unsqueeze(0).to(torch.bfloat16)

    return tensor


def extract_weights_from_torch_model(torch_model) -> dict:
    """
    Extract and organize weights from fused PyTorch model.

    Args:
        torch_model: Fused PyTorch YUNet model

    Returns:
        Dictionary of weight tensors
    """
    weights = {}

    # Backbone weights
    backbone = torch_model.backbone
    weights["backbone"] = {
        "p1": _extract_sequential_weights(backbone.p1),
        "p2": _extract_sequential_weights(backbone.p2),
        "p3": _extract_sequential_weights(backbone.p3),
        "p4": _extract_sequential_weights(backbone.p4),
        "p5": _extract_sequential_weights(backbone.p5),
    }

    # Neck weights
    neck = torch_model.neck
    weights["neck"] = {
        "conv1": _extract_dpunit_weights(neck.conv1),
        "conv2": _extract_dpunit_weights(neck.conv2),
        "conv3": _extract_dpunit_weights(neck.conv3),
    }

    # Head weights
    head = torch_model.head
    weights["head"] = {
        "m": [_extract_dpunit_weights(m) for m in head.m],
        "cls": [(h.weight, h.bias) for h in head.cls],
        "box": [(h.weight, h.bias) for h in head.box],
        "obj": [(h.weight, h.bias) for h in head.obj],
        "kpt": [(h.weight, h.bias) for h in head.kpt],
    }

    return weights


def _extract_sequential_weights(seq):
    """Extract weights from a Sequential module."""
    weights = []
    for module in seq:
        if hasattr(module, "conv"):
            # Conv or DPUnit
            if hasattr(module.conv, "weight"):
                weights.append((module.conv.weight, module.conv.bias))
            elif hasattr(module, "conv1"):
                weights.append(_extract_dpunit_weights(module))
        elif hasattr(module, "conv1"):
            # DPUnit
            weights.append(_extract_dpunit_weights(module))
    return weights


def _extract_dpunit_weights(dpunit):
    """Extract weights from a DPUnit module."""
    return {
        "conv1": (dpunit.conv1.weight, dpunit.conv1.bias),
        "conv2": (dpunit.conv2.conv.weight, dpunit.conv2.conv.bias),
    }

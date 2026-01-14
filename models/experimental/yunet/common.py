# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Common constants and utilities for YUNet face detection model."""

import torch

# Model constants
YUNET_INPUT_SIZE = 320  # Original repo uses 640, 320 is fast mode
YUNET_L1_SMALL_SIZE = 245760
YUNET_TRACE_REGION_SIZE = 6434816  # Trace region size for performant runner
YUNET_BATCH_SIZE = 1

# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_NMS_IOU_THRESHOLD = 0.4

# Feature map strides for multi-scale detection
STRIDES = [8, 16, 32]


def load_torch_model(weights_path: str = None):
    """Load PyTorch YUNet model with optional pretrained weights.

    Requires YUNet repo to be cloned to models/experimental/yunet/YUNet/
    Run: ./setup.sh or manually clone from https://github.com/jahongir7174/YUNet
    """
    import sys
    from models.experimental.yunet.YUNet.nets import nn as YUNet_nn

    model = YUNet_nn.version_n()

    if weights_path:
        # Temporarily add YUNet path for unpickling weights (they reference 'nets' module)
        yunet_path = "models/experimental/yunet/YUNet"
        sys.path.insert(0, yunet_path)
        try:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"].state_dict()
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        finally:
            sys.path.remove(yunet_path)

    return model.fuse().eval()


def get_default_weights_path():
    """Get default path to pretrained weights."""
    return "models/experimental/yunet/YUNet/weights/best.pt"

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Common constants and utilities for YUNet face detection model."""

import os
import subprocess
import torch

# Model constants
YUNET_INPUT_SIZE = 640
YUNET_L1_SMALL_SIZE = 24576  # Same as ResNet50, reduced from 245760 for Wormhole compatibility
YUNET_BATCH_SIZE = 1

# Detection thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_NMS_IOU_THRESHOLD = 0.4

# Feature map strides for multi-scale detection
STRIDES = [8, 16, 32]


def load_torch_model(weights_path: str = None):
    """Load PyTorch YUNet model with optional pretrained weights."""
    import sys

    sys.path.insert(0, "models/experimental/yunet/YUNet")
    from models.experimental.yunet.YUNet.nets import nn as YUNet_nn

    model = YUNet_nn.version_n()

    if weights_path:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"].state_dict()
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)

    return model.fuse().eval()


def get_default_weights_path():
    """Get default path to pretrained weights."""
    return "models/experimental/yunet/YUNet/weights/best.pt"


def setup_yunet_reference():
    """
    Setup YUNet reference model by cloning the repository if needed.

    This is called automatically by CI tests to ensure the reference model is available.
    """
    yunet_dir = "models/experimental/yunet/YUNet"

    # Clone if not exists
    if not os.path.exists(yunet_dir):
        print("Cloning YUNet reference repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/jahongir7174/YUNet.git", yunet_dir],
            check=True,
        )

    # Create __init__.py files to make it a proper Python package
    init_files = [
        os.path.join(yunet_dir, "__init__.py"),
        os.path.join(yunet_dir, "nets", "__init__.py"),
    ]
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("")

    return yunet_dir

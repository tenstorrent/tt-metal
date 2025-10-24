# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN implementation of Depthwise Convolution for YOLO11 Pose Estimation

DWConv is used in the cv3 (confidence) head of the pose model.
"""

from models.demos.yolov11.tt.common import Yolov11Conv2D


class TtnnDWConv:
    """
    TTNN Depthwise Convolution

    A depthwise convolution where groups = in_channels, making each
    input channel have its own filter (more parameter efficient).
    """

    def __init__(self, device, parameter, conv_pt, is_detect=False):
        """
        Initialize TTNN DWConv layer

        Args:
            device: TT device
            parameter: Parameter configuration
            conv_pt: PyTorch conv layer with pretrained weights
            is_detect: Whether this is part of detection head
        """
        self.device = device
        self.is_detect = is_detect

        # DWConv uses groups=in_channels
        # This is implemented as a regular Conv2d in TTNN
        self.conv = Yolov11Conv2D(
            parameter.conv,
            conv_pt.conv,
            bn=parameter.bn if hasattr(parameter, "bn") else None,
            device=device,
            activation="silu",  # DWConv uses SiLU activation
            is_detect=is_detect,
        )

        # BatchNorm and activation are handled within Yolov11Conv2D

    def __call__(self, device, x):
        """
        Forward pass

        Args:
            device: TT device
            x: Input tensor

        Returns:
            Output tensor after depthwise conv + bn + silu
        """
        x = self.conv(x)
        return x

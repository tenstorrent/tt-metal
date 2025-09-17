# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11m.tt.common import Yolov11Conv2D


class TtnnDWConv:
    """
    TTNN implementation of Depthwise Convolution for YOLOv11 OBB.
    
    Depthwise convolution uses groups=in_channels, making it computationally efficient
    while maintaining spatial feature extraction capabilities.
    """
    
    def __init__(self, device, parameter, conv_pt, enable_act=True, is_detect=False):
        """
        Initialize TtnnDWConv module.
        
        Args:
            device: TTNN device
            parameter: Parameter configuration for convolution
            conv_pt: PyTorch convolution weights/parameters
            enable_act: Whether to enable activation (SiLU)
            is_detect: Whether this is used in detection head
        """
        self.device = device
        self.enable_act = enable_act
        self.is_detect = is_detect
        
        # For depthwise convolution, groups should equal in_channels
        # This is handled in the parameter setup, but we validate here
        assert parameter.groups == parameter.in_channels, \
            f"DWConv requires groups=in_channels, got groups={parameter.groups}, in_channels={parameter.in_channels}"
        
        # Create the depthwise convolution using Yolov11Conv2D
        # The activation will be handled separately if enable_act is True
        activation = "silu" if enable_act else ""
        
        self.conv = Yolov11Conv2D(
            parameter, 
            conv_pt,
            device=device,
            activation=activation,
            is_detect=is_detect
        )
    
    def __call__(self, device, x):
        """
        Forward pass through depthwise convolution.
        
        Args:
            device: TTNN device
            x: Input tensor
            
        Returns:
            Output tensor after depthwise convolution (and optionally activation)
        """
        return self.conv(x)

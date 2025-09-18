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
        # The parameter is a DWConv module, so we access the inner conv layer
        assert parameter.conv.groups == parameter.conv.in_channels, \
            f"DWConv requires groups=in_channels, got groups={parameter.conv.groups}, in_channels={parameter.conv.in_channels}"
        
        # Create the depthwise convolution using Yolov11Conv2D
        # The activation will be handled separately if enable_act is True
        activation = "silu" if enable_act else ""
        
        # Extract and reformat DWConv parameters to match Yolov11Conv2D expectations
        # DWConv has nested structure: conv_pt.conv.weight, conv_pt.bn.bias
        # We need to flatten it to: conv_pt_flat.weight, conv_pt_flat.bias
        class FlattenedParams:
            def __init__(self, dwconv_params):
                self.weight = dwconv_params.conv.weight
                self.bias = None

                # For DWConv, bias comes from BatchNorm if it exists
                if "bn" in dwconv_params and "bias" in dwconv_params.bn:
                    self.bias = self._reshape_bias_for_ttnn(dwconv_params.bn.bias)
                # If no bias in bn, check conv layer
                elif "bias" in dwconv_params.conv and dwconv_params.conv.bias is not None:
                    self.bias = self._reshape_bias_for_ttnn(dwconv_params.conv.bias)
            
            def _reshape_bias_for_ttnn(self, bias_tensor):
                """
                Reshape bias tensor from [channels] to [1, 1, 1, channels] format for TTNN conv2d.
                
                Args:
                    bias_tensor: Input bias tensor (TTNN or torch tensor)
                    
                Returns:
                    Reshaped bias tensor in TTNN format
                """
                # Convert to torch tensor if it's not already
                if hasattr(bias_tensor, 'to_torch'):
                    bias_torch = ttnn.to_torch(bias_tensor)
                else:
                    bias_torch = bias_tensor
                
                # Reshape from [channels] to [1, 1, 1, channels]
                bias_reshaped = bias_torch.reshape((1, 1, 1, -1))
                
                # Convert back to TTNN tensor with original properties
                return ttnn.from_torch(bias_reshaped, dtype=bias_tensor.dtype, layout=bias_tensor.layout)
            
            def __contains__(self, key):
                """Support 'key in object' syntax like TTNN parameter containers"""
                return hasattr(self, key)
        
        conv_pt_flat = FlattenedParams(conv_pt)
        
        # Pass the inner conv layer and flattened parameters to Yolov11Conv2D
        self.conv = Yolov11Conv2D(
            parameter.conv, 
            conv_pt_flat,
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

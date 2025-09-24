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
    
    def __init__(self, device, parameter, conv_pt, enable_act=True, is_detect=False, layer_name="dwconv"):
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
                print(f"🔍 FlattenedParams [{layer_name}] - Input structure keys: {list(dwconv_params.keys())}")
                
                # Access weight using dictionary syntax
                self.weight = dwconv_params["conv"]["weight"]
                print(f"🔍 FlattenedParams [{layer_name}] - Weight shape: {self.weight.shape}")
                print(f"🔍 FlattenedParams [{layer_name}] - Weight range: {ttnn.to_torch(self.weight).min():.6f} to {ttnn.to_torch(self.weight).max():.6f}")
                
                self.bias = None

                # For DWConv, bias comes from BatchNorm if it exists
                if "bn" in dwconv_params and "bias" in dwconv_params["bn"]:
                    self.bias = dwconv_params["bn"]["bias"]
                    print(f"🔍 FlattenedParams [{layer_name}] - Using BN bias, shape: {self.bias.shape}")
                    print(f"🔍 FlattenedParams [{layer_name}] - BN bias range: {ttnn.to_torch(self.bias).min():.6f} to {ttnn.to_torch(self.bias).max():.6f}")
                # If no bias in bn, check conv layer  
                elif "bias" in dwconv_params["conv"] and dwconv_params["conv"]["bias"] is not None:
                    self.bias = dwconv_params["conv"]["bias"]
                    print(f"🔍 FlattenedParams [{layer_name}] - Using Conv bias, shape: {self.bias.shape}")
                    print(f"🔍 FlattenedParams [{layer_name}] - Conv bias range: {ttnn.to_torch(self.bias).min():.6f} to {ttnn.to_torch(self.bias).max():.6f}")
                else:
                    print(f"🔍 FlattenedParams [{layer_name}] - No bias found")
            
            def __contains__(self, key):
                """Support 'key in object' syntax like TTNN parameter containers"""
                return hasattr(self, key)
            
            def __getitem__(self, key):
                """Support dictionary-style access like conv_pth["bias"]"""
                if hasattr(self, key):
                    return getattr(self, key)
                else:
                    raise KeyError(f"'{key}' not found in FlattenedParams")
        
        print(f"🔍 TtnnDWConv [{layer_name}] - Creating FlattenedParams...")
        conv_pt_flat = FlattenedParams(conv_pt)
        
        print(f"🔍 TtnnDWConv [{layer_name}] - Final flattened bias: {'None' if conv_pt_flat.bias is None else f'shape={conv_pt_flat.bias.shape}, range={ttnn.to_torch(conv_pt_flat.bias).min():.6f} to {ttnn.to_torch(conv_pt_flat.bias).max():.6f}'}")
        
        # Pass the inner conv layer and flattened parameters to Yolov11Conv2D
        print(f"🔍 TtnnDWConv [{layer_name}] - Passing to Yolov11Conv2D...")
        self.conv = Yolov11Conv2D(
            parameter.conv, 
            conv_pt_flat,
            device=device,
            activation=activation,
            is_detect=is_detect,
            layer_name=f"{layer_name}_yolov11conv2d"
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

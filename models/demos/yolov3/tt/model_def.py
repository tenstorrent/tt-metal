# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLOv3 (416x416) implementation using TTNN APIs.

YOLOv3 architecture:
- Darknet-53 backbone with residual connections
- Multi-scale detection at 13x13, 26x26, 52x52
- 3 YOLO detection layers for different object sizes
"""

import torch
import ttnn
from typing import Dict, List, Optional, Tuple


def preprocess_conv(weights: torch.Tensor) -> torch.Tensor:
    """
    Preprocess convolution weights for TTNN.
    
    TTNN conv2d expects 4D weight tensors with shape
    (out_channels, in_channels, kernel_height, kernel_width).
    """
    # Preserve the original 4D layout
    return weights


def preprocess_batchnorm(
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse BatchNorm into scale and shift."""
    scale = weight / torch.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return scale, shift


class TtConvBnLeaky:
    """Convolution + BatchNorm + LeakyReLU block for TTNN."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Convert parameters to TTNN tensors
        self.parameters: Dict = {}
        if parameters is not None:
            for name, value in parameters.items():
                if isinstance(value, torch.Tensor):
                    self.parameters[name] = ttnn.from_torch(
                        value.to(dtype=torch.float32),
                        device=device,
                        layout=ttnn.TILE_LAYOUT
                    )
                else:
                    self.parameters[name] = value
        
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward through Conv + BN + LeakyReLU."""
        if not self.parameters:
            raise ValueError("Parameters must be provided for inference")
        
        # Convolution with proper API
        batch_size = x.shape[0]
        input_height = x.shape[2] if len(x.shape) > 2 else 1
        input_width = x.shape[3] if len(x.shape) > 3 else 1
        
        x, _, _ = ttnn.conv2d(
            x,
            self.parameters["conv_weight"],
            bias=self.parameters.get("conv_bias"),
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            batch_size=batch_size,
            input_height=input_height,
            input_width=input_width,
            kernel_size=(self.kernel_size, self.kernel_size),
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            device=self.device
        )
        
        # Fused BatchNorm (scale + shift)
        x = ttnn.mul(x, self.parameters["bn_scale"])
        x = ttnn.add(x, self.parameters["bn_shift"])
        
        # LeakyReLU (negative slope = 0.1)
        x = ttnn.leaky_relu(x, negative_slope=0.1)
        
        return x


class TtDarknetResidual:
    """Darknet residual block with skip connection."""
    
    def __init__(
        self,
        device,
        channels: int,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        
        # 1x1 conv to reduce channels
        self.conv1 = TtConvBnLeaky(
            device, channels, channels // 2, 1, 1, 0,
            parameters=parameters.get("conv1") if parameters else None
        )
        # 3x3 conv to restore channels
        self.conv2 = TtConvBnLeaky(
            device, channels // 2, channels, 3, 1, 1,
            parameters=parameters.get("conv2") if parameters else None
        )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = ttnn.add(x, residual)
        return x


class TtDarknet53:
    """Darknet-53 backbone for YOLOv3."""
    
    def __init__(self, device, parameters: Optional[Dict] = None):
        self.device = device
        
        # Initial conv layer
        self.conv0 = TtConvBnLeaky(
            device, 3, 32, 3, 1, 1,
            parameters=parameters.get("conv0") if parameters else None
        )
        
        # Downsampling stages with residual blocks
        # Stage 1: 416 -> 208, 1 residual block
        self.down1 = TtConvBnLeaky(
            device, 32, 64, 3, 2, 1,
            parameters=parameters.get("down1") if parameters else None
        )
        self.res1 = self._make_residuals(device, 64, 1, parameters, "res1")
        
        # Stage 2: 208 -> 104, 2 residual blocks
        self.down2 = TtConvBnLeaky(
            device, 64, 128, 3, 2, 1,
            parameters=parameters.get("down2") if parameters else None
        )
        self.res2 = self._make_residuals(device, 128, 2, parameters, "res2")
        
        # Stage 3: 104 -> 52, 8 residual blocks (output for scale 3)
        self.down3 = TtConvBnLeaky(
            device, 128, 256, 3, 2, 1,
            parameters=parameters.get("down3") if parameters else None
        )
        self.res3 = self._make_residuals(device, 256, 8, parameters, "res3")
        
        # Stage 4: 52 -> 26, 8 residual blocks (output for scale 2)
        self.down4 = TtConvBnLeaky(
            device, 256, 512, 3, 2, 1,
            parameters=parameters.get("down4") if parameters else None
        )
        self.res4 = self._make_residuals(device, 512, 8, parameters, "res4")
        
        # Stage 5: 26 -> 13, 4 residual blocks (output for scale 1)
        self.down5 = TtConvBnLeaky(
            device, 512, 1024, 3, 2, 1,
            parameters=parameters.get("down5") if parameters else None
        )
        self.res5 = self._make_residuals(device, 1024, 4, parameters, "res5")
    
    def _make_residuals(
        self,
        device,
        channels: int,
        num_blocks: int,
        parameters: Optional[Dict],
        name: str
    ) -> List:
        blocks = []
        for i in range(num_blocks):
            block_params = parameters.get(f"{name}_{i}") if parameters else None
            blocks.append(TtDarknetResidual(device, channels, block_params))
        return blocks
    
    def __call__(self, x: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]:
        """Forward through Darknet-53, returns 3 feature maps."""
        x = self.conv0(x)
        
        # Stage 1
        x = self.down1(x)
        for block in self.res1:
            x = block(x)
        
        # Stage 2
        x = self.down2(x)
        for block in self.res2:
            x = block(x)
        
        # Stage 3 - output for 52x52 scale
        x = self.down3(x)
        for block in self.res3:
            x = block(x)
        out_52 = x  # 256 channels
        
        # Stage 4 - output for 26x26 scale
        x = self.down4(x)
        for block in self.res4:
            x = block(x)
        out_26 = x  # 512 channels
        
        # Stage 5 - output for 13x13 scale
        x = self.down5(x)
        for block in self.res5:
            x = block(x)
        out_13 = x  # 1024 channels
        
        return out_13, out_26, out_52


class TtYOLOHead:
    """YOLO detection head for a single scale."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        num_classes: int = 80,
        num_anchors: int = 3,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.in_channels = in_channels
        
        # Output channels: (x, y, w, h, obj, classes) * num_anchors
        self.out_channels = num_anchors * (5 + num_classes)
        
        # 5 conv layers before detection
        self.conv1 = TtConvBnLeaky(
            device, in_channels, in_channels // 2, 1, 1, 0,
            parameters=parameters.get("conv1") if parameters else None
        )
        self.conv2 = TtConvBnLeaky(
            device, in_channels // 2, in_channels, 3, 1, 1,
            parameters=parameters.get("conv2") if parameters else None
        )
        self.conv3 = TtConvBnLeaky(
            device, in_channels, in_channels // 2, 1, 1, 0,
            parameters=parameters.get("conv3") if parameters else None
        )
        self.conv4 = TtConvBnLeaky(
            device, in_channels // 2, in_channels, 3, 1, 1,
            parameters=parameters.get("conv4") if parameters else None
        )
        self.conv5 = TtConvBnLeaky(
            device, in_channels, in_channels // 2, 1, 1, 0,
            parameters=parameters.get("conv5") if parameters else None
        )
        
        # Final detection convs
        self.conv6 = TtConvBnLeaky(
            device, in_channels // 2, in_channels, 3, 1, 1,
            parameters=parameters.get("conv6") if parameters else None
        )
        
        # Output layer (1x1 conv for predictions, no activation)
        self.output_conv = TtConvBnLeaky(
            device, in_channels, self.out_channels, 1, 1, 0,
            parameters=parameters.get("output_conv") if parameters else None
        )
    
    def __call__(self, x: ttnn.Tensor) -> Tuple[ttnn.Tensor, ttnn.Tensor]:
        """Forward through YOLO head."""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        route = self.conv5(x)  # Used for upsampling path
        
        x = self.conv6(route)
        detection = self.output_conv(x)  # Final detection output
        
        return detection, route


class TtYoloV3:
    """Complete YOLOv3 (416x416) model."""
    
    def __init__(
        self,
        device,
        num_classes: int = 80,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.num_classes = num_classes
        
        # Default to empty dict if None
        if parameters is None:
            parameters = {}
        
        # Backbone
        self.backbone = TtDarknet53(
            device,
            parameters=parameters.get("backbone")
        )
        
        # Detection heads for 3 scales
        # Note: Until upsampling is fully implemented, use backbone feature channel sizes
        self.head_13 = TtYOLOHead(
            device, 1024, num_classes,
            parameters=parameters.get("head_13")
        )
        # Use 512 directly (backbone output) until upsample+concat implemented
        self.head_26 = TtYOLOHead(
            device, 512, num_classes,
            parameters=parameters.get("head_26")
        )
        # Use 256 directly (backbone output) until upsample+concat implemented
        self.head_52 = TtYOLOHead(
            device, 256, num_classes,
            parameters=parameters.get("head_52")
        )
        
        # Lateral convs for FPN
        self.lateral_13 = TtConvBnLeaky(
            device, 512, 256, 1, 1, 0,
            parameters=parameters.get("lateral_13")
        )
        self.lateral_26 = TtConvBnLeaky(
            device, 256, 128, 1, 1, 0,
            parameters=parameters.get("lateral_26")
        )
    
    def __call__(self, x: ttnn.Tensor) -> List[ttnn.Tensor]:
        """Forward through YOLOv3."""
        # Extract backbone features
        out_13, out_26, out_52 = self.backbone(x)
        
        # Scale 1: 13x13 (large objects)
        det_13, route_13 = self.head_13(out_13)
        
        # Upsample and concatenate for scale 2
        route_13_up = self.lateral_13(route_13)
        route_13_up = ttnn.upsample(route_13_up, scale_factor=2.0)
        fused_26 = ttnn.concat([route_13_up, out_26], dim=1)
        
        # Scale 2: 26x26 (medium objects)
        det_26, route_26 = self.head_26(fused_26)
        
        # Upsample and concatenate for scale 3
        route_26_up = self.lateral_26(route_26)
        route_26_up = ttnn.upsample(route_26_up, scale_factor=2.0)
        fused_52 = ttnn.concat([route_26_up, out_52], dim=1)
        
        # Scale 3: 52x52 (small objects)
        det_52, _ = self.head_52(fused_52)
        
        return [det_13, det_26, det_52]


def custom_preprocessor(
    model,
    device: Optional[ttnn.Device] = None,
    name: str = ""
) -> Dict:
    """
    Preprocess PyTorch YOLOv3 weights for TTNN.
    
    If a TTNN device is provided, all returned tensors are converted
    to TTNN device tensors. Otherwise, returns torch tensors.
    """
    parameters: Dict = {}
    
    for param_name, param in model.named_parameters():
        full_name = f"{name}.{param_name}" if name else param_name
        
        # Preprocess convolution weights (keep 4D shape)
        if "weight" in param_name and len(param.shape) == 4:
            torch_tensor = preprocess_conv(param.data)
        else:
            torch_tensor = param.data
        
        if device is not None:
            parameters[full_name] = ttnn.from_torch(
                torch_tensor.to(dtype=torch.float32),
                device=device,
                layout=ttnn.TILE_LAYOUT
            )
        else:
            parameters[full_name] = torch_tensor
    
    # Fuse batch norm
    for module_name, module in model.named_modules():
        if hasattr(module, "bn") and hasattr(module.bn, "running_mean"):
            bn = module.bn
            scale, shift = preprocess_batchnorm(
                bn.weight, bn.bias, bn.running_mean, bn.running_var
            )
            if device is not None:
                parameters[f"{module_name}.bn_scale"] = ttnn.from_torch(
                    scale.to(dtype=torch.float32),
                    device=device,
                    layout=ttnn.TILE_LAYOUT
                )
                parameters[f"{module_name}.bn_shift"] = ttnn.from_torch(
                    shift.to(dtype=torch.float32),
                    device=device,
                    layout=ttnn.TILE_LAYOUT
                )
            else:
                parameters[f"{module_name}.bn_scale"] = scale
                parameters[f"{module_name}.bn_shift"] = shift
    
    return parameters

# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLOP-s (Panoptic Driving Perception) implementation using TTNN APIs.

YOLOP-s is a multi-task model for autonomous driving:
- Shared encoder (CSPDarknet backbone + SPP + FPN)
- Traffic object detection head
- Drivable area segmentation head
- Lane line detection head
"""

import torch
import ttnn
from typing import Dict, List, Optional, Tuple


def preprocess_conv(weights: torch.Tensor) -> torch.Tensor:
    """Preserve 4D weight shape for TTNN conv2d."""
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


class TtConvBnSiLU:
    """Convolution + BatchNorm + SiLU activation block."""
    
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
        if not self.parameters:
            raise ValueError("Parameters required for inference")
        
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
        
        x = ttnn.mul(x, self.parameters["bn_scale"])
        x = ttnn.add(x, self.parameters["bn_shift"])
        x = ttnn.silu(x)
        
        return x


class TtCSPBlock:
    """Cross Stage Partial block for feature extraction."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        hidden_ch = out_channels // 2
        
        self.conv1 = TtConvBnSiLU(
            device, in_channels, hidden_ch, 1, 1, 0,
            parameters=parameters.get("conv1") if parameters else None
        )
        self.conv2 = TtConvBnSiLU(
            device, in_channels, hidden_ch, 1, 1, 0,
            parameters=parameters.get("conv2") if parameters else None
        )
        self.conv3 = TtConvBnSiLU(
            device, 2 * hidden_ch, out_channels, 1, 1, 0,
            parameters=parameters.get("conv3") if parameters else None
        )
        
        self.bottlenecks = []
        for i in range(num_blocks):
            block_params = parameters.get(f"bottleneck_{i}") if parameters else None
            self.bottlenecks.append(
                TtConvBnSiLU(device, hidden_ch, hidden_ch, 3, 1, 1, parameters=block_params)
            )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
        
        x = ttnn.concat([x1, x2], dim=1)
        x = self.conv3(x)
        return x


class TtSPP:
    """Spatial Pyramid Pooling module."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        pool_sizes: Tuple[int, ...] = (5, 9, 13),
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.pool_sizes = pool_sizes
        
        hidden_ch = in_channels // 2
        self.conv1 = TtConvBnSiLU(
            device, in_channels, hidden_ch, 1, 1, 0,
            parameters=parameters.get("conv1") if parameters else None
        )
        # After concatenating: hidden_ch + hidden_ch * len(pool_sizes) = hidden_ch * 4 (for 3 pools)
        self.conv2 = TtConvBnSiLU(
            device, hidden_ch * (1 + len(pool_sizes)), out_channels, 1, 1, 0,
            parameters=parameters.get("conv2") if parameters else None
        )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv1(x)
        
        # Multi-scale pooling
        pooled = [x]
        for pool_size in self.pool_sizes:
            p = ttnn.max_pool2d(x, kernel_size=pool_size, stride=1, padding=pool_size // 2)
            pooled.append(p)
        
        x = ttnn.concat(pooled, dim=1)
        x = self.conv2(x)
        return x


class TtYOLOPEncoder:
    """Shared encoder for YOLOP-s (CSPDarknet + SPP + FPN)."""
    
    def __init__(self, device, parameters: Optional[Dict] = None):
        self.device = device
        
        # Stem
        self.stem = TtConvBnSiLU(
            device, 3, 32, 3, 1, 1,
            parameters=parameters.get("stem") if parameters else None
        )
        
        # Downsampling stages
        self.down1 = TtConvBnSiLU(device, 32, 64, 3, 2, 1,
                                   parameters=parameters.get("down1") if parameters else None)
        self.csp1 = TtCSPBlock(device, 64, 64, 1,
                                parameters=parameters.get("csp1") if parameters else None)
        
        self.down2 = TtConvBnSiLU(device, 64, 128, 3, 2, 1,
                                   parameters=parameters.get("down2") if parameters else None)
        self.csp2 = TtCSPBlock(device, 128, 128, 2,
                                parameters=parameters.get("csp2") if parameters else None)
        
        self.down3 = TtConvBnSiLU(device, 128, 256, 3, 2, 1,
                                   parameters=parameters.get("down3") if parameters else None)
        self.csp3 = TtCSPBlock(device, 256, 256, 8,
                                parameters=parameters.get("csp3") if parameters else None)
        
        self.down4 = TtConvBnSiLU(device, 256, 512, 3, 2, 1,
                                   parameters=parameters.get("down4") if parameters else None)
        self.csp4 = TtCSPBlock(device, 512, 512, 8,
                                parameters=parameters.get("csp4") if parameters else None)
        
        self.down5 = TtConvBnSiLU(device, 512, 1024, 3, 2, 1,
                                   parameters=parameters.get("down5") if parameters else None)
        self.spp = TtSPP(device, 1024, 1024,
                          parameters=parameters.get("spp") if parameters else None)
        self.csp5 = TtCSPBlock(device, 1024, 1024, 4,
                                parameters=parameters.get("csp5") if parameters else None)
    
    def __call__(self, x: ttnn.Tensor) -> Dict[str, ttnn.Tensor]:
        x = self.stem(x)
        
        x = self.down1(x)
        x = self.csp1(x)
        
        x = self.down2(x)
        x = self.csp2(x)
        c3 = x  # 256 channels, 1/8 scale
        
        x = self.down3(x)
        x = self.csp3(x)
        c4 = x  # 256 channels, 1/16 scale
        
        x = self.down4(x)
        x = self.csp4(x)
        c5 = x  # 512 channels, 1/32 scale
        
        x = self.down5(x)
        x = self.spp(x)
        x = self.csp5(x)
        
        return {"c3": c3, "c4": c4, "c5": c5, "spp": x}


class TtDetectionHead:
    """Object detection head for vehicles/pedestrians."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        num_classes: int = 80,
        num_anchors: int = 3,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.out_channels = num_anchors * (5 + num_classes)
        
        self.conv1 = TtConvBnSiLU(device, in_channels, in_channels // 2, 1, 1, 0,
                                   parameters=parameters.get("conv1") if parameters else None)
        self.conv2 = TtConvBnSiLU(device, in_channels // 2, in_channels, 3, 1, 1,
                                   parameters=parameters.get("conv2") if parameters else None)
        self.conv3 = TtConvBnSiLU(device, in_channels, in_channels // 2, 1, 1, 0,
                                   parameters=parameters.get("conv3") if parameters else None)
        self.output = TtConvBnSiLU(device, in_channels // 2, self.out_channels, 1, 1, 0,
                                    parameters=parameters.get("output") if parameters else None)
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.output(x)
        return x


class TtSegmentationHead:
    """Segmentation head for drivable area or lane detection."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int = 2,  # Binary segmentation
        parameters: Optional[Dict] = None
    ):
        self.device = device
        
        self.conv1 = TtConvBnSiLU(device, in_channels, 256, 3, 1, 1,
                                   parameters=parameters.get("conv1") if parameters else None)
        self.conv2 = TtConvBnSiLU(device, 256, 128, 3, 1, 1,
                                   parameters=parameters.get("conv2") if parameters else None)
        self.conv3 = TtConvBnSiLU(device, 128, 64, 3, 1, 1,
                                   parameters=parameters.get("conv3") if parameters else None)
        self.output = TtConvBnSiLU(device, 64, out_channels, 1, 1, 0,
                                    parameters=parameters.get("output") if parameters else None)
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        x = self.conv1(x)
        x = ttnn.upsample(x, scale_factor=2.0)
        x = self.conv2(x)
        x = ttnn.upsample(x, scale_factor=2.0)
        x = self.conv3(x)
        x = ttnn.upsample(x, scale_factor=2.0)
        x = self.output(x)
        return x


class TtYOLOP:
    """Complete YOLOP-s model for panoptic driving perception."""
    
    def __init__(
        self,
        device,
        num_classes: int = 80,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.num_classes = num_classes
        
        if parameters is None:
            parameters = {}
        
        # Shared encoder
        self.encoder = TtYOLOPEncoder(
            device, parameters=parameters.get("encoder")
        )
        
        # Detection head (vehicles, pedestrians)
        self.det_head = TtDetectionHead(
            device, 1024, num_classes,
            parameters=parameters.get("det_head")
        )
        
        # Drivable area segmentation head
        self.da_seg_head = TtSegmentationHead(
            device, 256, out_channels=2,
            parameters=parameters.get("da_seg_head")
        )
        
        # Lane line segmentation head
        self.ll_seg_head = TtSegmentationHead(
            device, 256, out_channels=2,
            parameters=parameters.get("ll_seg_head")
        )
    
    def __call__(self, x: ttnn.Tensor) -> Dict[str, ttnn.Tensor]:
        """Forward pass through YOLOP-s."""
        # Encode
        features = self.encoder(x)
        
        # Detection (on SPP output)
        det_output = self.det_head(features["spp"])
        
        # Segmentation (on C3 features for higher resolution)
        da_seg = self.da_seg_head(features["c3"])
        ll_seg = self.ll_seg_head(features["c3"])
        
        return {
            "detection": det_output,
            "drivable_area": da_seg,
            "lane_line": ll_seg
        }


def custom_preprocessor(
    model,
    device: Optional[ttnn.Device] = None,
    name: str = ""
) -> Dict:
    """Preprocess PyTorch YOLOP weights for TTNN."""
    parameters: Dict = {}
    
    for param_name, param in model.named_parameters():
        full_name = f"{name}.{param_name}" if name else param_name
        
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
    
    for module_name, module in model.named_modules():
        if hasattr(module, "bn") and hasattr(module.bn, "running_mean"):
            bn = module.bn
            scale, shift = preprocess_batchnorm(
                bn.weight, bn.bias, bn.running_mean, bn.running_var
            )
            if device is not None:
                parameters[f"{module_name}.bn_scale"] = ttnn.from_torch(
                    scale.to(dtype=torch.float32), device=device, layout=ttnn.TILE_LAYOUT
                )
                parameters[f"{module_name}.bn_shift"] = ttnn.from_torch(
                    shift.to(dtype=torch.float32), device=device, layout=ttnn.TILE_LAYOUT
                )
            else:
                parameters[f"{module_name}.bn_scale"] = scale
                parameters[f"{module_name}.bn_shift"] = shift
    
    return parameters

# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
YOLOX model implementation using TTNN APIs.

YOLOX is an anchor-free object detector with:
- CSPDarknet backbone for feature extraction
- FPN+PAN neck for multi-scale feature fusion
- Decoupled head for classification and regression
"""

import torch
import ttnn
from typing import Dict, List, Optional, Tuple

# Constants for YOLOX-S (small) variant
INPUT_SIZE = 640
NUM_CLASSES = 80  # COCO classes
DEPTH_MULTIPLIER = 0.33
WIDTH_MULTIPLIER = 0.50


def preprocess_conv(weights: torch.Tensor, bias: Optional[torch.Tensor] = None) -> Tuple:
    """Preprocess convolution weights for TTNN."""
    # Reshape to [out_channels, in_channels * kernel_h * kernel_w]
    out_c, in_c, kh, kw = weights.shape
    weights_flat = weights.view(out_c, -1)
    
    if bias is not None:
        return weights_flat, bias
    return weights_flat, None


def preprocess_batchnorm(
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps: float = 1e-5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fuse BatchNorm parameters into scale and shift for TTNN."""
    scale = weight / torch.sqrt(running_var + eps)
    shift = bias - running_mean * scale
    return scale, shift


class TtConvBnSiLU:
    """Convolution + BatchNorm + SiLU activation block for TTNN."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.parameters = parameters
        
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Forward pass through Conv + BN + SiLU."""
        # Convolution
        x = ttnn.conv2d(
            x,
            self.parameters["conv_weight"],
            bias=self.parameters.get("conv_bias"),
            stride=(self.stride, self.stride),
            padding=(self.padding, self.padding),
            groups=self.groups
        )
        
        # Fused BatchNorm (scale + shift)
        x = ttnn.mul(x, self.parameters["bn_scale"])
        x = ttnn.add(x, self.parameters["bn_shift"])
        
        # SiLU activation (x * sigmoid(x))
        x = ttnn.silu(x)
        
        return x


class TtDarknetBottleneck:
    """Darknet bottleneck block with residual connection."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.shortcut = shortcut and in_channels == out_channels
        hidden_channels = int(out_channels * expansion)
        
        self.conv1 = TtConvBnSiLU(
            device, in_channels, hidden_channels, 1, 1, 0,
            parameters=parameters.get("conv1") if parameters else None
        )
        self.conv2 = TtConvBnSiLU(
            device, hidden_channels, out_channels, 3, 1, 1,
            parameters=parameters.get("conv2") if parameters else None
        )
        
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        
        if self.shortcut:
            x = ttnn.add(x, residual)
        
        return x


class TtCSPLayer:
    """Cross Stage Partial layer for CSPDarknet."""
    
    def __init__(
        self,
        device,
        in_channels: int,
        out_channels: int,
        n_bottlenecks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        hidden_channels = int(out_channels * expansion)
        
        # Main branch convolutions
        self.conv1 = TtConvBnSiLU(
            device, in_channels, hidden_channels, 1, 1, 0,
            parameters=parameters.get("conv1") if parameters else None
        )
        self.conv2 = TtConvBnSiLU(
            device, in_channels, hidden_channels, 1, 1, 0,
            parameters=parameters.get("conv2") if parameters else None
        )
        self.conv3 = TtConvBnSiLU(
            device, 2 * hidden_channels, out_channels, 1, 1, 0,
            parameters=parameters.get("conv3") if parameters else None
        )
        
        # Bottleneck blocks
        self.bottlenecks = []
        for i in range(n_bottlenecks):
            block_params = parameters.get(f"bottleneck_{i}") if parameters else None
            self.bottlenecks.append(
                TtDarknetBottleneck(
                    device, hidden_channels, hidden_channels,
                    shortcut=shortcut, expansion=1.0,
                    parameters=block_params
                )
            )
    
    def __call__(self, x: ttnn.Tensor) -> ttnn.Tensor:
        # Split path
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        # Bottleneck path
        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
        
        # Concatenate and fuse
        x = ttnn.concat([x1, x2], dim=1)
        x = self.conv3(x)
        
        return x


class TtCSPDarknet:
    """CSPDarknet backbone for YOLOX."""
    
    def __init__(
        self,
        device,
        depth_multiplier: float = 1.0,
        width_multiplier: float = 1.0,
        out_features: Tuple[str, ...] = ("dark3", "dark4", "dark5"),
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.out_features = out_features
        
        # Base channel configuration
        base_channels = int(64 * width_multiplier)
        base_depth = max(round(3 * depth_multiplier), 1)
        
        # Stem: 6x6 conv
        self.stem = TtConvBnSiLU(
            device, 3, base_channels, 6, 2, 2,
            parameters=parameters.get("stem") if parameters else None
        )
        
        # Stage parameters: (out_channels, num_blocks)
        self.dark2 = self._make_stage(
            device, base_channels, base_channels * 2, base_depth,
            parameters=parameters.get("dark2") if parameters else None
        )
        self.dark3 = self._make_stage(
            device, base_channels * 2, base_channels * 4, base_depth * 3,
            parameters=parameters.get("dark3") if parameters else None
        )
        self.dark4 = self._make_stage(
            device, base_channels * 4, base_channels * 8, base_depth * 3,
            parameters=parameters.get("dark4") if parameters else None
        )
        self.dark5 = self._make_stage(
            device, base_channels * 8, base_channels * 16, base_depth,
            parameters=parameters.get("dark5") if parameters else None
        )
    
    def _make_stage(
        self,
        device,
        in_channels: int,
        out_channels: int,
        n_blocks: int,
        parameters: Optional[Dict] = None
    ) -> Tuple:
        """Create a CSP stage with downsampling conv + CSP layer."""
        downsample = TtConvBnSiLU(
            device, in_channels, out_channels, 3, 2, 1,
            parameters=parameters.get("downsample") if parameters else None
        )
        csp = TtCSPLayer(
            device, out_channels, out_channels, n_blocks,
            parameters=parameters.get("csp") if parameters else None
        )
        return (downsample, csp)
    
    def __call__(self, x: ttnn.Tensor) -> Dict[str, ttnn.Tensor]:
        outputs = {}
        
        x = self.stem(x)
        
        x = self.dark2[0](x)  # Downsample
        x = self.dark2[1](x)  # CSP
        
        x = self.dark3[0](x)
        x = self.dark3[1](x)
        if "dark3" in self.out_features:
            outputs["dark3"] = x
        
        x = self.dark4[0](x)
        x = self.dark4[1](x)
        if "dark4" in self.out_features:
            outputs["dark4"] = x
        
        x = self.dark5[0](x)
        x = self.dark5[1](x)
        if "dark5" in self.out_features:
            outputs["dark5"] = x
        
        return outputs


class TtYOLOXHead:
    """Decoupled head for YOLOX predictions."""
    
    def __init__(
        self,
        device,
        num_classes: int = 80,
        width_multiplier: float = 1.0,
        in_channels: Tuple[int, ...] = (256, 512, 1024),
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.num_classes = num_classes
        self.n_anchors = 1  # Anchor-free
        
        # Per-scale heads
        self.cls_convs = []
        self.reg_convs = []
        self.cls_preds = []
        self.reg_preds = []
        self.obj_preds = []
        
        for i, ch in enumerate(in_channels):
            hidden_ch = int(256 * width_multiplier)
            scale_params = parameters.get(f"scale_{i}") if parameters else None
            
            # Classification branch
            self.cls_convs.append([
                TtConvBnSiLU(device, ch, hidden_ch, 3, 1, 1, 
                            parameters=scale_params.get("cls_conv_0") if scale_params else None),
                TtConvBnSiLU(device, hidden_ch, hidden_ch, 3, 1, 1,
                            parameters=scale_params.get("cls_conv_1") if scale_params else None),
            ])
            
            # Regression branch
            self.reg_convs.append([
                TtConvBnSiLU(device, ch, hidden_ch, 3, 1, 1,
                            parameters=scale_params.get("reg_conv_0") if scale_params else None),
                TtConvBnSiLU(device, hidden_ch, hidden_ch, 3, 1, 1,
                            parameters=scale_params.get("reg_conv_1") if scale_params else None),
            ])
            
            # Prediction layers (1x1 conv, no activation)
            # These would be simple linear projections in TTNN
    
    def __call__(
        self,
        features: Dict[str, ttnn.Tensor]
    ) -> List[Tuple[ttnn.Tensor, ttnn.Tensor, ttnn.Tensor]]:
        """Forward through decoupled heads for each scale."""
        outputs = []
        
        feature_list = [features["dark3"], features["dark4"], features["dark5"]]
        
        for i, feat in enumerate(feature_list):
            # Classification branch
            cls_feat = feat
            for conv in self.cls_convs[i]:
                cls_feat = conv(cls_feat)
            
            # Regression branch
            reg_feat = feat
            for conv in self.reg_convs[i]:
                reg_feat = conv(reg_feat)
            
            # Predictions would be made here with final 1x1 convs
            outputs.append((cls_feat, reg_feat, feat))
        
        return outputs


class TtYOLOX:
    """Complete YOLOX model for object detection."""
    
    def __init__(
        self,
        device,
        num_classes: int = 80,
        depth_multiplier: float = 0.33,
        width_multiplier: float = 0.50,
        parameters: Optional[Dict] = None
    ):
        self.device = device
        self.num_classes = num_classes
        
        in_channels = [
            int(256 * width_multiplier),
            int(512 * width_multiplier),
            int(1024 * width_multiplier)
        ]
        
        self.backbone = TtCSPDarknet(
            device,
            depth_multiplier=depth_multiplier,
            width_multiplier=width_multiplier,
            parameters=parameters.get("backbone") if parameters else None
        )
        
        self.head = TtYOLOXHead(
            device,
            num_classes=num_classes,
            width_multiplier=width_multiplier,
            in_channels=tuple(in_channels),
            parameters=parameters.get("head") if parameters else None
        )
    
    def __call__(self, x: ttnn.Tensor) -> List:
        """Forward pass through YOLOX."""
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Get predictions from decoupled head
        outputs = self.head(features)
        
        return outputs


def custom_preprocessor(model, name: str = "") -> Dict:
    """
    Preprocess PyTorch YOLOX weights for TTNN.
    
    Args:
        model: PyTorch YOLOX model
        name: Parameter name prefix
        
    Returns:
        Dictionary of preprocessed parameters
    """
    parameters = {}
    
    for param_name, param in model.named_parameters():
        full_name = f"{name}.{param_name}" if name else param_name
        
        if "weight" in param_name and len(param.shape) == 4:
            # Conv weight preprocessing
            parameters[full_name] = preprocess_conv(param.data)[0]
        elif "running_mean" in param_name or "running_var" in param_name:
            # Skip, handled in batch norm fusion
            continue
        else:
            parameters[full_name] = param.data
    
    # Fuse batch norm if present
    for name, module in model.named_modules():
        if hasattr(module, 'bn') and hasattr(module.bn, 'running_mean'):
            bn = module.bn
            scale, shift = preprocess_batchnorm(
                bn.weight, bn.bias, bn.running_mean, bn.running_var
            )
            parameters[f"{name}.bn_scale"] = scale
            parameters[f"{name}.bn_shift"] = shift
    
    return parameters

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import torch

# Monkey patch ultralytics with missing classes
import ultralytics.nn.modules.block as ultralytics_block
import ultralytics.nn.modules.conv as ultralytics_conv
from torch.nn import AdaptiveAvgPool2d, BatchNorm2d, Conv2d, Linear, MaxPool2d, ReLU
from torch.nn.modules.activation import SiLU
from torch.nn.modules.container import ModuleList, Sequential
from ultralytics import YOLO
from ultralytics.nn.modules.conv import Conv as UltralyticsConv
from ultralytics.nn.tasks import DetectionModel

# Import only the missing custom classes
from models.demos.yolov13.reference.yolov13 import (
    C3AH,
    AdaHGComputation,
    AdaHGConv,
    AdaHyperedgeGen,
    DownsampleConv,
    DSBottleneck,
    DSC3k,
    DSC3k2,
    DsConv,
    FullPAD_Tunnel,
    FuseModule,
    HyperACE,
    Upsample,
    YoloV13,
)


# Create wrapper classes with correct module paths
class UltralyticsDsConv(DsConv):
    __module__ = "ultralytics.nn.modules.conv"
    __qualname__ = "DsConv"


class UltralyticsDSC3k2(DSC3k2):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "DSC3k2"


class UltralyticsDSBottleneck(DSBottleneck):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "DSBottleneck"


class UltralyticsDSC3k(DSC3k):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "DSC3k"


class UltralyticsUpsample(Upsample):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "Upsample"


class UltralyticsDownsampleConv(DownsampleConv):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "DownsampleConv"


class UltralyticsC3AH(C3AH):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "C3AH"


class UltralyticsAdaHyperedgeGen(AdaHyperedgeGen):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "AdaHyperedgeGen"


class UltralyticsAdaHGConv(AdaHGConv):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "AdaHGConv"


class UltralyticsAdaHGComputation(AdaHGComputation):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "AdaHGComputation"


class UltralyticsFuseModule(FuseModule):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "FuseModule"


class UltralyticsHyperACE(HyperACE):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "HyperACE"


class UltralyticsFullPAD_Tunnel(FullPAD_Tunnel):
    __module__ = "ultralytics.nn.modules.block"
    __qualname__ = "FullPAD_Tunnel"


# Patch the modules
ultralytics_conv.DsConv = UltralyticsDsConv
ultralytics_conv.DSConv = UltralyticsDsConv  # Handle case sensitivity
ultralytics_block.DSC3k2 = UltralyticsDSC3k2
ultralytics_block.DSBottleneck = UltralyticsDSBottleneck
ultralytics_block.DSC3k = UltralyticsDSC3k
ultralytics_block.Upsample = UltralyticsUpsample
ultralytics_block.DownsampleConv = UltralyticsDownsampleConv
ultralytics_block.C3AH = UltralyticsC3AH
ultralytics_block.AdaHyperedgeGen = UltralyticsAdaHyperedgeGen
ultralytics_block.AdaHGConv = UltralyticsAdaHGConv
ultralytics_block.AdaHGComputation = UltralyticsAdaHGComputation
ultralytics_block.FuseModule = UltralyticsFuseModule
ultralytics_block.HyperACE = UltralyticsHyperACE
ultralytics_block.FullPAD_Tunnel = UltralyticsFullPAD_Tunnel

YOLOV13_L1_SMALL_SIZE = 24576


def load_torch_model(model_location_generator=None):
    file_path = "https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13x.pt"

    # Load model with safe globals for weights_only=True
    with torch.serialization.safe_globals(
        [
            DetectionModel,
            Sequential,
            ModuleList,
            Conv2d,
            BatchNorm2d,
            ReLU,
            MaxPool2d,
            AdaptiveAvgPool2d,
            Linear,
            SiLU,
            UltralyticsConv,
            UltralyticsDsConv,
            UltralyticsDSC3k2,
            UltralyticsDSBottleneck,
            UltralyticsDSC3k,
            UltralyticsUpsample,
            UltralyticsDownsampleConv,
            UltralyticsC3AH,
            UltralyticsAdaHyperedgeGen,
            UltralyticsAdaHGConv,
            UltralyticsAdaHGComputation,
            UltralyticsFuseModule,
            UltralyticsHyperACE,
            UltralyticsFullPAD_Tunnel,
        ]
    ):
        torch_model = YoloV13()
        model = YOLO(file_path)
        # Get the actual PyTorch model from YOLO
        state_dict = {k.replace("model.", "", 1): v for k, v in model.state_dict().items()}
        torch_model.load_state_dict(state_dict)
        torch_model.eval()

        return torch_model

    return None

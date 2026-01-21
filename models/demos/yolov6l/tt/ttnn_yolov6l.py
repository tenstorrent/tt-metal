# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov6l.tt.ttnn_cspbep_backbone import TtCSPBepBackbone
from models.demos.yolov6l.tt.ttnn_csprep_bifpanneck import TtCSPRepBiFPANNeck
from models.demos.yolov6l.tt.ttnn_detect import TtDetect


class TtYolov6l:
    def __init__(self, device, parameters, model_params):
        self.backbone = TtCSPBepBackbone(device, parameters.backbone, model_params.backbone)
        self.neck = TtCSPRepBiFPANNeck(device, parameters.neck, model_params.neck)
        self.detect = TtDetect(device, parameters.detect, model_params.detect)

    def __call__(self, input_tensor):
        N, C, H, W = input_tensor.shape
        ## Padding from image channels (3) to min channels (16)
        min_channels = 16
        if C < min_channels:
            channel_padding_needed = min_channels - C
            nchw = ttnn.pad(input_tensor, ((0, 0), (0, channel_padding_needed), (0, 0), (0, 0)), value=0.0)
        else:
            nchw = input_tensor
        nhwc = ttnn.permute(nchw, (0, 2, 3, 1))  # NCHW -> NHWC
        ttnn.deallocate(nchw)
        ttnn.deallocate(input_tensor)
        nhwc = ttnn.reallocate(nhwc)
        input_tensor = ttnn.reshape(nhwc, [1, 1, nhwc.shape[0] * nhwc.shape[1] * nhwc.shape[2], nhwc.shape[-1]])

        backbone_outputs = self.backbone(input_tensor)
        neck_outputs = self.neck(backbone_outputs)
        output = self.detect(neck_outputs)
        return output

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from models.experimental.yolov6l.tt.ttnn_cspbep_backbone import TtCSPBepBackbone
from models.experimental.yolov6l.tt.ttnn_csprep_bifpanneck import TtCSPRepBiFPANNeck
from models.experimental.yolov6l.tt.ttnn_detect import TtDetect


class TtYolov6l:
    def __init__(self, device, parameters, model_params):
        self.backbone = TtCSPBepBackbone(device, parameters.backbone, model_params.backbone)
        self.neck = TtCSPRepBiFPANNeck(device, parameters.neck, model_params.neck)
        self.detect = TtDetect(device, parameters.detect, model_params.detect)

    def __call__(self, input_tensor):
        backbone_outputs = self.backbone(input_tensor)
        neck_outputs = self.neck(backbone_outputs)
        output = self.detect(neck_outputs)
        return output

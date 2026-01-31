# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from models.experimental.retinanet.tt.tt_backbone import TTBackbone
from models.experimental.retinanet.tt.tt_reg_head import TtnnRetinaNetRegressionHead
from models.experimental.retinanet.tt.tt_cls_head import TtnnRetinaNetClassificationHead


class TTRetinaNet:
    def __init__(self, parameters, model_config, device, model_args, name="backbone"):
        self.backbone = TTBackbone(
            parameters=parameters["backbone"], model_config=model_config, device=device, model_args=model_args
        )

        input_shapes = [(64, 64), (32, 32), (16, 16), (8, 8), (4, 4)]

        self.regression_head = TtnnRetinaNetRegressionHead(
            parameters=parameters["head"]["regression_head"],
            device=device,
            input_shapes=input_shapes,
            model_config=model_config,
        )
        self.classification_head = TtnnRetinaNetClassificationHead(
            parameters=parameters["head"]["classification_head"],
            device=device,
            input_shapes=input_shapes,
            model_config=model_config,
        )

    def __call__(self, x, device):
        backbone_output = self.backbone(x, device)
        fpn_features = [backbone_output[key] for key in ["0", "1", "2", "p6", "p7"]]

        regression_output = self.regression_head(fpn_features)
        classification_output = self.classification_head(fpn_features)

        self.output_tensor = {
            **backbone_output,
            "regression": regression_output,
            "classification": classification_output,
        }

        return self.output_tensor

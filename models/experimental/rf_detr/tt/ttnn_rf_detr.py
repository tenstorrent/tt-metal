# SPDX-License-Identifier: Apache-2.0
"""End-to-end RF-DETR on Tenstorrent.

On-device chain: windowed DINOv2 backbone -> C2f projector -> two-stage deformable
transformer + heads. The only remaining host glue is the backbone embeddings
(patch conv + window partition) and feature-map shaping (LN + window unpartition),
which are reshape-heavy; these are slated to move on-device next, after which the
whole model can be metal-traced with 2 command queues.
"""

import torch

import ttnn
from models.experimental.rf_detr.reference.modeling_rf_detr import RfDetrOutput
from models.experimental.rf_detr.tt.ttnn_backbone import TtDinoBackbone
from models.experimental.rf_detr.tt.ttnn_projector import TtProjector
from models.experimental.rf_detr.tt.ttnn_transformer import TtTransformer


class TtRfDetr:
    def __init__(self, ref_model, device):
        self.ref = ref_model.eval()
        self.device = device
        self.backbone = TtDinoBackbone(ref_model, device)
        self.projector = TtProjector(ref_model, device)
        self.transformer = TtTransformer(ref_model, device)

    def __call__(self, pixel_values):
        # backbone (device layers) -> 4 host feature maps [1,384,40,40]
        feats = self.backbone.feature_maps(pixel_values)
        # upload as channels-last [1,1600,384] for the on-device projector
        feats_cl = [
            ttnn.from_torch(
                f.flatten(2).transpose(1, 2).contiguous(),
                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device,
            )
            for f in feats
        ]
        source = self.projector(feats_cl)               # device [1,1600,256]
        logits, pred_boxes = self.transformer(source)    # torch [1,300,91],[1,300,4]
        return RfDetrOutput(logits=logits, pred_boxes=pred_boxes)

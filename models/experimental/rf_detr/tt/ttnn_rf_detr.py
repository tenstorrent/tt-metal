# SPDX-License-Identifier: Apache-2.0
"""End-to-end RF-DETR on Tenstorrent.

Baseline split (iteration 0): the windowed DINOv2 backbone (dominant compute) runs
on device; the projector + two-stage selection + deformable decoder + heads run on
host via the faithful torch reference modules. Subsequent optimization iterations
move these stages onto the device. The post-backbone tail mirrors
RfDetrForObjectDetection.forward exactly (so e2e accuracy is limited only by the
backbone's bf16 numerics).
"""

import torch
import torch.nn.functional as F

from models.experimental.rf_detr.reference.modeling_rf_detr import refine_bboxes, RfDetrOutput
from models.experimental.rf_detr.tt.ttnn_backbone import TtDinoBackbone


class TtRfDetr:
    def __init__(self, ref_model, device):
        self.ref = ref_model.eval()
        self.device = device
        self.backbone = TtDinoBackbone(ref_model, device)

    def backbone_feature_maps(self, pixel_values):
        """Device backbone (metal-traced) -> 4 torch feature maps [1,384,40,40]."""
        return self.backbone.feature_maps(pixel_values, use_trace=True)

    @torch.no_grad()
    def _tail_host(self, feature_maps, pixel_values):
        """Projector + transformer + heads on host (faithful to ref.forward)."""
        ref = self.ref
        b, _, height, width = pixel_values.shape
        device = pixel_values.device
        pixel_mask = torch.ones((b, height, width), dtype=torch.long, device=device)

        bb = ref.backbone[0]
        source = bb.projector(feature_maps)
        mask = F.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]

        source_flatten = source.flatten(2).transpose(1, 2)
        mask_flatten = mask.flatten(1)
        spatial_shapes_list = [tuple(source.shape[2:])]
        spatial_shapes = torch.as_tensor(spatial_shapes_list, dtype=torch.long, device=device)
        valid_ratios = ref._get_valid_ratio(mask).unsqueeze(1)

        object_query_embedding, output_proposals, invalid_mask = ref._gen_proposals(
            source_flatten, ~mask_flatten, spatial_shapes_list
        )
        topk = ref.num_queries
        tf = ref.transformer
        object_query = tf.enc_output[0](object_query_embedding)
        object_query = tf.enc_output_norm[0](object_query)
        enc_class = tf.enc_out_class_embed[0](object_query)
        enc_class = enc_class.masked_fill(invalid_mask, float("-inf"))
        delta_bbox = tf.enc_out_bbox_embed[0](object_query)
        enc_coord = refine_bboxes(output_proposals, delta_bbox)
        topk_proposals = torch.topk(enc_class.max(-1)[0], topk, dim=1)[1]
        topk_coords = torch.gather(enc_coord, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, 4))

        reference_points = ref.refpoint_embed.weight[: ref.num_queries].unsqueeze(0).expand(b, -1, -1)
        reference_points = refine_bboxes(topk_coords, reference_points)
        init_reference_points = reference_points
        target = ref.query_feat.weight[: ref.num_queries].unsqueeze(0).expand(b, -1, -1)

        intermediate, _ = tf.decoder(
            target=target,
            reference_points=reference_points,
            valid_ratios=valid_ratios,
            encoder_hidden_states=source_flatten,
            spatial_shapes=spatial_shapes,
            spatial_shapes_list=spatial_shapes_list,
            encoder_attention_mask=mask_flatten,
        )
        last_hidden_state = intermediate[-1]
        logits = ref.class_embed(last_hidden_state)
        pred_boxes = refine_bboxes(init_reference_points, ref.bbox_embed(last_hidden_state))
        return RfDetrOutput(logits=logits, pred_boxes=pred_boxes, init_reference_points=init_reference_points)

    def __call__(self, pixel_values):
        feats = self.backbone_feature_maps(pixel_values)
        return self._tail_host(feats, pixel_values)

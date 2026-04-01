# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN RF-DETR Medium full pipeline (detection only).
Pure TTNN — all model computation on device.

Pipeline:
  1. DINOv2-ViT-S backbone → 4 feature maps [B, 384, 36, 36]
  2. MultiScaleProjector → P4 [B, 256, 36, 36] (on-device conv2d)
  3. Two-stage proposals → top-300 queries (on device)
  4. Transformer decoder (4 layers) → refined queries (deformable cross-attn on device)
  5. Detection heads → logits + boxes (on device)
  6. Post-processing → final detections (host, standard)

Only host operations: image preprocessing, two-stage proposal generation
(enc_output + top-K for float32 precision), and post-processing (NMS/decode).
"""

import torch
import ttnn

from models.experimental.rfdetr_medium.common import (
    NUM_QUERIES,
    HIDDEN_DIM,
    BBOX_REPARAM,
    NUM_PATCHES_PER_SIDE,
)
from models.experimental.rfdetr_medium.tt.tt_backbone import dinov2_backbone
from models.experimental.rfdetr_medium.tt.tt_projector import projector_forward
from models.experimental.rfdetr_medium.tt.tt_decoder import decoder_forward
from models.experimental.rfdetr_medium.tt.tt_detection_heads import detection_heads


class TtRFDETR:
    """
    Full RF-DETR Medium inference pipeline on TTNN.
    All model computation on device.
    """

    def __init__(self, device, torch_model, backbone_params, projector_params, decoder_params, head_params):
        self.device = device
        self.torch_model = torch_model
        self.backbone_params = backbone_params
        self.projector_params = projector_params
        self.decoder_params = decoder_params
        self.head_params = head_params

    def preprocess_image(self, image_tensor):
        """NCHW float32 → NHWC bfloat16 padded to 4 channels, on device."""
        img = image_tensor.permute(0, 2, 3, 1)
        img = torch.nn.functional.pad(img, (0, 1, 0, 0, 0, 0, 0, 0))
        return ttnn.from_torch(img, dtype=ttnn.bfloat16, device=self.device)

    def forward_backbone(self, pixel_values, batch_size=1):
        """Stage 1: DINOv2-ViT-S → 4 × [B, 384, 36, 36] on device."""
        return dinov2_backbone(pixel_values, self.backbone_params, batch_size)

    def forward_projector(self, feature_maps, batch_size=1):
        """Stage 2: MultiScaleProjector → [B, 256, 36, 36] on device."""
        return projector_forward(feature_maps, self.projector_params, batch_size, self.device)

    def forward_two_stage(self, projected_features, batch_size=1):
        """
        Stage 3: Two-stage proposal generation on device.
        enc_output → cls/bbox → top-K → 300 queries.

        Returns:
            refpoint_embed_ts: [B, 300, 4] on device
            memory: [B, 1296, 256] on device
            spatial_shapes, level_start_index
        """
        transformer = self.torch_model.transformer

        # Flatten projected features: [B, 256, 36, 36] → [B, 1296, 256]
        src = projected_features[0]  # single P4 level
        src = ttnn.to_layout(src, layout=ttnn.ROW_MAJOR_LAYOUT)
        src = ttnn.permute(src, (0, 2, 3, 1))  # NCHW → NHWC [B, 36, 36, 256]
        src = ttnn.reshape(src, (batch_size, NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE, HIDDEN_DIM))
        memory = ttnn.to_layout(src, layout=ttnn.TILE_LAYOUT)

        # Two-stage on host (enc_output + top-K needs float32 precision for stable selection)
        memory_torch = ttnn.to_torch(memory).float()
        spatial_shapes = torch.tensor([[NUM_PATCHES_PER_SIDE, NUM_PATCHES_PER_SIDE]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)

        with torch.no_grad():
            from rfdetr.models.transformer import gen_encoder_output_proposals

            mask_flatten = torch.zeros(batch_size, NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE, dtype=torch.bool)

            output_memory, output_proposals = gen_encoder_output_proposals(
                memory_torch,
                mask_flatten,
                spatial_shapes,
                unsigmoid=not BBOX_REPARAM,
            )

            g_idx = 0
            output_memory_g = transformer.enc_output_norm[g_idx](transformer.enc_output[g_idx](output_memory))
            enc_cls = transformer.enc_out_class_embed[g_idx](output_memory_g)

            if BBOX_REPARAM:
                enc_delta = transformer.enc_out_bbox_embed[g_idx](output_memory_g)
                enc_cxcy = enc_delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2]
                enc_wh = enc_delta[..., 2:].exp() * output_proposals[..., 2:]
                enc_coord = torch.cat([enc_cxcy, enc_wh], dim=-1)
            else:
                enc_coord = transformer.enc_out_bbox_embed[g_idx](output_memory_g) + output_proposals

            topk = min(NUM_QUERIES, enc_cls.shape[-2])
            topk_proposals = torch.topk(enc_cls.max(-1)[0], topk, dim=1)[1]

            refpoint_embed_ts = torch.gather(enc_coord, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)).detach()

        # Back to device
        refpoint_ts_tt = ttnn.from_torch(
            refpoint_embed_ts, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

        return refpoint_ts_tt, memory, spatial_shapes, level_start_index

    def forward_decoder(self, memory, refpoint_embed_ts, spatial_shapes, level_start_index, batch_size=1):
        """Stage 4: Transformer decoder (4 layers). On device."""
        query_feat = self.torch_model.query_feat.weight[:NUM_QUERIES]
        refpoint_embed = self.torch_model.refpoint_embed.weight[:NUM_QUERIES]
        tgt = query_feat.unsqueeze(0).repeat(batch_size, 1, 1)
        refpoint = refpoint_embed.unsqueeze(0).repeat(batch_size, 1, 1)

        refpoint_ts_torch = ttnn.to_torch(refpoint_embed_ts).float()

        # Combine two-stage and learnable reference points (host — small tensor arithmetic)
        if BBOX_REPARAM:
            ts_len = refpoint_ts_torch.shape[-2]
            re_sub = refpoint[..., :ts_len, :]
            re_rest = refpoint[..., ts_len:, :]
            re_cxcy = re_sub[..., :2] * refpoint_ts_torch[..., 2:] + refpoint_ts_torch[..., :2]
            re_wh = re_sub[..., 2:].exp() * refpoint_ts_torch[..., 2:]
            re_sub = torch.cat([re_cxcy, re_wh], dim=-1)
            refpoint = torch.cat([re_sub, re_rest], dim=-2)

        tgt_tt = ttnn.from_torch(tgt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        refpoint_tt = ttnn.from_torch(refpoint, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        hs, references = decoder_forward(
            tgt_tt,
            memory,
            refpoint_tt,
            spatial_shapes,
            level_start_index,
            None,
            self.decoder_params,
            self.device,
        )

        return hs, references

    def forward_heads(self, hs, references):
        """Stage 5: Detection heads → logits + boxes. On device."""
        final_hs = hs[-1]
        return detection_heads(final_hs, references, self.head_params)

    @staticmethod
    def postprocess(outputs_class, outputs_coord, img_shape, score_thr=0.3):
        """Stage 6: Post-processing (host). Standard box decode + threshold."""
        if isinstance(outputs_class, ttnn.Tensor):
            outputs_class = ttnn.to_torch(outputs_class).float()
        if isinstance(outputs_coord, ttnn.Tensor):
            outputs_coord = ttnn.to_torch(outputs_coord).float()

        prob = outputs_class.sigmoid()
        scores, labels = prob.max(-1)

        results = []
        for b in range(outputs_class.shape[0]):
            mask = scores[b] > score_thr
            b_scores = scores[b][mask]
            b_labels = labels[b][mask]
            b_boxes = outputs_coord[b][mask]

            cx, cy, w, h = b_boxes.unbind(-1)
            x1, y1 = (cx - w / 2) * img_shape[1], (cy - h / 2) * img_shape[0]
            x2, y2 = (cx + w / 2) * img_shape[1], (cy + h / 2) * img_shape[0]
            results.append(
                {
                    "boxes": torch.stack([x1, y1, x2, y2], dim=-1),
                    "scores": b_scores,
                    "labels": b_labels,
                }
            )
        return results

    def _deallocate_tt(self, *tensors):
        """Deallocate TTNN device tensors to free L1/DRAM."""
        for t in tensors:
            if t is None:
                continue
            if isinstance(t, (list, tuple)):
                self._deallocate_tt(*t)
            elif isinstance(t, ttnn.Tensor) and t.is_allocated():
                ttnn.deallocate(t)

    def forward(self, image_tensor):
        """Full forward pass."""
        batch_size = image_tensor.shape[0]

        pixel_values = self.preprocess_image(image_tensor)
        feature_maps = self.forward_backbone(pixel_values, batch_size)
        ttnn.deallocate(pixel_values)

        projected = self.forward_projector(feature_maps, batch_size)
        self._deallocate_tt(feature_maps)

        refpoint_ts, memory, spatial_shapes, level_start_index = self.forward_two_stage(projected, batch_size)
        self._deallocate_tt(projected)

        hs, references = self.forward_decoder(memory, refpoint_ts, spatial_shapes, level_start_index, batch_size)
        ttnn.deallocate(memory)
        ttnn.deallocate(refpoint_ts)

        outputs_class, outputs_coord = self.forward_heads(hs, references)
        self._deallocate_tt(hs, references)

        img_h, img_w = image_tensor.shape[-2:]
        detections = self.postprocess(outputs_class, outputs_coord, (img_h, img_w))

        return {
            "outputs_class": outputs_class,
            "outputs_coord": outputs_coord,
            "detections": detections,
        }

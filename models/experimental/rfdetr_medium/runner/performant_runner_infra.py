# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Performance runner infrastructure for RF-DETR Medium.

Encapsulates model loading, weight preparation, input setup, and the
device forward pass split into two traceable segments:
  Trace A: backbone + projector  (pure device)
  Trace B: decoder + heads       (pure device)
with a host-side two-stage top-K selection in between.
"""

import torch
import ttnn

from models.experimental.rfdetr_medium.common import (
    RESOLUTION,
    NUM_QUERIES,
    HIDDEN_DIM,
    BBOX_REPARAM,
    NUM_PATCHES_PER_SIDE,
    load_torch_model,
)
from models.experimental.rfdetr_medium.tt.tt_backbone import dinov2_backbone
from models.experimental.rfdetr_medium.tt.tt_projector import projector_forward
from models.experimental.rfdetr_medium.tt.tt_decoder import decoder_forward
from models.experimental.rfdetr_medium.tt.tt_detection_heads import detection_heads
from models.experimental.rfdetr_medium.tt.model_preprocessing import (
    load_backbone_weights,
    load_projector_weights,
    load_decoder_weights,
    load_detection_head_weights,
)


class RFDETRPerformanceRunnerInfra:
    def __init__(self, device, batch_size=1):
        self.device = device
        self.batch_size = batch_size

        torch.manual_seed(0)
        self.torch_model = load_torch_model()

        self.backbone_params = load_backbone_weights(self.torch_model, device)
        self.projector_params = load_projector_weights(self.torch_model, device)
        self.decoder_params = load_decoder_weights(self.torch_model, device)
        self.head_params = load_detection_head_weights(self.torch_model, device)

        self.torch_input = torch.randn(batch_size, 3, RESOLUTION, RESOLUTION)

        self.input_tensor = None
        self.backbone_output = None
        self.projector_output = None
        self.memory = None
        self.decoder_tgt = None
        self.decoder_refpoint = None
        self.decoder_output = None
        self.decoder_references = None
        self.cls_output = None
        self.bbox_output = None

    def setup_l1_input(self, torch_input_tensor=None):
        """Prepare input tensor on device in L1 (NHWC padded to 4ch)."""
        inp = self.torch_input if torch_input_tensor is None else torch_input_tensor
        img = inp.permute(0, 2, 3, 1)  # NCHW → NHWC
        img = torch.nn.functional.pad(img, (0, 1, 0, 0, 0, 0, 0, 0))  # pad C=3→4
        tt_host = ttnn.from_torch(img, dtype=ttnn.bfloat16)
        return tt_host

    def setup_dram_input(self, torch_input_tensor=None):
        """Prepare input tensor in DRAM on device."""
        tt_host = self.setup_l1_input(torch_input_tensor)
        tt_dram = tt_host.to(self.device, ttnn.DRAM_MEMORY_CONFIG)
        return tt_host, tt_dram

    # ----- Trace A: backbone + projector (pure device) -----

    def run_backbone_projector(self):
        """backbone → projector. Input: self.input_tensor. Output: self.projector_output, self.memory."""
        feature_maps = dinov2_backbone(self.input_tensor, self.backbone_params, self.batch_size)
        projected = projector_forward(feature_maps, self.projector_params, self.batch_size, self.device)

        src = projected[0]
        src = ttnn.to_layout(src, layout=ttnn.ROW_MAJOR_LAYOUT)
        src = ttnn.permute(src, (0, 2, 3, 1))  # NCHW → NHWC
        src = ttnn.reshape(src, (self.batch_size, NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE, HIDDEN_DIM))
        self.memory = ttnn.to_layout(src, layout=ttnn.TILE_LAYOUT)
        self.projector_output = projected

    # ----- Host: two-stage top-K -----

    def run_two_stage_host(self):
        """Top-K proposal selection on host. Returns refpoint, tgt, refpoint tensors on device."""
        transformer = self.torch_model.transformer
        memory_torch = ttnn.to_torch(self.memory).float()

        spatial_shapes = torch.tensor([[NUM_PATCHES_PER_SIDE, NUM_PATCHES_PER_SIDE]], dtype=torch.long)
        level_start_index = torch.tensor([0], dtype=torch.long)
        self.spatial_shapes = spatial_shapes
        self.level_start_index = level_start_index

        with torch.no_grad():
            from rfdetr.models.transformer import gen_encoder_output_proposals

            mask = torch.zeros(self.batch_size, NUM_PATCHES_PER_SIDE * NUM_PATCHES_PER_SIDE, dtype=torch.bool)
            output_memory, output_proposals = gen_encoder_output_proposals(
                memory_torch,
                mask,
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

        query_feat = self.torch_model.query_feat.weight[:NUM_QUERIES]
        refpoint_embed = self.torch_model.refpoint_embed.weight[:NUM_QUERIES]
        tgt = query_feat.unsqueeze(0).repeat(self.batch_size, 1, 1)
        refpoint = refpoint_embed.unsqueeze(0).repeat(self.batch_size, 1, 1)

        if BBOX_REPARAM:
            ts_len = refpoint_embed_ts.shape[-2]
            re_sub = refpoint[..., :ts_len, :]
            re_rest = refpoint[..., ts_len:, :]
            re_cxcy = re_sub[..., :2] * refpoint_embed_ts[..., 2:] + refpoint_embed_ts[..., :2]
            re_wh = re_sub[..., 2:].exp() * refpoint_embed_ts[..., 2:]
            re_sub = torch.cat([re_cxcy, re_wh], dim=-1)
            refpoint = torch.cat([re_sub, re_rest], dim=-2)

        self.decoder_tgt = ttnn.from_torch(tgt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        self.decoder_refpoint = ttnn.from_torch(
            refpoint, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device
        )

    # ----- Trace B: decoder + detection heads (pure device) -----

    def run_decoder_heads(self):
        """decoder → heads. Input: self.decoder_tgt, self.decoder_refpoint, self.memory. Output: cls, bbox."""
        hs, references = decoder_forward(
            self.decoder_tgt,
            self.memory,
            self.decoder_refpoint,
            self.spatial_shapes,
            self.level_start_index,
            None,
            self.decoder_params,
            self.device,
        )

        self.decoder_output = hs
        self.decoder_references = references
        self.cls_output, self.bbox_output = detection_heads(hs[-1], references, self.head_params)

    # ----- Full forward -----

    def run_full(self):
        """Full forward: backbone+projector → host topk → decoder+heads."""
        self.run_backbone_projector()
        self.run_two_stage_host()
        self.run_decoder_heads()

    def dealloc_output(self):
        if self.cls_output is not None:
            ttnn.deallocate(self.cls_output)
            self.cls_output = None
        if self.bbox_output is not None:
            ttnn.deallocate(self.bbox_output)
            self.bbox_output = None
        if self.memory is not None:
            ttnn.deallocate(self.memory)
            self.memory = None

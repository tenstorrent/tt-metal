# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Full TTNN ATSS model: Swin-L (TTNN) + FPN (TTNN) + DyHead (hybrid) + ATSS Head (TTNN).

DyHead supports two modes (controlled by hybrid_dyhead flag):
  hybrid=True  (default):
    - DCNv2 spatial attention runs on CPU (no native TTNN kernel)
    - Scale-aware and task-aware attention run on TTNN device
  hybrid=False:
    - Entire DyHead runs on CPU via PyTorch

Data flow (hybrid mode):
  [device] Swin-L backbone   -> 3 NCHW feature maps
  [device] FPN                -> 5 NCHW feature maps
  [host]   DyHead spatial     -> DCNv2 on CPU
  [device] DyHead scale/task  -> attention on TTNN
  [device] ATSS Head          -> (cls, reg, centerness) per level
  [host]   Post-processing    -> bboxes, scores, labels
"""

import torch
import ttnn

from models.experimental.atss_swin_l_dyhead.tt.tt_swin_backbone import build_atss_backbone
from models.experimental.atss_swin_l_dyhead.common import (
    ATSS_FPN_IN_CHANNELS,
    ATSS_FPN_OUT_CHANNELS,
    ATSS_FPN_NUM_OUTS,
    ATSS_NUM_CLASSES,
    ATSS_NUM_ANCHORS,
    ATSS_PIXEL_MEAN,
    ATSS_PIXEL_STD,
    ATSS_PAD_SIZE_DIVISOR,
    ATSS_SCORE_THR,
    ATSS_NMS_IOU_THR,
    ATSS_MAX_PER_IMG,
)
from models.experimental.atss_swin_l_dyhead.tt.tt_fpn import TtFPN
from models.experimental.atss_swin_l_dyhead.tt.tt_atss_head import TtATSSHead
from models.experimental.atss_swin_l_dyhead.tt.weight_loading import (
    load_fpn_weights,
    load_atss_head_weights,
    load_dyhead_weights,
)
from models.experimental.atss_swin_l_dyhead.reference.dyhead import build_dyhead_for_atss
from models.experimental.atss_swin_l_dyhead.reference.postprocess import atss_postprocess
from models.experimental.atss_swin_l_dyhead.tt.tt_dyhead import TtHybridDyHead


class TtATSSModel:
    """
    Full ATSS detection model with TTNN backbone/FPN/head and PyTorch DyHead.

    Usage:
        model = TtATSSModel.from_checkpoint(checkpoint_path, device)
        results = model.predict(image_tensor, img_shape=(H, W))
    """

    def __init__(
        self,
        device,
        backbone,
        fpn,
        dyhead,
        head,
        pixel_mean=ATSS_PIXEL_MEAN,
        pixel_std=ATSS_PIXEL_STD,
        pad_size_divisor=ATSS_PAD_SIZE_DIVISOR,
        inputs_mesh_mapper=None,
        output_mesh_composer=None,
    ):
        self.device = device
        self.backbone = backbone
        self.fpn = fpn
        self.dyhead = dyhead
        self.head = head
        self.pixel_mean = torch.tensor(pixel_mean).view(1, 3, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(1, 3, 1, 1)
        self.pad_size_divisor = pad_size_divisor
        self.inputs_mesh_mapper = inputs_mesh_mapper
        self.output_mesh_composer = output_mesh_composer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: ttnn.Device,
        input_h=None,
        input_w=None,
        hybrid_dyhead: bool = True,
        inputs_mesh_mapper=None,
        output_mesh_composer=None,
    ):
        """Build the full model from an mmdet checkpoint.

        Args:
            checkpoint_path: path to mmdet .pth checkpoint.
            device: TTNN device.
            input_h: padded input height (default: ATSS_INPUT_H).
            input_w: padded input width  (default: ATSS_INPUT_W).
            hybrid_dyhead: if True, run scale/task attention on TTNN device
                (spatial DCNv2 stays on CPU). If False, run entire DyHead on CPU.
            inputs_mesh_mapper: mesh mapper for sharding inputs across devices.
            output_mesh_composer: mesh composer for gathering outputs from devices.
        """
        # 1. Swin-L backbone (TTNN)
        backbone = build_atss_backbone(checkpoint_path, device, input_h=input_h, input_w=input_w)

        # 2. FPN (TTNN)
        fpn_params = load_fpn_weights(
            checkpoint_path,
            device,
            in_channels=tuple(ATSS_FPN_IN_CHANNELS),
            out_channels=ATSS_FPN_OUT_CHANNELS,
            num_outs=ATSS_FPN_NUM_OUTS,
        )
        fpn = TtFPN(
            device,
            fpn_params,
            in_channels=tuple(ATSS_FPN_IN_CHANNELS),
            out_channels=ATSS_FPN_OUT_CHANNELS,
            num_outs=ATSS_FPN_NUM_OUTS,
        )

        # 3. DyHead — load PyTorch model first (needed in both modes for DCNv2)
        pt_dyhead = build_dyhead_for_atss()
        dyhead_sd = load_dyhead_weights(checkpoint_path, device)
        pt_dyhead.load_state_dict(dyhead_sd, strict=True)
        pt_dyhead.eval()

        if hybrid_dyhead:
            dyhead = TtHybridDyHead(
                device,
                pt_dyhead,
                inputs_mesh_mapper=inputs_mesh_mapper,
                output_mesh_composer=output_mesh_composer,
            )
        else:
            dyhead = pt_dyhead

        # 4. ATSS Head (TTNN)
        head_params = load_atss_head_weights(checkpoint_path, device)
        head = TtATSSHead(
            device,
            head_params,
            num_classes=ATSS_NUM_CLASSES,
            in_channels=ATSS_FPN_OUT_CHANNELS,
            num_anchors=ATSS_NUM_ANCHORS,
            num_levels=ATSS_FPN_NUM_OUTS,
        )

        return cls(
            device,
            backbone,
            fpn,
            dyhead,
            head,
            inputs_mesh_mapper=inputs_mesh_mapper,
            output_mesh_composer=output_mesh_composer,
        )

    def preprocess(self, img: torch.Tensor) -> torch.Tensor:
        """Normalize and pad image. Input: [1, 3, H, W] BGR float [0, 255]."""
        x = img.float()
        x = x[:, [2, 1, 0], :, :]  # BGR → RGB
        x = (x - self.pixel_mean) / self.pixel_std
        _, _, h, w = x.shape
        pad_h = (self.pad_size_divisor - h % self.pad_size_divisor) % self.pad_size_divisor
        pad_w = (self.pad_size_divisor - w % self.pad_size_divisor) % self.pad_size_divisor
        if pad_h > 0 or pad_w > 0:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), value=0)
        return x

    def forward_backbone_fpn(self, x_ttnn):
        """Run Swin-L backbone + FPN on device. Returns list of 5 NCHW ttnn tensors."""
        backbone_feats = self.backbone(x_ttnn)
        fpn_feats = self.fpn(backbone_feats)
        return fpn_feats

    def forward_dyhead(self, fpn_feats_ttnn):
        """Run DyHead on FPN features (hybrid TTNN or pure PyTorch)."""
        torch_feats = []
        for feat in fpn_feats_ttnn:
            t = ttnn.to_torch(ttnn.from_device(feat), mesh_composer=self.output_mesh_composer).float()
            torch_feats.append(t)

        with torch.no_grad():
            if isinstance(self.dyhead, TtHybridDyHead):
                dy_feats = self.dyhead(torch_feats)
            else:
                dy_feats = self.dyhead(torch_feats)
        return dy_feats

    def forward_head(self, dy_feats_torch):
        """Transfer DyHead features to device, run ATSS Head, return torch outputs."""
        ttnn_feats = []
        for feat in dy_feats_torch:
            t = ttnn.from_torch(
                feat,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=self.inputs_mesh_mapper,
            )
            ttnn_feats.append(t)

        cls_scores_ttnn, bbox_preds_ttnn, centernesses_ttnn = self.head(ttnn_feats)

        # Head returns NHWC (1, H, W, C); postprocess expects NCHW (1, C, H, W).
        # Single-device: use to_torch without mesh_composer for correct host copy.
        to_torch_kw = {} if self.output_mesh_composer is None else {"mesh_composer": self.output_mesh_composer}
        cls_scores = [
            ttnn.to_torch(ttnn.from_device(x), **to_torch_kw).float().permute(0, 3, 1, 2) for x in cls_scores_ttnn
        ]
        bbox_preds = [
            ttnn.to_torch(ttnn.from_device(x), **to_torch_kw).float().permute(0, 3, 1, 2) for x in bbox_preds_ttnn
        ]
        centernesses = [
            ttnn.to_torch(ttnn.from_device(x), **to_torch_kw).float().permute(0, 3, 1, 2) for x in centernesses_ttnn
        ]
        return cls_scores, bbox_preds, centernesses

    def forward(self, x_ttnn):
        """
        Full forward: backbone → FPN → DyHead (CPU) → ATSS Head.
        Input: preprocessed image as ttnn tensor [1, 3, H, W] NCHW.
        Returns: (cls_scores, bbox_preds, centernesses) as torch tensors.
        """
        fpn_feats = self.forward_backbone_fpn(x_ttnn)
        dy_feats = self.forward_dyhead(fpn_feats)
        cls_scores, bbox_preds, centernesses = self.forward_head(dy_feats)
        return cls_scores, bbox_preds, centernesses

    @torch.no_grad()
    def predict(
        self,
        img: torch.Tensor,
        img_shape,
        score_thr=ATSS_SCORE_THR,
        nms_iou_thr=ATSS_NMS_IOU_THR,
        max_per_img=ATSS_MAX_PER_IMG,
    ):
        """
        End-to-end inference.
        img: [1, 3, H, W] BGR float [0, 255].
        img_shape: (H, W) original image size.
        Returns: dict with 'bboxes', 'scores', 'labels'.
        """
        x = self.preprocess(img)
        x_ttnn = ttnn.from_torch(
            x,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        cls_scores, bbox_preds, centernesses = self.forward(x_ttnn)
        return atss_postprocess(
            cls_scores,
            bbox_preds,
            centernesses,
            img_shape=img_shape,
            score_thr=score_thr,
            nms_iou_thr=nms_iou_thr,
            max_per_img=max_per_img,
        )

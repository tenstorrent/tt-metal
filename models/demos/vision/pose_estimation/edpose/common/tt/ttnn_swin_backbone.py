# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Optimized Swin-L backbone for ED-Pose on TT P150.

Swin-L runs on CPU with torch.compile(mode="reduce-overhead") for ~20% speedup.
The device is not used for the Swin stages because window attention (144 tokens)
produces many small matmuls that don't saturate the device, and the host↔device
transfer overhead (769ms measured) far exceeds device compute savings (69ms).

Provides the same interface as EDPoseBackbone but with torch.compile optimization.
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

EDPOSE_ROOT = os.environ.get("EDPOSE_ROOT", os.path.expanduser("~/ttwork/ED-Pose"))


def _ensure_edpose_on_path():
    if EDPOSE_ROOT not in sys.path:
        sys.path.insert(0, EDPOSE_ROOT)


class TTSwinBackbone:
    """
    Swin-L backbone with torch.compile optimization.

    Uses the original Swin-L implementation on CPU with torch.compile for
    JIT optimization. Returns encoder-ready tensors (same interface as EDPoseBackbone).
    """

    def __init__(self, device, checkpoint_path=None, d_model=256, num_feature_levels=5,
                 use_compile=False):
        self.device = device  # TT device (stored but not used for backbone)
        self.d_model = d_model
        self.num_feature_levels = num_feature_levels

        if checkpoint_path is None:
            checkpoint_path = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

        _ensure_edpose_on_path()

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
        full_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self._build_swin(full_sd)
        self._build_input_proj(full_sd)
        self._build_position_embedding()
        self.level_embed = full_sd.get("transformer.level_embed", None)

        if use_compile:
            try:
                self.swin = torch.compile(self.swin, mode="reduce-overhead")
            except Exception:
                pass

    def _build_swin(self, full_sd):
        _ensure_edpose_on_path()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "swin_transformer",
            os.path.join(EDPOSE_ROOT, "models", "edpose", "backbones", "swin_transformer.py"),
        )
        swin_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(swin_mod)

        self.swin = swin_mod.build_swin_transformer(
            "swin_L_384_22k",
            pretrain_img_size=384,
            out_indices=(0, 1, 2, 3),
            dilation=False,
            use_checkpoint=False,
        )

        swin_sd = {k[len("backbone.0."):]: v for k, v in full_sd.items()
                   if k.startswith("backbone.0.")}
        self.swin.load_state_dict(swin_sd, strict=False)
        self.swin.eval()

    def _build_input_proj(self, full_sd):
        backbone_channels = self.swin.num_features
        self.input_proj = nn.ModuleList()
        for i in range(len(backbone_channels)):
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(backbone_channels[i], self.d_model, kernel_size=1),
                nn.GroupNorm(32, self.d_model),
            ))
        for i in range(self.num_feature_levels - len(backbone_channels)):
            in_ch = backbone_channels[-1] if i == 0 else self.d_model
            self.input_proj.append(nn.Sequential(
                nn.Conv2d(in_ch, self.d_model, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(32, self.d_model),
            ))
        proj_sd = {k[len("input_proj."):]: v for k, v in full_sd.items()
                   if k.startswith("input_proj.")}
        self.input_proj.load_state_dict(proj_sd, strict=True)
        self.input_proj.eval()

    def _build_position_embedding(self):
        from models.demos.vision.pose_estimation.edpose.common.tt.ttnn_edpose_backbone import (
            PositionEmbeddingSineHW,
        )
        self.pos_embed = PositionEmbeddingSineHW(
            num_pos_feats=self.d_model // 2,
            temperatureH=20, temperatureW=20, normalize=True,
        )

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        return torch.stack([valid_W.float() / W, valid_H.float() / H], -1)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing="ij",
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    @torch.inference_mode()
    def __call__(self, image_tensor, mask):
        """
        Run Swin-L backbone (CPU) + input_proj + pos_embed.

        Args:
            image_tensor: (N, 3, H, W) float tensor
            mask: (N, H, W) bool — True = padding

        Returns dict with same keys as EDPoseBackbone.
        """
        _ensure_edpose_on_path()
        from util.misc import NestedTensor

        nested = NestedTensor(image_tensor, mask)
        features_dict = self.swin(nested)

        srcs = []
        masks_list = []
        poss = []

        for lvl in range(len(features_dict)):
            feat = features_dict[lvl]
            src_l, mask_l = feat.tensors, feat.mask
            srcs.append(self.input_proj[lvl](src_l))
            masks_list.append(mask_l)
            poss.append(self.pos_embed(src_l, mask_l).to(src_l.dtype))

        if self.num_feature_levels > len(features_dict):
            _len_srcs = len(features_dict)
            for lvl in range(_len_srcs, self.num_feature_levels):
                if lvl == _len_srcs:
                    src_extra = self.input_proj[lvl](features_dict[_len_srcs - 1].tensors)
                else:
                    src_extra = self.input_proj[lvl](srcs[-1])
                m = mask
                mask_extra = F.interpolate(m[None].float(), size=src_extra.shape[-2:]).to(torch.bool)[0]
                pos_extra = self.pos_embed(src_extra, mask_extra).to(src_extra.dtype)
                srcs.append(src_extra)
                masks_list.append(mask_extra)
                poss.append(pos_extra)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src_l, mask_l, pos_l) in enumerate(zip(srcs, masks_list, poss)):
            bs, c, h, w = src_l.shape
            spatial_shapes.append((h, w))
            src_flatten.append(src_l.flatten(2).transpose(1, 2))
            mask_flatten.append(mask_l.flatten(1))
            if self.level_embed is not None:
                lvl_pos = pos_l.flatten(2).transpose(1, 2) + self.level_embed[lvl].view(1, 1, -1)
            else:
                lvl_pos = pos_l.flatten(2).transpose(1, 2)
            lvl_pos_embed_flatten.append(lvl_pos)

        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes_t = torch.as_tensor(spatial_shapes, dtype=torch.long)
        level_start_index = torch.cat((
            spatial_shapes_t.new_zeros((1,)),
            spatial_shapes_t.prod(1).cumsum(0)[:-1],
        ))

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks_list], 1)
        reference_points = self.get_reference_points(
            spatial_shapes_t, valid_ratios, device=src_flatten.device,
        )

        return {
            "src_flatten": src_flatten,
            "pos_flatten": lvl_pos_embed_flatten,
            "spatial_shapes": spatial_shapes_t,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            "mask_flatten": mask_flatten,
            "reference_points": reference_points,
        }

# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
ED-Pose backbone (Swin-L) + input projection + position encoding on CPU.

Runs Swin-L backbone and input projection on CPU (PyTorch), producing the
flattened multi-scale features, positional embeddings, spatial shapes, and
level start indices that feed directly into the ttnn deformable encoder.

ED-Pose Swin-L 5-scale config:
  - backbone: swin_L_384_22k (embed_dim=192, depths=[2,2,18,2], window_size=12)
  - return_interm_indices: [0,1,2,3]  (all 4 Swin stages)
  - num_feature_levels: 5  (4 backbone + 1 extra via Conv3x3 stride=2)
  - position_embedding: sine (temperatureH=20, temperatureW=20)
  - two_stage_type: standard
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


class PositionEmbeddingSineHW(nn.Module):
    """Sine position embedding with separate H/W temperatures (no external deps)."""

    def __init__(self, num_pos_feats=128, temperatureH=20, temperatureW=20, normalize=True, scale=2 * math.pi):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        self.scale = scale

    @torch.no_grad()
    def forward(self, tensor, mask):
        """
        Args:
            tensor: (N, C, H, W)
            mask: (N, H, W) bool — True = padding
        Returns:
            pos: (N, C, H, W) where C = 2 * num_pos_feats
        """
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=tensor.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class EDPoseBackbone:
    """
    Loads Swin-L backbone + input_proj from ED-Pose checkpoint.
    Runs on CPU to produce encoder-ready tensors.

    Usage:
        backbone = EDPoseBackbone(checkpoint_path)
        encoder_inputs = backbone(image_tensor, mask)
    """

    def __init__(self, checkpoint_path=None, device="cpu"):
        _ensure_edpose_on_path()

        if checkpoint_path is None:
            checkpoint_path = os.path.join(EDPOSE_ROOT, "weights", "edpose_swinl_5scale_coco.pth")

        self.device = device
        self.d_model = 256
        self.num_feature_levels = 5

        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = ckpt.get("model_ema", ckpt.get("model", ckpt))
        self.full_sd = {k.replace("module.", ""): v for k, v in state_dict.items()}

        self._build_swin()
        self._build_input_proj()
        self._build_position_embedding()
        self._load_level_embed()

    def _build_swin(self):
        _ensure_edpose_on_path()
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "swin_transformer",
            os.path.join(EDPOSE_ROOT, "models", "edpose", "backbones", "swin_transformer.py"),
        )
        swin_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(swin_mod)
        build_swin_transformer = swin_mod.build_swin_transformer

        self.swin = build_swin_transformer(
            "swin_L_384_22k",
            pretrain_img_size=384,
            out_indices=(0, 1, 2, 3),
            dilation=False,
            use_checkpoint=False,
        )

        swin_sd = {}
        prefix = "backbone.0."
        for k, v in self.full_sd.items():
            if k.startswith(prefix):
                swin_sd[k[len(prefix):]] = v

        missing, unexpected = self.swin.load_state_dict(swin_sd, strict=False)
        if missing:
            print(f"Swin-L: {len(missing)} missing keys (first 5: {missing[:5]})")
        self.swin.eval()
        self.swin.to(self.device)

    def _build_input_proj(self):
        backbone_channels = self.swin.num_features  # [192, 384, 768, 1536]
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

        proj_sd = {}
        for k, v in self.full_sd.items():
            if k.startswith("input_proj."):
                proj_sd[k[len("input_proj."):]] = v

        self.input_proj.load_state_dict(proj_sd, strict=True)
        self.input_proj.eval()
        self.input_proj.to(self.device)

    def _build_position_embedding(self):
        self.pos_embed = PositionEmbeddingSineHW(
            num_pos_feats=self.d_model // 2,
            temperatureH=20,
            temperatureW=20,
            normalize=True,
        )

    def _load_level_embed(self):
        self.level_embed = self.full_sd.get("transformer.level_embed", None)
        if self.level_embed is not None:
            self.level_embed = self.level_embed.to(self.device)

    @staticmethod
    def get_valid_ratio(mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

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

    @torch.no_grad()
    def __call__(self, image_tensor, mask):
        """
        Run backbone + input_proj + position encoding.

        Args:
            image_tensor: (N, 3, H, W) normalized image
            mask: (N, H, W) bool — True = padding pixels

        Returns dict with:
            src_flatten: (N, sum(Hi*Wi), 256) — flattened multi-scale features
            pos_flatten: (N, sum(Hi*Wi), 256) — positional embeddings with level_embed
            spatial_shapes: (num_levels, 2) — (H, W) per level
            level_start_index: (num_levels,)
            valid_ratios: (N, num_levels, 2)
            mask_flatten: (N, sum(Hi*Wi)) — bool
            reference_points: (N, sum(Hi*Wi), num_levels, 2) — for encoder self-attn
            backbone_features: list of (N, C_i, H_i, W_i) — raw Swin outputs (for extra level)
        """
        _ensure_edpose_on_path()
        from util.misc import NestedTensor  # noqa: E402

        nested = NestedTensor(image_tensor, mask)
        features_dict = self.swin(nested)

        srcs = []
        masks = []
        poss = []

        for lvl in range(len(features_dict)):
            feat = features_dict[lvl]
            src_l, mask_l = feat.tensors, feat.mask
            srcs.append(self.input_proj[lvl](src_l))
            masks.append(mask_l)
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
                masks.append(mask_extra)
                poss.append(pos_extra)

        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []

        for lvl, (src_l, mask_l, pos_l) in enumerate(zip(srcs, masks, poss)):
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
        spatial_shapes_t = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes_t.new_zeros((1,)), spatial_shapes_t.prod(1).cumsum(0)[:-1]))

        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        reference_points = self.get_reference_points(spatial_shapes_t, valid_ratios, device=src_flatten.device)

        return {
            "src_flatten": src_flatten,
            "pos_flatten": lvl_pos_embed_flatten,
            "spatial_shapes": spatial_shapes_t,
            "level_start_index": level_start_index,
            "valid_ratios": valid_ratios,
            "mask_flatten": mask_flatten,
            "reference_points": reference_points,
        }

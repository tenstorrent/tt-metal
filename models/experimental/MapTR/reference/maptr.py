# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Consolidated MapTR module combining detector, head, transformer, decoder, and builder

import copy
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import normal_
from torchvision.transforms.functional import rotate
from typing import List

from models.experimental.MapTR.reference.dependency import (
    DETECTORS,
    HEADS,
    TRANSFORMER,
    TRANSFORMER_LAYER_SEQUENCE,
    MVXTwoStageDetector,
    DETRHead,
    BaseModule,
    TransformerLayerSequence,
    Registry,
    build_loss,
    build_bbox_coder,
    build_transformer,
    build_transformer_layer_sequence,
    force_fp32,
    auto_fp16,
    Linear,
    bias_init_with_prob,
    inverse_sigmoid,
    LearnedPositionalEncoding,
    xavier_init,
    Voxelization,
    DynamicScatter,
    builder,
)

from models.experimental.MapTR.reference.utils import (
    GridMask,
    bbox_xyxy_to_cxcywh,
    denormalize_2d_pts,
)

# Import BEVFormer modules from consolidated bevformer.py
from models.experimental.MapTR.reference.bevformer import (
    TemporalSelfAttention,
    MSDeformableAttention3D,
    CustomMSDeformableAttention,
)


# ========== FUSERS Registry ==========
FUSERS = Registry("fusers")


def build_fuser(cfg):
    return FUSERS.build(cfg)


@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))


# ========== MapTR Decoder ==========
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class MapTRDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer."""

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(MapTRDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.fp16_enabled = False

    def forward(self, query, *args, reference_points=None, reg_branches=None, key_padding_mask=None, **kwargs):
        """Forward function for `Detr3DTransformerDecoder`."""
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            reference_points_input = reference_points[..., :2].unsqueeze(2)
            output = layer(
                output, *args, reference_points=reference_points_input, key_padding_mask=key_padding_mask, **kwargs
            )
            output = output.permute(1, 0, 2)

            if reg_branches is not None:
                tmp = reg_branches[lid](output)
                assert reference_points.shape[-1] == 2
                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


# ========== MapTR Perception Transformer ==========
@TRANSFORMER.register_module()
class MapTRPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer."""

    def __init__(
        self,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        fuser=None,
        encoder=None,
        decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        len_can_bus=18,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        modality="vision",
        **kwargs,
    ):
        super(MapTRPerceptionTransformer, self).__init__(**kwargs)
        if modality == "fusion":
            self.fuser = build_fuser(fuser)
        self.use_attn_bev = encoder["type"] == "BEVFormerEncoder"
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.len_can_bus = len_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(torch.Tensor(self.num_cams, self.embed_dims))
        self.feat_proj = None
        self.reference_points = nn.Linear(self.embed_dims, 2)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(self.len_can_bus, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module("norm", nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if (
                isinstance(m, MSDeformableAttention3D)
                or isinstance(m, TemporalSelfAttention)
                or isinstance(m, CustomMSDeformableAttention)
            ):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution="uniform", bias=0.0)
        xavier_init(self.can_bus_mlp, distribution="uniform", bias=0.0)

    def attn_bev_encode(
        self, mlvl_feats, bev_queries, bev_h, bev_w, grid_length=[0.512, 0.512], bev_pos=None, prev_bev=None, **kwargs
    ):
        bs = mlvl_feats[0].size(0)
        bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
        bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

        delta_x = np.array([each["can_bus"][0] if len(each["can_bus"]) > 0 else 0.0 for each in kwargs["img_metas"]])
        delta_y = np.array([each["can_bus"][1] if len(each["can_bus"]) > 1 else 0.0 for each in kwargs["img_metas"]])
        ego_angle = np.array(
            [each["can_bus"][-2] / np.pi * 180 if len(each["can_bus"]) >= 2 else 0.0 for each in kwargs["img_metas"]]
        )
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift_array = np.stack([shift_x, shift_y], axis=0)
        shift = torch.from_numpy(shift_array).to(bev_queries.device, dtype=bev_queries.dtype)
        if shift.dim() == 2:
            shift = shift.permute(1, 0)
        else:
            shift = shift.view(2, -1).permute(1, 0)

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = prev_bev.permute(1, 0, 2)
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = (
                        kwargs["img_metas"][i]["can_bus"][-1] if len(kwargs["img_metas"][i]["can_bus"]) > 0 else 0.0
                    )
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        can_bus_list = []
        for each in kwargs["img_metas"]:
            cb = each.get("can_bus", [0.0] * self.len_can_bus)
            if isinstance(cb, np.ndarray):
                cb = cb.tolist()
            elif not isinstance(cb, (list, tuple)):
                cb = [float(cb)] if isinstance(cb, (int, float)) else [0.0]
            else:
                cb = list(cb)
            if len(cb) < self.len_can_bus:
                cb = cb + [0.0] * (self.len_can_bus - len(cb))
            elif len(cb) > self.len_can_bus:
                cb = cb[: self.len_can_bus]
            can_bus_list.append(cb)
        can_bus_array = np.array(can_bus_list, dtype=np.float32)
        can_bus = torch.from_numpy(can_bus_array).to(device=bev_queries.device, dtype=bev_queries.dtype)
        can_bus = self.can_bus_mlp(can_bus[:, : self.len_can_bus])[None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            if feat.shape[-1] != self.embed_dims:
                if self.feat_proj is None or (
                    hasattr(self.feat_proj, "in_features") and self.feat_proj.in_features != feat.shape[-1]
                ):
                    self.feat_proj = nn.Linear(feat.shape[-1], self.embed_dims).to(feat.device, dtype=feat.dtype)
                feat = self.feat_proj(feat)
            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl : lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(0, 2, 1, 3)

        bev_embed = self.encoder(
            bev_queries,
            feat_flatten,
            feat_flatten,
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=bev_pos,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            prev_bev=prev_bev,
            shift=shift,
            **kwargs,
        )
        return bev_embed

    def lss_bev_encode(self, mlvl_feats, prev_bev=None, **kwargs):
        assert len(mlvl_feats) == 1, "Currently we only support single level feat in LSS"
        images = mlvl_feats[0]
        img_metas = kwargs["img_metas"]
        bev_embed = self.encoder(images, img_metas)
        bs, c, _, _ = bev_embed.shape
        bev_embed = bev_embed.view(bs, c, -1).permute(0, 2, 1).contiguous()
        return bev_embed

    def get_bev_features(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        """obtain bev features."""
        if self.use_attn_bev:
            bev_embed = self.attn_bev_encode(
                mlvl_feats,
                bev_queries,
                bev_h,
                bev_w,
                grid_length=grid_length,
                bev_pos=bev_pos,
                prev_bev=prev_bev,
                **kwargs,
            )
        else:
            bev_embed = self.lss_bev_encode(mlvl_feats, prev_bev=prev_bev, **kwargs)
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0, 3, 1, 2).contiguous()
            lidar_feat = lidar_feat.permute(0, 1, 3, 2).contiguous()
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h, bev_w), mode="bicubic", align_corners=False)
            fused_bev = self.fuser([bev_embed, lidar_feat])
            fused_bev = fused_bev.flatten(2).permute(0, 2, 1).contiguous()
            bev_embed = fused_bev

        return bev_embed

    def forward(
        self,
        mlvl_feats,
        lidar_feat,
        bev_queries,
        object_query_embed,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        prev_bev=None,
        **kwargs,
    ):
        """Forward function for `Detr3DTransformer`."""
        bev_embed = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )

        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
        query = query.unsqueeze(0).expand(bs, -1, -1)
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        debug_enabled = os.environ.get("MAPTR_DEBUG_EVAL", "0") == "1"
        if debug_enabled:
            print(f"\n=== Transformer Reference Points Initialization ===")
            print(f"reference_points (after sigmoid) shape: {reference_points.shape}")
            print(f"reference_points range: [{reference_points.min():.4f}, {reference_points.max():.4f}]")
            print(f"First 5 reference points: {reference_points[0, :5]}")
            print(f"Reference points should be in normalized [0, 1] range")

        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        bev_embed = bev_embed.permute(1, 0, 2)

        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
            level_start_index=torch.tensor([0], device=query.device),
            **kwargs,
        )

        inter_references_out = inter_references

        return bev_embed, inter_states, init_reference_out, inter_references_out


# ========== MapTR Head ==========
@HEADS.register_module()
class MapTRHead(DETRHead):
    """Head of Detr3D."""

    def __init__(
        self,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        num_vec=20,
        num_pts_per_vec=2,
        num_pts_per_gt_vec=2,
        query_embed_type="all_pts",
        transform_method="minmax",
        gt_shift_pts_pattern="v0",
        dir_interval=1,
        loss_pts=None,
        loss_dir=None,
        **kwargs,
    ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.bev_encoder_type = transformer.encoder.type
        if self.as_two_stage:
            transformer["as_two_stage"] = self.as_two_stage
        if "code_size" in kwargs:
            self.code_size = kwargs["code_size"]
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.query_embed_type = query_embed_type
        self.transform_method = transform_method
        self.gt_shift_pts_pattern = gt_shift_pts_pattern
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.dir_interval = dir_interval

        if transformer is not None and isinstance(transformer, dict):
            if "embed_dims" in transformer:
                self.embed_dims = transformer["embed_dims"]
            if "decoder" in transformer and isinstance(transformer["decoder"], dict):
                if "num_layers" in transformer["decoder"]:
                    self._decoder_num_layers = transformer["decoder"]["num_layers"]
                else:
                    self._decoder_num_layers = None
            else:
                self._decoder_num_layers = None
        else:
            self._decoder_num_layers = None

        super(MapTRHead, self).__init__(*args, transformer=transformer, **kwargs)

        if transformer is not None and isinstance(transformer, dict):
            if not hasattr(self, "transformer") or self.transformer is None:
                self.transformer = build_transformer(transformer)

        if not hasattr(self, "embed_dims") or self.embed_dims is None:
            if hasattr(self, "in_channels"):
                self.embed_dims = self.in_channels
            else:
                raise AttributeError("embed_dims not found in transformer config or in_channels")
        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        if loss_pts is not None:
            self.loss_pts = build_loss(loss_pts)
        else:
            self.loss_pts = None
        if loss_dir is not None:
            self.loss_dir = build_loss(loss_dir)
        else:
            self.loss_dir = None
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        if hasattr(self, "transformer") and hasattr(self.transformer, "decoder"):
            num_layers = self.transformer.decoder.num_layers
        elif hasattr(self, "_decoder_num_layers") and self._decoder_num_layers is not None:
            num_layers = self._decoder_num_layers
        else:
            raise AttributeError("num_layers not found in transformer decoder")
        num_pred = (num_layers + 1) if self.as_two_stage else num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            if self.bev_encoder_type == "BEVFormerEncoder":
                self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
                self.positional_encoding = LearnedPositionalEncoding(
                    self.embed_dims // 2, row_num_embed=self.bev_h, col_num_embed=self.bev_w
                )
            else:
                self.bev_embedding = None
                self.positional_encoding = None
            if self.query_embed_type == "all_pts":
                self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            elif self.query_embed_type == "instance_pts":
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    @force_fp32(apply_to=("mlvl_feats", "prev_bev"))
    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
        """Forward function."""
        batch_id = str(img_metas[0].get("sample_idx", "unknown")) if img_metas else "unknown"
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype

        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)
            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
            if self.positional_encoding is not None:
                bev_pos = self.positional_encoding(bev_mask).to(dtype)
            else:
                bev_pos = None
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        if only_bev:
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        else:
            outputs = self.transformer(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)

        if hasattr(self, "_debug_enabled") and self._debug_enabled:
            print(f"\n=== Query Initialization Debug ===")
            print(f"Query embeds shape: {object_query_embeds.shape}")
            print(f"Query embeds range: [{object_query_embeds.min():.4f}, {object_query_embeds.max():.4f}]")
            if init_reference is not None:
                print(f"Init reference points shape: {init_reference.shape}")
                print(
                    f"Init reference points (normalized) range: [{init_reference.min():.4f}, {init_reference.max():.4f}]"
                )
                print(f"First 5 reference points (normalized): {init_reference[0, :5]}")
                ref_pts_real = denormalize_2d_pts(init_reference[0].view(-1, 2), self.pc_range)
                print(
                    f"Reference points (denormalized) range: X[{ref_pts_real[:, 0].min():.2f}, {ref_pts_real[:, 0].max():.2f}], Y[{ref_pts_real[:, 1].min():.2f}, {ref_pts_real[:, 1].max():.2f}]"
                )
                print(f"First 5 reference points (real): {ref_pts_real[:5]}")
                print(f"PC range: {self.pc_range}")

        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])

            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()

            if hasattr(self, "_debug_enabled") and self._debug_enabled and lvl == hs.shape[0] - 1:
                print(f"\n=== Prediction Head Output (Layer {lvl}) ===")
                print(f"Raw coords (tmp) shape: {tmp.shape}")
                print(f"Raw coords (normalized) range: [{tmp.min():.4f}, {tmp.max():.4f}]")
                print(f"First 3 normalized coords: {tmp[0, :3, :2]}")
                denorm_pts = denormalize_2d_pts(tmp.view(tmp.shape[0], -1, 2), self.pc_range)
                print(
                    f"Denorm coords range: X[{denorm_pts[:, 0].min():.2f}, {denorm_pts[:, 0].max():.2f}], Y[{denorm_pts[:, 1].min():.2f}, {denorm_pts[:, 1].max():.2f}]"
                )
                print(f"First 3 denormalized coords: {denorm_pts[:3]}")
                print(f"PC range: {self.pc_range}")

            outputs_coord, outputs_pts_coord = self.transform_box(tmp)
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_pts_preds": outputs_pts_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "enc_pts_preds": None,
        }

        return outs

    def transform_box(self, pts, y_first=False):
        """Converting the points set into bounding box."""
        pts_reshape = pts.view(pts.shape[0], self.num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == "minmax":
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def loss(self, *args, **kwargs):
        raise NotImplementedError("MapTRHead training / loss computation has been removed in this reference build.")

    @force_fp32(apply_to=("preds_dicts"))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions."""
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list


# ========== MapTR Detector ==========
@DETECTORS.register_module()
class MapTR(MVXTwoStageDetector):
    """MapTR."""

    def __init__(
        self,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        modality="vision",
        lidar_encoder=None,
    ):
        super(MapTR, self).__init__(
            pts_voxel_layer,
            pts_voxel_encoder,
            pts_middle_encoder,
            pts_fusion_layer,
            img_backbone,
            pts_backbone,
            img_neck,
            pts_neck,
            pts_bbox_head,
            img_roi_head,
            img_rpn_head,
            train_cfg,
            test_cfg,
            pretrained,
        )
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.modality = modality
        if self.modality == "fusion" and lidar_encoder is not None:
            if lidar_encoder["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**lidar_encoder["voxelize"])
            else:
                voxelize_module = DynamicScatter(**lidar_encoder["voxelize"])
            self.lidar_modal_extractor = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": builder.build_middle_encoder(lidar_encoder["backbone"]),
                }
            )
            self.voxelize_reduce = lidar_encoder.get("voxelize_reduce", True)

    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        if img is None:
            return None
        if img.dim() == 4:
            img = img.unsqueeze(0)
        B = img.size(0)
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)

        img = img.to(dtype=next(self.img_backbone.parameters()).dtype)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=("img"), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        return img_feats

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether return_loss=True."""
        if return_loss:
            raise NotImplementedError(
                "MapTR training is disabled in this build. Call with return_loss=False for inference only."
            )

        aug_data = kwargs.pop("aug_data", None)
        if aug_data is not None:
            while isinstance(aug_data, (list, tuple)) and len(aug_data) > 0:
                if isinstance(aug_data[0], dict):
                    kwargs.update(aug_data[0])
                    break
                aug_data = aug_data[0]

        img = kwargs.get("img", None)
        if img is not None:
            if hasattr(img, "data"):
                img = img.data[0] if len(img.data) > 0 else None
            elif isinstance(img, (list, tuple)) and len(img) > 0 and hasattr(img[0], "data"):
                img = img[0].data[0] if len(img[0].data) > 0 else None
        kwargs["img"] = img

        img_metas = kwargs.pop("img_metas", None)
        if img_metas is not None:
            if hasattr(img_metas, "data"):
                img_metas = img_metas.data
            elif isinstance(img_metas, (list, tuple)) and len(img_metas) > 0 and hasattr(img_metas[0], "data"):
                img_metas = img_metas[0].data

        if img_metas is None or (isinstance(img_metas, (list, tuple)) and len(img_metas) == 0):
            num_cams = 6
            if img is not None and isinstance(img, torch.Tensor):
                num_cams = img.shape[1] if img.dim() == 5 else img.shape[0]
            lidar2img = kwargs.get("lidar2img", [np.eye(4, dtype=np.float32) for _ in range(num_cams)])
            can_bus = kwargs.get("can_bus", np.zeros(18, dtype=np.float32))
            img_metas = [
                {
                    "scene_token": kwargs.get("scene_token", ""),
                    "can_bus": can_bus,
                    "sample_idx": kwargs.get("sample_idx", None),
                    "lidar2img": lidar2img,
                }
            ]

        if not isinstance(img_metas, list):
            img_metas = [img_metas]
        if len(img_metas) > 0 and not isinstance(img_metas[0], list):
            img_metas = [img_metas]

        return self.forward_test(img_metas, **kwargs)

    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively."""
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs * len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                if not img_metas[0]["prev_bev_exists"]:
                    prev_bev = None
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(img_feats, None, img_metas, prev_bev, only_bev=True)
            return prev_bev

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.lidar_modal_extractor["voxelize"](res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("points"), out_fp32=True)
    def extract_lidar_feat(self, points):
        feats, coords, sizes = self.voxelize(points)
        batch_size = coords[-1, 0] + 1
        lidar_feat = self.lidar_modal_extractor["backbone"](feats, coords, batch_size, sizes=sizes)
        return lidar_feat

    def forward_test(self, img_metas, img=None, points=None, **kwargs):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if not isinstance(img, list) else img
        points = [points] if not isinstance(points, list) else points

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0]["can_bus"][-1] = 0
            img_metas[0][0]["can_bus"][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], points[0], prev_bev=self.prev_frame_info["prev_bev"], **kwargs
        )
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev
        return bbox_results

    def pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays."""
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu(), pts_3d=pts.to("cpu")
        )

        if attrs is not None:
            result_dict["attrs_3d"] = attrs.cpu()

        return result_dict

    def simple_test_pts(self, x, lidar_feat, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, lidar_feat, img_metas, prev_bev=prev_bev)
        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)
        bbox_results = [self.pred2result(bboxes, scores, labels, pts) for bboxes, scores, labels, pts in bbox_list]
        return outs["bev_embed"], bbox_results

    def simple_test(self, img_metas, img=None, points=None, prev_bev=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        lidar_feat = None
        if self.modality == "fusion":
            lidar_feat = self.extract_lidar_feat(points)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(img_feats, lidar_feat, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict["pts_bbox"] = pts_bbox
        return new_prev_bev, bbox_list

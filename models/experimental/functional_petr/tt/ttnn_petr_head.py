# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import math
from models.experimental.functional_petr.tt.ttnn_positional_encoding import ttnn_SinePositionalEncoding3D
from models.experimental.functional_petr.tt.ttnn_petr_transformer import TTPETRTransformer
from models.experimental.functional_petr.reference.nms_free_coder import NMSFreeCoder
from loguru import logger

from models.experimental.functional_petr.tt.utils import inverse_sigmoid as ttnn_inverse_sigmoid

from models.experimental.functional_petr.tt.common import Conv, Conv_with_split

import numpy as np
import torch.nn.functional as F


def ttnn_pos2posemb3d(pos, num_pos_feats=128, temperature=10000, device=None):  # input size of pos =(900,3)
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = ttnn.arange(end=num_pos_feats, dtype=ttnn.bfloat16, device=device)
    dim_t = ttnn.to_layout(dim_t, layout=ttnn.TILE_LAYOUT)
    dim_t = ttnn.reshape(dim_t, (1, -1))

    dim_t = ttnn.div(dim_t, 2)
    dim_t = ttnn.floor(dim_t)
    dim_t = 2 * ttnn.div(dim_t, num_pos_feats)

    # TORCH
    dim_t = temperature ** ttnn.to_torch(dim_t)
    pos = ttnn.to_layout(pos, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos_x = ttnn.to_torch(pos[..., 0:1]) / dim_t
    pos_y = ttnn.to_torch(pos[..., 1:2]) / dim_t
    pos_z = ttnn.to_torch(pos[..., 2:3]) / dim_t

    # Convert back to ttnn
    pos_x = ttnn.from_torch(pos_x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pos_y = ttnn.from_torch(pos_y, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    pos_z = ttnn.from_torch(pos_z, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    # TORCH
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


class ttnn_PETRHead:
    def __init__(
        self,
        num_classes,
        in_channels,
        num_query=100,
        num_reg_fcs=2,
        sync_cls_avg_factor=False,
        positional_encoding=dict(type="SinePositionalEncoding3D", num_feats=128, normalize=True),
        code_weights=None,
        with_position=True,
        with_multiview=False,
        depth_step=0.8,
        depth_num=64,
        LID=False,
        depth_start=1,
        position_range=[-65, -65, -8.0, 65, 65, 8.0],
        parameters=None,
        device=None,
        query_embedding_input=None,
    ):
        self.device = device
        self.code_size = 10
        self.query_embedding_input = query_embedding_input
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        self.code_weights = self.code_weights[: self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview
        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should" f" be exactly 2 times of num_feats. Found {self.embed_dims}" f" and {num_feats}."
        )
        self.parameters = parameters
        self.num_pred = 6
        self.cls_out_channels = num_classes
        self._init_layers(self.parameters)
        self.positional_encoding = ttnn_SinePositionalEncoding3D(num_feats=128, normalize=True)
        self.transformer = TTPETRTransformer(device=device, parameter=parameters["transformer"])
        self.bbox_coder = NMSFreeCoder(
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],  # self.point_cloud_range,
            max_num=300,
            voxel_size=[0.2, 0.2, 8],  # self.voxel_size,
            num_classes=10,
        )
        self.pc_range = self.bbox_coder.pc_range

    def _init_layers(self, parameters):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv([1, 1, 0, 0], parameters=parameters["input_proj"])
        else:
            self.input_proj = Conv([1, 1, 0, 0], parameters=parameters["input_proj"])

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(ttnn.linear)
            cls_branch.append(ttnn.layer_norm)
            cls_branch.append(ttnn.relu)

        cls_branch.append(ttnn.linear)
        fc_cls = cls_branch

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(ttnn.linear)
            reg_branch.append(ttnn.relu)
        reg_branch.append(ttnn.linear)

        # reg_branch.append(*reg_branch)

        self.cls_branches = [fc_cls for _ in range(self.num_pred)]
        self.reg_branches = [reg_branch for _ in range(self.num_pred)]

        if self.with_multiview:
            self.adapt_pos3d = [
                Conv_with_split([1, 1, 0, 0], parameters=parameters["adapt_pos3d"][0]),
                ttnn.relu,
                Conv([1, 1, 0, 0], parameters=parameters["adapt_pos3d"][2]),
            ]
        else:
            self.adapt_pos3d = [
                Conv([1, 1, 0, 0], parameters=parameters["adapt_pos3d"][0]),
                ttnn.relu,
                Conv([1, 1, 0, 0], parameters=parameters["adapt_pos3d"][2]),
            ]

        if self.with_position:
            self.position_encoder = [
                Conv([1, 1, 0, 0], parameters=parameters["position_encoder"][0]),
                ttnn.relu,
                Conv([1, 1, 0, 0], parameters=parameters["position_encoder"][2], height_sharding=False),
            ]

        self.reference_points = ttnn.embedding
        self.query_embedding = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
        ]

    def position_embeding(self, img_feats, img_metas, masks=None, device=None):
        eps = 1e-5
        pad_h, pad_w = img_metas[0]["pad_shape"]
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = ttnn.arange(0, H, 1, dtype=ttnn.bfloat16, device=device)
        coords_h = ttnn.to_layout(coords_h, ttnn.TILE_LAYOUT)
        coords_h = ttnn.div(coords_h * pad_h, H)
        coords_w = ttnn.arange(0, W, 1, dtype=ttnn.bfloat16, device=device)
        coords_w = ttnn.to_layout(coords_w, ttnn.TILE_LAYOUT)
        coords_w = ttnn.div(coords_w * pad_w, W)

        if self.LID:
            index = ttnn.arange(start=0, end=self.depth_num, step=1, dtype=ttnn.bfloat16, device=device)
            index = ttnn.to_layout(index, ttnn.TILE_LAYOUT)
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = ttnn.arange(start=0, end=self.depth_num, step=1, dtype=ttnn.bfloat16, device=device)
            index = ttnn.to_layout(index, ttnn.TILE_LAYOUT)
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        coords_h = ttnn.to_layout(coords_h, ttnn.ROW_MAJOR_LAYOUT)
        coords_w = ttnn.to_layout(coords_w, ttnn.ROW_MAJOR_LAYOUT)
        coords_d = ttnn.to_layout(coords_d, ttnn.ROW_MAJOR_LAYOUT)

        coords_h = ttnn.reshape(coords_h, [-1])
        coords_w = ttnn.reshape(coords_w, [-1])
        coords_d = ttnn.reshape(coords_d, [-1])

        coords_h = ttnn.to_torch(coords_h)
        coords_w = ttnn.to_torch(coords_w)
        coords_d = ttnn.to_torch(coords_d)

        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0)  # W, H, D, 3

        coords = ttnn.from_torch(coords, device=device)

        ones = ttnn.ones_like(coords[..., :1], dtype=ttnn.bfloat16)
        coords = ttnn.concat((coords, ones), dim=-1)
        coords = ttnn.to_layout(coords, ttnn.ROW_MAJOR_LAYOUT)
        torch_coords = ttnn.to_torch(coords)
        torch_coords[..., :2] = torch_coords[..., :2] * torch.maximum(
            torch_coords[..., 2:3], torch.ones_like(torch_coords[..., 2:3]) * eps
        )

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta["lidar2img"])):
                img2lidar.append(np.linalg.inv(img_meta["lidar2img"][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = torch_coords.new_tensor(img2lidars)

        W, H, D = torch_coords.shape[0], torch_coords.shape[1], torch_coords.shape[2]

        torch_coords = torch_coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, torch_coords).squeeze(-1)[..., :3]

        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0]
        )
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1]
        )
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2]
        )

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B * N, -1, H, W)
        coords3d = ttnn.from_torch(coords3d, layout=ttnn.TILE_LAYOUT, device=device)
        coords3d = ttnn_inverse_sigmoid(coords3d)

        coords_position_embeding = coords3d
        coords_position_embeding = ttnn.permute(coords_position_embeding, (0, 2, 3, 1))
        for i in self.position_encoder:
            if i != ttnn.relu:
                coords_position_embeding = i(device, coords_position_embeding)
            else:
                coords_position_embeding = i(coords_position_embeding)
        coords_position_embeding = ttnn.permute(coords_position_embeding, (0, 3, 1, 2))

        return ttnn.reshape(coords_position_embeding, (B, N, self.embed_dims, H, W)), coords_mask

    def __call__(self, mlvl_feats, img_metas, device=None):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format \
                (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        for i in range(len(mlvl_feats)):
            mlvl_feats[i] = ttnn.to_memory_config(mlvl_feats[i], memory_config=ttnn.DRAM_MEMORY_CONFIG)

        x = mlvl_feats[0]
        batch_size, num_cams = x.shape[0], x.shape[1]
        input_img_h, input_img_w = img_metas[0]["pad_shape"]
        x = ttnn.to_torch(x)
        masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))
        x = ttnn.from_torch(x, device=device)
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w = img_metas[img_id]["img_shape"][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        x = self.input_proj(
            device, ttnn.permute(ttnn.reshape(x, (-1, x.shape[2], x.shape[3], x.shape[4])), (0, 2, 3, 1))
        )
        x = ttnn.permute(x, (0, 3, 1, 2))  # converting from NHWC ot NCHW
        x = ttnn.reshape(x, (batch_size, num_cams, x.shape[-3], x.shape[-2], x.shape[-1]))
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=ttnn.to_torch(x).shape[-2:]).to(dtype=torch.bool)

        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks, device=device)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                masks = masks.to(dtype=torch.float16)
                masks = ttnn.from_torch(masks, layout=ttnn.TILE_LAYOUT, device=device)
                sin_embed = self.positional_encoding(masks)
                # sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1))
                masks = ttnn.to_torch(masks)
                masks = masks
                sin_embed = ttnn.permute(
                    ttnn.reshape(sin_embed, (-1, sin_embed.shape[2], sin_embed.shape[3], sin_embed.shape[4])),
                    (0, 2, 3, 1),
                )
                for i in self.adapt_pos3d:
                    if i == ttnn.relu:
                        sin_embed = i(sin_embed)
                    else:
                        sin_embed = i(device, sin_embed)
                sin_embed = ttnn.permute(sin_embed, (0, 3, 1, 2))
                sin_embed = ttnn.reshape(sin_embed, (x.shape[0], x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
                # .view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                # This is not invoked in our run
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            # This is not invoked in our run
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.parameters.reference_points.weight
        ref_check = ttnn.to_torch(reference_points)
        if torch.isnan(ref_check).any() or torch.isinf(ref_check).any():
            logger.error("reference_points contains NaN/Inf at initialization!")
            ref_check = torch.nan_to_num(ref_check, nan=0.5, posinf=1.0, neginf=0.0)
            self.parameters.reference_points.weight = ttnn.from_torch(ref_check, dtype=ttnn.bfloat16, device=device)
        for index, i in enumerate(
            self.query_embedding
        ):  # replaced this by preprocessing pos2posemb3d(reference_points))
            if index == 0:
                query_embeds = i(
                    self.query_embedding_input,
                    self.parameters["query_embedding"][index].weight,
                    bias=self.parameters["query_embedding"][index].bias,
                )
            elif i == ttnn.linear:
                query_embeds = i(
                    query_embeds,
                    self.parameters["query_embedding"][index].weight,
                    bias=self.parameters["query_embedding"][index].bias,
                )
            else:
                query_embeds = i(query_embeds)

        reference_points = ttnn.reshape(reference_points, (1, reference_points.shape[0], reference_points.shape[1]))
        reference_points = ttnn.repeat_interleave(reference_points, batch_size, dim=0)  # .sigmoid()
        masks = masks.to(dtype=torch.float16)
        masks = ttnn.from_torch(masks, device=device)

        outs_dec, _ = self.transformer(device, x, masks, query_embeds, pos_embed)  # , self.reg_branches)

        outs_dec_torch = ttnn.to_torch(outs_dec).to(torch.float32)
        outs_dec_torch = ttnn.to_torch(outs_dec).to(torch.float32)
        if torch.isnan(outs_dec_torch).any() or torch.isinf(outs_dec_torch).any():
            logger.warning(f"NaN/Inf detected in outs_dec! Applying nan_to_num")
            outs_dec_torch = torch.nan_to_num(outs_dec_torch, nan=0.0, posinf=1e6, neginf=-1e6)
            outs_dec = ttnn.from_torch(outs_dec_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            outs_dec = ttnn.to_device(outs_dec, device)
        outs_dec = ttnn.from_torch(outs_dec_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        outs_dec = ttnn.to_device(outs_dec, device)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = ttnn_inverse_sigmoid(ttnn.clone(reference_points))

            ref_torch = ttnn.to_torch(reference)
            if torch.isnan(ref_torch).any() or torch.isinf(ref_torch).any():
                logger.warning(f"Layer {lvl}: NaN/Inf in reference! Fixing...")
                ref_torch = torch.nan_to_num(ref_torch, nan=0.0, posinf=10.0, neginf=-10.0)
                reference = ttnn.from_torch(ref_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

            assert reference.shape[-1] == 3

            outputs_class_f32 = ttnn.from_torch(
                ttnn.to_torch(outs_dec[lvl : lvl + 1]).to(torch.float32),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )

            for index, operation in enumerate(self.cls_branches[lvl]):
                if operation == ttnn.linear:
                    # Keep using ttnn.linear but with float32 tensors
                    outputs_class_f32 = operation(
                        outputs_class_f32,
                        self.parameters["cls_branches"][lvl][index].weight,
                        bias=self.parameters["cls_branches"][lvl][index].bias,
                    )
                elif operation == ttnn.relu:
                    outputs_class_f32 = operation(outputs_class_f32)
                elif operation == ttnn.layer_norm:
                    outputs_class_f32 = operation(
                        outputs_class_f32,
                        weight=self.parameters["cls_branches"][lvl][index].weight,
                        bias=self.parameters["cls_branches"][lvl][index].bias,
                    )

            # Convert back to bfloat16
            outputs_class = ttnn.from_torch(
                ttnn.to_torch(outputs_class_f32).to(torch.bfloat16), dtype=ttnn.bfloat16, device=device
            )

            tmp = outs_dec[lvl : lvl + 1]
            for index, operation in enumerate(self.reg_branches[lvl]):
                if operation == ttnn.linear:
                    tmp = operation(
                        tmp,
                        self.parameters["reg_branches"][lvl][index].weight,
                        bias=self.parameters["reg_branches"][lvl][index].bias,
                    )
                elif operation == ttnn.relu:
                    tmp = operation(tmp)

            tmp_torch = ttnn.to_torch(tmp).to(torch.float32)
            if torch.isnan(tmp_torch).any() or torch.isinf(tmp_torch).any():
                logger.warning(f"Layer {lvl}: NaN/Inf in tmp before sigmoid! Fixing...")
                tmp_torch = torch.nan_to_num(tmp_torch, nan=0.0, posinf=10.0, neginf=-10.0)

            reference = ttnn.to_torch(reference).to(torch.float32)
            tmp_torch[..., 0:2] = tmp_torch[..., 0:2] + reference[..., 0:2]

            # Safety clamp before sigmoid to prevent overflow
            tmp_torch[..., 0:2] = torch.clamp(tmp_torch[..., 0:2], min=-10.0, max=10.0)
            tmp_torch[..., 0:2] = tmp_torch[..., 0:2].sigmoid()

            tmp_torch[..., 4:5] += reference[..., 2:3]
            tmp_torch[..., 4:5] = torch.clamp(tmp_torch[..., 4:5], min=-10.0, max=10.0)
            tmp_torch[..., 4:5] = tmp_torch[..., 4:5].sigmoid()

            # Final NaN check
            if torch.isnan(tmp_torch).any() or torch.isinf(tmp_torch).any():
                logger.warning(f"Layer {lvl}: NaN/Inf in tmp after sigmoid! Fixing...")
                tmp_torch = torch.nan_to_num(tmp_torch, nan=0.0, posinf=1.0, neginf=0.0)

            tmp = ttnn.from_torch(tmp_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            tmp = ttnn.to_device(tmp, device)
            reference = ttnn.from_torch(reference, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
            reference = ttnn.to_device(reference, device)

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = ttnn.concat(outputs_classes, dim=0)
        all_bbox_preds = ttnn.concat(outputs_coords, dim=0)

        all_cls_scores = ttnn.to_torch(all_cls_scores).to(torch.float32)
        all_bbox_preds = ttnn.to_torch(all_bbox_preds).to(torch.float32)

        if torch.isnan(all_bbox_preds).any() or torch.isinf(all_bbox_preds).any():
            logger.error("NaN/Inf detected in all_bbox_preds before scaling!")
            all_bbox_preds = torch.nan_to_num(all_bbox_preds, nan=0.0, posinf=50.0, neginf=-50.0)

        all_bbox_preds[..., 0:1] = all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
        all_bbox_preds[..., 1:2] = all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
        all_bbox_preds[..., 4:5] = all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

        if torch.isnan(all_bbox_preds).any() or torch.isinf(all_bbox_preds).any():
            logger.error("NaN/Inf still present after scaling!")
            all_bbox_preds = torch.nan_to_num(all_bbox_preds, nan=0.0, posinf=51.0, neginf=-51.0)

        all_cls_scores = ttnn.from_torch(all_cls_scores, device=device)
        all_cls_scores = ttnn.to_device(all_cls_scores, device)
        all_bbox_preds = ttnn.from_torch(all_bbox_preds, device=device)
        all_bbox_preds = ttnn.to_device(all_bbox_preds, device)

        outs = {
            "all_cls_scores": all_cls_scores,
            "all_bbox_preds": all_bbox_preds,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }
        return outs

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]["box_type_3d"](bboxes, bboxes.size(-1))
            scores = preds["scores"]
            labels = preds["labels"]
            ret_list.append([bboxes, scores, labels])
        return ret_list

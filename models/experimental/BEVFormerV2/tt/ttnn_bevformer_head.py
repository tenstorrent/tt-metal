# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.BEVFormerV2.tt.ttnn_perception_transformer import TtPerceptionTransformer
from models.experimental.BEVFormerV2.tt.ttnn_utils import inverse_sigmoid


class TtLearnedPositionalEncoding:
    """TTNN implementation of LearnedPositionalEncoding"""

    def __init__(
        self,
        params,
        device,
        num_feats,
        row_num_embed=200,
        col_num_embed=200,
        init_cfg=dict(type="Uniform", layer="Embedding"),
    ):
        self.row_embed = ttnn.embedding
        self.col_embed = ttnn.embedding
        self.params = params
        self.device = device
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def __call__(self, mask):
        _, h, w = mask.shape
        x = ttnn.arange(w, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.arange(h, device=self.device, memory_config=ttnn.L1_MEMORY_CONFIG)
        x_embed = self.col_embed(
            x,
            weight=self.params.col_embed.weight,
            layout=ttnn.TILE_LAYOUT,
        )
        y_embed = self.row_embed(y, weight=self.params.row_embed.weight, layout=ttnn.TILE_LAYOUT)
        x_embed = ttnn.unsqueeze(x_embed, 0)
        x_embed = ttnn.repeat(x_embed, (h, 1, 1))
        y_embed = ttnn.unsqueeze(y_embed, 1)
        y_embed = ttnn.repeat(y_embed, (1, w, 1))

        out = ttnn.concat((x_embed, y_embed), dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(y_embed)
        ttnn.deallocate(x_embed)
        out = ttnn.permute(out, (2, 0, 1))
        out = ttnn.unsqueeze(out, 0)
        out = ttnn.repeat(out, (mask.shape[0], 1, 1, 1))
        pos = out
        return pos


class TtBEVFormerHead:
    """TTNN implementation of BEVFormerHead"""

    def __init__(
        self,
        *args,
        params,
        device,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=100,
        bev_w=100,
        with_box_refine=False,
        as_two_stage=False,
        num_query=900,
        num_classes=10,
        embed_dims=256,
        num_reg_fcs=2,
        encoder_num_layers=6,
        decoder_num_layers=6,
        model_config=None,
        **kwargs,
    ):
        self.params = params
        self.device = device
        self.model_config = model_config or {
            "WEIGHTS_DTYPE": ttnn.bfloat16,
            "ACTIVATIONS_DTYPE": ttnn.bfloat16,
            "MATH_FIDELITY": ttnn.MathFidelity.HiFi4,
        }

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_query = num_query
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_reg_fcs = num_reg_fcs

        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = bbox_coder
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.transformer = TtPerceptionTransformer(
            params.head.transformer,
            params.head.branches,
            device,
            num_feature_levels=4,
            num_cams=6,
            two_stage_num_proposals=300,
            embed_dims=embed_dims,
            encoder_num_layers=encoder_num_layers,
            decoder_num_layers=decoder_num_layers,
            rotate_prev_bev=False,
            use_shift=False,
            use_can_bus=False,
            use_cams_embeds=True,
            decoder=True,
        )

        self.positional_encoding = TtLearnedPositionalEncoding(
            params.head.positional_encoding, device, embed_dims // 2, row_num_embed=bev_h, col_num_embed=bev_w
        )

        if not self.as_two_stage:
            self.bev_embedding = params.head.bev_embedding
            self.query_embedding = params.head.query_embedding

        self.cls_branches = params.head.cls_branches_torch
        self.reg_branches = params.head.reg_branches_torch

        self._init_layers()

    def _init_layers(self):
        pass

    def __call__(self, mlvl_feats, img_metas, prev_bev=None, only_bev=False):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape

        if not self.as_two_stage:
            object_query_embeds = self.query_embedding.weight
        bev_queries = self.bev_embedding.weight

        bev_mask = ttnn.zeros((bs, self.bev_h, self.bev_w), device=self.device, dtype=ttnn.bfloat16)
        bev_pos = self.positional_encoding(bev_mask)
        bev_pos = ttnn.to_layout(bev_pos, layout=ttnn.ROW_MAJOR_LAYOUT)

        if only_bev:
            bev_embed = self.transformer.get_bev_features(
                self.transformer.params,
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            ttnn.deallocate(bev_mask)
            ttnn.deallocate(bev_pos)
            if isinstance(bev_embed, ttnn.Tensor):
                bev_embed = ttnn.to_torch(bev_embed)
            return bev_embed
        else:
            dummy_map_query = ttnn.zeros(
                (1, self.embed_dims * 2), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
            outputs = self.transformer(
                mlvl_feats,
                bev_queries,
                object_query_embeds,
                dummy_map_query,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
            ttnn.deallocate(dummy_map_query)
        ttnn.deallocate(bev_mask)
        ttnn.deallocate(bev_pos)

        bev_embed, hs, init_reference, inter_references = outputs[0], outputs[1], outputs[2], outputs[3]

        num_levels = hs.shape[0]
        references_processed = []
        for lvl in range(num_levels):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            reference = ttnn.to_torch(reference).float()
            references_processed.append(reference)

        hs = ttnn.to_torch(hs).float()
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(num_levels):
            reference = references_processed[lvl]
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])

            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            tmp[..., 1:2] = tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            tmp[..., 4:5] = tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)

        if isinstance(bev_embed, ttnn.Tensor):
            bev_embed = ttnn.to_torch(bev_embed)

        outputs_classes = outputs_classes.float()
        outputs_coords = outputs_coords.float()

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
        }
        return outs

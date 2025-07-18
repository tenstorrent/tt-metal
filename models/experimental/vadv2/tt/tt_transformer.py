# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.experimental.vadv2.tt.tt_encoder import TtBEVFormerEncoder
from models.experimental.vadv2.tt.tt_decoder import TtDetectionTransformerDecoder, TtMapDetectionTransformerDecoder
import numpy as np


class TtVADPerceptionTransformer:
    def __init__(
        self,
        params,
        params_branches,
        device,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        encoder=None,
        decoder=None,
        map_decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        map_num_vec=50,
        map_num_pts_per_vec=10,
        **kwargs,
    ):
        super(TtVADPerceptionTransformer, self).__init__(**kwargs)
        point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        _dim_ = 256
        _pos_dim_ = _dim_ // 2
        _ffn_dim_ = _dim_ * 2
        self.device = device
        self.params = params
        self.params_branches = (params_branches,)
        self.encoder = TtBEVFormerEncoder(
            params.encoder,
            device,
            num_layers=3,
            pc_range=point_cloud_range,
            num_points_in_pillar=4,
            return_intermediate=False,
            embed_dims=_dim_,
            num_heads=4,
            dilation=1,
            kernel_size=(3, 5),
            im2col_step=192,
            feedforward_channels=_ffn_dim_,
            ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"),
        )
        if decoder is not None:
            self.decoder = TtDetectionTransformerDecoder(
                num_layers=3,
                embed_dim=_dim_,
                num_heads=8,
                params=params.decoder,
                params_branches=params_branches,
                device=self.device,
            )
        else:
            self.decoder = None
        if map_decoder is not None:
            self.map_decoder = TtMapDetectionTransformerDecoder(
                num_layers=3,
                embed_dim=_dim_,
                num_heads=8,
                params=params.map_decoder,
                params_branches=params_branches,
                device=self.device,
            )
        else:
            self.map_decoder = None

        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.fp16_enabled = False
        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds
        self.two_stage_num_proposals = two_stage_num_proposals
        self.rotate_center = rotate_center
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec

    def attn_bev_encode(
        self,
        params,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        shift=None,
        can_bus=None,
        **kwargs,
    ):
        bs = mlvl_feats[0].shape[0]

        bev_queries = ttnn.unsqueeze(bev_queries, 1)
        bev_queries = ttnn.repeat(bev_queries, (1, bs, 1))
        bev_pos = ttnn.reshape(bev_pos, [bev_pos.shape[0], bev_pos.shape[1], bev_pos.shape[2] * bev_pos.shape[3]])
        bev_pos = ttnn.permute(bev_pos, (2, 0, 1))
        bev_queries = ttnn.to_torch(bev_queries)
        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each["can_bus"][0] for each in kwargs["img_metas"]])
        delta_y = np.array([each["can_bus"][1] for each in kwargs["img_metas"]])
        ego_angle = np.array([each["can_bus"][-2] / np.pi * 180 for each in kwargs["img_metas"]])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift

        shift = bev_queries.new_tensor([shift_x, shift_y]).permute(1, 0)  # xy, bs -> bs, xy

        shift = ttnn.from_torch(shift, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        if prev_bev is not None:
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = ttnn.permute(prev_bev, (1, 0, 2))
            if self.rotate_prev_bev:
                for i in range(bs):
                    # num_prev_bev = prev_bev.size(1)
                    rotation_angle = kwargs["img_metas"][i]["can_bus"][-1]
                    tmp_prev_bev = prev_bev[:, i]
                    tmp_prev_bev = ttnn.reshape(tmp_prev_bev, (bev_h, bev_w, -1))
                    tmp_prev_bev = ttnn.permute(tmp_prev_bev, (2, 0, 1))
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = ttnn.permute(tmp_prev_bev, (1, 2, 0))
                    tmp_prev_bev = ttnn.reshape(tmp_prev_bev, (bev_h * bev_w, 1, -1))
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = bev_queries.new_tensor([each["can_bus"] for each in kwargs["img_metas"]])
        can_bus = ttnn.from_torch(can_bus, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)
        # [:, :]
        bev_queries = ttnn.from_torch(bev_queries, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device)

        can_bus = can_bus = ttnn.linear(can_bus, params.can_bus_mlp["0"].weight, bias=params.can_bus_mlp["0"].bias)
        can_bus = ttnn.relu(can_bus)
        can_bus = ttnn.linear(can_bus, params.can_bus_mlp["1"].weight, bias=params.can_bus_mlp["1"].bias)
        can_bus = ttnn.relu(can_bus)
        if self.can_bus_norm:
            can_bus = ttnn.layer_norm(
                can_bus,
                weight=self.params.can_bus_mlp.norm.weight,
                bias=self.params.can_bus_mlp.norm.bias,
            )
        can_bus = ttnn.reshape(can_bus, (1, can_bus.shape[0], can_bus.shape[1]))
        # [None, :, :]
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = ttnn.reshape(feat, (feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3] * feat.shape[4]))
            feat = ttnn.permute(feat, (1, 0, 3, 2))
            # ss
            if self.use_cams_embeds:
                cam_embeds = params.cams_embeds
                cam_embeds = ttnn.reshape(cam_embeds, (cam_embeds.shape[0], 1, 1, cam_embeds.shape[1]))
                feat = feat + cam_embeds
            level_embeds = params.level_embeds
            level_embeds = level_embeds[lvl : lvl + 1, :]
            level_embeds = ttnn.reshape(level_embeds, (1, 1, level_embeds.shape[0], level_embeds.shape[-1]))
            feat = feat + level_embeds
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = ttnn.concat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device="cpu")
        spatial_shapes = ttnn.from_torch(
            spatial_shapes, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
        )
        # spatial_shapes = ttnn.prod(spatial_shapes, 1)
        # spatial_shapes = ttnn.cumsum(spatial_shapes,0)[:-1]
        level_start_index = ttnn.zeros(
            (1,), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device
        )  ##---will take alook later
        feat_flatten = ttnn.permute(feat_flatten, (0, 2, 1, 3))  # (num_cam, H*W, bs, embed_dims)
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
        params,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        **kwargs,
    ):
        bev_embed = self.attn_bev_encode(
            params,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs,
        )

        return bev_embed

    def __call__(
        self,
        mlvl_feats,
        bev_queries,
        object_query_embed,
        map_query_embed,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        reg_branches=None,
        cls_branches=None,
        map_reg_branches=None,
        map_cls_branches=None,
        prev_bev=None,
        shift=None,
        can_bus=None,
        **kwargs,
    ):
        bev_embed = self.get_bev_features(
            self.params,
            mlvl_feats,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            shift=shift,
            can_bus=can_bus,
            **kwargs,
        )  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bs = mlvl_feats[0].shape[0]
        object_query_embed = ttnn.to_layout(object_query_embed, layout=ttnn.ROW_MAJOR_LAYOUT)
        query_pos, query = ttnn.split(object_query_embed, self.embed_dims, dim=1)
        query_pos = ttnn.unsqueeze(query_pos, 0)
        query_pos = ttnn.expand(query_pos, (bs, -1, -1))
        query_pos = ttnn.to_layout(query_pos, layout=ttnn.TILE_LAYOUT)
        query = ttnn.unsqueeze(query, 0)
        query = ttnn.expand(query, (bs, -1, -1))
        reference_points = ttnn.linear(
            query_pos, self.params.reference_points.weight, bias=self.params.reference_points.bias
        )
        reference_points = ttnn.sigmoid(reference_points)
        init_reference_out = reference_points
        map_query_embed = ttnn.to_layout(map_query_embed, layout=ttnn.ROW_MAJOR_LAYOUT)

        map_query_pos, map_query = ttnn.split(map_query_embed, self.embed_dims, dim=1)
        map_query_pos = ttnn.unsqueeze(map_query_pos, 0)
        map_query_pos = ttnn.expand(map_query_pos, (bs, -1, -1))

        map_query_pos = ttnn.to_layout(map_query_pos, layout=ttnn.TILE_LAYOUT)
        map_query = ttnn.unsqueeze(map_query, 0)
        map_query = ttnn.expand(map_query, (bs, -1, -1))
        map_reference_points = ttnn.linear(
            map_query_pos, self.params.map_reference_points.weight, bias=self.params.map_reference_points.bias
        )
        map_reference_points = ttnn.sigmoid(map_reference_points)
        map_init_reference_out = map_reference_points

        query = ttnn.permute(query, (1, 0, 2))
        query_pos = ttnn.permute(query_pos, (1, 0, 2))
        map_query = ttnn.permute(map_query, (1, 0, 2))
        map_query_pos = ttnn.permute(map_query_pos, (1, 0, 2))
        bev_embed = ttnn.permute(bev_embed, (1, 0, 2))

        if self.decoder is not None:
            spatial_shapes = torch.tensor([[bev_h, bev_w]], device="cpu")
            spatial_shapes = ttnn.from_torch(
                spatial_shapes, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=spatial_shapes,
                level_start_index=ttnn.zeros((1,), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device),
                **kwargs,
            )
            inter_references_out = inter_references
        else:
            inter_states = ttnn.unsqueeze(query, 0)
            inter_references_out = ttnn.unsqueeze(reference_points, 0)

        if self.map_decoder is not None:
            # [L, Q, B, D], [L, B, Q, D]
            spatial_shapes = torch.tensor([[bev_h, bev_w]], device="cpu")
            spatial_shapes = ttnn.from_torch(
                spatial_shapes, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=self.device
            )
            map_inter_states, map_inter_references = self.map_decoder(
                query=map_query,
                key=None,
                value=bev_embed,
                query_pos=map_query_pos,
                reference_points=map_reference_points,
                reg_branches=map_reg_branches,
                cls_branches=map_cls_branches,
                spatial_shapes=spatial_shapes,
                level_start_index=ttnn.zeros((1,), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=self.device),
                **kwargs,
            )
            map_inter_references_out = map_inter_references
        else:
            map_inter_states = ttnn.unsqueeze(map_query, 0)
            map_inter_references_out = ttnn.unsqueeze(map_reference_points, 0)

        return (
            bev_embed,
            inter_states,
            init_reference_out,
            inter_references_out,
            map_inter_states,
            map_init_reference_out,
            map_inter_references_out,
        )

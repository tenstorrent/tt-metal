# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import numpy as np

import ttnn

from models.experimental.uniad.tt.ttnn_encoder import TtBEVFormerEncoder
from models.experimental.uniad.tt.ttnn_decoder import TtDetectionTransformerDecoder
from torchvision.transforms.functional import rotate


class TtPerceptionTransformer:
    def __init__(
        self,
        parameters,
        device,
        num_feature_levels=4,
        num_cams=6,
        two_stage_num_proposals=300,
        encoder=None,
        decoder=None,
        embed_dims=256,
        rotate_prev_bev=True,
        use_shift=True,
        use_can_bus=True,
        can_bus_norm=True,
        use_cams_embeds=True,
        rotate_center=[100, 100],
        **kwargs,
    ):
        self.parameters = parameters
        self.device = device
        self.encoder = TtBEVFormerEncoder(
            params=parameters.encoder,
            device=device,
            num_layers=6,
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            num_points_in_pillar=4,
            return_intermediate=False,
        )
        self.decoder = TtDetectionTransformerDecoder(
            num_layers=6, embed_dim=256, num_heads=8, params=parameters.decoder, device=device
        )
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
        self.init_layers()
        self.rotate_center = rotate_center

    def init_layers(self):
        self.level_embeds = self.parameters["level_embeds"]
        self.cams_embeds = self.parameters["cams_embeds"]
        self.can_bus_mlp = [
            ttnn.linear,
            ttnn.relu,
            ttnn.linear,
            ttnn.relu,
        ]
        if self.can_bus_norm:
            self.can_bus_mlp.append(ttnn.layer_norm)

    def get_bev_features(
        self,
        mlvl_feats,
        bev_queries,
        bev_h,
        bev_w,
        grid_length=[0.512, 0.512],
        bev_pos=None,
        prev_bev=None,
        img_metas=None,
    ):
        bs = mlvl_feats[0].shape[0]
        bev_queries = ttnn.unsqueeze(bev_queries, 1)
        bev_queries = ttnn.repeat(bev_queries, (1, bs, 1))
        bev_pos = ttnn.reshape(bev_pos, (bev_pos.shape[0], bev_pos.shape[1], bev_pos.shape[2] * bev_pos.shape[3]))
        bev_pos = ttnn.permute(bev_pos, (2, 0, 1))
        # obtain rotation angle and shift with ego motion
        delta_x = np.array([each["can_bus"][0] for each in img_metas])
        delta_y = np.array([each["can_bus"][1] for each in img_metas])
        ego_angle = np.array([each["can_bus"][-2] / np.pi * 180 for each in img_metas])
        grid_length_y = grid_length[0]
        grid_length_x = grid_length[1]
        translation_length = np.sqrt(delta_x**2 + delta_y**2)
        translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
        bev_angle = ego_angle - translation_angle
        shift_y = translation_length * np.cos(bev_angle / 180 * np.pi) / grid_length_y / bev_h
        shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) / grid_length_x / bev_w
        shift_y = shift_y * self.use_shift
        shift_x = shift_x * self.use_shift
        shift = ttnn.Tensor(np.array([shift_x, shift_y]).astype("float32"), device=self.device, layout=ttnn.TILE_LAYOUT)
        shift = ttnn.permute(shift, (1, 0))  # xy, bs -> bs, xy

        if prev_bev is not None:
            assert False, "In our case prev_bev is None, So ttnn for the below is not supported"
            if prev_bev.shape[1] == bev_h * bev_w:
                prev_bev = ttnn.permute(prev_bev, (1, 0, 2))
            if self.rotate_prev_bev:
                for i in range(bs):
                    rotation_angle = img_metas[i]["can_bus"][-1]
                    tmp_prev_bev = prev_bev[:, i].reshape(bev_h, bev_w, -1).permute(2, 0, 1)
                    tmp_prev_bev = rotate(tmp_prev_bev, rotation_angle, center=self.rotate_center)
                    tmp_prev_bev = tmp_prev_bev.permute(1, 2, 0).reshape(bev_h * bev_w, 1, -1)
                    prev_bev[:, i] = tmp_prev_bev[:, 0]

        # add can bus signals
        can_bus = ttnn.Tensor(
            np.array([each["can_bus"] for each in img_metas]).astype("float32"), layout=ttnn.TILE_LAYOUT
        )  # [:, :]
        can_bus = ttnn.to_dtype(can_bus, dtype=ttnn.bfloat16)
        can_bus = ttnn.to_device(can_bus, device=self.device)

        can_bus = self.can_bus_mlp[0](
            can_bus, self.parameters.can_bus_mlp[0].weight, bias=self.parameters.can_bus_mlp[0].bias
        )
        can_bus = self.can_bus_mlp[1](can_bus)
        can_bus = self.can_bus_mlp[2](
            can_bus, self.parameters.can_bus_mlp[2].weight, bias=self.parameters.can_bus_mlp[2].bias
        )
        can_bus = self.can_bus_mlp[3](can_bus)
        if self.can_bus_norm:
            can_bus = ttnn.layer_norm(
                can_bus,
                weight=self.parameters.can_bus_mlp.norm.weight,
                bias=self.parameters.can_bus_mlp.norm.bias,
            )

        can_bus = ttnn.reshape(can_bus, (1, can_bus.shape[0], can_bus.shape[1]))
        bev_queries = bev_queries + can_bus * self.use_can_bus

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = ttnn.reshape(feat, (feat.shape[0], feat.shape[1], feat.shape[2], feat.shape[3] * feat.shape[4]))
            feat = ttnn.permute(feat, (1, 0, 3, 2))
            self.cams_embeds = ttnn.to_layout(self.cams_embeds, layout=ttnn.TILE_LAYOUT)
            if self.use_cams_embeds:
                feat = feat + ttnn.reshape(
                    self.cams_embeds, (self.cams_embeds.shape[0], 1, 1, self.cams_embeds.shape[1])
                )

            level_embeds = ttnn.to_layout(self.level_embeds, layout=ttnn.TILE_LAYOUT)
            level_embeds = level_embeds[lvl : lvl + 1, :]
            level_embeds = ttnn.reshape(level_embeds, (1, 1, level_embeds.shape[0], level_embeds.shape[-1]))
            feat = feat + level_embeds
            ttnn.deallocate(level_embeds)

            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = ttnn.concat(feat_flatten, 2)
        spatial_shapes = ttnn.Tensor(np.array(spatial_shapes), device=self.device, layout=ttnn.TILE_LAYOUT)
        level_start_index = ttnn.zeros((1,), dtype=ttnn.uint32, layout=ttnn.TILE_LAYOUT, device=self.device)

        feat_flatten = ttnn.permute(feat_flatten, (0, 2, 1, 3))  # (num_cam, H*W, bs, embed_dims)

        bev_embed = self.encoder(
            bev_queries,
            ttnn.to_layout(feat_flatten, layout=ttnn.ROW_MAJOR_LAYOUT),
            ttnn.to_layout(feat_flatten, layout=ttnn.ROW_MAJOR_LAYOUT),
            bev_h=bev_h,
            bev_w=bev_w,
            bev_pos=ttnn.to_layout(bev_pos, layout=ttnn.ROW_MAJOR_LAYOUT),
            spatial_shapes=ttnn.to_layout(spatial_shapes, layout=ttnn.ROW_MAJOR_LAYOUT),
            level_start_index=ttnn.to_layout(level_start_index, layout=ttnn.ROW_MAJOR_LAYOUT),
            prev_bev=prev_bev,
            shift=ttnn.to_layout(shift, layout=ttnn.ROW_MAJOR_LAYOUT),
            img_metas=img_metas,
        )

        return bev_embed

    def get_states_and_refs(
        self,
        bev_embed,
        object_query_embed,
        bev_h,
        bev_w,
        reference_points,
        reg_branches=None,
        cls_branches=None,
        img_metas=None,
    ):
        bs = bev_embed.shape[1]
        query_pos, query = object_query_embed[:, :256], object_query_embed[:, 256:]
        query_pos = ttnn.expand(ttnn.unsqueeze(query_pos, 0), (bs, -1, -1))
        query = ttnn.expand(ttnn.unsqueeze(query, 0), (bs, -1, -1))

        reference_points = ttnn.expand(ttnn.unsqueeze(reference_points, 0), (bs, -1, -1))
        reference_points = ttnn.sigmoid(reference_points)

        init_reference_out = ttnn.clone(reference_points)
        query = ttnn.permute(query, (1, 0, 2))
        query_pos = ttnn.permute(query_pos, (1, 0, 2))
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=bev_embed,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            cls_branches=cls_branches,
            spatial_shapes=ttnn.to_device(
                ttnn.to_dtype(
                    ttnn.Tensor(np.array([[bev_h, bev_w]]), layout=ttnn.ROW_MAJOR_LAYOUT), dtype=ttnn.bfloat16
                ),
                device=self.device,
            ),
            level_start_index=ttnn.Tensor(np.array([0]), device=self.device, layout=ttnn.TILE_LAYOUT),
            img_metas=img_metas,
        )
        inter_references_out = inter_references

        return inter_states, init_reference_out, inter_references_out

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
from models.experimental.vadv2.tt.tt_lanenet import TtLaneNet
from models.experimental.vadv2.tt.tt_decoder import TtCustomTransformerDecoder
from models.experimental.vadv2.tt.tt_transformer import TtVADPerceptionTransformer
from models.experimental.vadv2.reference.base_box3d import LiDARInstance3DBoxes
from models.experimental.vadv2.tt.tt_utils import inverse_sigmoid, bbox_xyxy_to_cxcywh
from models.experimental.vadv2.reference.nms_free_coder import MapNMSFreeCoder, CustomNMSFreeCoder


class TtLearnedPositionalEncoding:
    def __init__(
        self,
        params,
        device,
        num_feats,
        row_num_embed=50,
        col_num_embed=50,
        init_cfg=dict(type="Uniform", layer="Embedding"),
    ):
        super(TtLearnedPositionalEncoding, self).__init__()
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


class TtVADHead:
    def __init__(
        self,
        *args,
        params,
        device,
        with_box_refine=False,
        as_two_stage=False,
        bbox_coder=None,
        bev_h=30,
        bev_w=30,
        fut_ts=6,
        fut_mode=6,
        map_bbox_coder=None,
        map_num_vec=20,
        map_num_pts_per_vec=2,
        map_query_embed_type="all_pts",
        map_transform_method="minmax",
        tot_epoch=None,
        use_traj_lr_warmup=False,
        motion_decoder=None,
        motion_map_decoder=None,
        use_pe=False,
        motion_det_score=None,
        map_thresh=0.5,
        dis_thresh=0.2,
        pe_normalization=True,
        ego_his_encoder=None,
        ego_fut_mode=3,
        ego_agent_decoder=None,
        ego_map_decoder=None,
        query_thresh=None,
        query_use_fix_pad=None,
        ego_lcf_feat_idx=None,
        valid_fut_ts=6,
        **kwargs,
    ):
        super(TtVADHead, self).__init__()
        self.params = params
        self.device = device
        self.embed_dims = 256
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.tot_epoch = tot_epoch
        self.use_traj_lr_warmup = use_traj_lr_warmup
        self.motion_decoder = motion_decoder
        self.motion_map_decoder = motion_map_decoder
        self.use_pe = use_pe
        self.motion_det_score = motion_det_score
        self.map_thresh = map_thresh
        self.dis_thresh = dis_thresh
        self.pe_normalization = pe_normalization
        self.ego_his_encoder = ego_his_encoder
        self.ego_fut_mode = ego_fut_mode
        self.ego_agent_decoder = ego_agent_decoder
        self.ego_map_decoder = ego_map_decoder
        self.query_thresh = query_thresh
        self.query_use_fix_pad = query_use_fix_pad
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_reg_fcs = 2
        self.cls_out_channels = 10
        self.map_cls_out_channels = 3

        self.positional_encoding = TtLearnedPositionalEncoding(
            params.head.positional_encoding, device, self.embed_dims // 2, row_num_embed=100, col_num_embed=100
        )

        self.transformer = TtVADPerceptionTransformer(
            params=params.head.transformer,
            params_branches=params.head.branches,
            device=device,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            decoder=True,
            map_decoder=True,
            embed_dims=256,
        )

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage

        self.bbox_coder = bbox_coder
        self.pc_range = self.bbox_coder["pc_range"]

        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.map_bbox_coder = map_bbox_coder
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec

        if not self.as_two_stage:
            self.bev_embedding = self.params.head.bev_embedding
            self.query_embedding = self.params.head.query_embedding
            if self.map_query_embed_type == "all_pts":
                self.map_query_embedding = self.params.head.map_query_embedding
            elif self.map_query_embed_type == "instance_pts":
                self.map_query_embedding = None
                self.map_instance_embedding = self.params.head.map_instance_embedding
                self.map_pts_embedding = self.params.head.map_pts_embedding

        if self.motion_decoder is not None:
            self.motion_decoder = TtCustomTransformerDecoder(self.params.head.motion_decoder, self.device, num_layers=1)
            self.motion_mode_query = self.params.head.motion_mode_query
            if self.use_pe:
                self.pos_mlp_sa = ttnn.linear
        else:
            raise NotImplementedError("Not implement yet")

        if self.motion_map_decoder is not None:
            self.lane_encoder = TtLaneNet(self.params.head.lane_encoder, self.device, 256, 128, 3)

            self.motion_map_decoder = TtCustomTransformerDecoder(
                self.params.head.motion_map_decoder, self.device, num_layers=1
            )
            if self.use_pe:
                self.pos_mlp = ttnn.linear

        if self.ego_his_encoder is not None:
            self.ego_his_encoder = TtLaneNet(self.params.head.lane_encoder, self.device, 2, self.embed_dims // 2, 3)
        else:
            self.ego_query = ttnn.embedding

        if self.ego_agent_decoder is not None:
            self.ego_agent_decoder = TtCustomTransformerDecoder(
                self.params.head.ego_agent_decoder, self.device, num_layers=1
            )
            if self.use_pe:
                self.ego_agent_pos_mlp = ttnn.linear

        if self.ego_map_decoder is not None:
            self.ego_map_decoder = TtCustomTransformerDecoder(
                self.params.head.ego_map_decoder, self.device, num_layers=1
            )
            if self.use_pe:
                self.ego_map_pos_mlp = ttnn.linear

    def __call__(
        self,
        mlvl_feats,
        img_metas,
        prev_bev=None,
        only_bev=False,
        ego_his_trajs=None,
        ego_lcf_feat=None,
    ):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        if not self.as_two_stage:
            object_query_embeds = self.query_embedding.weight
        self.bbox_coder = CustomNMSFreeCoder(
            self.bbox_coder["pc_range"],
            voxel_size=self.bbox_coder["voxel_size"],
            post_center_range=self.bbox_coder["post_center_range"],
            max_num=self.bbox_coder["max_num"],
            num_classes=self.bbox_coder["num_classes"],
        )
        self.map_bbox_coder = MapNMSFreeCoder(
            self.map_bbox_coder["pc_range"],
            voxel_size=self.map_bbox_coder["voxel_size"],
            post_center_range=self.map_bbox_coder["post_center_range"],
            max_num=self.map_bbox_coder["max_num"],
            num_classes=self.map_bbox_coder["num_classes"],
        )

        if self.map_query_embed_type == "all_pts":
            map_query_embeds = self.map_query_embedding.weight
        elif self.map_query_embed_type == "instance_pts":
            map_pts_embeds = ttnn.unsqueeze(self.map_pts_embedding.weight, 0)
            map_instance_embeds = ttnn.unsqueeze(self.map_instance_embedding.weight, 1)
            map_query_embeds = map_pts_embeds + map_instance_embeds
            map_query_embeds = ttnn.reshape(
                map_query_embeds, (map_query_embeds.shape[0] * map_query_embeds.shape[1], map_query_embeds.shape[2])
            )

        bev_queries = self.bev_embedding.weight

        bev_mask = ttnn.zeros((bs, self.bev_h, self.bev_w), device=self.device, dtype=ttnn.bfloat16)
        bev_pos = self.positional_encoding(bev_mask)
        bev_pos = ttnn.to_layout(bev_pos, layout=ttnn.ROW_MAJOR_LAYOUT)

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
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
                bev_queries,
                object_query_embeds,
                map_query_embeds,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
                bev_pos=bev_pos,
                reg_branches=True if self.with_box_refine else None,  # noqa:E501
                cls_branches=True if self.as_two_stage else None,
                map_reg_branches=True if self.with_box_refine else None,  # noqa:E501
                map_cls_branches=True if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        (
            bev_embed,
            hs,
            init_reference,
            inter_references,
            map_hs,
            map_init_reference,
            map_inter_references,
        ) = outputs

        hs = ttnn.permute(hs, (0, 2, 1, 3))
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_hs = ttnn.permute(map_hs, (0, 2, 1, 3))
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []

        for lvl in range(hs.shape[0]):
            reference = init_reference if lvl == 0 else inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            cls_layers = self.params.head.branches.cls_branches[str(lvl)]
            cls_tmp = hs[lvl]

            for i in range(0, 5, 2):
                cls_tmp = ttnn.linear(
                    cls_tmp,
                    cls_layers[str(i)].weight,
                    bias=cls_layers[str(i)].bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )

                norm_key = f"{i+1}_norm"
                if norm_key in cls_layers:
                    norm_layer = cls_layers[norm_key]
                    cls_tmp = ttnn.layer_norm(cls_tmp, weight=norm_layer.weight, bias=norm_layer.bias)

                if i < 4:
                    cls_tmp = ttnn.relu(cls_tmp)

            outputs_class = cls_tmp

            reg_layers = self.params.head.branches.reg_branches[str(lvl)]

            tmp = hs[lvl]

            for i in range(3):
                tmp = ttnn.linear(
                    tmp, reg_layers[str(i)].weight, bias=reg_layers[str(i)].bias, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                if i < 2:
                    tmp = ttnn.relu(tmp)

            updated_xy = tmp[..., 0:2] + reference[..., 0:2]
            updated_xy = ttnn.sigmoid(updated_xy)
            updated_z = tmp[..., 4:5] + reference[..., 2:3]
            updated_z = ttnn.sigmoid(updated_z)

            outputs_coords_bev.append(ttnn.clone(updated_xy, memory_config=ttnn.L1_MEMORY_CONFIG))

            x = updated_xy[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            y = updated_xy[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            z = updated_z * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            tmp_out = ttnn.concat(
                [x, y, tmp[..., 2:4], z, tmp[..., 5:]], dim=-1  # 0  # 1  # 2:4 untouched  # 4  # 5:10 untouched
            )

            #     # TODO: check if using sigmoid
            outputs_coord = tmp_out
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        for lvl in range(map_hs.shape[0]):
            reference = map_init_reference if lvl == 0 else map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            # === Class Branch ===
            cls_input = map_hs[lvl]  # shape: [bs * num_vec * pts_per_vec, c]
            cls_input = ttnn.reshape(cls_input, (bs, self.map_num_vec, self.map_num_pts_per_vec, -1))
            cls_input = ttnn.mean(cls_input, dim=2)  # shape: [bs, num_vec, c]

            cls_params = self.params.head.branches.map_cls_branches[str(lvl)]
            cls_tmp = cls_input
            for i in range(0, 5, 2):
                cls_tmp = ttnn.linear(
                    cls_tmp,
                    cls_params[str(i)].weight,
                    bias=cls_params[str(i)].bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                norm_key = f"{i+1}_norm"
                if norm_key in cls_params:
                    cls_tmp = ttnn.layer_norm(
                        cls_tmp, weight=cls_params[norm_key].weight, bias=cls_params[norm_key].bias
                    )
                if i < 4:
                    cls_tmp = ttnn.relu(cls_tmp)
            map_outputs_class = cls_tmp

            # === Regression Branch ===
            reg_params = self.params.head.branches.map_reg_branches[str(lvl)]
            reg_tmp = map_hs[lvl]
            for i in range(3):  # 0,1,2
                reg_tmp = ttnn.linear(
                    reg_tmp,
                    reg_params[str(i)].weight,
                    bias=reg_params[str(i)].bias,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                if i < 2:
                    reg_tmp = ttnn.relu(reg_tmp)
            tmp = reg_tmp

            assert reference.shape[-1] == 2

            tmp_coord = tmp[..., 0:2] + reference[..., 0:2]
            tmp_coord = ttnn.sigmoid(tmp_coord)

            tmp_full = ttnn.concat([tmp_coord], dim=-1)
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp_full)
            map_outputs_coords_bev.append(ttnn.clone(map_outputs_pts_coord, memory_config=ttnn.L1_MEMORY_CONFIG))
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)

        if self.motion_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[0], outputs_coords_bev[-1].shape[1]
            # motion_query
            motion_query = ttnn.permute(hs[-1], (1, 0, 2))  # [A, B, D]
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]

            motion_query = ttnn.reshape(
                motion_query, (motion_query.shape[0], 1, motion_query.shape[1], motion_query.shape[2])
            ) + ttnn.reshape(mode_query, (1, mode_query.shape[0], 1, mode_query.shape[1]))
            motion_query = ttnn.reshape(
                motion_query,
                (motion_query.shape[0] * motion_query.shape[1], motion_query.shape[2], motion_query.shape[3]),
            )
            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_pos = self.pos_mlp_sa(
                    motion_coords, self.params.head.pos_mlp_sa.weight, bias=self.params.head.pos_mlp_sa.bias
                )  # [B, A, D]
                motion_pos = ttnn.unsqueeze(motion_pos, 2)
                motion_pos = ttnn.repeat(motion_pos, (1, 1, self.fut_mode, 1))
                motion_pos = ttnn.reshape(
                    motion_pos, (motion_pos.shape[0], motion_pos.shape[1] * motion_pos.shape[2], motion_pos.shape[3])
                )
                motion_pos = ttnn.permute(motion_pos, (1, 0, 2))  # [M, B, D]
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = ttnn.max(max_motion_score, dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = ttnn.unsqueeze(invalid_motion_idx, 2)
                invalid_motion_idx = ttnn.repeat(invalid_motion_idx, (1, 1, self.fut_mode))
                invalid_motion_idx = ttnn.reshape(
                    invalid_motion_idx,
                    (
                        invalid_motion_idx.shape[0],
                        invalid_motion_idx.shape[1] * invalid_motion_idx.shape[2],
                        invalid_motion_idx.shape[3],
                    ),
                )
            else:
                invalid_motion_idx = None

            motion_hs = self.motion_decoder(
                query=motion_query,
                key=motion_query,
                value=motion_query,
                query_pos=motion_pos,
                key_pos=motion_pos,
                key_padding_mask=invalid_motion_idx,
            )

            if self.motion_map_decoder is not None:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_coords = ttnn.unsqueeze(motion_coords, 2)
                motion_coords = ttnn.repeat(motion_coords, (1, 1, self.fut_mode, 1))
                motion_coords = ttnn.reshape(
                    motion_coords,
                    (motion_coords.shape[0], motion_coords.shape[1] * motion_coords.shape[2], motion_coords.shape[3]),
                )
                map_query = ttnn.reshape(map_hs[-1], (batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1))
                map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
                map_score = map_outputs_classes[-1]
                map_pos = map_outputs_coords_bev[-1]

                map_query, map_pos, key_padding_mask = self.select_and_pad_pred_map(
                    motion_coords,
                    map_query,
                    map_score,
                    map_pos,
                    map_thresh=self.map_thresh,
                    dis_thresh=self.dis_thresh,
                    pe_normalization=self.pe_normalization,
                    use_fix_pad=True,
                )

                map_query = ttnn.permute(map_query, (1, 0, 2))  # [P, B*M, D]
                ca_motion_query = ttnn.permute(motion_hs, (1, 0, 2))
                ca_motion_query = ttnn.reshape(
                    ca_motion_query, (ca_motion_query.shape[0] * ca_motion_query.shape[1], ca_motion_query.shape[2])
                )
                ca_motion_query = ttnn.unsqueeze(ca_motion_query, 0)

                if self.use_pe:
                    (num_query, batch) = ca_motion_query.shape[0], ca_motion_query.shape[1]
                    motion_pos = ttnn.zeros((num_query, batch, 2), device=self.device, layout=ttnn.TILE_LAYOUT)
                    motion_pos = self.pos_mlp(
                        motion_pos, self.params.head.pos_mlp.weight, bias=self.params.head.pos_mlp.bias
                    )
                    map_pos = ttnn.permute(map_pos, (1, 0, 2))
                    map_pos = self.pos_mlp(map_pos, self.params.head.pos_mlp.weight, bias=self.params.head.pos_mlp.bias)
                else:
                    motion_pos, map_pos = None, None

                ca_motion_query = self.motion_map_decoder(
                    query=ca_motion_query,
                    key=map_query,
                    value=map_query,
                    query_pos=motion_pos,
                    key_pos=map_pos,
                    key_padding_mask=key_padding_mask,
                )
            else:
                ca_motion_query = ttnn.permute(motion_hs, (1, 0, 2))
                ca_motion_query = ttnn.reshape(
                    ca_motion_query, ca_motion_query.shape[0] * ca_motion_query.shape[1], ca_motion_query.shape[2]
                )
                ca_motion_query = ttnn.unsqueeze(ca_motion_query, 0)

            batch_size = outputs_coords_bev[-1].shape[0]
            motion_hs = ttnn.permute(motion_hs, (1, 0, 2))
            B, S, D = motion_hs.shape
            motion_hs = ttnn.reshape(motion_hs, (B, num_agent, self.fut_mode, D))  # [B, A, T, D]
            ca_motion_query = ttnn.reshape(ca_motion_query, (batch_size, num_agent, self.fut_mode, -1))  # [B, A, T, D]

            motion_hs = ttnn.concat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]  #0.99
        else:
            raise NotImplementedError("Not implement yet")

        traj_branch_params = self.params.head.branches.traj_branches["0"]
        motion_h = motion_hs
        for i in range(3):  # 0,1,2
            motion_h = ttnn.linear(
                motion_h,
                traj_branch_params[str(i)].weight,
                bias=traj_branch_params[str(i)].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            if i < 2:
                motion_h = ttnn.relu(motion_h)
        outputs_traj = motion_h
        outputs_trajs.append(outputs_traj)
        cls_tmp = motion_hs
        traj_cls_params = self.params.head.traj_cls_branches["0"]
        for i in range(0, 5, 2):
            cls_tmp = ttnn.linear(
                cls_tmp,
                traj_cls_params[str(i)].weight,
                bias=traj_cls_params[str(i)].bias,
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            norm_key = f"{i+1}_norm"
            if norm_key in traj_cls_params:
                cls_tmp = ttnn.layer_norm(
                    cls_tmp, weight=traj_cls_params[norm_key].weight, bias=traj_cls_params[norm_key].bias
                )
            if i < 4:
                cls_tmp = ttnn.relu(cls_tmp)
        outputs_traj_class = cls_tmp
        outputs_trajs_classes.append(ttnn.squeeze(outputs_traj_class, -1))
        (batch, num_agent) = motion_hs.shape[0], motion_hs.shape[1]

        map_outputs_classes = ttnn.stack(map_outputs_classes, dim=0)
        map_outputs_coords = ttnn.stack(map_outputs_coords, dim=0)
        map_outputs_pts_coords = ttnn.stack(map_outputs_pts_coords, dim=0)

        outputs_classes = ttnn.stack(outputs_classes, dim=0)
        outputs_coords = ttnn.stack(outputs_coords, dim=0)
        outputs_trajs = ttnn.stack(outputs_trajs, dim=0)
        outputs_trajs_classes = ttnn.stack(outputs_trajs_classes, dim=0)

        # planning
        (batch, num_agent) = motion_hs.shape[0], motion_hs.shape[1]
        if self.ego_his_encoder is not None:
            ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
        else:
            ego_his_feats = ttnn.unsqueeze(self.params.head.ego_query.weight, 0)
            ego_his_feats = ttnn.repeat(ego_his_feats, (batch, 1, 1))
        # # Interaction
        ego_query = ego_his_feats
        ego_pos = ttnn.zeros((batch, 1, 2), device=self.device, layout=ttnn.TILE_LAYOUT)
        ego_pos_emb = ttnn.linear(
            ego_pos, self.params.head.ego_agent_pos_mlp.weight, bias=self.params.head.ego_agent_pos_mlp.bias
        )  # 0.9999987105141137
        agent_conf = outputs_classes[-1]
        agent_query = ttnn.reshape(motion_hs, (batch, num_agent, -1))
        cls_tmp = agent_query
        agent_fus_mlp_params = self.params.head.branches.agent_fus_mlp
        for i in range(4):
            if i in [0, 3]:  # Linear layers
                layer_params = agent_fus_mlp_params[str(i)]["0"]
                cls_tmp = ttnn.linear(
                    cls_tmp,
                    layer_params["weight"],
                    bias=layer_params["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            elif i == 1:  # LayerNorm
                norm_params = agent_fus_mlp_params[str(i)]["0_norm"]
                cls_tmp = ttnn.layer_norm(
                    cls_tmp,
                    weight=norm_params["weight"],
                    bias=norm_params["bias"],
                )
            elif i == 2:  # ReLU (no params)
                cls_tmp = ttnn.relu(cls_tmp)
        agent_query = cls_tmp
        agent_pos = outputs_coords_bev[-1]

        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf, score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )

        agent_pos = ttnn.to_layout(agent_pos, ttnn.TILE_LAYOUT)
        agent_pos_emb = self.ego_agent_pos_mlp(
            agent_pos, self.params.head.ego_agent_pos_mlp.weight, bias=self.params.head.ego_agent_pos_mlp.bias
        )
        # ego <-> agent interaction
        ego_agent_query = self.ego_agent_decoder(
            query=ttnn.permute(ego_query, (1, 0, 2)),
            key=ttnn.permute(agent_query, (1, 0, 2)),
            value=ttnn.permute(agent_query, (1, 0, 2)),
            query_pos=ttnn.permute(ego_pos_emb, (1, 0, 2)),
            key_pos=ttnn.permute(agent_pos_emb, (1, 0, 2)),
            key_padding_mask=agent_mask,
        )

        # # ego <-> map interaction
        ego_pos = ttnn.zeros((batch, 1, 2), device=self.device, layout=ttnn.TILE_LAYOUT)
        ego_pos_emb = self.ego_map_pos_mlp(
            ego_pos,
            self.params.head.ego_map_pos_mlp.weight,
            bias=self.params.head.ego_map_pos_mlp.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        map_query = map_hs[-1]
        map_query = ttnn.reshape(map_query, (batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1))
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_query = ttnn.unsqueeze(map_query, 0)
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        # # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[0], map_pos.shape[1]
        x = map_pos[:, :, :, 0]
        y = map_pos[:, :, :, 1]
        x_sq = ttnn.pow(x, 2)
        y_sq = ttnn.pow(y, 2)
        result = ttnn.add(x_sq, y_sq)
        map_dis = ttnn.sqrt(result)
        map_dis = ttnn.to_layout(map_dis, ttnn.ROW_MAJOR_LAYOUT)
        min_map_pos_idx = ttnn.argmax((map_dis * -1), dim=-1)  # [B, P]
        min_map_pos_idx = ttnn.reshape(min_map_pos_idx, [-1])
        min_map_pos = ttnn.reshape(map_pos, [map_pos.shape[0] * map_pos.shape[1], map_pos.shape[2], map_pos.shape[3]])
        min_map_pos = ttnn.to_torch(min_map_pos)
        min_map_pos_idx = ttnn.to_torch(min_map_pos_idx)
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]

        min_map_pos = ttnn.from_torch(min_map_pos, dtype=ttnn.bfloat16, device=self.device)

        min_map_pos = ttnn.reshape(min_map_pos, (batch, num_map, 2))  # [B, P, 2]
        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf, score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )

        map_pos = ttnn.to_layout(map_pos, ttnn.TILE_LAYOUT)
        map_pos_emb = self.ego_map_pos_mlp(
            map_pos,
            self.params.head.ego_map_pos_mlp.weight,
            bias=self.params.head.ego_map_pos_mlp.bias,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=ttnn.permute(map_query, (1, 0, 2)),
            value=ttnn.permute(map_query, (1, 0, 2)),
            query_pos=ttnn.permute(ego_pos_emb, (1, 0, 2)),
            key_pos=ttnn.permute(map_pos_emb, (1, 0, 2)),
            key_padding_mask=map_mask,
        )

        if self.ego_his_encoder is not None and self.ego_lcf_feat_idx is not None:
            ego_feats = ttnn.concat(
                [
                    ego_his_feats,
                    ttnn.permute(ego_map_query, (1, 0, 2)),
                    ttnn.squeeze(ego_lcf_feat, 1)[..., self.ego_lcf_feat_idx],
                ],
                dim=-1,
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is not None and self.ego_lcf_feat_idx is None:
            ego_feats = ttnn.concat([ego_his_feats, ttnn.permute(ego_map_query, (1, 0, 2))], dim=-1)  # [B, 1, 2D]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is not None:
            ego_feats = ttnn.concat(
                [
                    ttnn.permute(ego_agent_query, (1, 0, 2)),
                    ttnn.permute(ego_map_query, (1, 0, 2)),
                    ttnn.squeeze(ego_lcf_feat, 1)[..., self.ego_lcf_feat_idx],
                ],
                dim=-1,
            )
        if self.ego_his_encoder is None and self.ego_lcf_feat_idx is None:
            ego_feats = ttnn.concat(
                [
                    ttnn.permute(ego_agent_query, (1, 0, 2)),
                    ttnn.permute(ego_map_query, (1, 0, 2)),
                ],
                dim=-1,
            )
        # # Ego prediction
        ego_fut_decoder_params = self.params.head.branches.ego_fut_decoder
        cls_tmp = ego_feats
        for i in range(5):
            if i in [0, 2, 4]:  # Linear layers
                layer_params = ego_fut_decoder_params[str(i)]["0"]
                cls_tmp = ttnn.linear(
                    cls_tmp,
                    layer_params["weight"],
                    bias=layer_params["bias"],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            elif i in [1, 3]:  # ReLU layers
                cls_tmp = ttnn.relu(cls_tmp)
        outputs_ego_trajs = cls_tmp
        outputs_ego_trajs = ttnn.reshape(
            outputs_ego_trajs, (outputs_ego_trajs.shape[0], self.ego_fut_mode, self.fut_ts, 2)
        )
        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_traj_preds": ttnn.repeat(outputs_trajs, (outputs_coords.shape[0], 1, 1, 1, 1)),
            "all_traj_cls_scores": ttnn.repeat(outputs_trajs_classes, (outputs_coords.shape[0], 1, 1, 1)),
            "map_all_cls_scores": map_outputs_classes,
            "map_all_bbox_preds": map_outputs_coords,
            "map_all_pts_preds": map_outputs_pts_coords,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "map_enc_cls_scores": None,
            "map_enc_bbox_preds": None,
            "map_enc_pts_preds": None,
            "ego_fut_preds": outputs_ego_trajs,
        }

        return outs

    def map_transform_box(self, pts, y_first=False):
        pts_reshape = ttnn.reshape(pts, (pts.shape[0], self.map_num_vec, self.map_num_pts_per_vec, 2))
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == "minmax":
            # import pdb;pdb.set_trace()

            xmin = ttnn.min(pts_x, dim=2, keepdim=True)[0]
            xmax = ttnn.max(pts_x, dim=2, keepdim=True)[0]
            ymin = ttnn.min(pts_y, dim=2, keepdim=True)[0]
            ymax = ttnn.max(pts_y, dim=2, keepdim=True)[0]
            bbox = ttnn.concat([xmin, ymin, xmax, ymax], dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
        # map_bboxes: xmin, ymin, xmax, ymax
        map_preds_dicts = self.map_bbox_coder.decode(preds_dicts)

        num_samples = len(det_preds_dicts)
        assert len(det_preds_dicts) == len(map_preds_dicts), "len(preds_dict) should be equal to len(map_preds_dicts)"
        ret_list = []
        box_type_3d = LiDARInstance3DBoxes
        for i in range(num_samples):
            preds = det_preds_dicts[i]
            bboxes = preds["bboxes"]
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            code_size = bboxes.shape[-1]
            bboxes = box_type_3d(bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]
            trajs = preds["trajs"]

            map_preds = map_preds_dicts[i]
            map_bboxes = map_preds["map_bboxes"]
            map_scores = map_preds["map_scores"]
            map_labels = map_preds["map_labels"]
            map_pts = map_preds["map_pts"]

            ret_list.append([bboxes, scores, labels, trajs, map_bboxes, map_scores, map_labels, map_pts])

        return ret_list

    def select_and_pad_pred_map(
        self,
        motion_pos,
        map_query,
        map_score,
        map_pos,
        map_thresh=0.5,
        dis_thresh=None,
        pe_normalization=True,
        use_fix_pad=False,
    ):
        map_query = ttnn.unsqueeze(map_query, 0)
        batch, num_map = map_pos.shape[0], map_pos.shape[1]
        map_pos = ttnn.to_layout(map_pos, ttnn.TILE_LAYOUT)
        map_score = ttnn.to_layout(map_score, ttnn.TILE_LAYOUT)
        x = map_pos[:, :, :, 0]
        y = map_pos[:, :, :, 1]
        x_sq = ttnn.pow(x, 2)
        y_sq = ttnn.pow(y, 2)
        result = ttnn.add(x_sq, y_sq)
        map_dis = ttnn.sqrt(result)
        map_dis = ttnn.to_layout(map_dis, ttnn.ROW_MAJOR_LAYOUT)
        min_map_pos_idx = ttnn.argmax((map_dis * -1), dim=-1)  # [B, P]
        min_map_pos_idx = ttnn.reshape(min_map_pos_idx, [-1])
        min_map_pos = ttnn.reshape(map_pos, [map_pos.shape[0] * map_pos.shape[1], map_pos.shape[2], map_pos.shape[3]])
        min_map_pos = ttnn.to_torch(min_map_pos)
        min_map_pos_idx = ttnn.to_torch(min_map_pos_idx)

        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]

        min_map_pos = ttnn.from_torch(min_map_pos, dtype=ttnn.bfloat16, device=self.device)  # [B*P, 2]
        min_map_pos = ttnn.reshape(min_map_pos, (batch, num_map, 2))  # [B, P, 2]
        min_map_pos = ttnn.to_layout(min_map_pos, layout=ttnn.TILE_LAYOUT)

        map_score = ttnn.sigmoid_accurate(map_score)
        map_max_score = ttnn.max(map_score, dim=-1)[0]
        map_max_score = ttnn.unsqueeze(map_max_score, 0)
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = ttnn.sum(map_idx[i])
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum
        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = ttnn.sum(map_idx[i])
            map_query = ttnn.to_torch(map_query)
            min_map_pos = ttnn.to_torch(min_map_pos)
            map_idx = ttnn.to_torch(map_idx, dtype=torch.bool)
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            valid_map_query = ttnn.from_torch(valid_map_query, dtype=ttnn.bfloat16, device=self.device)
            valid_map_pos = ttnn.from_torch(valid_map_pos, dtype=ttnn.bfloat16, device=self.device)
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device="cpu")
            padding_mask = ttnn.from_torch(padding_mask, dtype=ttnn.bfloat16, device=self.device)
            padding_mask = ttnn.repeat(padding_mask, [int(batch_max_pnum.item())])
            if pad_pnum.item() != 0:
                valid_map_query = ttnn.concat(
                    [valid_map_query, ttnn.zeros((pad_pnum, dim), device=self.device)],
                    dim=0,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                valid_map_pos = ttnn.concat(
                    [valid_map_pos, ttnn.zeros((pad_pnum, 2), device=self.device)],
                    dim=0,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = ttnn.stack(selected_map_query, dim=0)
        selected_map_pos = ttnn.stack(selected_map_pos, dim=0)
        selected_padding_mask = ttnn.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = ttnn.unsqueeze(selected_map_query, 1)
        selected_map_query = ttnn.repeat(selected_map_query, [1, num_agent, 1, 1])

        selected_map_pos = ttnn.unsqueeze(selected_map_pos, 1)
        selected_map_pos = ttnn.repeat(selected_map_pos, [1, num_agent, 1, 1])

        selected_padding_mask = ttnn.unsqueeze(selected_padding_mask, 1)
        selected_padding_mask = ttnn.repeat(selected_padding_mask, [1, num_agent, 1])

        # move lane to per-car coords system
        B, A, D = motion_pos.shape  # Get the original shape
        motion_pos = ttnn.reshape(motion_pos, (B, A, 1, D))
        selected_map_pos = ttnn.to_layout(selected_map_pos, layout=ttnn.TILE_LAYOUT)
        motion_pos = ttnn.to_layout(motion_pos, layout=ttnn.TILE_LAYOUT)
        selected_map_dist = selected_map_pos - motion_pos  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos  # [B, A, max_P, 2]

        # filter far map inst for each agent
        x = selected_map_dist[:, :, :, 0]
        y = selected_map_dist[:, :, :, 1]
        x_sq = ttnn.pow(x, 2)
        y_sq = ttnn.pow(y, 2)
        result = ttnn.add(x_sq, y_sq)
        map_dis = ttnn.sqrt(result)
        valid_map_inst = map_dis <= dis_thresh  # [B, A, max_P]
        invalid_map_inst = valid_map_inst == False
        selected_padding_mask = selected_padding_mask + invalid_map_inst
        B, N, P, D = selected_map_query.shape
        selected_map_query = ttnn.reshape(selected_map_query, (N, P, D))  # [1800, 3, 256]
        selected_map_pos = ttnn.reshape(selected_map_pos, (N, P, 2))  # [1800, 3, 2]
        selected_padding_mask = ttnn.reshape(selected_padding_mask, (N, P))  # [1800, 3]

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = ttnn.zeros((num_batch, 1, feat_dim), device=self.device)
            pad_map_pos = ttnn.ones((num_batch, 1, 2), device=self.device)
            a = torch.tensor([False])
            a = ttnn.from_torch(a, dtype=ttnn.bfloat16, device=self.device)
            pad_lane_mask = ttnn.repeat(ttnn.unsqueeze(a, 0), [num_batch, 1])

            selected_map_query = ttnn.concat([selected_map_query, pad_map_query], dim=1)
            pad_map_pos = ttnn.to_layout(pad_map_pos, layout=ttnn.TILE_LAYOUT)
            selected_map_pos = ttnn.concat([selected_map_pos, pad_map_pos], dim=1)
            pad_lane_mask = ttnn.to_layout(pad_lane_mask, layout=ttnn.TILE_LAYOUT)
            selected_padding_mask = ttnn.concat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask

    def select_and_pad_query(self, query, query_pos, query_score, score_thresh=0.5, use_fix_pad=True):
        # select & pad query for different batch using score_thresh
        query_score = ttnn.to_layout(query_score, layout=ttnn.TILE_LAYOUT)
        query_score = ttnn.sigmoid_accurate(query_score)
        query_score = ttnn.to_layout(query_score, layout=ttnn.ROW_MAJOR_LAYOUT)
        query_score = ttnn.add(query_score, 0.0, dtype=ttnn.float32)
        query_score = ttnn.max(query_score, dim=-1)[0]
        query_score = ttnn.unsqueeze(query_score, 0)
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = ttnn.sum(query_idx[i])
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = ttnn.sum(query_idx[i])
            query = ttnn.to_torch(query)
            query_pos = ttnn.to_torch(query_pos)
            query_idx = ttnn.to_torch(query_idx, dtype=torch.bool)
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            valid_query = ttnn.from_torch(valid_query, dtype=ttnn.bfloat16, device=self.device)
            valid_query_pos = ttnn.from_torch(valid_query_pos, dtype=ttnn.bfloat16, device=self.device)

            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False])
            padding_mask = ttnn.from_torch(padding_mask, dtype=ttnn.bfloat16, device=self.device)
            padding_mask = ttnn.repeat(padding_mask, [int(batch_max_qnum.item())])
            if pad_qnum.item() != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat(
                    [valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0
                )
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = ttnn.stack(selected_query, dim=0)
        selected_query_pos = ttnn.stack(selected_query_pos, dim=0)
        selected_padding_mask = ttnn.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        return selected_query, selected_query_pos, selected_padding_mask

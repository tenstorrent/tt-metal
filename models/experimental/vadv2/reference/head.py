# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import copy
import torch
import torch.nn as nn
from models.experimental.vadv2.reference.nms_free_coder import MapNMSFreeCoder, CustomNMSFreeCoder
from models.experimental.vadv2.reference.decoder import CustomTransformerDecoder
from models.experimental.vadv2.reference.transformer import VADPerceptionTransformer
from models.experimental.vadv2.reference.utils import inverse_sigmoid, bbox_xyxy_to_cxcywh
from models.experimental.vadv2.reference.base_box3d import LiDARInstance3DBoxes


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, init_cfg=dict(type="Uniform", layer="Embedding")):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = (
            torch.cat((x_embed.unsqueeze(0).repeat(h, 1, 1), y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(mask.shape[0], 1, 1, 1)
        )
        return pos


class MLP(nn.Module):
    def __init__(self, in_channels, hidden_unit, verbose=False):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels, hidden_unit), nn.LayerNorm(hidden_unit), nn.ReLU())

    def forward(self, x):
        x = self.mlp(x)
        return x


class LaneNet(nn.Module):
    def __init__(self, in_channels, hidden_unit, num_subgraph_layers):
        super(LaneNet, self).__init__()
        self.num_subgraph_layers = num_subgraph_layers
        self.layer_seq = nn.Sequential()
        for i in range(num_subgraph_layers):
            self.layer_seq.add_module(f"lmlp_{i}", MLP(in_channels, hidden_unit))
            in_channels = hidden_unit * 2

    def forward(self, pts_lane_feats):
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


class VADHead(nn.Module):
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
        fut_ts=6,
        fut_mode=6,
        loss_traj_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.8),
        map_bbox_coder=None,
        map_num_query=900,
        map_num_classes=3,
        map_num_vec=20,
        map_num_pts_per_vec=2,
        map_num_pts_per_gt_vec=2,
        map_query_embed_type="all_pts",
        map_transform_method="minmax",
        map_gt_shift_pts_pattern="v0",
        map_dir_interval=1,
        map_code_size=None,
        map_code_weights=None,
        loss_map_cls=dict(
            type="CrossEntropyLoss", bg_cls_weight=0.1, use_sigmoid=False, loss_weight=1.0, class_weight=1.0
        ),
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
        super(VADHead, self).__init__()
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

        self.positional_encoding = LearnedPositionalEncoding(self.embed_dims // 2, row_num_embed=100, col_num_embed=100)

        self.transformer = VADPerceptionTransformer(
            rotate_prev_bev=True, use_shift=True, use_can_bus=True, decoder=True, map_decoder=True, embed_dims=256
        )

        if loss_traj_cls["use_sigmoid"] == True:
            self.traj_num_cls = 1
        else:
            self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
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
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = bbox_coder
        self.pc_range = self.bbox_coder["pc_range"]

        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        self.map_bbox_coder = map_bbox_coder
        self.map_query_embed_type = map_query_embed_type
        self.map_transform_method = map_transform_method
        self.map_gt_shift_pts_pattern = map_gt_shift_pts_pattern
        map_num_query = map_num_vec * map_num_pts_per_vec
        self.map_num_query = map_num_query
        self.map_num_classes = map_num_classes
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec
        self.map_num_pts_per_gt_vec = map_num_pts_per_gt_vec
        self.map_dir_interval = map_dir_interval
        self.num_query = 300
        if loss_map_cls["use_sigmoid"] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1

        self.map_bg_cls_weight = 0
        map_class_weight = loss_map_cls.get("class_weight", None)
        if map_class_weight is not None and (self.__class__ is VADHead):
            assert isinstance(map_class_weight, float), (
                "Expected " "class_weight to have type float. Found " f"{type(map_class_weight)}."
            )

            map_bg_cls_weight = loss_map_cls.get("bg_cls_weight", map_class_weight)
            assert isinstance(map_bg_cls_weight, float), (
                "Expected " "bg_cls_weight to have type float. Found " f"{type(map_bg_cls_weight)}."
            )
            map_class_weight = torch.ones(map_num_classes + 1) * map_class_weight
            # set background class as the last indice
            map_class_weight[map_num_classes] = map_bg_cls_weight
            loss_map_cls.update({"class_weight": map_class_weight})
            if "bg_cls_weight" in loss_map_cls:
                loss_map_cls.pop("bg_cls_weight")
            self.map_bg_cls_weight = map_bg_cls_weight

        self.traj_bg_cls_weight = 0

        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(
            torch.tensor(self.map_code_weights, requires_grad=False), requires_grad=False
        )
        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(nn.Linear(self.embed_dims * 2, self.embed_dims * 2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(nn.Linear(self.embed_dims * 2, self.fut_ts * 2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(nn.Linear(self.embed_dims * 2, self.embed_dims * 2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims * 2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(nn.Linear(self.embed_dims * 2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(nn.Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(nn.Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = 3
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = 3
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList([cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList([traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList([traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList([map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList([map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            if self.map_query_embed_type == "all_pts":
                self.map_query_embedding = nn.Embedding(self.map_num_query, self.embed_dims * 2)
            elif self.map_query_embed_type == "instance_pts":
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)

        if self.motion_decoder is not None:
            self.motion_decoder = CustomTransformerDecoder(num_layers=1)
            self.motion_mode_query = nn.Embedding(self.fut_mode, self.embed_dims)
            self.motion_mode_query.weight.requires_grad = True
            if self.use_pe:
                self.pos_mlp_sa = nn.Linear(2, self.embed_dims)
        else:
            raise NotImplementedError("Not implement yet")

        if self.motion_map_decoder is not None:
            self.lane_encoder = LaneNet(256, 128, 3)

            self.motion_map_decoder = CustomTransformerDecoder(num_layers=1)
            if self.use_pe:
                self.pos_mlp = nn.Linear(2, self.embed_dims)

        if self.ego_his_encoder is not None:
            self.ego_his_encoder = LaneNet(2, self.embed_dims // 2, 3)
        else:
            self.ego_query = nn.Embedding(1, self.embed_dims)

        if self.ego_agent_decoder is not None:
            self.ego_agent_decoder = CustomTransformerDecoder(num_layers=1)
            if self.use_pe:
                self.ego_agent_pos_mlp = nn.Linear(2, self.embed_dims)

        if self.ego_map_decoder is not None:
            self.ego_map_decoder = CustomTransformerDecoder(num_layers=1)
            if self.use_pe:
                self.ego_map_pos_mlp = nn.Linear(2, self.embed_dims)

        ego_fut_decoder = []
        ego_fut_dec_in_dim = (
            self.embed_dims * 2 + len(self.ego_lcf_feat_idx)
            if self.ego_lcf_feat_idx is not None
            else self.embed_dims * 2
        )
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(nn.Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(nn.Linear(ego_fut_dec_in_dim, self.ego_fut_mode * self.fut_ts * 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)

        self.agent_fus_mlp = nn.Sequential(
            nn.Linear(self.fut_mode * 2 * self.embed_dims, self.embed_dims, bias=True),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims, bias=True),
        )

    def forward(
        self,
        mlvl_feats,
        img_metas,
        prev_bev=None,
        only_bev=False,
        ego_his_trajs=None,
        ego_lcf_feat=None,
    ):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = self.query_embedding.weight.to(dtype)
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
            map_query_embeds = self.map_query_embedding.weight.to(dtype)
        elif self.map_query_embed_type == "instance_pts":
            map_pts_embeds = self.map_pts_embedding.weight.unsqueeze(0)
            map_instance_embeds = self.map_instance_embedding.weight.unsqueeze(1)
            map_query_embeds = (map_pts_embeds + map_instance_embeds).flatten(0, 1).to(dtype)

        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)

        if only_bev:
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
                reg_branches=self.reg_branches if self.with_box_refine else None,
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,
                map_cls_branches=self.map_cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        bev_embed, hs, init_reference, inter_references, map_hs, map_init_reference, map_inter_references = outputs

        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_coords_bev = []
        outputs_trajs = []
        outputs_trajs_classes = []

        map_hs = map_hs.permute(0, 2, 1, 3)
        map_outputs_classes = []
        map_outputs_coords = []
        map_outputs_pts_coords = []
        map_outputs_coords_bev = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            tmp[..., 1:2] = tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            tmp[..., 4:5] = tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        for lvl in range(map_hs.shape[0]):
            if lvl == 0:
                reference = map_init_reference
            else:
                reference = map_inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            map_outputs_class = self.map_cls_branches[lvl](
                map_hs[lvl].view(bs, self.map_num_vec, self.map_num_pts_per_vec, -1).mean(2)
            )
            tmp = self.map_reg_branches[lvl](map_hs[lvl])

            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            tmp = tmp.sigmoid()  # cx,cy,w,h
            map_outputs_coord, map_outputs_pts_coord = self.map_transform_box(tmp)
            map_outputs_coords_bev.append(map_outputs_pts_coord.clone().detach())
            map_outputs_classes.append(map_outputs_class)
            map_outputs_coords.append(map_outputs_coord)
            map_outputs_pts_coords.append(map_outputs_pts_coord)

        if self.motion_decoder is not None:
            batch_size, num_agent = outputs_coords_bev[-1].shape[:2]

            # motion_query
            motion_query = hs[-1].permute(1, 0, 2)  # [A, B, D]
            mode_query = self.motion_mode_query.weight  # [fut_mode, D]
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)

            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1)
                motion_pos = motion_pos.flatten(1, 2)
                motion_pos = motion_pos.permute(1, 0, 2)  # [M, B, D]
            else:
                motion_pos = None

            if self.motion_det_score is not None:
                motion_score = outputs_classes[-1]
                max_motion_score = motion_score.max(dim=-1)[0]
                invalid_motion_idx = max_motion_score < self.motion_det_score  # [B, A]
                invalid_motion_idx = invalid_motion_idx.unsqueeze(2).repeat(1, 1, self.fut_mode).flatten(1, 2)
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
                # map preprocess
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_coords = motion_coords.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
                map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
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
                map_query = map_query.permute(1, 0, 2)  # [P, B*M, D]
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

                # position encoding
                if self.use_pe:
                    (num_query, batch) = ca_motion_query.shape[:2]
                    motion_pos = torch.zeros((num_query, batch, 2), device=motion_hs.device)
                    motion_pos = self.pos_mlp(motion_pos)
                    map_pos = map_pos.permute(1, 0, 2)
                    map_pos = self.pos_mlp(map_pos)
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
                ca_motion_query = motion_hs.permute(1, 0, 2).flatten(0, 1).unsqueeze(0)

            batch_size = outputs_coords_bev[-1].shape[0]
            motion_hs = motion_hs.permute(1, 0, 2).unflatten(dim=1, sizes=(num_agent, self.fut_mode))
            ca_motion_query = ca_motion_query.squeeze(0).unflatten(dim=0, sizes=(batch_size, num_agent, self.fut_mode))
            motion_hs = torch.cat([motion_hs, ca_motion_query], dim=-1)  # [B, A, fut_mode, 2D]
        else:
            raise NotImplementedError("Not implement yet")

        outputs_traj = self.traj_branches[0](motion_hs)
        outputs_trajs.append(outputs_traj)
        outputs_traj_class = self.traj_cls_branches[0](motion_hs)
        outputs_trajs_classes.append(outputs_traj_class.squeeze(-1))
        (batch, num_agent) = motion_hs.shape[:2]

        map_outputs_classes = torch.stack(map_outputs_classes)
        map_outputs_coords = torch.stack(map_outputs_coords)
        map_outputs_pts_coords = torch.stack(map_outputs_pts_coords)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outputs_trajs = torch.stack(outputs_trajs)
        outputs_trajs_classes = torch.stack(outputs_trajs_classes)

        # planning
        (batch, num_agent) = motion_hs.shape[:2]
        if self.ego_his_encoder is not None:
            ego_his_feats = self.ego_his_encoder(ego_his_trajs)  # [B, 1, dim]
        else:
            ego_his_feats = self.ego_query.weight.unsqueeze(0).repeat(batch, 1, 1)
        # # Interaction
        ego_query = ego_his_feats
        ego_pos = torch.zeros((batch, 1, 2), device=ego_query.device)
        ego_pos_emb = self.ego_agent_pos_mlp(ego_pos)
        agent_conf = outputs_classes[-1]
        agent_query = motion_hs.reshape(batch, num_agent, -1)
        agent_query = self.agent_fus_mlp(agent_query)  # [B, A, fut_mode, 2*D] -> [B, A, D]
        agent_pos = outputs_coords_bev[-1]
        agent_query, agent_pos, agent_mask = self.select_and_pad_query(
            agent_query, agent_pos, agent_conf, score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        agent_pos_emb = self.ego_agent_pos_mlp(agent_pos)

        # # ego <-> agent interaction
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=agent_pos_emb.permute(1, 0, 2),
            key_padding_mask=agent_mask,
        )

        # # ego <-> map interaction
        ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)
        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        # # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0] ** 2 + map_pos[..., 1] ** 2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]
        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]
        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]
        map_query, map_pos, map_mask = self.select_and_pad_query(
            map_query, min_map_pos, map_conf, score_thresh=self.query_thresh, use_fix_pad=self.query_use_fix_pad
        )
        map_pos_emb = self.ego_map_pos_mlp(map_pos)
        ego_map_query = self.ego_map_decoder(
            query=ego_agent_query,
            key=map_query.permute(1, 0, 2),
            value=map_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=map_pos_emb.permute(1, 0, 2),
            key_padding_mask=map_mask,
        )

        if self.ego_his_encoder is not None and self.ego_lcf_feat_idx is not None:
            ego_feats = torch.cat(
                [ego_his_feats, ego_map_query.permute(1, 0, 2), ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx]],
                dim=-1,
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is not None and self.ego_lcf_feat_idx is None:
            ego_feats = torch.cat([ego_his_feats, ego_map_query.permute(1, 0, 2)], dim=-1)  # [B, 1, 2D]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is not None:
            ego_feats = torch.cat(
                [
                    ego_agent_query.permute(1, 0, 2),
                    ego_map_query.permute(1, 0, 2),
                    ego_lcf_feat.squeeze(1)[..., self.ego_lcf_feat_idx],
                ],
                dim=-1,
            )  # [B, 1, 2D+2]
        elif self.ego_his_encoder is None and self.ego_lcf_feat_idx is None:
            ego_feats = torch.cat(
                [ego_agent_query.permute(1, 0, 2), ego_map_query.permute(1, 0, 2)], dim=-1
            )  # [B, 1, 2D]

        # Ego prediction
        outputs_ego_trajs = self.ego_fut_decoder(ego_feats)
        outputs_ego_trajs = outputs_ego_trajs.reshape(outputs_ego_trajs.shape[0], self.ego_fut_mode, self.fut_ts, 2)

        outs = {
            "bev_embed": bev_embed,
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_traj_preds": outputs_trajs.repeat(outputs_coords.shape[0], 1, 1, 1, 1),
            "all_traj_cls_scores": outputs_trajs_classes.repeat(outputs_coords.shape[0], 1, 1, 1),
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
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec, self.map_num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == "minmax":
            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        det_preds_dicts = self.bbox_coder.decode(preds_dicts)
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
        if dis_thresh is None:
            raise NotImplementedError("Not implement yet")
        # use the most close pts pos in each map inst as the inst's pos
        batch, num_map = map_pos.shape[:2]
        map_dis = torch.sqrt(map_pos[..., 0] ** 2 + map_pos[..., 1] ** 2)
        min_map_pos_idx = map_dis.argmin(dim=-1).flatten()  # [B*P]

        min_map_pos = map_pos.flatten(0, 1)  # [B*P, pts, 2]

        min_map_pos = min_map_pos[range(min_map_pos.shape[0]), min_map_pos_idx]  # [B*P, 2]
        min_map_pos = min_map_pos.view(batch, num_map, 2)  # [B, P, 2]

        # select & pad map vectors for different batch using map_thresh
        map_score = map_score.sigmoid()
        map_max_score = map_score.max(dim=-1)[0]
        map_idx = map_max_score > map_thresh
        batch_max_pnum = 0
        for i in range(map_score.shape[0]):
            pnum = map_idx[i].sum()
            if pnum > batch_max_pnum:
                batch_max_pnum = pnum

        selected_map_query, selected_map_pos, selected_padding_mask = [], [], []
        for i in range(map_score.shape[0]):
            dim = map_query.shape[-1]
            valid_pnum = map_idx[i].sum()
            valid_map_query = map_query[i, map_idx[i]]
            valid_map_pos = min_map_pos[i, map_idx[i]]
            pad_pnum = batch_max_pnum - valid_pnum
            padding_mask = torch.tensor([False], device=map_score.device).repeat(batch_max_pnum)
            if pad_pnum != 0:
                valid_map_query = torch.cat(
                    [valid_map_query, torch.zeros((pad_pnum, dim), device=map_score.device)], dim=0
                )
                valid_map_pos = torch.cat([valid_map_pos, torch.zeros((pad_pnum, 2), device=map_score.device)], dim=0)
                padding_mask[valid_pnum:] = True
            selected_map_query.append(valid_map_query)
            selected_map_pos.append(valid_map_pos)
            selected_padding_mask.append(padding_mask)

        selected_map_query = torch.stack(selected_map_query, dim=0)
        selected_map_pos = torch.stack(selected_map_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        # generate different pe for map vectors for each agent
        num_agent = motion_pos.shape[1]
        selected_map_query = selected_map_query.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, D]
        selected_map_pos = selected_map_pos.unsqueeze(1).repeat(1, num_agent, 1, 1)  # [B, A, max_P, 2]
        selected_padding_mask = selected_padding_mask.unsqueeze(1).repeat(1, num_agent, 1)  # [B, A, max_P]
        # move lane to per-car coords system
        selected_map_dist = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]
        if pe_normalization:
            selected_map_pos = selected_map_pos - motion_pos[:, :, None, :]  # [B, A, max_P, 2]

        # filter far map inst for each agent
        map_dis = torch.sqrt(selected_map_dist[..., 0] ** 2 + selected_map_dist[..., 1] ** 2)
        valid_map_inst = map_dis <= dis_thresh  # [B, A, max_P]
        invalid_map_inst = valid_map_inst == False
        selected_padding_mask = selected_padding_mask + invalid_map_inst

        selected_map_query = selected_map_query.flatten(0, 1)
        selected_map_pos = selected_map_pos.flatten(0, 1)
        selected_padding_mask = selected_padding_mask.flatten(0, 1)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_map_query.shape[-1]
        if use_fix_pad:
            pad_map_query = torch.zeros((num_batch, 1, feat_dim), device=selected_map_query.device)
            pad_map_pos = torch.ones((num_batch, 1, 2), device=selected_map_pos.device)
            pad_lane_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_map_query = torch.cat([selected_map_query, pad_map_query], dim=1)
            selected_map_pos = torch.cat([selected_map_pos, pad_map_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_lane_mask], dim=1)

        return selected_map_query, selected_map_pos, selected_padding_mask

    def select_and_pad_query(self, query, query_pos, query_score, score_thresh=0.5, use_fix_pad=True):
        # select & pad query for different batch using score_thresh
        query_score = query_score.sigmoid()
        query_score = query_score.max(dim=-1)[0]
        query_idx = query_score > score_thresh
        batch_max_qnum = 0
        for i in range(query_score.shape[0]):
            qnum = query_idx[i].sum()
            if qnum > batch_max_qnum:
                batch_max_qnum = qnum

        selected_query, selected_query_pos, selected_padding_mask = [], [], []
        for i in range(query_score.shape[0]):
            dim = query.shape[-1]
            valid_qnum = query_idx[i].sum()
            valid_query = query[i, query_idx[i]]
            valid_query_pos = query_pos[i, query_idx[i]]
            pad_qnum = batch_max_qnum - valid_qnum
            padding_mask = torch.tensor([False], device=query_score.device).repeat(batch_max_qnum)
            if pad_qnum != 0:
                valid_query = torch.cat([valid_query, torch.zeros((pad_qnum, dim), device=query_score.device)], dim=0)
                valid_query_pos = torch.cat(
                    [valid_query_pos, torch.zeros((pad_qnum, 2), device=query_score.device)], dim=0
                )
                padding_mask[valid_qnum:] = True
            selected_query.append(valid_query)
            selected_query_pos.append(valid_query_pos)
            selected_padding_mask.append(padding_mask)

        selected_query = torch.stack(selected_query, dim=0)
        selected_query_pos = torch.stack(selected_query_pos, dim=0)
        selected_padding_mask = torch.stack(selected_padding_mask, dim=0)

        num_batch = selected_padding_mask.shape[0]
        feat_dim = selected_query.shape[-1]
        if use_fix_pad:
            pad_query = torch.zeros((num_batch, 1, feat_dim), device=selected_query.device)
            pad_query_pos = torch.ones((num_batch, 1, 2), device=selected_query_pos.device)
            pad_mask = torch.tensor([False], device=selected_padding_mask.device).unsqueeze(0).repeat(num_batch, 1)
            selected_query = torch.cat([selected_query, pad_query], dim=1)
            selected_query_pos = torch.cat([selected_query_pos, pad_query_pos], dim=1)
            selected_padding_mask = torch.cat([selected_padding_mask, pad_mask], dim=1)

        return selected_query, selected_query_pos, selected_padding_mask

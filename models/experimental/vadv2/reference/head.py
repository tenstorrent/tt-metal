import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from VADv2.reference.nms_free_coder import MapNMSFreeCoder, CustomNMSFreeCoder
from VADv2.reference.decoder import CustomTransformerDecoder
from VADv2.reference.transformer import VADPerceptionTransformer
from VADv2.reference.utils import inverse_sigmoid, bbox_xyxy_to_cxcywh
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes


class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, num_feats, row_num_embed=50, col_num_embed=50, init_cfg=dict(type="Uniform", layer="Embedding")):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
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
        """
            Extract lane_feature from vectorized lane representation

        Args:
            pts_lane_feats: [batch size, max_pnum, pts, D]

        Returns:
            inst_lane_feats: [batch size, max_pnum, D]
        """
        x = pts_lane_feats
        for name, layer in self.layer_seq.named_modules():
            if isinstance(layer, MLP):
                # x [bs,max_lane_num,9,dim]
                x = layer(x)
                x_max = torch.max(x, -2)[0]
                x_max = x_max.unsqueeze(2).repeat(1, 1, x.shape[2], 1)
                x = torch.cat([x, x_max], dim=-1)
        x_max = torch.max(x, -2)[0]
        return x_max


class VADHead(nn.Module):
    """Head of VAD model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

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
        loss_traj=dict(type="L1Loss", loss_weight=0.25),
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
        loss_map_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_map_iou=dict(type="GIoULoss", loss_weight=2.0),
        loss_map_pts=dict(type="ChamferDistance", loss_src_weight=1.0, loss_dst_weight=1.0),
        loss_map_dir=dict(type="PtsDirCosLoss", loss_weight=2.0),
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
        loss_plan_reg=dict(type="L1Loss", loss_weight=0.25),
        loss_plan_bound=dict(type="PlanMapBoundLoss", loss_weight=0.1),
        loss_plan_col=dict(type="PlanAgentDisLoss", loss_weight=0.1),
        loss_plan_dir=dict(type="PlanMapThetaLoss", loss_weight=0.1),
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
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
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
        # if kwargs['train_cfg'] is not None:
        #     assert 'map_assigner' in kwargs['train_cfg'], 'map assigner should be provided '\
        #         'when train_cfg is set.'
        #     map_assigner = kwargs['train_cfg']['map_assigner']
        #     assert loss_map_cls['loss_weight'] == map_assigner['cls_cost']['weight'], \
        #         'The classification weight for loss and matcher should be' \
        #         'exactly the same.'
        # assert loss_map_bbox['loss_weight'] == map_assigner['reg_cost'][
        #     'weight'], 'The regression L1 weight for loss and matcher ' \
        #     'should be exactly the same.'
        # assert loss_map_iou['loss_weight'] == map_assigner['iou_cost']['weight'], \
        #     'The regression iou weight for loss and matcher should be' \
        #     'exactly the same.'
        # assert loss_map_pts['loss_weight'] == map_assigner['pts_cost']['weight'], \
        #     'The regression l1 weight for map pts loss and matcher should be' \
        #     'exactly the same.'

        # self.map_assigner = build_assigner(map_assigner)
        # # DETR sampling=False, so use PseudoSampler
        # sampler_cfg = dict(type='PseudoSampler')
        # self.map_sampler = build_sampler(sampler_cfg, context=self)

        # self.loss_traj = build_loss(loss_traj)
        # self.loss_traj_cls = build_loss(loss_traj_cls)
        # self.loss_map_bbox = build_loss(loss_map_bbox)
        # self.loss_map_cls = build_loss(loss_map_cls)
        # self.loss_map_iou = build_loss(loss_map_iou)
        # self.loss_map_pts = build_loss(loss_map_pts)
        # self.loss_map_dir = build_loss(loss_map_dir)
        # self.loss_plan_reg = build_loss(loss_plan_reg)
        # self.loss_plan_bound = build_loss(loss_plan_bound)
        # self.loss_plan_col = build_loss(loss_plan_col)
        # self.loss_plan_dir = build_loss(loss_plan_dir)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
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

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
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
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

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
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                map_reg_branches=self.map_reg_branches if self.with_box_refine else None,  # noqa:E501
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

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] = tmp[..., 0:2] + reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            outputs_coords_bev.append(tmp[..., 0:2].clone().detach())
            tmp[..., 4:5] = tmp[..., 4:5] + reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()
            tmp[..., 0:1] = tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) + self.pc_range[0]
            tmp[..., 1:2] = tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) + self.pc_range[1]
            tmp[..., 4:5] = tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) + self.pc_range[2]

            # TODO: check if using sigmoid
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
            # TODO: check the shape of reference
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
            # [M, B, D], M=A*fut_mode
            motion_query = (motion_query[:, None, :, :] + mode_query[None, :, None, :]).flatten(0, 1)
            if self.use_pe:
                motion_coords = outputs_coords_bev[-1]  # [B, A, 2]
                motion_pos = self.pos_mlp_sa(motion_coords)  # [B, A, D]
                motion_pos = motion_pos.unsqueeze(2).repeat(1, 1, self.fut_mode, 1).flatten(1, 2)
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
        # Interaction
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
        # ego <-> agent interaction
        ego_agent_query = self.ego_agent_decoder(
            query=ego_query.permute(1, 0, 2),
            key=agent_query.permute(1, 0, 2),
            value=agent_query.permute(1, 0, 2),
            query_pos=ego_pos_emb.permute(1, 0, 2),
            key_pos=agent_pos_emb.permute(1, 0, 2),
            key_padding_mask=agent_mask,
        )

        # ego <-> map interaction
        ego_pos = torch.zeros((batch, 1, 2), device=agent_query.device)
        ego_pos_emb = self.ego_map_pos_mlp(ego_pos)
        map_query = map_hs[-1].view(batch_size, self.map_num_vec, self.map_num_pts_per_vec, -1)
        map_query = self.lane_encoder(map_query)  # [B, P, pts, D] -> [B, P, D]
        map_conf = map_outputs_classes[-1]
        map_pos = map_outputs_coords_bev[-1]
        # use the most close pts pos in each map inst as the inst's pos
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
            "hs": hs,
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
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """
        pts_reshape = pts.view(pts.shape[0], self.map_num_vec, self.map_num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.map_transform_method == "minmax":
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape

    def _get_target_single(self, cls_score, bbox_pred, gt_labels, gt_bboxes, gt_attr_labels, gt_bboxes_ignore=None):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 10].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 9) in [x,y,z,w,l,h,yaw,vx,vy] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_fut_trajs = gt_attr_labels[:, : self.fut_ts * 2]
        gt_fut_masks = gt_attr_labels[:, self.fut_ts * 2 : self.fut_ts * 3]
        gt_bbox_c = gt_bboxes.shape[-1]
        num_gt_bbox, gt_traj_c = gt_fut_trajs.shape

        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_bbox_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # trajs targets
        traj_targets = torch.zeros((num_bboxes, gt_traj_c), dtype=torch.float32, device=bbox_pred.device)
        traj_weights = torch.zeros_like(traj_targets)
        traj_targets[pos_inds] = gt_fut_trajs[sampling_result.pos_assigned_gt_inds]
        traj_weights[pos_inds] = 1.0

        # Filter out invalid fut trajs
        traj_masks = torch.zeros_like(traj_targets)  # [num_bboxes, fut_ts*2]
        gt_fut_masks = gt_fut_masks.unsqueeze(-1).repeat(1, 1, 2).view(num_gt_bbox, -1)  # [num_gt_bbox, fut_ts*2]
        traj_masks[pos_inds] = gt_fut_masks[sampling_result.pos_assigned_gt_inds]
        traj_weights = traj_weights * traj_masks

        # Extra future timestamp mask for controlling pred horizon
        fut_ts_mask = torch.zeros((num_bboxes, self.fut_ts, 2), dtype=torch.float32, device=bbox_pred.device)
        fut_ts_mask[:, : self.valid_fut_ts, :] = 1.0
        fut_ts_mask = fut_ts_mask.view(num_bboxes, -1)
        traj_weights = traj_weights * fut_ts_mask

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            traj_targets,
            traj_weights,
            traj_masks.view(-1, self.fut_ts, 2)[..., 0],
            pos_inds,
            neg_inds,
        )

    def _map_get_target_single(
        self, cls_score, bbox_pred, pts_pred, gt_labels, gt_bboxes, gt_shifts_pts, gt_bboxes_ignore=None
    ):
        """ "Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        assign_result, order_index = self.map_assigner.assign(
            bbox_pred, cls_score, pts_pred, gt_bboxes, gt_labels, gt_shifts_pts, gt_bboxes_ignore
        )

        sampling_result = self.map_sampler.sample(assign_result, bbox_pred, gt_bboxes)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        # label targets
        labels = gt_bboxes.new_full((num_bboxes,), self.map_num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)
        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # pts targets
        if order_index is None:
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0), pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds, assigned_shift, :, :]
        return (labels, label_weights, bbox_targets, bbox_weights, pts_targets, pts_weights, pos_inds, neg_inds)

    def get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_attr_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            traj_targets_list,
            traj_weights_list,
            gt_fut_masks_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            cls_scores_list,
            bbox_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_attr_labels_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            traj_targets_list,
            traj_weights_list,
            gt_fut_masks_list,
            num_total_pos,
            num_total_neg,
        )

    def map_get_targets(
        self,
        cls_scores_list,
        bbox_preds_list,
        pts_preds_list,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, "Only supports for gt_bboxes_ignore setting to None."
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [gt_bboxes_ignore_list for _ in range(num_imgs)]

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._map_get_target_single,
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_labels_list,
            gt_bboxes_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def loss_planning(
        self,
        ego_fut_preds,
        ego_fut_gt,
        ego_fut_masks,
        ego_fut_cmd,
        lane_preds,
        lane_score_preds,
        agent_preds,
        agent_fut_preds,
        agent_score_preds,
        agent_fut_cls_preds,
    ):
        """ "Loss function for ego vehicle planning.
        Args:
            ego_fut_preds (Tensor): [B, ego_fut_mode, fut_ts, 2]
            ego_fut_gt (Tensor): [B, fut_ts, 2]
            ego_fut_masks (Tensor): [B, fut_ts]
            ego_fut_cmd (Tensor): [B, ego_fut_mode]
            lane_preds (Tensor): [B, num_vec, num_pts, 2]
            lane_score_preds (Tensor): [B, num_vec, 3]
            agent_preds (Tensor): [B, num_agent, 2]
            agent_fut_preds (Tensor): [B, num_agent, fut_mode, fut_ts, 2]
            agent_score_preds (Tensor): [B, num_agent, 10]
            agent_fut_cls_scores (Tensor): [B, num_agent, fut_mode]
        Returns:
            loss_plan_reg (Tensor): planning reg loss.
            loss_plan_bound (Tensor): planning map boundary constraint loss.
            loss_plan_col (Tensor): planning col constraint loss.
            loss_plan_dir (Tensor): planning directional constraint loss.
        """

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(ego_fut_preds, ego_fut_gt, loss_plan_l1_weight)

        loss_plan_bound = self.loss_plan_bound(
            ego_fut_preds[ego_fut_cmd == 1], lane_preds, lane_score_preds, weight=ego_fut_masks
        )

        loss_plan_col = self.loss_plan_col(
            ego_fut_preds[ego_fut_cmd == 1],
            agent_preds,
            agent_fut_preds,
            agent_score_preds,
            agent_fut_cls_preds,
            weight=ego_fut_masks[:, :, None].repeat(1, 1, 2),
        )

        loss_plan_dir = self.loss_plan_dir(
            ego_fut_preds[ego_fut_cmd == 1], lane_preds, lane_score_preds, weight=ego_fut_masks
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_plan_l1 = torch.nan_to_num(loss_plan_l1)
            loss_plan_bound = torch.nan_to_num(loss_plan_bound)
            loss_plan_col = torch.nan_to_num(loss_plan_col)
            loss_plan_dir = torch.nan_to_num(loss_plan_dir)

        loss_plan_dict = dict()
        loss_plan_dict["loss_plan_reg"] = loss_plan_l1
        loss_plan_dict["loss_plan_bound"] = loss_plan_bound
        loss_plan_dict["loss_plan_col"] = loss_plan_col
        loss_plan_dict["loss_plan_dir"] = loss_plan_dir

        return loss_plan_dict

    def loss_single(
        self,
        cls_scores,
        bbox_preds,
        traj_preds,
        traj_cls_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_attr_labels_list,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(
            cls_scores_list, bbox_preds_list, gt_bboxes_list, gt_labels_list, gt_attr_labels_list, gt_bboxes_ignore_list
        )

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            traj_targets_list,
            traj_weights_list,
            gt_fut_masks_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        traj_targets = torch.cat(traj_targets_list, 0)
        traj_weights = torch.cat(traj_weights_list, 0)
        gt_fut_masks = torch.cat(gt_fut_masks_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights
        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos,
        )

        # traj regression loss
        best_traj_preds = self.get_best_fut_preds(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks,
        )

        neg_inds = bbox_weights[:, 0] == 0
        traj_labels = self.get_traj_cls_target(
            traj_preds.reshape(-1, self.fut_mode, self.fut_ts, 2),
            traj_targets.reshape(-1, self.fut_ts, 2),
            gt_fut_masks,
            neg_inds,
        )

        loss_traj = self.loss_traj(
            best_traj_preds[isnotnan], traj_targets[isnotnan], traj_weights[isnotnan], avg_factor=num_total_pos
        )

        if self.use_traj_lr_warmup:
            loss_scale_factor = get_traj_warmup_loss_weight(self.epoch, self.tot_epoch)
            loss_traj = loss_scale_factor * loss_traj

        # traj classification loss
        traj_cls_scores = traj_cls_preds.reshape(-1, self.fut_mode)
        # construct weighted avg_factor to match with the official DETR repo
        traj_cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.traj_bg_cls_weight
        if self.sync_cls_avg_factor:
            traj_cls_avg_factor = reduce_mean(traj_cls_scores.new_tensor([traj_cls_avg_factor]))

        traj_cls_avg_factor = max(traj_cls_avg_factor, 1)
        loss_traj_cls = self.loss_traj_cls(traj_cls_scores, traj_labels, label_weights, avg_factor=traj_cls_avg_factor)

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_traj = torch.nan_to_num(loss_traj)
            loss_traj_cls = torch.nan_to_num(loss_traj_cls)

        return loss_cls, loss_bbox, loss_traj, loss_traj_cls

    def get_best_fut_preds(self, traj_preds, traj_targets, gt_fut_masks):
        """ "Choose best preds among all modes.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            pred_box_centers (Tensor): Pred box centers with shape (num_box_preds, 2).
            gt_box_centers (Tensor): Ground truth box centers with shape (num_box_preds, 2).

        Returns:
            best_traj_preds (Tensor): best traj preds (min displacement error with gt)
                with shape (num_box_preds, fut_ts*2).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        min_mode_idxs = torch.argmin(dist, dim=-1).tolist()
        box_idxs = torch.arange(traj_preds.shape[0]).tolist()
        best_traj_preds = traj_preds[box_idxs, min_mode_idxs, :, :].reshape(-1, self.fut_ts * 2)

        return best_traj_preds

    def get_traj_cls_target(self, traj_preds, traj_targets, gt_fut_masks, neg_inds):
        """ "Get Trajectory mode classification target.
        Args:
            traj_preds (Tensor): MultiModal traj preds with shape (num_box_preds, fut_mode, fut_ts, 2).
            traj_targets (Tensor): Ground truth traj for each pred box with shape (num_box_preds, fut_ts, 2).
            gt_fut_masks (Tensor): Ground truth traj mask with shape (num_box_preds, fut_ts).
            neg_inds (Tensor): Negtive indices with shape (num_box_preds,)

        Returns:
            traj_labels (Tensor): traj cls labels (num_box_preds,).
        """

        cum_traj_preds = traj_preds.cumsum(dim=-2)
        cum_traj_targets = traj_targets.cumsum(dim=-2)

        # Get min pred mode indices.
        # (num_box_preds, fut_mode, fut_ts)
        dist = torch.linalg.norm(cum_traj_targets[:, None, :, :] - cum_traj_preds, dim=-1)
        dist = dist * gt_fut_masks[:, None, :]
        dist = dist[..., -1]
        dist[torch.isnan(dist)] = dist[torch.isnan(dist)] * 0
        traj_labels = torch.argmin(dist, dim=-1)
        traj_labels[neg_inds] = self.fut_mode

        return traj_labels

    def map_loss_single(
        self,
        cls_scores,
        bbox_preds,
        pts_preds,
        gt_bboxes_list,
        gt_labels_list,
        gt_shifts_pts_list,
        gt_bboxes_ignore_list=None,
    ):
        """ "Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_pts_list (list[Tensor]): Ground truth pts for each image
                with shape (num_gts, fixed_num, 2) in [x,y] format.
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        pts_preds_list = [pts_preds[i] for i in range(num_imgs)]

        cls_reg_targets = self.map_get_targets(
            cls_scores_list,
            bbox_preds_list,
            pts_preds_list,
            gt_bboxes_list,
            gt_labels_list,
            gt_shifts_pts_list,
            gt_bboxes_ignore_list,
        )
        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            pts_targets_list,
            pts_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.map_cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + num_total_neg * self.map_bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_map_cls(cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes accross all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.map_code_weights

        loss_bbox = self.loss_map_bbox(
            bbox_preds[isnotnan, :4],
            normalized_bbox_targets[isnotnan, :4],
            bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos,
        )

        # regression pts CD loss
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        if self.map_num_pts_per_vec != self.map_num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0, 2, 1)
            pts_preds = F.interpolate(pts_preds, size=(self.map_num_pts_per_gt_vec), mode="linear", align_corners=True)
            pts_preds = pts_preds.permute(0, 2, 1).contiguous()

        loss_pts = self.loss_map_pts(
            pts_preds[isnotnan, :, :],
            normalized_pts_targets[isnotnan, :, :],
            pts_weights[isnotnan, :, :],
            avg_factor=num_total_pos,
        )

        dir_weights = pts_weights[:, : -self.map_dir_interval, 0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = (
            denormed_pts_preds[:, self.map_dir_interval :, :] - denormed_pts_preds[:, : -self.map_dir_interval, :]
        )
        pts_targets_dir = pts_targets[:, self.map_dir_interval :, :] - pts_targets[:, : -self.map_dir_interval, :]

        loss_dir = self.loss_map_dir(
            denormed_pts_preds_dir[isnotnan, :, :],
            pts_targets_dir[isnotnan, :, :],
            dir_weights[isnotnan, :],
            avg_factor=num_total_pos,
        )

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_map_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], avg_factor=num_total_pos
        )

        if digit_version(TORCH_VERSION) >= digit_version("1.8"):
            loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)

        return loss_cls, loss_bbox, loss_iou, loss_pts, loss_dir

    def loss(
        self,
        gt_bboxes_list,
        gt_labels_list,
        map_gt_bboxes_list,
        map_gt_labels_list,
        preds_dicts,
        ego_fut_gt,
        ego_fut_masks,
        ego_fut_cmd,
        gt_attr_labels,
        gt_bboxes_ignore=None,
        map_gt_bboxes_ignore=None,
        img_metas=None,
    ):
        """ "Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, (
            f"{self.__class__.__name__} only supports " f"for gt_bboxes_ignore setting to None."
        )

        map_gt_vecs_list = copy.deepcopy(map_gt_bboxes_list)

        all_cls_scores = preds_dicts["all_cls_scores"]
        all_bbox_preds = preds_dicts["all_bbox_preds"]
        all_traj_preds = preds_dicts["all_traj_preds"]
        all_traj_cls_scores = preds_dicts["all_traj_cls_scores"]
        enc_cls_scores = preds_dicts["enc_cls_scores"]
        enc_bbox_preds = preds_dicts["enc_bbox_preds"]
        map_all_cls_scores = preds_dicts["map_all_cls_scores"]
        map_all_bbox_preds = preds_dicts["map_all_bbox_preds"]
        map_all_pts_preds = preds_dicts["map_all_pts_preds"]
        map_enc_cls_scores = preds_dicts["map_enc_cls_scores"]
        map_enc_bbox_preds = preds_dicts["map_enc_bbox_preds"]
        map_enc_pts_preds = preds_dicts["map_enc_pts_preds"]
        ego_fut_preds = preds_dicts["ego_fut_preds"]

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1).to(device)
            for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_attr_labels_list = [gt_attr_labels for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [gt_bboxes_ignore for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, loss_traj, loss_traj_cls = multi_apply(
            self.loss_single,
            all_cls_scores,
            all_bbox_preds,
            all_traj_preds,
            all_traj_cls_scores,
            all_gt_bboxes_list,
            all_gt_labels_list,
            all_gt_attr_labels_list,
            all_gt_bboxes_ignore_list,
        )

        num_dec_layers = len(map_all_cls_scores)
        device = map_gt_labels_list[0].device

        map_gt_bboxes_list = [map_gt_bboxes.bbox.to(device) for map_gt_bboxes in map_gt_vecs_list]
        map_gt_pts_list = [map_gt_bboxes.fixed_num_sampled_points.to(device) for map_gt_bboxes in map_gt_vecs_list]
        if self.map_gt_shift_pts_pattern == "v0":
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points.to(device) for gt_bboxes in map_gt_vecs_list
            ]
        elif self.map_gt_shift_pts_pattern == "v1":
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v1.to(device) for gt_bboxes in map_gt_vecs_list
            ]
        elif self.map_gt_shift_pts_pattern == "v2":
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v2.to(device) for gt_bboxes in map_gt_vecs_list
            ]
        elif self.map_gt_shift_pts_pattern == "v3":
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v3.to(device) for gt_bboxes in map_gt_vecs_list
            ]
        elif self.map_gt_shift_pts_pattern == "v4":
            map_gt_shifts_pts_list = [
                gt_bboxes.shift_fixed_num_sampled_points_v4.to(device) for gt_bboxes in map_gt_vecs_list
            ]
        else:
            raise NotImplementedError
        map_all_gt_bboxes_list = [map_gt_bboxes_list for _ in range(num_dec_layers)]
        map_all_gt_labels_list = [map_gt_labels_list for _ in range(num_dec_layers)]
        map_all_gt_pts_list = [map_gt_pts_list for _ in range(num_dec_layers)]
        map_all_gt_shifts_pts_list = [map_gt_shifts_pts_list for _ in range(num_dec_layers)]
        map_all_gt_bboxes_ignore_list = [map_gt_bboxes_ignore for _ in range(num_dec_layers)]

        map_losses_cls, map_losses_bbox, map_losses_iou, map_losses_pts, map_losses_dir = multi_apply(
            self.map_loss_single,
            map_all_cls_scores,
            map_all_bbox_preds,
            map_all_pts_preds,
            map_all_gt_bboxes_list,
            map_all_gt_labels_list,
            map_all_gt_shifts_pts_list,
            map_all_gt_bboxes_ignore_list,
        )

        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict["loss_cls"] = losses_cls[-1]
        loss_dict["loss_bbox"] = losses_bbox[-1]
        loss_dict["loss_traj"] = loss_traj[-1]
        loss_dict["loss_traj_cls"] = loss_traj_cls[-1]
        # loss from the last decoder layer
        loss_dict["loss_map_cls"] = map_losses_cls[-1]
        loss_dict["loss_map_bbox"] = map_losses_bbox[-1]
        loss_dict["loss_map_iou"] = map_losses_iou[-1]
        loss_dict["loss_map_pts"] = map_losses_pts[-1]
        loss_dict["loss_map_dir"] = map_losses_dir[-1]

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        batch, num_agent = all_traj_preds[-1].shape[:2]
        agent_fut_preds = all_traj_preds[-1].view(batch, num_agent, self.fut_mode, self.fut_ts, 2)
        agent_fut_cls_preds = all_traj_cls_scores[-1].view(batch, num_agent, self.fut_mode)
        loss_plan_input = [
            ego_fut_preds,
            ego_fut_gt,
            ego_fut_masks,
            ego_fut_cmd,
            map_all_pts_preds[-1],
            map_all_cls_scores[-1].sigmoid(),
            all_bbox_preds[-1][..., 0:2],
            agent_fut_preds,
            all_cls_scores[-1].sigmoid(),
            agent_fut_cls_preds.sigmoid(),
        ]

        loss_planning_dict = self.loss_planning(*loss_plan_input)
        loss_dict["loss_plan_reg"] = loss_planning_dict["loss_plan_reg"]
        loss_dict["loss_plan_bound"] = loss_planning_dict["loss_plan_bound"]
        loss_dict["loss_plan_col"] = loss_planning_dict["loss_plan_col"]
        loss_dict["loss_plan_dir"] = loss_planning_dict["loss_plan_dir"]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f"d{num_dec_layer}.loss_cls"] = loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_bbox"] = loss_bbox_i
            num_dec_layer += 1
        # loss from other decoder layers
        num_dec_layer = 0
        for map_loss_cls_i, map_loss_bbox_i, map_loss_iou_i, map_loss_pts_i, map_loss_dir_i in zip(
            map_losses_cls[:-1], map_losses_bbox[:-1], map_losses_iou[:-1], map_losses_pts[:-1], map_losses_dir[:-1]
        ):
            loss_dict[f"d{num_dec_layer}.loss_map_cls"] = map_loss_cls_i
            loss_dict[f"d{num_dec_layer}.loss_map_bbox"] = map_loss_bbox_i
            loss_dict[f"d{num_dec_layer}.loss_map_iou"] = map_loss_iou_i
            loss_dict[f"d{num_dec_layer}.loss_map_pts"] = map_loss_pts_i
            loss_dict[f"d{num_dec_layer}.loss_map_dir"] = map_loss_dir_i
            num_dec_layer += 1

        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [torch.zeros_like(gt_labels_list[i]) for i in range(len(all_gt_labels_list))]
            enc_loss_cls, enc_losses_bbox = self.loss_single(
                enc_cls_scores, enc_bbox_preds, gt_bboxes_list, binary_labels_list, gt_bboxes_ignore
            )
            loss_dict["enc_loss_cls"] = enc_loss_cls
            loss_dict["enc_loss_bbox"] = enc_losses_bbox

        if map_enc_cls_scores is not None:
            map_binary_labels_list = [
                torch.zeros_like(map_gt_labels_list[i]) for i in range(len(map_all_gt_labels_list))
            ]
            # TODO bug here, but we dont care enc_loss now
            (
                map_enc_loss_cls,
                map_enc_loss_bbox,
                map_enc_loss_iou,
                map_enc_loss_pts,
                map_enc_loss_dir,
            ) = self.map_loss_single(
                map_enc_cls_scores,
                map_enc_bbox_preds,
                map_enc_pts_preds,
                map_gt_bboxes_list,
                map_binary_labels_list,
                map_gt_pts_list,
                map_gt_bboxes_ignore,
            )
            loss_dict["enc_loss_map_cls"] = map_enc_loss_cls
            loss_dict["enc_loss_map_bbox"] = map_enc_loss_bbox
            loss_dict["enc_loss_map_iou"] = map_enc_loss_iou
            loss_dict["enc_loss_map_pts"] = map_enc_loss_pts
            loss_dict["enc_loss_map_dir"] = map_enc_loss_dir

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """

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
        """select_and_pad_pred_map.
        Args:
            motion_pos: [B, A, 2]
            map_query: [B, P, D].
            map_score: [B, P, 3].
            map_pos: [B, P, pts, 2].
            map_thresh: map confidence threshold for filtering low-confidence preds
            dis_thresh: distance threshold for masking far maps for each agent in cross-attn
            use_fix_pad: always pad one lane instance for each batch
        Returns:
            selected_map_query: [B*A, P1(+1), D], P1 is the max inst num after filter and pad.
            selected_map_pos: [B*A, P1(+1), 2]
            selected_padding_mask: [B*A, P1(+1)]
        """

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
        """select_and_pad_query.
        Args:
            query: [B, Q, D].
            query_pos: [B, Q, 2]
            query_score: [B, Q, C].
            score_thresh: confidence threshold for filtering low-confidence query
            use_fix_pad: always pad one query instance for each batch
        Returns:
            selected_query: [B, Q', D]
            selected_query_pos: [B, Q', 2]
            selected_padding_mask: [B, Q']
        """

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

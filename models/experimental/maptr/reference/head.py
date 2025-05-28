import copy
import torch
import torch.nn as nn

from models.experimental.maptr.reference.utils import inverse_sigmoid, bbox_xyxy_to_cxcywh
from models.experimental.maptr.reference.nms_free_coder import MapTRNMSFreeCoder
from models.experimental.maptr.reference.transformer import MapTRPerceptionTransformer


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


class MapTRHead(nn.Module):
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
        loss_pts=dict(type="ChamferDistance", loss_src_weight=1.0, loss_dst_weight=1.0),
        loss_dir=dict(type="PtsDirCosLoss", loss_weight=2.0),
        **kwargs,
    ):
        super(MapTRHead, self).__init__()
        self.bbox_coder = bbox_coder
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.embed_dims = 256
        self.cls_out_channels = 3
        self.num_layers = 2
        self.bev_encoder_type = "BEVFormerEncoder"

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.num_reg_fcs = 2

        self.positional_encoding = LearnedPositionalEncoding(self.embed_dims // 2, row_num_embed=80, col_num_embed=40)

        self.transformer = MapTRPerceptionTransformer(
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=256,
        )

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

        self.code_weights = nn.Parameter(torch.tensor(self.code_weights, requires_grad=False), requires_grad=False)
        # self.loss_pts = build_loss(loss_pts)
        # self.loss_dir = build_loss(loss_dir)
        num_query = num_vec * num_pts_per_vec
        self.num_query = num_query
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self._init_layers()

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        # cls_branch.append(Linear(self.embed_dims * 2, self.embed_dims))
        # cls_branch.append(nn.LayerNorm(self.embed_dims))
        # cls_branch.append(nn.ReLU(inplace=True))
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))  # TODO Check if this is the workaround
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.num_layers + 1) if self.as_two_stage else self.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList([fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList([reg_branch for _ in range(num_pred)])
        if not self.as_two_stage:
            if self.bev_encoder_type == "BEVFormerEncoder":
                self.bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
            else:
                self.bev_embedding = None
            if self.query_embed_type == "all_pts":
                self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
            elif self.query_embed_type == "instance_pts":
                self.query_embedding = None
                self.instance_embedding = nn.Embedding(self.num_vec, self.embed_dims * 2)
                self.pts_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims * 2)

    def forward(self, mlvl_feats, lidar_feat, img_metas, prev_bev=None, only_bev=False):
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

        self.bbox_coder = MapTRNMSFreeCoder(
            self.bbox_coder["pc_range"],
            voxel_size=self.bbox_coder["voxel_size"],
            post_center_range=self.bbox_coder["post_center_range"],
            max_num=self.bbox_coder["max_num"],
            num_classes=self.bbox_coder["num_classes"],
        )

        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        # import pdb;pdb.set_trace()
        if self.query_embed_type == "all_pts":
            object_query_embeds = self.query_embedding.weight.to(dtype)
        elif self.query_embed_type == "instance_pts":
            pts_embeds = self.pts_embedding.weight.unsqueeze(0)
            instance_embeds = self.instance_embedding.weight.unsqueeze(1)
            object_query_embeds = (pts_embeds + instance_embeds).flatten(0, 1).to(dtype)
        if self.bev_embedding is not None:
            bev_queries = self.bev_embedding.weight.to(dtype)

            bev_mask = torch.zeros((bs, self.bev_h, self.bev_w), device=bev_queries.device).to(dtype)
            bev_pos = self.positional_encoding(bev_mask).to(dtype)
        else:
            bev_queries = None
            bev_mask = None
            bev_pos = None

        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                lidar_feat,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(0.75, 0.75),
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
                grid_length=(0.75, 0.75),
                bev_pos=bev_pos,
                reg_branches=self.reg_branches if self.with_box_refine else None,  # noqa:E501
                cls_branches=self.cls_branches if self.as_two_stage else None,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )

        bev_embed, hs, init_reference, inter_references = outputs
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []
        outputs_pts_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                # import pdb;pdb.set_trace()
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            # import pdb;pdb.set_trace()
            # vec_embedding = hs[lvl].reshape(bs, self.num_vec, -1)
            outputs_class = self.cls_branches[lvl](hs[lvl].view(bs, self.num_vec, self.num_pts_per_vec, -1).mean(2))
            tmp = self.reg_branches[lvl](hs[lvl])

            # TODO: check the shape of reference
            assert reference.shape[-1] == 2
            tmp[..., 0:2] += reference[..., 0:2]
            # tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp = tmp.sigmoid()  # cx,cy,w,h
            # import pdb;pdb.set_trace()
            # tmp[..., 0:1] = (tmp[..., 0:1] * (self.pc_range[3] -
            #                  self.pc_range[0]) + self.pc_range[0])
            # tmp[..., 1:2] = (tmp[..., 1:2] * (self.pc_range[4] -
            #                  self.pc_range[1]) + self.pc_range[1])
            # tmp = tmp.reshape(bs, self.num_vec,-1)
            # TODO: check if using sigmoid
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
        pts_reshape = pts.view(pts.shape[0], self.num_vec, self.num_pts_per_vec, 2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == "minmax":
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

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds["bboxes"]
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds["scores"]
            labels = preds["labels"]
            pts = preds["pts"]

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list

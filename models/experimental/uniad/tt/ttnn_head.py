# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn

from models.experimental.uniad.tt.ttnn_perception_transformer import TtPerceptionTransformer
from models.experimental.uniad.tt.ttnn_nms_free_coder import TtNMSFreeCoder
from models.experimental.uniad.tt.ttnn_utils import inverse_sigmoid


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


class TtBEVFormerTrackHead:
    def __init__(
        self,
        parameters,
        device,
        *args,
        with_box_refine=False,
        as_two_stage=False,
        transformer=None,
        bbox_coder=None,
        num_cls_fcs=2,
        code_weights=None,
        bev_h=30,
        bev_w=30,
        past_steps=4,
        fut_steps=4,
        **kwargs,
    ):
        self.parameters = parameters
        self.device = device
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False

        self.with_box_refine = with_box_refine

        assert as_two_stage is False, "as_two_stage is not supported yet."
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

        self.bbox_coder = TtNMSFreeCoder(
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            voxel_size=[0.2, 0.2, 8],
            num_classes=10,
        )
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1
        self.past_steps = past_steps
        self.fut_steps = fut_steps

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = False if "sync_cls_avg_factor" not in kwargs else kwargs["sync_cls_avg_factor"]

        self.num_query = 100 if "num_query" not in kwargs else kwargs["num_query"]
        self.num_classes = kwargs["num_classes"]
        self.in_channels = kwargs["in_channels"]
        self.num_reg_fcs = 2 if "num_reg_fcs" not in kwargs else kwargs["num_reg_fcs"]
        self.fp16_enabled = False
        self.cls_out_channels = kwargs["num_classes"]

        self.activate = ttnn.relu
        self.positional_encoding = TtLearnedPositionalEncoding(
            params=parameters.positional_encoding, device=device, num_feats=128, row_num_embed=50, col_num_embed=50
        )
        self.transformer = TtPerceptionTransformer(
            parameters=parameters.transformer,
            device=device,
            num_feature_levels=4,
            num_cams=6,
            two_stage_num_proposals=300,
            embed_dims=256,
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            can_bus_norm=True,
            use_cams_embeds=True,
            rotate_center=[100, 100],
        )
        self.embed_dims = self.transformer.embed_dims
        positional_encoding = dict(type="SinePositionalEncoding", num_feats=128, normalize=True)
        assert "num_feats" in positional_encoding
        num_feats = positional_encoding["num_feats"]
        assert num_feats * 2 == self.embed_dims, (
            "embed_dims should" f" be exactly 2 times of num_feats. Found {self.embed_dims}" f" and {num_feats}."
        )
        self._init_layers()

        self.code_weights = parameters.code_weights

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(ttnn.linear)
            cls_branch.append(ttnn.layer_norm)
            cls_branch.append(ttnn.relu)
        cls_branch.append(ttnn.linear)
        fc_cls = [*cls_branch]

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(ttnn.linear)
            reg_branch.append(ttnn.relu)
        reg_branch.append(ttnn.linear)
        reg_branch = [*reg_branch]

        past_traj_reg_branch = []
        for _ in range(self.num_reg_fcs):
            past_traj_reg_branch.append(ttnn.linear)
            past_traj_reg_branch.append(ttnn.relu)
        past_traj_reg_branch.append(ttnn.linear)
        past_traj_reg_branch = [*past_traj_reg_branch]

        def _get_clones(module, N):
            return [module for i in range(N)]

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (
            (self.transformer.decoder.num_layers + 1) if self.as_two_stage else self.transformer.decoder.num_layers
        )

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.past_traj_reg_branches = _get_clones(past_traj_reg_branch, num_pred)
        else:
            self.cls_branches = [fc_cls for _ in range(num_pred)]
            self.reg_branches = [reg_branch for _ in range(num_pred)]
            self.past_traj_reg_branches = [past_traj_reg_branch for _ in range(num_pred)]
        if not self.as_two_stage:
            self.bev_embedding = ttnn.embedding

    def get_bev_features(self, mlvl_feats, img_metas, prev_bev=None):
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        bev_queries = self.parameters.bev_embedding["weight"]

        bev_mask = ttnn.zeros((bs, self.bev_h, self.bev_w), device=self.device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        bev_pos = self.positional_encoding(bev_mask)
        bev_embed = self.transformer.get_bev_features(
            mlvl_feats,
            bev_queries,
            self.bev_h,
            self.bev_w,
            grid_length=(self.real_h / self.bev_h, self.real_w / self.bev_w),
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            img_metas=img_metas,
        )
        return bev_embed, bev_pos

    def get_detections(
        self,
        bev_embed,
        object_query_embeds=None,
        ref_points=None,
        img_metas=None,
    ):
        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        hs, init_reference, inter_references = self.transformer.get_states_and_refs(
            bev_embed,
            object_query_embeds,
            self.bev_h,
            self.bev_w,
            reference_points=ref_points,
            reg_branches=self.parameters["reg_branches"] if self.with_box_refine else None,
            cls_branches=self.cls_branches if self.as_two_stage else None,
            img_metas=img_metas,
        )
        hs = ttnn.permute(hs, (0, 2, 1, 3))
        outputs_classes = []
        outputs_coords = []
        outputs_trajs = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = ttnn.sigmoid(ref_points)
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)

            outputs_class = ttnn.clone(hs[lvl])
            for i, layer in enumerate(self.cls_branches[lvl]):
                if layer == ttnn.linear:
                    outputs_class = ttnn.linear(
                        outputs_class,
                        self.parameters["cls_branches"][lvl][i]["weight"],
                        bias=self.parameters["cls_branches"][lvl][i]["bias"],
                    )
                elif layer == ttnn.layer_norm:
                    outputs_class = ttnn.layer_norm(
                        outputs_class,
                        weight=self.parameters["cls_branches"][lvl][i]["weight"],
                        bias=self.parameters["cls_branches"][lvl][i]["bias"],
                    )
                else:
                    outputs_class = ttnn.relu(outputs_class)

            tmp = ttnn.clone(hs[lvl])
            for i, layer in enumerate(self.reg_branches[lvl]):
                if layer == ttnn.linear:
                    tmp = ttnn.linear(
                        tmp,
                        self.parameters["reg_branches"][lvl][i]["weight"],
                        bias=self.parameters["reg_branches"][lvl][i]["bias"],
                    )
                else:
                    tmp = ttnn.relu(tmp)

            outputs_past_traj = ttnn.clone(hs[lvl])
            for i, layer in enumerate(self.past_traj_reg_branches[lvl]):
                if layer == ttnn.linear:
                    outputs_past_traj = ttnn.linear(
                        outputs_past_traj,
                        self.parameters["past_traj_reg_branches"][lvl][i]["weight"],
                        bias=self.parameters["past_traj_reg_branches"][lvl][i]["bias"],
                    )
                else:
                    outputs_past_traj = ttnn.relu(outputs_past_traj)

            outputs_past_traj = ttnn.reshape(outputs_past_traj, (tmp.shape[0], -1, self.past_steps + self.fut_steps, 2))

            # TODO: check the shape of reference
            assert reference.shape[-1] == 3

            tmp_temp_1 = ttnn.clone(tmp[..., 0:2])
            tmp_temp_2 = ttnn.clone(tmp[..., 2:4])
            tmp_temp_3 = ttnn.clone(tmp[..., 4:5])
            tmp_temp_4 = ttnn.clone(tmp[..., 5:])
            tmp_temp_1 = ttnn.add(tmp_temp_1, reference[..., 0:2])
            tmp_temp_1 = ttnn.sigmoid(tmp_temp_1)
            tmp_temp_3 = ttnn.add(tmp_temp_3, reference[..., 2:3])
            tmp_temp_3 = ttnn.sigmoid(tmp_temp_3)
            tmp = ttnn.concat([tmp_temp_1, tmp_temp_2, tmp_temp_3, tmp_temp_4], dim=-1)

            last_ref_points = ttnn.concat(
                [tmp[..., 0:2], tmp[..., 4:5]],
                dim=-1,
            )

            tmp_temp_1 = ttnn.clone(tmp[..., 0:1])
            tmp_temp_2 = ttnn.clone(tmp[..., 1:2])
            tmp_temp_3 = ttnn.clone(tmp[..., 2:4])
            tmp_temp_4 = ttnn.clone(tmp[..., 4:5])
            tmp_temp_5 = ttnn.clone(tmp[..., 5:])

            tmp_temp_1 = ttnn.add(ttnn.mul(tmp_temp_1, (self.pc_range[3] - self.pc_range[0])), self.pc_range[0])
            tmp_temp_2 = ttnn.add(ttnn.mul(tmp_temp_2, (self.pc_range[4] - self.pc_range[1])), self.pc_range[1])
            tmp_temp_4 = ttnn.add(ttnn.mul(tmp_temp_4, (self.pc_range[5] - self.pc_range[2])), self.pc_range[2])

            tmp = ttnn.concat([tmp_temp_1, tmp_temp_2, tmp_temp_3, tmp_temp_4, tmp_temp_5], dim=-1)

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
            outputs_trajs.append(outputs_past_traj)

        outputs_classes = ttnn.stack(outputs_classes, dim=0)
        outputs_coords = ttnn.stack(outputs_coords, dim=0)
        outputs_trajs = ttnn.stack(outputs_trajs, dim=0)
        last_ref_points = inverse_sigmoid(last_ref_points)
        outs = {
            "all_cls_scores": outputs_classes,
            "all_bbox_preds": outputs_coords,
            "all_past_traj_preds": outputs_trajs,
            "enc_cls_scores": None,
            "enc_bbox_preds": None,
            "last_ref_points": last_ref_points,
            "query_feats": hs,
        }
        return outs

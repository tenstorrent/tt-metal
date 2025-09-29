# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import copy
import math
import torch
from torch import nn

from models.experimental.uniad.reference.planning_head import PlanningHeadSingleMode
from models.experimental.uniad.reference.pan_segformer_head import PansegformerHead
from models.experimental.uniad.reference.motion_head import MotionHead
from models.experimental.uniad.reference.occ_head import OccHead

from einops import rearrange

from models.experimental.uniad.reference.head import BEVFormerTrackHead
from models.experimental.uniad.reference.resnet import ResNet
from models.experimental.uniad.reference.fpn import FPN
from models.experimental.uniad.reference.grid_mask import GridMask
from models.experimental.uniad.reference.runtime_tracker_base import RuntimeTrackerBase, Instances
from models.experimental.uniad.reference.query_interaction_module import QueryInteractionModule
from models.experimental.uniad.reference.memory_bank import MemoryBank
from models.experimental.uniad.reference.detr_track_3d_coder import DETRTrack3DCoder


class UniAD(nn.Module):
    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(track=1.0, map=1.0, motion=1.0, occ=1.0, planning=1.0),
        use_grid_mask=False,
        img_backbone=None,
        img_neck=None,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        loss_cfg=None,
        qim_args=dict(
            qim_type="QIMBase",
            update_query_pos=False,
            fp_ratio=0.3,
        ),
        mem_args=dict(
            memory_bank_type="MemoryBank",
            memory_bank_score_thresh=0.0,
            memory_bank_len=4,
        ),
        bbox_coder=dict(
            type="DETRTrack3DCoder",
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        ),
        pc_range=None,
        embed_dims=256,
        num_query=900,
        num_classes=10,
        vehicle_id_list=None,
        score_thresh=0.2,
        filter_score_thresh=0.1,
        miss_tolerance=5,
        gt_iou_threshold=0.0,
        freeze_img_backbone=False,
        freeze_img_neck=False,
        freeze_bn=False,
        freeze_bev_encoder=False,
        queue_length=3,
        **kwargs,
    ):
        super().__init__()

        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = BEVFormerTrackHead(
                args=(),
                with_box_refine=True,
                as_two_stage=False,
                num_cls_fcs=2,
                code_weights=None,
                bev_h=50,
                bev_w=50,
                past_steps=4,
                fut_steps=4,
                **{
                    "num_query": 900,
                    "num_classes": 10,
                    "in_channels": 256,
                    "sync_cls_avg_factor": True,
                    "positional_encoding": {
                        "type": "LearnedPositionalEncoding",
                        "num_feats": 128,
                        "row_num_embed": 200,
                        "col_num_embed": 200,
                    },
                    "loss_cls": {
                        "type": "FocalLoss",
                        "use_sigmoid": True,
                        "gamma": 2.0,
                        "alpha": 0.25,
                        "loss_weight": 2.0,
                    },
                    "loss_bbox": {"type": "L1Loss", "loss_weight": 0.25},
                    "loss_iou": {"type": "GIoULoss", "loss_weight": 0.0},
                    "train_cfg": None,
                    "test_cfg": None,
                },
            )

        if img_backbone:
            self.img_backbone = ResNet(
                depth=101,
                in_channels=3,
                stem_channels=None,
                base_channels=64,
                num_stages=4,
                strides=(1, 2, 2, 2),
                dilations=(1, 1, 1, 1),
                out_indices=(1, 2, 3),
                style="caffe",
                deep_stem=False,
                avg_down=False,
                frozen_stages=4,
                conv_cfg=None,
                norm_cfg={"type": "BN2d", "requires_grad": False},
                norm_eval=True,
                dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
                stage_with_dcn=(False, False, True, True),
                plugins=None,
                with_cp=False,
                zero_init_residual=True,
                pretrained=None,
                init_cfg=None,
            )
        if img_neck is not None:
            self.img_neck = FPN(
                in_channels=[512, 1024, 2048],
                out_channels=256,
                start_level=0,
                add_extra_convs="on_output",
                num_outs=4,
                relu_before_extra_convs=True,
            )

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if pretrained is None:
            img_pretrained = None
            pts_pretrained = None
        elif isinstance(pretrained, dict):
            img_pretrained = pretrained.get("img", None)
            pts_pretrained = pretrained.get("pts", None)
        else:
            raise ValueError(f"pretrained should be a dict, got {type(pretrained)}")

        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length
        if freeze_img_backbone:
            if freeze_bn:
                self.img_backbone.eval()
            for param in self.img_backbone.parameters():
                param.requires_grad = False

        if freeze_img_neck:
            if freeze_bn:
                self.img_neck.eval()
            for param in self.img_neck.parameters():
                param.requires_grad = False

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = nn.Embedding(
            self.num_query + 1, self.embed_dims * 2
        )  # the final one is ego query, which constantly models ego-vehicle
        self.reference_points = nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = QueryInteractionModule(
            # qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
            update_query_pos=qim_args["update_query_pos"],
        )

        # {'type': 'DETRTrack3DCoder', 'post_center_range': [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0], 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'max_num': 300, 'num_classes': 10, 'score_threshold': 0.0, 'with_nms': False, 'iou_thres': 0.3}
        self.bbox_coder = DETRTrack3DCoder(
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
            max_num=300,
            num_classes=10,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3,
        )

        self.memory_bank = MemoryBank(
            # mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
            memory_bank_len=mem_args["memory_bank_len"],
        )
        self.mem_bank_len = 0 if self.memory_bank is None else self.memory_bank.max_his_length

        self.test_track_instances = None
        self.l2g_r_mat = None
        self.l2g_t = None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h, self.bev_w = self.pts_bbox_head.bev_h, self.pts_bbox_head.bev_w
        self.freeze_bev_encoder = freeze_bev_encoder
        if seg_head:
            self.seg_head = PansegformerHead(
                bev_h=50,
                bev_w=50,
                canvas_size=(50, 50),
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                with_box_refine=True,
                as_two_stage=False,
                **{
                    "num_query": 300,
                    "num_classes": 4,
                    "num_things_classes": 3,
                    "num_stuff_classes": 1,
                    "in_channels": 2048,
                    "sync_cls_avg_factor": True,
                    "positional_encoding": {
                        "type": "SinePositionalEncoding",
                        "num_feats": 128,
                        "normalize": True,
                        "offset": -0.5,
                    },
                    "loss_cls": {
                        "type": "FocalLoss",
                        "use_sigmoid": True,
                        "gamma": 2.0,
                        "alpha": 0.25,
                        "loss_weight": 2.0,
                    },
                    "loss_bbox": {"type": "L1Loss", "loss_weight": 5.0},
                    "loss_iou": {"type": "GIoULoss", "loss_weight": 2.0},
                },
            )
        if occ_head:
            occflow_grid_conf = {
                "xbound": [-50.0, 50.0, 0.5],
                "ybound": [-50.0, 50.0, 0.5],
                "zbound": [-10.0, 10.0, 20.0],
            }
            self.occ_head = OccHead(
                grid_conf=occflow_grid_conf,
                ignore_index=255,
                bev_proj_dim=256,
                bev_proj_nlayers=4,
                # Transformer
                attn_mask_thresh=0.3,
            )
        if motion_head:
            self.motion_head = MotionHead(
                args=(),
                predict_steps=12,
                transformerlayers={
                    "type": "MotionTransformerDecoder",
                    "pc_range": [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                    "embed_dims": 256,
                    "num_layers": 3,
                    "transformerlayers": {
                        "type": "MotionTransformerAttentionLayer",
                        "batch_first": True,
                        "attn_cfgs": [
                            {
                                "type": "MotionDeformableAttention",
                                "num_steps": 12,
                                "embed_dims": 256,
                                "num_levels": 1,
                                "num_heads": 8,
                                "num_points": 4,
                                "sample_index": -1,
                            }
                        ],
                        "feedforward_channels": 512,
                        "operation_order": ("cross_attn", "norm", "ffn", "norm"),
                    },
                },
                bbox_coder=None,
                num_cls_fcs=3,
                bev_h=50,
                bev_w=50,
                embed_dims=256,
                num_anchor=6,
                det_layer_num=6,
                group_id_list=[[0, 1, 2, 3, 4], [6, 7], [8], [5, 9]],
                pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                use_nonlinear_optimizer=True,
                anchor_info_path="models/experimental/uniad/reference/motion_anchor_infos_mode6.pkl",
                loss_traj={
                    "type": "TrajLoss",
                    "use_variance": True,
                    "cls_loss_weight": 0.5,
                    "nll_loss_weight": 0.5,
                    "loss_weight_minade": 0.0,
                    "loss_weight_minfde": 0.25,
                },
                num_classes=10,
                vehicle_id_list=[0, 1, 2, 3, 4, 6, 7],
                num_query=300,
                predict_modes=6,
            )
        if planning_head:
            self.planning_head = PlanningHeadSingleMode(
                bev_h=50, bev_w=50, embed_dims=256, planning_steps=6, planning_eval=True
            )

    @property
    def with_planning_head(self):
        return hasattr(self, "planning_head") and self.planning_head is not None

    @property
    def with_occ_head(self):
        return hasattr(self, "occ_head") and self.occ_head is not None

    @property
    def with_motion_head(self):
        return hasattr(self, "motion_head") and self.motion_head is not None

    @property
    def with_seg_head(self):
        return hasattr(self, "seg_head") and self.seg_head is not None

    def forward(self, return_loss=True, **kwargs):
        return self.forward_test(**kwargs)

    def forward_test(
        self,
        img=None,
        img_metas=None,
        l2g_t=None,
        l2g_r_mat=None,
        timestamp=None,
        gt_lane_labels=None,
        gt_lane_masks=None,
        rescale=False,
        # planning gt(for evaluation only)
        sdc_planning=None,
        sdc_planning_mask=None,
        command=None,
        # Occ_gt (for evaluation only)
        gt_segmentation=None,
        gt_instance=None,
        gt_occ_img_is_valid=None,
        **kwargs,
    ):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            # the first sample of each scene is truncated
            self.prev_frame_info["prev_bev"] = None
        # update idx
        self.prev_frame_info["scene_token"] = img_metas[0][0]["scene_token"]

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]["can_bus"][-1])
        # first frame
        if self.prev_frame_info["scene_token"] is None:
            img_metas[0][0]["can_bus"][:3] = 0
            img_metas[0][0]["can_bus"][-1] = 0
        # following frames
        else:
            img_metas[0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle

        img = img[0]
        img_metas = img_metas[0]
        timestamp = timestamp[0] if timestamp is not None else None

        result = [dict() for i in range(len(img_metas))]
        result_track = self.simple_test_track(img, l2g_t, l2g_r_mat, img_metas, timestamp)

        # Upsample bev for tiny model
        result_track[0] = self.upsample_bev_if_tiny(result_track[0])

        bev_embed = result_track[0]["bev_embed"]

        if self.with_seg_head:
            result_seg = self.seg_head.forward_test(bev_embed, gt_lane_labels, gt_lane_masks, img_metas, rescale)

        if self.with_motion_head:
            result_motion, outs_motion = self.motion_head.forward_test(
                bev_embed, outs_track=result_track[0], outs_seg=result_seg[0]
            )
            outs_motion["bev_pos"] = result_track[0]["bev_pos"]

        outs_occ = dict()
        if self.with_occ_head:
            occ_no_query = outs_motion["track_query"].shape[1] == 0
            outs_occ = self.occ_head(
                bev_embed,
                outs_motion,
                no_query=occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid,
            )
            result[0]["occ"] = outs_occ

        if self.with_planning_head:
            planning_gt = dict(
                segmentation=gt_segmentation,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
            )
            result_planning = self.planning_head.forward_test(bev_embed, outs_motion, outs_occ, command)
            result[0]["planning"] = dict(
                planning_gt=planning_gt,
                result_planning=result_planning,
            )

        pop_track_list = ["prev_bev", "bev_pos", "bev_embed", "track_query_embeddings", "sdc_embedding"]
        result_track[0] = pop_elem_in_result(result_track[0], pop_track_list)

        if self.with_seg_head:
            result_seg[0] = pop_elem_in_result(result_seg[0], pop_list=["pts_bbox", "args_tuple"])
        if self.with_motion_head:
            result_motion[0] = pop_elem_in_result(result_motion[0])
        if self.with_occ_head:
            result[0]["occ"] = pop_elem_in_result(
                result[0]["occ"],
                pop_list=[
                    "seg_out_mask",
                    "flow_out",
                    "future_states_occ",
                    "pred_ins_masks",
                    "pred_raw_occ",
                    "pred_ins_logits",
                    "pred_ins_sigmoid",
                ],
            )

        for i, res in enumerate(result):
            res["token"] = img_metas[i]["sample_idx"]
            res.update(result_track[i])
            if self.with_motion_head:
                res.update(result_motion[i])
            if self.with_seg_head:
                res.update(result_seg[i])

        return result

    def extract_img_feat(self, img, len_queue=None):
        if img is None:
            return None
        assert img.dim() == 5

        B, N, C, H, W = img.size()
        img = img.reshape(B * N, C, H, W)
        if self.use_grid_mask:
            img = self.grid_mask(img)
        img_feats = self.img_backbone(img)
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
        if True:  # self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            _, c, h, w = img_feat.size()
            if len_queue is not None:
                img_feat_reshaped = img_feat.view(B // len_queue, len_queue, N, c, h, w)
            else:
                img_feat_reshaped = img_feat.view(B, N, c, h, w)
            img_feats_reshaped.append(img_feat_reshaped)
        return img_feats_reshaped

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape  # (300, 256 * 2)
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(query[..., : dim // 2])

        # init boxes: xy, wl, z, h, sin, cos, vx, vy, vz
        pred_boxes_init = torch.zeros((len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.query = query

        track_instances.output_embedding = torch.zeros((num_queries, dim >> 1), device=device)

        track_instances.obj_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full((len(track_instances),), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros((len(track_instances),), dtype=torch.long, device=device)

        track_instances.iou = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros((len(track_instances),), dtype=torch.float, device=device)
        # xy, wl, z, h, sin, cos, vx, vy, vz
        track_instances.pred_boxes = pred_boxes_init

        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.num_classes), dtype=torch.float, device=device
        )

        mem_bank_len = self.mem_bank_len
        track_instances.mem_bank = torch.zeros(
            (len(track_instances), mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device,
        )
        track_instances.mem_padding_mask = torch.ones(
            (len(track_instances), mem_bank_len), dtype=torch.bool, device=device
        )
        track_instances.save_period = torch.zeros((len(track_instances),), dtype=torch.float32, device=device)

        return track_instances.to(self.query_embedding.weight.device)

    # Generate bev using bev_encoder in BEVFormer
    def get_bevs(self, imgs, img_metas, prev_img=None, prev_img_metas=None, prev_bev=None):
        if prev_img is not None and prev_img_metas is not None:
            assert prev_bev is None
            prev_bev = self.get_history_bev(prev_img, prev_img_metas)

        img_feats = self.extract_img_feat(img=imgs)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                    mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
                )
        else:
            bev_embed, bev_pos = self.pts_bbox_head.get_bev_features(
                mlvl_feats=img_feats, img_metas=img_metas, prev_bev=prev_bev
            )

        if bev_embed.shape[1] == self.bev_h * self.bev_w:
            bev_embed = bev_embed.permute(1, 0, 2)

        assert bev_embed.shape[0] == self.bev_h * self.bev_w
        return bev_embed, bev_pos

    def select_active_track_query(self, track_instances, active_index, img_metas, with_mask=True):
        result_dict = self._track_instances2results(track_instances[active_index], img_metas, with_mask=with_mask)
        result_dict["track_query_embeddings"] = track_instances.output_embedding[active_index][
            result_dict["bbox_index"]
        ][result_dict["mask"]]
        result_dict["track_query_matched_idxes"] = track_instances.matched_gt_idxes[active_index][
            result_dict["bbox_index"]
        ][result_dict["mask"]]
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        out = dict()
        result_dict = self._track_instances2results(sdc_instance, img_metas, with_mask=False)
        out["sdc_boxes_3d"] = result_dict["boxes_3d"]
        out["sdc_scores_3d"] = result_dict["scores_3d"]
        out["sdc_track_scores"] = result_dict["track_scores"]
        out["sdc_track_bbox_results"] = result_dict["track_bbox_results"]
        out["sdc_embedding"] = sdc_instance.output_embedding[0]
        return out

    def upsample_bev_if_tiny(self, outs_track):
        if outs_track["bev_embed"].size(0) == 100 * 100:
            # For tiny model
            # bev_emb
            bev_embed = outs_track["bev_embed"]  # [10000, 1, 256]
            dim, _, _ = bev_embed.size()
            w = h = int(math.sqrt(dim))
            assert h == w == 100

            bev_embed = rearrange(bev_embed, "(h w) b c -> b c h w", h=h, w=w)  # [1, 256, 100, 100]
            bev_embed = nn.Upsample(scale_factor=2)(bev_embed)  # [1, 256, 200, 200]
            bev_embed = rearrange(bev_embed, "b c h w -> (h w) b c")
            outs_track["bev_embed"] = bev_embed

            # prev_bev
            prev_bev = outs_track.get("prev_bev", None)
            if prev_bev is not None:
                if self.training:
                    #  [1, 10000, 256]
                    prev_bev = rearrange(prev_bev, "b (h w) c -> b c h w", h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, "b c h w -> b (h w) c")
                    outs_track["prev_bev"] = prev_bev
                else:
                    #  [10000, 1, 256]
                    prev_bev = rearrange(prev_bev, "(h w) b c -> b c h w", h=h, w=w)
                    prev_bev = nn.Upsample(scale_factor=2)(prev_bev)  # [1, 256, 200, 200]
                    prev_bev = rearrange(prev_bev, "b c h w -> (h w) b c")
                    outs_track["prev_bev"] = prev_bev

            # bev_pos
            bev_pos = outs_track["bev_pos"]  # [1, 256, 100, 100]
            bev_pos = nn.Upsample(scale_factor=2)(bev_pos)  # [1, 256, 200, 200]
            outs_track["bev_pos"] = bev_pos
        return outs_track

    def _forward_single_frame_inference(
        self,
        img,
        img_metas,
        track_instances,
        prev_bev=None,
        l2g_r1=None,
        l2g_t1=None,
        l2g_r2=None,
        l2g_t2=None,
        time_delta=None,
    ):
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]

        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = active_inst.ref_pts
            velo = active_inst.pred_boxes[:, -2:]
            ref_pts = self.velo_update(ref_pts, velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2, time_delta=time_delta)
            ref_pts = ref_pts.squeeze(0)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(active_inst.query[..., : dim // 2])
            active_inst.ref_pts[..., :2] = ref_pts[..., :2]

        track_instances = Instances.cat([other_inst, active_inst])

        # NOTE: You can replace BEVFormer with other BEV encoder and provide bev_embed here
        bev_embed, bev_pos = self.get_bevs(img, img_metas, prev_bev=prev_bev)
        det_output = self.pts_bbox_head.get_detections(
            bev_embed,
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts,
            img_metas=img_metas,
        )
        output_classes = det_output["all_cls_scores"]
        output_coords = det_output["all_bbox_preds"]
        last_ref_pts = det_output["last_ref_points"]
        query_feats = det_output["query_feats"]

        out = {
            "pred_logits": output_classes,
            "pred_boxes": output_coords,
            "ref_pts": last_ref_pts,
            "bev_embed": bev_embed,
            "query_embeddings": query_feats,
            "all_past_traj_preds": det_output["all_past_traj_preds"],
            "bev_pos": bev_pos,
        }

        track_scores = output_classes[-1, 0, :].sigmoid().max(dim=-1).values
        track_instances.scores = track_scores
        track_instances.pred_logits = output_classes[-1, 0]  # [300, num_cls]
        track_instances.pred_boxes = output_coords[-1, 0]  # [300, box_dim]
        track_instances.output_embedding = query_feats[-1][0]  # [300, feat_dim]
        track_instances.ref_pts = last_ref_pts[0]
        # hard_code: assume the 901 query is sdc query
        track_instances.obj_idxes[900] = -2
        """ update track base """
        self.track_base.update(track_instances, None)

        active_index = (track_instances.obj_idxes >= 0) & (
            track_instances.scores >= self.track_base.filter_score_thresh
        )  # filter out sleep objects
        out.update(self.select_active_track_query(track_instances, active_index, img_metas))
        out.update(self.select_sdc_track_query(track_instances[track_instances.obj_idxes == -2], img_metas))

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)

        tmp = {}
        tmp["init_track_instances"] = self._generate_empty_tracks()
        tmp["track_instances"] = track_instances
        out_track_instances = self.query_interact(tmp)
        out["track_instances_fordet"] = track_instances
        out["track_instances"] = out_track_instances
        out["track_obj_idxes"] = track_instances.obj_idxes
        return out

    def simple_test_track(
        self,
        img=None,
        l2g_t=None,
        l2g_r_mat=None,
        img_metas=None,
        timestamp=None,
    ):
        bs = img.size(0)

        if self.test_track_instances is None or img_metas[0]["scene_token"] != self.scene_token:
            self.timestamp = timestamp
            self.scene_token = img_metas[0]["scene_token"]
            self.prev_bev = None
            track_instances = self._generate_empty_tracks()
            time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2 = None, None, None, None, None
        else:
            track_instances = self.test_track_instances
            time_delta = timestamp - self.timestamp
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            l2g_r2 = l2g_r_mat
            l2g_t2 = l2g_t

        self.timestamp = timestamp
        self.l2g_t = l2g_t
        self.l2g_r_mat = l2g_r_mat

        prev_bev = self.prev_bev
        frame_res = self._forward_single_frame_inference(
            img,
            img_metas,
            track_instances,
            prev_bev,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta,
        )

        self.prev_bev = frame_res["bev_embed"]
        track_instances = frame_res["track_instances"]
        track_instances_fordet = frame_res["track_instances_fordet"]

        self.test_track_instances = track_instances
        results = [dict()]
        get_keys = [
            "bev_embed",
            "bev_pos",
            "track_query_embeddings",
            "track_bbox_results",
            "boxes_3d",
            "scores_3d",
            "labels_3d",
            "track_scores",
            "track_ids",
        ]
        if self.with_motion_head:
            get_keys += ["sdc_boxes_3d", "sdc_scores_3d", "sdc_track_scores", "sdc_track_bbox_results", "sdc_embedding"]
        results[0].update({k: frame_res[k] for k in get_keys})
        results = self._det_instances2results(track_instances_fordet, results, img_metas)
        return results

    def _track_instances2results(self, track_instances, img_metas, with_mask=True):
        bbox_dict = dict(
            cls_scores=track_instances.pred_logits,
            bbox_preds=track_instances.pred_boxes,
            track_scores=track_instances.scores,
            obj_idxes=track_instances.obj_idxes,
        )

        bboxes_dict = self.bbox_coder.decode(bbox_dict, with_mask=with_mask, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]
        bbox_index = bboxes_dict["bbox_index"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = dict(
            boxes_3d=bboxes.to("cpu"),
            scores_3d=scores.cpu(),
            labels_3d=labels.cpu(),
            track_scores=track_scores.cpu(),
            bbox_index=bbox_index.cpu(),
            track_ids=obj_idxes.cpu(),
            mask=bboxes_dict["mask"].cpu(),
            track_bbox_results=[
                [bboxes.to("cpu"), scores.cpu(), labels.cpu(), bbox_index.cpu(), bboxes_dict["mask"].cpu()]
            ],
        )
        return result_dict

    def _det_instances2results(self, instances, results, img_metas):
        if instances.pred_logits.numel() == 0:
            return [None]
        bbox_dict = dict(
            cls_scores=instances.pred_logits,
            bbox_preds=instances.pred_boxes,
            track_scores=instances.scores,
            obj_idxes=instances.obj_idxes,
        )
        bboxes_dict = self.bbox_coder.decode(bbox_dict, img_metas=img_metas)[0]
        bboxes = bboxes_dict["bboxes"]
        bboxes = img_metas[0]["box_type_3d"](bboxes, 9)
        labels = bboxes_dict["labels"]
        scores = bboxes_dict["scores"]

        track_scores = bboxes_dict["track_scores"]
        obj_idxes = bboxes_dict["obj_idxes"]
        result_dict = results[0]
        result_dict_det = dict(
            boxes_3d_det=bboxes.to("cpu"),
            scores_3d_det=scores.cpu(),
            labels_3d_det=labels.cpu(),
        )
        if result_dict is not None:
            result_dict.update(result_dict_det)
        else:
            result_dict = None

        return [result_dict]


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith("query") or k.endswith("query_pos") or k.endswith("embedding"):
            task_result.pop(k)

    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result

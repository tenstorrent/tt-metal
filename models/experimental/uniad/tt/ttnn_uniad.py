# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn


from models.experimental.uniad.tt.ttnn_head import TtBEVFormerTrackHead
from models.experimental.uniad.tt.ttnn_resnet import TtResNet
from models.experimental.uniad.reference.fpn import FPN
from models.experimental.uniad.reference.runtime_tracker_base import RuntimeTrackerBase
from models.experimental.uniad.tt.ttnn_query_interaction import TtQueryInteractionModule
from models.experimental.uniad.tt.ttnn_memory_bank import TtMemoryBank
from models.experimental.uniad.reference.detr_track_3d_coder import DETRTrack3DCoder
from models.experimental.uniad.tt.ttnn_planning_head import TtPlanningHeadSingleMode
from models.experimental.uniad.reference.pan_segformer_head import PansegformerHead
from models.experimental.uniad.tt.ttnn_motion_head import TtMotionHead
from models.experimental.uniad.tt.ttnn_occ_head import TtOccHead


class TtUniAD:
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """

    def __init__(
        self,
        parameters=None,
        device=None,
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
            merger_dropout=0,
            update_query_pos=False,
            fp_ratio=0.3,
            random_drop=0.1,
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
        # super().__init__()
        # ------mvx starting

        if pts_bbox_head:
            # pts_train_cfg = train_cfg.pts if train_cfg else None
            # pts_bbox_head.update(train_cfg=pts_train_cfg)
            # pts_test_cfg = test_cfg.pts if test_cfg else None
            # pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = TtBEVFormerTrackHead(
                parameters=parameters.pts_bbox_head,
                device=device,
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
            self.img_backbone = TtResNet(
                conv_args=parameters.img_backbone.conv_args,
                conv_pth=parameters.img_backbone,
                device=device,
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
                # norm_cfg={"type": "BN2d", "requires_grad": False},
                # norm_eval=True,
                dcn={"type": "DCNv2", "deform_groups": 1, "fallback_on_stride": False},
                stage_with_dcn=(False, False, True, True),
                # plugins=None,
                # with_cp=False,
                # zero_init_residual=True,
                pretrained=None,
                init_cfg=None,
            )
        if img_neck is not None:
            self.img_neck = FPN(
                in_channels=[512, 1024, 2048],
                out_channels=256,
                num_outs=4,
            )

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # if pretrained is None:
        #     img_pretrained = None
        #     pts_pretrained = None
        # elif isinstance(pretrained, dict):
        #     img_pretrained = pretrained.get("img", None)
        #     pts_pretrained = pretrained.get("pts", None)
        # else:
        #     raise ValueError(f"pretrained should be a dict, got {type(pretrained)}")

        # if self.with_img_backbone:
        #     if img_pretrained is not None:
        #         warnings.warn("DeprecationWarning: pretrained is a deprecated " "key, please consider using init_cfg.")
        #         self.img_backbone.init_cfg = dict(type="Pretrained", checkpoint=img_pretrained)

        # -------------- mvx complte

        # self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.vehicle_id_list = vehicle_id_list
        self.pc_range = pc_range
        self.queue_length = queue_length
        # if freeze_img_backbone:
        #     if freeze_bn:
        #         self.img_backbone.eval()
        #     for param in self.img_backbone.parameters():
        #         param.requires_grad = False

        # if freeze_img_neck:
        #     if freeze_bn:
        #         self.img_neck.eval()
        #     for param in self.img_neck.parameters():
        #         param.requires_grad = False

        # temporal
        self.video_test_mode = video_test_mode
        assert self.video_test_mode

        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.query_embedding = ttnn.embedding
        self.reference_points = ttnn.linear  # nn.Linear(self.embed_dims, 3)

        self.mem_bank_len = mem_args["memory_bank_len"]
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance,
        )  # hyper-param for removing inactive queries

        self.query_interact = TtQueryInteractionModule(
            # qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
            update_query_pos=qim_args["update_query_pos"],
            params=parameters.query_interact,
            device=device,
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

        self.memory_bank = TtMemoryBank(
            # mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims,
            memory_bank_len=mem_args["memory_bank_len"],
            params=parameters.memory_bank,
            device=device,
        )
        self.mem_bank_len = 0 if self.memory_bank is None else self.memory_bank.max_his_length
        # self.criterion = build_loss(loss_cfg)
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
            self.occ_head = TtOccHead(
                device=device,
                # grid_conf=occflow_grid_conf,
                ignore_index=255,
                # bev_proj_dim=256,
                # bev_proj_nlayers=4,
                # Transformer
                # attn_mask_thresh=0.3,
            )
        if motion_head:
            self.motion_head = TtMotionHead(
                parameters=parameters.motion_head,
                device=device,
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
                        "ffn_dropout": 0.1,
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
            self.planning_head = TtPlanningHeadSingleMode(
                device=device,
                conv_pt=parameters["planning_head"]["model_args"],
                parameters=parameters.planning_head,
                bev_h=50,
                bev_w=50,
                embed_dims=256,
                planning_steps=6,
                planning_eval=True,
            )

        self.task_loss_weight = task_loss_weight
        assert set(task_loss_weight.keys()) == {"track", "occ", "motion", "map", "planning"}

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

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """

        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)

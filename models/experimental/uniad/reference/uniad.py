# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


# from mmcv.runner import auto_fp16
# from mmdet.models import DETECTORS
import copy

# from ..dense_heads.seg_head_plugin import IOU
from .uniad_track import UniADTrack
from models.experimental.uniad.reference.planning_head import PlanningHeadSingleMode
from models.experimental.uniad.reference.pan_segformer_head import PansegformerHead
from models.experimental.uniad.reference.motion_head import MotionHead
from models.experimental.uniad.reference.occ_head import OccHead

# from mmdet.models.builder import build_head


class UniAD(UniADTrack):
    """
    UniAD: Unifying Detection, Tracking, Segmentation, Motion Forecasting, Occupancy Prediction and Planning for Autonomous Driving
    """

    def __init__(
        self,
        seg_head=None,
        motion_head=None,
        occ_head=None,
        planning_head=None,
        task_loss_weight=dict(track=1.0, map=1.0, motion=1.0, occ=1.0, planning=1.0),
        **kwargs,
    ):
        super(UniAD, self).__init__(**kwargs)
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
                anchor_info_path="models/experimental/uniad/reference/motion_head/motion_anchor_infos_mode6.pkl",
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
        """Test function"""

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
            outs_occ = self.occ_head.forward_test(
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


def pop_elem_in_result(task_result: dict, pop_list: list = None):
    # assert False, "pop_elem_in_result"  # added by me
    all_keys = list(task_result.keys())
    for k in all_keys:
        if k.endswith("query") or k.endswith("query_pos") or k.endswith("embedding"):
            task_result.pop(k)

    if pop_list is not None:
        for pop_k in pop_list:
            task_result.pop(pop_k, None)
    return task_result

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import copy
import ttnn
import os
from scipy.optimize import linear_sum_assignment
from models.experimental.vadv2.tt.tt_backbone import TtResnet50
from models.experimental.vadv2.reference.planning_metric import PlanningMetric
from models.experimental.vadv2.tt.tt_fpn import TtFPN
from models.experimental.vadv2.tt.tt_head import TtVADHead


def bbox3d2result(bboxes, scores, labels, attrs=None):
    result_dict = dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)

    if attrs is not None:
        result_dict["attrs_3d"] = attrs

    return result_dict


class TtVAD:
    def __init__(
        self,
        device,
        params,
        use_grid_mask=False,
        pts_voxel_layer=None,
        pts_voxel_encoder=None,
        pts_middle_encoder=None,
        pts_fusion_layer=None,
        img_backbone=None,
        pts_backbone=None,
        img_neck=None,
        pts_neck=None,
        pts_bbox_head=None,
        img_roi_head=None,
        img_rpn_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        video_test_mode=False,
        fut_ts=6,
        fut_mode=6,
    ):
        super(TtVAD, self).__init__()
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.params = params
        self.device = device

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            "prev_bev": None,
            "scene_token": None,
            "prev_pos": 0,
            "prev_angle": 0,
        }
        self.with_img_neck = True
        self.planning_metric = None
        self.img_backbone = TtResnet50(params.conv_args["img_backbone"], params.img_backbone, device)
        self.img_neck = TtFPN(params.conv_args["img_neck"], params.img_neck, device)
        self.pts_bbox_head = TtVADHead(
            params=params,
            device=device,
            with_box_refine=True,
            as_two_stage=False,
            transformer=True,
            bbox_coder={
                "type": "CustomNMSFreeCoder",
                "post_center_range": [-20, -35, -10.0, 20, 35, 10.0],
                "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                "max_num": 100,
                "voxel_size": [0.15, 0.15, 4],
                "num_classes": 10,
            },
            num_cls_fcs=2,
            code_weights=None,
            bev_h=100,
            bev_w=100,
            fut_ts=6,
            fut_mode=6,
            map_bbox_coder={
                "type": "MapNMSFreeCoder",
                "post_center_range": [-20, -35, -20, -35, 20, 35, 20, 35],
                "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                "max_num": 50,
                "voxel_size": [0.15, 0.15, 4],
                "num_classes": 3,
            },
            map_num_query=900,
            map_num_classes=3,
            map_num_vec=100,
            map_num_pts_per_vec=20,
            map_num_pts_per_gt_vec=20,
            map_query_embed_type="instance_pts",
            map_transform_method="minmax",
            map_gt_shift_pts_pattern="v2",
            map_dir_interval=1,
            map_code_size=2,
            map_code_weights=[1.0, 1.0, 1.0, 1.0],
            tot_epoch=12,
            use_traj_lr_warmup=False,
            motion_decoder=True,
            motion_map_decoder=True,
            use_pe=True,
            motion_det_score=None,
            map_thresh=0.5,
            dis_thresh=0.2,
            pe_normalization=True,
            ego_his_encoder=None,
            ego_fut_mode=3,
            ego_agent_decoder=True,
            ego_map_decoder=True,
            query_thresh=0.0,
            query_use_fix_pad=False,
            ego_lcf_feat_idx=None,
            valid_fut_ts=6,
        )
        self.valid_fut_ts = self.pts_bbox_head.valid_fut_ts

    def extract_img_feat(self, img, img_metas, len_queue=None):
        B = img.shape[0]

        if img is not None:
            if img.shape[0] == 1 and len(img.shape) == 5:
                img = ttnn.squeeze(img, 0)

            elif len(img.shape) == 4 and img.shape[0] > 1:
                B, N, C, H, W = img.shape
                img = ttnn.reshape(img, (B * N, C, H, W))
            img = ttnn.permute(img, (0, 2, 3, 1))
            N, H, W, C = img.shape
            batch_size = img.shape[0]
            img = ttnn.reshape(img, (1, 1, N * H * W, C))

            img_feats = self.img_backbone(img, batch_size=batch_size)

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        img_feats_reshaped = []
        for img_feat in img_feats:
            img_feat = ttnn.unsqueeze(img_feat, 0)
            img_feat = ttnn.to_layout(img_feat, layout=ttnn.ROW_MAJOR_LAYOUT)
            img_feat = ttnn.sharded_to_interleaved(img_feat)
            img_feat = ttnn.reshape(img_feat, (6, 12, 20, img_feat.shape[3]))
            img_feat = ttnn.permute(img_feat, (0, 3, 1, 2))
            BN, C, H, W = img_feat.shape
            if len_queue is not None:
                img_feat = ttnn.reshape(img_feat, (int(B / len_queue), len_queue, int(BN / B), C, H, W))
                img_feats_reshaped.append(img_feat)
            else:
                img_feat = ttnn.reshape(img_feat, (B, int(BN / B), C, H, W))
                img_feats_reshaped.append(img_feat)
        ttnn.deallocate(img_feats[0])
        return img_feats_reshaped

    def extract_feat(self, img, img_metas=None, len_queue=None):
        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)

        return img_feats

    def __call__(self, return_loss=True, **kwargs):
        return self.forward_test(**kwargs)

    def forward_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs,
    ):
        for var, name in [(img_metas, "img_metas")]:
            if not isinstance(var, list):
                raise TypeError("{} must be a list, but got {}".format(name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0][0]["scene_token"] != self.prev_frame_info["scene_token"]:
            self.prev_frame_info["prev_bev"] = None
        self.prev_frame_info["scene_token"] = img_metas[0][0][0]["scene_token"]

        if not self.video_test_mode:
            self.prev_frame_info["prev_bev"] = None

        tmp_pos = copy.deepcopy(img_metas[0][0][0]["can_bus"][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0][0]["can_bus"][-1])
        if self.prev_frame_info["prev_bev"] is not None:
            img_metas[0][0][0]["can_bus"][:3] -= self.prev_frame_info["prev_pos"]
            img_metas[0][0][0]["can_bus"][-1] -= self.prev_frame_info["prev_angle"]
        else:
            img_metas[0][0][0]["can_bus"][-1] = 0
            img_metas[0][0][0]["can_bus"][:3] = 0
        img = ttnn.unsqueeze(img[0][0], 0)
        new_prev_bev, bbox_results = self.simple_test(
            img_metas=img_metas[0][0],
            img=img,
            prev_bev=self.prev_frame_info["prev_bev"],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0][0],
            ego_fut_trajs=ego_fut_trajs[0][0],
            ego_fut_cmd=ego_fut_cmd[0][0],
            ego_lcf_feat=ego_lcf_feat[0][0],
            gt_attr_labels=gt_attr_labels,
            **kwargs,
        )
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info["prev_pos"] = tmp_pos
        self.prev_frame_info["prev_angle"] = tmp_angle
        self.prev_frame_info["prev_bev"] = new_prev_bev

        return bbox_results

    def simple_test(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs,
    ):
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_list = self.simple_test_pts(
            img_feats,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        x,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        x[0] = ttnn.to_layout(x[0], layout=ttnn.TILE_LAYOUT)
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev, ego_his_trajs=None, ego_lcf_feat=None)

        outs["bev_embed"] = ttnn.to_torch(outs["bev_embed"]).float()
        outs["all_cls_scores"] = ttnn.to_torch(outs["all_cls_scores"]).float()
        outs["all_bbox_preds"] = ttnn.to_torch(outs["all_bbox_preds"]).float()
        outs["all_traj_preds"] = ttnn.to_torch(outs["all_traj_preds"]).float()
        outs["all_traj_cls_scores"] = ttnn.to_torch(outs["all_traj_cls_scores"]).float()
        outs["map_all_cls_scores"] = ttnn.to_torch(outs["map_all_cls_scores"]).float()
        outs["map_all_bbox_preds"] = ttnn.to_torch(outs["map_all_bbox_preds"]).float()
        outs["map_all_pts_preds"] = ttnn.to_torch(outs["map_all_pts_preds"]).float()
        outs["ego_fut_preds"] = ttnn.to_torch(outs["ego_fut_preds"]).float()

        save_path = "models/experimental/vadv2/tt/dumps"
        os.makedirs(save_path, exist_ok=True)

        keys_to_save = [
            "bev_embed",
            "all_cls_scores",
            "all_bbox_preds",
            "all_traj_preds",
            "all_traj_cls_scores",
            "map_all_cls_scores",
            "map_all_bbox_preds",
            "map_all_pts_preds",
            "ego_fut_preds",
        ]

        for key in keys_to_save:
            tensor = outs[key]
            torch.save(tensor, os.path.join(save_path, f"{key}.pt"))

        bbox_results = self.post_process_with_metrics(
            outs,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            gt_attr_labels,
            ego_fut_trajs,
            ego_fut_cmd,
            fut_valid_flag,
            rescale,
        )

        return outs["bev_embed"], bbox_results

    def post_process_with_metrics(
        self,
        outs,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        gt_attr_labels,
        ego_fut_trajs,
        ego_fut_cmd,
        fut_valid_flag,
        rescale=False,
    ):
        mapped_class_names = [
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ]

        bbox_list = self.pts_bbox_head.get_bboxes(outs, img_metas, rescale=rescale)

        bbox_results = []
        for i, (bboxes, scores, labels, trajs, map_bboxes, map_scores, map_labels, map_pts) in enumerate(bbox_list):
            bbox_result = bbox3d2result(bboxes, scores, labels)
            bbox_result["trajs_3d"] = trajs.cpu()
            map_bbox_result = self.map_pred2result(map_bboxes, map_scores, map_labels, map_pts)
            bbox_result.update(map_bbox_result)
            bbox_result["ego_fut_preds"] = outs["ego_fut_preds"][i].cpu()
            bbox_result["ego_fut_cmd"] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, "Only batch size 1 supported"

        score_threshold = 0.6
        with torch.no_grad():
            c_bbox_results = copy.deepcopy(bbox_results)
            bbox_result = c_bbox_results[0]
            gt_bbox = gt_bboxes_3d[0][0][0]
            gt_label = gt_labels_3d[0][0][0].to("cpu")
            gt_attr_label = gt_attr_labels[0][0][0].to("cpu")

            fut_valid_flag = bool(fut_valid_flag[0][0])
            mask = bbox_result["scores_3d"] > score_threshold
            bbox_result["boxes_3d"] = bbox_result["boxes_3d"][mask]
            bbox_result["scores_3d"] = bbox_result["scores_3d"][mask]
            bbox_result["labels_3d"] = bbox_result["labels_3d"][mask]
            bbox_result["trajs_3d"] = bbox_result["trajs_3d"][mask]

            matched_bbox_result = self.assign_pred_to_gt_vip3d(bbox_result, gt_bbox, gt_label)

            metric_dict = self.compute_motion_metric_vip3d(
                gt_bbox, gt_label, gt_attr_label, bbox_result, matched_bbox_result, mapped_class_names
            )

            # Planning metrics
            ego_fut_preds = bbox_result["ego_fut_preds"]
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]
            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)

            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs=ego_fut_pred[None],
                gt_ego_fut_trajs=ego_fut_trajs[None],
                gt_agent_boxes=gt_bbox,
                gt_agent_feats=gt_attr_label.unsqueeze(0),
                fut_valid_flag=fut_valid_flag,
            )
            metric_dict.update(metric_dict_planner_stp3)

        bbox_list = [dict() for _ in range(len(img_metas))]
        for result_dict, pts_bbox in zip(bbox_list, bbox_results):
            result_dict["pts_bbox"] = pts_bbox
            result_dict["metric_results"] = metric_dict

        return bbox_list

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        result_dict = dict(
            map_boxes_3d=bboxes.to("cpu"),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to("cpu"),
        )

        if attrs is not None:
            result_dict["map_attrs_3d"] = attrs.cpu()

        return result_dict

    def assign_pred_to_gt_vip3d(self, bbox_result, gt_bbox, gt_label, match_dis_thresh=2.0):
        dynamic_list = [0, 1, 3, 4, 6, 7, 8]
        matched_bbox_result = torch.ones((len(gt_bbox)), dtype=torch.long) * -1  # -1: not assigned
        gt_centers = gt_bbox.center[:, :2]
        pred_centers = bbox_result["boxes_3d"].center[:, :2]
        dist = torch.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], dim=-1)
        pred_not_dyn = [label not in dynamic_list for label in bbox_result["labels_3d"]]
        gt_not_dyn = [label not in dynamic_list for label in gt_label]
        dist[pred_not_dyn] = 1e6
        dist[:, gt_not_dyn] = 1e6
        dist[dist > match_dis_thresh] = 1e6

        r_list, c_list = linear_sum_assignment(dist)

        for i in range(len(r_list)):
            if dist[r_list[i], c_list[i]] <= match_dis_thresh:
                matched_bbox_result[c_list[i]] = r_list[i]

        return matched_bbox_result

    def compute_motion_metric_vip3d(
        self,
        gt_bbox,
        gt_label,
        gt_attr_label,
        pred_bbox,
        matched_bbox_result,
        mapped_class_names,
        match_dis_thresh=2.0,
    ):
        motion_cls_names = ["car", "pedestrian"]
        motion_metric_names = ["gt", "cnt_ade", "cnt_fde", "hit", "fp", "ADE", "FDE", "MR"]

        metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                metric_dict[met + "_" + cls] = 0.0

        veh_list = [0, 1, 3, 4]
        ignore_list = ["construction_vehicle", "barrier", "traffic_cone", "motorcycle", "bicycle"]

        for i in range(pred_bbox["labels_3d"].shape[0]):
            pred_bbox["labels_3d"][i] = 0 if pred_bbox["labels_3d"][i] in veh_list else pred_bbox["labels_3d"][i]
            box_name = mapped_class_names[pred_bbox["labels_3d"][i]]
            if box_name in ignore_list:
                continue
            if i not in matched_bbox_result:
                metric_dict["fp_" + box_name] += 1

        for i in range(gt_label.shape[0]):
            gt_label[i] = 0 if gt_label[i] in veh_list else gt_label[i]
            box_name = mapped_class_names[gt_label[i]]
            if box_name in ignore_list:
                continue
            gt_fut_masks = gt_attr_label[i][self.fut_ts * 2 : self.fut_ts * 3]
            num_valid_ts = sum(gt_fut_masks == 1)
            if num_valid_ts == self.fut_ts:
                metric_dict["gt_" + box_name] += 1
            if matched_bbox_result[i] >= 0 and num_valid_ts > 0:
                metric_dict["cnt_ade_" + box_name] += 1
                m_pred_idx = matched_bbox_result[i]
                gt_fut_trajs = gt_attr_label[i][: self.fut_ts * 2].reshape(-1, 2)
                gt_fut_trajs = gt_fut_trajs[:num_valid_ts]
                pred_fut_trajs = pred_bbox["trajs_3d"][m_pred_idx].reshape(self.fut_mode, self.fut_ts, 2)
                pred_fut_trajs = pred_fut_trajs[:, :num_valid_ts, :]
                gt_fut_trajs = gt_fut_trajs.cumsum(dim=-2)
                pred_fut_trajs = pred_fut_trajs.cumsum(dim=-2)
                gt_fut_trajs = gt_fut_trajs + gt_bbox[i].center[0, :2]
                pred_fut_trajs = pred_fut_trajs + pred_bbox["boxes_3d"][int(m_pred_idx)].center[0, :2]

                dist = torch.linalg.norm(gt_fut_trajs[None, :, :] - pred_fut_trajs, dim=-1)
                ade = dist.sum(-1) / num_valid_ts
                ade = ade.min()

                metric_dict["ADE_" + box_name] += ade
                if num_valid_ts == self.fut_ts:
                    fde = dist[:, -1].min()
                    metric_dict["cnt_fde_" + box_name] += 1
                    metric_dict["FDE_" + box_name] += fde
                    if fde <= match_dis_thresh:
                        metric_dict["hit_" + box_name] += 1
                    else:
                        metric_dict["MR_" + box_name] += 1

        return metric_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self, pred_ego_fut_trajs, gt_ego_fut_trajs, gt_agent_boxes, gt_agent_feats, fut_valid_flag
    ):
        metric_dict = {
            "plan_L2_1s": 0,
            "plan_L2_2s": 0,
            "plan_L2_3s": 0,
            "plan_obj_col_1s": 0,
            "plan_obj_col_2s": 0,
            "plan_obj_col_3s": 0,
            "plan_obj_box_col_1s": 0,
            "plan_obj_box_col_2s": 0,
            "plan_obj_box_col_3s": 0,
        }
        metric_dict["fut_valid_flag"] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, "only support bs=1"
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian = self.planning_metric.get_label(gt_agent_boxes, gt_agent_feats)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i + 1) * 2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time],
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(), gt_ego_fut_trajs[:, :cur_time], occupancy
                )
                metric_dict["plan_L2_{}s".format(i + 1)] = traj_L2
                metric_dict["plan_obj_col_{}s".format(i + 1)] = obj_coll.mean().item()
                metric_dict["plan_obj_box_col_{}s".format(i + 1)] = obj_box_coll.mean().item()
            else:
                metric_dict["plan_L2_{}s".format(i + 1)] = 0.0
                metric_dict["plan_obj_col_{}s".format(i + 1)] = 0.0
                metric_dict["plan_obj_box_col_{}s".format(i + 1)] = 0.0

        return metric_dict

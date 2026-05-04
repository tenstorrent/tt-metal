import copy
import numpy as np
import torch
import mmcv
import math
from mmdet3d.registry import DATASETS
from mmdet3d.datasets.transforms.formating import to_tensor
from .my_dataset import CustomNuScenesDataset
from mmdet3d.structures import LiDARInstance3DBoxes
from os import path as osp
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .uniad_custom_nuscenes_eval import NuScenesEval_custom
from nuscenes.eval.tracking.evaluate import TrackingEval
import mmengine
from nuscenes.eval.common.config import config_factory
import tempfile
from models.experimental.uniad.demo.data_container import DataContainer as DC
import cv2
import random
import pickle

from nuscenes import NuScenes

from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer

from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C - 1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0

    return mask


def mask_for_lines(lines, mask, thickness, idx, type="index", angle_class=36):
    coords = np.asarray(list(lines.coords), np.int32)
    coords = coords.reshape((-1, 2))
    if len(coords) < 2:
        return mask, idx
    if type == "backward":
        coords = np.flip(coords, 0)

    if type == "index":
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(
                mask,
                [coords[i:]],
                False,
                color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class),
                thickness=thickness,
            )
    return mask, idx


def get_patch_coord(patch_box, patch_angle=0.0):
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def line_geom_to_mask(
    layer_geom, confidence_levels, local_box, canvas_size, thickness, idx, type="index", angle_class=36
):
    patch_x, patch_y, patch_h, patch_w = local_box

    patch = get_patch_coord(local_box)

    canvas_h = canvas_size[0]
    canvas_w = canvas_size[1]
    scale_height = canvas_h / patch_h
    scale_width = canvas_w / patch_w

    trans_x = -patch_x + patch_w / 2.0
    trans_y = -patch_y + patch_h / 2.0

    map_mask = np.zeros(canvas_size, np.uint8)

    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, trans_x, trans_y])
            new_line = affinity.scale(new_line, xfact=scale_width, yfact=scale_height, origin=(0, 0))
            confidence_levels.append(confidence)
            if new_line.geom_type == "MultiLineString":
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def preprocess_map(vectors, patch_size, canvas_size, num_classes, thickness, angle_class):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(num_classes):
        vector_num_list[i] = []

    for vector in vectors:
        if vector["pts_num"] >= 2:
            vector_num_list[vector["type"]].append(LineString(vector["pts"][: vector["pts_num"]]))

    local_box = (0.0, 0.0, patch_size[0], patch_size[1])

    idx = 1
    filter_masks = []
    instance_masks = []
    forward_masks = []
    backward_masks = []
    for i in range(num_classes):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, local_box, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(
            vector_num_list[i], confidence_levels, local_box, canvas_size, thickness + 4, 1
        )
        filter_masks.append(filter_mask)
        forward_mask, _ = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness,
            1,
            type="forward",
            angle_class=angle_class,
        )
        forward_masks.append(forward_mask)
        backward_mask, _ = line_geom_to_mask(
            vector_num_list[i],
            confidence_levels,
            local_box,
            canvas_size,
            thickness,
            1,
            type="backward",
            angle_class=angle_class,
        )
        backward_masks.append(backward_mask)

    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    forward_masks = np.stack(forward_masks)
    backward_masks = np.stack(backward_masks)

    instance_masks = overlap_filter(instance_masks, filter_masks)
    forward_masks = overlap_filter(forward_masks, filter_masks).sum(0).astype("int32")
    backward_masks = overlap_filter(backward_masks, filter_masks).sum(0).astype("int32")

    semantic_masks = instance_masks != 0

    return semantic_masks, instance_masks, forward_masks, backward_masks


def output_to_nusc_box(detection):
    """Convert the output to the box class in the nuScenes.
    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "track_ids" in detection:
        ids = detection["track_ids"].numpy()
    else:
        ids = np.ones_like(labels)

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # TODO(box3d): convert bbox_yaw and bbox_dims to mmdet3d v1.0.0rc6 format. [DONE]
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = NuScenesBox(box_gravity_center[i], box_dims[i], quat, label=labels[i], score=scores[i], velocity=velocity)
        box.token = ids[i]
        box_list.append(box)
    return box_list


def output_to_nusc_box_det(detection):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    if "boxes_3d_det" in detection:
        box3d = detection["boxes_3d_det"]
        scores = detection["scores_3d_det"].numpy()
        labels = detection["labels_3d_det"].numpy()
    else:
        box3d = detection["boxes_3d"]
        scores = detection["scores_3d"].numpy()
        labels = detection["labels_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    # TODO(box3d): convert bbox_yaw and bbox_dims to mmdet3d v1.0.0rc6 format. [DONE]
    # box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        box = NuScenesBox(box_gravity_center[i], box_dims[i], quat, label=labels[i], score=scores[i], velocity=velocity)
        box_list.append(box)
    return box_list


def obtain_map_info(
    nusc,
    nusc_maps,
    sample,
    patch_size=(102.4, 102.4),
    canvas_size=(256, 256),
    layer_names=["lane_divider", "road_divider"],
    thickness=10,
):
    """
    Export 2d annotation from the info file and raw data.
    """
    l2e_r = sample["lidar2ego_rotation"]
    l2e_t = sample["lidar2ego_translation"]
    e2g_r = sample["ego2global_rotation"]
    e2g_t = sample["ego2global_translation"]
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix

    scene = nusc.get("scene", sample["scene_token"])
    log = nusc.get("log", scene["log_token"])
    nusc_map = nusc_maps[log["location"]]
    if layer_names is None:
        layer_names = nusc_map.non_geometric_layers

    l2g_r_mat = (l2e_r_mat.T @ e2g_r_mat.T).T
    l2g_t = l2e_t @ e2g_r_mat.T + e2g_t
    patch_box = (l2g_t[0], l2g_t[1], patch_size[0], patch_size[1])
    patch_angle = math.degrees(Quaternion(matrix=l2g_r_mat).yaw_pitch_roll[0])

    map_mask = nusc_map.get_map_mask(patch_box, patch_angle, layer_names, canvas_size=canvas_size)
    map_mask = map_mask[-2] | map_mask[-1]
    map_mask = map_mask[np.newaxis, :]
    map_mask = map_mask.transpose((2, 1, 0)).squeeze(2)  # (H, W, C)

    erode = nusc_map.get_map_mask(patch_box, patch_angle, ["drivable_area"], canvas_size=canvas_size)
    erode = erode.transpose((2, 1, 0)).squeeze(2)

    map_mask = np.concatenate([erode[None], map_mask[None]], axis=0)
    return map_mask


def lidar_nusc_box_to_global(info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"):
    box_list = []
    keep_idx = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
        keep_idx.append(i)
    return box_list, keep_idx


import numpy as np
from nuscenes.prediction import PredictHelper, convert_local_coords_to_global, convert_global_coords_to_local
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from mmdet3d.datasets.transforms.formating import to_tensor


class NuScenesTraj(object):
    def __init__(
        self,
        nusc,
        predict_steps,
        planning_steps,
        past_steps,
        fut_steps,
        with_velocity,
        CLASSES,
        box_mode_3d,
        use_nonlinear_optimizer=False,
    ):
        super().__init__()
        self.nusc = nusc
        self.prepare_sdc_vel_info()
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.with_velocity = with_velocity
        self.CLASSES = CLASSES
        self.box_mode_3d = box_mode_3d
        self.predict_helper = PredictHelper(self.nusc)
        self.use_nonlinear_optimizer = use_nonlinear_optimizer

    def get_traj_label(self, sample_token, ann_tokens):
        sd_rec = self.nusc.get("sample", sample_token)
        fut_traj_all = []
        fut_traj_valid_mask_all = []
        past_traj_all = []
        past_traj_valid_mask_all = []
        _, boxes, _ = self.nusc.get_sample_data(sd_rec["data"]["LIDAR_TOP"], selected_anntokens=ann_tokens)
        for i, ann_token in enumerate(ann_tokens):
            box = boxes[i]
            instance_token = self.nusc.get("sample_annotation", ann_token)["instance_token"]
            fut_traj_local = self.predict_helper.get_future_for_agent(
                instance_token, sample_token, seconds=6, in_agent_frame=True
            )
            past_traj_local = self.predict_helper.get_past_for_agent(
                instance_token, sample_token, seconds=2, in_agent_frame=True
            )

            fut_traj = np.zeros((self.predict_steps, 2))
            fut_traj_valid_mask = np.zeros((self.predict_steps, 2))
            past_traj = np.zeros((self.past_steps + self.fut_steps, 2))
            past_traj_valid_mask = np.zeros((self.past_steps + self.fut_steps, 2))
            if fut_traj_local.shape[0] > 0:
                if self.use_nonlinear_optimizer:
                    trans = box.center
                else:
                    trans = np.array([0, 0, 0])
                rot = Quaternion(matrix=box.rotation_matrix)
                fut_traj_scence_centric = convert_local_coords_to_global(fut_traj_local, trans, rot)
                fut_traj[: fut_traj_scence_centric.shape[0], :] = fut_traj_scence_centric
                fut_traj_valid_mask[: fut_traj_scence_centric.shape[0], :] = 1
            if past_traj_local.shape[0] > 0:
                trans = np.array([0, 0, 0])
                rot = Quaternion(matrix=box.rotation_matrix)
                past_traj_scence_centric = convert_local_coords_to_global(past_traj_local, trans, rot)
                past_traj[: past_traj_scence_centric.shape[0], :] = past_traj_scence_centric
                past_traj_valid_mask[: past_traj_scence_centric.shape[0], :] = 1

                if fut_traj_local.shape[0] > 0:
                    fut_steps = min(self.fut_steps, fut_traj_scence_centric.shape[0])
                    past_traj[self.past_steps : self.past_steps + fut_steps, :] = fut_traj_scence_centric[:fut_steps]
                    past_traj_valid_mask[self.past_steps : self.past_steps + fut_steps, :] = 1

            fut_traj_all.append(fut_traj)
            fut_traj_valid_mask_all.append(fut_traj_valid_mask)
            past_traj_all.append(past_traj)
            past_traj_valid_mask_all.append(past_traj_valid_mask)
        if len(ann_tokens) > 0:
            fut_traj_all = np.stack(fut_traj_all, axis=0)
            fut_traj_valid_mask_all = np.stack(fut_traj_valid_mask_all, axis=0)
            past_traj_all = np.stack(past_traj_all, axis=0)
            past_traj_valid_mask_all = np.stack(past_traj_valid_mask_all, axis=0)
        else:
            fut_traj_all = np.zeros((0, self.predict_steps, 2))
            fut_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))
            past_traj_all = np.zeros((0, self.predict_steps, 2))
            past_traj_valid_mask_all = np.zeros((0, self.predict_steps, 2))
        return fut_traj_all, fut_traj_valid_mask_all, past_traj_all, past_traj_valid_mask_all

    def get_vel_transform_mats(self, sample):
        sd_rec = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = self.nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        l2e_r = cs_record["rotation"]
        l2e_t = cs_record["translation"]
        e2g_r = pose_record["rotation"]
        e2g_t = pose_record["translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, e2g_r_mat

    def get_vel_and_time(self, sample):
        lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_top = self.nusc.get("sample_data", lidar_token)
        pose = self.nusc.get("ego_pose", lidar_top["ego_pose_token"])
        xyz = pose["translation"]
        timestamp = sample["timestamp"]
        return xyz, timestamp

    def prepare_sdc_vel_info(self):
        # generate sdc velocity info for all samples
        # Note that these velocity values are converted from
        # global frame to lidar frame
        # as aligned with bbox gts

        self.sdc_vel_info = {}
        for scene in self.nusc.scene:
            scene_token = scene["token"]

            # we cannot infer vel for the last sample, therefore we skip it
            last_sample_token = scene["last_sample_token"]
            sample_token = scene["first_sample_token"]
            sample = self.nusc.get("sample", sample_token)
            xyz, time = self.get_vel_and_time(sample)
            while sample["token"] != last_sample_token:
                next_sample_token = sample["next"]
                next_sample = self.nusc.get("sample", next_sample_token)
                next_xyz, next_time = self.get_vel_and_time(next_sample)
                dc = np.array(next_xyz) - np.array(xyz)
                dt = (next_time - time) / 1e6
                vel = dc / dt

                # global frame to lidar frame
                l2e_r_mat, e2g_r_mat = self.get_vel_transform_mats(sample)
                vel = vel @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                vel = vel[:2]

                self.sdc_vel_info[sample["token"]] = vel
                xyz, time = next_xyz, next_time
                sample = next_sample

            # set first sample's vel equal to second sample's
            last_sample = self.nusc.get("sample", last_sample_token)
            second_last_sample_token = last_sample["prev"]
            self.sdc_vel_info[last_sample_token] = self.sdc_vel_info[second_last_sample_token]

    def generate_sdc_info(self, sdc_vel, as_lidar_instance3d_box=False):
        # sdc dim from https://forum.nuscenes.org/t/dimensions-of-the-ego-vehicle-used-to-gather-data/550
        # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, wlh->lwh -pi->0.5pi
        psudo_sdc_bbox = np.array([0.0, 0.0, 0.0, 4.08, 1.73, 1.56, 0.5 * np.pi])
        if self.with_velocity:
            psudo_sdc_bbox = np.concatenate([psudo_sdc_bbox, sdc_vel], axis=-1)
        gt_bboxes_3d = np.array([psudo_sdc_bbox]).astype(np.float32)
        gt_names_3d = ["car"]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        if as_lidar_instance3d_box:
            # if we do not want the batch the box in to DataContrainer
            return gt_bboxes_3d

        gt_labels_3d = DC(to_tensor(gt_labels_3d))
        gt_bboxes_3d = DC(gt_bboxes_3d, cpu_only=True)

        return gt_bboxes_3d, gt_labels_3d

    def get_sdc_traj_label(self, sample_token):
        sd_rec = self.nusc.get("sample", sample_token)
        lidar_top_data_start = self.nusc.get("sample_data", sd_rec["data"]["LIDAR_TOP"])
        ego_pose_start = self.nusc.get("ego_pose", lidar_top_data_start["ego_pose_token"])

        sdc_fut_traj = []
        for _ in range(self.predict_steps):
            next_annotation_token = sd_rec["next"]
            if next_annotation_token == "":
                break
            sd_rec = self.nusc.get("sample", next_annotation_token)
            lidar_top_data_next = self.nusc.get("sample_data", sd_rec["data"]["LIDAR_TOP"])
            ego_pose_next = self.nusc.get("ego_pose", lidar_top_data_next["ego_pose_token"])
            sdc_fut_traj.append(ego_pose_next["translation"][:2])  # global xy pos of sdc at future step i

        sdc_fut_traj_all = np.zeros((1, self.predict_steps, 2))
        sdc_fut_traj_valid_mask_all = np.zeros((1, self.predict_steps, 2))
        n_valid_timestep = len(sdc_fut_traj)
        if n_valid_timestep > 0:
            sdc_fut_traj = np.stack(sdc_fut_traj, axis=0)  # (t,2)
            sdc_fut_traj = convert_global_coords_to_local(
                coordinates=sdc_fut_traj,
                translation=ego_pose_start["translation"],
                rotation=ego_pose_start["rotation"],
            )
            sdc_fut_traj_all[:, :n_valid_timestep, :] = sdc_fut_traj
            sdc_fut_traj_valid_mask_all[:, :n_valid_timestep, :] = 1

        return sdc_fut_traj_all, sdc_fut_traj_valid_mask_all

    def get_l2g_transform(self, sample):
        sd_rec = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cs_record = self.nusc.get("calibrated_sensor", sd_rec["calibrated_sensor_token"])
        pose_record = self.nusc.get("ego_pose", sd_rec["ego_pose_token"])

        l2e_r = cs_record["rotation"]
        l2e_t = np.array(cs_record["translation"])
        e2g_r = pose_record["rotation"]
        e2g_t = np.array(pose_record["translation"])
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        return l2e_r_mat, l2e_t, e2g_r_mat, e2g_t

    def get_sdc_planning_label(self, sample_token):
        sd_rec = self.nusc.get("sample", sample_token)
        l2e_r_mat_init, l2e_t_init, e2g_r_mat_init, e2g_t_init = self.get_l2g_transform(sd_rec)

        planning = []
        for _ in range(self.planning_steps):
            next_annotation_token = sd_rec["next"]
            if next_annotation_token == "":
                break
            sd_rec = self.nusc.get("sample", next_annotation_token)
            l2e_r_mat_curr, l2e_t_curr, e2g_r_mat_curr, e2g_t_curr = self.get_l2g_transform(
                sd_rec
            )  # (lidar to global at current frame)

            # bbox of sdc under current lidar frame
            next_bbox3d = self.generate_sdc_info(self.sdc_vel_info[next_annotation_token], as_lidar_instance3d_box=True)

            # to bbox under curr ego frame
            next_bbox3d.rotate(l2e_r_mat_curr.T)
            next_bbox3d.translate(l2e_t_curr)

            # to bbox under world frame
            next_bbox3d.rotate(e2g_r_mat_curr.T)
            next_bbox3d.translate(e2g_t_curr)

            # to bbox under initial ego frame, first inverse translate, then inverse rotate
            next_bbox3d.translate(-e2g_t_init)
            m1 = np.linalg.inv(e2g_r_mat_init)
            next_bbox3d.rotate(m1.T)

            # to bbox under curr ego frame, first inverse translate, then inverse rotate
            next_bbox3d.translate(-l2e_t_init)
            m2 = np.linalg.inv(l2e_r_mat_init)
            next_bbox3d.rotate(m2.T)

            planning.append(next_bbox3d)

        planning_all = np.zeros((1, self.planning_steps, 3))
        planning_mask_all = np.zeros((1, self.planning_steps, 2))
        n_valid_timestep = len(planning)
        if n_valid_timestep > 0:
            planning = [p.tensor.squeeze(0) for p in planning]
            planning = np.stack(planning, axis=0)  # (valid_t, 9)
            planning = planning[:, [0, 1, 6]]  # (x, y, yaw)
            planning_all[:, :n_valid_timestep, :] = planning
            planning_mask_all[:, :n_valid_timestep, :] = 1

        mask = planning_mask_all[0].any(axis=1)
        if mask.sum() == 0:
            command = 2  #'FORWARD'
        elif planning_all[0, mask][-1][0] >= 2:
            command = 0  #'RIGHT'
        elif planning_all[0, mask][-1][0] <= -2:
            command = 1  #'LEFT'
        else:
            command = 2  #'FORWARD'

        return planning_all, planning_mask_all, command


CLASS2LABEL = {"road_divider": 0, "lane_divider": 0, "ped_crossing": 1, "contours": 2, "others": -1}


class VectorizedLocalMap(object):
    def __init__(
        self,
        dataroot,
        patch_size,
        canvas_size,
        line_classes=["road_divider", "lane_divider"],
        ped_crossing_classes=["ped_crossing"],
        contour_classes=["road_segment", "lane"],
        sample_dist=1,
        num_samples=250,
        padding=False,
        normalize=False,
        fixed_num=-1,
    ):
        """
        Args:
            fixed_num = -1 : no fixed num
        """
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ["boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"]
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.normalize = normalize
        self.fixed_num = fixed_num

    def gen_vectorized_samples(self, location, ego2global_translation, ego2global_rotation):
        map_pose = ego2global_translation[:2]
        rotation = Quaternion(ego2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180

        line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
        line_vector_dict = self.line_geoms_to_vectors(line_geom)

        ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
        # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
        ped_vector_list = self.line_geoms_to_vectors(ped_geom)["ped_crossing"]

        polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
        poly_bound_list = self.poly_geoms_to_vectors(polygon_geom)

        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, CLASS2LABEL.get(line_type, -1)))

        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, CLASS2LABEL.get("ped_crossing", -1)))

        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length, CLASS2LABEL.get("contours", -1)))

        # filter out -1
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({"pts": pts, "pts_num": pts_num, "type": type})

        return filtered_vectors

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
        return map_geom

    def _one_type_line_geom_to_vectors(self, line_geom):
        line_vectors = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        line_vectors.append(self.sample_pts_from_line(single_line))
                elif line.geom_type == "LineString":
                    line_vectors.append(self.sample_pts_from_line(line))
                else:
                    raise NotImplementedError
        return line_vectors

    def poly_geoms_to_vectors(self, polygon_geom):
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        exteriors = []
        interiors = []
        if union_segments.geom_type != "MultiPolygon":
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)

        results = []
        for ext in exteriors:
            if ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        for inter in interiors:
            if not inter.is_ccw:
                inter.coords = list(inter.coords)[::-1]
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def ped_geoms_to_vectors(self, ped_geom):
        ped_geom = ped_geom[0][1]
        union_ped = ops.unary_union(ped_geom)
        if union_ped.geom_type != "MultiPolygon":
            union_ped = MultiPolygon([union_ped])

        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        results = []
        for ped_poly in union_ped:
            # rect = ped_poly.minimum_rotated_rectangle
            ext = ped_poly.exterior
            if not ext.is_ccw:
                ext.coords = list(ext.coords)[::-1]
            lines = ext.intersection(local_patch)
            results.append(lines)

        return self._one_type_line_geom_to_vectors(results)

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
            points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx : idx + 2], poly_xy[1, idx : idx + 2])]
            line = LineString(points)
            line = line.intersection(patch)
            if not line.is_empty:
                line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(line)

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
        line_list = []
        records = getattr(self.nusc_maps[location], "ped_crossing")
        for record in records:
            polygon = self.map_explorer[location].extract_polygon(record["polygon_token"])
            poly_xy = np.array(polygon.exterior.xy)
            dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
            x1, x2 = np.argsort(dist)[-2:]

            add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
            add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)

        return line_list

    def sample_pts_from_line(self, line):
        if self.fixed_num < 0:
            distances = np.arange(0, line.length, self.sample_dist)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2
            )
        else:
            # fixed number of points, so distance is line.length / self.fixed_num
            distances = np.linspace(0, line.length, self.fixed_num)
            sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2
            )

        if self.normalize:
            sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

        num_valid = len(sampled_points)

        if not self.padding or self.fixed_num > 0:
            # fixed num sample can return now!
            return sampled_points, num_valid

        # fixed distance sampling need padding!
        num_valid = len(sampled_points)

        if self.fixed_num < 0:
            if num_valid < self.num_samples:
                padding = np.zeros((self.num_samples - len(sampled_points), 2))
                sampled_points = np.concatenate([sampled_points, padding], axis=0)
            else:
                sampled_points = sampled_points[: self.num_samples, :]
                num_valid = self.num_samples

            if self.normalize:
                sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
                num_valid = len(sampled_points)

        return sampled_points, num_valid


@DATASETS.register_module()
class CustomNuScenesE2EDataset(CustomNuScenesDataset):
    r"""NuScenes E2E Dataset.

    This dataset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(
        self,
        queue_length=4,
        bev_size=(200, 200),
        patch_size=(102.4, 102.4),
        canvas_size=(200, 200),
        overlap_test=False,
        predict_steps=12,
        planning_steps=6,
        past_steps=4,
        fut_steps=4,
        use_nonlinear_optimizer=False,
        lane_ann_file=None,
        eval_mod=None,
        # For debug
        is_debug=False,
        len_debug=30,
        # Occ dataset
        enbale_temporal_aug=False,
        occ_receptive_field=3,
        occ_n_future=4,
        occ_filter_invalid_sample=False,
        occ_filter_by_valid_flag=False,
        file_client_args=dict(backend="disk"),
        *args,
        **kwargs,
    ):
        # init before super init since it is called in parent class
        self.file_client_args = file_client_args
        self.file_client = mmengine.FileClient(**file_client_args)

        self.is_debug = is_debug
        self.len_debug = len_debug
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.predict_steps = predict_steps
        self.planning_steps = planning_steps
        self.past_steps = past_steps
        self.fut_steps = fut_steps
        self.scene_token = None
        self.lane_infos = self.load_annotations(lane_ann_file) if lane_ann_file else None
        self.eval_mod = eval_mod

        self.use_nonlinear_optimizer = use_nonlinear_optimizer

        self.nusc = NuScenes(version="v1.0-mini", dataroot=self.data_root, verbose=True)

        self.map_num_classes = 3
        if canvas_size[0] == 50:
            self.thickness = 1
        elif canvas_size[0] == 200:
            self.thickness = 2
        else:
            assert False
        self.angle_class = 36
        self.patch_size = patch_size
        self.canvas_size = canvas_size
        self.nusc_maps = {
            "boston-seaport": NuScenesMap(dataroot=self.data_root, map_name="boston-seaport"),
            "singapore-hollandvillage": NuScenesMap(dataroot=self.data_root, map_name="singapore-hollandvillage"),
            "singapore-onenorth": NuScenesMap(dataroot=self.data_root, map_name="singapore-onenorth"),
            "singapore-queenstown": NuScenesMap(dataroot=self.data_root, map_name="singapore-queenstown"),
        }
        self.vector_map = VectorizedLocalMap(self.data_root, patch_size=self.patch_size, canvas_size=self.canvas_size)
        self.traj_api = NuScenesTraj(
            self.nusc,  # changed by me vi
            self.predict_steps,
            self.planning_steps,
            self.past_steps,
            self.fut_steps,
            self.with_velocity,
            self.CLASSES,
            self.box_mode_3d,
            self.use_nonlinear_optimizer,
        )

        # Occ
        self.enbale_temporal_aug = enbale_temporal_aug
        assert self.enbale_temporal_aug is False

        self.occ_receptive_field = occ_receptive_field  # past + current
        self.occ_n_future = occ_n_future  # future only
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        self.occ_filter_by_valid_flag = occ_filter_by_valid_flag
        self.occ_only_total_frames = 7  # NOTE: hardcode, not influenced by planning

    def __len__(self):
        if not self.is_debug:
            return len(self.data_infos)
        else:
            return self.len_debug

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        if self.file_client_args["backend"] == "disk":
            # data_infos = mmcv.load(ann_file)
            data = pickle.loads(self.file_client.get(ann_file.name))
            data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
            data_infos = data_infos[:: self.load_interval]
            self.metadata = data["metadata"]
            self.version = self.metadata["version"]
        elif self.file_client_args["backend"] == "petrel":
            data = pickle.loads(self.file_client.get(ann_file))
            data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
            data_infos = data_infos[:: self.load_interval]
            self.metadata = data["metadata"]
            self.version = self.metadata["version"]
        else:
            assert False, "Invalid file_client_args!"
        return data_infos

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_globals_3d: gt_globals of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame (list)
        """
        data_queue = []
        self.enbale_temporal_aug = False
        if self.enbale_temporal_aug:
            # temporal aug
            prev_indexs_list = list(range(index - self.queue_length, index))
            random.shuffle(prev_indexs_list)
            prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
            input_dict = self.get_data_info(index)
        else:
            # ensure the first and final frame in same scene
            final_index = index
            first_index = index - self.queue_length + 1
            if first_index < 0:
                return None
            if self.data_infos[first_index]["scene_token"] != self.data_infos[final_index]["scene_token"]:
                return None
            # current timestamp
            input_dict = self.get_data_info(final_index)
            prev_indexs_list = list(reversed(range(first_index, final_index)))
        if input_dict is None:
            return None
        frame_idx = input_dict["frame_idx"]
        scene_token = input_dict["scene_token"]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        assert example["gt_labels_3d"].data.shape[0] == example["gt_fut_traj"].shape[0]
        assert example["gt_labels_3d"].data.shape[0] == example["gt_past_traj"].shape[0]

        if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
            return None
        data_queue.insert(0, example)

        # retrieve previous infos

        for i in prev_indexs_list:
            if self.enbale_temporal_aug:
                i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict["frame_idx"] < frame_idx and input_dict["scene_token"] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                if self.filter_empty_gt and (example is None or ~(example["gt_labels_3d"]._data != -1).any()):
                    return None
                frame_idx = input_dict["frame_idx"]
            assert example["gt_labels_3d"].data.shape[0] == example["gt_fut_traj"].shape[0]
            assert example["gt_labels_3d"].data.shape[0] == example["gt_past_traj"].shape[0]
            data_queue.insert(0, copy.deepcopy(example))
        data_queue = self.union2one(data_queue)
        return data_queue

    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
                img: queue_length, 6, 3, H, W
                img_metas: img_metas of each frame (list)
                gt_labels_3d: gt_labels of each frame (list)
                gt_bboxes_3d: gt_bboxes of each frame (list)
                gt_inds: gt_inds of each frame(list)
        """

        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        data_dict = {}
        for key, value in example.items():
            if "l2g" in key:
                data_dict[key] = to_tensor(value[0])
            else:
                data_dict[key] = value
        return data_dict

    def union2one(self, queue):
        """
        convert sample dict into one single sample.
        """
        imgs_list = [each["img"].data for each in queue]
        gt_labels_3d_list = [each["gt_labels_3d"].data for each in queue]
        gt_sdc_label_list = [each["gt_sdc_label"].data for each in queue]
        gt_inds_list = [to_tensor(each["gt_inds"]) for each in queue]
        gt_bboxes_3d_list = [each["gt_bboxes_3d"].data for each in queue]
        gt_past_traj_list = [to_tensor(each["gt_past_traj"]) for each in queue]
        gt_past_traj_mask_list = [to_tensor(each["gt_past_traj_mask"]) for each in queue]
        gt_sdc_bbox_list = [each["gt_sdc_bbox"].data for each in queue]
        l2g_r_mat_list = [to_tensor(each["l2g_r_mat"]) for each in queue]
        l2g_t_list = [to_tensor(each["l2g_t"]) for each in queue]
        timestamp_list = [to_tensor(each["timestamp"]) for each in queue]
        gt_fut_traj = to_tensor(queue[-1]["gt_fut_traj"])
        gt_fut_traj_mask = to_tensor(queue[-1]["gt_fut_traj_mask"])
        gt_sdc_fut_traj = to_tensor(queue[-1]["gt_sdc_fut_traj"])
        gt_sdc_fut_traj_mask = to_tensor(queue[-1]["gt_sdc_fut_traj_mask"])
        gt_future_boxes_list = queue[-1]["gt_future_boxes"]
        gt_future_labels_list = [to_tensor(each) for each in queue[-1]["gt_future_labels"]]

        metas_map = {}
        prev_pos = None
        prev_angle = None
        for i, each in enumerate(queue):
            metas_map[i] = each["img_metas"].data
            if i == 0:
                metas_map[i]["prev_bev"] = False
                prev_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                prev_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] = 0
                metas_map[i]["can_bus"][-1] = 0
            else:
                metas_map[i]["prev_bev"] = True
                tmp_pos = copy.deepcopy(metas_map[i]["can_bus"][:3])
                tmp_angle = copy.deepcopy(metas_map[i]["can_bus"][-1])
                metas_map[i]["can_bus"][:3] -= prev_pos
                metas_map[i]["can_bus"][-1] -= prev_angle
                prev_pos = copy.deepcopy(tmp_pos)
                prev_angle = copy.deepcopy(tmp_angle)

        queue[-1]["img"] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]["img_metas"] = DC(metas_map, cpu_only=True)

        queue = queue[-1]

        queue["gt_labels_3d"] = DC(gt_labels_3d_list)
        queue["gt_sdc_label"] = DC(gt_sdc_label_list)
        queue["gt_inds"] = DC(gt_inds_list)
        queue["gt_bboxes_3d"] = DC(gt_bboxes_3d_list, cpu_only=True)
        queue["gt_sdc_bbox"] = DC(gt_sdc_bbox_list, cpu_only=True)
        queue["l2g_r_mat"] = DC(l2g_r_mat_list)
        queue["l2g_t"] = DC(l2g_t_list)
        queue["timestamp"] = DC(timestamp_list)
        queue["gt_fut_traj"] = DC(gt_fut_traj)
        queue["gt_fut_traj_mask"] = DC(gt_fut_traj_mask)
        queue["gt_past_traj"] = DC(gt_past_traj_list)
        queue["gt_past_traj_mask"] = DC(gt_past_traj_mask_list)
        queue["gt_future_boxes"] = DC(gt_future_boxes_list, cpu_only=True)
        queue["gt_future_labels"] = DC(gt_future_labels_list)
        return queue

    def get_ann_info(self, index):
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
        gt_inds = info["gt_inds"][mask]

        sample = self.nusc.get("sample", info["token"])
        ann_tokens = np.array(sample["anns"])[mask]
        assert ann_tokens.shape[0] == gt_bboxes_3d.shape[0]

        gt_fut_traj, gt_fut_traj_mask, gt_past_traj, gt_past_traj_mask = self.traj_api.get_traj_label(
            info["token"], ann_tokens
        )

        sdc_vel = self.traj_api.sdc_vel_info[info["token"]]
        gt_sdc_bbox, gt_sdc_label = self.traj_api.generate_sdc_info(sdc_vel)
        gt_sdc_fut_traj, gt_sdc_fut_traj_mask = self.traj_api.get_sdc_traj_label(info["token"])

        sdc_planning, sdc_planning_mask, command = self.traj_api.get_sdc_planning_label(info["token"])

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
            gt_fut_traj=gt_fut_traj,
            gt_fut_traj_mask=gt_fut_traj_mask,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            gt_sdc_fut_traj=gt_sdc_fut_traj,
            gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,
            sdc_planning=sdc_planning,
            sdc_planning_mask=sdc_planning_mask,
            command=command,
        )
        assert gt_fut_traj.shape[0] == gt_labels_3d.shape[0]
        assert gt_past_traj.shape[0] == gt_labels_3d.shape[0]
        return anns_results

    def get_data_info(self, index):
        info = self.data_infos[index]

        # semantic format
        lane_info = self.lane_infos[index] if self.lane_infos else None
        # panoptic format
        location = self.nusc.get("log", self.nusc.get("scene", info["scene_token"])["log_token"])["location"]
        vectors = self.vector_map.gen_vectorized_samples(
            location, info["ego2global_translation"], info["ego2global_rotation"]
        )
        semantic_masks, instance_masks, forward_masks, backward_masks = preprocess_map(
            vectors, self.patch_size, self.canvas_size, self.map_num_classes, self.thickness, self.angle_class
        )
        instance_masks = np.rot90(instance_masks, k=-1, axes=(1, 2))
        instance_masks = torch.tensor(instance_masks.copy())
        gt_labels = []
        gt_bboxes = []
        gt_masks = []
        for cls in range(self.map_num_classes):
            for i in np.unique(instance_masks[cls]):
                if i == 0:
                    continue
                gt_mask = (instance_masks[cls] == i).to(torch.uint8)
                ys, xs = np.where(gt_mask)
                gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
                gt_labels.append(cls)
                gt_bboxes.append(gt_bbox)
                gt_masks.append(gt_mask)
        map_mask = obtain_map_info(
            self.nusc,
            self.nusc_maps,
            info,
            patch_size=self.patch_size,
            canvas_size=self.canvas_size,
            layer_names=["lane_divider", "road_divider"],
        )
        map_mask = np.flip(map_mask, axis=1)
        map_mask = np.rot90(map_mask, k=-1, axes=(1, 2))
        map_mask = torch.tensor(map_mask.copy())
        for i, gt_mask in enumerate(map_mask[:-1]):
            ys, xs = np.where(gt_mask)
            gt_bbox = [min(xs), min(ys), max(xs), max(ys)]
            gt_labels.append(i + self.map_num_classes)
            gt_bboxes.append(gt_bbox)
            gt_masks.append(gt_mask)
        gt_labels = torch.tensor(gt_labels)
        gt_bboxes = torch.tensor(np.stack(gt_bboxes))
        gt_masks = torch.stack(gt_masks)

        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            prev_idx=info["prev"],
            next_idx=info["next"],
            scene_token=info["scene_token"],
            can_bus=info["can_bus"],
            frame_idx=info["frame_idx"],
            timestamp=info["timestamp"] / 1e6,
            map_filename=lane_info["maps"]["map_mask"] if lane_info else None,
            gt_lane_labels=gt_labels,
            gt_lane_bboxes=gt_bboxes,
            gt_lane_masks=gt_masks,
        )

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        l2g_r_mat = l2e_r_mat.T @ e2g_r_mat.T
        l2g_t = l2e_t @ e2g_r_mat.T + e2g_t

        input_dict.update(dict(l2g_r_mat=l2g_r_mat.astype(np.float32), l2g_t=l2g_t.astype(np.float32)))

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(cam_info["data_path"])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info["sensor2lidar_rotation"])
                lidar2cam_t = cam_info["sensor2lidar_translation"] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = cam_info["cam_intrinsic"]
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt.T
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        # if not self.test_mode:
        annos = self.get_ann_info(index)
        input_dict["ann_info"] = annos
        if "sdc_planning" in input_dict["ann_info"].keys():
            input_dict["sdc_planning"] = input_dict["ann_info"]["sdc_planning"]
            input_dict["sdc_planning_mask"] = input_dict["ann_info"]["sdc_planning_mask"]
            input_dict["command"] = input_dict["ann_info"]["command"]

        rotation = Quaternion(input_dict["ego2global_rotation"])
        translation = input_dict["ego2global_translation"]
        can_bus = input_dict["can_bus"]
        can_bus[:3] = translation
        # NOTE(lty): fix can_bus format, in https://github.com/OpenDriveLab/UniAD/pull/214
        can_bus[3:7] = rotation.elements
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        # TODO: Warp all those below occupancy-related codes into a function
        prev_indices, future_indices = self.occ_get_temporal_indices(index, self.occ_receptive_field, self.occ_n_future)

        # ego motions of all frames are needed
        all_frames = prev_indices + [index] + future_indices

        # whether invalid frames is present
        #
        has_invalid_frame = -1 in all_frames[: self.occ_only_total_frames]
        # NOTE: This can only represent 7 frames in total as it influence evaluation
        input_dict["occ_has_invalid_frame"] = has_invalid_frame
        input_dict["occ_img_is_valid"] = np.array(all_frames) >= 0

        # might have None if not in the same sequence
        future_frames = [index] + future_indices

        # get lidar to ego to global transforms for each curr and fut index
        occ_transforms = self.occ_get_transforms(future_frames)  # might have None
        input_dict.update(occ_transforms)

        # for (current and) future frames, detection labels are needed
        # generate detection labels for current + future frames
        input_dict["occ_future_ann_infos"] = self.get_future_detection_infos(future_frames)
        return input_dict

    def get_future_detection_infos(self, future_frames):
        detection_ann_infos = []
        for future_frame in future_frames:
            if future_frame >= 0:
                detection_ann_infos.append(
                    self.occ_get_detection_ann_info(future_frame),
                )
            else:
                detection_ann_infos.append(None)
        return detection_ann_infos

    def occ_get_temporal_indices(self, index, receptive_field, n_future):
        current_scene_token = self.data_infos[index]["scene_token"]

        # generate the past
        previous_indices = []

        for t in range(-receptive_field + 1, 0):
            index_t = index + t
            if index_t >= 0 and self.data_infos[index_t]["scene_token"] == current_scene_token:
                previous_indices.append(index_t)
            else:
                previous_indices.append(-1)  # for invalid indices

        # generate the future
        future_indices = []

        for t in range(1, n_future + 1):
            index_t = index + t
            if index_t < len(self.data_infos) and self.data_infos[index_t]["scene_token"] == current_scene_token:
                future_indices.append(index_t)
            else:
                # NOTE: How to deal the invalid indices???
                future_indices.append(-1)

        return previous_indices, future_indices

    def occ_get_transforms(self, indices, data_type=torch.float32):
        """
        get l2e, e2g rotation and translation for each valid frame
        """
        l2e_r_mats = []
        l2e_t_vecs = []
        e2g_r_mats = []
        e2g_t_vecs = []

        for index in indices:
            if index == -1:
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
            else:
                info = self.data_infos[index]
                l2e_r = info["lidar2ego_rotation"]
                l2e_t = info["lidar2ego_translation"]
                e2g_r = info["ego2global_rotation"]
                e2g_t = info["ego2global_translation"]

                l2e_r_mat = torch.from_numpy(Quaternion(l2e_r).rotation_matrix)
                e2g_r_mat = torch.from_numpy(Quaternion(e2g_r).rotation_matrix)

                l2e_r_mats.append(l2e_r_mat.to(data_type))
                l2e_t_vecs.append(torch.tensor(l2e_t).to(data_type))
                e2g_r_mats.append(e2g_r_mat.to(data_type))
                e2g_t_vecs.append(torch.tensor(e2g_t).to(data_type))

        res = {
            "occ_l2e_r_mats": l2e_r_mats,
            "occ_l2e_t_vecs": l2e_t_vecs,
            "occ_e2g_r_mats": e2g_r_mats,
            "occ_e2g_t_vecs": e2g_t_vecs,
        }

        return res

    def occ_get_detection_ann_info(self, index):
        info = self.data_infos[index].copy()
        gt_bboxes_3d = info["gt_boxes"].copy()
        gt_names_3d = info["gt_names"].copy()
        gt_ins_inds = info["gt_inds"].copy()

        gt_vis_tokens = info.get("visibility_tokens", None)

        if self.use_valid_flag:
            gt_valid_flag = info["valid_flag"]
        else:
            gt_valid_flag = info["num_lidar_pts"] > 0

        assert self.occ_filter_by_valid_flag is False
        if self.occ_filter_by_valid_flag:
            gt_bboxes_3d = gt_bboxes_3d[gt_valid_flag]
            gt_names_3d = gt_names_3d[gt_valid_flag]
            gt_ins_inds = gt_ins_inds[gt_valid_flag]
            gt_vis_tokens = gt_vis_tokens[gt_valid_flag]

        # cls_name to cls_id
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            # gt_names=gt_names_3d,
            gt_inds=gt_ins_inds,
            gt_vis_tokens=gt_vis_tokens,
        )

        return anns_results

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        nusc_map_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]["token"]

            if "map" in self.eval_mod:
                map_annos = {}
                for key, value in det["ret_iou"].items():
                    map_annos[key] = float(value.numpy()[0])
                    nusc_map_annos[sample_token] = map_annos

            if "boxes_3d" not in det:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(
                self.data_infos[sample_id], boxes, mapped_class_names, self.eval_detection_configs, self.eval_version
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = CustomNuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = CustomNuScenesDataset.DefaultAttribute[name]

                # center_ = box.center.tolist()
                # change from ground height to center height
                # center_[2] = center_[2] + (box.wlh.tolist()[2] / 2.0)
                if name not in [
                    "car",
                    "truck",
                    "bus",
                    "trailer",
                    "motorcycle",
                    "bicycle",
                    "pedestrian",
                ]:
                    continue

                box_ego = boxes_ego[keep_idx[i]]
                trans = box_ego.center
                if "traj" in det:
                    traj_local = det["traj"][keep_idx[i]].numpy()[..., :2]
                    traj_scores = det["traj_scores"][keep_idx[i]].numpy()
                else:
                    traj_local = np.zeros((0,))
                    traj_scores = np.zeros((0,))
                traj_ego = np.zeros_like(traj_local)
                rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi / 2)
                for kk in range(traj_ego.shape[0]):
                    traj_ego[kk] = convert_local_coords_to_global(traj_local[kk], trans, rot)

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                    tracking_name=name,
                    tracking_score=box.score,
                    tracking_id=box.token,
                    predict_traj=traj_ego,
                    predict_traj_score=traj_scores,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
            "map_results": nusc_map_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        result_files = self._format_bbox(results, jsonfile_prefix)

        return result_files, tmp_dir

    def _format_bbox_det(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            sample_token = self.data_infos[sample_id]["token"]

            if det is None:
                nusc_annos[sample_token] = annos
                continue

            boxes = output_to_nusc_box_det(det)
            boxes_ego = copy.deepcopy(boxes)
            boxes, keep_idx = lidar_nusc_box_to_global(
                self.data_infos[sample_id], boxes, mapped_class_names, self.eval_detection_configs, self.eval_version
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = CustomNuScenesDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = CustomNuScenesDataset.DefaultAttribute[name]

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                    detection_name=name,
                    detection_score=box.score,
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc_det.json")
        print("Results writes to", res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def format_results_det(self, results, jsonfile_prefix=None):
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(self), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results_det")
        else:
            tmp_dir = None

        result_files = self._format_bbox_det(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
        planning_evaluation_strategy="uniad",
    ):
        if isinstance(results, dict):
            if "occ_results_computed" in results.keys():
                occ_results_computed = results["occ_results_computed"]
                out_metrics = ["iou"]

                # pan_eval
                if occ_results_computed.get("pq", None) is not None:
                    out_metrics = ["iou", "pq", "sq", "rq"]

                print("Occ-flow Val Results:")
                for panoptic_key in out_metrics:
                    print(panoptic_key)
                    # HERE!! connect
                    print(" & ".join([f"{x:.1f}" for x in occ_results_computed[panoptic_key]]))

                if "num_occ" in occ_results_computed.keys() and "ratio_occ" in occ_results_computed.keys():
                    print(f"num occ evaluated:{occ_results_computed['num_occ']}")
                    print(f"ratio occ evaluated: {occ_results_computed['ratio_occ'] * 100:.1f}%")
            if "planning_results_computed" in results.keys():
                planning_results_computed = results["planning_results_computed"]
                planning_tab = PrettyTable()
                planning_tab.title = f"{planning_evaluation_strategy}'s definition planning metrics"
                planning_tab.field_names = ["metrics", "0.5s", "1.0s", "1.5s", "2.0s", "2.5s", "3.0s"]
                for key in planning_results_computed.keys():
                    value = planning_results_computed[key]
                    row_value = []
                    row_value.append(key)
                    for i in range(len(value)):
                        if planning_evaluation_strategy == "stp3":
                            row_value.append("%.4f" % float(value[: i + 1].mean()))
                        elif planning_evaluation_strategy == "uniad":
                            row_value.append("%.4f" % float(value[i]))
                        else:
                            raise ValueError("planning_evaluation_strategy should be uniad or spt3")
                    planning_tab.add_row(row_value)
                print(planning_tab)
            results = results["bbox_results"]  # get bbox_results

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)
        result_files_det, tmp_dir = self.format_results_det(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print("Evaluating bboxes of {}".format(name))
                ret_dict = self._evaluate_single(result_files[name], result_files_det[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, result_files_det)

        if "map" in self.eval_mod:
            drivable_intersection = 0
            drivable_union = 0
            lanes_intersection = 0
            lanes_union = 0
            divider_intersection = 0
            divider_union = 0
            crossing_intersection = 0
            crossing_union = 0
            contour_intersection = 0
            contour_union = 0
            for i in range(len(results)):
                drivable_intersection += results[i]["ret_iou"]["drivable_intersection"]
                drivable_union += results[i]["ret_iou"]["drivable_union"]
                lanes_intersection += results[i]["ret_iou"]["lanes_intersection"]
                lanes_union += results[i]["ret_iou"]["lanes_union"]
                divider_intersection += results[i]["ret_iou"]["divider_intersection"]
                divider_union += results[i]["ret_iou"]["divider_union"]
                crossing_intersection += results[i]["ret_iou"]["crossing_intersection"]
                crossing_union += results[i]["ret_iou"]["crossing_union"]
                contour_intersection += results[i]["ret_iou"]["contour_intersection"]
                contour_union += results[i]["ret_iou"]["contour_union"]
            results_dict.update(
                {
                    "drivable_iou": float(drivable_intersection / drivable_union),
                    "lanes_iou": float(lanes_intersection / lanes_union),
                    "divider_iou": float(divider_intersection / divider_union),
                    "crossing_iou": float(crossing_intersection / crossing_union),
                    "contour_iou": float(contour_intersection / contour_union),
                }
            )

            print(results_dict)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict

    def _evaluate_single(self, result_path, result_path_det, logger=None, metric="bbox", result_name="pts_bbox"):
        # TODO: fix the evaluation pipelines

        output_dir = osp.join(*osp.split(result_path)[:-1])
        output_dir_det = osp.join(output_dir, "det")
        output_dir_track = osp.join(output_dir, "track")
        output_dir_motion = osp.join(output_dir, "motion")
        mmcv.mkdir_or_exist(output_dir_det)
        mmcv.mkdir_or_exist(output_dir_track)
        mmcv.mkdir_or_exist(output_dir_motion)

        eval_set_map = {
            "v1.0-mini": "mini_val",
            # 'v1.0-trainval': 'val',
        }
        detail = dict()

        if "det" in self.eval_mod:
            self.nusc_eval = NuScenesEval_custom(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path_det,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_det,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
            )
            self.nusc_eval.main(plot_examples=0, render_curves=False)
            # record metrics
            metrics = mmcv.load(osp.join(output_dir_det, "metrics_summary.json"))
            metric_prefix = f"{result_name}_NuScenes"
            for name in self.CLASSES:
                for k, v in metrics["label_aps"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_AP_dist_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["label_tp_errors"][name].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}_{}".format(metric_prefix, name, k)] = val
                for k, v in metrics["tp_errors"].items():
                    val = float("{:.4f}".format(v))
                    detail["{}/{}".format(metric_prefix, self.ErrNameMapping[k])] = val
            detail["{}/NDS".format(metric_prefix)] = metrics["nd_score"]
            detail["{}/mAP".format(metric_prefix)] = metrics["mean_ap"]

        if "track" in self.eval_mod:
            cfg = config_factory("tracking_nips_2019")
            self.nusc_eval_track = TrackingEval(
                config=cfg,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_track,
                verbose=True,
                nusc_version=self.version,
                nusc_dataroot=self.data_root,
            )
            self.nusc_eval_track.main()
            # record metrics
            metrics = mmcv.load(osp.join(output_dir_track, "metrics_summary.json"))
            keys = [
                "amota",
                "amotp",
                "recall",
                "motar",
                "gt",
                "mota",
                "motp",
                "mt",
                "ml",
                "faf",
                "tp",
                "fp",
                "fn",
                "ids",
                "frag",
                "tid",
                "lgd",
            ]
            for key in keys:
                detail["{}/{}".format(metric_prefix, key)] = metrics[key]

        # if 'map' in self.eval_mod:
        #     for i, ret_iou in enumerate(ret_ious):
        #         detail['iou_{}'.format(i)] = ret_iou

        if "motion" in self.eval_mod:
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type="motion_category",
            )
            print("-" * 50)
            print("Evaluate on motion category, merge class for vehicles and pedestrians...")
            print("evaluate standard motion metrics...")
            self.nusc_eval_motion.main(plot_examples=0, render_curves=False, eval_mode="standard")
            print("evaluate motion mAP-minFDE metrics...")
            self.nusc_eval_motion.main(plot_examples=0, render_curves=False, eval_mode="motion_map")
            print("evaluate EPA motion metrics...")
            self.nusc_eval_motion.main(plot_examples=0, render_curves=False, eval_mode="epa")
            print("-" * 50)
            print("Evaluate on detection category...")
            self.nusc_eval_motion = MotionEval(
                self.nusc,
                config=self.eval_detection_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir,
                verbose=True,
                overlap_test=self.overlap_test,
                data_infos=self.data_infos,
                category_convert_type="detection_category",
            )
            print("evaluate standard motion metrics...")
            self.nusc_eval_motion.main(plot_examples=0, render_curves=False, eval_mode="standard")
            print("evaluate EPA motion metrics...")
            self.nusc_eval_motion.main(plot_examples=0, render_curves=False, eval_mode="motion_map")
            print("evaluate EPA motion metrics...")
            self.nusc_eval_motion.main(plot_examples=0, render_curves=False, eval_mode="epa")

        return detail

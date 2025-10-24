# Copyright (c) Facebook, Inc. and its affiliates.

"""
Modified from https://github.com/facebookresearch/votenet
Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points

# Import from reference utils instead of source
from models.experimental.detr3d.reference.utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)


MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATA_PATH_V1 = ""  ## Replace with path to dataset
DATA_PATH_V2 = ""  ## Not used in the codebase.


class SunrgbdDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 10
        self.num_angle_bin = 12
        self.max_num_obj = 64
        self.type2class = {
            "bed": 0,
            "table": 1,
            "sofa": 2,
            "chair": 3,
            "toilet": 4,
            "desk": 5,
            "dresser": 6,
            "night_stand": 7,
            "bookshelf": 8,
            "bathtub": 9,
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.type2onehotclass = {
            "bed": 0,
            "table": 1,
            "sofa": 2,
            "chair": 3,
            "toilet": 4,
            "desk": 5,
            "dresser": 6,
            "night_stand": 7,
            "bookshelf": 8,
            "bathtub": 9,
        }

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that
            class*(2pi/N) + number = angle
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (class_id * angle_per_class + angle_per_class / 2)
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def my_compute_box_3d(self, center, size, heading_angle):
        # Simple rotation matrix for Z-axis
        R = np.array(
            [
                [np.cos(-heading_angle), -np.sin(-heading_angle), 0],
                [np.sin(-heading_angle), np.cos(-heading_angle), 0],
                [0, 0, 1],
            ]
        )
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)


class SunrgbdDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        root_dir=None,
        num_points=20000,
        use_color=False,
        use_height=False,
        use_v1=True,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
    ):
        assert num_points <= 50000
        assert split_set in ["train", "val", "trainval"]
        self.dataset_config = dataset_config
        self.use_v1 = use_v1

        if root_dir is None:
            root_dir = DATA_PATH_V1 if use_v1 else DATA_PATH_V2

        self.data_path = root_dir + "_%s" % (split_set)

        if split_set in ["train", "val"]:
            self.scan_names = sorted(list(set([os.path.basename(x)[0:6] for x in os.listdir(self.data_path)])))
        elif split_set in ["trainval"]:
            # combine names from both
            sub_splits = ["train", "val"]
            all_paths = []
            for sub_split in sub_splits:
                data_path = self.data_path.replace("trainval", sub_split)
                basenames = sorted(list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)])))
                basenames = [os.path.join(data_path, x) for x in basenames]
                all_paths.extend(basenames)
            all_paths.sort()
            self.scan_names = all_paths

        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid

        # Simplified random cuboid for minimal dependencies
        self.random_cuboid_augmentor = None
        if use_random_cuboid:
            # Create a minimal random cuboid implementation
            self.random_cuboid_augmentor = MinimalRandomCuboid(
                min_points=random_cuboid_min_points,
                aspect=0.75,
                min_crop=0.75,
                max_crop=1.0,
            )

        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        if scan_name.startswith("/"):
            scan_path = scan_name
        else:
            scan_path = os.path.join(self.data_path, scan_name)
        point_cloud = np.load(scan_path + "_pc.npz")["pc"]  # Nx6
        bboxes = np.load(scan_path + "_bbox.npy")  # K,8

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            assert point_cloud.shape[1] == 6
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = point_cloud[:, 3:] - MEAN_COLOR_RGB

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)  # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = self._rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:, 3:6] + MEAN_COLOR_RGB
                rgb_color *= 1 + 0.4 * np.random.random(3) - 0.2  # brightness change for each channel
                rgb_color += 0.1 * np.random.random(3) - 0.05  # color shift for each channel
                rgb_color += np.expand_dims(
                    (0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1
                )  # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(np.random.random(point_cloud.shape[0]) > 0.3, -1)
                point_cloud[:, 3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio

            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

            if self.use_random_cuboid and self.random_cuboid_augmentor:
                point_cloud, bboxes, _ = self.random_cuboid_augmentor(point_cloud, bboxes)

        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        label_mask = np.zeros((self.max_num_obj))
        label_mask[0 : bboxes.shape[0]] = 1
        max_bboxes = np.zeros((self.max_num_obj, 8))
        max_bboxes[0 : bboxes.shape[0], :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            raw_angles[i] = bbox[6] % 2 * np.pi
            box3d_size = bbox[3:6] * 2
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox

        point_cloud, choices = self._random_sampling(point_cloud, self.num_points, return_choices=True)

        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = self._scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = self._shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(angle_classes, angle_residuals)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(np.float32)
        target_bboxes_semcls = np.zeros((self.max_num_obj))
        target_bboxes_semcls[0 : bboxes.shape[0]] = bboxes[:, -1]  # from 0 to 9
        ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max
        return ret_dict

    def _rotz(self, t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def _random_sampling(self, pc, num_sample, replace=None, return_choices=False):
        """Input is NxC, output is num_samplexC"""
        if replace is None:
            replace = pc.shape[0] < num_sample
        choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
        if return_choices:
            return pc[choices], choices
        else:
            return pc[choices]

    def _shift_scale_points(self, pred_xyz, src_range, dst_range=None):
        """
        pred_xyz: B x N x 3
        src_range: [[B x 3], [B x 3]] - min and max XYZ coords
        dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
        """
        if dst_range is None:
            dst_range = [
                np.zeros((src_range[0].shape[0], 3)),
                np.ones((src_range[0].shape[0], 3)),
            ]

        if pred_xyz.ndim == 4:
            src_range = [x[:, None] for x in src_range]
            dst_range = [x[:, None] for x in dst_range]

        assert src_range[0].shape[0] == pred_xyz.shape[0]
        assert dst_range[0].shape[0] == pred_xyz.shape[0]
        assert src_range[0].shape[-1] == pred_xyz.shape[-1]
        assert src_range[0].shape == src_range[1].shape
        assert dst_range[0].shape == dst_range[1].shape
        assert src_range[0].shape == dst_range[1].shape

        src_diff = src_range[1][:, None, :] - src_range[0][:, None, :]
        dst_diff = dst_range[1][:, None, :] - dst_range[0][:, None, :]
        prop_xyz = (((pred_xyz - src_range[0][:, None, :]) * dst_diff) / src_diff) + dst_range[0][:, None, :]
        return prop_xyz

    def _scale_points(self, pred_xyz, mult_factor):
        if pred_xyz.ndim == 4:
            mult_factor = mult_factor[:, None]
        scaled_xyz = pred_xyz * mult_factor[:, None, :]
        return scaled_xyz


class MinimalRandomCuboid(object):
    """
    Minimal implementation of RandomCuboid augmentation without external dependencies
    """

    def __init__(self, min_points, aspect=0.8, min_crop=0.5, max_crop=1.0):
        self.aspect = aspect
        self.min_crop = min_crop
        self.max_crop = max_crop
        self.min_points = min_points

    def _check_aspect(self, crop_range, aspect_min):
        xy_aspect = np.min(crop_range[:2]) / np.max(crop_range[:2])
        xz_aspect = np.min(crop_range[[0, 2]]) / np.max(crop_range[[0, 2]])
        yz_aspect = np.min(crop_range[1:]) / np.max(crop_range[1:])
        return (xy_aspect >= aspect_min) or (xz_aspect >= aspect_min) or (yz_aspect >= aspect_min)

    def __call__(self, point_cloud, target_boxes, per_point_labels=None):
        range_xyz = np.max(point_cloud[:, 0:3], axis=0) - np.min(point_cloud[:, 0:3], axis=0)

        for _ in range(100):
            crop_range = self.min_crop + np.random.rand(3) * (self.max_crop - self.min_crop)
            if not self._check_aspect(crop_range, self.aspect):
                continue

            sample_center = point_cloud[np.random.choice(len(point_cloud)), 0:3]

            new_range = range_xyz * crop_range / 2.0

            max_xyz = sample_center + new_range
            min_xyz = sample_center - new_range

            upper_idx = np.sum((point_cloud[:, 0:3] <= max_xyz).astype(np.int32), 1) == 3
            lower_idx = np.sum((point_cloud[:, 0:3] >= min_xyz).astype(np.int32), 1) == 3

            new_pointidx = (upper_idx) & (lower_idx)

            if np.sum(new_pointidx) < self.min_points:
                continue

            new_point_cloud = point_cloud[new_pointidx, :]

            # filtering policy
            new_boxes = target_boxes
            if target_boxes.sum() > 0:  # ground truth contains no bounding boxes. Common in SUNRGBD.
                box_centers = target_boxes[:, 0:3]
                new_pc_min_max = np.min(new_point_cloud[:, 0:3], axis=0), np.max(new_point_cloud[:, 0:3], axis=0)
                keep_boxes = np.logical_and(
                    np.all(box_centers >= new_pc_min_max[0], axis=1),
                    np.all(box_centers <= new_pc_min_max[1], axis=1),
                )
                if keep_boxes.sum() == 0:
                    # current data augmentation removes all boxes in the pointcloud. fail!
                    continue
                new_boxes = target_boxes[keep_boxes]
            if per_point_labels is not None:
                new_per_point_labels = [x[new_pointidx] for x in per_point_labels]
            else:
                new_per_point_labels = None
            # if we are here, all conditions are met. return boxes
            return new_point_cloud, new_boxes, new_per_point_labels

        # fallback
        return point_cloud, target_boxes, per_point_labels

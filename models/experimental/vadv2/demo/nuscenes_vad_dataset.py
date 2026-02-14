# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import json
import copy
import tempfile
import random
import torch
import mmengine
import numpy as np
import pyquaternion
from typing import Dict, List
from mmengine.utils import ProgressBar
from models.experimental.vadv2.demo.my_dataset import CustomNuScenesDataset
from models.experimental.vadv2.demo.data_container import DataContainer as DC
from os import path as osp
from mmdet3d.registry import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from .vad_custom_nuscenes_eval import NuScenesEval_custom
from nuscenes.eval.common.utils import center_distance
from mmdet3d.structures import LiDARInstance3DBoxes
from models.experimental.vadv2.demo.nuscenes_box import CustomNuscenesBox
from shapely import affinity, ops
from mmdet3d.datasets.transforms.formating import to_tensor
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.detection.constants import DETECTION_NAMES


class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates"""

    def __init__(
        self,
        instance_line_list,
        sample_dist=1,
        num_samples=250,
        padding=False,
        fixed_num=-1,
        padding_value=-10000,
        patch_size=None,
    ):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value

        self.instance_list = instance_line_list

    @property
    def start_end_points(self):
        """
        return torch.Tensor([N,4]), in xstart, ystart, xend, yend form
        """
        assert len(self.instance_list) != 0
        instance_se_points_list = []
        for instance in self.instance_list:
            se_points = []
            se_points.extend(instance.coords[0])
            se_points.extend(instance.coords[-1])
            instance_se_points_list.append(se_points)
        instance_se_points_array = np.array(instance_se_points_list)
        instance_se_points_tensor = to_tensor(instance_se_points_array)
        instance_se_points_tensor = instance_se_points_tensor.to(dtype=torch.float32)
        instance_se_points_tensor[:, 0] = torch.clamp(instance_se_points_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_se_points_tensor[:, 1] = torch.clamp(instance_se_points_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_se_points_tensor[:, 2] = torch.clamp(instance_se_points_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_se_points_tensor[:, 3] = torch.clamp(instance_se_points_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_se_points_tensor

    @property
    def bbox(self):
        """
        return torch.Tensor([N,4]), in xmin, ymin, xmax, ymax form
        """
        assert len(self.instance_list) != 0
        instance_bbox_list = []
        for instance in self.instance_list:
            # bounds is bbox: [xmin, ymin, xmax, ymax]
            instance_bbox_list.append(instance.bounds)
        instance_bbox_array = np.array(instance_bbox_list)
        instance_bbox_tensor = to_tensor(instance_bbox_array)
        instance_bbox_tensor = instance_bbox_tensor.to(dtype=torch.float32)
        instance_bbox_tensor[:, 0] = torch.clamp(instance_bbox_tensor[:, 0], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 1] = torch.clamp(instance_bbox_tensor[:, 1], min=-self.max_y, max=self.max_y)
        instance_bbox_tensor[:, 2] = torch.clamp(instance_bbox_tensor[:, 2], min=-self.max_x, max=self.max_x)
        instance_bbox_tensor[:, 3] = torch.clamp(instance_bbox_tensor[:, 3], min=-self.max_y, max=self.max_y)
        return instance_bbox_tensor

    @property
    def fixed_num_sampled_points(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2
            )
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_ambiguity(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(
                -1, 2
            )
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        instance_points_tensor = instance_points_tensor.unsqueeze(1)
        return instance_points_tensor

    @property
    def fixed_num_sampled_points_torch(self):
        """
        return torch.Tensor([N,fixed_num,2]), in xmin, ymin, xmax, ymax form
            N means the num of instances
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            # distances = np.linspace(0, instance.length, self.fixed_num)
            # sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            poly_pts = to_tensor(np.array(list(instance.coords)))
            poly_pts = poly_pts.unsqueeze(0).permute(0, 2, 1)
            sampled_pts = torch.nn.functional.interpolate(
                poly_pts, size=(self.fixed_num), mode="linear", align_corners=True
            )
            sampled_pts = sampled_pts.permute(0, 2, 1).squeeze(0)
            instance_points_list.append(sampled_pts)
        # instance_points_array = np.array(instance_points_list)
        # instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = torch.stack(instance_points_list, dim=0)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:, :, 0] = torch.clamp(instance_points_tensor[:, :, 0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:, :, 1] = torch.clamp(instance_points_tensor[:, :, 1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor

    @property
    def shift_fixed_num_sampled_points(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num - shift_pts.shape[0], fixed_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v1(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
            shift_pts_list = []
            if is_poly:
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num, pts_num, num_coords))
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num - shift_pts.shape[0], pts_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v2(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]
                ).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape

            if shifts_num > final_shift_num:
                index = np.random.choice(multi_shifts_pts.shape[0], final_shift_num, replace=False)
                multi_shifts_pts = multi_shifts_pts[index]

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(dtype=torch.float32)

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0], min=-self.max_x, max=self.max_x
            )
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1], min=-self.max_y, max=self.max_y
            )
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < final_shift_num:
                padding = torch.full(
                    [final_shift_num - multi_shifts_pts_tensor.shape[0], self.fixed_num, 2], self.padding_value
                )
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v3(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        assert len(self.instance_list) != 0
        instances_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            poly_pts = np.array(list(instance.coords))
            start_pts = poly_pts[0]
            end_pts = poly_pts[-1]
            is_poly = np.equal(start_pts, end_pts)
            is_poly = is_poly.all()
            shift_pts_list = []
            pts_num, coords_num = poly_pts.shape
            shift_num = pts_num - 1
            final_shift_num = self.fixed_num - 1
            if is_poly:
                pts_to_shift = poly_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                flip_pts_to_shift = np.flip(pts_to_shift, axis=0)
                for shift_right_i in range(shift_num):
                    shift_pts = np.roll(flip_pts_to_shift, shift_right_i, axis=0)
                    pts_to_concat = shift_pts[0]
                    pts_to_concat = np.expand_dims(pts_to_concat, axis=0)
                    shift_pts = np.concatenate((shift_pts, pts_to_concat), axis=0)
                    shift_instance = LineString(shift_pts)
                    shift_sampled_points = np.array(
                        [list(shift_instance.interpolate(distance).coords) for distance in distances]
                    ).reshape(-1, 2)
                    shift_pts_list.append(shift_sampled_points)
                # import pdb;pdb.set_trace()
            else:
                sampled_points = np.array(
                    [list(instance.interpolate(distance).coords) for distance in distances]
                ).reshape(-1, 2)
                flip_sampled_points = np.flip(sampled_points, axis=0)
                shift_pts_list.append(sampled_points)
                shift_pts_list.append(flip_sampled_points)

            multi_shifts_pts = np.stack(shift_pts_list, axis=0)
            shifts_num, _, _ = multi_shifts_pts.shape
            # import pdb;pdb.set_trace()
            if shifts_num > 2 * final_shift_num:
                index = np.random.choice(shift_num, final_shift_num, replace=False)
                flip0_shifts_pts = multi_shifts_pts[index]
                flip1_shifts_pts = multi_shifts_pts[index + shift_num]
                multi_shifts_pts = np.concatenate((flip0_shifts_pts, flip1_shifts_pts), axis=0)

            multi_shifts_pts_tensor = to_tensor(multi_shifts_pts)
            multi_shifts_pts_tensor = multi_shifts_pts_tensor.to(dtype=torch.float32)

            multi_shifts_pts_tensor[:, :, 0] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 0], min=-self.max_x, max=self.max_x
            )
            multi_shifts_pts_tensor[:, :, 1] = torch.clamp(
                multi_shifts_pts_tensor[:, :, 1], min=-self.max_y, max=self.max_y
            )
            # if not is_poly:
            if multi_shifts_pts_tensor.shape[0] < 2 * final_shift_num:
                padding = torch.full(
                    [final_shift_num * 2 - multi_shifts_pts_tensor.shape[0], self.fixed_num, 2], self.padding_value
                )
                multi_shifts_pts_tensor = torch.cat([multi_shifts_pts_tensor, padding], dim=0)
            instances_list.append(multi_shifts_pts_tensor)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_v4(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            pts_num = fixed_num_pts.shape[0]
            shift_num = pts_num - 1
            shift_pts_list = []
            if is_poly:
                pts_to_shift = fixed_num_pts[:-1, :]
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(pts_to_shift.roll(shift_right_i, 0))
                flip_pts_to_shift = pts_to_shift.flip(0)
                for shift_right_i in range(shift_num):
                    shift_pts_list.append(flip_pts_to_shift.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            if is_poly:
                _, _, num_coords = shift_pts.shape
                tmp_shift_pts = shift_pts.new_zeros((shift_num * 2, pts_num, num_coords))
                tmp_shift_pts[:, :-1, :] = shift_pts
                tmp_shift_pts[:, -1, :] = shift_pts[:, 0, :]
                shift_pts = tmp_shift_pts

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([shift_num * 2 - shift_pts.shape[0], pts_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    @property
    def shift_fixed_num_sampled_points_torch(self):
        """
        return  [instances_num, num_shifts, fixed_num, 2]
        """
        fixed_num_sampled_points = self.fixed_num_sampled_points_torch
        instances_list = []
        is_poly = False
        # is_line = False
        # import pdb;pdb.set_trace()
        for fixed_num_pts in fixed_num_sampled_points:
            # [fixed_num, 2]
            is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])
            fixed_num = fixed_num_pts.shape[0]
            shift_pts_list = []
            if is_poly:
                # import pdb;pdb.set_trace()
                for shift_right_i in range(fixed_num):
                    shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))
            else:
                shift_pts_list.append(fixed_num_pts)
                shift_pts_list.append(fixed_num_pts.flip(0))
            shift_pts = torch.stack(shift_pts_list, dim=0)

            shift_pts[:, :, 0] = torch.clamp(shift_pts[:, :, 0], min=-self.max_x, max=self.max_x)
            shift_pts[:, :, 1] = torch.clamp(shift_pts[:, :, 1], min=-self.max_y, max=self.max_y)

            if not is_poly:
                padding = torch.full([fixed_num - shift_pts.shape[0], fixed_num, 2], self.padding_value)
                shift_pts = torch.cat([shift_pts, padding], dim=0)
                # padding = np.zeros((self.num_samples - len(sampled_points), 2))
                # sampled_points = np.concatenate([sampled_points, padding], axis=0)
            instances_list.append(shift_pts)
        instances_tensor = torch.stack(instances_list, dim=0)
        instances_tensor = instances_tensor.to(dtype=torch.float32)
        return instances_tensor

    # @property
    # def polyline_points(self):
    #     """
    #     return [[x0,y0],[x1,y1],...]
    #     """
    #     assert len(self.instance_list) != 0
    #     for instance in self.instance_list:


class VectorizedLocalMap(object):
    CLASS2LABEL = {"road_divider": 0, "lane_divider": 0, "ped_crossing": 1, "contours": 2, "others": -1}

    def __init__(
        self,
        dataroot,
        patch_size,
        map_classes=["divider", "ped_crossing", "boundary"],
        line_classes=["road_divider", "lane_divider"],
        ped_crossing_classes=["ped_crossing"],
        contour_classes=["road_segment", "lane"],
        sample_dist=1,
        num_samples=250,
        padding=False,
        fixed_ptsnum_per_line=-1,
        padding_value=-10000,
    ):
        """
        Args:
            fixed_ptsnum_per_line = -1 : no fixed num
        """
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ["boston-seaport", "singapore-hollandvillage", "singapore-onenorth", "singapore-queenstown"]
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(dataroot=self.data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])

        self.patch_size = patch_size
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value

    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        """
        use lidar2global to get gt map layers
        """

        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)

        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        # import pdb;pdb.set_trace()
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == "divider":
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == "ped_crossing":
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
                # ped_vector_list = self.ped_geoms_to_vectors(ped_geom)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                # import pdb;pdb.set_trace()
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get("ped_crossing", -1)))
            elif vec_class == "boundary":
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
                # import pdb;pdb.set_trace()
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                # import pdb;pdb.set_trace()
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get("contours", -1)))
            else:
                raise ValueError(f"WRONG vec_class: {vec_class}")

        # filter out -1
        filtered_vectors = []
        gt_pts_loc_3d = []
        gt_pts_num_3d = []
        gt_labels = []
        gt_instance = []
        for instance, type in vectors:
            if type != -1:
                gt_instance.append(instance)
                gt_labels.append(type)

        gt_instance = LiDARInstanceLines(
            gt_instance,
            self.sample_dist,
            self.num_samples,
            self.padding,
            self.fixed_num,
            self.padding_value,
            patch_size=self.patch_size,
        )

        anns_results = dict(
            gt_vecs_pts_loc=gt_instance,
            gt_vecs_label=gt_labels,
        )
        # import pdb;pdb.set_trace()
        return anns_results

    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                # import pdb;pdb.set_trace()
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name, location)
                # import pdb;pdb.set_trace()
                # geoms = self.map_explorer[location]._get_layer_line(patch_box, patch_angle, layer_name)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name, location)
                # geoms = self.map_explorer[location]._get_layer_polygon(patch_box, patch_angle, layer_name)
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

    def _one_type_line_geom_to_instances(self, line_geom):
        line_instances = []

        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == "MultiLineString":
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == "LineString":
                    line_instances.append(line)
                else:
                    raise NotImplementedError
        return line_instances

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

    def ped_poly_geoms_to_instances(self, ped_geom):
        # import pdb;pdb.set_trace()
        ped = ped_geom[0][1]
        union_segments = ops.unary_union(ped)
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        # local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
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

        return self._one_type_line_geom_to_instances(results)

    def poly_geoms_to_instances(self, polygon_geom):
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

        return self._one_type_line_geom_to_instances(results)

    def line_geoms_to_vectors(self, line_geom):
        line_vectors_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_vectors = self._one_type_line_geom_to_vectors(a_type_of_lines)
            line_vectors_dict[line_type] = one_type_vectors

        return line_vectors_dict

    def line_geoms_to_instances(self, line_geom):
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances

        return line_instances_dict

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

    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_polygon_layers:
            raise ValueError("{} is not a polygonal layer".format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        records = getattr(self.map_explorer[location].map_api, layer_name)

        polygon_list = []
        if layer_name == "drivable_area":
            for record in records:
                polygons = [
                    self.map_explorer[location].map_api.extract_polygon(polygon_token)
                    for polygon_token in record["polygon_tokens"]
                ]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(
                            new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False
                        )
                        new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == "Polygon":
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.map_explorer[location].map_api.extract_polygon(record["polygon_token"])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(
                            new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False
                        )
                        new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == "Polygon":
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list

    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        if layer_name not in self.map_explorer[location].map_api.non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name == "traffic_light":
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.map_explorer[location].map_api, layer_name)
        for record in records:
            line = self.map_explorer[location].map_api.extract_line(record["line_token"])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list

    def get_ped_crossing_line(self, patch_box, patch_angle, location):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        polygon_list = []
        records = getattr(self.map_explorer[location].map_api, "ped_crossing")
        # records = getattr(self.nusc_maps[location], 'ped_crossing')
        for record in records:
            polygon = self.map_explorer[location].map_api.extract_polygon(record["polygon_token"])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(
                        new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False
                    )
                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == "Polygon":
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)

        return polygon_list

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

            # tmpdistances = np.linspace(0, line.length, 2)
            # tmpsampled_points = np.array([list(line.interpolate(tmpdistance).coords) for tmpdistance in tmpdistances]).reshape(-1, 2)
        # import pdb;pdb.set_trace()
        # if self.normalize:
        #     sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])

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

            # if self.normalize:
            #     sampled_points = sampled_points / np.array([self.patch_size[1], self.patch_size[0]])
            #     num_valid = len(sampled_points)

        return sampled_points, num_valid


###############################################################################################################
###############################################################################################################
###############################################################################################################


class v1CustomDetectionConfig:
    """Data class that specifies the detection evaluation settings."""

    def __init__(
        self,
        class_range_x: Dict[str, int],
        class_range_y: Dict[str, int],
        dist_fcn: str,
        dist_ths: List[float],
        dist_th_tp: float,
        min_recall: float,
        min_precision: float,
        max_boxes_per_sample: int,
        mean_ap_weight: int,
    ):
        assert set(class_range_x.keys()) == set(DETECTION_NAMES), "Class count mismatch."
        assert dist_th_tp in dist_ths, "dist_th_tp must be in set of dist_ths."

        self.class_range_x = class_range_x
        self.class_range_y = class_range_y
        self.dist_fcn = dist_fcn
        self.dist_ths = dist_ths
        self.dist_th_tp = dist_th_tp
        self.min_recall = min_recall
        self.min_precision = min_precision
        self.max_boxes_per_sample = max_boxes_per_sample
        self.mean_ap_weight = mean_ap_weight

        self.class_names = self.class_range_y.keys()

    def __eq__(self, other):
        eq = True
        for key in self.serialize().keys():
            eq = eq and np.array_equal(getattr(self, key), getattr(other, key))
        return eq

    def serialize(self) -> dict:
        """Serialize instance into json-friendly format."""
        return {
            "class_range_x": self.class_range_x,
            "class_range_y": self.class_range_y,
            "dist_fcn": self.dist_fcn,
            "dist_ths": self.dist_ths,
            "dist_th_tp": self.dist_th_tp,
            "min_recall": self.min_recall,
            "min_precision": self.min_precision,
            "max_boxes_per_sample": self.max_boxes_per_sample,
            "mean_ap_weight": self.mean_ap_weight,
        }

    @classmethod
    def deserialize(cls, content: dict):
        """Initialize from serialized dictionary."""
        return cls(
            content["class_range_x"],
            content["class_range_y"],
            content["dist_fcn"],
            content["dist_ths"],
            content["dist_th_tp"],
            content["min_recall"],
            content["min_precision"],
            content["max_boxes_per_sample"],
            content["mean_ap_weight"],
        )

    @property
    def dist_fcn_callable(self):
        """Return the distance function corresponding to the dist_fcn string."""
        if self.dist_fcn == "center_distance":
            return center_distance
        else:
            raise Exception("Error: Unknown distance function %s!" % self.dist_fcn)


@DATASETS.register_module()
class VADCustomNuScenesDataset(CustomNuScenesDataset):
    r"""Custom NuScenes Dataset."""
    MAPCLASSES = ("divider",)

    def __init__(
        self,
        queue_length=4,
        bev_size=(200, 200),
        overlap_test=False,
        with_attr=True,
        fut_ts=6,
        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        map_classes=None,
        map_ann_file=None,
        map_fixed_ptsnum_per_line=-1,
        map_eval_use_same_gt_sample_num_flag=False,
        padding_value=-10000,
        use_pkl_result=False,
        custom_eval_version="vad_nusc_detection_cvpr_2019",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.queue_length = queue_length
        self.overlap_test = overlap_test
        self.bev_size = bev_size
        self.with_attr = with_attr
        self.fut_ts = fut_ts
        self.use_pkl_result = use_pkl_result

        self.custom_eval_version = custom_eval_version
        # Check if config exists.
        this_dir = os.path.dirname(os.path.abspath(__file__))
        cfg_path = os.path.join(this_dir, "%s.json" % self.custom_eval_version)
        assert os.path.exists(cfg_path), "Requested unknown configuration {}".format(self.custom_eval_version)
        # Load config file and deserialize it.
        with open(cfg_path, "r") as f:
            data = json.load(f)
        self.custom_eval_detection_configs = v1CustomDetectionConfig.deserialize(data)

        self.map_ann_file = map_ann_file
        self.MAPCLASSES = self.get_map_classes(map_classes)
        self.NUM_MAPCLASSES = len(self.MAPCLASSES)
        self.pc_range = pc_range
        patch_h = pc_range[4] - pc_range[1]
        patch_w = pc_range[3] - pc_range[0]
        self.patch_size = (patch_h, patch_w)
        self.padding_value = padding_value
        self.fixed_num = map_fixed_ptsnum_per_line
        self.eval_use_same_gt_sample_num_flag = map_eval_use_same_gt_sample_num_flag
        self.vector_map = VectorizedLocalMap(
            kwargs["data_root"],
            patch_size=self.patch_size,
            map_classes=self.MAPCLASSES,
            fixed_ptsnum_per_line=map_fixed_ptsnum_per_line,
            padding_value=self.padding_value,
        )
        self.is_vis_on_test = True

    @classmethod
    def get_map_classes(cls, map_classes=None):
        """Get class names of current dataset.

        Args:
            classes (Sequence[str] | str | None): If classes is None, use
                default CLASSES defined by builtin dataset. If classes is a
                string, take it as a file name. The file contains the name of
                classes where each line contains one class name. If classes is
                a tuple or list, override the CLASSES defined by the dataset.

        Return:
            list[str]: A list of class names.
        """
        if map_classes is None:
            return cls.MAPCLASSES

        if isinstance(map_classes, str):
            # take it as a file path
            class_names = mmengine.list_from_file(map_classes)
        elif isinstance(map_classes, (tuple, list)):
            class_names = map_classes
        else:
            raise ValueError(f"Unsupported type {type(map_classes)} of map classes.")

        return class_names

    def vectormap_pipeline(self, example, input_dict):
        """
        `example` type: <class 'dict'>
            keys: 'img_metas', 'gt_bboxes_3d', 'gt_labels_3d', 'img';
                  all keys type is 'DataContainer';
                  'img_metas' cpu_only=True, type is dict, others are false;
                  'gt_labels_3d' shape torch.size([num_samples]), stack=False,
                                padding_value=0, cpu_only=False
                  'gt_bboxes_3d': stack=False, cpu_only=True
        """
        # import pdb;pdb.set_trace()
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(input_dict["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = input_dict["lidar2ego_translation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(input_dict["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = input_dict["ego2global_translation"]

        lidar2global = ego2global @ lidar2ego

        lidar2global_translation = list(lidar2global[:3, 3])
        lidar2global_rotation = list(Quaternion(matrix=lidar2global).q)

        location = input_dict["map_location"]
        ego2global_translation = input_dict["ego2global_translation"]
        ego2global_rotation = input_dict["ego2global_rotation"]
        anns_results = self.vector_map.gen_vectorized_samples(location, lidar2global_translation, lidar2global_rotation)

        """
        anns_results, type: dict
            'gt_vecs_pts_loc': list[num_vecs], vec with num_points*2 coordinates
            'gt_vecs_pts_num': list[num_vecs], vec with num_points
            'gt_vecs_label': list[num_vecs], vec with cls index
        """
        gt_vecs_label = to_tensor(anns_results["gt_vecs_label"])
        if isinstance(anns_results["gt_vecs_pts_loc"], LiDARInstanceLines):
            gt_vecs_pts_loc = anns_results["gt_vecs_pts_loc"]
        else:
            gt_vecs_pts_loc = to_tensor(anns_results["gt_vecs_pts_loc"])
            try:
                gt_vecs_pts_loc = gt_vecs_pts_loc.flatten(1).to(dtype=torch.float32)
            except:
                # empty tensor, will be passed in train,
                # but we preserve it for test
                gt_vecs_pts_loc = gt_vecs_pts_loc

        example[0]["map_gt_labels_3d"] = DC(gt_vecs_label, cpu_only=False)
        example[0]["map_gt_bboxes_3d"] = DC(gt_vecs_pts_loc, cpu_only=True)

        return example[0]

    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        data_queue = []

        # temporal aug
        prev_indexs_list = list(range(index - self.queue_length, index))
        random.shuffle(prev_indexs_list)
        prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        ##

        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        frame_idx = input_dict["frame_idx"]
        scene_token = input_dict["scene_token"]
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        example = self.vectormap_pipeline(example, input_dict)
        if self.filter_empty_gt and (
            (example is None or ~(example["gt_labels_3d"]._data != -1).any())
            or (example is None or ~(example["map_gt_labels_3d"]._data != -1).any())
        ):
            return None
        data_queue.insert(0, example)
        for i in prev_indexs_list:
            i = max(0, i)
            input_dict = self.get_data_info(i)
            if input_dict is None:
                return None
            if input_dict["frame_idx"] < frame_idx and input_dict["scene_token"] == scene_token:
                self.pre_pipeline(input_dict)
                example = self.pipeline(input_dict)
                example = self.vectormap_pipeline(example, input_dict)
                if (
                    self.filter_empty_gt
                    and (example is None or ~(example["gt_labels_3d"]._data != -1).any())
                    and (example is None or ~(example["map_gt_labels_3d"]._data != -1).any())
                ):
                    return None
                frame_idx = input_dict["frame_idx"]
            data_queue.insert(0, copy.deepcopy(example))
        return self.union2one(data_queue)

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)

        if self.is_vis_on_test:
            example = self.vectormap_pipeline(example, input_dict)

        return example

    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        """
        imgs_list = [each["img"].data for each in queue]
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
        return queue

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0
        gt_bboxes_3d = info["gt_boxes"][mask]
        gt_names_3d = info["gt_names"][mask]
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

        if self.with_attr:
            gt_fut_trajs = info["gt_agent_fut_trajs"][mask]
            gt_fut_masks = info["gt_agent_fut_masks"][mask]
            gt_fut_goal = info["gt_agent_fut_goal"][mask]
            gt_lcf_feat = info["gt_agent_lcf_feat"][mask]
            gt_fut_yaw = info["gt_agent_fut_yaw"][mask]
            attr_labels = np.concatenate(
                [gt_fut_trajs, gt_fut_masks, gt_fut_goal[..., None], gt_lcf_feat, gt_fut_yaw], axis=-1
            ).astype(np.float32)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, gt_names=gt_names_3d, attr_labels=attr_labels
        )

        return anns_results

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info["token"],
            pts_filename=info["lidar_path"],
            sweeps=info["sweeps"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            prev_idx=info["prev"],
            next_idx=info["next"],
            scene_token=info["scene_token"],
            can_bus=info["can_bus"],
            frame_idx=info["frame_idx"],
            timestamp=info["timestamp"] / 1e6,
            fut_valid_flag=info["fut_valid_flag"],
            map_location=info["map_location"],
            ego_his_trajs=info["gt_ego_his_trajs"],
            ego_fut_trajs=info["gt_ego_fut_trajs"],
            ego_fut_masks=info["gt_ego_fut_masks"],
            ego_fut_cmd=info["gt_ego_fut_cmd"],
            ego_lcf_feat=info["gt_ego_lcf_feat"],
        )
        # lidar to ego transform
        lidar2ego = np.eye(4).astype(np.float32)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = info["lidar2ego_translation"]
        input_dict["lidar2ego"] = lidar2ego

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            input_dict["camera2ego"] = []
            input_dict["camera_intrinsics"] = []
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

                # camera to ego transform
                camera2ego = np.eye(4).astype(np.float32)
                camera2ego[:3, :3] = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
                camera2ego[:3, 3] = cam_info["sensor2ego_translation"]
                input_dict["camera2ego"].append(camera2ego)
                # camera intrinsics
                camera_intrinsics = np.eye(4).astype(np.float32)
                camera_intrinsics[:3, :3] = cam_info["cam_intrinsic"]
                input_dict["camera_intrinsics"].append(camera_intrinsics)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        # NOTE: now we load gt in test_mode for evaluating
        # if not self.test_mode:
        #     annos = self.get_ann_info(index)
        #     input_dict['ann_info'] = annos

        annos = self.get_ann_info(index)
        input_dict["ann_info"] = annos

        rotation = Quaternion(input_dict["ego2global_rotation"])
        translation = input_dict["ego2global_translation"]
        can_bus = input_dict["can_bus"]
        can_bus[:3] = translation
        can_bus[3:7] = rotation
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        if patch_angle < 0:
            patch_angle += 360
        can_bus[-2] = patch_angle / 180 * np.pi
        can_bus[-1] = patch_angle

        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(input_dict["lidar2ego_rotation"]).rotation_matrix
        lidar2ego[:3, 3] = input_dict["lidar2ego_translation"]
        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(input_dict["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = input_dict["ego2global_translation"]
        lidar2global = ego2global @ lidar2ego
        input_dict["lidar2global"] = lidar2global

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """

        return self.prepare_test_data(idx)

    def _format_gt(self):
        gt_annos = []
        print("Start to convert gt map format...")
        # assert self.map_ann_file is not None
        if not os.path.exists(self.map_ann_file):
            dataset_length = len(self)
            prog_bar = ProgressBar(dataset_length)
            mapped_class_names = self.MAPCLASSES
            for sample_id in range(dataset_length):
                sample_token = self.data_infos[sample_id]["token"]
                gt_anno = {}
                gt_anno["sample_token"] = sample_token
                # gt_sample_annos = []
                gt_sample_dict = {}
                gt_sample_dict = self.vectormap_pipeline(gt_sample_dict, self.data_infos[sample_id])
                gt_labels = gt_sample_dict["map_gt_labels_3d"].data.numpy()
                gt_vecs = gt_sample_dict["map_gt_bboxes_3d"].data.instance_list
                gt_vec_list = []
                for i, (gt_label, gt_vec) in enumerate(zip(gt_labels, gt_vecs)):
                    name = mapped_class_names[gt_label]
                    anno = dict(
                        pts=np.array(list(gt_vec.coords)),
                        pts_num=len(list(gt_vec.coords)),
                        cls_name=name,
                        type=gt_label,
                    )
                    gt_vec_list.append(anno)
                gt_anno["vectors"] = gt_vec_list
                gt_annos.append(gt_anno)

                prog_bar.update()
            nusc_submissions = {"GTs": gt_annos}
            print("\n GT anns writes to", self.map_ann_file)
            mmengine.dump(nusc_submissions, self.map_ann_file)
        else:
            print(f"{self.map_ann_file} exist, not update")

    def _format_bbox(self, results, jsonfile_prefix=None, score_thresh=0.2):
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
        det_mapped_class_names = self.CLASSES

        # assert self.map_ann_file is not None
        map_pred_annos = {}
        map_mapped_class_names = self.MAPCLASSES

        plan_annos = {}

        print("Start to convert detection format...")
        for sample_id, det in enumerate(mmengine.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(det)
            sample_token = self.data_infos[sample_id]["token"]

            plan_annos[sample_token] = [det["ego_fut_preds"], det["ego_fut_cmd"]]

            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                det_mapped_class_names,
                self.custom_eval_detection_configs,
                self.eval_version,
            )
            for i, box in enumerate(boxes):
                if box.score < score_thresh:
                    continue
                name = det_mapped_class_names[box.label]
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
                    fut_traj=box.fut_trajs.tolist(),
                )
                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos

            map_pred_anno = {}
            vecs = output_to_vecs(det)
            sample_token = self.data_infos[sample_id]["token"]
            map_pred_anno["sample_token"] = sample_token
            pred_vec_list = []
            for i, vec in enumerate(vecs):
                name = map_mapped_class_names[vec["label"]]
                anno = dict(
                    # sample_token=sample_token,
                    pts=vec["pts"],
                    pts_num=len(vec["pts"]),
                    cls_name=name,
                    type=vec["label"],
                    confidence_level=vec["score"],
                )
                pred_vec_list.append(anno)
                # annos.append(nusc_anno)
            # nusc_annos[sample_token] = annos
            map_pred_anno["vectors"] = pred_vec_list
            map_pred_annos[sample_token] = map_pred_anno

        if not os.path.exists(self.map_ann_file):
            self._format_gt()
        else:
            print(f"{self.map_ann_file} exist, not update")
        # with open(self.map_ann_file,'r') as f:
        #     GT_anns = json.load(f)
        # gt_annos = GT_anns['GTs']

        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
            "map_results": map_pred_annos,
            "plan_results": plan_annos
            # 'GTs': gt_annos
        }

        mmengine.mkdir_or_exist(jsonfile_prefix)
        if self.use_pkl_result:
            res_path = osp.join(jsonfile_prefix, "results_nusc.pkl")
        else:
            res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Results writes to", res_path)
        mmengine.dump(nusc_submissions, res_path)
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

        if isinstance(results, dict):
            # print(f'results must be a list, but get dict, keys={results.keys()}')
            # assert isinstance(results, list)
            results = results["bbox_results"]
        assert isinstance(results, list)
        assert len(results) == len(self), "The length of results is not equal to the dataset len: {} != {}".format(
            len(results), len(self)
        )

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in results[0]:
                if name == "metric_results":
                    continue
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update({name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def _evaluate_single(self, result_path, logger=None, metric="bbox", map_metric="chamfer", result_name="pts_bbox"):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation. Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        detail = dict()
        from nuscenes import NuScenes

        self.nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)

        output_dir = osp.join(*osp.split(result_path)[:-1])

        eval_set_map = {
            "v1.0-mini": "mini_val",
            # 'v1.0-trainval': 'val',
        }
        self.nusc_eval = NuScenesEval_custom(
            self.nusc,
            config=self.custom_eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False,
            overlap_test=self.overlap_test,
            data_infos=self.data_infos,
        )
        self.nusc_eval.main(plot_examples=0, render_curves=False)
        # record metrics
        metrics = mmengine.load(osp.join(output_dir, "metrics_summary.json"))
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

        from models.experimental.vadv2.demo.utils.mean_op import eval_map, format_res_gt_by_classes

        result_path = osp.abspath(result_path)

        print("Formating results & gts by classes")
        pred_results = mmengine.load(result_path)
        map_results = pred_results["map_results"]
        gt_anns = mmengine.load(self.map_ann_file)
        map_annotations = gt_anns["GTs"]
        cls_gens, cls_gts = format_res_gt_by_classes(
            result_path,
            map_results,
            map_annotations,
            cls_names=self.MAPCLASSES,
            num_pred_pts_per_instance=self.fixed_num,
            eval_use_same_gt_sample_num_flag=self.eval_use_same_gt_sample_num_flag,
            pc_range=self.pc_range,
        )
        map_metrics = map_metric if isinstance(map_metric, list) else [map_metric]
        allowed_metrics = ["chamfer", "iou"]
        for metric in map_metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported")
        for metric in map_metrics:
            print("-*" * 10 + f"use metric:{metric}" + "-*" * 10)
            if metric == "chamfer":
                thresholds = [0.5, 1.0, 1.5]
            elif metric == "iou":
                thresholds = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
            cls_aps = np.zeros((len(thresholds), self.NUM_MAPCLASSES))
            for i, thr in enumerate(thresholds):
                print("-*" * 10 + f"threshhold:{thr}" + "-*" * 10)
                mAP, cls_ap = eval_map(
                    map_results,
                    map_annotations,
                    cls_gens,
                    cls_gts,
                    threshold=thr,
                    cls_names=self.MAPCLASSES,
                    logger=logger,
                    num_pred_pts_per_instance=self.fixed_num,
                    pc_range=self.pc_range,
                    metric=metric,
                )
                for j in range(self.NUM_MAPCLASSES):
                    cls_aps[i, j] = cls_ap[j]["ap"]
            for i, name in enumerate(self.MAPCLASSES):
                print("{}: {}".format(name, cls_aps.mean(0)[i]))
                detail["NuscMap_{}/{}_AP".format(metric, name)] = cls_aps.mean(0)[i]
            print("map: {}".format(cls_aps.mean(0).mean()))
            detail["NuscMap_{}/mAP".format(metric)] = cls_aps.mean(0).mean()
            for i, name in enumerate(self.MAPCLASSES):
                for j, thr in enumerate(thresholds):
                    if metric == "chamfer":
                        detail["NuscMap_{}/{}_AP_thr_{}".format(metric, name, thr)] = cls_aps[j][i]
                    elif metric == "iou":
                        if thr == 0.5 or thr == 0.75:
                            detail["NuscMap_{}/{}_AP_thr_{}".format(metric, name, thr)] = cls_aps[j][i]

        return detail

    def evaluate(
        self,
        results,
        metric="bbox",
        map_metric="chamfer",
        logger=None,
        jsonfile_prefix=None,
        result_names=["pts_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        result_metric_names = ["EPA", "ADE", "FDE", "MR"]
        motion_cls_names = ["car", "pedestrian"]
        motion_metric_names = ["gt", "cnt_ade", "cnt_fde", "hit", "fp", "ADE", "FDE", "MR"]
        all_metric_dict = {}
        for met in motion_metric_names:
            for cls in motion_cls_names:
                all_metric_dict[met + "_" + cls] = 0.0
        result_dict = {}
        for met in result_metric_names:
            for cls in motion_cls_names:
                result_dict[met + "_" + cls] = 0.0

        alpha = 0.5

        for i in range(len(results)):
            for key in all_metric_dict.keys():
                all_metric_dict[key] += results[i]["metric_results"][key]

        for cls in motion_cls_names:
            result_dict["EPA_" + cls] = (
                all_metric_dict["hit_" + cls] - alpha * all_metric_dict["fp_" + cls]
            ) / all_metric_dict["gt_" + cls]
            result_dict["ADE_" + cls] = all_metric_dict["ADE_" + cls] / all_metric_dict["cnt_ade_" + cls]
            result_dict["FDE_" + cls] = all_metric_dict["FDE_" + cls] / all_metric_dict["cnt_fde_" + cls]
            result_dict["MR_" + cls] = all_metric_dict["MR_" + cls] / all_metric_dict["cnt_fde_" + cls]

        print("\n")
        print("-------------- Motion Prediction --------------")
        for k, v in result_dict.items():
            print(f"{k}: {v}")

        # NOTE: print planning metric
        print("\n")
        print("-------------- Planning --------------")
        metric_dict = None
        num_valid = 0
        for res in results:
            if res["metric_results"]["fut_valid_flag"]:
                num_valid += 1
            else:
                continue
            if metric_dict is None:
                metric_dict = copy.deepcopy(res["metric_results"])
            else:
                for k in res["metric_results"].keys():
                    metric_dict[k] += res["metric_results"][k]

        for k in metric_dict:
            metric_dict[k] = metric_dict[k] / num_valid
            print("{}:{}".format(k, metric_dict[k]))

        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print("Evaluating bboxes of {}".format(name))
                ret_dict = self._evaluate_single(result_files[name], metric=metric, map_metric=map_metric)
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files, metric=metric, map_metric=map_metric)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show:
            self.show(results, out_dir, pipeline=pipeline)
        return results_dict


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
    trajs = detection["trajs_3d"].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()
    # TODO: check whether this is necessary
    # with dir_offset & dir_limit in the head
    box_yaw = -box_yaw - np.pi / 2

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        velocity = (*box3d.tensor[i, 7:9], 0.0)
        # velo_val = np.linalg.norm(box3d[i, 7:9])
        # velo_ori = box3d[i, 6]
        # velocity = (
        # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
        box = CustomNuscenesBox(
            center=box_gravity_center[i],
            size=box_dims[i],
            orientation=quat,
            fut_trajs=trajs[i],
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_x_map = eval_configs.class_range_x
        cls_range_y_map = eval_configs.class_range_y
        x_distance, y_distance = box.center[0], box.center[1]
        det_range_x = cls_range_x_map[classes[box.label]]
        det_range_y = cls_range_y_map[classes[box.label]]
        if abs(x_distance) > det_range_x or abs(y_distance) > det_range_y:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list


def output_to_vecs(detection):
    box3d = detection["map_boxes_3d"].numpy()
    scores = detection["map_scores_3d"].numpy()
    labels = detection["map_labels_3d"].numpy()
    pts = detection["map_pts_3d"].numpy()

    vec_list = []
    # import pdb;pdb.set_trace()
    for i in range(box3d.shape[0]):
        vec = dict(
            bbox=box3d[i],  # xyxy
            label=labels[i],
            score=scores[i],
            pts=pts[i],
        )
        vec_list.append(vec)
    return vec_list

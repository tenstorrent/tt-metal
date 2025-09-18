# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Tuple
import numpy as np


def query_ball_point(new_xyz: torch.Tensor, xyz: torch.Tensor, radius: float, nsample: int) -> torch.Tensor:
    b, m, _ = new_xyz.shape
    _, n, _ = xyz.shape
    device = new_xyz.device
    radius2 = radius * radius

    diff = new_xyz.unsqueeze(2) - xyz.unsqueeze(1)
    dist2 = torch.sum(diff**2, dim=3)
    mask = dist2 < radius2

    idx = torch.zeros((b, m, nsample), dtype=torch.int32, device=device)

    arange_n = torch.arange(n, device=device).view(1, 1, n).expand(b, m, n)
    arange_n_masked = torch.where(mask, arange_n, torch.full_like(arange_n, n + 1))

    sorted_indices, _ = torch.sort(arange_n_masked, dim=2)
    first_nsample = sorted_indices[:, :, :nsample]

    invalid_mask = first_nsample == (n + 1)
    first_valid = first_nsample[:, :, 0].unsqueeze(2).expand_as(first_nsample)
    first_nsample[invalid_mask] = first_valid[invalid_mask]

    return first_nsample.to(torch.int32)


class BallQuery(nn.Module):
    def __init__(self, radius, nsample):
        super().__init__()
        self.radius = radius
        self.nsample = nsample

    def __call__(self, xyz, new_xyz):
        return query_ball_point(new_xyz, xyz, self.radius, self.nsample)


class GroupingOperation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, points, idx):
        """
        points: (B, C, N)
        idx: (B, npoint, nsample)
        """
        B, C, N = points.shape
        _, npoint, nsample = idx.shape

        idx = idx.to(torch.int64)
        # Flatten points and idx for gather operation
        points_flat = points.view(B * C, N)
        idx_flat = idx.view(B, npoint * nsample)
        # Expand idx to match points channels dimension
        idx_expand = idx_flat.unsqueeze(1).expand(-1, C, -1).contiguous().view(B * C, npoint * nsample)
        # Gather the points
        out_flat = torch.gather(points_flat, 1, idx_expand)
        # Reshape to (B, C, npoint, nsample)
        output = out_flat.view(B, C, npoint, nsample)
        return output


def furthest_point_sampling(points: torch.Tensor, n_samples: int) -> torch.Tensor:
    B, N, _ = points.shape
    device = points.device
    centroids = torch.zeros((B, n_samples), dtype=torch.long, device=device)
    distance = torch.ones((B, N), dtype=points.dtype, device=device) * 1e10
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    batch_indices = torch.arange(B, dtype=torch.long, device=device)

    for i in range(n_samples):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=2)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, dim=1)[1]

    return centroids


class FurthestPointSampling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xyz, npoint):
        """
        Args:
            xyz: (B, N, 3) input points
            npoint: int, number of points to sample

        Returns:
            fps_inds: (B, npoint) sampled indices
        """
        return furthest_point_sampling(xyz, npoint)


def gather_points(points, idx):
    B, C, N = points.shape
    M = idx.shape[1]
    idx_expand = idx.long().unsqueeze(1).expand(B, C, M)
    out = torch.gather(points, 2, idx_expand)
    return out


class GatherOperation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features, idx):
        """
        Args:
            features: (B, C, N) tensor
            idx: (B, S) or (B, S, K) tensor of indices to gather from `features`

        Returns:
            gathered: (B, C, S) or (B, C, S, K) tensor depending on idx shape
        """
        return gather_points(features, idx)


class QueryAndGroup(nn.Module):
    r"""
    Groups with a ball query of radius

    Parameters
    ---------
    radius : float32
        Radius of ball
    nsample : int32
        Maximum number of features to gather in the ball
    """

    def __init__(
        self,
        radius,
        nsample,
        use_xyz=True,
        ret_grouped_xyz=False,
        normalize_xyz=False,
        sample_uniformly=False,
        ret_unique_cnt=False,
    ):
        # type: (QueryAndGroup, float, int, bool) -> None
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz
        self.ret_grouped_xyz = ret_grouped_xyz
        self.normalize_xyz = normalize_xyz
        self.sample_uniformly = sample_uniformly
        self.ret_unique_cnt = ret_unique_cnt
        self.ball_query = BallQuery(radius=self.radius, nsample=self.nsample)
        self.grouping_operation = GroupingOperation()
        if self.ret_unique_cnt:
            assert self.sample_uniformly

    def forward(self, xyz, new_xyz, features=None):
        # type: (QueryAndGroup, torch.Tensor. torch.Tensor, torch.Tensor) -> Tuple[Torch.Tensor]
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            xyz coordinates of the features (B, N, 3)
        new_xyz : torch.Tensor
            centriods (B, npoint, 3)
        features : torch.Tensor
            Descriptors of the features (B, C, N)

        Returns
        -------
        new_features : torch.Tensor
            (B, 3 + C, npoint, nsample) tensor
        """
        idx = self.ball_query(xyz, new_xyz)

        if self.sample_uniformly:
            unique_cnt = torch.zeros((idx.shape[0], idx.shape[1]))
            for i_batch in range(idx.shape[0]):
                for i_region in range(idx.shape[1]):
                    unique_ind = torch.unique(idx[i_batch, i_region, :])
                    num_unique = unique_ind.shape[0]
                    unique_cnt[i_batch, i_region] = num_unique
                    sample_ind = torch.randint(0, num_unique, (self.nsample - num_unique,), dtype=torch.long)
                    all_ind = torch.cat((unique_ind, unique_ind[sample_ind]))
                    idx[i_batch, i_region, :] = all_ind

        xyz_trans = xyz.transpose(1, 2).contiguous()

        grouped_xyz = self.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)
        if self.normalize_xyz:
            grouped_xyz /= self.radius

        if features is not None:
            grouped_features = self.grouping_operation(features, idx)
            if self.use_xyz:
                new_features = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            assert self.use_xyz, "Cannot have not features and not use xyz as a feature!"
            new_features = grouped_xyz

        ret = [new_features]
        if self.ret_grouped_xyz:
            ret.append(grouped_xyz)
        if self.ret_unique_cnt:
            ret.append(unique_cnt)
        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


class _ConvBase(nn.Sequential):
    def __init__(
        self,
        in_size,
        out_size,
        kernel_size,
        stride,
        padding,
        activation,
        bn,
        init,
        conv=None,
        batch_norm=None,
        bias=True,
        preact=False,
        name="",
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(in_size, out_size, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)

        self.add_module(name + "conv", conv_unit)

        if not preact:
            if bn:
                self.add_module(name + "bn", bn_unit)

            if activation is not None:
                self.add_module(name + "activation", activation)


class _BNBase(nn.Sequential):
    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm2d(_BNBase):
    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class Conv2d(_ConvBase):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        *,
        kernel_size: Tuple[int, int] = (1, 1),
        stride: Tuple[int, int] = (1, 1),
        padding: Tuple[int, int] = (0, 0),
        activation=nn.ReLU(inplace=True),
        bn: bool = False,
        init=nn.init.kaiming_normal_,
        bias: bool = True,
        preact: bool = False,
        name: str = "",
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name,
        )


def shift_scale_points(pred_xyz, src_range, dst_range=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """
    if dst_range is None:
        dst_range = [
            torch.zeros((src_range[0].shape[0], 3), device=src_range[0].device),
            torch.ones((src_range[0].shape[0], 3), device=src_range[0].device),
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


def scale_points(pred_xyz, mult_factor):
    if pred_xyz.ndim == 4:
        mult_factor = mult_factor[:, None]
    scaled_xyz = pred_xyz * mult_factor[:, None, :]
    return scaled_xyz


class BoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config):
        self.dataset_config = dataset_config

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points(center_unnormalized, src_range=point_cloud_dims)
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = torch.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = angle.squeeze(-1).clamp(min=0)
        else:
            angle_per_cls = 2 * np.pi / self.dataset_config.num_angle_bin
            pred_angle_class = angle_logits.argmax(dim=-1).detach()
            angle_center = angle_per_cls * pred_angle_class
            angle = angle_center + angle_residual.gather(2, pred_angle_class.unsqueeze(-1)).squeeze(-1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = torch.nn.functional.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(self, box_center_unnorm, box_size_unnorm, box_angle):
        return self.dataset_config.box_parametrization_to_corners(box_center_unnorm, box_size_unnorm, box_angle)


def flip_axis_to_camera_tensor(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = torch.clone(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def roty_batch_tensor(t):
    input_shape = t.shape
    output = torch.zeros(tuple(list(input_shape) + [3, 3]), dtype=torch.float32, device=t.device)
    c = torch.cos(t)
    s = torch.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def get_3d_box_batch_tensor(box_size, angle, center):
    assert isinstance(box_size, torch.Tensor)
    assert isinstance(angle, torch.Tensor)
    assert isinstance(center, torch.Tensor)

    reshape_final = False
    if angle.ndim == 2:
        assert box_size.ndim == 3
        assert center.ndim == 3
        bsize = box_size.shape[0]
        nprop = box_size.shape[1]
        box_size = box_size.reshape(-1, box_size.shape[-1])
        angle = angle.reshape(-1)
        center = center.reshape(-1, 3)
        reshape_final = True

    input_shape = angle.shape
    R = roty_batch_tensor(angle)
    l = torch.unsqueeze(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = torch.unsqueeze(box_size[..., 1], -1)
    h = torch.unsqueeze(box_size[..., 2], -1)
    corners_3d = torch.zeros(tuple(list(input_shape) + [8, 3]), device=box_size.device, dtype=torch.float32)
    corners_3d[..., :, 0] = torch.cat((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = torch.cat((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = torch.cat((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = torch.matmul(corners_3d, R.permute(tlist))
    corners_3d += torch.unsqueeze(center, -2)
    if reshape_final:
        corners_3d = corners_3d.reshape(bsize, nprop, 8, 3)
    return corners_3d


def flip_axis_to_camera_np(pc):
    """Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
    Input and output are both (N,3) array
    """
    pc2 = pc.copy()
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[..., 1] *= -1
    return pc2


def roty_batch(t):
    """Rotation about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape) + [3, 3]))
    c = np.cos(t)
    s = np.sin(t)
    output[..., 0, 0] = c
    output[..., 0, 2] = s
    output[..., 1, 1] = 1
    output[..., 2, 0] = -s
    output[..., 2, 2] = c
    return output


def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def get_3d_box_batch_np(box_size, angle, center):
    input_shape = angle.shape
    R = roty_batch(angle)
    l = np.expand_dims(box_size[..., 0], -1)  # [x1,...,xn,1]
    w = np.expand_dims(box_size[..., 1], -1)
    h = np.expand_dims(box_size[..., 2], -1)
    corners_3d = np.zeros(tuple(list(input_shape) + [8, 3]))
    corners_3d[..., :, 0] = np.concatenate((l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2), -1)
    corners_3d[..., :, 1] = np.concatenate((h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2), -1)
    corners_3d[..., :, 2] = np.concatenate((w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2), -1)
    tlist = [i for i in range(len(input_shape))]
    tlist += [len(input_shape) + 1, len(input_shape)]
    corners_3d = np.matmul(corners_3d, np.transpose(R, tuple(tlist)))
    corners_3d += np.expand_dims(center, -2)
    return corners_3d


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
        R = rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)

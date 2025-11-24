# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import math
import itertools
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

from typing import Any, Dict, List, Sequence, Tuple, Union

Pose = Tuple[float, float, float]  # (x, y, yaw)


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def multi_scale_deformable_attn_pytorch(
    value: torch.Tensor,
    value_spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: torch.Tensor,
) -> torch.Tensor:
    bs, num_keys, num_heads, head_dim = value.shape
    num_levels = value_spatial_shapes.shape[0]
    num_queries = sampling_locations.shape[1]
    num_points = sampling_locations.shape[4]

    # Split value into a list of tensors for each level
    value_list = []
    start = 0
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        len_l = h_l * w_l
        value_l = value[:, start : start + len_l, :, :]
        value_list.append(value_l)
        start += len_l

    # Normalize sampling locations to [-1, 1]
    sampling_grids = []
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        grid = sampling_locations[:, :, :, lvl, :, :]
        grid = grid.clone()
        grid[..., 0] = grid[..., 0] / w_l * 2 - 1
        grid[..., 1] = grid[..., 1] / h_l * 2 - 1
        sampling_grids.append(grid)

    # Perform sampling and attention
    output = torch.zeros(bs, num_queries, num_heads, head_dim, device=value.device)
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        value_l = value_list[lvl].permute(0, 2, 3, 1).reshape(bs * num_heads, head_dim, h_l, w_l)
        grid = sampling_grids[lvl].permute(0, 2, 1, 3, 4).reshape(bs * num_heads, num_queries * num_points, 1, 2)
        sampled = F.grid_sample(value_l, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        sampled = sampled.view(bs, num_heads, head_dim, num_queries, num_points).permute(0, 3, 1, 4, 2)
        attn = attention_weights[:, :, :, lvl, :].unsqueeze(-1)
        output += (sampled * attn).sum(-2)

    return output.view(bs, num_queries, num_heads * head_dim)


# taken from  mmdet3d.structures.bbox_3d.utils import limit_period
def limit_period(
    val: Union[np.ndarray, Tensor], offset: float = 0.5, period: float = np.pi
) -> Union[np.ndarray, Tensor]:
    limited_val = val - torch.floor(val / period + offset) * period
    return limited_val


# taken from projects/DETR3D/detr3d/util import denormalize_bbox
def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)
    rot = -rot - np.pi / 2
    rot = limit_period(rot, period=np.pi * 2)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    length = normalized_bboxes[..., 2:3]
    width = normalized_bboxes[..., 3:4]
    height = normalized_bboxes[..., 5:6]

    width = width.exp()
    length = length.exp()
    height = height.exp()
    if normalized_bboxes.size(-1) > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, length, width, height, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, length, width, height, rot], dim=-1)

    return denormalized_bboxes


from abc import ABCMeta, abstractmethod


# taken from mmdet.models.task_modules import BaseBBoxCoder
class BaseBBoxCoder(metaclass=ABCMeta):
    encode_size = 4

    def __init__(self, use_box_type: bool = False, **kwargs):
        self.use_box_type = use_box_type

    @abstractmethod
    def encode(self, bboxes, gt_bboxes):
        """Encode deltas between bboxes and ground truth boxes."""

    @abstractmethod
    def decode(self, bboxes, bboxes_pred):
        """Decode the predicted bboxes according to prediction and base
        boxes."""


def multi_apply(func, *args, **kwargs):
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


class Instances:
    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        return self._image_size

    def __setattr__(self, name: str, val: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, "Adding a field of length {} to a Instances of length {}".format(
                data_len, len(self)
            )
        self._fields[name] = value

    def has(self, name: str) -> bool:
        return name in self._fields

    def remove(self, name: str) -> None:
        del self._fields[name]

    def get(self, name: str) -> Any:
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "to"):
                v = v.to(*args, **kwargs)
            ret.set(k, v)
        return ret

    def numpy(self):
        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if hasattr(v, "numpy"):
                v = v.numpy()
            ret.set(k, v)
        return ret

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "Instances":
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            if k == "kalman_models" and isinstance(item, torch.Tensor):
                ret_list = []
                for i, if_true in enumerate(item):
                    if if_true:
                        ret_list.append(self.kalman_models[i])
                ret.set(k, ret_list)

            else:
                ret.set(k, v[item])
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"]) -> "Instances":
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                values = type(v0).cat(values)
            else:
                raise ValueError("Unsupported type {} for concatenation".format(type(v0)))
            ret.set(k, values)
        return ret

    def __str__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "image_height={}, ".format(self._image_size[0])
        s += "image_width={}, ".format(self._image_size[1])
        s += "fields=[{}])".format(", ".join((f"{k}: {v}" for k, v in self._fields.items())))
        return s

    __repr__ = __str__


class CollisionNonlinearOptimizer:
    def __init__(
        self,
        trajectory_len: int,
        dt: float,
        sigma: float,
        alpha_collision: float,
        obj_pixel_pos: List[List[Tuple[float, float]]],
        device: str = "cpu",
    ):
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.sigma = sigma
        self.alpha_collision = alpha_collision
        self.obj_pixel_pos = obj_pixel_pos
        self.device = device

        # Initialize state trajectory as a learnable parameter (x, y)
        self.state = torch.nn.Parameter(torch.zeros(2, trajectory_len, device="cpu", dtype=torch.float32))

        # Reference trajectory placeholder
        self.ref_traj = torch.zeros(2, trajectory_len, device="cpu", dtype=torch.float32)

    def set_reference_trajectory(self, reference_trajectory: Sequence["Pose"]):
        reference_tensor = torch.tensor(reference_trajectory, dtype=torch.float32, device="cpu").T
        self.ref_traj = reference_tensor.clone()

        with torch.no_grad():
            self.state.copy_(self.ref_traj)

    def _compute_cost(self) -> torch.Tensor:
        # Stage cost: follow reference trajectory
        alpha_xy = 1.0

        diff = self.state - self.ref_traj
        squared_error = diff**2
        error_sum = torch.sum(squared_error)
        cost_stage = alpha_xy * error_sum

        # Collision cost
        cost_collision = 0.0
        normalizer = 1 / (2.507 * self.sigma)

        for t, obstacles in enumerate(self.obj_pixel_pos):
            x, y = self.state[0, t], self.state[1, t]
            for col_x, col_y in obstacles:
                dist_sq = (x - col_x) ** 2 + (y - col_y) ** 2
                cost_collision += self.alpha_collision * normalizer * torch.exp(-dist_sq / (2 * self.sigma**2))

        return cost_stage + cost_collision

    def solve(self, lr: float = 0.05, steps: int = 200):
        """
        Optimize the trajectory using gradient descent (Adam).
        """
        optimizer = torch.optim.Adam([self.state], lr=lr)

        for step in range(steps):
            optimizer.zero_grad()
            cost = self._compute_cost()
            cost.backward()
            optimizer.step()

        return self.state.detach().cpu().T.numpy()


def bivariate_gaussian_activation(ip):
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = torch.exp(sig_x)
    sig_y = torch.exp(sig_y)
    rho = torch.tanh(rho)
    out = torch.cat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


def rot_2d(yaw):
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2, 0, 1])
    return out


def norm_points(pos, pc_range):
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    return torch.stack([x_norm, y_norm], dim=-1)


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb


def trajectory_coordinate_transform(
    trajectory, bbox_results, with_translation_transform=True, with_rotation_transform=True
):
    batch_size = len(bbox_results)
    batched_trajectories = []
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(trajectory.device)
        bbox_centers = bboxes.gravity_center.to(trajectory.device)
        transformed_trajectory = trajectory[i, ...]
        if with_rotation_transform:
            angle = -(yaw - 3.1415953)
            rot_yaw = rot_2d(angle)
            rot_yaw = rot_yaw[:, None, None, :, :]  # A, 1, 1, 2, 2
            transformed_trajectory = rearrange(
                transformed_trajectory, "a g p t c -> a g p c t"
            )  # A, G, P, 12 ,2 -> # A, G, P, 2, 12
            transformed_trajectory = torch.matmul(rot_yaw, transformed_trajectory)  # -> A, G, P, 12, 2
            transformed_trajectory = rearrange(transformed_trajectory, "a g p c t -> a g p t c")
        if with_translation_transform:
            transformed_trajectory = bbox_centers[:, None, None, None, :2] + transformed_trajectory
        batched_trajectories.append(transformed_trajectory)
    return torch.stack(batched_trajectories)


def anchor_coordinate_transform(anchors, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    batch_size = len(bbox_results)
    batched_anchors = []
    transformed_anchors = anchors[None, ...]  # expand num agents: num_groups, num_modes, 12, 2 -> 1, ...
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(transformed_anchors.device)
        bbox_centers = bboxes.gravity_center.to(transformed_anchors.device)
        if with_rotation_transform:
            angle = yaw - 3.1415953  # num_agents, 1
            rot_yaw = rot_2d(angle)  # num_agents, 2, 2
            rot_yaw = rot_yaw[:, None, None, :, :]  # num_agents, 1, 1, 2, 2
            transformed_anchors = rearrange(
                transformed_anchors, "b g m t c -> b g m c t"
            )  # 1, num_groups, num_modes, 12, 2 -> 1, num_groups, num_modes, 2, 12
            transformed_anchors = torch.matmul(
                rot_yaw, transformed_anchors
            )  # -> num_agents, num_groups, num_modes, 12, 2
            transformed_anchors = rearrange(transformed_anchors, "b g m c t -> b g m t c")
        if with_translation_transform:
            transformed_anchors = bbox_centers[:, None, None, None, :2] + transformed_anchors
        batched_anchors.append(transformed_anchors)
    return torch.stack(batched_anchors)


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor(
        [(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long
    )

    return bev_resolution, bev_start_position, bev_dimension


# taken from mmdet3d/structures/bbox_3d/base_box3d.py
class BaseInstance3DBoxes:
    YAW_AXIS: int = 0

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 0.5, 0),
    ) -> None:
        if isinstance(tensor, Tensor):
            device = tensor.device
        else:
            device = torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.float32, device=device)
        if tensor.numel() == 0:
            tensor = tensor.reshape((-1, box_dim))
        assert tensor.dim() == 2 and tensor.size(-1) == box_dim, (
            "The box dimension must be 2 and the length of the last "
            f"dimension must be {box_dim}, but got boxes with shape "
            f"{tensor.shape}."
        )

        if tensor.shape[-1] == 6:
            assert box_dim == 6
            fake_rot = tensor.new_zeros(tensor.shape[0], 1)
            tensor = torch.cat((tensor, fake_rot), dim=-1)
            self.box_dim = box_dim + 1
            self.with_yaw = False
        else:
            self.box_dim = box_dim
            self.with_yaw = with_yaw
        self.tensor = tensor.clone()

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    def __getitem__(self, item: Union[int, slice, np.ndarray, Tensor]) -> "BaseInstance3DBoxes":
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "BaseInstance3DBoxes":
        original_type = type(self)
        return original_type(self.tensor.to(device, *args, **kwargs), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def gravity_center(self):
        pass

    @property
    def bottom_center(self):
        return self.tensor[:, :3]

    @property
    def yaw(self):
        return self.tensor[:, 6]


# taken from mmdet3d/structures/bbox_3d/lidar_box3d.py, removed all the functions from this class as they are not  invoked
class LiDARInstance3DBoxes(BaseInstance3DBoxes):
    YAW_AXIS = 2

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = torch.zeros_like(bottom_center)
        gravity_center[:, :2] = bottom_center[:, :2]
        gravity_center[:, 2] = bottom_center[:, 2] + self.tensor[:, 5] * 0.5
        return gravity_center

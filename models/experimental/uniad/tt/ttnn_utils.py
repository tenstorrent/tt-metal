# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn

import torch
import numpy as np
from torch import Tensor
import math
from typing import Any, Sequence, Tuple, Union, Dict, List

from abc import ABCMeta, abstractmethod


# taken from mmdet.models.task_modules import BaseBBoxCoder
class TtBaseBBoxCoder(metaclass=ABCMeta):
    """Base bounding box coder.

    Args:
        use_box_type (bool): Whether to warp decoded boxes with the
            box type data structure. Defaults to False.
    """

    # The size of the last of dimension of the encoded tensor.
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


def inverse_sigmoid(x, eps: float = 1e-5):
    x = ttnn.to_layout(x, layout=ttnn.TILE_LAYOUT)
    x = ttnn.clamp(x, min=0, max=1)
    x1 = ttnn.clamp(x, min=eps)
    if len(x.shape) == 3:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1], x.shape[2]], layout=ttnn.TILE_LAYOUT, device=x.device())
    elif len(x.shape) == 2:
        x_temp = ttnn.ones(shape=[x.shape[0], x.shape[1]], layout=ttnn.TILE_LAYOUT, device=x.device())
    else:
        x_temp = ttnn.ones(
            shape=[x.shape[0], x.shape[1], x.shape[2], x.shape[3]], layout=ttnn.TILE_LAYOUT, device=x.device()
        )
    x_temp = x_temp - x
    x2 = ttnn.clamp(x_temp, min=eps)
    return ttnn.log(ttnn.div(x1, x2))


def multi_scale_deformable_attn_pytorch(
    value,
    value_spatial_shapes,
    level_start_index,
    sampling_locations,
    attention_weights,
    im2col_step,
    device,
):
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
        grid = ttnn.to_memory_config(grid, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        grid = ttnn.clone(grid)
        grid = ttnn.to_torch(grid)
        grid[..., 0] = grid[..., 0] / w_l * 2 - 1
        grid[..., 1] = grid[..., 1] / h_l * 2 - 1
        grid = ttnn.from_torch(grid, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        sampling_grids.append(grid)

    # Perform sampling and attention
    output = ttnn.zeros(
        [bs, num_queries, num_heads, head_dim], device=device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )
    for lvl in range(num_levels):
        h_l, w_l = value_spatial_shapes[lvl]
        h_l = int(h_l.item())
        w_l = int(w_l.item())
        value_l = ttnn.permute(value_list[lvl], (0, 2, 3, 1))
        value_l = ttnn.reshape(value_l, (bs * num_heads, head_dim, h_l, w_l))
        grid = ttnn.permute(sampling_grids[lvl], (0, 2, 1, 3, 4))
        grid = ttnn.reshape(grid, (bs * num_heads, num_queries * num_points, 1, 2))
        value_l = ttnn.typecast(value_l, ttnn.bfloat16)
        grid = ttnn.typecast(grid, ttnn.bfloat16)
        value_l = ttnn.to_layout(value_l, layout=ttnn.ROW_MAJOR_LAYOUT)
        grid = ttnn.to_layout(grid, layout=ttnn.ROW_MAJOR_LAYOUT)
        value_l = ttnn.permute(value_l, (0, 2, 3, 1))
        sampled = ttnn.grid_sample(value_l, grid)
        sampled = ttnn.permute(sampled, (0, 3, 1, 2))
        sampled = ttnn.reshape(sampled, (bs, num_heads, head_dim, num_queries, num_points))
        sampled = ttnn.permute(sampled, (0, 3, 1, 4, 2))
        attn = attention_weights[:, :, :, lvl, :]
        attn = ttnn.unsqueeze(attn, -1)
        output += ttnn.sum((sampled * attn), -2)

    output = ttnn.reshape(output, (bs, num_queries, num_heads * head_dim))

    return output


def limit_period(val, offset: float = 0.5, period: float = np.pi):
    # limited_val = val - torch.floor(val / period + offset) * period
    tmp_val = ttnn.add(ttnn.div(val, period), offset)
    tmp_val = ttnn.floor(tmp_val)
    tmp_val = ttnn.mul(tmp_val, period)

    limited_val = ttnn.sub(val, tmp_val)
    return limited_val


def bivariate_gaussian_activation_motion_head(ip):
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    sig_x = ip[..., 2:3]
    sig_y = ip[..., 3:4]
    rho = ip[..., 4:5]
    sig_x = ttnn.exp(sig_x)
    sig_y = ttnn.exp(sig_y)
    rho = ttnn.tanh(rho)
    out = ttnn.concat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    return out


def anchor_coordinate_transform(anchors, bbox_results, with_translation_transform=True, with_rotation_transform=True):
    batch_size = len(bbox_results)
    batched_anchors = []
    transformed_anchors = ttnn.unsqueeze(anchors, 0)  # expand num agents: num_groups, num_modes, 12, 2 -> 1, ...
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw
        bbox_centers = bboxes.gravity_center
        if with_rotation_transform:
            # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, maybe we should change this.
            angle = ttnn.sub(yaw, 3.1415953)  # num_agents, 1
            rot_yaw = rot_2d(angle)  # num_agents, 2, 2
            rot_yaw = ttnn.reshape(
                rot_yaw, (rot_yaw.shape[0], 1, 1, rot_yaw.shape[1], rot_yaw.shape[2])
            )  # num_agents, 1, 1, 2, 2
            transformed_anchors = ttnn.permute(
                transformed_anchors, (0, 1, 2, 4, 3)  # "b g m t c -> b g m c t"
            )  # 1, num_groups, num_modes, 12, 2 -> 1, num_groups, num_modes, 2, 12

            rot_yaw = ttnn.squeeze(rot_yaw, 0)
            transformed_anchors = ttnn.squeeze(transformed_anchors, 0)

            input_rot_yaw_ttnn_bc = ttnn.experimental.broadcast_to(
                rot_yaw,
                ttnn.Shape(
                    (
                        transformed_anchors.shape[0],
                        transformed_anchors.shape[1],
                        transformed_anchors.shape[2],
                        rot_yaw.shape[-1],
                    )
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

            transformed_anchors = ttnn.matmul(
                input_rot_yaw_ttnn_bc, transformed_anchors
            )  # -> num_agents, num_groups, num_modes, 12, 2

            rot_yaw = ttnn.unsqueeze(rot_yaw, 0)
            transformed_anchors = ttnn.unsqueeze(transformed_anchors, 0)

            transformed_anchors = ttnn.permute(transformed_anchors, (0, 1, 2, 4, 3))  # , "b g m c t -> b g m t c")
        if with_translation_transform:
            transformed_anchors = (
                ttnn.reshape(bbox_centers[:, :2], (bbox_centers[:, :2].shape[0], 1, 1, 1, 2)) + transformed_anchors
            )
        batched_anchors.append(transformed_anchors)
    return ttnn.stack(batched_anchors, dim=0)


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    device = pos.device()
    dim_t = ttnn.arange(num_pos_feats, device=device)
    dim_t = ttnn.to_layout(dim_t, layout=ttnn.TILE_LAYOUT)
    dim_t = ttnn.div(dim_t, 2)
    dim_t = ttnn.floor(dim_t)

    dim_t = 2 * ttnn.div(dim_t, num_pos_feats)
    dim_t = ttnn.pow(temperature, dim_t)
    pos_x = ttnn.div(ttnn.unsqueeze(pos[..., 0], -1), dim_t)
    pos_y = ttnn.div(ttnn.unsqueeze(pos[..., 1], -1), dim_t)
    pos_x = ttnn.stack((ttnn.sin(pos_x[..., 0::2]), ttnn.cos(pos_x[..., 1::2])), dim=-1)
    if len(pos_x.shape) == 4:
        pos_x = ttnn.reshape(pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2] * pos_x.shape[3]))
    elif len(pos_x.shape) == 5:
        pos_x = ttnn.reshape(pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], pos_x.shape[3] * pos_x.shape[4]))
    elif len(pos_x.shape) == 6:
        pos_x = ttnn.reshape(
            pos_x, (pos_x.shape[0], pos_x.shape[1], pos_x.shape[2], pos_x.shape[3], pos_x.shape[4] * pos_x.shape[5])
        )

    pos_y = ttnn.stack((ttnn.sin(pos_y[..., 0::2]), ttnn.cos(pos_y[..., 1::2])), dim=-1)

    if len(pos_y.shape) == 4:
        pos_y = ttnn.reshape(pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2] * pos_y.shape[3]))
    elif len(pos_y.shape) == 5:
        pos_y = ttnn.reshape(pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], pos_y.shape[3] * pos_y.shape[4]))
    elif len(pos_y.shape) == 6:
        pos_y = ttnn.reshape(
            pos_y, (pos_y.shape[0], pos_y.shape[1], pos_y.shape[2], pos_y.shape[3], pos_y.shape[4] * pos_y.shape[5])
        )

    posemb = ttnn.concat((pos_y, pos_x), dim=-1)
    return posemb


def norm_points(pos, pc_range):
    x_norm = ttnn.div((ttnn.sub(pos[..., 0], pc_range[0])), (pc_range[3] - pc_range[0]))
    y_norm = ttnn.div((ttnn.sub(pos[..., 1], pc_range[1])), (pc_range[4] - pc_range[1]))
    return ttnn.stack([x_norm, y_norm], dim=-1)


def trajectory_coordinate_transform(
    trajectory, bbox_results, with_translation_transform=True, with_rotation_transform=True
):
    batch_size = len(bbox_results)
    batched_trajectories = []
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw
        bbox_centers = bboxes.gravity_center
        transformed_trajectory = trajectory[i]
        if with_rotation_transform:
            # we take negtive here, to reverse the trajectory back to ego centric coordinate
            # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, maybe we should change this.
            angle = ttnn.mul((ttnn.sub(yaw, 3.1415953)), -1)
            rot_yaw = rot_2d(angle)
            rot_yaw = ttnn.reshape(
                rot_yaw, (rot_yaw.shape[0], 1, 1, rot_yaw.shape[1], rot_yaw.shape[2])
            )  # A, 1, 1, 2, 2
            transformed_trajectory = ttnn.permute(
                transformed_trajectory, (0, 1, 2, 4, 3)  # "a g p t c -> a g p c t"
            )  # A, G, P, 12 ,2 -> # A, G, P, 2, 12

            # Squeezing because matmul doesn't support 5D
            rot_yaw = ttnn.squeeze(rot_yaw, 0)
            transformed_trajectory = ttnn.squeeze(transformed_trajectory, 0)

            input_rot_yaw_ttnn_bc = ttnn.experimental.broadcast_to(
                rot_yaw,
                ttnn.Shape(
                    (
                        transformed_trajectory.shape[0],
                        transformed_trajectory.shape[1],
                        transformed_trajectory.shape[2],
                        rot_yaw.shape[-1],
                    )
                ),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            transformed_trajectory = ttnn.matmul(input_rot_yaw_ttnn_bc, transformed_trajectory)  # -> A, G, P, 12, 2

            rot_yaw = ttnn.unsqueeze(rot_yaw, 0)
            transformed_trajectory = ttnn.unsqueeze(transformed_trajectory, 0)

            transformed_trajectory = ttnn.permute(transformed_trajectory, (0, 1, 2, 4, 3))  # "a g p c t -> a g p t c"
        if with_translation_transform:
            transformed_trajectory = (
                ttnn.reshape(bbox_centers[:, :2], (bbox_centers[:, :2].shape[0], 1, 1, 1, 2)) + transformed_trajectory
            )
        batched_trajectories.append(transformed_trajectory)
    return ttnn.stack(batched_trajectories, dim=0)


def denormalize_bbox(normalized_bboxes, pc_range, device=None):
    # rotation
    rot_sine = normalized_bboxes[..., 6:7]
    rot_cosine = normalized_bboxes[..., 7:8]

    rot_sine = ttnn.typecast(rot_sine, ttnn.bfloat16)
    rot_cosine = ttnn.typecast(rot_cosine, ttnn.bfloat16)
    if rot_sine.shape[0] == 0:
        rot = rot_sine
    else:
        rot = ttnn.atan2(rot_sine, rot_cosine)  # Does not support float32

    rot = -1 * rot
    rot = rot - np.pi / 2

    rot = limit_period(rot, period=np.pi * 2)
    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    length = normalized_bboxes[..., 2:3]
    width = normalized_bboxes[..., 3:4]
    height = normalized_bboxes[..., 5:6]

    width = ttnn.exp(width)
    length = ttnn.exp(length)
    height = ttnn.exp(height)

    if normalized_bboxes.shape[-1] > 8:
        # velocity
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]

        if ttnn.to_torch(cx).numel() == 0:
            denormalized_bboxes = ttnn.from_torch(
                torch.empty(0, 9), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )
            print("Tensor is empty, Creating empty ttnn tensor")
        else:
            denormalized_bboxes = ttnn.concat([cx, cy, cz, length, width, height, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = ttnn.concat([cx, cy, cz, length, width, height, rot], dim=-1)

    return denormalized_bboxes


def rot_2d(yaw):
    sy, cy = ttnn.sin(yaw), ttnn.cos(yaw)
    out = ttnn.permute(
        ttnn.stack([ttnn.stack([cy, ttnn.mul(sy, -1)], dim=0), ttnn.stack([sy, cy], dim=0)], dim=0), (2, 0, 1)
    )
    return out


# taken from mmdet3d/structures/bbox_3d/base_box3d.py
class TtBaseInstance3DBoxes:
    """Base class for 3D Boxes.
    Note:
        The box is bottom centered, i.e. the relative position of origin in the
        box is (0.5, 0.5, 0).
    Args:
        tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
            data with shape (N, box_dim).
        box_dim (int): Number of the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw). Defaults to 7.
        with_yaw (bool): Whether the box is with yaw rotation. If False, the
            value of yaw will be set to 0 as minmax boxes. Defaults to True.
        origin (Tuple[float]): Relative position of the box origin.
            Defaults to (0.5, 0.5, 0). This will guide the box be converted to
            (0.5, 0.5, 0) mode.
    Attributes:
        tensor (Tensor): Float matrix with shape (N, box_dim).
        box_dim (int): Integer indicating the dimension of a box. Each row is
            (x, y, z, x_size, y_size, z_size, yaw, ...).
        with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
            boxes.
    """

    YAW_AXIS: int = 0

    def __init__(
        self,
        tensor: Union[Tensor, np.ndarray, Sequence[Sequence[float]]],
        box_dim: int = 7,
        with_yaw: bool = True,
        origin: Tuple[float, float, float] = (0.5, 0.5, 0),
    ) -> None:
        # tensor = torch.Tensor(tensor, dtype=ttnn.bfloat16, device=device)
        if tensor.shape[-1] == 0:
            # Use reshape, so we don't end up creating a new tensor that does
            # not depend on the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, box_dim))
        assert len(tensor.shape) == 2 and tensor.shape[-1] == box_dim, (
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
        self.tensor = ttnn.clone(tensor)

        if origin != (0.5, 0.5, 0):
            dst = self.tensor.new_tensor((0.5, 0.5, 0))
            src = self.tensor.new_tensor(origin)
            self.tensor[:, :3] += self.tensor[:, 3:6] * (dst - src)

    def __getitem__(self, item: Union[int, slice, np.ndarray, Tensor]) -> "TtBaseInstance3DBoxes":
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def to(self, device, *args, **kwargs) -> "TtBaseInstance3DBoxes":
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
class TtLiDARInstance3DBoxes(TtBaseInstance3DBoxes):
    YAW_AXIS = 2

    @property
    def gravity_center(self):
        bottom_center = self.bottom_center
        gravity_center = ttnn.zeros(bottom_center.shape)  # no use
        gravity_center1 = bottom_center[:, :2]
        gravity_center2 = bottom_center[:, 2:3] + self.tensor[:, 5] * 0.5
        gravity_center = ttnn.concat([gravity_center1, gravity_center2], dim=1)
        return gravity_center


class Instances:
    """
    This class represents a list of instances in an image.
    It stores the attributes of instances (e.g., boxes, masks, labels, scores) as "fields".
    All fields must have the same ``__len__`` which is the number of instances.
    All other (non-field) attributes of this class are considered private:
    they must start with '_' and are not modifiable by a user.
    Some basic usage:
    1. Set/get/check a field:
       .. code-block:: python
          instances.gt_boxes = Boxes(...)
          print(instances.pred_masks)  # a tensor of shape (N, H, W)
          print('gt_masks' in instances)
    2. ``len(instances)`` returns the number of instances
    3. Indexing: ``instances[indices]`` will apply the indexing on all the fields
       and returns a new :class:`Instances`.
       Typically, ``indices`` is a integer vector of indices,
       or a binary mask of length ``num_instances``
       .. code-block:: python
          category_3_detections = instances[instances.pred_classes == 3]
          confident_detections = instances[instances.scores > 0.9]
    """

    def __init__(self, image_size: Tuple[int, int], ttnn_device=None, **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)
        self.device = ttnn_device

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
        return self._image_size

    def __setattr__(self, name, val):
        if name.startswith("_") or name in {"device", "device_id", "dtype"}:
            super().__setattr__(name, val)
        else:
            self.set(name, val)

    def __getattr__(self, name: str) -> Any:
        if name == "_fields" or name not in self._fields:
            raise AttributeError("Cannot find field '{}' in the given Instances!".format(name))
        return self._fields[name]

    def set(self, name: str, value: Any) -> None:
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = value.shape[0]
        self._fields[name] = value

    def has(self, name: str) -> bool:
        """
        Returns:
            bool: whether the field called `name` exists.
        """
        return name in self._fields

    def remove(self, name: str) -> None:
        """
        Remove the field called `name`.
        """
        del self._fields[name]

    def get(self, name: str) -> Any:
        """
        Returns the field called `name`.
        """
        return self._fields[name]

    def get_fields(self) -> Dict[str, Any]:
        """
        Returns:
            dict: a dict which maps names (str) to data of the fields
        Modifying the returned dict will modify this instance.
        """
        return self._fields

    # Tensor-like methods
    def to(self, *args: Any, **kwargs: Any) -> "Instances":
        """
        Returns:
            Instances: all fields are called with a `to(device)`, if the field has this method.
        """
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
        """
        Args:
            item: an index-like object and will be used to index all the fields.
        Returns:
            If `item` is a string, return the data in the corresponding field.
            Otherwise, returns an `Instances` where all fields are indexed by `item`.
        """
        if type(item) == int:
            if item >= len(self) or item < -len(self):
                raise IndexError("Instances index out of range!")
            else:
                item = slice(item, None, len(self))

        ret = Instances(self._image_size, ttnn_device=self.device)
        for k, v in self._fields.items():
            # if index by torch.BoolTensor
            if k == "kalman_models" and isinstance(item, torch.Tensor):
                ret_list = []
                for i, if_true in enumerate(item):
                    if if_true:
                        ret_list.append(self.kalman_models[i])
                ret.set(k, ret_list)

            else:
                if isinstance(v, ttnn.Tensor):
                    v = ttnn.to_torch(v)
                    if isinstance(item, ttnn.Tensor):
                        item = ttnn.to_torch(item).bool()
                    v = v[item]
                    v = ttnn.from_torch(v, device=self.device, layout=ttnn.TILE_LAYOUT)
                    ret.set(k, v)
                else:
                    ret.set(k, v)
        return ret

    def __len__(self) -> int:
        for v in self._fields.values():
            # use __len__ because len() has to be int and is not friendly to tracing
            return v.__len__()
        raise NotImplementedError("Empty Instances does not support __len__!")

    def __iter__(self):
        raise NotImplementedError("`Instances` object is not iterable!")

    @staticmethod
    def cat(instance_lists: List["Instances"], device=None) -> "Instances":
        """
        Args:
            instance_lists (list[Instances])
        Returns:
            Instances
        """
        assert all(isinstance(i, Instances) for i in instance_lists)
        assert len(instance_lists) > 0
        if len(instance_lists) == 1:
            return instance_lists[0]

        image_size = instance_lists[0].image_size
        for i in instance_lists[1:]:
            assert i.image_size == image_size
        ret = Instances(image_size, ttnn_device=device)
        for k in instance_lists[0]._fields.keys():
            values = [i.get(k) for i in instance_lists]
            v0 = values[0]
            for i in range(len(values)):
                values[i] = ttnn.to_layout(values[i], layout=ttnn.TILE_LAYOUT)
                # values[i] = ttnn.to_torch(values[i])
            if isinstance(v0, torch.Tensor):
                values = torch.cat(values, dim=0)
            elif isinstance(v0, list):
                print("Here 2")
                # values = list(itertools.chain(*values))
            elif hasattr(type(v0), "cat"):
                print("Here 3")
                # values = type(v0).cat(values)
            else:
                if values[1].shape[0] > 0:
                    values = ttnn.concat(values, dim=1)
                else:
                    values = values[0]

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


def bivariate_gaussian_activation_plan_head(ip):
    mu_x = ip[..., 0:1]
    mu_y = ip[..., 1:2]
    # sig_x = ip[..., 2:] # empty tensors
    # sig_y = ip[..., 3:]
    # rho = ip[..., 4:]
    # sig_x = ttnn.exp(sig_x)
    # sig_y = ttnn.exp(sig_y)
    # rho = ttnn.tanh(rho)
    # out = ttnn.concat([mu_x, mu_y, sig_x, sig_y, rho], dim=-1)
    out = ttnn.concat([mu_x, mu_y], dim=-1)
    return out

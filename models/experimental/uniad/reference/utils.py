# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import math
from einops import rearrange
import torch
import itertools
import torch.nn.functional as F
from functools import partial
import numpy as np
from typing import Any, Dict, List, Tuple, Union

from torch import Tensor
from typing import Union

from typing import Any, Dict, List, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt
from casadi import DM, Opti, sumsqr, vertcat, exp

Pose = Tuple[float, float, float]  # (x, y, yaw)


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
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
    """Limit the value into a period for periodic function.

    Args:
        val (np.ndarray or Tensor): The value to be converted.
        offset (float): Offset to set the value range. Defaults to 0.5.
        period (float): Period of the value. Defaults to np.pi.

    Returns:
        np.ndarray or Tensor: Value in the range of
        [-offset * period, (1-offset) * period].
    """
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


def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))


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

    def __init__(self, image_size: Tuple[int, int], **kwargs: Any):
        """
        Args:
            image_size (height, width): the spatial size of the image.
            kwargs: fields to add to this `Instances`.
        """
        self._image_size = image_size
        self._fields: Dict[str, Any] = {}
        for k, v in kwargs.items():
            self.set(k, v)

    @property
    def image_size(self) -> Tuple[int, int]:
        """
        Returns:
            tuple: height, width
        """
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
        """
        Set the field named `name` to `value`.
        The length of `value` must be the number of instances,
        and must agree with other existing fields in this object.
        """
        data_len = len(value)
        if len(self._fields):
            assert len(self) == data_len, "Adding a field of length {} to a Instances of length {}".format(
                data_len, len(self)
            )
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

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
            # print(k, type(item), 'getitem', item.type(), item.dtype)
            # if index by torch.BoolTensor
            if k == "kalman_models" and isinstance(item, torch.Tensor):
                ret_list = []
                for i, if_true in enumerate(item):
                    if if_true:
                        ret_list.append(self.kalman_models[i])
                ret.set(k, ret_list)

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
    def cat(instance_lists: List["Instances"]) -> "Instances":
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
    def __init__(self, trajectory_len: int, dt: float, sigma, alpha_collision, obj_pixel_pos):
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.current_index = 0
        self.sigma = sigma
        self.alpha_collision = alpha_collision
        self.obj_pixel_pos = obj_pixel_pos
        # Use a array of dts to make it compatible to situations with varying dts across different time steps.
        self._dts: npt.NDArray[np.float32] = np.asarray([[dt] * trajectory_len])
        self._init_optimization()

    def _init_optimization(self) -> None:
        self.nx = 2  # state dim

        self._optimizer = Opti()  # Optimization problem
        self._create_decision_variables()
        self._create_parameters()
        self._set_objective()

        # Set default solver options (quiet)
        self._optimizer.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})

    def set_reference_trajectory(self, reference_trajectory: Sequence[Pose]) -> None:
        self._optimizer.set_value(self.ref_traj, DM(reference_trajectory).T)
        self._set_initial_guess(reference_trajectory)

    def set_solver_optimizerons(self, options: Dict[str, Any]) -> None:
        self._optimizer.solver("ipopt", options)

    def solve(self):
        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
        # State trajectory (x, y)
        self.state = self._optimizer.variable(self.nx, self.trajectory_len)
        self.position_x = self.state[0, :]
        self.position_y = self.state[1, :]

    def _create_parameters(self) -> None:
        self.ref_traj = self._optimizer.parameter(2, self.trajectory_len)  # (x, y)

    def _set_objective(self) -> None:
        # Follow reference, minimize control rates and absolute inputs
        alpha_xy = 1.0
        cost_stage = alpha_xy * sumsqr(self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y))

        alpha_collision = self.alpha_collision

        cost_collision = 0
        normalizer = 1 / (2.507 * self.sigma)
        # TODO: vectorize this
        for t in range(len(self.obj_pixel_pos)):
            x, y = self.position_x[t], self.position_y[t]
            for i in range(len(self.obj_pixel_pos[t])):
                col_x, col_y = self.obj_pixel_pos[t][i]
                cost_collision += (
                    alpha_collision * normalizer * exp(-((x - col_x) ** 2 + (y - col_y) ** 2) / 2 / self.sigma**2)
                )
        self._optimizer.minimize(cost_stage + cost_collision)

    def _set_initial_guess(self, reference_trajectory: Sequence[Pose]) -> None:
        """Set a warm-start for the solver based on the reference trajectory."""
        # Initialize state guess based on reference
        self._optimizer.set_initial(self.state[:2, :], DM(reference_trajectory).T)  # (x, y, yaw)


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
    """
    Compute 2D rotation matrix for a given yaw angle tensor.

    Args:
        yaw (torch.Tensor): Input yaw angle tensor.

    Returns:
        torch.Tensor: 2D rotation matrix tensor.
    """
    sy, cy = torch.sin(yaw), torch.cos(yaw)
    out = torch.stack([torch.stack([cy, -sy]), torch.stack([sy, cy])]).permute([2, 0, 1])
    return out


def norm_points(pos, pc_range):
    """
    Normalize the end points of a given position tensor.

    Args:
        pos (torch.Tensor): Input position tensor.
        pc_range (List[float]): Point cloud range.

    Returns:
        torch.Tensor: Normalized end points tensor.
    """
    x_norm = (pos[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    y_norm = (pos[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    return torch.stack([x_norm, y_norm], dim=-1)


def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    """
    Convert 2D position into positional embeddings.

    Args:
        pos (torch.Tensor): Input 2D position tensor.
        num_pos_feats (int, optional): Number of positional features. Default is 128.
        temperature (int, optional): Temperature factor for positional embeddings. Default is 10000.

    Returns:
        torch.Tensor: Positional embeddings tensor.
    """
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
    """
    Transform trajectory coordinates with respect to detected bounding boxes in the batch.
    Args:
        trajectory (torch.Tensor): predicted trajectory.
        bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
        with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
        with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the transformed trajectory coordinates.
    """
    batch_size = len(bbox_results)
    batched_trajectories = []
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(trajectory.device)
        bbox_centers = bboxes.gravity_center.to(trajectory.device)
        transformed_trajectory = trajectory[i, ...]
        if with_rotation_transform:
            # we take negtive here, to reverse the trajectory back to ego centric coordinate
            # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, maybe we should change this.
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
    """
    Transform anchor coordinates with respect to detected bounding boxes in the batch.

    Args:
        anchors (torch.Tensor): A tensor containing the k-means anchor values.
        bbox_results (List[Tuple[torch.Tensor]]): A list of tuples containing the bounding box results for each image in the batch.
        with_translate (bool, optional): Whether to perform translation transformation. Defaults to True.
        with_rot (bool, optional): Whether to perform rotation transformation. Defaults to True.

    Returns:
        torch.Tensor: A tensor containing the transformed anchor coordinates.
    """
    batch_size = len(bbox_results)
    batched_anchors = []
    transformed_anchors = anchors[None, ...]  # expand num agents: num_groups, num_modes, 12, 2 -> 1, ...
    for i in range(batch_size):
        bboxes, scores, labels, bbox_index, mask = bbox_results[i]
        yaw = bboxes.yaw.to(transformed_anchors.device)
        bbox_centers = bboxes.gravity_center.to(transformed_anchors.device)
        if with_rotation_transform:
            # TODO(box3d): we have changed yaw to mmdet3d 1.0.0rc6 format, maybe we should change this.
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

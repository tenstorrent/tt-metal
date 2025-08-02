# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Tuple, Union, Sequence
import torch
import ttnn
from torch import Tensor
import numpy as np


def bivariate_gaussian_activation(ip):
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
        print("image_size in init", image_size)
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

        ret = Instances(self._image_size)
        for k, v in self._fields.items():
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
                print("values for ttnn concat", len(values), values[0].shape, values[1].shape)
                if len(values[0].shape) == 1:
                    values[0] = ttnn.add(values[0], 0, dtype=ttnn.bfloat16)
                    values[1] = ttnn.add(values[1], 0, dtype=ttnn.bfloat16)
                    values[0] = ttnn.unsqueeze(values[0], 0)
                    values[1] = ttnn.unsqueeze(values[1], 0)
                    print(
                        values[0].shape,
                        values[1].shape,
                        values[0].get_dtype(),
                        values[1].get_dtype(),
                    )
                    values = ttnn.concat(values, dim=1)
                else:
                    values = ttnn.concat(values, dim=0)
                print("After concat ttnn", values.shape)

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


# taken from mmdet3d/structures/bbox_3d/base_box3d.py
class TtBaseInstance3DBoxes:
    # """Base class for 3D Boxes.
    # Note:
    #     The box is bottom centered, i.e. the relative position of origin in the
    #     box is (0.5, 0.5, 0).
    # Args:
    #     tensor (Tensor or np.ndarray or Sequence[Sequence[float]]): The boxes
    #         data with shape (N, box_dim).
    #     box_dim (int): Number of the dimension of a box. Each row is
    #         (x, y, z, x_size, y_size, z_size, yaw). Defaults to 7.
    #     with_yaw (bool): Whether the box is with yaw rotation. If False, the
    #         value of yaw will be set to 0 as minmax boxes. Defaults to True.
    #     origin (Tuple[float]): Relative position of the box origin.
    #         Defaults to (0.5, 0.5, 0). This will guide the box be converted to
    #         (0.5, 0.5, 0) mode.
    # Attributes:
    #     tensor (Tensor): Float matrix with shape (N, box_dim).
    #     box_dim (int): Integer indicating the dimension of a box. Each row is
    #         (x, y, z, x_size, y_size, z_size, yaw, ...).
    #     with_yaw (bool): If True, the value of yaw will be set to 0 as minmax
    #         boxes.
    # """

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
        """
        Args:
            item (int or slice or np.ndarray or Tensor): Index of boxes.
        Note:
            The following usage are allowed:
            1. `new_boxes = boxes[3]`: Return a `Boxes` that contains only one
               box.
            2. `new_boxes = boxes[2:10]`: Return a slice of boxes.
            3. `new_boxes = boxes[vector]`: Where vector is a
               torch.BoolTensor with `length = len(boxes)`. Nonzero elements in
               the vector will be selected.
            Note that the returned Boxes might share storage with this Boxes,
            subject to PyTorch's indexing semantics.
        Returns:
            :obj:`BaseInstance3DBoxes`: A new object of
            :class:`BaseInstance3DBoxes` after indexing.
        """
        original_type = type(self)
        if isinstance(item, int):
            return original_type(self.tensor[item].view(1, -1), box_dim=self.box_dim, with_yaw=self.with_yaw)
        b = self.tensor[item]
        assert b.dim() == 2, f"Indexing on Boxes with {item} failed to return a matrix!"
        return original_type(b, box_dim=self.box_dim, with_yaw=self.with_yaw)

    def to(self, device: Union[str, torch.device], *args, **kwargs) -> "TtBaseInstance3DBoxes":
        """Convert current boxes to a specific device.
        Args:
            device (str or :obj:`torch.device`): The name of the device.
        Returns:
            :obj:`BaseInstance3DBoxes`: A new boxes object on the specific
            device.
        """
        original_type = type(self)
        return original_type(self.tensor.to(device, *args, **kwargs), box_dim=self.box_dim, with_yaw=self.with_yaw)

    def gravity_center(self):
        pass

    @property
    def bottom_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        return self.tensor[:, :3]

    @property
    def yaw(self):
        """torch.Tensor: A vector with yaw of each box in shape (N, )."""
        return self.tensor[:, 6]


# taken from mmdet3d/structures/bbox_3d/lidar_box3d.py, removed all the functions from this class as they are not  invoked
class TtLiDARInstance3DBoxes(TtBaseInstance3DBoxes):
    YAW_AXIS = 2

    @property
    def gravity_center(self):
        """torch.Tensor: A tensor with center of each box in shape (N, 3)."""
        bottom_center = self.bottom_center
        gravity_center = ttnn.zeros(bottom_center.shape)  # no use
        gravity_center1 = bottom_center[:, :2]
        gravity_center2 = bottom_center[:, 2:3] + self.tensor[:, 5] * 0.5
        gravity_center = ttnn.concat([gravity_center1, gravity_center2], dim=1)
        return gravity_center

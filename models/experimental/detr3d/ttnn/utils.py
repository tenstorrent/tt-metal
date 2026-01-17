# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import numpy as np
from models.common.lightweightmodule import LightweightModule


class TtnnConv1D(LightweightModule):
    def __init__(
        self,
        conv,
        parameters,
        device,
        activation_dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        shard_layout=None,
        fp32_accum=False,
        packer_l1_acc=False,
        activation=None,
        deallocate_activation=False,
        math_fidelity=ttnn.MathFidelity.LoFi,
        return_dims=False,
        reshape_output=False,
        memory_config=None,
    ):
        super().__init__()
        self.conv = conv
        self.device = device
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = conv.kernel_size[0]
        self.padding = conv.padding[0]
        self.stride = conv.stride[0]
        self.groups = conv.groups
        self.conv_config = ttnn.Conv1dConfig(
            weights_dtype=weights_dtype,
            shard_layout=shard_layout,
            deallocate_activation=deallocate_activation,
            activation=activation,
        )
        self.compute_config = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_accum,
            packer_l1_acc=packer_l1_acc,
        )
        self.weight = ttnn.from_device(parameters.weight)
        self.bias = None
        if "bias" in parameters and parameters["bias"] is not None:
            bias = ttnn.from_device(parameters.bias)
            self.bias = bias
        self.activation_dtype = activation_dtype
        self.return_dims = return_dims
        self.reshape_output = reshape_output
        self.memory_config = memory_config

    def forward(self, x, shape=None):
        if shape is not None:
            batch_size = shape[0]
            input_length = shape[1]
        else:
            batch_size = x.shape[0]
            input_length = x.shape[1]

        [tt_output_tensor_on_device, out_length, [weights_device, bias_device]] = ttnn.conv1d(
            input_tensor=x,
            weight_tensor=self.weight,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            device=self.device,
            bias_tensor=self.bias,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            batch_size=batch_size,
            input_length=input_length,
            conv_config=self.conv_config,
            compute_config=self.compute_config,
            groups=self.groups,
            return_output_dim=True,
            return_weights_and_bias=True,
            memory_config=self.memory_config,
            dtype=self.activation_dtype,
        )
        shape = (batch_size, out_length, tt_output_tensor_on_device.shape[-1])
        if self.reshape_output:
            tt_output_tensor_on_device = ttnn.reshape(tt_output_tensor_on_device, shape)
        if self.return_dims:
            return tt_output_tensor_on_device, shape
        return tt_output_tensor_on_device


def shift_scale_points_ttnn(pred_xyz, src_range, device=None):
    """
    pred_xyz: B x N x 3
    src_range: [[B x 3], [B x 3]] - min and max XYZ coords
    dst_range: [[B x 3], [B x 3]] - min and max XYZ coords
    """

    dst_range = [
        ttnn.zeros((src_range[0].shape[0], 3), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
        ttnn.ones((src_range[0].shape[0], 3), dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT),
    ]

    assert src_range[0].shape[0] == pred_xyz.shape[0]
    assert dst_range[0].shape[0] == pred_xyz.shape[0]
    assert src_range[0].shape[-1] == pred_xyz.shape[-1]
    assert src_range[0].shape == src_range[1].shape
    assert dst_range[0].shape == dst_range[1].shape
    assert src_range[0].shape == dst_range[1].shape

    src_range[0] = ttnn.unsqueeze(src_range[0], 1)
    src_range[1] = ttnn.unsqueeze(src_range[1], 1)
    dst_range[0] = ttnn.unsqueeze(dst_range[0], 1)
    dst_range[1] = ttnn.unsqueeze(dst_range[1], 1)

    src_diff = src_range[1] - src_range[0]
    dst_diff = dst_range[1] - dst_range[0]
    prop_xyz = pred_xyz - src_range[0]
    prop_xyz = prop_xyz * dst_diff
    prop_xyz = ttnn.div(
        prop_xyz,
        src_diff,
        accurate_mode=True,
        rounding_mode=None,
    )
    prop_xyz = prop_xyz + dst_range[0]

    ttnn.deallocate(src_diff)
    ttnn.deallocate(dst_diff)
    ttnn.deallocate(dst_range[0])
    ttnn.deallocate(dst_range[1])

    return prop_xyz


def scale_points(pred_xyz, mult_factor):
    if len(pred_xyz.shape) == 4:
        mult_factor = ttnn.unsqueeze(mult_factor, 1)
    mult_factor = ttnn.unsqueeze(mult_factor, 1)
    scaled_xyz = pred_xyz * mult_factor
    return scaled_xyz


class TtnnBoxProcessor(object):
    """
    Class to convert 3DETR MLP head outputs into bounding boxes
    """

    def __init__(self, dataset_config, device):
        self.dataset_config = dataset_config
        self.device = device

    def compute_predicted_center(self, center_offset, query_xyz, point_cloud_dims):
        center_unnormalized = query_xyz + center_offset
        center_normalized = shift_scale_points_ttnn(center_unnormalized, src_range=point_cloud_dims)
        return center_normalized, center_unnormalized

    def compute_predicted_size(self, size_normalized, point_cloud_dims):
        scene_scale = point_cloud_dims[1] - point_cloud_dims[0]
        scene_scale = ttnn.clamp(scene_scale, min=1e-1)
        size_unnormalized = scale_points(size_normalized, mult_factor=scene_scale)
        ttnn.deallocate(scene_scale)
        return size_unnormalized

    def compute_predicted_angle(self, angle_logits, angle_residual):
        if angle_logits.shape[-1] == 1:
            # special case for datasets with no rotation angle
            # we still use the predictions so that model outputs are used
            # in the backwards pass (DDP may complain otherwise)
            angle = angle_logits * 0 + angle_residual * 0
            angle = ttnn.squeeze(angle, -1)
            angle = ttnn.clamp(angle, min=0)
        else:
            angle_per_cls = (2 * np.pi) / self.dataset_config.num_angle_bin
            pred_angle_class = ttnn.argmax(angle_logits, dim=-1)
            angle_center = angle_per_cls * pred_angle_class
            pred_angle_class = ttnn.unsqueeze(pred_angle_class, -1)
            angle_residual_gathered = ttnn.gather(angle_residual, 2, pred_angle_class)
            angle = angle_center + ttnn.squeeze(angle_residual_gathered, -1)
            mask = angle > np.pi
            angle[mask] = angle[mask] - (2 * np.pi)
        return angle

    def compute_objectness_and_cls_prob(self, cls_logits):
        assert cls_logits.shape[-1] == self.dataset_config.num_semcls + 1
        cls_prob = ttnn.softmax(cls_logits, dim=-1)
        objectness_prob = 1 - cls_prob[..., -1]
        return cls_prob[..., :-1], objectness_prob

    def box_parametrization_to_corners(self, box_center_unnorm, box_size_unnorm, box_angle):
        torch_box_center_unnorm = ttnn.to_torch(box_center_unnorm, dtype=torch.float32)
        torch_box_size_unnorm = ttnn.to_torch(box_size_unnorm, dtype=torch.float32)
        torch_box_angle = ttnn.to_torch(box_angle, dtype=torch.float32)
        return self.dataset_config.box_parametrization_to_corners(
            torch_box_center_unnorm, torch_box_size_unnorm, torch_box_angle
        )

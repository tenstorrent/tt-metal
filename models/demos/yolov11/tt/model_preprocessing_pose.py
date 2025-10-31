# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Model preprocessing for YOLO11 Pose Estimation

This module handles weight loading and parameter setup for the
TTNN YoloV11 Pose model.
"""

import torch
import torch.nn as nn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.yolov11.reference.yolov11 import Conv  # Conv is same for both
from models.demos.yolov11.reference.yolov11_pose_correct import YoloV11Pose
from models.demos.yolov11.tt.common import get_mesh_mappers
from models.demos.yolov11.tt.model_preprocessing import make_anchors


def custom_preprocessor_pose(model, name, mesh_mapper=None):
    """
    Custom preprocessor for pose model
    Handles Conv layers and DWConv layers
    """
    parameters = {}

    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    if isinstance(model, Conv):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
        bias = bias.reshape((1, 1, 1, -1))
        if model.split_weights:
            chunk_size = bias.shape[-1] // 2
            parameters["a"] = {}
            parameters["a"]["conv"] = {}
            parameters["a"]["conv"]["weight"] = ttnn.from_torch(
                weight[:chunk_size, :, :, :], dtype=ttnn.float32, mesh_mapper=mesh_mapper
            )
            parameters["a"]["conv"]["bias"] = ttnn.from_torch(
                bias[:, :, :, :chunk_size], dtype=ttnn.float32, mesh_mapper=mesh_mapper
            )
            parameters["b"] = {}
            parameters["b"]["conv"] = {}
            parameters["b"]["conv"]["weight"] = ttnn.from_torch(
                weight[chunk_size:, :, :, :], dtype=ttnn.float32, mesh_mapper=mesh_mapper
            )
            parameters["b"]["conv"]["bias"] = ttnn.from_torch(
                bias[:, :, :, chunk_size:], dtype=ttnn.float32, mesh_mapper=mesh_mapper
            )
        else:
            parameters["conv"] = {}
            parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
            parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    # Handle DWConv (Depthwise Convolution) - same as Conv but with groups=in_channels
    # The preprocessing is the same as regular Conv
    from models.demos.yolov11.reference.yolov11_pose_correct import DWConv

    if isinstance(model, DWConv):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv"] = {}
        parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    return parameters


def create_custom_mesh_preprocessor_pose(mesh_mapper=None):
    """Create custom mesh preprocessor for pose model"""

    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor_pose(model, name, mesh_mapper)

    return custom_mesh_preprocessor


def create_yolov11_pose_model_parameters(model: YoloV11Pose, input_tensor: torch.Tensor, device):
    """
    Create model parameters for YOLO11 Pose

    Args:
        model: YoloV11Pose model with pretrained weights
        input_tensor: Sample input tensor
        device: TT device

    Returns:
        Preprocessed parameters for TTNN model
    """
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor_pose(weights_mesh_mapper),
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    parameters["model_args"] = model

    # Create anchors for pose detection
    feats = [
        input_tensor.shape[3] // 8,  # 80 for 640x640 input
        input_tensor.shape[3] // 16,  # 40
        input_tensor.shape[3] // 32,  # 20
    ]
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides, mesh_mapper=weights_mesh_mapper)

    if "model" in parameters:
        parameters.model[23]["anchors"] = anchors
        parameters.model[23]["strides"] = strides

    return parameters


def create_yolov11_pose_model_parameters_head(
    model: YoloV11Pose, input_tensor_1: torch.Tensor, input_tensor_2, input_tensor_3, device
):
    """
    Create model parameters specifically for the pose head (layer 23)

    Args:
        model: YoloV11Pose model
        input_tensor_1, 2, 3: Feature map inputs to the pose head
        device: TT device

    Returns:
        Preprocessed parameters for pose head
    """
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor_pose(weights_mesh_mapper),
        device=device,
    )

    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensor_1, input_tensor_2, input_tensor_3), device=None
    )

    # Anchors for 640x640 input
    feats = [80, 40, 20]  # Feature map sizes
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides, mesh_mapper=weights_mesh_mapper)

    parameters["anchors"] = anchors
    parameters["strides"] = strides
    parameters["model"] = model

    return parameters

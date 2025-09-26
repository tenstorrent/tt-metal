# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.yolov11m.reference.yolov11 import Conv, YoloV11
from models.demos.yolov11m.tt.common import get_mesh_mappers


def create_yolov11_input_tensors(
    device, batch=1, input_channels=3, input_height=320, input_width=320, is_sub_module=True
):
    num_devices = device.get_num_devices()
    inputs_mesh_mapper, _, _ = get_mesh_mappers(device)
    torch_input_tensor = torch.randn(batch * device.get_num_devices(), input_channels, input_height, input_width)
    if is_sub_module:
        ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        ttnn_input_tensor = torch.reshape(
            ttnn_input_tensor,
            (
                1,
                1,
                ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
                ttnn_input_tensor.shape[-1],
            ),
        )
        ttnn_input_tensor = ttnn.from_torch(
            ttnn_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, mesh_mapper=inputs_mesh_mapper
        )
    else:
        n, c, h, w = torch_input_tensor.shape
        if c == 3:
            c = 16
        n = n // num_devices if n // num_devices != 0 else n
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        # Debug: Check diversity before scaling
        pre_scale_flat = torch_input_tensor.flatten()
        pre_scale_unique = torch.unique(pre_scale_flat)
        print(f"🔍 [SCALING DEBUG] BEFORE scaling: {len(pre_scale_unique)} unique values")
        print(f"    Range: [{pre_scale_flat.min()}, {pre_scale_flat.max()}], Mean: {pre_scale_flat.mean()}")
        
        # CRITICAL FIX: Scale tensor to better utilize bfloat16's range
        # Scale by 8x to move from [-5, +5] to [-40, +40] range
        # This utilizes more of bfloat16's 65K capacity instead of just 3K
        scale_factor = 8.0
        torch_input_tensor_scaled = torch_input_tensor * scale_factor
        
        # Debug: Check diversity after scaling
        post_scale_flat = torch_input_tensor_scaled.flatten()
        post_scale_unique = torch.unique(post_scale_flat)
        print(f"🔍 [SCALING DEBUG] AFTER scaling by {scale_factor}x: {len(post_scale_unique)} unique values")
        print(f"    Range: [{post_scale_flat.min()}, {post_scale_flat.max()}], Mean: {post_scale_flat.mean()}")
        
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor_scaled,  # Use scaled tensor for better bfloat16 utilization
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
            mesh_mapper=inputs_mesh_mapper,
        )
        
        # Debug: Check diversity after ttnn.from_torch conversion (ELSE branch)
        post_conversion_debug = ttnn.to_torch(ttnn_input_tensor)
        post_conversion_flat = post_conversion_debug.flatten()
        post_conversion_unique = torch.unique(post_conversion_flat)
        print(f"🔍 [PREPROCESSING DEBUG - ELSE] AFTER ttnn.from_torch: {len(post_conversion_unique)} unique values out of {len(post_conversion_flat)} total")
        print(f"    Range: [{post_conversion_flat.min()}, {post_conversion_flat.max()}], Mean: {post_conversion_flat.mean()}")
        print(f"    Dtype: {post_conversion_debug.dtype}")
        print(f"🔍 [PREPROCESSING DEBUG - ELSE] DIVERSITY LOSS: {len(torch_input_tensor)} → {len(post_conversion_unique)} ({100*(len(pre_conversion_unique)-len(post_conversion_unique))/len(pre_conversion_unique):.2f}% loss)")
    return torch_input_tensor, ttnn_input_tensor


def make_anchors(device, feats, strides, grid_cell_offset=0.5, mesh_mapper=None):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i], feats[i]
        sx = torch.arange(end=w) + grid_cell_offset
        sy = torch.arange(end=h) + grid_cell_offset
        sy, sx = torch.meshgrid(sy, sx)
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride))

    a = torch.cat(anchor_points).transpose(0, 1).unsqueeze(0)
    b = torch.cat(stride_tensor).transpose(0, 1)

    return (
        ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper),
        ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=mesh_mapper),
    )


def custom_preprocessor(model, name, mesh_mapper=None):
    parameters = {}
    if isinstance(model, nn.Conv2d):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        if model.bias is not None:
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    if isinstance(model, Conv):
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
        parameters["conv"] = {}
        parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

    return parameters


def create_yolov11_model_parameters(model: YoloV11, input_tensor: torch.Tensor, device):
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)

    parameters["model_args"] = model

    feats = [
        input_tensor.shape[3] // 8,
        input_tensor.shape[3] // 16,
        input_tensor.shape[3] // 32,
    ]
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides, mesh_mapper=weights_mesh_mapper)

    if "model" in parameters:
        parameters.model[23]["anchors"] = anchors
        parameters.model[23]["strides"] = strides

    return parameters


def create_yolov11_model_parameters_detect(
    model: YoloV11, input_tensor_1: torch.Tensor, input_tensor_2, input_tensor_3, device
):
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)

    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensor_1, input_tensor_2, input_tensor_3), device=None
    )

    feats = [28, 14, 7]
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides, mesh_mapper=weights_mesh_mapper)

    parameters["anchors"] = anchors
    parameters["strides"] = strides

    parameters["model"] = model

    return parameters


def create_custom_mesh_preprocessor(mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(model, name, mesh_mapper)

    return custom_mesh_preprocessor

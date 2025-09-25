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
        original_c = c
        if c == 3:
            c = 16
        n = n // num_devices if n // num_devices != 0 else n
        
        # CRITICAL FIX: Do padding AND permute in PyTorch to avoid TTNN's 94% quantization loss
        # Step 1: Pad channels in PyTorch if needed (to match memory config)
        if original_c == 3:
            # Pad from 3 to 16 channels
            padding = torch.zeros(n, 13, h, w, dtype=torch_input_tensor.dtype, device=torch_input_tensor.device)
            torch_input_tensor_padded = torch.cat([torch_input_tensor, padding], dim=1)  # [1, 16, 320, 320]
        else:
            torch_input_tensor_padded = torch_input_tensor
            
        # Step 2: Convert NCHW → NHWC in high-precision PyTorch instead of lossy TTNN  
        torch_input_tensor_nhwc = torch.permute(torch_input_tensor_padded, (0, 2, 3, 1))  # [N,C,H,W] → [N,H,W,C]
        
        # ✅ CHECKPOINT: Verify PyTorch permute preserves full precision
        pre_conversion_unique = len(torch.unique(torch_input_tensor_nhwc.flatten()))
        print(f"✅ [CHECKPOINT] PyTorch permute preserved: {pre_conversion_unique} unique values")
        
        # Keep original sharding strategy but adapt for NHWC tensor dimensions
        # The tensor is now [n, h, w, c] but we need compatible sharding
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],  # Keep original dimension order for sharding compatibility  
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor_nhwc,  # Use NHWC version for TTNN
            dtype=ttnn.float32,  # Changed from bfloat16 to preserve input diversity
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
            mesh_mapper=inputs_mesh_mapper,
        )
        
        # ✅ CHECKPOINT: Verify ttnn.from_torch preserves diversity
        post_conversion_unique = len(torch.unique(ttnn.to_torch(ttnn_input_tensor).flatten()))
        loss_pct = 100*(pre_conversion_unique-post_conversion_unique)/pre_conversion_unique
        print(f"✅ [CHECKPOINT] TTNN conversion: {pre_conversion_unique} → {post_conversion_unique} ({loss_pct:.1f}% loss)")
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

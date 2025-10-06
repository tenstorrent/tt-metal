# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.yolov11m.reference.yolov11 import Conv, YoloV11
from models.demos.yolov11m.tt.common import get_mesh_mappers, analyze_tensor_precision


def create_yolov11_input_tensors(
    device, batch=1, input_channels=3, input_height=320, input_width=320, is_sub_module=True, input_tensor=None
):
    num_devices = device.get_num_devices()
    inputs_mesh_mapper, _, _ = get_mesh_mappers(device)
    
    # 🔍 DEBUG: Check input tensor parameter
    print(f"🔍 [DEBUG] input_tensor is None: {input_tensor is None}")
    if input_tensor is not None:
        print(f"🔍 [DEBUG] input_tensor shape: {input_tensor.shape}")
        print(f"🔍 [DEBUG] input_tensor dtype: {input_tensor.dtype}")
    
    # Use provided input tensor or generate random data as fallback
    if input_tensor is not None:
        # 🔍 PRECISION TRACKING: Analyze original input tensor
        analyze_tensor_precision(input_tensor, "PREPROCESSING", "ORIGINAL_INPUT")
        
        torch_input_tensor = input_tensor
        # Ensure the tensor has the right batch size for multi-device
        if torch_input_tensor.shape[0] != batch * device.get_num_devices():
            # Repeat the tensor if needed for multi-device setup
            torch_input_tensor = torch_input_tensor.repeat(device.get_num_devices(), 1, 1, 1)
            analyze_tensor_precision(torch_input_tensor, "PREPROCESSING", "AFTER_REPEAT")
    else:
        print(f"🔍 [DEBUG] input_tensor is None, using random fallback")
        torch_input_tensor = torch.randn(batch * device.get_num_devices(), input_channels, input_height, input_width)
        analyze_tensor_precision(torch_input_tensor, "PREPROCESSING", "RANDOM_FALLBACK")
    if is_sub_module:
        # 🔍 PRECISION TRACKING: Sub-module path (not used in main model)
        analyze_tensor_precision(torch_input_tensor, "PREPROCESSING", "BEFORE_SUB_MODULE_PERMUTE")
        
        ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
        analyze_tensor_precision(ttnn_input_tensor, "PREPROCESSING", "AFTER_SUB_MODULE_PERMUTE")
        
        ttnn_input_tensor = torch.reshape(
            ttnn_input_tensor,
            (
                1,
                1,
                ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
                ttnn_input_tensor.shape[-1],
            ),
        )
        analyze_tensor_precision(ttnn_input_tensor, "PREPROCESSING", "AFTER_SUB_MODULE_RESHAPE")
        
        ttnn_input_tensor = ttnn.from_torch(
            ttnn_input_tensor, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, mesh_mapper=inputs_mesh_mapper
        )
        # Convert back to torch for analysis
        ttnn_torch_converted = ttnn.to_torch(ttnn_input_tensor)
        analyze_tensor_precision(ttnn_torch_converted, "PREPROCESSING", "AFTER_SUB_MODULE_TTNN_CONVERSION_bfloat8_b")
    else:
        # 🔍 PRECISION TRACKING: Main module path (used by actual model)
        analyze_tensor_precision(torch_input_tensor, "PREPROCESSING", "BEFORE_MAIN_MODULE_PROCESSING")
        
        n, c, h, w = torch_input_tensor.shape
        print(f"🔍 [PREPROCESSING] Original tensor shape: {torch_input_tensor.shape}")
        print(f"🔍 [PREPROCESSING] Original channels: {c}")
        
        if c == 3:
            c = 16
            print(f"🔍 [PREPROCESSING] Channels will be padded from 3 to 16 during model processing")
        n = n // num_devices if n // num_devices != 0 else n
        
        print(f"🔍 [PREPROCESSING] Memory config shape: [{n}, {c}, {h}, {w}]")
        print(f"🔍 [PREPROCESSING] Converting to TTNN with dtype=bfloat16")
        
        input_mem_config = ttnn.create_sharded_memory_config(
            [n, c, h, w],
            ttnn.CoreGrid(x=8, y=8),
            ttnn.ShardStrategy.HEIGHT,
        )
        
        ttnn_input_tensor = ttnn.from_torch(
            torch_input_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=input_mem_config,
            mesh_mapper=inputs_mesh_mapper,
        )
        
        # 🔍 PRECISION TRACKING: Analyze after TTNN conversion
        ttnn_torch_converted = ttnn.to_torch(ttnn_input_tensor)
        analyze_tensor_precision(ttnn_torch_converted, "PREPROCESSING", "AFTER_MAIN_MODULE_TTNN_CONVERSION_bfloat16")
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
        # 🔍 PRECISION TRACKING: Analyze original PyTorch weights before TTNN conversion
        analyze_tensor_precision(model.weight, f"WEIGHT_CONVERSION-{name}", "PYTORCH_WEIGHT")
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        # 🔍 PRECISION TRACKING: Analyze converted TTNN weights
        analyze_tensor_precision(parameters["weight"], f"WEIGHT_CONVERSION-{name}", "TTNN_WEIGHT_float32")
        
        if model.bias is not None:
            # 🔍 PRECISION TRACKING: Analyze original PyTorch bias before TTNN conversion
            analyze_tensor_precision(model.bias, f"BIAS_CONVERSION-{name}", "PYTORCH_BIAS")
            bias = model.bias.reshape((1, 1, 1, -1))
            parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
            # 🔍 PRECISION TRACKING: Analyze converted TTNN bias
            analyze_tensor_precision(parameters["bias"], f"BIAS_CONVERSION-{name}", "TTNN_BIAS_float32")

    if isinstance(model, Conv):
        # 🔍 PRECISION TRACKING: Analyze original PyTorch weights before batch norm folding
        analyze_tensor_precision(model.conv.weight, f"CONV_WEIGHT_CONVERSION-{name}", "PYTORCH_CONV_WEIGHT")
        if model.bn.bias is not None:
            analyze_tensor_precision(model.bn.bias, f"BN_BIAS_CONVERSION-{name}", "PYTORCH_BN_BIAS")
        
        weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
        # 🔍 PRECISION TRACKING: Analyze weights after batch norm folding
        analyze_tensor_precision(weight, f"CONV_WEIGHT_CONVERSION-{name}", "AFTER_BN_FOLDING")
        analyze_tensor_precision(bias, f"CONV_BIAS_CONVERSION-{name}", "AFTER_BN_FOLDING")
        
        parameters["conv"] = {}
        parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        # 🔍 PRECISION TRACKING: Analyze converted TTNN weights
        analyze_tensor_precision(parameters["conv"]["weight"], f"CONV_WEIGHT_CONVERSION-{name}", "TTNN_WEIGHT_float32")
        
        bias = bias.reshape((1, 1, 1, -1))
        parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        # 🔍 PRECISION TRACKING: Analyze converted TTNN bias
        analyze_tensor_precision(parameters["conv"]["bias"], f"CONV_BIAS_CONVERSION-{name}", "TTNN_BIAS_float32")

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

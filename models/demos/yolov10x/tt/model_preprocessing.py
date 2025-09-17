# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from ttnn.model_preprocessing import fold_batch_norm2d_into_conv2d, infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.utils.common_demo_utils import get_mesh_mappers
from models.demos.yolov10x.reference.yolov10x import YOLOv10


def fold_batch_norm2d_into_conv2d(device, state_dict, path, eps=1e-03, mesh_mapper=None, split=False):
    bn_weight = state_dict[path + f".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path + f".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = state_dict[path + f".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path + f".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = state_dict[path + f".conv.weight"]
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)

    if split:
        chunk_size = bias.shape[-1] // 2
        return (
            ttnn.from_torch(weight[:chunk_size, :, :, :], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias[:, :, :, :chunk_size], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(weight[chunk_size:, :, :, :], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias[:, :, :, chunk_size:], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
        )
    else:
        return (
            ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper),
        )


def preprocess_parameters(state_dict, path, bias=True, mesh_mapper=None):
    if bias:
        conv_weight = state_dict[f"{path}.weight"]
        conv_bias = state_dict[f"{path}.bias"]

        conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        conv_bias = ttnn.reshape(ttnn.from_torch(conv_bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper), (1, 1, 1, -1))
        return (conv_weight, conv_bias)

    else:
        conv_weight = state_dict[f"{path}.conv.weight"]
        conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        return (conv_weight, None)


def custom_preprocessor(device, model, name, mesh_mapper=None):
    state_dict = model.state_dict()

    pairs = [
        "model.0",
        "model.1",
        "model.2.cv1",
        "model.2.cv2",
        "model.2.m.0.cv1",
        "model.2.m.0.cv2",
        "model.2.m.1.cv1",
        "model.2.m.1.cv2",
        "model.2.m.2.cv1",
        "model.2.m.2.cv2",
        "model.3",
        "model.4.cv1",
        "model.4.cv2",
        "model.4.m.0.cv1",
        "model.4.m.0.cv2",
        "model.4.m.1.cv1",
        "model.4.m.1.cv2",
        "model.4.m.2.cv1",
        "model.4.m.2.cv2",
        "model.4.m.3.cv1",
        "model.4.m.3.cv2",
        "model.4.m.4.cv1",
        "model.4.m.4.cv2",
        "model.4.m.5.cv1",
        "model.4.m.5.cv2",
        "model.5.cv1",
        "model.5.cv2",
        "model.6.cv1",
        "model.6.cv2",
        "model.6.m.0.cv1.0",
        "model.6.m.0.cv1.1",
        "model.6.m.0.cv1.2",
        "model.6.m.0.cv1.3",
        "model.6.m.0.cv1.4",
        "model.6.m.1.cv1.0",
        "model.6.m.1.cv1.1",
        "model.6.m.1.cv1.2",
        "model.6.m.1.cv1.3",
        "model.6.m.1.cv1.4",
        "model.6.m.2.cv1.0",
        "model.6.m.2.cv1.1",
        "model.6.m.2.cv1.2",
        "model.6.m.2.cv1.3",
        "model.6.m.2.cv1.4",
        "model.6.m.3.cv1.0",
        "model.6.m.3.cv1.1",
        "model.6.m.3.cv1.2",
        "model.6.m.3.cv1.3",
        "model.6.m.3.cv1.4",
        "model.6.m.4.cv1.0",
        "model.6.m.4.cv1.1",
        "model.6.m.4.cv1.2",
        "model.6.m.4.cv1.3",
        "model.6.m.4.cv1.4",
        "model.6.m.5.cv1.0",
        "model.6.m.5.cv1.1",
        "model.6.m.5.cv1.2",
        "model.6.m.5.cv1.3",
        "model.6.m.5.cv1.4",
        "model.7.cv1",
        "model.7.cv2",
        "model.8.cv1",
        "model.8.cv2",
        "model.8.m.0.cv1.0",
        "model.8.m.0.cv1.1",
        "model.8.m.0.cv1.2",
        "model.8.m.0.cv1.3",
        "model.8.m.0.cv1.4",
        "model.8.m.1.cv1.0",
        "model.8.m.1.cv1.1",
        "model.8.m.1.cv1.2",
        "model.8.m.1.cv1.3",
        "model.8.m.1.cv1.4",
        "model.8.m.2.cv1.0",
        "model.8.m.2.cv1.1",
        "model.8.m.2.cv1.2",
        "model.8.m.2.cv1.3",
        "model.8.m.2.cv1.4",
        "model.9.cv1",
        "model.9.cv2",
        "model.10.cv1",
        "model.10.cv2",
        "model.10.attn.qkv",
        "model.10.attn.proj",
        "model.10.attn.pe",
        "model.10.ffn.0",
        "model.10.ffn.1",
        "model.13.cv1",
        "model.13.cv2",
        "model.13.m.0.cv1.0",
        "model.13.m.0.cv1.1",
        "model.13.m.0.cv1.2",
        "model.13.m.0.cv1.3",
        "model.13.m.0.cv1.4",
        "model.13.m.1.cv1.0",
        "model.13.m.1.cv1.1",
        "model.13.m.1.cv1.2",
        "model.13.m.1.cv1.3",
        "model.13.m.1.cv1.4",
        "model.13.m.2.cv1.0",
        "model.13.m.2.cv1.1",
        "model.13.m.2.cv1.2",
        "model.13.m.2.cv1.3",
        "model.13.m.2.cv1.4",
        "model.16.cv1",
        "model.16.cv2",
        "model.16.m.0.cv1",
        "model.16.m.0.cv2",
        "model.16.m.1.cv1",
        "model.16.m.1.cv2",
        "model.16.m.2.cv1",
        "model.16.m.2.cv2",
        "model.17",
        "model.19.cv1",
        "model.19.cv2",
        "model.19.m.0.cv1.0",
        "model.19.m.0.cv1.1",
        "model.19.m.0.cv1.2",
        "model.19.m.0.cv1.3",
        "model.19.m.0.cv1.4",
        "model.19.m.1.cv1.0",
        "model.19.m.1.cv1.1",
        "model.19.m.1.cv1.2",
        "model.19.m.1.cv1.3",
        "model.19.m.1.cv1.4",
        "model.19.m.2.cv1.0",
        "model.19.m.2.cv1.1",
        "model.19.m.2.cv1.2",
        "model.19.m.2.cv1.3",
        "model.19.m.2.cv1.4",
        "model.20.cv1",
        "model.20.cv2",
        "model.22.cv1",
        "model.22.cv2",
        "model.22.m.0.cv1.0",
        "model.22.m.0.cv1.1",
        "model.22.m.0.cv1.2",
        "model.22.m.0.cv1.3",
        "model.22.m.0.cv1.4",
        "model.22.m.1.cv1.0",
        "model.22.m.1.cv1.1",
        "model.22.m.1.cv1.2",
        "model.22.m.1.cv1.3",
        "model.22.m.1.cv1.4",
        "model.22.m.2.cv1.0",
        "model.22.m.2.cv1.1",
        "model.22.m.2.cv1.2",
        "model.22.m.2.cv1.3",
        "model.22.m.2.cv1.4",
        "model.23.one2one_cv2.0.0",
        "model.23.one2one_cv2.0.1",
        "model.23.one2one_cv2.1.0",
        "model.23.one2one_cv2.1.1",
        "model.23.one2one_cv2.2.0",
        "model.23.one2one_cv2.2.1",
        "model.23.one2one_cv3.0.0.0",
        "model.23.one2one_cv3.0.0.1",
        "model.23.one2one_cv3.0.1.0",
        "model.23.one2one_cv3.0.1.1",
        "model.23.one2one_cv3.1.0.0",
        "model.23.one2one_cv3.1.0.1",
        "model.23.one2one_cv3.1.1.0",
        "model.23.one2one_cv3.1.1.1",
        "model.23.one2one_cv3.2.0.0",
        "model.23.one2one_cv3.2.0.1",
        "model.23.one2one_cv3.2.1.0",
        "model.23.one2one_cv3.2.1.1",
        "model.6.cv1",
        "model.8.cv1",
        "model.9.cv1",
        "model.10.cv1",
        "model.13.cv1",
        "model.19.cv1",
        "model.20.cv1",
        "model.22.cv1",
    ]

    parameters = {}

    split_paths = [
        "model.2.cv1",
        "model.4.cv1",
        "model.16.cv1",
    ]

    for path in pairs:
        parameters[path] = fold_batch_norm2d_into_conv2d(device, state_dict, path=path, mesh_mapper=mesh_mapper)
        if path in split_paths:
            parameters_modified = fold_batch_norm2d_into_conv2d(
                device,
                state_dict,
                path=path,
                mesh_mapper=mesh_mapper,
                split=True,
            )
            parameters[path + "_a"] = parameters_modified[:2]
            parameters[path + "_b"] = parameters_modified[2:]

    parameters["model.23.one2one_cv2.0.2"] = preprocess_parameters(
        state_dict, "model.23.one2one_cv2.0.2", mesh_mapper=mesh_mapper
    )
    parameters["model.23.one2one_cv2.1.2"] = preprocess_parameters(
        state_dict, "model.23.one2one_cv2.1.2", mesh_mapper=mesh_mapper
    )
    parameters["model.23.one2one_cv2.2.2"] = preprocess_parameters(
        state_dict, "model.23.one2one_cv2.2.2", mesh_mapper=mesh_mapper
    )
    parameters["model.23.one2one_cv3.0.2"] = preprocess_parameters(
        state_dict, "model.23.one2one_cv3.0.2", mesh_mapper=mesh_mapper
    )
    parameters["model.23.one2one_cv3.1.2"] = preprocess_parameters(
        state_dict, "model.23.one2one_cv3.1.2", mesh_mapper=mesh_mapper
    )
    parameters["model.23.one2one_cv3.2.2"] = preprocess_parameters(
        state_dict, "model.23.one2one_cv3.2.2", mesh_mapper=mesh_mapper
    )
    parameters["model.23.dfl"] = preprocess_parameters(state_dict, "model.23.dfl", bias=False, mesh_mapper=mesh_mapper)

    return parameters


def make_anchors(device, feats, strides, grid_cell_offset=0.5, weights_mesh_mapper=None):
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
        ttnn.from_torch(
            a,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=weights_mesh_mapper,
        ),
        ttnn.from_torch(
            b,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.L1_MEMORY_CONFIG,
            mesh_mapper=weights_mesh_mapper,
        ),
    )


# def custom_preprocessor(model, name, mesh_mapper=None):
#     parameters = {}
#     if isinstance(model, nn.Conv2d):
#         parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
#         if model.bias is not None:
#             bias = model.bias.reshape((1, 1, 1, -1))
#             parameters["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

#     if isinstance(model, Conv):
#         weight, bias = fold_batch_norm2d_into_conv2d(model.conv, model.bn)
#         parameters["conv"] = {}
#         parameters["conv"]["weight"] = ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
#         bias = bias.reshape((1, 1, 1, -1))
#         parameters["conv"]["bias"] = ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper)

#     return parameters


def create_yolov10x_model_parameters(model: YOLOv10, input_tensor: torch.Tensor, device):
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(model=model, run_model=lambda model: model(input_tensor), device=None)
    feats = [80, 40, 20]
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides, weights_mesh_mapper=weights_mesh_mapper)

    parameters["anchors"] = anchors
    parameters["strides"] = strides

    parameters["model_args"] = model

    return parameters


def create_yolov10_model_parameters_detect(
    model, input_tensor_1, input_tensor_2, input_tensor_3, device, weights_mesh_mapper=None
):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=create_custom_mesh_preprocessor(weights_mesh_mapper),
        device=None,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model([input_tensor_1, input_tensor_2, input_tensor_3]), device=None
    )

    feats = [80, 40, 20]
    strides = torch.tensor([8.0, 16.0, 32.0])

    anchors, strides = make_anchors(
        device, feats, strides, weights_mesh_mapper=weights_mesh_mapper
    )  # Optimization: Processing make anchors outside model run

    parameters["anchors"] = anchors
    parameters["strides"] = strides
    parameters["model_args"] = model

    parameters["model"] = model

    return parameters


def create_yolov10x_input_tensors(
    device, batch_size=1, input_channels=3, input_height=640, input_width=640, input_dtype=ttnn.bfloat8_b
):
    inputs_mesh_mapper, _, _ = get_mesh_mappers(device)
    torch_input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)
    n, c, h, w = torch_input_tensor.shape
    if c == 3:
        c = 16
    num_devices = device.get_num_devices()
    n = n // num_devices if n // num_devices != 0 else n
    input_mem_config = ttnn.create_sharded_memory_config(
        [n, c, h, w],
        ttnn.CoreGrid(x=8, y=8),
        ttnn.ShardStrategy.HEIGHT,
    )
    ttnn_input_host = ttnn.from_torch(
        torch_input_tensor,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=input_mem_config,
        mesh_mapper=inputs_mesh_mapper,
    )
    return torch_input_tensor, ttnn_input_host


def create_yolov10x_input_tensors_submodules(
    device, batch_size=1, input_channels=3, input_height=640, input_width=640, input_dtype=ttnn.bfloat8_b
):
    torch_input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)
    ttnn_input_tensor = torch.permute(torch_input_tensor, (0, 2, 3, 1))
    ttnn_input_tensor = ttnn_input_tensor.reshape(
        1,
        1,
        ttnn_input_tensor.shape[0] * ttnn_input_tensor.shape[1] * ttnn_input_tensor.shape[2],
        ttnn_input_tensor.shape[3],
    )
    ttnn_input_tensor = ttnn.from_torch(
        ttnn_input_tensor,
        dtype=input_dtype,
        device=device,
        layout=ttnn.TILE_LAYOUT,
    )
    return torch_input_tensor, ttnn_input_tensor


def create_custom_mesh_preprocessor(device, mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(device, model, name, mesh_mapper)

    return custom_mesh_preprocessor

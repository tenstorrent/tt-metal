# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
from ttnn.model_preprocessing import infer_ttnn_module_args, preprocess_model_parameters

import ttnn
from models.demos.yolov9c.reference.yolov9c import YoloV9


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
        "model.2.cv2.0.cv1",
        "model.2.cv2.0.cv2",
        "model.2.cv2.0.cv3",
        "model.2.cv2.0.m.0.cv1.conv1",
        "model.2.cv2.0.m.0.cv1.conv2",
        "model.2.cv2.0.m.0.cv2",
        "model.2.cv2.1",
        "model.2.cv3.0.cv1",
        "model.2.cv3.0.cv2",
        "model.2.cv3.0.cv3",
        "model.2.cv3.0.m.0.cv1.conv1",
        "model.2.cv3.0.m.0.cv1.conv2",
        "model.2.cv3.0.m.0.cv2",
        "model.2.cv3.1",
        "model.2.cv4",
        "model.3.cv1",
        "model.3.cv2",
        "model.4.cv1",
        "model.4.cv2.0.cv1",
        "model.4.cv2.0.cv2",
        "model.4.cv2.0.cv3",
        "model.4.cv2.0.m.0.cv1.conv1",
        "model.4.cv2.0.m.0.cv1.conv2",
        "model.4.cv2.0.m.0.cv2",
        "model.4.cv2.1",
        "model.4.cv3.0.cv1",
        "model.4.cv3.0.cv2",
        "model.4.cv3.0.cv3",
        "model.4.cv3.0.m.0.cv1.conv1",
        "model.4.cv3.0.m.0.cv1.conv2",
        "model.4.cv3.0.m.0.cv2",
        "model.4.cv3.1",
        "model.4.cv4",
        "model.5.cv1",
        "model.5.cv2",
        "model.6.cv1",
        "model.6.cv2.0.cv1",
        "model.6.cv2.0.cv2",
        "model.6.cv2.0.cv3",
        "model.6.cv2.0.m.0.cv1.conv1",
        "model.6.cv2.0.m.0.cv1.conv2",
        "model.6.cv2.0.m.0.cv2",
        "model.6.cv2.1",
        "model.6.cv3.0.cv1",
        "model.6.cv3.0.cv2",
        "model.6.cv3.0.cv3",
        "model.6.cv3.0.m.0.cv1.conv1",
        "model.6.cv3.0.m.0.cv1.conv2",
        "model.6.cv3.0.m.0.cv2",
        "model.6.cv3.1",
        "model.6.cv4",
        "model.7.cv1",
        "model.7.cv2",
        "model.8.cv1",
        "model.8.cv2.0.cv1",
        "model.8.cv2.0.cv2",
        "model.8.cv2.0.cv3",
        "model.8.cv2.0.m.0.cv1.conv1",
        "model.8.cv2.0.m.0.cv1.conv2",
        "model.8.cv2.0.m.0.cv2",
        "model.8.cv2.1",
        "model.8.cv3.0.cv1",
        "model.8.cv3.0.cv2",
        "model.8.cv3.0.cv3",
        "model.8.cv3.0.m.0.cv1.conv1",
        "model.8.cv3.0.m.0.cv1.conv2",
        "model.8.cv3.0.m.0.cv2",
        "model.8.cv3.1",
        "model.8.cv4",
        "model.9.cv1",
        "model.9.cv5",
        "model.12.cv1",
        "model.12.cv2.0.cv1",
        "model.12.cv2.0.cv2",
        "model.12.cv2.0.cv3",
        "model.12.cv2.0.m.0.cv1.conv1",
        "model.12.cv2.0.m.0.cv1.conv2",
        "model.12.cv2.0.m.0.cv2",
        "model.12.cv2.1",
        "model.12.cv3.0.cv1",
        "model.12.cv3.0.cv2",
        "model.12.cv3.0.cv3",
        "model.12.cv3.0.m.0.cv1.conv1",
        "model.12.cv3.0.m.0.cv1.conv2",
        "model.12.cv3.0.m.0.cv2",
        "model.12.cv3.1",
        "model.12.cv4",
        "model.15.cv1",
        "model.15.cv2.0.cv1",
        "model.15.cv2.0.cv2",
        "model.15.cv2.0.cv3",
        "model.15.cv2.0.m.0.cv1.conv1",
        "model.15.cv2.0.m.0.cv1.conv2",
        "model.15.cv2.0.m.0.cv2",
        "model.15.cv2.1",
        "model.15.cv3.0.cv1",
        "model.15.cv3.0.cv2",
        "model.15.cv3.0.cv3",
        "model.15.cv3.0.m.0.cv1.conv1",
        "model.15.cv3.0.m.0.cv1.conv2",
        "model.15.cv3.0.m.0.cv2",
        "model.15.cv3.1",
        "model.15.cv4",
        "model.16.cv1",
        "model.16.cv2",
        "model.18.cv1",
        "model.18.cv2.0.cv1",
        "model.18.cv2.0.cv2",
        "model.18.cv2.0.cv3",
        "model.18.cv2.0.m.0.cv1.conv1",
        "model.18.cv2.0.m.0.cv1.conv2",
        "model.18.cv2.0.m.0.cv2",
        "model.18.cv2.1",
        "model.18.cv3.0.cv1",
        "model.18.cv3.0.cv2",
        "model.18.cv3.0.cv3",
        "model.18.cv3.0.m.0.cv1.conv1",
        "model.18.cv3.0.m.0.cv1.conv2",
        "model.18.cv3.0.m.0.cv2",
        "model.18.cv3.1",
        "model.18.cv4",
        "model.19.cv1",
        "model.19.cv2",
        "model.21.cv1",
        "model.21.cv2.0.cv1",
        "model.21.cv2.0.cv2",
        "model.21.cv2.0.cv3",
        "model.21.cv2.0.m.0.cv1.conv1",
        "model.21.cv2.0.m.0.cv1.conv2",
        "model.21.cv2.0.m.0.cv2",
        "model.21.cv2.1",
        "model.21.cv3.0.cv1",
        "model.21.cv3.0.cv2",
        "model.21.cv3.0.cv3",
        "model.21.cv3.0.m.0.cv1.conv1",
        "model.21.cv3.0.m.0.cv1.conv2",
        "model.21.cv3.0.m.0.cv2",
        "model.21.cv3.1",
        "model.21.cv4",
        "model.22.cv2.0.0",
        "model.22.cv2.0.1",
        "model.22.cv2.1.0",
        "model.22.cv2.1.1",
        "model.22.cv2.2.0",
        "model.22.cv2.2.1",
        "model.22.cv3.0.0",
        "model.22.cv3.0.1",
        "model.22.cv3.1.0",
        "model.22.cv3.1.1",
        "model.22.cv3.2.0",
        "model.22.cv3.2.1",
    ]

    parameters = {}

    split_paths = [
        "model.2.cv1",
        "model.4.cv1",
        "model.6.cv1",
        "model.8.cv1",
        "model.12.cv1",
        "model.15.cv1",
        "model.18.cv1",
        "model.21.cv1",
    ]
    model_type = model.model[-1].__class__.__name__
    if model_type == "Segment":
        segment_pairs = [
            "model.22.proto.cv1",
            "model.22.proto.cv2",
            "model.22.proto.cv3",
            "model.22.cv4.0.0",
            "model.22.cv4.0.1",
            "model.22.cv4.1.0",
            "model.22.cv4.1.1",
            "model.22.cv4.2.0",
            "model.22.cv4.2.1",
        ]
        pairs.extend(segment_pairs)

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

    parameters["model.22.cv2.0.2"] = preprocess_parameters(state_dict, "model.22.cv2.0.2", mesh_mapper=mesh_mapper)
    parameters["model.22.cv2.1.2"] = preprocess_parameters(state_dict, "model.22.cv2.1.2", mesh_mapper=mesh_mapper)
    parameters["model.22.cv2.2.2"] = preprocess_parameters(state_dict, "model.22.cv2.2.2", mesh_mapper=mesh_mapper)
    parameters["model.22.cv3.0.2"] = preprocess_parameters(state_dict, "model.22.cv3.0.2", mesh_mapper=mesh_mapper)
    parameters["model.22.cv3.1.2"] = preprocess_parameters(state_dict, "model.22.cv3.1.2", mesh_mapper=mesh_mapper)
    parameters["model.22.cv3.2.2"] = preprocess_parameters(state_dict, "model.22.cv3.2.2", mesh_mapper=mesh_mapper)
    parameters["model.22.dfl"] = preprocess_parameters(state_dict, "model.22.dfl", bias=False, mesh_mapper=mesh_mapper)
    if model_type == "Segment":
        parameters["model.22.proto.upsample"] = preprocess_parameters(
            state_dict, "model.22.proto.upsample", mesh_mapper=mesh_mapper
        )
        parameters["model.22.cv4.0.2"] = preprocess_parameters(state_dict, "model.22.cv4.0.2", mesh_mapper=mesh_mapper)
        parameters["model.22.cv4.1.2"] = preprocess_parameters(state_dict, "model.22.cv4.1.2", mesh_mapper=mesh_mapper)
        parameters["model.22.cv4.2.2"] = preprocess_parameters(state_dict, "model.22.cv4.2.2", mesh_mapper=mesh_mapper)
    return parameters


def create_yolov9c_input_tensors(
    device, batch_size=1, input_channels=3, input_height=640, input_width=640, model=False
):
    inputs_mesh_mapper, _, _ = get_mesh_mappers(device)

    torch_input_tensor = torch.randn(batch_size * device.get_num_devices(), input_channels, input_height, input_width)
    ttnn_input_tensor = None
    if model:
        n, c, h, w = torch_input_tensor.shape
        if c == 3:
            c = 16
        input_mem_config = ttnn.create_sharded_memory_config(
            [n // device.get_num_devices(), c, h, w],
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
        # ttnn_input_tensor = ttnn.to_device(ttnn_input_host, device, memory_config=input_mem_config)
    return torch_input_tensor, ttnn_input_host


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
            a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=weights_mesh_mapper
        ),
        ttnn.from_torch(
            b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, mesh_mapper=weights_mesh_mapper
        ),
    )


def create_yolov9c_model_parameters(model: YoloV9, input_tensor: torch.Tensor, device):
    _, weights_mesh_mapper, _ = get_mesh_mappers(device)
    state_dict = model.state_dict()

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

    anchors, strides = make_anchors(device, feats, strides, weights_mesh_mapper=weights_mesh_mapper)
    parameters["anchors"] = anchors
    parameters["strides"] = strides

    return parameters


def create_yolov9c_model_parameters_detect(
    model: YoloV9,
    input_tensor_1: torch.Tensor,
    input_tensor_2: torch.Tensor,
    input_tensor_3: torch.Tensor,
    device,
    weights_mesh_mapper=None,
):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    parameters.conv_args = {}
    parameters.conv_args = infer_ttnn_module_args(
        model=model, run_model=lambda model: model(input_tensor_1, input_tensor_2, input_tensor_3), device=None
    )
    parameters["model_args"] = model

    feats = [80, 40, 20]  # Values depends on input resolution. Current: 640x640
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(
        device, feats, strides, weights_mesh_mapper=weights_mesh_mapper
    )  # Optimization: Processing make anchors outside model run

    parameters["anchors"] = anchors
    parameters["strides"] = strides
    parameters["model"] = model

    return parameters


def get_mesh_mappers(device):
    if device.get_num_devices() > 1:
        inputs_mesh_mapper = ttnn.ShardTensorToMesh(device, dim=0)
        weights_mesh_mapper = ttnn.ReplicateTensorToMesh(device)
        output_mesh_composer = ttnn.ConcatMeshToTensor(device, dim=0)
    else:
        inputs_mesh_mapper = None
        weights_mesh_mapper = None
        output_mesh_composer = None
    return inputs_mesh_mapper, weights_mesh_mapper, output_mesh_composer


def create_custom_mesh_preprocessor(device, mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(device, model, name, mesh_mapper)

    return custom_mesh_preprocessor

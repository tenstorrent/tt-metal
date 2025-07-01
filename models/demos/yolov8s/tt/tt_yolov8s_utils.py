# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch

import ttnn


def fold_batch_norm2d_into_conv2d(device, state_dict, path, eps=1e-03, bfloat8=False, mesh_mapper=None):
    bn_weight = state_dict[path + f".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path + f".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = state_dict[path + f".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path + f".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = state_dict[path + f".conv.weight"]
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)
    if bfloat8:
        return (
            ttnn.from_torch(weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper),
        )

    return (
        ttnn.from_torch(weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        ttnn.from_torch(bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    )


def fold_batch_norm2d_into_conv2d_split(device, state_dict, path, eps=1e-03, bfloat8=False, mesh_mapper=None):
    bn_weight = state_dict[path + f".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path + f".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = state_dict[path + f".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path + f".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = state_dict[path + f".conv.weight"]
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)

    chunk_size = bias.shape[-1] // 2

    if bfloat8:
        return (
            ttnn.from_torch(weight[:chunk_size, :, :, :], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias[:, :, :, :chunk_size], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(weight[chunk_size:, :, :, :], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
            ttnn.from_torch(bias[:, :, :, chunk_size:], dtype=ttnn.float32, mesh_mapper=mesh_mapper),
        )

    return (
        ttnn.from_torch(weight[:chunk_size, :, :, :], dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        ttnn.from_torch(bias[:, :, :, :chunk_size], dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        ttnn.from_torch(weight[chunk_size:, :, :, :], dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
        ttnn.from_torch(bias[:, :, :, chunk_size:], dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper),
    )


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p if isinstance(p, int) else p[0]


def make_anchors(device, feats, strides, grid_cell_offset=0.5, mesh_mapper=None):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    for i, stride in enumerate(strides):
        h, w = feats[i]
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


def ttnn_decode_bboxes(device, distance, anchor_points, xywh=True, dim=1):
    lt, rb = ttnn.split(distance, 2, 1, memory_config=ttnn.L1_MEMORY_CONFIG)  # if done in tile : tt-metal issue #17017

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = x1y1 + x2y2
        c_xy = ttnn.div(c_xy, 2, dtype=ttnn.bfloat8_b)
        wh = ttnn.subtract(x2y2, x1y1, dtype=ttnn.bfloat8_b)
        return ttnn.concat([c_xy, wh], 1, memory_config=ttnn.L1_MEMORY_CONFIG)


def preprocess_parameters(state_dict, path, bias=True, bfloat8=True, mesh_mapper=None):
    if bias:
        conv_weight = state_dict[f"{path}.2.weight"]
        conv_bias = state_dict[f"{path}.2.bias"]

        if bfloat8:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
            conv_bias = ttnn.reshape(
                ttnn.from_torch(conv_bias, dtype=ttnn.float32, mesh_mapper=mesh_mapper), (1, 1, 1, -1)
            )
        else:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)
            conv_bias = ttnn.reshape(
                ttnn.from_torch(conv_bias, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper), (1, 1, 1, -1)
            )

        return (conv_weight, conv_bias)

    else:
        conv_weight = state_dict[f"{path}.conv.weight"]

        if bfloat8:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32, mesh_mapper=mesh_mapper)
        else:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16, mesh_mapper=mesh_mapper)

        return (conv_weight, None)


def custom_preprocessor(device, state_dict, inp_h=640, inp_w=640, mesh_mapper=None):
    pairs = [
        ("model.0", True),
        ("model.1", True),
        ("model.2.cv1", True),
        ("model.2.m.0.cv1", True),
        ("model.2.m.0.cv2", True),
        ("model.2.cv2", True),
        ("model.3", True),
        ("model.4.cv1", True),
        ("model.4.m.0.cv1", True),
        ("model.4.m.0.cv2", True),
        ("model.4.m.1.cv1", True),
        ("model.4.m.1.cv2", True),
        ("model.4.cv2", True),
        ("model.5", True),
        ("model.6.cv1", True),
        ("model.6.m.0.cv1", True),
        ("model.6.m.0.cv2", True),
        ("model.6.m.1.cv1", True),
        ("model.6.m.1.cv2", True),
        ("model.6.cv2", True),
        ("model.7", True),
        ("model.8.cv1", True),
        ("model.8.m.0.cv1", True),
        ("model.8.m.0.cv2", True),
        ("model.8.cv2", True),
        ("model.9.cv1", True),
        ("model.9.cv2", True),
        ("model.12.cv1", True),
        ("model.12.m.0.cv1", True),
        ("model.12.m.0.cv2", True),
        ("model.12.cv2", True),
        ("model.15.cv1", True),
        ("model.15.m.0.cv1", True),
        ("model.15.m.0.cv2", True),
        ("model.15.cv2", True),
        ("model.16", True),
        ("model.18.cv1", True),
        ("model.18.m.0.cv1", True),
        ("model.18.m.0.cv2", True),
        ("model.18.cv2", True),
        ("model.19", True),
        ("model.21.cv1", True),
        ("model.21.m.0.cv1", True),
        ("model.21.m.0.cv2", True),
        ("model.21.cv2", True),
        ("model.22.cv2.0.0", True),
        ("model.22.cv2.0.1", True),
        ("model.22.cv3.0.0", True),
        ("model.22.cv3.0.1", True),
        ("model.22.cv2.1.0", True),
        ("model.22.cv2.1.1", True),
        ("model.22.cv3.1.0", True),
        ("model.22.cv3.1.1", True),
        ("model.22.cv2.2.0", True),
        ("model.22.cv2.2.1", True),
        ("model.22.cv3.2.0", True),
        ("model.22.cv3.2.1", True),
    ]

    parameters = {}

    c2f_paths = [
        "model.2.cv1",
        "model.4.cv1",
        "model.6.cv1",
        "model.8.cv1",
        "model.12.cv1",
        "model.15.cv1",
        "model.18.cv1",
        "model.21.cv1",
    ]

    for path, bfloat8 in pairs:
        parameters[path] = fold_batch_norm2d_into_conv2d(
            device, state_dict, path=path, bfloat8=True, mesh_mapper=mesh_mapper
        )

        if path in c2f_paths:
            parameters_modified = fold_batch_norm2d_into_conv2d_split(
                device, state_dict, path=path, bfloat8=True, mesh_mapper=mesh_mapper
            )
            parameters[path + "_a"] = parameters_modified[:2]
            parameters[path + "_b"] = parameters_modified[2:]

    parameters["model.22.cv2.0"] = preprocess_parameters(
        state_dict, "model.22.cv2.0", bfloat8=True, mesh_mapper=mesh_mapper
    )
    parameters["model.22.cv3.0"] = preprocess_parameters(
        state_dict, "model.22.cv3.0", bfloat8=True, mesh_mapper=mesh_mapper
    )

    parameters["model.22.cv2.1"] = preprocess_parameters(
        state_dict, "model.22.cv2.1", bfloat8=True, mesh_mapper=mesh_mapper
    )
    parameters["model.22.cv3.1"] = preprocess_parameters(
        state_dict, "model.22.cv3.1", bfloat8=True, mesh_mapper=mesh_mapper
    )

    parameters["model.22.cv2.2"] = preprocess_parameters(
        state_dict, "model.22.cv2.2", bfloat8=True, mesh_mapper=mesh_mapper
    )
    parameters["model.22.cv3.2"] = preprocess_parameters(
        state_dict, "model.22.cv3.2", bfloat8=True, mesh_mapper=mesh_mapper
    )

    parameters["model.22.dfl"] = preprocess_parameters(
        state_dict, "model.22.dfl", bias=False, bfloat8=True, mesh_mapper=mesh_mapper
    )

    strides = [8, 16, 32]

    feats = [(inp_h // 8, inp_w // 8), (inp_h // 16, inp_w // 16), (inp_h // 32, inp_w // 32)]  # value depend on res

    anchors, strides = make_anchors(
        device, feats, strides, mesh_mapper=mesh_mapper
    )  # Optimization: Processing make anchors outside model run

    parameters["anchors"] = anchors
    parameters["strides"] = strides

    return parameters


def create_custom_mesh_preprocessor(device, mesh_mapper=None):
    def custom_mesh_preprocessor(model, name, ttnn_module_args, convert_to_ttnn):
        return custom_preprocessor(
            device,
            model.state_dict(),
            inp_h=640,
            inp_w=640,
            mesh_mapper=mesh_mapper,  # model, name, ttnn_module_args, convert_to_ttnn, custom_mesh_preprocessor, mesh_mapper
        )

    return custom_mesh_preprocessor

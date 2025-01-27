# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


def fold_batch_norm2d_into_conv2d(device, state_dict, path, eps=1e-03, bfloat8=False):
    bn_weight = state_dict[path + f".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path + f".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = state_dict[path + f".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path + f".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = state_dict[path + f".conv.weight"]
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)

    if bfloat8:
        return (ttnn.from_torch(weight, dtype=ttnn.float32), ttnn.from_torch(bias, dtype=ttnn.float32))

    return (ttnn.from_torch(weight, dtype=ttnn.bfloat16), ttnn.from_torch(bias, dtype=ttnn.bfloat16))


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p if isinstance(p, int) else p[0]


def ttnn_decode_bboxes(device, distance, anchor_points, xywh=True, dim=1):
    lt, rb = ttnn.split(distance, 2, 1)
    lt = ttnn.to_layout(lt, ttnn.TILE_LAYOUT)
    rb = ttnn.to_layout(rb, ttnn.TILE_LAYOUT)

    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = x1y1 + x2y2
        c_xy = ttnn.div(c_xy, 2)
        wh = x2y2 - x1y1
        return ttnn.concat([c_xy, wh], 1)


def make_anchors(device, feats, strides, grid_cell_offset=0.5):
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
        ttnn.from_torch(a, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
        ttnn.from_torch(b, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device),
    )


def preprocess_parameters(state_dict, path, bias=True):
    if bias:
        conv_weight = state_dict[f"{path}.2.weight"]
        conv_bias = state_dict[f"{path}.2.bias"]

        conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
        conv_bias = ttnn.reshape(ttnn.from_torch(conv_bias, dtype=ttnn.bfloat16), (1, 1, 1, -1))

        return (conv_weight, conv_bias)

    else:
        conv_weight = state_dict[f"{path}.conv.weight"]
        conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)

        return conv_weight


def custom_preprocessor(device, state_dict):
    pairs = [
        ("model.0", False),
        ("model.1", False),
        ("model.2.cv1", False),
        ("model.2.m.0.cv1", False),
        ("model.2.m.0.cv2", False),
        ("model.2.m.1.cv1", False),
        ("model.2.m.1.cv2", False),
        ("model.2.cv2", False),
        ("model.3", False),
        ("model.4.cv1", True),
        ("model.4.m.0.cv1", False),
        ("model.4.m.0.cv2", False),
        ("model.4.m.1.cv1", False),
        ("model.4.m.1.cv2", False),
        ("model.4.m.2.cv1", False),
        ("model.4.m.2.cv2", False),
        ("model.4.m.3.cv1", False),
        ("model.4.m.3.cv2", False),
        ("model.4.cv2", True),
        ("model.5", False),
        ("model.6.cv1", False),
        ("model.6.m.0.cv1", False),
        ("model.6.m.0.cv2", False),
        ("model.6.m.1.cv1", False),
        ("model.6.m.1.cv2", False),
        ("model.6.m.2.cv1", False),
        ("model.6.m.2.cv2", False),
        ("model.6.m.3.cv1", False),
        ("model.6.m.3.cv2", False),
        ("model.6.cv2", False),
        ("model.7", False),
        ("model.8.cv1", True),
        ("model.8.m.0.cv1", False),
        ("model.8.m.0.cv2", False),
        ("model.8.m.1.cv1", False),
        ("model.8.m.1.cv2", False),
        ("model.8.cv2", True),
        ("model.9.cv1", False),
        ("model.9.cv2", False),
        ("model.12.cv1", True),
        ("model.12.m.0.cv1", False),
        ("model.12.m.0.cv2", False),
        ("model.12.m.1.cv1", False),
        ("model.12.m.1.cv2", False),
        ("model.12.cv2", True),
        ("model.15.cv1", False),
        ("model.15.m.0.cv1", False),
        ("model.15.m.0.cv2", False),
        ("model.15.m.1.cv1", False),
        ("model.15.m.1.cv2", False),
        ("model.15.cv2", False),
        ("model.16", False),
        ("model.18.cv1", True),
        ("model.18.m.0.cv1", False),
        ("model.18.m.0.cv2", False),
        ("model.18.m.1.cv1", False),
        ("model.18.m.1.cv2", False),
        ("model.18.cv2", True),
        ("model.19", False),
        ("model.21.cv1", True),
        ("model.21.m.0.cv1", False),
        ("model.21.m.0.cv2", False),
        ("model.21.m.1.cv1", False),
        ("model.21.m.1.cv2", False),
        ("model.21.cv2", True),
        ("model.22.cv2.0.0", False),
        ("model.22.cv2.0.1", False),
        ("model.22.cv3.0.0", True),
        ("model.22.cv3.0.1", True),
        ("model.22.cv2.1.0", False),
        ("model.22.cv2.1.1", False),
        ("model.22.cv3.1.0", True),
        ("model.22.cv3.1.1", True),
        ("model.22.cv2.2.0", False),
        ("model.22.cv2.2.1", False),
        ("model.22.cv3.2.0", True),
        ("model.22.cv3.2.1", True),
    ]

    parameters = {}

    for path, bfloat8 in pairs:
        parameters[path] = fold_batch_norm2d_into_conv2d(device, state_dict, path=path, bfloat8=bfloat8)

    # Detect

    parameters["model.22.cv2.0"] = preprocess_parameters(state_dict, "model.22.cv2.0")
    parameters["model.22.cv3.0"] = preprocess_parameters(state_dict, "model.22.cv3.0")

    parameters["model.22.cv2.1"] = preprocess_parameters(state_dict, "model.22.cv2.1")
    parameters["model.22.cv3.1"] = preprocess_parameters(state_dict, "model.22.cv3.1")

    parameters["model.22.cv2.2"] = preprocess_parameters(state_dict, "model.22.cv2.2")
    parameters["model.22.cv3.2"] = preprocess_parameters(state_dict, "model.22.cv3.2")

    # DFL conv

    parameters["model.22.dfl"] = preprocess_parameters(state_dict, "model.22.dfl", bias=False)

    feats = [80, 40, 20]  # Values depends on input resolution. Current: 640*640
    strides = [8.0, 16.0, 32.0]

    anchors, strides = make_anchors(device, feats, strides)  # Optimization: Processing make anchors outside model run

    parameters["anchors"] = anchors
    parameters["strides"] = strides

    return parameters

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0


import ttnn
import torch


def fold_batch_norm2d_into_conv2d(device, state_dict, path1, path2, eps=1e-05, bfloat8=False):
    bn_weight = state_dict[path2 + f".weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path2 + f".bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    bn_running_mean = state_dict[path2 + f".running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path2 + f".running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    weight = state_dict[path1 + f".weight"]
    weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
    bias = bias.reshape(1, 1, 1, -1)

    if bfloat8:
        return (ttnn.from_torch(weight, dtype=ttnn.float32), ttnn.from_torch(bias, dtype=ttnn.float32))

    return (ttnn.from_torch(weight, dtype=ttnn.bfloat16), ttnn.from_torch(bias, dtype=ttnn.bfloat16))


def preprocess_parameters(state_dict, path, bias=True, bfloat8=False):
    if bias:
        conv_weight = state_dict[f"{path}.weight"]
        conv_bias = state_dict[f"{path}.bias"]

        if bfloat8:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
            conv_bias = ttnn.reshape(ttnn.from_torch(conv_bias, dtype=ttnn.float32), (1, 1, 1, -1))
        else:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
            conv_bias = ttnn.reshape(ttnn.from_torch(conv_bias, dtype=ttnn.bfloat16), (1, 1, 1, -1))

        return (conv_weight, conv_bias)
    else:
        conv_weight = state_dict[f"{path}.weight"]
        if bfloat8:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
        else:
            conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
        return (conv_weight, None)


def preprocess_bn_weight(state_dict, path, eps=1e-05, device=None):
    bn_weight = state_dict[path + f".weight"].reshape(1, -1, 1, 1)  # .unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = state_dict[path + f".bias"].reshape(1, -1, 1, 1)  # .unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_mean = state_dict[path + f".running_mean"].reshape(
        1, -1, 1, 1
    )  # .unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = state_dict[path + f".running_var"].reshape(1, -1, 1, 1)  # .unsqueeze(1).unsqueeze(1).unsqueeze(1)

    return (
        ttnn.from_torch(
            bn_weight,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,  # , memory_config=ttnn.L1_MEMORY_CONFIG
        ),
        ttnn.from_torch(
            bn_bias,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,  # , memory_config=ttnn.L1_MEMORY_CONFIG
        ),
        ttnn.from_torch(
            bn_running_mean,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
        ),
        ttnn.from_torch(
            bn_running_var,
            dtype=ttnn.bfloat16,
            device=device,
            layout=ttnn.TILE_LAYOUT,
            # memory_config=ttnn.L1_MEMORY_CONFIG,
        ),
    )


def custom_preprocessor(device, state_dict):
    pairs = [
        ("backbone.conv1", "backbone.bn1"),
        ("backbone.layer1.0.conv1", "backbone.layer1.0.bn1"),
        ("backbone.layer1.0.conv2", "backbone.layer1.0.bn2"),
        ("backbone.layer1.1.conv1", "backbone.layer1.1.bn1"),
        ("backbone.layer1.1.conv2", "backbone.layer1.1.bn2"),
        ("backbone.layer2.0.conv1", "backbone.layer2.0.bn1"),
        ("backbone.layer2.0.conv2", "backbone.layer2.0.bn2"),
        ("backbone.layer2.0.downsample.0", "backbone.layer2.0.downsample.1"),
        ("backbone.layer2.1.conv1", "backbone.layer2.1.bn1"),
        ("backbone.layer2.1.conv2", "backbone.layer2.1.bn2"),
        ("backbone.layer3.0.conv1", "backbone.layer3.0.bn1"),
        ("backbone.layer3.0.conv2", "backbone.layer3.0.bn2"),
        ("backbone.layer3.0.downsample.0", "backbone.layer3.0.downsample.1"),
        ("backbone.layer3.1.conv1", "backbone.layer3.1.bn1"),
        ("backbone.layer3.1.conv2", "backbone.layer3.1.bn2"),
        ("backbone.layer4.0.conv1", "backbone.layer4.0.bn1"),
        ("backbone.layer4.0.conv2", "backbone.layer4.0.bn2"),
        ("backbone.layer4.0.downsample.0", "backbone.layer4.0.downsample.1"),
        ("backbone.layer4.1.conv1", "backbone.layer4.1.bn1"),
        ("backbone.layer4.1.conv2", "backbone.layer4.1.bn2"),
    ]

    parameters = {}

    for path1, path2 in pairs:
        parameters[path1 + ".weight"], parameters[path1 + ".bias"] = fold_batch_norm2d_into_conv2d(
            device, state_dict, path1=path1, path2=path2
        )

    heads = [
        ("bbox_head.heatmap_head.0", True),
        ("bbox_head.heatmap_head.2", True),
        ("bbox_head.wh_head.0", True),
        ("bbox_head.wh_head.2", True),
        ("bbox_head.offset_head.0", True),
        ("bbox_head.offset_head.2", True),
        ("neck.deconv_layers.0.conv", False),
        ("neck.deconv_layers.0.conv.conv_offset", True),
        ("neck.deconv_layers.1.conv", False),
        ("neck.deconv_layers.2.conv", False),
        ("neck.deconv_layers.2.conv.conv_offset", True),
        ("neck.deconv_layers.3.conv", False),
        ("neck.deconv_layers.4.conv", False),
        ("neck.deconv_layers.4.conv.conv_offset", True),
        ("neck.deconv_layers.5.conv", False),
    ]

    for paths, bias in heads:
        parameters[paths + ".weight"], parameters[paths + ".bias"] = preprocess_parameters(
            state_dict, path=paths, bias=bias
        )

    bn = [
        "neck.deconv_layers.0.bn",
        "neck.deconv_layers.1.bn",
        "neck.deconv_layers.2.bn",
        "neck.deconv_layers.3.bn",
        "neck.deconv_layers.4.bn",
        "neck.deconv_layers.5.bn",
    ]

    for path in bn:
        (
            parameters[path + ".weight"],
            parameters[path + ".bias"],
            parameters[path + ".running_mean"],
            parameters[path + ".running_var"],
        ) = preprocess_bn_weight(state_dict, path=path, device=device)

    return parameters

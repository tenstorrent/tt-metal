# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch


# def fold_batch_norm2d_into_conv2d(device, state_dict, path, eps=1e-03, bfloat8=False, seperable_conv_norm = False):
#     bn_weight = state_dict[path + f".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
#     bn_bias = state_dict[path + f".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
#     bn_running_mean = state_dict[path + f".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
#     bn_running_var = state_dict[path + f".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
#     if seperable_conv_norm:
#         weight = state_dict[path + f".conv_pw.weight"]
#     else:
#         weight = state_dict[path + f".conv.weight"]
#     weight = (weight / torch.sqrt(bn_running_var + eps)) * bn_weight
#     bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var + eps)) + bn_bias
#     bias = bias.reshape(1, 1, 1, -1)

#     if bfloat8:
#         return (ttnn.from_torch(weight, dtype=ttnn.float32), ttnn.from_torch(bias, dtype=ttnn.float32))

#     return (ttnn.from_torch(weight, dtype=ttnn.bfloat16), ttnn.from_torch(bias, dtype=ttnn.bfloat16))


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


def preprocess_parameters(state_dict, path, seperable_conv_norm=False, effective_se=False):
    # if bias:
    #     conv_weight = state_dict[f"{path}.2.weight"]
    #     conv_bias = state_dict[f"{path}.2.bias"]

    #     if bfloat8:
    #         conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
    #         conv_bias = ttnn.reshape(ttnn.from_torch(conv_bias, dtype=ttnn.float32), (1, 1, 1, -1))
    #     else:
    #         conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.bfloat16)
    #         conv_bias = ttnn.reshape(ttnn.from_torch(conv_bias, dtype=ttnn.bfloat16), (1, 1, 1, -1))

    #     return (conv_weight, conv_bias)

    # else:
    if effective_se:
        conv_weight = state_dict[f"{path}.fc.weight"]
        conv_bias = state_dict[f"{path}.fc.bias"].reshape(1, 1, 1, -1)
        conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
        conv_bias = ttnn.from_torch(conv_bias, dtype=ttnn.float32)
        return (conv_weight, conv_bias)
    elif seperable_conv_norm:
        if "dw" in path:
            conv_weight = state_dict[f"{path}.weight"]
        if "pw" in path:
            conv_weight = state_dict[f"{path}.weight"]
    else:
        conv_weight = state_dict[f"{path}.weight"]

    # if bfloat8:
    #     conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32)
    # else:
    conv_weight = ttnn.from_torch(conv_weight, dtype=ttnn.float32)

    return (conv_weight, None)


def preprocess_linear_weight(state_dict, path, device=None):
    weight = torch.transpose(state_dict[f"{path}.weight"], 0, 1)
    weight = ttnn.from_torch(
        weight.unsqueeze(0).unsqueeze(0), dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device
    )
    bias = ttnn.from_torch(
        state_dict[f"{path}.bias"].unsqueeze(0).unsqueeze(0).unsqueeze(0),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    return weight, bias


def custom_preprocessor(device, state_dict):
    pairs = [
        ("stem.0.conv", False),
        ("stem.0.bn", False),
        ("stem.1.conv_dw", True),
        ("stem.1.conv_pw", True),
        ("stem.1.bn", True),
        ("stem.2.conv_dw", True),
        ("stem.2.conv_pw", True),
        ("stem.2.bn", True),
        ("stages.0.blocks.0.conv_reduction.conv", False),
        ("stages.0.blocks.0.conv_reduction.bn", False),
        ("stages.0.blocks.0.conv_mid.0.conv_dw", True),
        ("stages.0.blocks.0.conv_mid.0.conv_pw", True),
        ("stages.0.blocks.0.conv_mid.0.bn", True),
        ("stages.0.blocks.0.conv_mid.1.conv_dw", True),
        ("stages.0.blocks.0.conv_mid.1.conv_pw", True),
        ("stages.0.blocks.0.conv_mid.1.bn", True),
        ("stages.0.blocks.0.conv_mid.2.conv_dw", True),
        ("stages.0.blocks.0.conv_mid.2.conv_pw", True),
        ("stages.0.blocks.0.conv_mid.2.bn", True),
        ("stages.0.blocks.0.conv_concat.conv", False),
        ("stages.0.blocks.0.conv_concat.bn", False),
        ("stages.1.blocks.0.conv_reduction.conv", False),
        ("stages.1.blocks.0.conv_reduction.bn", False),
        ("stages.1.blocks.0.conv_mid.0.conv_dw", True),
        ("stages.1.blocks.0.conv_mid.0.conv_pw", True),
        ("stages.1.blocks.0.conv_mid.0.bn", True),
        ("stages.1.blocks.0.conv_mid.1.conv_dw", True),
        ("stages.1.blocks.0.conv_mid.1.conv_pw", True),
        ("stages.1.blocks.0.conv_mid.1.bn", True),
        ("stages.1.blocks.0.conv_mid.2.conv_dw", True),
        ("stages.1.blocks.0.conv_mid.2.conv_pw", True),
        ("stages.1.blocks.0.conv_mid.2.bn", True),
        ("stages.1.blocks.0.conv_concat.conv", False),
        ("stages.1.blocks.0.conv_concat.bn", False),
        ("stages.2.blocks.0.conv_reduction.conv", False),
        ("stages.2.blocks.0.conv_reduction.bn", False),
        ("stages.2.blocks.0.conv_mid.0.conv_dw", True),
        ("stages.2.blocks.0.conv_mid.0.conv_pw", True),
        ("stages.2.blocks.0.conv_mid.0.bn", True),
        ("stages.2.blocks.0.conv_mid.1.conv_dw", True),
        ("stages.2.blocks.0.conv_mid.1.conv_pw", True),
        ("stages.2.blocks.0.conv_mid.1.bn", True),
        ("stages.2.blocks.0.conv_mid.2.conv_dw", True),
        ("stages.2.blocks.0.conv_mid.2.conv_pw", True),
        ("stages.2.blocks.0.conv_mid.2.bn", True),
        ("stages.2.blocks.0.conv_concat.conv", False),
        ("stages.2.blocks.0.conv_concat.bn", False),
        ("stages.3.blocks.0.conv_reduction.conv", False),
        ("stages.3.blocks.0.conv_reduction.bn", False),
        ("stages.3.blocks.0.conv_mid.0.conv_dw", True),
        ("stages.3.blocks.0.conv_mid.0.conv_pw", True),
        ("stages.3.blocks.0.conv_mid.0.bn", True),
        ("stages.3.blocks.0.conv_mid.1.conv_dw", True),
        ("stages.3.blocks.0.conv_mid.1.conv_pw", True),
        ("stages.3.blocks.0.conv_mid.1.bn", True),
        ("stages.3.blocks.0.conv_mid.2.conv_dw", True),
        ("stages.3.blocks.0.conv_mid.2.conv_pw", True),
        ("stages.3.blocks.0.conv_mid.2.bn", True),
        ("stages.3.blocks.0.conv_concat.conv", False),
        ("stages.3.blocks.0.conv_concat.bn", False),
    ]

    parameters = {}

    for path, seperable_conv_norm in pairs:
        if "bn" in path:
            (
                parameters[path + ".weight"],
                parameters[path + ".bias"],
                parameters[path + ".running_mean"],
                parameters[path + ".running_var"],
            ) = preprocess_bn_weight(state_dict, path=path, device=device)

        else:
            parameters[path + ".weight"], parameters[path + ".bias"] = preprocess_parameters(
                state_dict, path=path, seperable_conv_norm=seperable_conv_norm
            )

    # Detect

    # parameters["stem.1.conv_dw.weight"],parameters["stem.1.conv_dw.bias"] = preprocess_parameters(state_dict, "stem.1", seperable_conv_norm=True)
    # parameters["stem.2.conv_dw.weight"],parameters["stem.2.conv_dw.bias"] = preprocess_parameters(state_dict, "stem.2", seperable_conv_norm=True)

    # parameters["stages.0.blocks.0.conv_mid.0.conv_dw.weight"],parameters["stages.0.blocks.0.conv_mid.0.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.0.blocks.0.conv_mid.0", seperable_conv_norm=True)
    # parameters["stages.0.blocks.0.conv_mid.1.conv_dw.weight"],parameters["stages.0.blocks.0.conv_mid.1.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.0.blocks.0.conv_mid.1", seperable_conv_norm=True)
    # parameters["stages.0.blocks.0.conv_mid.2.conv_dw.weight"],parameters["stages.0.blocks.0.conv_mid.2.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.0.blocks.0.conv_mid.2", seperable_conv_norm=True)

    # parameters["stages.1.blocks.0.conv_mid.0.conv_dw.weight"],parameters["stages.1.blocks.0.conv_mid.0.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.1.blocks.0.conv_mid.0", seperable_conv_norm=True)
    # parameters["stages.1.blocks.0.conv_mid.1.conv_dw.weight"],parameters["stages.1.blocks.0.conv_mid.1.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.1.blocks.0.conv_mid.1", seperable_conv_norm=True)
    # parameters["stages.1.blocks.0.conv_mid.2.conv_dw.weight"],parameters["stages.1.blocks.0.conv_mid.2.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.1.blocks.0.conv_mid.2", seperable_conv_norm=True)

    # parameters["stages.2.blocks.0.conv_mid.0.conv_dw.weight"],parameters["stages.2.blocks.0.conv_mid.0.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.2.blocks.0.conv_mid.0", seperable_conv_norm=True)
    # parameters["stages.2.blocks.0.conv_mid.1.conv_dw.weight"],parameters["stages.2.blocks.0.conv_mid.1.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.2.blocks.0.conv_mid.1", seperable_conv_norm=True)
    # parameters["stages.2.blocks.0.conv_mid.2.conv_dw.weight"],parameters["stages.2.blocks.0.conv_mid.2.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.2.blocks.0.conv_mid.2", seperable_conv_norm=True)

    # parameters["stages.3.blocks.0.conv_mid.0.conv_dw.weight"],parameters["stages.3.blocks.0.conv_mid.0.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.3.blocks.0.conv_mid.0", seperable_conv_norm=True)
    # parameters["stages.3.blocks.0.conv_mid.1.conv_dw.weight"],parameters["stages.3.blocks.0.conv_mid.1.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.3.blocks.0.conv_mid.1", seperable_conv_norm=True)
    # parameters["stages.3.blocks.0.conv_mid.2.conv_dw.weight"],parameters["stages.3.blocks.0.conv_mid.2.conv_dw.bias"] = preprocess_parameters(state_dict, "stages.3.blocks.0.conv_mid.2", seperable_conv_norm=True)

    parameters["stages.0.blocks.0.attn.weight"], parameters["stages.0.blocks.0.attn.bias"] = preprocess_parameters(
        state_dict, "stages.0.blocks.0.attn", seperable_conv_norm=False, effective_se=True
    )
    parameters["stages.1.blocks.0.attn.weight"], parameters["stages.1.blocks.0.attn.bias"] = preprocess_parameters(
        state_dict, "stages.1.blocks.0.attn", seperable_conv_norm=False, effective_se=True
    )
    parameters["stages.2.blocks.0.attn.weight"], parameters["stages.2.blocks.0.attn.bias"] = preprocess_parameters(
        state_dict, "stages.2.blocks.0.attn", seperable_conv_norm=False, effective_se=True
    )
    parameters["stages.3.blocks.0.attn.weight"], parameters["stages.3.blocks.0.attn.bias"] = preprocess_parameters(
        state_dict, "stages.3.blocks.0.attn", seperable_conv_norm=False, effective_se=True
    )
    parameters["head.fc.weight"], parameters["head.fc.bias"] = preprocess_linear_weight(
        state_dict, "head.fc", device=device
    )

    return parameters


"""
def create_vovnet_model_parameters(model, device):
    parameters = preprocess_model_parameters(
        initialize_model=lambda: model,
        custom_preprocessor=custom_preprocessor,
        device=None,
    )

    return parameters
"""

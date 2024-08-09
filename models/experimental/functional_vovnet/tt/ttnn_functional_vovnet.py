# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import tt_lib
import torch.nn as nn
from models.experimental.functional_vovnet.reference import torch_functional_vovnet
import math

from models.experimental.vovnet.vovnet_utils import *
import torch.nn.functional as F


def fold_bn_to_conv_weights_bias(model, path, conv="conv"):
    bn_weight = model[path + ".bn.weight"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_running_var = model[path + ".bn.running_var"].unsqueeze(1).unsqueeze(1).unsqueeze(1)

    weight = model[path + f".{conv}.weight"]
    weight = (weight / torch.sqrt(bn_running_var)) * bn_weight

    bn_running_mean = model[path + ".bn.running_mean"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bn_bias = model[path + ".bn.bias"].unsqueeze(1).unsqueeze(1).unsqueeze(1)
    bias = -(bn_weight) * (bn_running_mean / torch.sqrt(bn_running_var)) + bn_bias

    bias = bias.reshape(1, 1, 1, -1)
    return (
        ttnn.from_torch(
            weight,
        ),
        ttnn.from_torch(bias),
    )


def conv(
    device,
    input_tensor,
    model,
    path,
    input_params,
    conv_params,
    *,
    act_block_h=64,
    reshard=False,
    deallocate=False,
    height_sharding=True,
    activation="",
    fused_op=True,
    use_shallow_conv_variant=False,
    fp32_accum=False,
    packer_l1_acc=False,
    enable_act_double_buffer=False,
    enable_split_reader=False,
    enable_subblock_padding=False,
    reallocate_halo_output=True,
    debug=False,
    groups=1,
    bias=False,
    conv=f"conv",
    reshard_if_not_optimal=False,
):
    if fused_op:
        weights, bias = fold_bn_to_conv_weights_bias(model, path, conv)
    else:
        weight = model[path + ".weight"]
        weights = ttnn.from_torch(weight)
        if bias:
            bias = model[path + ".bias"]
            bias = bias.reshape(1, 1, 1, -1)
            bias = ttnn.from_torch(bias)
    kernel_size = (weights.shape[2], weights.shape[3])
    out_channels = weights.shape[0]
    reader_patterns_cache = {}

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat8_b,
        math_fidelity=ttnn.MathFidelity.LoFi,
        activation=activation,
        height_sharding=height_sharding,
        input_channels_alignment=16 if use_shallow_conv_variant or input_params[3] < 16 else 32,
        transpose_shards=True if deallocate else False,
        # reshard_if_not_optimal=reshard,
        deallocate_activation=deallocate,
        fp32_dest_acc_enabled=fp32_accum,
        packer_l1_accum_enabled=packer_l1_acc,
        enable_act_double_buffer=enable_act_double_buffer,
        enable_split_reader=enable_split_reader,
        enable_subblock_padding=enable_subblock_padding,
    )
    if act_block_h is not None:
        conv_config.act_block_h_override = act_block_h

    [output_tensor, _out_height, _out_width, weights, bias] = ttnn.conv2d(
        input_tensor=input_tensor,
        weight_tensor=weights,
        in_channels=input_params[3],
        out_channels=out_channels,
        device=device,
        bias_tensor=None if not bias else bias,
        kernel_size=kernel_size,
        stride=(conv_params[0], conv_params[1]),
        padding=(conv_params[2], conv_params[3]),
        batch_size=input_params[0],
        input_height=input_params[1],
        input_width=input_params[2],
        conv_config=conv_config,
        conv_op_cache=reader_patterns_cache,
        debug=debug,
        groups=groups,
    )

    return output_tensor, _out_height, _out_width


def effective_se_module(
    device,
    torch_model,
    path,
    input_tensor,
    input_params,
    conv_params,
    parameters,
    add_maxpool=False,
    debug=False,
    bias=False,
):
    x_se = ttnn.mean(input_tensor, (2, 3), keepdim=True)
    if add_maxpool:
        # experimental codepath, may remove or change
        x_se = 0.5 * x_se + 0.5 * input_tensor.amax((2, 3), keepdim=True)

    x_se_torch = ttnn.to_torch(x_se)
    x_se_permute = x_se_torch.permute(0, 2, 3, 1)
    x_se_permute = ttnn.from_torch(x_se_permute, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

    if path == "stages.3.blocks.0.attn":
        x_se_permute = ttnn.to_torch(x_se_permute).permute(0, 3, 1, 2)
        conv1 = nn.Conv2d(
            int(1024 * 1.0),
            1024,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=True,
        )
        conv1.weight = parameters.fc.weight
        x_se = conv1(x_se_permute.float())
        x_se = ttnn.from_torch(x_se, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    else:
        x_se, _x_se_h, _x_se_w = conv(
            device,
            x_se_permute,
            torch_model,
            f"{path}.fc",
            x_se_permute.shape,
            conv_params,
            act_block_h=None,
            reshard=False,
            deallocate=True,
            height_sharding=True,
            activation="",
            fused_op=False,
            debug=debug,
            bias=bias,
            reallocate_halo_output=False,
        )

        x_se_torch = ttnn.to_torch(x_se)
        x_se = x_se_torch.permute(0, 3, 1, 2)
        x_se = ttnn.from_torch(x_se, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    input_tensor = ttnn.to_layout(input_tensor, layout=ttnn.TILE_LAYOUT)
    y = input_tensor * ttnn.hardsigmoid(x_se)

    return y


def conv_norm_act(
    device,
    x,
    torch_model,
    path,
    input_params,
    conv_params,
    debug=False,
    bias=False,
    reshard=False,
    act_block_h=None,
    activation="",
    # act_block_h=64,
):
    if x.layout == ttnn.TILE_LAYOUT:
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    x, _x_h, _x_w = conv(
        device,
        x,
        torch_model,
        path,
        input_params,
        conv_params,
        act_block_h=act_block_h,
        reshard=reshard,
        deallocate=False,
        height_sharding=True,
        activation=activation,
        fused_op=True,
        debug=debug,
        bias=bias,
        reallocate_halo_output=True,
    )

    return x, _x_h, _x_w


def seperable_conv_norm_act(
    device,
    x,
    torch_model,
    path,
    input_params,
    conv_params1,
    conv_params2,
    debug,
    groups,
    bias=False,
    act_block_h=None,
):
    x, _x_h, _x_w = conv(
        device,
        x,
        torch_model,
        f"{path}.conv_dw",
        x.shape,
        conv_params1,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="",
        fused_op=False,
        debug=debug,
        groups=groups,
        bias=False,
    )

    x, _x1_h, _x1_w = conv(
        device,
        x,
        torch_model,
        f"{path}",
        x.shape,
        conv_params2,
        act_block_h=None,
        reshard=False,
        deallocate=True,
        height_sharding=True,
        activation="relu",
        fused_op=True,
        debug=debug,
        conv=f"conv_pw",
        bias=False,
    )

    return x, _x_h, _x_w


def sequential_append_list(
    device,
    input_tensor,
    torch_model,
    path,
    concat_list,
    conv_params1,
    conv_params2,
    debug,
    groups,
    layers_per_block,
    bias=False,
    act_block_h=None,
):
    input_tensor = ttnn.from_device(input_tensor)
    input_tensor = ttnn.to_layout(input_tensor, ttnn.ROW_MAJOR_LAYOUT)
    for i in range(layers_per_block):
        conv, conv_h, conv_w = seperable_conv_norm_act(
            device=device,
            x=input_tensor if i == 0 else conv,
            torch_model=torch_model,
            path=f"{path}.{i}",
            input_params=input_tensor.shape if i == 0 else conv.shape,
            conv_params1=conv_params1,
            conv_params2=conv_params2,
            debug=False,
            groups=groups,
            bias=False,
            # act_block_h=None,
        )
        conv = ttnn.to_layout(conv, ttnn.ROW_MAJOR_LAYOUT)
        conv = ttnn.reshape(conv, (1, conv_h, conv_w, conv.shape[-1]))
        concat_list.append(conv)
        conv = ttnn.from_device(conv)

    for i in range(len(concat_list)):
        concat_list[i] = ttnn.to_device(concat_list[i], device)
        concat_list[i] = ttnn.to_memory_config(concat_list[i], memory_config=ttnn.L1_MEMORY_CONFIG)

    seq_out = ttnn.concat(concat_list, dim=-1)
    seq_out = seq_out.reshape(1, conv_h, conv_w, seq_out.shape[-1])

    return seq_out, conv_h, conv_w


def osa_block(
    device,
    x,
    torch_model,
    path,
    parameters,
    model,
    groups,
    conv_norm_act_params,
    conv_params1,
    conv_params2,
    layers_per_block=3,
    residual=True,
    depthwise=True,
    debug=False,
    bias=False,
    act_block_h=64,
):
    outputs = [x]

    if depthwise:
        assert not residual
        x, x_h, x_w = conv_norm_act(
            device=device,
            x=x,
            torch_model=torch_model,
            path=f"{path}.conv_reduction",
            input_params=x.shape,
            conv_params=conv_norm_act_params,
            debug=debug,
            bias=bias,
            activation="relu",
        )
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

    x = x.reshape(1, x_h, x_w, x.shape[-1])
    x, x_h, x_w = sequential_append_list(
        device=device,
        input_tensor=x,
        torch_model=torch_model,
        path=f"{path}.conv_mid",
        concat_list=outputs,
        conv_params1=conv_params1,
        conv_params2=conv_params2,
        debug=False,
        groups=groups,
        layers_per_block=layers_per_block,
        bias=False,
    )

    # Statically allocated circular buffer for the convs below mentioned stages in conv_norm_act.
    if path == "stages.2.blocks.0" or path == "stages.3.blocks.0":
        x = ttnn.to_torch(x).permute(0, 3, 1, 2)
        x = torch_functional_vovnet.conv_norm_act(
            x=x.float(),
            in_channels=1088 if path == "stages.2.blocks.0" else 1440,
            out_channels=768 if path == "stages.2.blocks.0" else 1024,
            parameters=parameters.conv_concat,
            running_mean=model.conv_concat.bn.running_mean,
            running_var=model.conv_concat.bn.running_var,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            channel_multiplier=1.0,
            apply_act="True",
        )
        x = ttnn.from_torch(
            x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )  # , layout=ttnn.TILE_LAYOUT, device=device)

    else:
        x = ttnn.from_device(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x, x_h, x_w = conv_norm_act(
            device=device,
            x=x,
            torch_model=torch_model,
            path=f"{path}.conv_concat",
            input_params=x.shape,
            conv_params=[1, 1, 0, 0],
            debug=debug,
            bias=bias,
            activation="relu",
        )
        x = x.reshape(1, x_h, x_w, x.shape[-1])

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x_torch = ttnn.to_torch(x)
        x = x_torch.permute(0, 3, 1, 2)
        x = ttnn.from_torch(x, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)

    x = effective_se_module(
        device=device,
        torch_model=torch_model,
        path=f"{path}.attn",
        input_tensor=x,
        input_params=x.shape,
        conv_params=conv_norm_act_params,
        parameters=parameters.attn,
        debug=debug,
        bias=True,
    )

    return x


def osa_stage(
    device,
    x,
    torch_model,
    path,
    parameters,
    model,
    groups,
    block_per_stage=1,
    layer_per_block=3,
    residual=False,
    depthwise=True,
    debug=False,
    bias=False,
    downsample=False,
):
    x = ttnn.to_device(x, device)
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    if downsample:
        # Output Shape Mismatch with ttnn.MaxPool2d as ceil_mode is true in torch.MaxPool2d

        # in_n, in_c, in_h, in_w = x.shape
        # x_shape = (in_n, 1, in_h * in_w, in_c)
        # x_torch = ttnn.to_torch(x)
        # x_permuted = torch.permute(x_torch, (0, 2, 3, 1))
        # x_permuted = ttnn.from_torch(x_permuted, dtype=ttnn.bfloat16)
        # x_shape = (x.shape[0], 1, x.shape[2] * x.shape[3], x.shape[1])
        # x_reshaped = x_permuted.reshape(x_shape)

        # max_pool_reader_patterns_cache = {}
        # max_pool_parallel_config_override = {}
        # maxpool = ttnn.MaxPool2d(
        #     kernel_size=(3, 3),
        #     stride=(2, 2),
        #     padding=(0, 0),
        #     dilation=(1, 1),
        #     dtype=ttnn.bfloat16,
        #     device=device,
        #     batch_size=in_n,
        #     input_height=int(math.sqrt(x_reshaped.shape[-2])),
        #     input_width=int(math.sqrt(x_reshaped.shape[-2])),
        #     deallocate_activation=True,
        #     parallel_config_override=max_pool_parallel_config_override,
        #     reader_patterns_cache=max_pool_reader_patterns_cache,
        #     channels=in_c,
        # )

        # ttact_d = maxpool.copy_input_to_device(x_reshaped)
        # tt_x = maxpool(ttact_d)
        # ttnn.deallocate(ttact_d)
        # x = maxpool.copy_output_from_device(tt_x)
        # max_pool_reader_patterns_cache.clear()
        # max_pool_parallel_config_override.clear()

        pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        x_torch = ttnn.to_torch(x)
        x_torch = pool(x_torch)
        x_torch_permute = x_torch.permute(0, 2, 3, 1)
        x = ttnn.from_torch(x_torch_permute, dtype=ttnn.bfloat16, device=device)

    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.from_device(x)

    x = osa_block(
        device=device,
        x=x,
        torch_model=torch_model,
        path=f"{path}.blocks.0",
        parameters=parameters.blocks[0],
        model=model.blocks[0],
        groups=groups,
        conv_norm_act_params=[1, 1, 0, 0],
        conv_params1=[1, 1, 1, 1],
        conv_params2=[1, 1, 0, 0],
        layers_per_block=3,
        residual=False,
        depthwise=True,
        debug=debug,
        bias=bias,
    )

    return x


def classifier_head(
    device,
    x,
    torch_model,
    path,
):
    x = ttnn.global_avg_pool2d(x)

    weights_tensor = torch_model[path + ".fc.weight"]
    weights_tensor = weights_tensor.permute(1, 0)
    weights_tensor = ttnn.from_torch(weights_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

    bias_tensor = torch_model[path + ".fc.bias"]
    bias_tensor = ttnn.from_torch(bias_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    x = ttnn.linear(
        x,
        weights_tensor,
        bias=bias_tensor,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )

    return x


def vovnet(
    device,
    x,
    torch_model,
    parameters,
    model,
    block_per_stage=1,
    residual=False,
    depthwise=True,
    debug=False,
    bias=False,
):
    x, x_h, x_w = conv_norm_act(
        device=device,
        x=x,
        torch_model=torch_model,
        path=f"stem.0",
        input_params=x.shape,
        conv_params=[2, 2, 1, 1],
        debug=debug,
        bias=bias,  # bias,
        activation="relu",
    )

    x = x.reshape(1, x_h, x_w, x.shape[-1])
    x, x_h, x_w = seperable_conv_norm_act(
        device=device,
        x=x,
        torch_model=torch_model,
        path=f"stem.1",
        input_params=x.shape,
        conv_params1=[1, 1, 1, 1],
        conv_params2=[1, 1, 0, 0],
        debug=debug,
        groups=64,
        bias=bias,
    )
    x = x.reshape(1, x_h, x_w, x.shape[-1])
    x, x_h, x_w = seperable_conv_norm_act(
        device=device,
        x=x,
        torch_model=torch_model,
        path=f"stem.2",
        input_params=x.shape,
        conv_params1=[2, 2, 1, 1],
        conv_params2=[1, 1, 0, 0],
        debug=debug,
        groups=64,
        bias=bias,
    )

    x = x.reshape(1, x_h, x_w, x.shape[-1])
    groups = [128, 160, 192, 224]
    for i in range(4):
        x = osa_stage(
            device=device,
            x=x,
            torch_model=torch_model,
            path=f"stages.{i}",
            parameters=parameters.stages[i],
            model=model.stages[i],
            block_per_stage=block_per_stage,
            groups=groups[i],
            residual=True,
            depthwise=True,
            debug=debug,
            bias=bias,
            downsample=False if i == 0 else True,
        )

    x_torch = ttnn.to_torch(x)
    x_torch = x_torch.permute(0, 2, 3, 1)
    x_permute = ttnn.from_torch(x_torch, dtype=ttnn.bfloat16, device=device)

    x4 = classifier_head(
        device=device,
        x=x_permute,
        torch_model=torch_model,
        path=f"head",
    )

    return x4

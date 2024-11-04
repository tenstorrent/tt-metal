# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import torch.nn.functional as F
import types
import functools
import timm
from typing import Optional, Tuple, Union, Type, List
from models.experimental.vovnet.vovnet_utils import *


def effective_se_module(x, channels, parameters, add_maxpool=False):
    add_maxpool = add_maxpool
    fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    fc.weight = parameters.fc.weight
    fc.bias = parameters.fc.bias
    gate_layer = nn.Hardsigmoid()

    x_se = x.mean((2, 3), keepdim=True)

    if add_maxpool:
        x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
    x_se = fc(x_se)
    output = x * gate_layer(x_se)

    return output


def batch_norm_act_2d(
    x,
    num_features,
    running_mean,
    running_var,
    parameters,
    eps,
    momentum,
    affine=True,
    track_running_stats=True,
    apply_act=True,
    act_kwargs={},
    inplace=True,
):
    act_layer = nn.ReLU
    act = create_act(act_layer, act_kwargs=act_kwargs, inplace=inplace, apply_act=apply_act)

    assert x.ndim == 4, f"expected 4D input (got {x.ndim}D input)"

    if momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = momentum

    x = F.batch_norm(
        x,
        running_mean,
        running_var,
        parameters.weight,
        parameters.bias,
        False,
        exponential_average_factor,
        eps,
    )
    x = act(x)

    return x


def seperable_conv_norm_act(
    x,
    in_channels,
    out_channels,
    parameters,
    running_mean,
    running_var,
    groups,
    kernel_size,
    stride,
    padding="",
    bias=False,
    channel_multiplier=1,
    pw_kernel_size=1,
    apply_act=True,
):
    norm_layer = nn.BatchNorm2d
    act_layer = nn.ReLU
    conv_dw = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=1,
        bias=False,
        groups=in_channels,
    )
    conv_dw.weight = parameters.conv_dw.weight

    conv_pw = nn.Conv2d(int(in_channels * channel_multiplier), out_channels, pw_kernel_size, stride=1, bias=False)
    conv_pw.weight = parameters.conv_pw.weight

    x = conv_dw(x)
    x = conv_pw(x)

    momentum = 0.1

    assert x.ndim == 4, f"expected 4D input (got {x.ndim}D input)"

    if momentum is None:
        exponential_average_factor = 0.0
    else:
        exponential_average_factor = momentum

    x = batch_norm_act_2d(
        x,
        out_channels,
        running_mean,
        running_var,
        parameters.bn,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_kwargs=None,
        inplace=True,
    )

    return x


def conv_norm_act(
    x,
    in_channels,
    out_channels,
    parameters,
    running_mean,
    running_var,
    kernel_size,
    stride,
    padding,
    dilation,
    bias,
    channel_multiplier,
    apply_act,
    groups,
):
    norm_layer = nn.BatchNorm2d
    act_layer = nn.ReLU
    conv = nn.Conv2d(
        int(in_channels * channel_multiplier),
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=1,
        bias=False,
    )
    conv.weight = parameters.conv.weight
    x = conv(x)

    x = batch_norm_act_2d(
        x,
        out_channels,
        running_mean,
        running_var,
        parameters.bn,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        apply_act=True,
        act_kwargs=None,
        inplace=True,
    )

    return x


def sequential_append_list(
    x,
    parameters,
    running_mean,
    running_var,
    concat_list,
    in_channels,
    layer_per_block,
    groups,
):
    layer_per_block = layer_per_block
    for i in range(layer_per_block):
        if i == 0:
            rm = running_mean[i].bn.running_mean
            rv = running_var[i].bn.running_var
            conv = seperable_conv_norm_act(
                x,
                in_channels,
                in_channels,
                parameters[i],
                rm,
                rv,
                groups,
                kernel_size=3,
                stride=1,
                padding="",
                bias=False,
                channel_multiplier=1.0,
                pw_kernel_size=1,
                apply_act=True,
            )
            concat_list.append(conv)

        else:
            rm = running_mean[i].bn.running_mean
            rv = running_var[i].bn.running_var
            conv = seperable_conv_norm_act(
                concat_list[-1],
                in_channels,
                in_channels,
                parameters[i],
                rm,
                rv,
                groups,
                kernel_size=3,
                stride=1,
                padding="",
                bias=False,
                channel_multiplier=1.0,
                pw_kernel_size=1,
                apply_act=True,
            )
            concat_list.append(conv)

        x = torch.cat(concat_list, dim=1)

    return x


def select_adaptive_pool2d(
    x,
    output_size,
    pool_type,
    flatten,
    input_fmt,
):
    assert input_fmt in ("NCHW", "NHWC")
    pool_type = pool_type or ""
    if not pool_type:
        pool = nn.Identity()
        flatten_layer = nn.Flatten(1) if flatten else nn.Identity()
    else:
        assert input_fmt == "NCHW"
        if pool_type == "max":
            pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            pool = nn.AdaptiveAvgPool2d(output_size)
        flatten_layer = nn.Flatten(1) if flatten else nn.Identity()

    x = pool(x)
    x = flatten_layer(x)
    x = torch.reshape(x, (1, 1024))
    return x


def classifier_head(x, parameters, in_features, num_classes, pool_type, use_conv, input_fmt, pre_logits):
    x = create_classifier(
        x,
        parameters,
        in_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    flatten = nn.Flatten(1) if use_conv and pool_type else nn.Identity()

    if pre_logits:
        x = flatten

    x = x @ parameters.fc.weight
    x += parameters.fc.bias
    x = flatten(x)

    return x


def create_classifier(
    x,
    parameters,
    num_features,
    num_classes,
    pool_type,
    use_conv,
    input_fmt,
):
    global_pool = _create_pool(
        x,
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )

    return global_pool


def _create_pool(
    x,
    num_features,
    num_classes,
    pool_type,
    use_conv,
    input_fmt,
):
    flatten_in_pool = not use_conv
    if not pool_type:
        assert num_classes == 0 or use_conv
        flatten_in_pool = False

    global_pool = select_adaptive_pool2d(
        x,
        output_size=1,
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    return global_pool


def osa_block(
    x,
    parameters,
    running_mean,
    running_var,
    in_chs,
    mid_chs,
    out_chs,
    layer_per_block,
    residual=False,
    depthwise=True,
):
    output = [x]
    next_in_chs = in_chs

    rm_conv_red = running_mean.conv_reduction.bn.running_mean
    rv_conv_red = running_mean.conv_reduction.bn.running_var
    print(x.shape)
    return x
    if depthwise and next_in_chs != mid_chs:
        assert not residual
        print("!!!!!!!!!!!!!!!!!")
        x = conv_norm_act(
            x,
            next_in_chs,
            mid_chs,
            parameters.conv_reduction,
            rm_conv_red,
            rv_conv_red,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            channel_multiplier=1.0,
            apply_act=True,
            groups=1,
        )
    else:
        conv_reduction = None

    rm_conv_mid = running_mean.conv_mid
    rv_conv_mid = running_var.conv_mid

    x = sequential_append_list(
        x,
        parameters.conv_mid,
        rm_conv_mid,
        rv_conv_mid,
        output,
        in_channels=mid_chs,
        layer_per_block=3,
        groups=mid_chs,
    )

    next_in_chs = next_in_chs = in_chs + layer_per_block * mid_chs

    rm_conv_concat = running_mean.conv_concat.bn.running_mean
    rv_conv_concat = running_var.conv_concat.bn.running_var

    x = conv_norm_act(
        x,
        next_in_chs,
        out_chs,
        parameters.conv_concat,
        rm_conv_concat,
        rv_conv_concat,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
        channel_multiplier=1.0,
        apply_act=True,
        groups=1,
    )

    x = effective_se_module(x, out_chs, parameters.attn, add_maxpool=False)

    if residual:
        x = x + output[0]

    return x


def osa_stage(
    x,
    parameters,
    running_mean,
    running_var,
    in_chs,
    mid_chs,
    out_chs,
    block_per_stage,
    layer_per_block,
    downsample,
    residual=True,
    depthwise=True,
):
    if downsample:
        pool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        x = pool(x)

    rm = running_mean.blocks[0]
    rv = running_var.blocks[0]

    x = osa_block(
        x,
        parameters.blocks[0],
        rm,
        rv,
        in_chs,
        mid_chs,
        out_chs,
        layer_per_block,
        residual=False,
        depthwise=True,
    )

    return x


def vovnet(
    cfg,
    x,
    parameters,
    running_mean,
    running_var,
    in_chans=3,
    num_classes=1000,
    global_pool="avg",
    output_stride=32,
    stem_stride=4,
    depthwise=True,
):
    assert stem_stride in (4, 2)
    assert output_stride == 32

    stem_chs = cfg["stem_chs"]
    stage_conv_chs = cfg["stage_conv_chs"]
    stage_out_chs = cfg["stage_out_chs"]
    block_per_stage = cfg["block_per_stage"]
    layer_per_block = cfg["layer_per_block"]
    last_stem_stride = stem_stride // 2

    rm = running_mean.stem[0].bn.running_mean
    rv = running_var.stem[0].bn.running_var
    x = conv_norm_act(
        x,
        in_chans,
        stem_chs[0],
        parameters.stem[0],
        rm,
        rv,
        kernel_size=3,
        stride=2,
        padding=1,
        dilation=1,
        bias=False,
        channel_multiplier=1,
        apply_act=True,
        groups=1,
    )
    if depthwise:
        for i in range(0, 2):
            if i == 0:
                stride = 1
            else:
                stride = 2

            x = seperable_conv_norm_act(
                x,
                stem_chs[i],
                stem_chs[i + 1],
                parameters.stem[i + 1],
                running_mean.stem[i + 1].bn.running_mean,
                running_var.stem[i + 1].bn.running_var,
                stem_chs[i],
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
                channel_multiplier=1,
                pw_kernel_size=1,
                apply_act=True,
            )

    feature_info = [dict(num_chs=stem_chs[1], reduction=2, module=f"stem.{1 if stem_stride == 4 else 2}")]
    current_stride = stem_stride
    stage_dpr = torch.split(torch.linspace(0, 0, sum(block_per_stage)), block_per_stage)
    in_ch_list = stem_chs[-1:] + stage_out_chs[:-1]
    stage_args = dict(residual=cfg["residual"], depthwise=cfg["depthwise"], attn=cfg["attn"])

    for i in range(4):
        downsample = stem_stride == 2 or i > 0

        x = osa_stage(
            x,
            parameters.stages[i],
            running_mean.stages[i],
            running_var.stages[i],
            in_ch_list[i],
            stage_conv_chs[i],
            stage_out_chs[i],
            block_per_stage[i],
            layer_per_block,
            downsample=downsample,
            residual=True,
            depthwise=True,
        )

    num_features = stage_out_chs[i]
    current_stride *= 2 if downsample else 1
    feature_info += [dict(num_chs=num_features, reduction=current_stride, module=f"stages.{i}")]

    x = classifier_head(
        x,
        parameters.head,
        num_features,
        num_classes,
        pool_type="avg",
        use_conv=False,
        input_fmt="NCHW",
        pre_logits=False,
    )

    return x

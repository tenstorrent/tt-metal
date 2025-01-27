# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import math
import torch.nn as nn

from models.experimental.functional_yolov8x.tt.ttnn_yolov8x_utils import (
    autopad,
    ttnn_decode_bboxes,
)


def Conv(
    device,
    x,
    parameters,
    path,
    in_h,
    in_w,
    k=1,
    s=1,
    p=None,
    g=1,
    d=1,
    act_block_h=False,
    block_shard=None,
    bfloat8=True,
    change_shard=False,
    deallocate_activation=False,
    output_layout=ttnn.TILE_LAYOUT,
    is_fused=True,
    is_dfl=False,
    is_detect_cv2=False,
):
    p = autopad(k, p, d)

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=(16 if False or (x.shape[-1] == 16 and x.shape[-2] == 115) else 32),
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=output_layout,
    )

    if deallocate_activation:
        conv_config.deallocate_activation = deallocate_activation

    if change_shard:
        conv_config.shard_layout = None

    if act_block_h:
        conv_config.act_block_h_override = 32

    if block_shard:
        conv_config.shard_layout = ttnn.TensorMemoryLayout.BLOCK_SHARDED

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    if is_fused:
        fused_weight, fused_bias = parameters[path]
    else:
        conv_weight, conv_bias = parameters[path]

    if bfloat8:
        conv_config.weights_dtype = ttnn.bfloat8_b

    [x, [out_height, out_width]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=fused_weight if is_fused else conv_weight,
        in_channels=fused_weight.shape[1] if is_fused else conv_weight.shape[1],
        out_channels=fused_weight.shape[0] if is_fused else conv_weight.shape[0],
        device=device,
        bias_tensor=fused_bias if is_fused else conv_bias,
        kernel_size=(k, k),
        stride=(s, s),
        padding=(p, p),
        dilation=(d, d),
        batch_size=x.shape[0],
        input_height=in_h,
        input_width=in_w,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=False,
        groups=g,
        memory_config=None,
        return_weights_and_bias=False,
        return_output_dim=True,
    )

    if is_dfl:
        return ttnn.reshape(x, (x.shape[0], 4, -1))

    if is_detect_cv2:
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        return (x, out_height, out_width)

    x = ttnn.silu(x)

    return (x, out_height, out_width)


def Bottleneck(
    device,
    x,
    parameters,
    path,
    in_h,
    in_w,
    shortcut=True,
    g=1,
    k=(3, 3),
    e=0.5,
    act_block_h=True,
    change_shard=None,
    deallocate_activation=False,
    output_layout=ttnn.TILE_LAYOUT,
):
    cv1, out_h, out_w = Conv(
        device,
        x,
        parameters,
        f"{path}.cv1",
        in_h,
        in_w,
        k[0][0],
        1,
        act_block_h=act_block_h,
        change_shard=change_shard,
        deallocate_activation=deallocate_activation,
        output_layout=output_layout,
    )

    cv2, out_h, out_w = Conv(
        device,
        cv1,
        parameters,
        f"{path}.cv2",
        in_h,
        in_w,
        k[1][1],
        1,
        g=g,
        act_block_h=act_block_h,
        deallocate_activation=deallocate_activation,
    )

    ttnn.deallocate(cv1)

    cv2 = ttnn.sharded_to_interleaved(cv2, ttnn.L1_MEMORY_CONFIG)
    cv2 = ttnn.reshape(cv2, (1, out_h, out_w, cv2.shape[-1]))

    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT, device=device)
    cv2 = ttnn.to_layout(cv2, ttnn.TILE_LAYOUT, device=device)

    add = shortcut

    return x + cv2 if add else cv2


def C2f(
    device,
    x,
    parameters,
    path,
    in_h,
    in_w,
    n=1,
    shortcut=False,
    g=1,
    e=0.5,
    act_block_h=False,
    bfloat8=True,
    block_shard=False,
    change_shard=None,
    deallocate_activation=False,
    output_layout=ttnn.ROW_MAJOR_LAYOUT,
):
    cv1, out_h, out_w = Conv(
        device,
        x,
        parameters,
        f"{path}.cv1",
        in_h,
        in_w,
        1,
        1,
        bfloat8=bfloat8,
        change_shard=change_shard,
        deallocate_activation=deallocate_activation,
        output_layout=output_layout,
    )

    cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)
    cv1 = ttnn.reshape(cv1, (1, out_h, out_w, cv1.shape[-1]))

    y = list(ttnn.split(cv1, 2, 3))

    for i in range(n):
        z = Bottleneck(
            device,
            y[-1],
            parameters,
            f"{path}.m.{i}",
            out_h,
            out_w,
            shortcut,
            g,
            k=((3, 3), (3, 3)),
            e=1.0,
            act_block_h=act_block_h,
            change_shard=change_shard,
            deallocate_activation=deallocate_activation,
        )
        y.append(z)

    y[0] = ttnn.to_layout(y[0], layout=ttnn.TILE_LAYOUT)
    y[1] = ttnn.to_layout(y[1], layout=ttnn.TILE_LAYOUT)

    x = ttnn.concat(y, 3)

    for i in range(len(y)):
        ttnn.deallocate(y[i])

    x, out_h, out_w = Conv(
        device,
        x,
        parameters,
        f"{path}.cv2",
        out_h,
        out_w,
        1,
        bfloat8=bfloat8,
        block_shard=block_shard,
        change_shard=change_shard,
        deallocate_activation=deallocate_activation,
    )
    return x, out_h, out_w


def SPPF(device, x, parameters, path, in_h, in_w, k=5):
    cv1, out_h, out_w = Conv(device, x, parameters, f"{path}.cv1", in_h, in_w, 1, 1)

    cv1 = ttnn.sharded_to_interleaved(cv1, ttnn.L1_MEMORY_CONFIG)
    cv1 = ttnn.reshape(cv1, (1, out_h, out_w, cv1.shape[-1]))
    cv1 = ttnn.permute(cv1, (0, 3, 1, 2))

    # tt maxpool2d low pcc case : submitted unit test.

    cv1 = ttnn.to_torch(cv1)
    m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    y = [cv1]
    y.extend(m(y[-1]) for _ in range(3))

    tt_y = []
    for i in range(len(y)):
        y[i] = ttnn.from_torch(y[i], device=device)
        y[i] = ttnn.permute(y[i], (0, 2, 3, 1))
        tt_y.append(y[i])

    x = ttnn.concat(tt_y, 3)

    for i in range(len(y)):
        ttnn.deallocate(tt_y[i])

    x, out_h, out_w = Conv(device, x, parameters, f"{path}.cv2", x.shape[1], x.shape[2], 1, 1, change_shard=True)

    return x, out_h, out_w


def Detect_cv2(device, x, parameters, path, inp_h, inp_w, k, reg_max, bfloat8=True):
    x, out_h, out_w = Conv(device, x, parameters, f"{path}.0", inp_h, inp_w, k, bfloat8=bfloat8)

    x, out_h, out_w = Conv(device, x, parameters, f"{path}.1", out_h, out_w, k, bfloat8=bfloat8)

    x, out_h, out_w = Conv(
        device,
        x,
        parameters,
        path,
        out_h,
        out_w,
        k=1,
        s=1,
        p=0,
        g=1,
        d=1,
        bfloat8=True,
        is_fused=False,
        change_shard=True,
        is_detect_cv2=True,
    )
    return x, out_h, out_w


def DFL(device, x, parameters, path, c1=16):
    b, _, a = x.shape

    x = ttnn.reshape(x, (b, 4, c1, a))

    x = ttnn.softmax(x, dim=2)

    x = ttnn.permute(x, (0, 1, 3, 2))

    x = Conv(
        device,
        x,
        parameters,
        path,
        x.shape[1],
        x.shape[2],
        k=1,
        s=1,
        p=0,
        g=1,
        d=1,
        bfloat8=True,
        is_fused=False,
        is_dfl=True,
        change_shard=True,
    )

    return x


def Detect(device, x, parameters, path, nc=80, ch=(), bfloat8=True):
    nc = nc
    nl = len(ch)
    reg_max = 16
    no = nc + reg_max * 4

    for i in range(nl):
        inp_h = inp_w = int(math.sqrt(x[i].shape[2]))
        a = Detect_cv2(
            device,
            x[i],
            parameters,
            path=f"{path}.cv2.{i}",
            inp_h=inp_h,
            inp_w=inp_w,
            k=3,
            reg_max=4 * reg_max,
        )[0]
        b = Detect_cv2(
            device,
            x[i],
            parameters,
            path=f"{path}.cv3.{i}",
            inp_h=inp_h,
            inp_w=inp_w,
            k=3,
            reg_max=nc,
            bfloat8=bfloat8,
        )[0]
        x[i] = ttnn.concat((a, b), dim=3)

    shape = x[0].shape

    anchors, strides = parameters["anchors"], parameters["strides"]

    xi = []
    for i in x:
        i = ttnn.permute(i, (0, 3, 1, 2))
        i = ttnn.reshape(i, (shape[0], no, -1))
        xi.append(i)

    x_cat = ttnn.concat(xi, 2)

    box = ttnn.slice(x_cat, [0, 0, 0], [1, 64, 8400])
    cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, 8400])

    dfl = DFL(device, box, parameters, f"{path}.dfl")

    dbox = ttnn_decode_bboxes(device, dfl, anchors)
    dbox = dbox * strides

    return [ttnn.concat((dbox, ttnn.sigmoid(cls)), dim=1), x]


def DetectionModel(device, x, parameters):
    Conv_0, out_h, out_w = Conv(
        device, x, parameters, "model.0", x.shape[1], x.shape[2], 3, 2, 1, change_shard=True, deallocate_activation=True
    )
    ttnn.deallocate(x)

    Conv_1, out_h, out_w = Conv(
        device, Conv_0, parameters, "model.1", out_h, out_w, 3, 2, 1, act_block_h=True, deallocate_activation=True
    )
    ttnn.deallocate(Conv_0)

    C2f_2, out_h, out_w = C2f(
        device, Conv_1, parameters, "model.2", out_h, out_w, n=3, shortcut=True, change_shard=True
    )
    ttnn.deallocate(Conv_1)

    Conv_3, out_h, out_w = Conv(
        device, C2f_2, parameters, "model.3", out_h, out_w, 3, 2, 1, change_shard=True, deallocate_activation=False
    )
    ttnn.deallocate(C2f_2)

    C2f_4, out_h, out_w = C2f(
        device, Conv_3, parameters, "model.4", out_h, out_w, n=6, shortcut=True, change_shard=True
    )
    ttnn.deallocate(Conv_3)

    four = ttnn.clone(C2f_4, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    Conv_5, out_h, out_w = Conv(
        device,
        C2f_4,
        parameters,
        "model.5",
        out_h,
        out_w,
        3,
        2,
        1,
        change_shard=True,
        block_shard=False,
        deallocate_activation=True,
    )
    ttnn.deallocate(C2f_4)

    C2f_6, out_h, out_w = C2f(
        device, Conv_5, parameters, "model.6", out_h, out_w, n=6, shortcut=True, block_shard=False, change_shard=True
    )
    ttnn.deallocate(Conv_5)

    six = ttnn.clone(C2f_6, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    Conv_7, out_h, out_w = Conv(
        device,
        C2f_6,
        parameters,
        "model.7",
        out_h,
        out_w,
        3,
        2,
        1,
        block_shard=False,
        change_shard=True,
        deallocate_activation=True,
    )
    ttnn.deallocate(C2f_6)

    Conv_7 = ttnn.sharded_to_interleaved(Conv_7, ttnn.L1_MEMORY_CONFIG)

    C2f_8, out_h, out_w = C2f(
        device, Conv_7, parameters, "model.8", out_h, out_w, n=3, shortcut=True, change_shard=True
    )
    ttnn.deallocate(Conv_7)

    SPPF_9, out_h, out_w = SPPF(device, C2f_8, parameters, "model.9", out_h, out_w)
    ttnn.deallocate(C2f_8)

    nine = ttnn.clone(SPPF_9, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    SPPF_9 = ttnn.to_layout(SPPF_9, ttnn.ROW_MAJOR_LAYOUT)

    SPPF_9 = ttnn.reshape(SPPF_9, (1, out_h, out_w, SPPF_9.shape[-1]))
    SPPF_9 = ttnn.upsample(SPPF_9, scale_factor=(2, 2))

    x = ttnn.reshape(SPPF_9, (1, 1, SPPF_9.shape[1] * SPPF_9.shape[2], SPPF_9.shape[-1]))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    x = ttnn.concat([x, six], dim=3)
    ttnn.deallocate(six)

    C2f_12, out_h, out_w = C2f(
        device, x, parameters, "model.12", SPPF_9.shape[1], SPPF_9.shape[2], n=3, shortcut=False, bfloat8=True
    )
    ttnn.deallocate(x)
    ttnn.deallocate(SPPF_9)

    C2f_12 = ttnn.sharded_to_interleaved(C2f_12, ttnn.L1_MEMORY_CONFIG)
    C2f_12 = ttnn.to_layout(C2f_12, ttnn.ROW_MAJOR_LAYOUT)

    twelve = ttnn.clone(C2f_12, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    C2f_12 = ttnn.reshape(C2f_12, (1, out_h, out_w, C2f_12.shape[-1]))

    C2f_12 = ttnn.upsample(C2f_12, scale_factor=(2, 2))

    x = ttnn.reshape(C2f_12, (1, 1, C2f_12.shape[1] * C2f_12.shape[2], C2f_12.shape[-1]))

    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    x = ttnn.concat([x, four], dim=3)

    ttnn.deallocate(four)
    ttnn.deallocate(C2f_12)

    C2f_15, out_h, out_w = C2f(device, x, parameters, "model.15", C2f_12.shape[1], C2f_12.shape[2], n=3, shortcut=False)
    ttnn.deallocate(x)

    C2f_15 = ttnn.sharded_to_interleaved(C2f_15, ttnn.L1_MEMORY_CONFIG)

    fifteen = ttnn.clone(C2f_15, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    Conv_16, out_h, out_w = Conv(device, C2f_15, parameters, "model.16", out_h, out_w, 3, 2, 1)
    ttnn.deallocate(C2f_15)

    Conv_16 = ttnn.sharded_to_interleaved(Conv_16, ttnn.L1_MEMORY_CONFIG)

    x = ttnn.concat([Conv_16, twelve], dim=3)
    ttnn.deallocate(twelve)
    ttnn.deallocate(Conv_16)

    C2f_18, out_h, out_w = C2f(device, x, parameters, "model.18", out_h, out_w, n=3, shortcut=False)
    ttnn.deallocate(x)

    C2f_18 = ttnn.sharded_to_interleaved(C2f_18, ttnn.L1_MEMORY_CONFIG)

    eighteen = ttnn.clone(C2f_18, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)

    Conv_19, out_h, out_w = Conv(device, C2f_18, parameters, "model.19", out_h, out_w, 3, 2, 1, block_shard=True)
    ttnn.deallocate(C2f_18)

    Conv_19 = ttnn.sharded_to_interleaved(Conv_19, ttnn.L1_MEMORY_CONFIG)

    x = ttnn.concat([Conv_19, nine], dim=3)
    ttnn.deallocate(nine)
    ttnn.deallocate(Conv_19)

    C2f_21, out_h, out_w = C2f(device, x, parameters, "model.21", out_h, out_w, n=3, shortcut=False)
    ttnn.deallocate(x)

    C2f_21 = ttnn.sharded_to_interleaved(C2f_21, ttnn.L1_MEMORY_CONFIG)

    twentyone = ttnn.clone(C2f_21, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(C2f_21)

    x = [fifteen, eighteen, twentyone]

    x = Detect(device, x, parameters, "model.22", nc=80, ch=(320, 640, 640))

    return x


def YOLOv8x(device, x, parameters):
    return DetectionModel(device, x, parameters)

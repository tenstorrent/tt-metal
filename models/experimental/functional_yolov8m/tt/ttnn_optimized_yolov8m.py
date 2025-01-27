# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn

from models.experimental.functional_yolov8m.tt.ttnn_yolov8m_utils import (
    autopad,
    ttnn_make_anchors,
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
    bfloat8=False,
    change_shard=False,
):
    p = autopad(k, p, d)
    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat8_b,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=(16 if False or (x.shape[-1] == 16 and x.shape[-2] == 115) else 32),
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.TILE_LAYOUT,
    )

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

    fused_weight, fused_bias = parameters[path]
    if bfloat8:
        conv_config.weights_dtype = ttnn.bfloat8_b

    [x, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=fused_weight,
        in_channels=fused_weight.shape[1],
        out_channels=fused_weight.shape[0],
        device=device,
        bias_tensor=fused_bias,
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
        return_weights_and_bias=True,
        return_output_dim=True,
    )

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
    )

    cv2, out_h, out_w = Conv(
        device,
        cv1,
        parameters,
        f"{path}.cv2",
        out_h,
        out_w,
        k[1][1],
        1,
        g=g,
        act_block_h=act_block_h,
        change_shard=change_shard,
    )

    ttnn.deallocate(cv1)

    cv2 = ttnn.sharded_to_interleaved(cv2, ttnn.L1_MEMORY_CONFIG)
    cv2 = ttnn.to_layout(cv2, ttnn.ROW_MAJOR_LAYOUT)
    cv2 = ttnn.reshape(cv2, (1, out_h, out_w, x.shape[-1]))

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
    bfloat8=False,
    block_shard=False,
    change_shard=None,
):
    cv1, out_h, out_w = Conv(
        device, x, parameters, f"{path}.cv1", in_h, in_w, 1, 1, bfloat8=bfloat8, change_shard=change_shard
    )

    cv1 = ttnn.sharded_to_interleaved(cv1, ttnn.L1_MEMORY_CONFIG)
    cv1 = ttnn.to_layout(cv1, ttnn.ROW_MAJOR_LAYOUT)
    cv1 = ttnn.reshape(cv1, (1, out_h, out_w, cv1.shape[-1]))
    y = list(ttnn.split(cv1, 2, 3))
    ttnn.deallocate(cv1)
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
    )
    return x, out_h, out_w


def SPPF(device, x, parameters, path, in_h, in_w, k=5, bfloat8=False):
    cv1, out_h, out_w = Conv(device, x, parameters, f"{path}.cv1", in_h, in_w, 1, 1)

    p = k // 2

    cv1 = ttnn.to_torch(cv1)

    cv1 = torch.reshape(cv1, (1, out_h, out_w, cv1.shape[-1]))
    cv1 = torch.permute(cv1, (0, 3, 1, 2))

    # tt maxpool2d low pcc case : submitted unit test.

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


def Detect_cv2(device, x, parameters, path, in_h, in_w, k, reg_max, bfloat8=False):
    x, out_h, out_w = Conv(device, x, parameters, f"{path}.0", in_h, in_w, k, bfloat8=bfloat8)

    x, out_h, out_w = Conv(device, x, parameters, f"{path}.1", out_h, out_w, k, bfloat8=bfloat8)

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
        output_layout=ttnn.TILE_LAYOUT,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    conv_weight, conv_bias = parameters[path]

    [x, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv_weight,
        in_channels=conv_weight.shape[1],
        out_channels=conv_weight.shape[0],
        device=device,
        bias_tensor=conv_bias,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        batch_size=x.shape[0],
        input_height=out_h,
        input_width=out_w,
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        memory_config=None,
        return_weights_and_bias=True,
        return_output_dim=True,
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (1, out_height, out_width, x.shape[-1]))
    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

    return x, out_height, out_width


def DFL(device, x, parameters, path, c1=16):
    c1 = c1
    b, _, a = x.shape

    x = ttnn.reshape(x, (b, 4, c1, a))
    x = ttnn.permute(x, (0, 2, 1, 3))

    x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
    x = ttnn.softmax(x, dim=1)

    x = ttnn.permute(x, (0, 2, 3, 1))

    conv_config = ttnn.Conv2dConfig(
        dtype=ttnn.bfloat16,
        weights_dtype=ttnn.bfloat16,
        activation="",
        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        input_channels_alignment=(16 if False or (c1 == 16 and x.shape[-2] == 115) else 32),
        deallocate_activation=False,
        enable_act_double_buffer=False,
        enable_split_reader=False,
        enable_subblock_padding=False,
        output_layout=ttnn.TILE_LAYOUT,
    )

    compute_config = ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.LoFi,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    conv_weight = parameters[path]

    [x, [out_height, out_width], [weights_device, bias_device]] = ttnn.conv2d(
        input_tensor=x,
        weight_tensor=conv_weight,
        in_channels=c1,
        out_channels=1,
        device=device,
        bias_tensor=None,
        kernel_size=(1, 1),
        stride=(1, 1),
        padding=(0, 0),
        dilation=(1, 1),
        batch_size=x.shape[0],
        input_height=x.shape[1],
        input_width=x.shape[2],
        conv_config=conv_config,
        compute_config=compute_config,
        conv_op_cache={},
        debug=False,
        groups=1,
        memory_config=None,
        return_weights_and_bias=True,
        return_output_dim=True,
    )

    x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
    x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    x = ttnn.reshape(x, (b, 4, -1))

    return x


def Detect(device, x, parameters, path, nc=80, ch=()):
    dynamic = False
    format = None
    self_shape = None
    nc = nc
    nl = len(ch)
    reg_max = 16
    no = nc + reg_max * 4

    stride = [8.0, 16.0, 32.0]

    for i in range(nl):
        dim = int(x[i].shape[2] ** 0.5)
        a = Detect_cv2(
            device,
            x[i],
            parameters,
            path=f"{path}.cv2.{i}",
            in_h=dim,
            in_w=dim,
            k=3,
            reg_max=4 * reg_max,
        )[0]
        b = Detect_cv2(
            device,
            x[i],
            parameters,
            path=f"{path}.cv3.{i}",
            in_h=dim,
            in_w=dim,
            k=3,
            reg_max=nc,
            bfloat8=True,
        )[0]

        x[i] = ttnn.concat((a, b), dim=3)

    shape = x[0].shape

    if format != "imx" and (dynamic or self_shape != shape):
        temp = ttnn_make_anchors(device, x, stride, 0.5)
        ls = []
        for i in temp:
            i = ttnn.sharded_to_interleaved(i, ttnn.L1_MEMORY_CONFIG)
            i = ttnn.to_layout(i, ttnn.ROW_MAJOR_LAYOUT)
            i = ttnn.permute(i, (1, 0))
            ls.append(i)

        anchors, strides = ls[0], ls[1]
        anchors = ttnn.reshape(anchors, (-1, anchors.shape[0], anchors.shape[1]))
        self_shape = shape

    xi = []
    for i in x:
        i = ttnn.sharded_to_interleaved(i, ttnn.L1_MEMORY_CONFIG)
        i = ttnn.to_layout(i, ttnn.ROW_MAJOR_LAYOUT)
        i = ttnn.permute(i, (0, 3, 1, 2))
        i = ttnn.reshape(i, (shape[0], no, -1))
        i = ttnn.to_layout(i, ttnn.TILE_LAYOUT)
        xi.append(i)

    x_cat = ttnn.concat(xi, 2)

    box = ttnn.slice(x_cat, [0, 0, 0], [1, 64, 8400])
    cls = ttnn.slice(x_cat, [0, 64, 0], [1, 144, 8400])

    dfl = DFL(device, box, parameters, f"{path}.dfl")

    anchors = ttnn.to_layout(anchors, ttnn.TILE_LAYOUT)
    strides = ttnn.to_layout(strides, ttnn.TILE_LAYOUT)

    dbox = ttnn_decode_bboxes(device, dfl, anchors)

    dbox = dbox * strides

    return [ttnn.concat((dbox, ttnn.sigmoid(cls)), dim=1), x]


def DetectionModel(device, x, parameters):
    Conv_0, out_h, out_w = Conv(device, x, parameters, "model.0", x.shape[1], x.shape[2], 3, 2, 1, act_block_h=True)
    ttnn.deallocate(x)

    Conv_1, out_h, out_w = Conv(device, Conv_0, parameters, "model.1", out_h, out_w, 3, 2, 1, act_block_h=True)
    ttnn.deallocate(Conv_0)

    C2f_2, out_h, out_w = C2f(device, Conv_1, parameters, "model.2", out_h, out_w, n=2, shortcut=True, act_block_h=True)
    ttnn.deallocate(Conv_1)

    conv_3, out_h, out_w = Conv(device, C2f_2, parameters, "model.3", out_h, out_w, 3, 2, 1)
    ttnn.deallocate(C2f_2)

    c2f_4, out_h, out_w = C2f(device, conv_3, parameters, "model.4", out_h, out_w, n=4, shortcut=True, bfloat8=True)
    c2f_4 = ttnn.sharded_to_interleaved(c2f_4, ttnn.L1_MEMORY_CONFIG)
    four = ttnn.clone(c2f_4, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    conv_5, out_h, out_w = Conv(
        device, c2f_4, parameters, "model.5", out_h, out_w, 3, 2, 1, block_shard=True, bfloat8=False
    )
    ttnn.deallocate(c2f_4)

    c2f_6, out_h, out_w = C2f(device, conv_5, parameters, "model.6", out_h, out_w, n=4, shortcut=True, block_shard=True)
    ttnn.deallocate(conv_5)

    c2f_6 = ttnn.sharded_to_interleaved(c2f_6, ttnn.L1_MEMORY_CONFIG)
    six = ttnn.clone(c2f_6, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    conv_7, out_h, out_w = Conv(device, c2f_6, parameters, "model.7", out_h, out_w, 3, 2, 1, block_shard=True)
    ttnn.deallocate(c2f_6)

    c2f_8, out_h, out_w = C2f(device, conv_7, parameters, "model.8", out_h, out_w, n=2, shortcut=True, bfloat8=True)
    ttnn.deallocate(conv_7)

    spppf_9, out_h, out_w = SPPF(device, c2f_8, parameters, "model.9", out_h, out_w)
    nine = ttnn.clone(spppf_9, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    spppf_9 = ttnn.to_layout(spppf_9, ttnn.ROW_MAJOR_LAYOUT)
    spppf_9 = ttnn.reshape(spppf_9, (1, out_h, out_w, spppf_9.shape[-1]))
    up = ttnn.upsample(spppf_9, scale_factor=(2, 2))
    ttnn.deallocate(spppf_9)

    x = ttnn.reshape(up, (up.shape[0], 1, (up.shape[1] * up.shape[2]), up.shape[-1]))
    x = ttnn.concat([x, six], dim=3)
    ttnn.deallocate(six)

    c2f_12, out_h, out_w = C2f(
        device, x, parameters, "model.12", up.shape[1], up.shape[2], n=2, shortcut=False, bfloat8=True
    )
    ttnn.deallocate(x)

    c2f_12 = ttnn.sharded_to_interleaved(c2f_12, ttnn.L1_MEMORY_CONFIG)
    twelve = ttnn.clone(c2f_12, dtype=ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    c2f_12 = ttnn.to_layout(c2f_12, ttnn.ROW_MAJOR_LAYOUT)
    c2f_12 = ttnn.reshape(c2f_12, (1, out_h, out_w, c2f_12.shape[-1]))
    up = ttnn.upsample(c2f_12, scale_factor=(2, 2))
    ttnn.deallocate(c2f_12)

    x = ttnn.reshape(up, (up.shape[0], 1, (up.shape[1] * up.shape[2]), up.shape[-1]))
    x = ttnn.concat([x, four], dim=3)
    ttnn.deallocate(four)

    c2f_15, out_h, out_w = C2f(device, x, parameters, "model.15", up.shape[1], up.shape[2], n=2, shortcut=False)
    ttnn.deallocate(x)

    c2f_15 = ttnn.sharded_to_interleaved(c2f_15, ttnn.L1_MEMORY_CONFIG)
    fifteen = ttnn.clone(c2f_15, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    conv_16, out_h, out_w = Conv(device, c2f_15, parameters, "model.16", out_h, out_w, 3, 2, 1)
    ttnn.deallocate(c2f_15)

    conv_16 = ttnn.sharded_to_interleaved(conv_16, ttnn.L1_MEMORY_CONFIG)
    conv_16 = ttnn.concat([conv_16, twelve], dim=3)
    ttnn.deallocate(twelve)

    c2f_18, out_h, out_w = C2f(device, conv_16, parameters, "model.18", out_h, out_w, n=2, shortcut=False, bfloat8=True)
    ttnn.deallocate(conv_16)
    c2f_18 = ttnn.sharded_to_interleaved(c2f_18, ttnn.L1_MEMORY_CONFIG)

    eighteen = ttnn.clone(c2f_18, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    conv_19, out_h, out_w = Conv(device, c2f_18, parameters, "model.19", out_h, out_w, 3, 2, 1, block_shard=True)
    ttnn.deallocate(c2f_18)

    conv_19 = ttnn.sharded_to_interleaved(conv_19, ttnn.L1_MEMORY_CONFIG)
    conv_19 = ttnn.concat([conv_19, nine], dim=3)
    ttnn.deallocate(nine)

    c2f_21, out_h, out_w = C2f(device, conv_19, parameters, "model.21", out_h, out_w, n=2, shortcut=False, bfloat8=True)
    ttnn.deallocate(conv_19)
    c2f_21 = ttnn.sharded_to_interleaved(c2f_21, ttnn.L1_MEMORY_CONFIG)

    twentyone = ttnn.clone(c2f_21, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ttnn.deallocate(c2f_21)
    x = [fifteen, eighteen, twentyone]

    detect = Detect(device, x, parameters, "model.22", nc=80, ch=(192, 384, 576))
    ttnn.deallocate(x[0])
    ttnn.deallocate(x[1])
    ttnn.deallocate(x[2])

    return detect


def YOLOv8m(device, x, parameters):
    return DetectionModel(device, x, parameters)

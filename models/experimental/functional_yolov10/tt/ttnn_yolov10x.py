# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.experimental.functional_yolov10.tt.common import Yolov10_Conv2D, Conv
from models.experimental.functional_yolov10.reference.yolov10 import Conv as Torch_conv
import math


class ttnn_Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        out = ttnn.concat(x, self.d)
        return out


class ttnn_BottleNeck:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
        )

    def __call__(self, x):
        cv1 = self.cv1(x)
        cv2 = self.cv2(cv1)
        ttnn.deallocate(cv1)
        x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        return ttnn.add(x, cv2, memory_config=ttnn.L1_MEMORY_CONFIG) if self.shortcut else cv2


class ttnn_SCDown:
    def __init__(self, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        if torch_conv:
            self.cv2 = nn.Conv2d(
                in_channels=640,
                out_channels=640,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=640,
                bias=False,
            )
            self.cv2.weight = torch.nn.Parameter(ttnn.to_torch(ttnn.from_device(self.conv_pt.cv2.conv.weight)))
            self.cv2.bias = torch.nn.Parameter(
                ttnn.to_torch(ttnn.from_device(self.conv_pt.cv2.conv.bias)).squeeze(0).squeeze(0).squeeze(0)
            )

        else:
            self.cv2 = Conv(
                device,
                parameters.cv2,
                self.conv_pt.cv2,
                enable_identity=True,
                use_1d_systolic_array=False,
                config_override={"act_block_h": 512},
            )

    def __call__(self, x):
        x = self.cv1(x)

        x = ttnn.to_torch(x)
        x = torch.permute(x, (0, 3, 1, 2)).float()
        x = torch.reshape(x, (1, x.shape[1], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))))
        x = self.cv2(x)
        x = ttnn.from_torch(
            x, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x = ttnn.identity(x, memory_config=ttnn.L1_MEMORY_CONFIG)
        return x


class ttnn_SPPF:
    def __init__(self, device=None, parameters=None, conv_pt=None):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
        )

    def __call__(self, x):
        y = [self.cv1(x)]
        y[0] = ttnn.sharded_to_interleaved(y[0], ttnn.L1_MEMORY_CONFIG)
        y[0] = ttnn.to_layout(y[0], ttnn.ROW_MAJOR_LAYOUT)

        for i in range(3):
            tt_max = y[-1]
            tt_out = ttnn.max_pool2d(
                input_tensor=y[-1],
                batch_size=x.shape[0],
                input_h=20,
                input_w=20,
                channels=320,
                kernel_size=[5, 5],
                stride=[1, 1],
                padding=[2, 2],
                dilation=[1, 1],
                memory_config=ttnn.L1_MEMORY_CONFIG,
            )
            y.append(tt_out)

        out = ttnn.concat(y, dim=-1)

        out = self.cv2(out)
        return out


class ttnn_CIB:
    def __init__(self, shortcut=True, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.conv0 = Conv(
            device,
            parameters.cv1[0],
            self.conv_pt.cv1[0],
        )

        self.conv1 = Conv(
            device,
            parameters.cv1[1],
            self.conv_pt.cv1[1],
        )

        if torch_conv:
            self.conv2 = nn.Conv2d(
                in_channels=640,
                out_channels=640,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=640,
                bias=False,
            )
            self.conv2.weight = torch.nn.Parameter(ttnn.to_torch(ttnn.from_device(self.conv_pt.cv1[2].conv.weight)))
            self.conv2.bias = torch.nn.Parameter(
                ttnn.to_torch(ttnn.from_device(self.conv_pt.cv1[2].conv.bias)).squeeze(0).squeeze(0).squeeze(0)
            )

        else:
            self.conv2 = Conv(
                device,
                parameters.cv1[2],
                self.conv_pt.cv1[2],
            )

        self.conv3 = Conv(
            device,
            parameters.cv1[3],
            self.conv_pt.cv1[3],
        )

        self.conv4 = Conv(
            device,
            parameters.cv1[4],
            self.conv_pt.cv1[4],
        )

    def __call__(self, x):
        input_tensor = x
        x = self.conv0(x)
        x = self.conv1(x)

        x = ttnn.to_torch(x)
        x = torch.permute(x, (0, 3, 1, 2)).float()
        x = torch.reshape(x, (1, x.shape[1], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))))
        x = self.conv2(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        x = ttnn.from_torch(
            x, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x = ttnn.silu(x)

        x = self.conv3(x)
        x = self.conv4(x)

        x = input_tensor + x
        return x


class ttnn_Attention:
    def __init__(self, dim, num_heads=8, attn_ratio=0.5, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2

        self.qkv = Conv(
            device,
            parameters.qkv,
            self.conv_pt.qkv,
            enable_identity=True,
        )
        self.proj = Conv(
            device,
            parameters.proj,
            self.conv_pt.proj,
            enable_identity=True,
        )

        self.pe = Conv(
            device,
            parameters.pe,
            self.conv_pt.pe,
            enable_identity=True,
        )

    def __call__(self, x):
        B, C, H, W = (1, x.shape[-1], int(math.sqrt(x.shape[2])), int(math.sqrt(x.shape[2])))
        N = H * W

        qkv = self.qkv(x)
        qkv = ttnn.from_device(qkv)
        qkv = ttnn.to_dtype(qkv, ttnn.bfloat16)
        qkv = ttnn.to_device(qkv, self.device)
        qkv = ttnn.permute(qkv, (0, 3, 1, 2))
        qkv = ttnn.reshape(qkv, (1, qkv.shape[1], int(math.sqrt(qkv.shape[-1])), int(math.sqrt(qkv.shape[-1]))))
        qkv = ttnn.reshape(qkv, (B, self.num_heads, self.key_dim * 2 + self.head_dim, N))

        qkv = ttnn.to_layout(qkv, ttnn.ROW_MAJOR_LAYOUT)

        qk, v = ttnn.split(qkv, 2, dim=2)
        q, k = ttnn.split(qk, 2, dim=2)

        q = ttnn.permute(q, (0, 1, 3, 2))

        q = ttnn.to_layout(q, ttnn.TILE_LAYOUT)
        k = ttnn.to_layout(k, ttnn.TILE_LAYOUT)

        attn = ttnn.matmul(q, k)
        attn = attn * self.scale

        attn = ttnn.softmax(attn, dim=-1)

        v = ttnn.to_layout(v, ttnn.TILE_LAYOUT)
        attn = ttnn.permute(attn, (0, 1, 3, 2))
        attn = ttnn.matmul(v, attn)
        attn = ttnn.reshape(attn, (B, C, H, W))

        v = ttnn.reshape(v, (B, C, H, W))

        v = ttnn.permute(v, (0, 2, 3, 1))  # for conv
        v = ttnn.reshape(v, (1, 1, v.shape[0] * v.shape[1] * v.shape[2], v.shape[3]))  # for conv

        v = self.pe(v)

        attn = ttnn.permute(attn, (0, 2, 3, 1))  # for conv
        attn = ttnn.reshape(attn, (1, 1, attn.shape[0] * attn.shape[1] * attn.shape[2], attn.shape[3]))  # for conv

        x = attn + v

        x = self.proj(x)

        return x


class ttnn_PSA:
    def __init__(self, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )

        self.cv2 = Conv(
            device,
            parameters.cv2,
            self.conv_pt.cv2,
        )

        self.attn = ttnn_Attention(
            dim=320,
            num_heads=5,
            attn_ratio=0.5,
            device=self.device,
            parameters=self.parameters.attn,
            conv_pt=self.conv_pt.attn,
        )

        self.ffn_0 = Conv(
            device,
            parameters.ffn[0],
            self.conv_pt.ffn[0],
        )

        self.ffn_1 = Conv(
            device,
            parameters.ffn[1],
            self.conv_pt.ffn[1],
            enable_identity=True,
        )

    def __call__(self, x):
        x = self.cv1(x)

        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.reshape(x, (1, x.shape[1], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))))

        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        a, b = ttnn.split(x, 2, dim=1)

        b = ttnn.permute(b, (0, 2, 3, 1))
        b = ttnn.reshape(b, (1, 1, b.shape[0] * b.shape[1] * b.shape[2], b.shape[3]))

        out = self.attn(b)

        b = ttnn.to_layout(b, ttnn.TILE_LAYOUT)
        b = b + out

        out = self.ffn_0(b)
        out = self.ffn_1(out)

        b = b + out

        a = ttnn.permute(a, (0, 2, 3, 1))
        a = ttnn.reshape(a, (1, 1, a.shape[0] * a.shape[1] * a.shape[2], a.shape[3]))
        a = ttnn.to_layout(a, ttnn.TILE_LAYOUT)

        out = ttnn.concat([a, b], dim=3)

        out = self.cv2(out)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

        return out


class ttnn_DFL:
    def __init__(self, c1=16, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.c1 = c1

        self.conv = Conv(device, parameters, self.conv_pt, enable_act=False, enable_identity=False, is_dfl=True)

    def __call__(self, x):
        b, _, a = x.shape
        x = ttnn.reshape(x, (b, 4, self.c1, a))
        x = ttnn.permute(x, (0, 1, 3, 2))
        x = ttnn.softmax(x, dim=-1)
        x = self.conv(x)
        x = ttnn.sharded_to_interleaved(x, ttnn.L1_MEMORY_CONFIG)
        x = ttnn.reshape(x, (b, 4, a))
        return x


class ttnn_C2f:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.shortcut = shortcut
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )
        self.cv2 = Conv(device, parameters.cv2, self.conv_pt.cv2)
        self.m = [
            ttnn_BottleNeck(self.shortcut, device=self.device, parameters=self.parameters[_], conv_pt=self.conv_pt.m[_])
            for _ in range(n)
        ]

    def __call__(self, x):
        x = self.cv1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.reshape(x, (1, x.shape[1], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))))
        y = list(ttnn.split(x, 2, 1))
        y[0] = ttnn.permute(y[0], (0, 2, 3, 1))
        y[0] = ttnn.reshape(y[0], (1, 1, y[0].shape[0] * y[0].shape[1] * y[0].shape[2], y[0].shape[3]))
        y[0] = ttnn.to_layout(y[0], ttnn.TILE_LAYOUT)

        y[1] = ttnn.permute(y[1], (0, 2, 3, 1))
        y[1] = ttnn.reshape(y[1], (1, 1, y[1].shape[0] * y[1].shape[1] * y[1].shape[2], y[1].shape[3]))
        y[1] = ttnn.to_layout(y[1], ttnn.TILE_LAYOUT)

        y = [y[0], y[1]]

        for m in self.m:
            out = m(y[-1])

            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = ttnn.concat(y, -1)

        out = self.cv2(out)

        return out


class ttnn_C2fCIB:
    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.device = device
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.shortcut = shortcut

        self.cv1 = Conv(
            device,
            parameters.cv1,
            self.conv_pt.cv1,
        )
        if torch_conv:
            self.cv2 = nn.Conv2d(
                in_channels=2560,
                out_channels=640,
                kernel_size=1,
                stride=1,
                bias=False,
            )
            self.cv2.weight = torch.nn.Parameter(ttnn.to_torch(ttnn.from_device(self.conv_pt.cv2.conv.weight)))
            self.cv2.bias = torch.nn.Parameter(
                ttnn.to_torch(ttnn.from_device(self.conv_pt.cv2.conv.bias)).squeeze(0).squeeze(0).squeeze(0)
            )
        else:
            self.cv2 = Conv(
                device,
                parameters.cv2,
                self.conv_pt.cv2,
                auto_shard=True,
            )

        self.m = [
            ttnn_CIB(
                shortcut=self.shortcut,
                device=self.device,
                parameters=self.parameters[2],
                conv_pt=self.conv_pt.m[_],
                torch_conv=True,
            )
            for _ in range(n)
        ]

    def __call__(self, x):
        x = self.cv1(x)
        x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.permute(x, (0, 3, 1, 2))
        x = ttnn.reshape(x, (1, x.shape[1], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))))
        y = list(ttnn.split(x, 2, 1))

        y[0] = ttnn.permute(y[0], (0, 2, 3, 1))
        y[0] = ttnn.reshape(y[0], (1, 1, y[0].shape[0] * y[0].shape[1] * y[0].shape[2], y[0].shape[3]))
        y[0] = ttnn.to_layout(y[0], ttnn.TILE_LAYOUT)

        y[1] = ttnn.permute(y[1], (0, 2, 3, 1))
        y[1] = ttnn.reshape(y[1], (1, 1, y[1].shape[0] * y[1].shape[1] * y[1].shape[2], y[1].shape[3]))
        y[1] = ttnn.to_layout(y[1], ttnn.TILE_LAYOUT)

        y = [y[0], y[1]]

        for m in self.m:
            out = m(y[-1])

            y.append(ttnn.to_layout(out, ttnn.TILE_LAYOUT))

        out = ttnn.concat(y, -1)

        x = ttnn.to_torch(out)
        x = torch.permute(x, (0, 3, 1, 2)).float()
        x = torch.reshape(x, (1, x.shape[1], int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))))
        x = self.cv2(x)
        x = torch.permute(x, (0, 2, 3, 1))
        x = x.reshape(1, 1, x.shape[0] * x.shape[1] * x.shape[2], x.shape[3])
        x = ttnn.from_torch(
            x, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x = ttnn.silu(x)

        return x


class ttnn_V10Detect:
    end2end = True

    def __init__(self, shortcut=True, n=3, device=None, parameters=None, conv_pt=None, torch_conv=False):
        self.parameters = parameters
        self.conv_pt = conv_pt
        self.device = device

        self.cv2_0_0 = Conv(device, parameters.cv2[0][0], self.conv_pt.cv2[0][0], is_detect=True)
        self.cv2_0_1 = Conv(device, parameters.cv2[0][1], self.conv_pt.cv2[0][1], is_detect=True)
        self.cv2_0_2 = Yolov10_Conv2D(parameters.cv2[0][2], self.conv_pt.cv2[0][2], device=device, is_detect=True)

        self.cv2_1_0 = Conv(device, parameters.cv2[1][0], self.conv_pt.cv2[1][0], is_detect=True)
        self.cv2_1_1 = Conv(device, parameters.cv2[1][1], self.conv_pt.cv2[1][1], is_detect=True)
        self.cv2_1_2 = Yolov10_Conv2D(parameters.cv2[1][2], self.conv_pt.cv2[1][2], device=device, is_detect=True)

        self.cv2_2_0 = Conv(device, parameters.cv2[2][0], self.conv_pt.cv2[2][0], is_detect=True)
        self.cv2_2_1 = Conv(device, parameters.cv2[2][1], self.conv_pt.cv2[2][1], is_detect=True)
        self.cv2_2_2 = Yolov10_Conv2D(parameters.cv2[2][2], self.conv_pt.cv2[2][2], device=device, is_detect=True)

        self.cv3_0_0_0 = Conv(device, parameters.cv3[0][0][0], conv_pt.cv3[0][0][0], is_detect=True)
        self.cv3_0_0_1 = Conv(device, parameters.cv3[0][0][1], conv_pt.cv3[0][0][1], is_detect=True)
        self.cv3_0_1_0 = Conv(device, parameters.cv3[0][1][0], conv_pt.cv3[0][1][0], is_detect=True)
        self.cv3_0_1_1 = Conv(device, parameters.cv3[0][1][1], conv_pt.cv3[0][1][1], is_detect=True)
        self.cv3_0_2_0 = Yolov10_Conv2D(parameters.cv3[0][2], conv_pt.cv3[0][2], device=device, is_detect=True)

        self.cv3_1_0_0 = Conv(
            device, parameters.cv3[1][0][0], conv_pt.cv3[1][0][0], is_detect=True, use_1d_systolic_array=False
        )
        self.cv3_1_0_1 = Conv(device, parameters.cv3[1][0][1], conv_pt.cv3[1][0][1], is_detect=True)
        self.cv3_1_1_0 = Conv(device, parameters.cv3[1][1][0], conv_pt.cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = Conv(device, parameters.cv3[1][1][1], conv_pt.cv3[1][1][1], is_detect=True)
        self.cv3_1_2_0 = Yolov10_Conv2D(parameters.cv3[1][2], conv_pt.cv3[1][2], device=device, is_detect=True)

        if torch_conv:
            self.cv3_1_0_0 = nn.Conv2d(
                in_channels=640,
                out_channels=640,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=640,
                bias=False,
            )
        else:
            self.cv3_1_0_0 = Conv(
                device, parameters.cv3[1][0][0], conv_pt.cv3[1][0][0], is_detect=True, use_1d_systolic_array=True
            )

        self.cv3_1_0_1 = Conv(device, parameters.cv3[1][0][1], conv_pt.cv3[1][0][1], is_detect=True)
        self.cv3_1_1_0 = Conv(device, parameters.cv3[1][1][0], conv_pt.cv3[1][1][0], is_detect=True)
        self.cv3_1_1_1 = Conv(device, parameters.cv3[1][1][1], conv_pt.cv3[1][1][1], is_detect=True)
        self.cv3_1_2_0 = Yolov10_Conv2D(parameters.cv3[1][2], conv_pt.cv3[1][2], device=device, is_detect=True)

        if torch_conv:
            self.cv3_2_0_0 = nn.Conv2d(
                in_channels=640,
                out_channels=640,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=640,
                bias=False,
            )
        else:
            self.cv3_2_0_0 = Conv(device, parameters.cv3[2][0][0], conv_pt.cv3[2][0][0], is_detect=True)

        self.cv3_2_0_1 = Conv(device, parameters.cv3[2][0][1], conv_pt.cv3[2][0][1], is_detect=True)
        self.cv3_2_1_0 = Conv(device, parameters.cv3[2][1][0], conv_pt.cv3[2][1][0], is_detect=True)
        self.cv3_2_1_1 = Conv(device, parameters.cv3[2][1][1], conv_pt.cv3[2][1][1], is_detect=True)
        self.cv3_2_2_0 = Yolov10_Conv2D(parameters.cv3[2][2], conv_pt.cv3[2][2], device=device, is_detect=True)
        self.dfl = Conv(device, parameters.dfl, self.conv_pt.dfl, is_dfl=True)

        self.anchors = conv_pt.anchors
        self.strides = conv_pt.strides

    def __call__(self, y1, y2, y3):
        x1 = self.cv2_0_0(y1)
        x1 = self.cv2_0_1(x1)
        x1 = self.cv2_0_2(x1)
        x2 = self.cv2_1_0(y2)
        x2 = self.cv2_1_1(x2)
        x2 = self.cv2_1_2(x2)

        x3 = self.cv2_2_0(y3)
        x3 = self.cv2_2_1(x3)
        x3 = self.cv2_2_2(x3)

        x4 = self.cv3_0_0_0(y1)
        x4 = self.cv3_0_0_1(x4)
        x4 = self.cv3_0_1_0(x4)
        x4 = self.cv3_0_1_1(x4)
        x4 = self.cv3_0_2_0(x4)

        y2 = ttnn.to_torch(y2)
        y2 = torch.permute(y2, (0, 3, 1, 2)).float()
        y2 = torch.reshape(y2, (1, y2.shape[1], int(math.sqrt(y2.shape[-1])), int(math.sqrt(y2.shape[-1]))))
        x5 = self.cv3_1_0_0(y2)
        x5 = torch.permute(x5, (0, 2, 3, 1))
        x5 = x5.reshape(1, 1, x5.shape[0] * x5.shape[1] * x5.shape[2], x5.shape[3])
        x5 = ttnn.from_torch(
            x5, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x5 = ttnn.silu(x5)

        x5 = self.cv3_1_0_1(x5)
        x5 = self.cv3_1_1_0(x5)
        x5 = self.cv3_1_1_1(x5)
        x5 = self.cv3_1_2_0(x5)

        y3 = ttnn.to_torch(y3)
        y3 = torch.permute(y3, (0, 3, 1, 2)).float()
        y3 = torch.reshape(y3, (1, y3.shape[1], int(math.sqrt(y3.shape[-1])), int(math.sqrt(y3.shape[-1]))))
        x6 = self.cv3_2_0_0(y3)
        x6 = torch.permute(x6, (0, 2, 3, 1))
        x6 = x6.reshape(1, 1, x6.shape[0] * x6.shape[1] * x6.shape[2], x6.shape[3])
        x6 = ttnn.from_torch(
            x6, device=self.device, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat8_b, memory_config=ttnn.L1_MEMORY_CONFIG
        )
        x6 = ttnn.silu(x6)
        x6 = self.cv3_2_0_1(x6)
        x6 = self.cv3_2_1_0(x6)
        x6 = self.cv3_2_1_1(x6)
        x6 = self.cv3_2_2_0(x6)

        x1 = ttnn.sharded_to_interleaved(x1, memory_config=ttnn.L1_MEMORY_CONFIG)
        x2 = ttnn.sharded_to_interleaved(x2, memory_config=ttnn.L1_MEMORY_CONFIG)
        x3 = ttnn.sharded_to_interleaved(x3, memory_config=ttnn.L1_MEMORY_CONFIG)
        x4 = ttnn.sharded_to_interleaved(x4, memory_config=ttnn.L1_MEMORY_CONFIG)
        x5 = ttnn.sharded_to_interleaved(x5, memory_config=ttnn.L1_MEMORY_CONFIG)
        x6 = ttnn.sharded_to_interleaved(x6, memory_config=ttnn.L1_MEMORY_CONFIG)

        y1 = ttnn.concat((x1, x4), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y2 = ttnn.concat((x2, x5), -1, memory_config=ttnn.L1_MEMORY_CONFIG)
        y3 = ttnn.concat((x3, x6), -1, memory_config=ttnn.L1_MEMORY_CONFIG)

        y = ttnn.concat((y1, y2, y3), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        y = ttnn.squeeze(y, dim=0)  # PCC = 0.9706918150614874

        ya, yb = y[:, :, :64], y[:, :, 64:144]  # PCC = 0.8106240777832109, #PCC = 0.6693110387385858
        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)
        ttnn.deallocate(x1)
        ttnn.deallocate(x2)
        ttnn.deallocate(x3)
        ttnn.deallocate(x4)
        ttnn.deallocate(x5)
        ttnn.deallocate(x6)
        ttnn.deallocate(y)
        ya = ttnn.reallocate(ya)
        yb = ttnn.reallocate(yb)
        ya = ttnn.reshape(ya, (ya.shape[0], y.shape[1], 4, 16))
        ya = ttnn.permute(ya, (0, 2, 1, 3))
        ya = ttnn.softmax(ya, dim=-1)
        c = self.dfl(ya)
        ttnn.deallocate(ya)
        c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.to_layout(c, layout=ttnn.ROW_MAJOR_LAYOUT)
        c = ttnn.permute(c, (0, 3, 1, 2))
        c = ttnn.reshape(c, (c.shape[0], 1, 4, int(c.shape[3] / 4)))
        c = ttnn.reshape(c, (c.shape[0], c.shape[1] * c.shape[2], c.shape[3]))
        c1, c2 = c[:, :2, :], c[:, 2:4, :]

        anchor, strides = self.anchors, self.strides
        anchor = ttnn.to_memory_config(anchor, memory_config=ttnn.L1_MEMORY_CONFIG)
        strides = ttnn.to_memory_config(strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        c1 = ttnn.to_layout(c1, layout=ttnn.TILE_LAYOUT)
        c2 = ttnn.to_layout(c2, layout=ttnn.TILE_LAYOUT)

        c1 = anchor - c1
        c2 = anchor + c2

        z1 = c2 - c1
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)

        z = ttnn.concat((z2, z1), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        z = ttnn.multiply(z, strides)
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)
        ttnn.deallocate(c)
        ttnn.deallocate(z1)
        ttnn.deallocate(z2)
        ttnn.deallocate(c1)
        ttnn.deallocate(c2)
        ttnn.deallocate(anchor)
        ttnn.deallocate(strides)
        z = ttnn.reallocate(z)
        yb = ttnn.reallocate(yb)
        z = ttnn.to_layout(z, layout=ttnn.ROW_MAJOR_LAYOUT)
        yb = ttnn.to_layout(yb, layout=ttnn.ROW_MAJOR_LAYOUT)
        out = ttnn.concat((z, yb), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(yb)
        ttnn.deallocate(z)
        return out

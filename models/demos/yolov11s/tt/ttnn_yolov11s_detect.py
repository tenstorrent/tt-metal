# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.demos.yolov11s.tt.common import TtnnConv, Yolov11sConv2D, deallocate_tensors, sharded_concat_2

_DETECT_CV3_STEM = {
    "act_block_h": 32,
    "enable_weights_double_buffer": True,
    "enable_act_double_buffer": False,
    "config_tensors_in_dram": True,
}
_DETECT_DRAM_HEAVY = {
    "act_block_h": 32,
    "enable_weights_double_buffer": True,
    "enable_act_double_buffer": False,
    "config_tensors_in_dram": True,
}
_DETECT_PW64 = {
    "act_block_h": 64,
    "enable_weights_double_buffer": True,
    "enable_act_double_buffer": True,
}
_DETECT_PW128 = {
    "act_block_h": 64,
    "enable_weights_double_buffer": True,
    "enable_act_double_buffer": False,
}
_DETECT_PW256 = {
    "act_block_h": 64,
    "enable_weights_double_buffer": True,
    "enable_act_double_buffer": False,
}
_DETECT_DEPTHWISE = {
    "act_block_h": 64,
    "enable_weights_double_buffer": True,
    "enable_act_double_buffer": False,
}
_DETECT_DFL = {
    "act_block_h": 32,
    "enable_weights_double_buffer": True,
    "enable_act_double_buffer": True,
}


def _run_detect_conv_chain(device, layers, x):
    """Apply TtnnConv (needs device) or Yolov11sConv2D in order. Same conv count as unrolled forward."""
    for layer in layers:
        if isinstance(layer, TtnnConv):
            x = layer(device, x)
        else:
            x = layer(x)
    return x


class TtnnDetect:
    def __init__(self, device, parameter, conv_pt):
        stem = _DETECT_CV3_STEM
        dram = _DETECT_DRAM_HEAVY
        pw64 = _DETECT_PW64
        pw128 = _DETECT_PW128
        pw256 = _DETECT_PW256
        dw = _DETECT_DEPTHWISE
        dfl = _DETECT_DFL

        # Three FPN scales; each cv2 is 2× TtnnConv + 1× depthwise Yolov11sConv2D (matches torch Sequential).
        self._cv2_chains = [
            [
                TtnnConv(device, parameter.cv2[0][0], conv_pt.cv2[0][0], is_detect=True, config_override=pw128),
                TtnnConv(device, parameter.cv2[0][1], conv_pt.cv2[0][1], is_detect=True, config_override=pw64),
                Yolov11sConv2D(
                    parameter.cv2[0][2], conv_pt.cv2[0][2], device=device, is_detect=True, config_override=dw
                ),
            ],
            [
                TtnnConv(device, parameter.cv2[1][0], conv_pt.cv2[1][0], is_detect=True, config_override=pw256),
                TtnnConv(device, parameter.cv2[1][1], conv_pt.cv2[1][1], is_detect=True, config_override=pw64),
                Yolov11sConv2D(
                    parameter.cv2[1][2], conv_pt.cv2[1][2], device=device, is_detect=True, config_override=dw
                ),
            ],
            [
                TtnnConv(device, parameter.cv2[2][0], conv_pt.cv2[2][0], is_detect=True, config_override=dram),
                TtnnConv(device, parameter.cv2[2][1], conv_pt.cv2[2][1], is_detect=True, config_override=pw64),
                Yolov11sConv2D(
                    parameter.cv2[2][2], conv_pt.cv2[2][2], device=device, is_detect=True, config_override=dw
                ),
            ],
        ]

        self._cv3_chains = [
            [
                TtnnConv(device, parameter.cv3[0][0][0], conv_pt.cv3[0][0][0], is_detect=True, config_override=stem),
                TtnnConv(device, parameter.cv3[0][0][1], conv_pt.cv3[0][0][1], is_detect=True, config_override=pw128),
                TtnnConv(device, parameter.cv3[0][1][0], conv_pt.cv3[0][1][0], is_detect=True, config_override=pw128),
                TtnnConv(device, parameter.cv3[0][1][1], conv_pt.cv3[0][1][1], is_detect=True, config_override=pw128),
                Yolov11sConv2D(
                    parameter.cv3[0][2], conv_pt.cv3[0][2], device=device, is_detect=True, config_override=dw
                ),
            ],
            [
                TtnnConv(
                    device,
                    parameter.cv3[1][0][0],
                    conv_pt.cv3[1][0][0],
                    is_detect=True,
                    config_override={**stem, "enable_act_double_buffer": True, "enable_weights_double_buffer": False},
                ),
                TtnnConv(device, parameter.cv3[1][0][1], conv_pt.cv3[1][0][1], is_detect=True, config_override=pw256),
                TtnnConv(device, parameter.cv3[1][1][0], conv_pt.cv3[1][1][0], is_detect=True, config_override=pw128),
                TtnnConv(device, parameter.cv3[1][1][1], conv_pt.cv3[1][1][1], is_detect=True, config_override=pw128),
                Yolov11sConv2D(
                    parameter.cv3[1][2], conv_pt.cv3[1][2], device=device, is_detect=True, config_override=dw
                ),
            ],
            [
                TtnnConv(
                    device,
                    parameter.cv3[2][0][0],
                    conv_pt.cv3[2][0][0],
                    is_detect=True,
                    config_override={**stem, "enable_act_double_buffer": True, "enable_weights_double_buffer": False},
                ),
                TtnnConv(device, parameter.cv3[2][0][1], conv_pt.cv3[2][0][1], is_detect=True, config_override=dram),
                TtnnConv(device, parameter.cv3[2][1][0], conv_pt.cv3[2][1][0], is_detect=True, config_override=pw128),
                TtnnConv(device, parameter.cv3[2][1][1], conv_pt.cv3[2][1][1], is_detect=True, config_override=pw128),
                Yolov11sConv2D(
                    parameter.cv3[2][2], conv_pt.cv3[2][2], device=device, is_detect=True, config_override=dw
                ),
            ],
        ]

        self.dfl = Yolov11sConv2D(parameter.dfl.conv, conv_pt.dfl.conv, device=device, is_dfl=True, config_override=dfl)
        self.anchors = ttnn.to_memory_config(conv_pt.anchors, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.strides = ttnn.to_memory_config(conv_pt.strides, memory_config=ttnn.L1_MEMORY_CONFIG)
        self.anchor = self.anchors

    def __call__(self, device, y1, y2, y3, tile_size=32):
        feats = (y1, y2, y3)
        x1, x2, x3 = tuple(_run_detect_conv_chain(device, chain, y) for chain, y in zip(self._cv2_chains, feats))
        x4, x5, x6 = tuple(_run_detect_conv_chain(device, chain, y) for chain, y in zip(self._cv3_chains, feats))

        y1 = sharded_concat_2(x1, x4)
        y2 = sharded_concat_2(x2, x5)
        y3 = sharded_concat_2(x3, x6)

        if y1.is_sharded():
            y1 = ttnn.sharded_to_interleaved(y1, memory_config=ttnn.L1_MEMORY_CONFIG)
        if y2.is_sharded():
            y2 = ttnn.sharded_to_interleaved(y2, memory_config=ttnn.L1_MEMORY_CONFIG)
        if y3.is_sharded():
            y3 = ttnn.sharded_to_interleaved(y3, memory_config=ttnn.L1_MEMORY_CONFIG)

        y = ttnn.concat((y1, y2, y3), dim=2, memory_config=ttnn.L1_MEMORY_CONFIG)
        ttnn.deallocate(y1)
        ttnn.deallocate(y2)
        ttnn.deallocate(y3)
        # y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        # if y.layout != ttnn.TILE_LAYOUT:
        y = ttnn.to_layout(y, ttnn.TILE_LAYOUT)
        y = ttnn.squeeze(y, dim=0)
        ya, yb = y[:, :, :64], y[:, :, 64:144]
        # deallocate_tensors(y1, y2, y3, x1, x2, x3, x4, x5, x6, y)
        ya = ttnn.reshape(ya, (ya.shape[0], y.shape[1], 4, 16))  # ya
        ya = ttnn.softmax_in_place(ya, dim=-1, numeric_stable=False)
        ya = ttnn.permute(ya, (0, 2, 1, 3))
        c = self.dfl(ya)
        ttnn.deallocate(ya)
        deallocate_tensors(x1, x2, x3, x4, x5, x6)
        if c.is_sharded():
            c = ttnn.sharded_to_interleaved(c, memory_config=ttnn.L1_MEMORY_CONFIG)
        c = ttnn.permute(c, (0, 3, 1, 2))
        b0, d3 = int(c.shape[0]), int(c.shape[3])
        c = ttnn.reshape(c, (b0, 4, d3 // 4))
        c1, c2 = c[:, :2, :], c[:, 2:4, :]
        c1 = self.anchor - c1
        c2 = self.anchor + c2
        z1 = c2 - c1
        z2 = c1 + c2
        z2 = ttnn.div(z2, 2)
        z = ttnn.concat((z2, z1), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        z = ttnn.multiply(z, self.strides)
        yb = ttnn.permute(yb, (0, 2, 1))
        yb = ttnn.sigmoid(yb)
        deallocate_tensors(c, z1, z2, c1, c2)
        out = ttnn.concat((z, yb), dim=1, memory_config=ttnn.L1_MEMORY_CONFIG)
        deallocate_tensors(yb, z)
        return out

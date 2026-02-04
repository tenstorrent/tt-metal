# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
TTNN YUNet Face Detection Model.

"""

import ttnn
from ttnn.device import Arch
from typing import Tuple
from models.tt_cnn.tt.builder import (
    Conv2dConfiguration,
    MaxPool2dConfiguration,
    TtConv2d,
    TtMaxPool2d,
    HeightShardedStrategyConfiguration,
)


class TTNNDPUnit:
    """
    Optimized TTNN DPUnit: Pointwise 1x1 → Depthwise 3x3 with ReLU.

    Key difference: NO reshape between conv1 and conv2!
    """

    def __init__(self, device, in_channels: int, out_channels: int, name: str):
        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = name
        self.conv1_weight = self.conv1_bias = None
        self.conv2_weight = self.conv2_bias = None
        self._conv1_cache = {}
        self._conv2_cache = {}

    def load_weights(self, conv1_weight, conv1_bias, conv2_weight, conv2_bias):
        self.conv1_weight = conv1_weight
        self.conv1_bias = conv1_bias
        self.conv2_weight = conv2_weight
        self.conv2_bias = conv2_bias

    def _get_conv1(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv1_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                dilation=(1, 1),
                weight=self.conv1_weight,
                bias=self.conv1_bias,
                sharding_strategy=HeightShardedStrategyConfiguration(),
                weights_dtype=ttnn.bfloat8_b,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
            )
            self._conv1_cache[key] = TtConv2d(config, self.device)
        return self._conv1_cache[key]

    def _get_conv2(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv2_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                batch_size=batch_size,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                groups=self.out_channels,
                dilation=(1, 1),
                weight=self.conv2_weight,
                bias=self.conv2_bias,
                sharding_strategy=HeightShardedStrategyConfiguration(),
                activation=ttnn.UnaryOpType.RELU,
                weights_dtype=ttnn.bfloat8_b,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
            )
            self._conv2_cache[key] = TtConv2d(config, self.device)
        return self._conv2_cache[key]

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        """Forward - NO reshape between conv1 and conv2!"""
        conv1 = self._get_conv1(batch_size, height, width)
        x, h_w = conv1(x, return_output_dim=True)
        # NO RESHAPE - pass directly to conv2
        conv2 = self._get_conv2(batch_size, h_w[0], h_w[1])
        x, h_w = conv2(x, return_output_dim=True)
        return x, h_w[0], h_w[1]


class TTNNConvBN:
    """Optimized Conv + BatchNorm (fused) with ReLU."""

    def __init__(self, device, in_ch: int, out_ch: int, kernel: int, stride: int, pad: int, name: str):
        self.device = device
        self.in_ch, self.out_ch = in_ch, out_ch
        self.kernel, self.stride, self.pad = kernel, stride, pad
        self.name = name
        self.weight = self.bias = None
        self._conv_cache = {}

    def load_weights(self, weight, bias):
        self.weight, self.bias = weight, bias

    def _get_conv(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                batch_size=batch_size,
                kernel_size=(self.kernel, self.kernel),
                stride=(self.stride, self.stride),
                padding=(self.pad, self.pad),
                groups=1,
                dilation=(1, 1),
                weight=self.weight,
                bias=self.bias,
                sharding_strategy=HeightShardedStrategyConfiguration(),
                activation=ttnn.UnaryOpType.RELU,
                weights_dtype=ttnn.bfloat8_b,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
            )
            self._conv_cache[key] = TtConv2d(config, self.device)
        return self._conv_cache[key]

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        conv = self._get_conv(batch_size, height, width)
        x, h_w = conv(x, return_output_dim=True)
        return x, h_w[0], h_w[1]


class TTNNConv1x1:
    """Optimized 1x1 convolution for head outputs."""

    def __init__(self, device, in_ch: int, out_ch: int, name: str):
        self.device = device
        self.in_ch, self.out_ch = in_ch, out_ch
        self.name = name
        self.weight = self.bias = None
        self._conv_cache = {}

    def load_weights(self, weight, bias):
        self.weight, self.bias = weight, bias

    def _get_conv(self, batch_size: int, input_height: int, input_width: int):
        key = (batch_size, input_height, input_width)
        if key not in self._conv_cache:
            config = Conv2dConfiguration(
                input_height=input_height,
                input_width=input_width,
                in_channels=self.in_ch,
                out_channels=self.out_ch,
                batch_size=batch_size,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
                groups=1,
                dilation=(1, 1),
                weight=self.weight,
                bias=self.bias,
                sharding_strategy=HeightShardedStrategyConfiguration(),
                weights_dtype=ttnn.bfloat8_b,
                enable_act_double_buffer=True,
                enable_weights_double_buffer=True,
            )
            self._conv_cache[key] = TtConv2d(config, self.device)
        return self._conv_cache[key]

    def __call__(self, x: ttnn.Tensor, batch_size: int, height: int, width: int) -> Tuple[ttnn.Tensor, int, int]:
        conv = self._get_conv(batch_size, height, width)
        x, h_w = conv(x, return_output_dim=True)
        return x, h_w[0], h_w[1]


def safe_reshape(x: ttnn.Tensor, shape, memory_config=ttnn.L1_MEMORY_CONFIG) -> ttnn.Tensor:
    """Safely reshape tensor - converts sharded to interleaved first if needed."""
    if x.memory_config().is_sharded():
        x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(x, shape, memory_config=memory_config)


class TTNNMaxPool:
    """Optimized MaxPool2d - reshapes at boundary."""

    def __init__(self, device, kernel_size: int = 2, stride: int = 2, padding: int = 0):
        self.device = device
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self._pool_cache = {}
        # Check if Wormhole - needs ROW_MAJOR workaround for max_pool2d
        self._is_wormhole = device.arch() == Arch.WORMHOLE_B0

    def _get_pool(self, batch_size: int, h: int, w: int, c: int):
        key = (batch_size, h, w, c)
        if key not in self._pool_cache:
            config = MaxPool2dConfiguration(
                input_height=h,
                input_width=w,
                channels=c,
                batch_size=batch_size,
                kernel_size=[self.kernel_size, self.kernel_size],
                stride=[self.stride, self.stride],
                padding=[self.padding, self.padding],
                dilation=[1, 1],
            )
            self._pool_cache[key] = TtMaxPool2d(config, self.device)
        return self._pool_cache[key]

    def __call__(
        self, x: ttnn.Tensor, batch_size: int, height: int, width: int, channels: int
    ) -> Tuple[ttnn.Tensor, int, int]:
        """Pool needs NHWC. On Wormhole, requires ROW_MAJOR to avoid sharding hang."""
        if self._is_wormhole:
            # Wormhole workaround: convert to DRAM + ROW_MAJOR before pool
            if x.memory_config().is_sharded():
                x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            x = ttnn.reshape(x, [batch_size, height, width, channels])
            if x.layout == ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)

            pool = self._get_pool(batch_size, height, width, channels)
            x = pool(x)

            if x.memory_config().is_sharded():
                x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            # Blackhole: original fast path
            x = safe_reshape(x, [batch_size, height, width, channels], memory_config=ttnn.L1_MEMORY_CONFIG)
            pool = self._get_pool(batch_size, height, width, channels)
            x = pool(x)

        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        x = ttnn.reshape(x, [batch_size, out_h, out_w, channels])
        return x, out_h, out_w


class TTNNUpsample:
    """Optimized Upsample - reshapes at boundary."""

    def __init__(self, scale_factor: int = 2):
        self.scale_factor = scale_factor

    def __call__(
        self, x: ttnn.Tensor, batch_size: int, height: int, width: int, channels: int
    ) -> Tuple[ttnn.Tensor, int, int]:
        x = safe_reshape(x, [batch_size, height, width, channels], memory_config=ttnn.L1_MEMORY_CONFIG)
        x = ttnn.to_layout(x, layout=ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.upsample(x, scale_factor=self.scale_factor, mode="nearest")
        return x, height * self.scale_factor, width * self.scale_factor


def to_nhwc(
    x: ttnn.Tensor, batch_size: int, height: int, width: int, channels: int, to_dram: bool = False
) -> ttnn.Tensor:
    """Reshape to NHWC. If to_dram=True, prepare for to_torch() by converting to DRAM + ROW_MAJOR."""
    x = safe_reshape(x, [batch_size, height, width, channels], memory_config=ttnn.L1_MEMORY_CONFIG)
    if to_dram:
        # Convert sharded to DRAM interleaved
        if x.memory_config().is_sharded():
            x = ttnn.sharded_to_interleaved(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        else:
            x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        # Convert TILE to ROW_MAJOR - required for to_torch() on tensors with small last dim
        if x.layout == ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
    return x


class TtYUNet:
    """
    Optimized YUNet - NO reshape between consecutive convs in DPUnit.

    High performance with bfloat8_b weights and minimized reshapes
    """

    def __init__(self, device, filters=(3, 16, 64, 64, 64, 64)):
        self.device = device
        self.filters = filters

        # Backbone
        self.p1_conv = TTNNConvBN(device, filters[0], filters[1], 3, 2, 1, "backbone.p1.0")
        self.p1_dpunit = TTNNDPUnit(device, filters[1], filters[1], "backbone.p1.1")

        self.p2_pool = TTNNMaxPool(device, 2, 2)
        self.p2_dpunit1 = TTNNDPUnit(device, filters[1], filters[2], "backbone.p2.1")
        self.p2_dpunit2 = TTNNDPUnit(device, filters[2], filters[2], "backbone.p2.2")
        self.p2_dpunit3 = TTNNDPUnit(device, filters[2], filters[2], "backbone.p2.3")

        self.p3_pool = TTNNMaxPool(device, 2, 2)
        self.p3_dpunit1 = TTNNDPUnit(device, filters[2], filters[3], "backbone.p3.1")
        self.p3_dpunit2 = TTNNDPUnit(device, filters[3], filters[3], "backbone.p3.2")

        self.p4_pool = TTNNMaxPool(device, 2, 2)
        self.p4_dpunit1 = TTNNDPUnit(device, filters[3], filters[4], "backbone.p4.1")
        self.p4_dpunit2 = TTNNDPUnit(device, filters[4], filters[4], "backbone.p4.2")

        self.p5_pool = TTNNMaxPool(device, 2, 2)
        self.p5_dpunit1 = TTNNDPUnit(device, filters[4], filters[5], "backbone.p5.1")
        self.p5_dpunit2 = TTNNDPUnit(device, filters[5], filters[5], "backbone.p5.2")

        # Neck
        self.up = TTNNUpsample(2)
        self.neck_conv1 = TTNNDPUnit(device, filters[5], filters[4], "neck.conv1")
        self.neck_conv2 = TTNNDPUnit(device, filters[4], filters[3], "neck.conv2")
        self.neck_conv3 = TTNNDPUnit(device, filters[3], filters[3], "neck.conv3")

        # Head
        self.head_m, self.head_cls, self.head_box, self.head_obj, self.head_kpt = [], [], [], [], []
        head_filters = (filters[3], filters[3], filters[4])
        for i, f in enumerate(head_filters):
            self.head_m.append(TTNNDPUnit(device, f, f, f"head.m.{i}"))
            self.head_cls.append(TTNNConv1x1(device, f, 1, f"head.cls.{i}"))
            self.head_box.append(TTNNConv1x1(device, f, 4, f"head.box.{i}"))
            self.head_obj.append(TTNNConv1x1(device, f, 1, f"head.obj.{i}"))
            self.head_kpt.append(TTNNConv1x1(device, f, 10, f"head.kpt.{i}"))

    def load_weights_from_torch(self, torch_model):
        """Load weights from PyTorch YUNet model."""

        def to_ttnn(w, b):
            return Conv2dConfiguration.convert_torch_weight_and_bias_to_ttnn(w, b)

        def load_dpunit(ttnn_dp, torch_dp):
            w1, b1 = to_ttnn(torch_dp.conv1.weight, torch_dp.conv1.bias)
            w2, b2 = to_ttnn(torch_dp.conv2.conv.weight, torch_dp.conv2.conv.bias)
            ttnn_dp.load_weights(w1, b1, w2, b2)

        # Backbone
        w, b = to_ttnn(torch_model.backbone.p1[0].conv.weight, torch_model.backbone.p1[0].conv.bias)
        self.p1_conv.load_weights(w, b)
        load_dpunit(self.p1_dpunit, torch_model.backbone.p1[1])

        load_dpunit(self.p2_dpunit1, torch_model.backbone.p2[1])
        load_dpunit(self.p2_dpunit2, torch_model.backbone.p2[2])
        load_dpunit(self.p2_dpunit3, torch_model.backbone.p2[3])

        load_dpunit(self.p3_dpunit1, torch_model.backbone.p3[1])
        load_dpunit(self.p3_dpunit2, torch_model.backbone.p3[2])

        load_dpunit(self.p4_dpunit1, torch_model.backbone.p4[1])
        load_dpunit(self.p4_dpunit2, torch_model.backbone.p4[2])

        load_dpunit(self.p5_dpunit1, torch_model.backbone.p5[1])
        load_dpunit(self.p5_dpunit2, torch_model.backbone.p5[2])

        # Neck
        load_dpunit(self.neck_conv1, torch_model.neck.conv1)
        load_dpunit(self.neck_conv2, torch_model.neck.conv2)
        load_dpunit(self.neck_conv3, torch_model.neck.conv3)

        # Head
        for i in range(3):
            load_dpunit(self.head_m[i], torch_model.head.m[i])
            for head_list, torch_head in [
                (self.head_cls, torch_model.head.cls),
                (self.head_box, torch_model.head.box),
                (self.head_obj, torch_model.head.obj),
                (self.head_kpt, torch_model.head.kpt),
            ]:
                w, b = to_ttnn(torch_head[i].weight, torch_head[i].bias)
                head_list[i].load_weights(w, b)

    def __call__(self, x: ttnn.Tensor):
        """Forward pass with minimized reshapes."""
        batch_size = x.shape[0]
        h, w = x.shape[1], x.shape[2]

        # P1
        x, h, w = self.p1_conv(x, batch_size, h, w)
        x, h, w = self.p1_dpunit(x, batch_size, h, w)

        # P2
        x, h, w = self.p2_pool(x, batch_size, h, w, self.filters[1])
        x, h, w = self.p2_dpunit1(x, batch_size, h, w)
        x, h, w = self.p2_dpunit2(x, batch_size, h, w)
        x, h, w = self.p2_dpunit3(x, batch_size, h, w)

        # P3
        x, h, w = self.p3_pool(x, batch_size, h, w, self.filters[2])
        x, h, w = self.p3_dpunit1(x, batch_size, h, w)
        p3, p3_h, p3_w = self.p3_dpunit2(x, batch_size, h, w)

        # P4
        x, h, w = self.p4_pool(p3, batch_size, p3_h, p3_w, self.filters[3])
        x, h, w = self.p4_dpunit1(x, batch_size, h, w)
        p4, p4_h, p4_w = self.p4_dpunit2(x, batch_size, h, w)

        # P5
        x, h, w = self.p5_pool(p4, batch_size, p4_h, p4_w, self.filters[4])
        x, h, w = self.p5_dpunit1(x, batch_size, h, w)
        p5, p5_h, p5_w = self.p5_dpunit2(x, batch_size, h, w)

        # Neck
        p5_out, p5_out_h, p5_out_w = self.neck_conv1(p5, batch_size, p5_h, p5_w)
        p5_up, _, _ = self.up(p5_out, batch_size, p5_out_h, p5_out_w, self.filters[4])
        p4_nhwc = to_nhwc(p4, batch_size, p4_h, p4_w, self.filters[4])
        p4_fused = ttnn.add(p4_nhwc, p5_up)

        p4_out, p4_out_h, p4_out_w = self.neck_conv2(p4_fused, batch_size, p4_h, p4_w)
        p4_up, _, _ = self.up(p4_out, batch_size, p4_out_h, p4_out_w, self.filters[3])
        p3_nhwc = to_nhwc(p3, batch_size, p3_h, p3_w, self.filters[3])
        p3_fused = ttnn.add(p3_nhwc, p4_up)

        p3_out, p3_out_h, p3_out_w = self.neck_conv3(p3_fused, batch_size, p3_h, p3_w)

        # Head
        features = [
            (p3_out, p3_out_h, p3_out_w, self.filters[3]),
            (p4_out, p4_out_h, p4_out_w, self.filters[3]),
            (p5_out, p5_out_h, p5_out_w, self.filters[4]),
        ]

        cls_out, box_out, obj_out, kpt_out = [], [], [], []

        for i, (feat, fh, fw, fc) in enumerate(features):
            feat, fh, fw = self.head_m[i](feat, batch_size, fh, fw)

            cls, _, _ = self.head_cls[i](feat, batch_size, fh, fw)
            box, _, _ = self.head_box[i](feat, batch_size, fh, fw)
            obj, _, _ = self.head_obj[i](feat, batch_size, fh, fw)
            kpt, _, _ = self.head_kpt[i](feat, batch_size, fh, fw)

            # Use to_dram=True for outputs - converts to DRAM + ROW_MAJOR for safe to_torch()
            cls_out.append(to_nhwc(cls, batch_size, fh, fw, 1, to_dram=True))
            box_out.append(to_nhwc(box, batch_size, fh, fw, 4, to_dram=True))
            obj_out.append(to_nhwc(obj, batch_size, fh, fw, 1, to_dram=True))
            kpt_out.append(to_nhwc(kpt, batch_size, fh, fw, 10, to_dram=True))

        return cls_out, box_out, obj_out, kpt_out


def create_yunet_model(device, torch_model=None) -> TtYUNet:
    """Factory function."""
    model = TtYUNet(device)
    if torch_model is not None:
        model.load_weights_from_torch(torch_model)
    return model
